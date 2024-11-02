#!/usr/bin/env python3
"""
Streamlit UI for NepaliGov-RAG-Bench

Interactive interface with clickable bbox highlights and comprehensive
CP6/CP7/CP9/CP10 diagnostics integration.
"""

import streamlit as st
import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import base64
from PIL import Image, ImageDraw
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8002"
PAGE_IMAGES_DIR = Path("data/page_images")

# Page configuration
st.set_page_config(
    page_title="NepaliGov-RAG-Bench",
    page_icon="🇳🇵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.citation-chip {
    display: inline-block;
    background-color: #e1f5fe;
    border: 1px solid #0277bd;
    border-radius: 16px;
    padding: 4px 12px;
    margin: 2px;
    font-size: 12px;
    cursor: pointer;
}

.citation-chip:hover {
    background-color: #b3e5fc;
}

.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 10px;
    font-weight: bold;
    margin: 1px;
}

.badge-en { background-color: #e8f5e8; color: #2e7d32; }
.badge-ne { background-color: #fff3e0; color: #f57c00; }
.badge-auth { background-color: #e8f5e8; color: #2e7d32; }
.badge-ocr-high { background-color: #e8f5e8; color: #2e7d32; }
.badge-ocr-medium { background-color: #fff3e0; color: #f57c00; }
.badge-ocr-low { background-color: #ffebee; color: #c62828; }
.badge-watermark { background-color: #fce4ec; color: #ad1457; }
.badge-table { background-color: #e3f2fd; color: #1565c0; }
.badge-text { background-color: #f3e5f5; color: #7b1fa2; }
.badge-expanded { background-color: #e0f2f1; color: #00695c; }
.badge-diversified { background-color: #f1f8e9; color: #558b2f; }

.diagnostics-section {
    background-color: #f8f9fa;
    padding: 16px;
    border-radius: 8px;
    border: 1px solid #dee2e6;
    margin: 8px 0;
}

.metric-card {
    background-color: white;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #e9ecef;
    margin: 4px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


def make_api_request(endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Make API request with error handling."""
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse API response: {e}")
        return None


def render_badge(text: str, badge_type: str) -> str:
    """Render a styled badge."""
    return f'<span class="badge badge-{badge_type}">{text}</span>'


def render_citation_chip(citation_token: str, doc_id: str, page_num: int) -> str:
    """Render a clickable citation chip with XSS protection."""
    import html
    safe_doc_id = html.escape(doc_id)
    safe_citation_token = html.escape(citation_token)
    return f'<span class="citation-chip" onclick="showPageImage(\'{safe_doc_id}\', {page_num})">{safe_citation_token}</span>'


def load_and_overlay_image(doc_id: str, page_num: int, bbox: List[float]) -> Optional[str]:
    """Load page image and overlay bbox highlight."""
    try:
        # Look for page image
        possible_paths = [
            PAGE_IMAGES_DIR / f"{doc_id}_{page_num:03d}.png",
            PAGE_IMAGES_DIR / f"{doc_id}_page_{page_num:03d}.png",
            PAGE_IMAGES_DIR / f"{doc_id}.png"
        ]
        
        image_path = None
        for path in possible_paths:
            if path.exists():
                image_path = path
                break
        
        if not image_path:
            return None
        
        # Load image
        image = Image.open(image_path)
        
        # Create overlay if bbox provided
        if bbox and len(bbox) >= 4:
            draw = ImageDraw.Draw(image)
            x1, y1, x2, y2 = bbox[:4]
            
            # Draw highlight rectangle
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3, fill=None)
            
            # Add semi-transparent fill
            overlay = Image.new('RGBA', image.size, (255, 0, 0, 30))
            mask = Image.new('RGBA', image.size, (0, 0, 0, 0))
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 30))
            image = Image.alpha_composite(image.convert('RGBA'), mask)
        
        # Convert to base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        logger.error(f"Failed to load image {doc_id}_{page_num}: {e}")
        return None


def main():
    """Main Streamlit application."""
    
    st.title("🇳🇵 NepaliGov-RAG-Bench")
    st.markdown("*Advanced Multilingual RAG System for Nepali Government Documents*")
    
    # Sidebar controls
    st.sidebar.header("🔧 Configuration")
    
    # Query input
    question = st.sidebar.text_area(
        "Question",
        placeholder="Enter your question in English or Nepali...",
        height=100
    )
    
    # Language controls
    answer_language = st.sidebar.selectbox(
        "Answer Language",
        options=["Auto", "English", "Nepali", "Bilingual"],
        index=0
    )
    
    language_map = {
        "Auto": "auto",
        "English": "en", 
        "Nepali": "ne",
        "Bilingual": "bilingual"
    }
    
    # Retrieval parameters
    st.sidebar.subheader("🔍 Retrieval Settings")
    k = st.sidebar.slider("Number of candidates (k)", min_value=1, max_value=20, value=10)
    allow_distractors = st.sidebar.checkbox("Allow distractors", value=False)
    
    # Advanced toggles
    st.sidebar.subheader("⚙️ Advanced Options")
    use_query_expansion = st.sidebar.checkbox("Use query expansion", value=True)
    diversify_results = st.sidebar.checkbox("Diversify results", value=True)
    show_diagnostics = st.sidebar.checkbox("Show evidence diagnostics", value=False)
    per_claim_justifications = st.sidebar.checkbox("Per-claim justifications", value=False)
    
    # Cross-language settings
    fallback_threshold = st.sidebar.slider(
        "Cross-language fallback threshold", 
        min_value=0.0, max_value=1.0, value=0.5, step=0.1
    )
    
    # Main content area
    if question:
        if st.button("🚀 Search & Answer", type="primary"):
            
            with st.spinner("Processing your question..."):
                # Prepare API request
                request_data = {
                    "query": question,
                    "k": k,
                    "query_lang": "auto",
                    "output_mode": language_map[answer_language],
                    "allow_distractors": allow_distractors,
                    "table_context": True,
                    "cross_lang_fallback_threshold": fallback_threshold
                }
                
                # Make API call
                result = make_api_request("/answer", request_data)
                
                if result:
                    # Display results
                    display_answer_results(result, show_diagnostics, per_claim_justifications)
                else:
                    st.error("Failed to get response from API. Please check if the API server is running.")
    
    # Benchmark tab
    with st.expander("📊 Benchmark Dashboard", expanded=False):
        display_benchmark_tab()


def display_answer_results(result: Dict[str, Any], show_diagnostics: bool, per_claim_justifications: bool):
    """Display answer results with comprehensive formatting."""
    
    # Main answer display
    st.subheader("💬 Answer")
    
    if result.get('is_refusal', False):
        st.warning(f"**Refusal:** {result.get('refusal_reason', 'Insufficient evidence')}")
        st.markdown(f"*{result.get('answer_text', '')}*")
    else:
        # Handle bilingual vs single language
        if result.get('is_bilingual', False):
            # Bilingual tabs
            en_tab, ne_tab = st.tabs(["🇺🇸 English", "🇳🇵 नेपाली"])
            
            with en_tab:
                if result.get('english_answer'):
                    st.markdown(result['english_answer'])
                else:
                    st.info("No English answer available")
            
            with ne_tab:
                if result.get('nepali_answer'):
                    st.markdown(result['nepali_answer'])
                else:
                    st.info("नेपाली उत्तर उपलब्ध छैन")
        else:
            # Single language answer
            answer_lang = result.get('answer_language', 'en')
            lang_flag = "🇺🇸" if answer_lang == 'en' else "🇳🇵"
            st.markdown(f"{lang_flag} **{answer_lang.upper()}:** {result.get('answer_text', '')}")
    
    # Citations section
    citations = result.get('citations', [])
    if citations:
        st.subheader("📚 Citations")
        
        citation_cols = st.columns(min(len(citations), 3))
        
        for i, citation in enumerate(citations):
            with citation_cols[i % len(citation_cols)]:
                doc_id = citation.get('doc_id', 'unknown')
                page_num = citation.get('page_num', 1)
                
                # Citation card
                with st.container():
                    st.markdown(f"**{doc_id}** (Page {page_num})")
                    
                    # Badges
                    badges_html = ""
                    if citation.get('language') == 'en':
                        badges_html += render_badge("EN", "en")
                    elif citation.get('language') == 'ne':
                        badges_html += render_badge("NE", "ne")
                    
                    if citation.get('is_authoritative'):
                        badges_html += render_badge("AUTH", "auth")
                    
                    st.markdown(badges_html, unsafe_allow_html=True)
                    
                    # Citation text preview
                    text_preview = citation.get('text', '')[:100] + "..." if len(citation.get('text', '')) > 100 else citation.get('text', '')
                    st.markdown(f"*{text_preview}*")
                    
                    # Clickable highlight button
                    if st.button(f"🖼️ View Highlight", key=f"cite_{i}"):
                        bbox = citation.get('bbox', [])
                        image_data = load_and_overlay_image(doc_id, page_num, bbox)
                        
                        if image_data:
                            st.image(image_data, caption=f"{doc_id} - Page {page_num}", use_column_width=True)
                        else:
                            st.warning(f"Page image not found: {doc_id}_page_{page_num:03d}.png")
    
    # Processing metrics
    st.subheader("📊 Processing Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Processing Time",
            f"{result.get('processing_time_ms', 0):.1f}ms"
        )
    
    with col2:
        st.metric(
            "Language Consistency",
            f"{result.get('language_consistency_rate', 0):.2f}"
        )
    
    with col3:
        st.metric(
            "Same Lang Citations",
            f"{result.get('same_lang_citation_rate', 0):.2f}"
        )
    
    with col4:
        st.metric(
            "Cross Lang Success@5",
            f"{result.get('cross_lang_success_at_5', 0):.2f}"
        )
    
    # CP9/CP10 diagnostics
    if show_diagnostics:
        st.subheader("🔍 Evidence Diagnostics")
        
        with st.expander("CP9 Bilingual Processing", expanded=True):
            diag_col1, diag_col2 = st.columns(2)
            
            with diag_col1:
                st.markdown("**Processing Strategy:**")
                st.code(result.get('processing_strategy', 'monolingual'))
                
                st.markdown("**Query Domain:**")
                st.code(result.get('query_domain', 'general'))
                
                st.markdown("**Bilingual Confidence:**")
                st.progress(result.get('bilingual_confidence', 1.0))
            
            with diag_col2:
                st.markdown("**Semantic Mappings:**")
                st.code(str(result.get('semantic_mappings_count', 0)))
                
                st.markdown("**Language Segments:**")
                st.code(str(result.get('language_segments_count', 1)))
        
        with st.expander("CP10 Enhanced Evidence Selection", expanded=True):
            thresholds = result.get('evidence_thresholds_used', {})
            st.json({
                "Evidence Thresholds": thresholds,
                "Authority Detection Sources": result.get('authority_detection_sources', []),
                "Fallbacks Used": result.get('used_fallbacks', {}),
                "Preprocessing Summary": result.get('candidate_preprocessing_summary', {}),
                "Formatting Info": result.get('formatting_info', {})
            })
    
    # Per-claim justifications
    if per_claim_justifications:
        st.subheader("⚖️ Per-Claim Validation")
        
        validations = result.get('per_claim_validation', [])
        if validations:
            for i, validation in enumerate(validations):
                with st.expander(f"Citation {i+1}: {validation.get('citation_token', '')}", expanded=False):
                    val_col1, val_col2 = st.columns(2)
                    
                    with val_col1:
                        st.markdown("**Validation Status:**")
                        if validation.get('is_valid', False):
                            st.success("✅ Valid")
                        else:
                            st.error("❌ Invalid")
                        
                        st.markdown("**Authority Verified:**")
                        if validation.get('authority_verified', False):
                            st.success("✅ Authoritative")
                        else:
                            st.warning("⚠️ Not Authoritative")
                    
                    with val_col2:
                        st.markdown("**Character IoU:**")
                        st.progress(validation.get('char_iou', 0.0))
                        
                        st.markdown("**Bbox IoU:**")
                        st.progress(validation.get('bbox_iou', 0.0))
        else:
            st.info("No citation validations available")


def display_benchmark_tab():
    """Display benchmark dashboard."""
    st.markdown("### 📈 System Performance Metrics")
    
    # Sample queries for benchmarking
    sample_queries = [
        "What are fundamental rights in Nepal?",
        "नेपालमा स्वास्थ्य सेवा कस्तो छ?",
        "COVID-19 response measures",
        "Emergency procedures आपातकालीन"
    ]
    
    if st.button("🧪 Run Sample Benchmark"):
        with st.spinner("Running benchmark queries..."):
            request_data = {
                "queries": sample_queries,
                "k": 10,
                "output_mode": "auto"
            }
            
            result = make_api_request("/eval/sample", request_data)
            
            if result:
                # Display aggregated metrics
                metrics = result.get('aggregated_metrics', {})
                
                met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                
                with met_col1:
                    st.metric(
                        "Authority Purity",
                        f"{metrics.get('average_authority_purity', 0):.3f}"
                    )
                
                with met_col2:
                    st.metric(
                        "Refusal Rate", 
                        f"{metrics.get('refusal_rate', 0):.2f}"
                    )
                
                with met_col3:
                    st.metric(
                        "Avg Citations",
                        f"{metrics.get('average_citations_per_query', 0):.1f}"
                    )
                
                with met_col4:
                    st.metric(
                        "Processing Time",
                        f"{result.get('processing_time_ms', 0):.0f}ms"
                    )
                
                # Detailed results
                st.markdown("### 📋 Query Results")
                results = result.get('results', [])
                
                for i, query_result in enumerate(results):
                    with st.expander(f"Query {i+1}: {query_result.get('query', '')}", expanded=False):
                        res_col1, res_col2 = st.columns(2)
                        
                        with res_col1:
                            st.markdown("**Candidates Found:**")
                            st.code(str(query_result.get('candidates_count', 0)))
                            
                            st.markdown("**Authority Purity:**")
                            st.progress(query_result.get('authority_purity', 0.0))
                        
                        with res_col2:
                            st.markdown("**Refusal:**")
                            if query_result.get('is_refusal', False):
                                st.error("Yes")
                            else:
                                st.success("No")
                            
                            st.markdown("**Citations:**")
                            st.code(str(query_result.get('citations_count', 0)))
            else:
                st.error("Benchmark failed")
    
    # System information
    st.markdown("### ℹ️ System Information")
    
    if st.button("🔍 Check API Health"):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                st.success("✅ API is healthy")
                st.json(health_data)
            else:
                st.error(f"❌ API health check failed: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Cannot reach API: {e}")


if __name__ == "__main__":
    main()
