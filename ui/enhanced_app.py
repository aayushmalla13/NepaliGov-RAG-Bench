#!/usr/bin/env python3
"""
Enhanced Streamlit UI for CP11

Improved interface with better UX, performance, real-time updates,
advanced visualizations, and superior user experience.
"""

import streamlit as st
import requests
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import base64
from PIL import Image, ImageDraw, ImageEnhance
import io
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    # Fallback for systems without plotly
    px = None
    go = None
    PLOTLY_AVAILABLE = False
from datetime import datetime, timedelta
import pandas as pd

# Enhanced configuration
API_BASE_URL = "http://localhost:8002"
PAGE_IMAGES_DIR = Path("data/page_images")

# Enhanced page configuration
st.set_page_config(
    page_title="🇳🇵 NepaliGov-RAG-Bench Enhanced",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/help',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': "Enhanced NepaliGov-RAG-Bench with advanced features"
    }
)

# Enhanced CSS with modern styling
st.markdown("""
<style>
/* Enhanced modern styling */
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    margin-bottom: 20px;
    text-align: center;
}

.metric-card {
    background: white;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-left: 4px solid #667eea;
    margin: 5px;
}

.citation-chip-enhanced {
    display: inline-block;
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white;
    border-radius: 20px;
    padding: 8px 16px;
    margin: 4px;
    font-size: 12px;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.citation-chip-enhanced:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.3);
}

.confidence-bar {
    background: #f0f0f0;
    border-radius: 10px;
    height: 8px;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #0abde3);
    transition: width 0.5s ease;
}

.answer-container {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
}

.diagnostic-panel {
    background: #ffffff;
    border: 2px solid #e9ecef;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
}

.performance-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}

.status-excellent { background-color: #28a745; }
.status-good { background-color: #ffc107; }
.status-poor { background-color: #dc3545; }

.image-overlay-container {
    position: relative;
    display: inline-block;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid #f3f3f3;
    border-top: 3px solid #667eea;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)


# Session state initialization
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'performance_data' not in st.session_state:
    st.session_state.performance_data = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None


def make_enhanced_api_request(endpoint: str, data: Dict[str, Any], timeout: int = 60) -> Optional[Dict[str, Any]]:
    """Enhanced API request with better error handling and timeout."""
    try:
        with st.spinner(f"🔄 Processing {endpoint}..."):
            start_time = time.time()
            response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=timeout)
            response_time = time.time() - start_time
            
            response.raise_for_status()
            result = response.json()
            
            # Store performance data
            st.session_state.performance_data.append({
                'timestamp': datetime.now(),
                'endpoint': endpoint,
                'response_time': response_time,
                'cache_hit': result.get('cache_hit', False)
            })
            
            return result
            
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please try again or reduce the complexity of your query.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Cannot connect to API server. Please ensure the server is running at http://localhost:8000")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"🚨 API request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"📝 Failed to parse API response: {e}")
        return None


def get_api_health():
    """Get API health status with enhanced metrics."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None


def get_api_metrics():
    """Get API performance metrics."""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None


def render_confidence_bar(confidence: float, label: str = "") -> str:
    """Render enhanced confidence bar."""
    width_percent = confidence * 100
    color = "#28a745" if confidence > 0.8 else "#ffc107" if confidence > 0.5 else "#dc3545"
    
    return f"""
    <div style="margin: 5px 0;">
        <small>{label}</small>
        <div class="confidence-bar">
            <div style="width: {width_percent}%; background-color: {color}; height: 100%; border-radius: 10px;"></div>
        </div>
        <small>{confidence:.2f}</small>
    </div>
    """


def render_enhanced_citation_chip(citation: Dict[str, Any], index: int) -> str:
    """Render enhanced citation chip with metadata."""
    doc_id = citation.get('doc_id', 'unknown')
    page_num = citation.get('page_num', 1)
    confidence = citation.get('confidence_score', 0.5)
    
    confidence_color = "#28a745" if confidence > 0.8 else "#ffc107" if confidence > 0.5 else "#dc3545"
    
    return f"""
    <div class="citation-chip-enhanced" onclick="showCitation({index})" style="border-left: 4px solid {confidence_color};">
        📄 {doc_id} (p.{page_num})
        <br><small>Confidence: {confidence:.2f}</small>
    </div>
    """


def load_and_enhance_image(doc_id: str, page_num: int, bbox: List[float], confidence: float = 1.0) -> Optional[str]:
    """Load and enhance page image with advanced highlighting."""
    try:
        # Look for page image with multiple patterns
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
        
        # Load and enhance image
        image = Image.open(image_path).convert('RGBA')
        
        # Enhance image quality
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        if bbox and len(bbox) >= 4:
            draw = ImageDraw.Draw(image)
            x1, y1, x2, y2 = bbox[:4]
            
            # Confidence-based highlighting
            if confidence > 0.8:
                outline_color = "green"
                fill_alpha = 40
            elif confidence > 0.5:
                outline_color = "orange"
                fill_alpha = 30
            else:
                outline_color = "red"
                fill_alpha = 20
            
            # Draw enhanced highlight
            draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=4)
            
            # Add semi-transparent overlay
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            if outline_color == "green":
                fill_color = (0, 255, 0, fill_alpha)
            elif outline_color == "orange":
                fill_color = (255, 165, 0, fill_alpha)
            else:
                fill_color = (255, 0, 0, fill_alpha)
            
            overlay_draw.rectangle([x1, y1, x2, y2], fill=fill_color)
            image = Image.alpha_composite(image, overlay)
            
            # Add confidence indicator
            draw = ImageDraw.Draw(image)
            draw.text((x1, y1-20), f"Confidence: {confidence:.2f}", fill=outline_color)
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        st.error(f"Failed to load image {doc_id}_{page_num}: {e}")
        return None


def display_performance_dashboard():
    """Display enhanced performance dashboard."""
    st.subheader("📊 Performance Dashboard")
    
    # Get API metrics
    api_metrics = get_api_metrics()
    api_health = get_api_health()
    
    if api_metrics and api_health:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Requests",
                f"{api_metrics['total_requests']:,}",
                delta=None
            )
        
        with col2:
            avg_time = api_metrics['avg_response_time']
            status_color = "🟢" if avg_time < 1.0 else "🟡" if avg_time < 3.0 else "🔴"
            st.metric(
                "Avg Response Time",
                f"{avg_time:.2f}s",
                delta=None,
                help=f"{status_color} Performance indicator"
            )
        
        with col3:
            cache_rate = api_metrics['cache_hit_rate']
            st.metric(
                "Cache Hit Rate",
                f"{cache_rate:.1f}%",
                delta=None
            )
        
        with col4:
            status = api_health['status']
            status_emoji = "🟢" if status == "healthy" else "🟡" if status == "degraded" else "🔴"
            st.metric(
                "System Status",
                f"{status_emoji} {status.title()}",
                delta=None
            )
        
        # Performance chart
        if st.session_state.performance_data and PLOTLY_AVAILABLE:
            df = pd.DataFrame(st.session_state.performance_data)
            
            fig = px.line(
                df, 
                x='timestamp', 
                y='response_time',
                title="Response Time Trend",
                labels={'response_time': 'Response Time (s)', 'timestamp': 'Time'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        elif st.session_state.performance_data:
            # Fallback without plotly
            df = pd.DataFrame(st.session_state.performance_data)
            st.line_chart(df.set_index('timestamp')['response_time'])


def main():
    """Enhanced main application."""
    
    # Enhanced header
    st.markdown("""
    <div class="main-header">
        <h1>🇳🇵 NepaliGov-RAG-Bench Enhanced</h1>
        <p>Advanced Multilingual RAG System with Superior User Experience</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with enhanced controls
    with st.sidebar:
        st.header("🔧 Enhanced Configuration")
        
        # Query input with suggestions
        question = st.text_area(
            "Question",
            placeholder="Enter your question in English or Nepali...",
            height=100,
            help="💡 Try questions about Nepal's constitution, health policies, or COVID-19 response"
        )
        
        # Quick question suggestions
        if st.button("💡 Suggest Questions"):
            suggestions = [
                "What are the fundamental rights in Nepal?",
                "नेपालमा स्वास्थ्य सेवा कस्तो छ?",
                "COVID-19 response measures in Nepal",
                "संविधानमा आपातकालीन अवस्थाको व्यवस्था के छ?"
            ]
            st.write("**Suggested Questions:**")
            for i, suggestion in enumerate(suggestions, 1):
                if st.button(f"{i}. {suggestion}", key=f"suggestion_{i}"):
                    st.session_state.suggested_question = suggestion
        
        if 'suggested_question' in st.session_state:
            question = st.session_state.suggested_question
        
        # Enhanced language controls
        st.subheader("🌐 Language Settings")
        answer_language = st.selectbox(
            "Answer Language",
            options=["Auto", "English", "Nepali", "Bilingual"],
            index=0,
            help="Choose the language for the answer"
        )
        
        language_map = {
            "Auto": "auto",
            "English": "en", 
            "Nepali": "ne",
            "Bilingual": "bilingual"
        }
        
        # Enhanced retrieval settings
        st.subheader("🔍 Retrieval Settings")
        k = st.slider("Number of candidates (k)", min_value=1, max_value=30, value=10)
        allow_distractors = st.checkbox("Include distractors", value=False, help="Include non-authoritative content")
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            use_cache = st.checkbox("Enable caching", value=True, help="Use cached results for faster responses")
            include_debug = st.checkbox("Debug information", value=False, help="Show detailed processing information")
            include_confidence = st.checkbox("Confidence scores", value=True, help="Show confidence metrics")
            max_answer_length = st.slider("Max answer length", 500, 5000, 2000, help="Maximum characters in answer")
            
            # FAISS parameters
            st.write("**FAISS Parameters:**")
            nprobe = st.slider("nprobe", 1, 100, 10, help="Number of clusters to search")
            efSearch = st.slider("efSearch", 1, 1000, 100, help="Search effort parameter")
        
        # Cross-language settings
        fallback_threshold = st.slider(
            "Cross-language fallback threshold", 
            min_value=0.0, max_value=1.0, value=0.5, step=0.1,
            help="When to use cross-language evidence"
        )
    
    # Main content area
    if question:
        if st.button("🚀 Enhanced Search & Answer", type="primary"):
            
            # Prepare enhanced API request
            request_data = {
                "query": question,
                "k": k,
                "query_lang": "auto",
                "output_mode": language_map[answer_language],
                "allow_distractors": allow_distractors,
                "table_context": True,
                "cross_lang_fallback_threshold": fallback_threshold,
                "use_cache": use_cache,
                "include_confidence_scores": include_confidence,
                "max_answer_length": max_answer_length,
                "nprobe": nprobe,
                "efSearch": efSearch,
                "include_debug_info": include_debug
            }
            
            # Make enhanced API call
            result = make_enhanced_api_request("/answer", request_data)
            
            if result:
                st.session_state.current_result = result
                
                # Add to query history
                st.session_state.query_history.append({
                    'timestamp': datetime.now(),
                    'query': question,
                    'language': answer_language,
                    'processing_time': result.get('processing_time_ms', 0),
                    'cache_hit': result.get('cache_hit', False)
                })
                
                display_enhanced_results(result, include_debug, include_confidence)
    
    # Enhanced tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Main", "📊 Performance", "📈 Analytics", "🔧 Settings"])
    
    with tab1:
        if st.session_state.current_result:
            st.write("### Current Results")
            display_enhanced_results(st.session_state.current_result, False, True)
    
    with tab2:
        display_performance_dashboard()
    
    with tab3:
        display_analytics_dashboard()
    
    with tab4:
        display_settings_panel()


def display_enhanced_results(result: Dict[str, Any], include_debug: bool, include_confidence: bool):
    """Display enhanced results with better visualization."""
    
    # Enhanced answer display
    st.subheader("💬 Enhanced Answer")
    
    # Performance indicators
    processing_time = result.get('processing_time_ms', 0)
    cache_hit = result.get('cache_hit', False)
    overall_confidence = result.get('overall_confidence', 0.5)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Processing Time", f"{processing_time:.1f}ms")
    with col2:
        cache_status = "🎯 Cache Hit" if cache_hit else "🔄 Fresh"
        st.metric("Cache Status", cache_status)
    with col3:
        confidence_emoji = "🟢" if overall_confidence > 0.8 else "🟡" if overall_confidence > 0.5 else "🔴"
        st.metric("Confidence", f"{confidence_emoji} {overall_confidence:.2f}")
    with col4:
        reading_time = result.get('estimated_reading_time', 0)
        st.metric("Reading Time", f"{reading_time}s")
    
    # Answer content
    if result.get('is_refusal', False):
        st.warning(f"**Refusal:** {result.get('refusal_reason', 'Insufficient evidence')}")
        st.markdown(f"*{result.get('answer_text', '')}*")
        
        # Show why it was refused
        with st.expander("🔍 Refusal Analysis"):
            st.write("**Possible reasons:**")
            st.write("- Insufficient authoritative evidence")
            st.write("- Query too ambiguous")
            st.write("- No relevant documents found")
            
            evidence_thresholds = result.get('evidence_thresholds_used', {})
            st.write("**Evidence Requirements:**")
            st.json(evidence_thresholds)
    else:
        # Enhanced answer display
        if result.get('is_bilingual', False):
            en_tab, ne_tab = st.tabs(["🇺🇸 English", "🇳🇵 नेपाली"])
            
            with en_tab:
                if result.get('english_answer'):
                    st.markdown(f"""
                    <div class="answer-container">
                        {result['english_answer']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No English answer available")
            
            with ne_tab:
                if result.get('nepali_answer'):
                    st.markdown(f"""
                    <div class="answer-container">
                        {result['nepali_answer']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("नेपाली उत्तर उपलब्ध छैन")
        else:
            # Single language answer
            answer_lang = result.get('answer_language', 'en')
            lang_flag = "🇺🇸" if answer_lang == 'en' else "🇳🇵"
            
            st.markdown(f"""
            <div class="answer-container">
                <h4>{lang_flag} {answer_lang.upper()} Answer</h4>
                {result.get('answer_text', '')}
            </div>
            """, unsafe_allow_html=True)
    
    # Enhanced citations section
    citations = result.get('citations', [])
    if citations:
        st.subheader("📚 Enhanced Citations")
        
        # Citation overview
        st.write(f"Found **{len(citations)}** citations with enhanced metadata:")
        
        for i, citation in enumerate(citations):
            with st.expander(f"📄 Citation {i+1}: {citation.get('doc_id', 'unknown')} (Page {citation.get('page_num', 1)})", expanded=False):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Citation text
                    text_preview = citation.get('text', '')
                    if len(text_preview) > 300:
                        text_preview = text_preview[:297] + "..."
                    
                    st.markdown(f"**Content Preview:**")
                    st.markdown(f"*{text_preview}*")
                    
                    # Enhanced metadata
                    if include_confidence:
                        confidence = citation.get('confidence_score', 0.5)
                        st.markdown(render_confidence_bar(confidence, "Citation Confidence"), unsafe_allow_html=True)
                        
                        if 'relevance_explanation' in citation:
                            st.write(f"**Relevance:** {citation['relevance_explanation']}")
                
                with col2:
                    # Citation actions
                    doc_id = citation.get('doc_id', 'unknown')
                    page_num = citation.get('page_num', 1)
                    bbox = citation.get('bbox', [])
                    confidence = citation.get('confidence_score', 0.5)
                    
                    if st.button(f"🖼️ View Enhanced Highlight", key=f"enhanced_cite_{i}"):
                        image_data = load_and_enhance_image(doc_id, page_num, bbox, confidence)
                        
                        if image_data:
                            st.markdown(f"""
                            <div class="image-overlay-container">
                                <img src="{image_data}" style="max-width: 100%; border-radius: 8px;">
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning(f"Page image not found: {doc_id}_page_{page_num:03d}.png")
                    
                    # Additional citation info
                    if citation.get('is_authoritative'):
                        st.success("✅ Authoritative")
                    else:
                        st.warning("⚠️ Non-authoritative")
                    
                    if citation.get('image_available'):
                        st.info("🖼️ Image Available")
                    
                    reading_time = citation.get('estimated_reading_time', 0)
                    if reading_time > 0:
                        st.info(f"⏱️ {reading_time}s read")
    
    # Enhanced diagnostics
    if include_debug:
        st.subheader("🔍 Enhanced Diagnostics")
        
        with st.expander("🧠 Processing Intelligence", expanded=True):
            diag_col1, diag_col2 = st.columns(2)
            
            with diag_col1:
                st.markdown("**CP9 Bilingual Processing:**")
                processing_data = {
                    "Strategy": result.get('processing_strategy', 'unknown'),
                    "Domain": result.get('query_domain', 'unknown'),
                    "Bilingual Confidence": f"{result.get('bilingual_confidence', 0):.2f}",
                    "Semantic Mappings": result.get('semantic_mappings_count', 0),
                    "Language Segments": result.get('language_segments_count', 0)
                }
                
                for key, value in processing_data.items():
                    st.write(f"- **{key}:** {value}")
            
            with diag_col2:
                st.markdown("**CP10 Enhanced Evidence:**")
                evidence_data = result.get('evidence_thresholds_used', {})
                authority_sources = result.get('authority_detection_sources', [])
                fallbacks = result.get('used_fallbacks', {})
                
                st.write(f"- **Min Characters:** {evidence_data.get('min_chars', 'N/A')}")
                st.write(f"- **Min Confidence:** {evidence_data.get('min_confidence', 'N/A')}")
                st.write(f"- **Authority Sources:** {len(authority_sources)}")
                st.write(f"- **Fallback Used:** {fallbacks.get('fallback_triggered', False)}")
        
        # Performance breakdown
        with st.expander("⚡ Performance Breakdown"):
            perf_data = {
                "Total Processing": f"{result.get('processing_time_ms', 0):.1f}ms",
                "Cache Hit": result.get('cache_hit', False),
                "Answer Quality": f"{result.get('answer_quality_score', 0):.2f}",
                "Language Consistency": f"{result.get('language_consistency_rate', 0):.2f}",
                "Same Lang Citations": f"{result.get('same_lang_citation_rate', 0):.2f}"
            }
            
            st.json(perf_data)


def display_analytics_dashboard():
    """Display enhanced analytics dashboard."""
    st.subheader("📈 Query Analytics")
    
    if st.session_state.query_history:
        df = pd.DataFrame(st.session_state.query_history)
        
        if PLOTLY_AVAILABLE:
            # Query frequency by language
            lang_counts = df['language'].value_counts()
            fig_lang = px.pie(values=lang_counts.values, names=lang_counts.index, title="Queries by Language")
            st.plotly_chart(fig_lang, use_container_width=True)
            
            # Response time distribution
            fig_time = px.histogram(df, x='processing_time', title="Response Time Distribution", nbins=20)
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Cache hit rate over time
            fig_cache = px.scatter(df, x='timestamp', y='cache_hit', title="Cache Hits Over Time")
            st.plotly_chart(fig_cache, use_container_width=True)
        else:
            # Fallback charts without plotly
            st.write("### Query Distribution by Language")
            lang_counts = df['language'].value_counts()
            st.bar_chart(lang_counts)
            
            st.write("### Response Time Distribution")
            st.bar_chart(df['processing_time'].value_counts().head(10))
            
            st.write("### Cache Hit Analysis")
            cache_summary = df['cache_hit'].value_counts()
            st.bar_chart(cache_summary)
        
        # Recent queries
        st.write("### Recent Queries")
        recent_queries = df.tail(10)[['timestamp', 'query', 'language', 'processing_time', 'cache_hit']]
        st.dataframe(recent_queries)
    else:
        st.info("No query history available yet. Start asking questions to see analytics!")


def display_settings_panel():
    """Display enhanced settings panel."""
    st.subheader("🔧 System Settings")
    
    # API configuration
    st.write("### API Configuration")
    current_api = st.text_input("API Base URL", value=API_BASE_URL)
    
    if st.button("Test API Connection"):
        try:
            response = requests.get(f"{current_api}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ API connection successful!")
                health_data = response.json()
                st.json(health_data)
            else:
                st.error(f"❌ API returned status code: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Failed to connect to API: {e}")
    
    # Cache management
    st.write("### Cache Management")
    if st.button("Clear API Cache"):
        try:
            response = requests.delete(f"{API_BASE_URL}/cache/clear", timeout=10)
            if response.status_code == 200:
                st.success("✅ API cache cleared successfully!")
            else:
                st.error("❌ Failed to clear cache")
        except Exception as e:
            st.error(f"❌ Cache clear failed: {e}")
    
    if st.button("Clear Local History"):
        st.session_state.query_history = []
        st.session_state.performance_data = []
        st.success("✅ Local history cleared!")
    
    # Export options
    st.write("### Export Options")
    if st.session_state.query_history:
        if st.button("📥 Export Query History"):
            df = pd.DataFrame(st.session_state.query_history)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
