#!/usr/bin/env python3
"""
Simple User-Friendly UI for CP11

Clean, intuitive interface focused on getting answers from Nepal's constitution and government documents.
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

# Configuration
API_BASE_URL = "http://localhost:8002"
PAGE_IMAGES_DIR = Path("data/page_images")

# Simple page configuration
st.set_page_config(
    page_title="🇳🇵 Nepal Government Q&A",
    page_icon="🇳🇵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, simple CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    margin-bottom: 20px;
    text-align: center;
}

.question-box {
    background: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    border: 2px solid #e9ecef;
    margin: 10px 0;
}

.answer-box {
    background: #ffffff;
    padding: 20px;
    border-radius: 10px;
    border-left: 4px solid #28a745;
    margin: 10px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.citation-chip {
    display: inline-block;
    background: #007bff;
    color: white;
    border-radius: 15px;
    padding: 5px 12px;
    margin: 3px;
    font-size: 12px;
    cursor: pointer;
}

.citation-chip:hover {
    background: #0056b3;
}

.simple-metric {
    background: white;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #dee2e6;
    text-align: center;
    margin: 5px;
}

.error-box {
    background: #f8d7da;
    color: #721c24;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #f5c6cb;
    margin: 10px 0;
}

.success-box {
    background: #d4edda;
    color: #155724;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #c3e6cb;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)


def make_api_request(endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Make API request with simple error handling."""
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"🔌 Cannot connect to the system. Please try again.")
        return None
    except json.JSONDecodeError as e:
        st.error(f"📝 System response error. Please try again.")
        return None


def load_page_image(doc_id: str, page_num: int, bbox: List[float]) -> Optional[str]:
    """Load page image with simple highlighting."""
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
        
        # Load and highlight image
        image = Image.open(image_path).convert('RGBA')
        
        if bbox and len(bbox) >= 4:
            draw = ImageDraw.Draw(image)
            x1, y1, x2, y2 = bbox[:4]
            
            # Simple red highlight
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            
            # Semi-transparent overlay
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 50))
            image = Image.alpha_composite(image, overlay)
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        return None


def main():
    """Main simple application."""
    
    # Simple header
    st.markdown("""
    <div class="main-header">
        <h1>🇳🇵 Nepal Government Q&A System</h1>
        <p>Ask questions about Nepal's Constitution, Health Policies, and Government Services</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple sidebar with clear options
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Language selection
        st.subheader("🌐 Answer Language")
        answer_language = st.selectbox(
            "Choose language for answers:",
            options=["Auto (Recommended)", "English", "Nepali", "Both Languages"],
            index=0,
            help="Auto will choose the best language based on your question"
        )
        
        # Simple search settings
        st.subheader("🔍 Search Settings")
        
        # Number of sources to check
        num_sources = st.slider(
            "Number of sources to check:",
            min_value=3,
            max_value=10,
            value=5,
            help="More sources = better answers but slower"
        )
        
        # Include non-government sources
        include_all = st.checkbox(
            "Include all sources (not just government)",
            value=False,
            help="Check this if you want answers from all available sources"
        )
        
        # Show advanced options
        with st.expander("🔧 Advanced Options"):
            st.write("**Cross-language search:**")
            cross_lang = st.slider(
                "How much to search in other languages:",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="0.0 = only same language, 1.0 = search all languages"
            )
            
            st.write("**Search quality:**")
            strict_mode = st.checkbox(
                "Strict government-only mode",
                value=True,
                help="Only use official government documents"
            )
    
    # Main question area
    st.header("❓ Ask Your Question")
    
    # Question input with examples
    question = st.text_area(
        "Type your question here:",
        placeholder="Examples:\n• What are the fundamental rights in Nepal?\n• नेपालमा स्वास्थ्य सेवा कस्तो छ?\n• What are the COVID-19 policies?\n• संविधानमा आपातकालीन अवस्थाको व्यवस्था के छ?",
        height=100,
        help="You can ask in English or Nepali"
    )
    
    # Quick question buttons
    st.write("**Quick Questions:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏛️ Constitution Rights"):
            question = "What are the fundamental rights in Nepal?"
    
    with col2:
        if st.button("🏥 Health Services"):
            question = "What health services are available in Nepal?"
    
    with col3:
        if st.button("🦠 COVID-19 Policies"):
            question = "What are the COVID-19 response policies in Nepal?"
    
    # Search button
    if st.button("🔍 Search for Answer", type="primary", disabled=not question):
        
        if not question.strip():
            st.warning("Please enter a question first!")
            return
        
        # Show loading
        with st.spinner("🔍 Searching government documents..."):
            
            # Prepare request
            language_map = {
                "Auto (Recommended)": "auto",
                "English": "en",
                "Nepali": "ne", 
                "Both Languages": "bilingual"
            }
            
            request_data = {
                "query": question,
                "k": num_sources,
                "query_lang": "auto",
                "output_mode": language_map[answer_language],
                "allow_distractors": include_all,
                "cross_lang_fallback_threshold": cross_lang,
                "table_context": True
            }
            
            # Make API call
            result = make_api_request("/answer", request_data)
            
            if result:
                display_simple_results(result, question)
            else:
                st.error("❌ Could not get answer. Please check if the system is running.")


def display_simple_results(result: Dict[str, Any], question: str):
    """Display results in a simple, user-friendly way."""
    
    # Question
    st.markdown(f"""
    <div class="question-box">
        <h3>❓ Your Question:</h3>
        <p><strong>{question}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Answer
    if result.get('is_refusal', False):
        # Refusal with helpful message
        st.markdown(f"""
        <div class="error-box">
            <h4>❌ Could not find a clear answer</h4>
            <p><strong>Reason:</strong> {result.get('refusal_reason', 'Insufficient evidence')}</p>
            <p><strong>Suggestions:</strong></p>
            <ul>
                <li>Try rephrasing your question</li>
                <li>Use simpler words</li>
                <li>Check if you included "Include all sources" in settings</li>
                <li>Try asking in Nepali if you asked in English (or vice versa)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Show what we found
        if result.get('answer_text'):
            st.write("**Partial answer found:**")
            st.write(result['answer_text'])
    
    else:
        # Success - show answer
        answer_text = result.get('answer_text', '')
        
        st.markdown(f"""
        <div class="answer-box">
            <h3>✅ Answer Found:</h3>
            <div style="font-size: 16px; line-height: 1.6;">
                {answer_text}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Citations
        citations = result.get('citations', [])
        if citations:
            st.subheader("📚 Sources Used:")
            
            for i, citation in enumerate(citations):
                with st.expander(f"📄 Source {i+1}: {citation.get('doc_id', 'Unknown Document')}", expanded=False):
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Citation text
                        text_preview = citation.get('text', '')
                        if len(text_preview) > 300:
                            text_preview = text_preview[:297] + "..."
                        
                        st.write("**Content:**")
                        st.write(text_preview)
                        
                        # Document info
                        doc_id = citation.get('doc_id', 'unknown')
                        page_num = citation.get('page_num', 1)
                        st.write(f"**Document:** {doc_id} (Page {page_num})")
                    
                    with col2:
                        # View image button
                        bbox = citation.get('bbox', [])
                        if st.button(f"🖼️ View Page", key=f"view_{i}"):
                            image_data = load_page_image(doc_id, page_num, bbox)
                            
                            if image_data:
                                st.image(image_data, caption=f"{doc_id} - Page {page_num}", use_column_width=True)
                            else:
                                st.warning("Page image not available")
                        
                        # Authority indicator
                        if citation.get('is_authoritative'):
                            st.success("✅ Official Document")
                        else:
                            st.info("ℹ️ Reference Source")
    
    # Simple metrics
    st.subheader("📊 Search Information")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Processing Time",
            f"{result.get('processing_time_ms', 0):.0f}ms"
        )
    
    with col2:
        citations_count = len(result.get('citations', []))
        st.metric(
            "Sources Found",
            citations_count
        )
    
    with col3:
        lang_consistency = result.get('language_consistency_rate', 0)
        st.metric(
            "Language Match",
            f"{lang_consistency:.0%}"
        )
    
    with col4:
        processing_strategy = result.get('processing_strategy', 'unknown')
        st.metric(
            "Search Method",
            processing_strategy.replace('_', ' ').title()
        )


if __name__ == "__main__":
    main()
