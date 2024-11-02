#!/usr/bin/env python3
"""
🇳🇵 Nepal Government Q&A Web Interface

Beautiful, user-friendly web interface for asking questions about
any Nepal government document - Constitution, Health, COVID-19, etc.
"""

import streamlit as st
import requests
import json
import time
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="🇳🇵 Nepal Government Q&A",
    page_icon="🇳🇵",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Ask questions about Nepal's constitution, health policies, COVID-19 response, and more!"
    }
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #DC143C 0%, #FF6347 50%, #1E3A8A 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.question-box {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #DC143C;
    margin: 1rem 0;
}

.answer-box {
    background: #ffffff;
    padding: 2rem;
    border-radius: 10px;
    border-left: 5px solid #28a745;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.source-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #dee2e6;
    margin: 0.5rem 0;
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #dee2e6;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.document-chip {
    display: inline-block;
    background: #007bff;
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 15px;
    font-size: 0.8rem;
    margin: 0.2rem;
}

.confidence-high { border-left-color: #28a745; }
.confidence-medium { border-left-color: #ffc107; }
.confidence-low { border-left-color: #dc3545; }

.stButton > button {
    background: linear-gradient(90deg, #DC143C, #FF6347);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 0.5rem 2rem;
    font-weight: bold;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(220, 20, 60, 0.3);
}
</style>
""", unsafe_allow_html=True)

def check_api_health() -> bool:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_system_info() -> Optional[Dict[str, Any]]:
    """Get system information from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def get_documents() -> Optional[List[Dict[str, Any]]]:
    """Get information about available documents."""
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def ask_question(question: str, max_results: int = 5, language_pref: str = "auto") -> Optional[Dict[str, Any]]:
    """Ask a question to the API."""
    try:
        payload = {
            "question": question,
            "max_results": max_results,
            "include_sources": True,
            "language_preference": language_pref
        }
        
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. Please try again with a simpler question.")
        return None
    except Exception as e:
        st.error(f"❌ Error connecting to the system: {str(e)}")
        return None

def render_confidence_indicator(confidence: float):
    """Render a confidence indicator."""
    if confidence >= 0.7:
        color = "#28a745"
        label = "High Confidence"
    elif confidence >= 0.4:
        color = "#ffc107"
        label = "Medium Confidence"
    else:
        color = "#dc3545"
        label = "Low Confidence"
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin: 0.5rem 0;">
        <div style="width: 100px; height: 8px; background: #e9ecef; border-radius: 4px; margin-right: 10px;">
            <div style="width: {confidence*100}%; height: 100%; background: {color}; border-radius: 4px;"></div>
        </div>
        <span style="color: {color}; font-weight: bold; font-size: 0.9rem;">{label} ({confidence:.1%})</span>
    </div>
    """, unsafe_allow_html=True)

def render_sources(sources: List[Dict[str, Any]]):
    """Render source information in an organized way."""
    if not sources:
        return
    
    st.subheader("📚 Sources")
    
    for i, source in enumerate(sources, 1):
        relevance = source['relevance_score']
        
        # Color code by relevance
        if relevance >= 15:
            border_color = "#28a745"
        elif relevance >= 10:
            border_color = "#ffc107"
        else:
            border_color = "#6c757d"
        
        st.markdown(f"""
        <div class="source-card" style="border-left: 4px solid {border_color};">
            <h4 style="margin: 0 0 0.5rem 0; color: {border_color};">
                📄 {i}. {source['doc_title']}
            </h4>
            <div style="font-size: 0.9rem; color: #6c757d; margin-bottom: 0.5rem;">
                Page {source['page_num']} • Relevance: {relevance:.1f}/20 • Language: {source['language'].upper()} • {source['char_count']} characters
            </div>
            <div style="font-size: 0.95rem; line-height: 1.4;">
                {source['text'][:300]}{"..." if len(source['text']) > 300 else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_document_coverage(coverage: Dict[str, int], documents: List[Dict[str, Any]]):
    """Render document coverage visualization."""
    if not coverage or not documents:
        return
    
    # Create document lookup
    doc_lookup = {doc['doc_id']: doc['title'] for doc in documents}
    
    # Prepare data for visualization
    doc_names = []
    chunk_counts = []
    
    for doc_id, count in coverage.items():
        doc_names.append(doc_lookup.get(doc_id, doc_id)[:30] + "..." if len(doc_lookup.get(doc_id, doc_id)) > 30 else doc_lookup.get(doc_id, doc_id))
        chunk_counts.append(count)
    
    # Create bar chart
    fig = px.bar(
        x=chunk_counts,
        y=doc_names,
        orientation='h',
        title="📊 Information Sources by Document",
        labels={'x': 'Number of Relevant Sections', 'y': 'Document'},
        color=chunk_counts,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=max(300, len(doc_names) * 40),
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🇳🇵 Nepal Government Q&A System</h1>
        <p>Ask questions about Nepal's Constitution, Health Policies, COVID-19 Response, Emergency Services, and more!</p>
        <p><em>Powered by 22 official government documents with 1,917+ information sections</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("🔌 **Cannot connect to the Q&A system.** Please ensure the API server is running on port 8000.")
        st.code("To start the server, run: python nepal_gov_api.py")
        return
    
    # Get system info
    system_info = get_system_info()
    documents = get_documents()
    
    # Sidebar with system information
    with st.sidebar:
        st.header("📋 System Information")
        
        if system_info:
            st.success("✅ System Online")
            st.metric("Documents", system_info.get('total_documents', 0))
            st.metric("Information Sections", system_info.get('total_chunks', 0))
            
            st.subheader("📚 Available Topics")
            topics = system_info.get('available_topics', [])
            for topic in topics:
                st.markdown(f"• {topic}")
        else:
            st.warning("⚠️ System information unavailable")
        
        st.markdown("---")
        
        # Search settings
        st.subheader("⚙️ Search Settings")
        max_results = st.slider("Max results to analyze", 3, 10, 5)
        language_pref = st.selectbox(
            "Language preference",
            ["auto", "en", "ne"],
            format_func=lambda x: {"auto": "🌐 Auto-detect", "en": "🇺🇸 English", "ne": "🇳🇵 Nepali"}[x]
        )
        
        # Show documents if available
        if documents:
            st.subheader("📄 Available Documents")
            with st.expander("View all documents"):
                for doc in documents[:10]:  # Show first 10
                    st.markdown(f"**{doc['title']}**")
                    st.markdown(f"• {doc['total_chunks']} sections")
                    st.markdown(f"• Language: {doc['language'].upper()}")
                    st.markdown(f"• Topics: {', '.join(doc['topics'])}")
                    st.markdown("---")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("❓ Ask Your Question")
        
        # Question input
        question = st.text_area(
            "What would you like to know about Nepal's government?",
            placeholder="Examples:\n• What are the fundamental rights in Nepal?\n• What health services are available?\n• How did Nepal respond to COVID-19?\n• What is the structure of Nepal's government?\n• What emergency health protocols exist?",
            height=120
        )
        
        # Quick question buttons
        st.subheader("🚀 Quick Questions")
        col_q1, col_q2, col_q3 = st.columns(3)
        
        with col_q1:
            if st.button("🏛️ Constitution & Rights"):
                question = "What are the fundamental rights guaranteed by Nepal's constitution?"
        
        with col_q2:
            if st.button("🏥 Health Services"):
                question = "What health services and policies are available in Nepal?"
        
        with col_q3:
            if st.button("🦠 COVID-19 Response"):
                question = "How did Nepal respond to the COVID-19 pandemic?"
        
        # Search button
        if st.button("🔍 **Get Answer**", type="primary") and question:
            with st.spinner("🔍 Searching through Nepal government documents..."):
                start_time = time.time()
                result = ask_question(question, max_results, language_pref)
                search_time = time.time() - start_time
            
            if result:
                # Display question
                st.markdown(f"""
                <div class="question-box">
                    <h3>❓ Your Question:</h3>
                    <p><strong>{question}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display answer with confidence styling
                confidence_class = "confidence-high" if result['confidence'] >= 0.7 else "confidence-medium" if result['confidence'] >= 0.4 else "confidence-low"
                
                st.markdown(f"""
                <div class="answer-box {confidence_class}">
                    <h3>✅ Answer:</h3>
                    <div style="font-size: 1.1rem; line-height: 1.6;">
                        {result['answer'].replace('**', '<strong>').replace('**', '</strong>').replace('\n', '<br>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence and metrics
                col_conf1, col_conf2, col_conf3, col_conf4 = st.columns(4)
                
                with col_conf1:
                    st.metric("🎯 Confidence", f"{result['confidence']:.1%}")
                
                with col_conf2:
                    st.metric("📚 Sources Found", result['total_sources_found'])
                
                with col_conf3:
                    st.metric("⚡ Processing Time", f"{result['processing_time_ms']:.0f}ms")
                
                with col_conf4:
                    st.metric("📄 Documents Used", len(result['document_coverage']))
                
                # Confidence indicator
                render_confidence_indicator(result['confidence'])
                
                # Sources
                if result['sources']:
                    render_sources(result['sources'])
                
                # Document coverage visualization
                if result['document_coverage'] and documents:
                    st.subheader("📊 Information Analysis")
                    render_document_coverage(result['document_coverage'], documents)
    
    with col2:
        st.header("💡 Tips for Better Results")
        
        st.markdown("""
        **🎯 Ask Specific Questions:**
        - Instead of "Tell me about Nepal"
        - Try "What are Nepal's fundamental rights?"
        
        **📋 Topics We Cover:**
        - Constitutional rights and duties
        - Health services and policies
        - COVID-19 response and vaccination
        - Emergency health protocols
        - Government structure and budget
        - Public service laws and acts
        
        **🌐 Language Support:**
        - Ask in English or Nepali
        - Get answers from both languages
        - Auto-detection works well
        
        **⚡ Performance:**
        - Simple questions: ~200-500ms
        - Complex questions: ~1-3 seconds
        - 1,917+ information sections searched
        """)
        
        # System performance metrics
        if system_info:
            st.subheader("📈 System Stats")
            
            # Create a simple performance chart
            perf_data = {
                'Metric': ['Documents', 'Sections', 'Languages', 'Topics'],
                'Count': [
                    system_info.get('total_documents', 0),
                    system_info.get('total_chunks', 0),
                    2,  # EN, NE
                    len(system_info.get('available_topics', []))
                ]
            }
            
            fig = px.bar(
                perf_data,
                x='Metric',
                y='Count',
                title="System Coverage",
                color='Count',
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()


