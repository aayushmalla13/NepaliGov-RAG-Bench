#!/usr/bin/env python3
"""
🇳🇵 Enhanced Web App for Nepal Government Q&A
Features:
- Language selection (English/Nepali/Auto)
- Advanced UI with preferences
- Better text processing and display
- Bilingual support
"""

import pandas as pd
import re
import time
import json
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import sys
import os
import unicodedata
import yaml
import hashlib
from datetime import datetime
# Removed auto-reload imports that were causing issues

# Add api directory to path for translation module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))

# Add ui directory to path for font utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ui'))

# Import our dynamic search functions
def load_corpus():
    """Load corpus dynamically."""
    corpus_path = Path("data/real_corpus.parquet")
    if not corpus_path.exists():
        return None
    return pd.read_parquet(corpus_path)

def load_title_mapping():
    """Load proper PDF titles from JSON file."""
    title_file = Path("data/pdf_titles.json")
    if title_file.exists():
        with open(title_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def detect_language(text: str) -> str:
    """Detect if text is primarily Nepali or English."""
    if not text:
        return "en"
    
    # Count Devanagari characters
    nepali_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
    
    # Also check for romanized Nepali patterns (common transliteration patterns)
    romanized_patterns = ['kf', 'sf]', ';+', 'df', 'gf]', 'xf]', 'g]', 'df]', 'tf', 'sf', 'nf', 'rf', 'hf', 'wf']
    romanized_count = sum(1 for pattern in romanized_patterns if pattern in text.lower())
    
    total_chars = len([c for c in text if c.isalpha()])
    
    if total_chars == 0:
        return "en"
    
    # Consider both Devanagari and romanized patterns
    nepali_ratio = (nepali_chars + romanized_count * 2) / total_chars
    return "ne" if nepali_ratio > 0.3 else "en"

def generate_dynamic_title(doc_id: str, text_sample: str = "", title_mapping=None) -> str:
    """Generate document title using proper titles from mapping."""
    if title_mapping and doc_id in title_mapping:
        return title_mapping[doc_id]
    
    # Fallback to dynamic generation
    title = doc_id.replace('_', ' ').replace('-', ' ')
    title = re.sub(r'\.pdf$', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^[0-9]+\s*', '', title)
    return ' '.join(word.capitalize() for word in title.split())

def calculate_semantic_similarity(query: str, text: str) -> float:
    """Calculate semantic similarity with language awareness."""
    query_lower = query.lower()
    text_lower = text.lower()
    
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    text_words = set(re.findall(r'\b\w+\b', text_lower))
    
    score = 0.0
    
    # Detect query and text languages
    query_lang = detect_language(query)
    text_lang = detect_language(text)
    
    # Language matching bonus - prioritize matching language types
    if query_lang == text_lang:
        score += 15.0
    
    # MAJOR BONUS: If query is in Devanagari, heavily prioritize Devanagari text over romanized
    query_has_devanagari = any('\u0900' <= char <= '\u097F' for char in query)
    text_has_devanagari = any('\u0900' <= char <= '\u097F' for char in text)
    
    if query_has_devanagari and text_has_devanagari:
        score += 50.0  # HUGE bonus for Devanagari-to-Devanagari matching
    elif query_has_devanagari and not text_has_devanagari:
        score -= 30.0  # Penalty for Devanagari query matching romanized text
    
    # Exact phrase matching (higher weight for exact matches)
    if query_lower in text_lower:
        score += 30.0
    
    # Multi-word phrases
    query_tokens = query_lower.split()
    for i in range(len(query_tokens)):
        for j in range(i + 2, min(i + 6, len(query_tokens) + 1)):
            phrase = ' '.join(query_tokens[i:j])
            if phrase in text_lower:
                score += 8.0 + ((j - i) * 3.0)
    
    # Word overlap with language detection
    if query_words and text_words:
        intersection = query_words.intersection(text_words)
        union = query_words.union(text_words)
        jaccard = len(intersection) / len(union) if union else 0
        coverage = len(intersection) / len(query_words)
        score += jaccard * 20.0 + coverage * 15.0
    
    # Structural content bonuses
    if re.search(r'\(\d+\)', text):
        score += 5.0
    if re.search(r'Article \d+|Section \d+|धारा \d+', text, re.IGNORECASE):
        score += 4.0
    
    # Quality bonus for proper Devanagari text
    if text_has_devanagari:
        devanagari_ratio = sum(1 for char in text if '\u0900' <= char <= '\u097F') / len(text)
        if devanagari_ratio > 0.1:  # At least 10% Devanagari
            score += 10.0
    
    return score

def dynamic_search(df, query: str, title_mapping=None, max_results: int = 10):
    """Enhanced dynamic search with language awareness."""
    results = []
    
    for _, row in df.iterrows():
        text = str(row['text'])
        doc_id = str(row['doc_id'])
        
        similarity_score = calculate_semantic_similarity(query, text)
        
        if similarity_score > 1.0:  # Minimum threshold
            doc_title = generate_dynamic_title(doc_id, text[:100], title_mapping)
            
            results.append({
                'text': text,
                'doc_id': doc_id,
                'title': doc_title,  # Use 'title' for consistency
                'doc_title': doc_title,  # Keep both for backward compatibility
                'page_num': row.get('page_num', 1),
                'similarity_score': similarity_score
            })
    
    # Sort by similarity score
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results[:max_results]

def clean_text(text: str) -> str:
    """Clean and format text for better display."""
    if not text:
        return text
    
    # Remove control characters but preserve Nepali text
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Clean up multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove table markers
    text = re.sub(r'\[TABLE[^\]]*\]', '', text)
    
    # If text is primarily romanized Nepali, add a note
    has_devanagari = any('\u0900' <= char <= '\u097F' for char in text)
    romanized_patterns = ['kf', 'sf]', ';+', 'df', 'gf]', 'xf]', 'g]']
    has_romanized = sum(1 for pattern in romanized_patterns if pattern in text.lower()) > 3
    
    if not has_devanagari and has_romanized and len(text) > 50:
        # This appears to be romanized Nepali text
        text = f"[Romanized Nepali Text] {text}"
    
    return text.strip()

def extract_relevant_portion(text: str, query: str, language: str = "auto") -> str:
    """Extract relevant text portion with language-aware formatting."""
    # Clean the text first
    text = clean_text(text)
    
    if len(text) <= 400:
        return text
    
    query_words = set(query.lower().split())
    
    # Use appropriate sentence delimiters based on language
    if language == "ne" or detect_language(text) == "ne":
        sentences = re.split(r'[.!?\u0964\u0965]+', text)  # Nepali sentence delimiters
    else:
        sentences = re.split(r'[.!?]+', text)
    
    sentence_scores = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:
            continue
        
        sentence_words = set(sentence.lower().split())
        overlap = len(query_words.intersection(sentence_words))
        score = overlap / len(query_words) if query_words else 0
        
        # Boost score for sentences containing key terms
        if any(word in sentence.lower() for word in query_words):
            score += 0.3
        
        sentence_scores.append((sentence, score))
    
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    relevant_sentences = [s[0] for s in sentence_scores[:4] if s[1] > 0]
    
    if relevant_sentences:
        result = '. '.join(relevant_sentences) + '.'
        return result[:500] + "..." if len(result) > 500 else result
    else:
        return text[:400] + "..."

def generate_answer(results, query: str, language_preference: str = "auto"):
    """Generate answer with language-aware formatting and translation support."""
    if not results:
        return {
            'query': query,
            'answer': f"I couldn't find specific information about '{query}' in the documents.",
            'confidence': 0.0,
            'sources': [],
            'detected_language': detect_language(query),
            'ui_translation_applied': False
        }
    
    detected_lang = detect_language(query)
    top_results = results[:3]
    answer_parts = []
    sources = []
    
    # Check if we have same-language content
    same_lang_sources = 0
    for result in top_results:
        result_lang = detect_language(result['text'])
        if result_lang == detected_lang:
            same_lang_sources += 1
    
    same_lang_citation_rate = same_lang_sources / len(top_results) if top_results else 0
    
    for i, result in enumerate(top_results):
        text = result.get('text', '')
        # Be robust to missing keys
        doc_title = result.get('doc_title') or result.get('title') or 'Unknown Document'
        
        relevant_text = extract_relevant_portion(text, query, detected_lang)
        
        if i == 0:
            answer_parts.append(f"**According to {doc_title}:**\n{relevant_text}")
        else:
            answer_parts.append(f"**{doc_title} also states:**\n{relevant_text}")
        
        sources.append({
            'doc_title': doc_title,
            'doc_id': result.get('doc_id', 'unknown'),
            'page_num': result.get('page_num') or result.get('page') or 1,
            'similarity_score': result.get('similarity_score') or result.get('confidence') or 0.0
        })
    
    # Calculate confidence based on top score and number of results
    top_score = results[0].get('similarity_score') or results[0].get('confidence') or 0.0
    confidence = min(1.0, (top_score / 50.0) * (len(results) / 10.0))
    
    answer = '\n\n'.join(answer_parts)
    answer_language = detect_language(answer)
    
    # Prepare base response
    response = {
        'query': query,
        'answer': answer,
        'confidence': confidence,
        'sources': sources,
        'detected_language': detected_lang,
        'answer_language': answer_language,
        'same_lang_citation_rate': same_lang_citation_rate,
        'processing_info': {
            'total_results': len(results),
            'top_score': top_score
        },
        'ui_translation_applied': False,
        'ui_translation_backend': None,
        'ui_translation_reason': None,
        'original_english': None
    }
    
    # SIMPLIFIED: No automatic translation, just return the answer as-is
    # Users can choose their language preference but we don't force translate
    
    return response

# Load corpus and titles at startup
print("🔍 Loading enhanced corpus...")
import pandas as pd  # Fix pandas import
CORPUS_DF = load_corpus()
TITLE_MAPPING = load_title_mapping()

if CORPUS_DF is not None:
    print(f"✅ Loaded {len(CORPUS_DF)} chunks from {CORPUS_DF['doc_id'].nunique()} documents")
    print(f"✅ Loaded {len(TITLE_MAPPING)} proper document titles")
else:
    print("❌ Failed to load corpus")

class EnhancedHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Health check endpoint for Docker
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "service": "nepali-gov-rag-bench",
                "version": "1.0.0"
            }
            self.wfile.write(json.dumps(health_status).encode('utf-8'))
            return
            
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            # Strong no-cache to ensure UI updates reflect without hard refresh
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            
            # Inject font CSS for CP11.5
            try:
                from utils.fonts import inject_devanagari_fonts
                font_css = inject_devanagari_fonts()
            except ImportError:
                font_css = ""
            
            html = """
<!DOCTYPE html>
<html lang="ne">
<head>
    <title>🇳🇵 Nepal Government Q&A - Enhanced</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    """ + font_css + """
    <style>
        :root {
            --primary-color: #2563eb;
            --primary-hover: #1d4ed8;
            --secondary-color: #64748b;
            --accent-color: #dc2626;
            --success-color: #059669;
            --warning-color: #d97706;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --text-muted: #94a3b8;
            --bg-primary: #ffffff;
            --bg-secondary: #f8fafc;
            --bg-tertiary: #f1f5f9;
            --border-color: #e2e8f0;
            --border-hover: #cbd5e1;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
            --radius-sm: 0.375rem;
            --radius-md: 0.5rem;
            --radius-lg: 0.75rem;
            --spacing-xs: 0.5rem;
            --spacing-sm: 0.75rem;
            --spacing-md: 1rem;
            --spacing-lg: 1.5rem;
            --spacing-xl: 2rem;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body { 
            font-family: 'Inter', 'Noto Sans Devanagari', -apple-system, BlinkMacSystemFont, sans-serif; 
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
            min-height: 100vh;
            line-height: 1.6;
            color: var(--text-primary);
            font-size: 16px;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            padding-bottom: 90px; /* room for fixed footer */
        }
        
        .container { 
            max-width: 1440px; 
            margin: 0 auto; 
            padding: var(--spacing-lg); 
            display: grid;
            grid-template-columns: 1fr 1fr; /* landscape two-column */
            grid-auto-rows: minmax(min-content, max-content);
            gap: var(--spacing-lg);
            align-items: start;
        }
        
        .header { 
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)); 
            color: white; 
            padding: 30px; 
            border-radius: 15px; 
            text-align: center; 
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            position: relative;
        }
        
        .system-status {
            position: absolute;
            top: 15px;
            right: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9em;
            background: rgba(255,255,255,0.2);
            padding: 5px 12px;
            border-radius: 20px;
        }
        
        .status-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4CAF50;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 700;
        }
        
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .language-section {
            background: var(--white);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        }
        
        
        .general-question-section {
            background: linear-gradient(135deg, #fff3e0, #fce4ec);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 25px;
            border: 2px solid #2196F3;
        }
        
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid rgba(0,0,0,0.1);
        }
        
        .section-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.3rem;
            font-weight: 600;
            color: #333;
        }
        
        .section-icon {
            font-size: 1.5rem;
        }
        
        .status-badge {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .status-badge.available {
            background: #4CAF50;
            color: white;
        }
        
        .status-badge.ready {
            background: #2196F3;
            color: white;
        }
        
        .status-badge.processing {
            background: #FF9800;
            color: white;
        }
        
        .section-content {
            padding: 0;
        }
        
        .document-info {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
            padding: 10px;
            background: rgba(255,255,255,0.7);
            border-radius: 8px;
            font-weight: 500;
        }
        
        .question-input-group {
            margin-bottom: 20px;
        }
        
        .question-input-group textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            resize: vertical;
            min-height: 80px;
            font-family: inherit;
            margin-bottom: 15px;
        }
        
        .question-input-group textarea:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .input-actions {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        .language-tools {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            padding: 15px;
            background: rgba(255,255,255,0.5);
            border-radius: 8px;
        }
        
        .quick-question-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 10px;
            margin-bottom: 15px;
        }
        
        
        .question-section { 
            background: var(--white); 
            padding: 30px; 
            border-radius: 15px; 
            margin-bottom: 30px; 
            box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        }
        
        
        .input-group {
            margin-bottom: 20px;
        }
        
        .input-group textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            resize: vertical;
            min-height: 100px;
            font-family: inherit;
        }
        
        .input-group textarea:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .input-buttons {
            display: flex;
            gap: 10px;
            margin-top: 10px;
            flex-wrap: wrap;
        }
        
        .btn-small {
            padding: 8px 16px;
            font-size: 14px;
            border-radius: 6px;
        }
        
        
        
        
        .doc-buttons .btn {
            min-width: 160px;
        }
        
        .language-selector {
            margin-bottom: 20px;
            padding: 15px;
            background: var(--light-bg);
            border-radius: 10px;
        }
        
        .language-selector h4 {
            margin-bottom: 10px;
            color: var(--text-dark);
        }
        
        .lang-options {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .lang-option {
            padding: 8px 16px;
            border: 2px solid var(--border-color);
            border-radius: 25px;
            background: var(--white);
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.9rem;
        }
        
        .lang-option.active {
            background: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }
        
        .lang-option:hover {
            border-color: var(--primary-color);
            transform: translateY(-2px);
        }
        
        .question-input {
            position: relative;
            margin-bottom: 20px;
        }
        
        .input-container {
            position: relative;
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }
        
        input[type="text"] { 
            flex: 1;
            padding: 18px 20px; 
            border: 2px solid var(--border-color); 
            border-radius: 12px; 
            font-size: 16px; 
            font-family: 'Inter', 'Noto Sans Devanagari', Arial, sans-serif;
            transition: all 0.3s ease;
            background: var(--white);
        }
        
        input[type="text"]:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(220, 20, 60, 0.1);
        }
        
        .input-tools {
            display: flex;
            gap: 5px;
        }
        
        .btn-tool {
            padding: 10px 12px;
            background: var(--light-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 14px;
            white-space: nowrap;
        }
        
        .btn-tool:hover {
            background: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }
        
        .nepali-keyboard {
            background: var(--white);
            border: 2px solid var(--border-color);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .keyboard-row {
            display: flex;
            gap: 5px;
            margin-bottom: 8px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .key {
            min-width: 40px;
            height: 40px;
            background: var(--light-bg);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
            font-family: 'Noto Sans Devanagari', Arial, sans-serif;
            font-size: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .key:hover {
            background: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
            transform: translateY(-2px);
        }
        
        .key.wide {
            min-width: 80px;
        }
        
        .translation-badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 8px;
            background: #17a2b8;
            color: white;
            border-radius: 12px;
            font-size: 12px;
            margin-left: 10px;
        }
        
        .original-english-panel {
            border-top: 1px solid var(--border-color);
            margin-top: 20px;
            padding-top: 15px;
        }
        
        .transliteration-active {
            background: var(--success-color) !important;
            color: white !important;
        }
        
        .btn {
            padding: 15px 30px; 
            border: none; 
            border-radius: 12px; 
            font-size: 16px; 
            cursor: pointer; 
            font-weight: 600;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--primary-color), #B91C3C);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(220, 20, 60, 0.3);
        }
        
        .btn-secondary {
            background: var(--light-bg);
            color: var(--text-dark);
            border: 1px solid var(--border-color);
            margin: 5px;
            font-size: 14px;
            padding: 10px 15px;
        }
        
        .btn-secondary:hover {
            background: var(--primary-color);
            color: white;
            border-color: var(--primary-color);
        }
        
        .quick-questions {
            margin-top: 25px;
        }
        
        .quick-questions h4 {
            margin-bottom: 15px;
            color: var(--text-dark);
        }
        
        .answer-section { 
            background: var(--white); 
            padding: 30px; 
            border-radius: 15px; 
            border-left: 5px solid var(--success-color); 
            margin: 30px 0; 
            box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        }
        
        .answer-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .confidence-badge {
            display: inline-flex;
            align-items: center;
            padding: 8px 16px; 
            border-radius: 20px; 
            color: white; 
            font-size: 14px;
            font-weight: 600;
            gap: 5px;
        }
        
        .conf-high { background: var(--success-color); }
        .conf-medium { background: var(--warning-color); color: var(--text-dark); }
        .conf-low { background: var(--danger-color); }
        
        .answer-content { 
            white-space: pre-wrap; 
            line-height: 1.8; 
            font-family: 'Inter', 'Noto Sans Devanagari', Arial, sans-serif;
            background: var(--light-bg); 
            padding: 20px; 
            border-radius: 10px;
            font-size: 15px;
        }
        
        .sources-section { 
            margin-top: 30px; 
            padding: 25px; 
            background: var(--light-bg); 
            border-radius: 12px;
        }
        
        .sources-section h4 {
            margin-bottom: 15px;
            color: var(--text-dark);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .source-item { 
            margin: 15px 0; 
            padding: 15px; 
            background: var(--white); 
            border-radius: 8px;
            border-left: 3px solid var(--primary-color);
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        
        .source-title {
            font-weight: 600;
            color: var(--text-dark);
            margin-bottom: 5px;
        }
        
        .source-meta {
            font-size: 14px;
            color: #6c757d;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        .loading { 
            text-align: center; 
            padding: 40px;
            font-size: 18px;
            color: var(--text-dark);
        }
        
        .loading::after {
            content: '';
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid var(--border-color);
            border-radius: 50%;
            border-top-color: var(--primary-color);
            animation: spin 1s ease-in-out infinite;
            margin-left: 10px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .nepali-text { 
            font-family: 'Noto Sans Devanagari', Arial, sans-serif;
            font-weight: 500;
        }
        
        .error-message {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #f5c6cb;
            margin: 20px 0;
        }
        
        /* Modern Header Styles */
        .modern-header {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-hover) 100%);
            border-radius: var(--radius-lg);
            padding: var(--spacing-xl);
            color: white;
            box-shadow: var(--shadow-lg);
            grid-column: 1 / -1; /* full width at top */
        }
        
        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header-main .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            display: flex;
            align-items: center;
            gap: var(--spacing-md);
        }
        
        .flag-icon {
            font-size: 2.5rem;
        }
        
        .subtitle {
            font-size: 1.125rem;
            opacity: 0.9;
            margin: var(--spacing-sm) 0 0 0;
            font-weight: 400;
        }
        
        .system-status {
            display: flex;
            align-items: center;
            gap: var(--spacing-sm);
            background: rgba(255, 255, 255, 0.1);
            padding: var(--spacing-sm) var(--spacing-md);
            border-radius: var(--radius-md);
            backdrop-filter: blur(10px);
        }
        
        .status-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--success-color);
            box-shadow: 0 0 8px rgba(5, 150, 105, 0.5);
        }
        
        .status-text {
            font-size: 0.875rem;
            font-weight: 500;
        }
        
        /* Feature Card Styles */
        .feature-card {
            background: var(--bg-primary);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-color);
            overflow: hidden;
            transition: all 0.2s ease;
        }
        
        .feature-card:hover {
            box-shadow: var(--shadow-lg);
            transform: translateY(-1px);
        }
        
        .card-header {
            padding: var(--spacing-md);
            border-bottom: 1px solid var(--border-color);
            background: var(--bg-secondary);
        }
        
        .card-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin: 0;
            display: flex;
            align-items: center;
            gap: var(--spacing-sm);
            color: var(--text-primary);
        }
        
        .card-subtitle {
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin: var(--spacing-xs) 0 0 0;
        }
        
        .card-content {
            padding: var(--spacing-md);
        }
        
        .icon {
            font-size: 1.25rem;
        }
        
        /* Language Selector Styles */
        .lang-options {
            display: flex;
            gap: var(--spacing-sm);
            flex-wrap: wrap;
        }
        
        .lang-option {
            display: flex;
            align-items: center;
            gap: var(--spacing-xs);
            padding: 6px 10px;
            border: 2px solid var(--border-color);
            border-radius: var(--radius-md);
            background: var(--bg-primary);
            color: var(--text-secondary);
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            text-decoration: none;
        }
        
        .lang-option:hover {
            border-color: var(--border-hover);
            background: var(--bg-tertiary);
        }
        
        .lang-option.active {
            border-color: var(--primary-color);
            background: var(--primary-color);
            color: white;
        }
        
        .lang-icon {
            font-size: 1.125rem;
        }
        
        .lang-text {
            font-size: 0.875rem;
        }
        
        .help-text {
            background: rgba(37, 99, 235, 0.05);
            border: 1px solid rgba(37, 99, 235, 0.1);
            border-radius: var(--radius-md);
            padding: var(--spacing-md);
            margin-top: var(--spacing-md);
            font-size: 0.875rem;
            display: flex;
            align-items: flex-start;
            gap: var(--spacing-sm);
        }
        
        .help-icon {
            font-size: 1rem;
            margin-top: 1px;
        }
        
        /* Question Input Styles */
        .main-input-card { border: 2px solid var(--primary-color); }

        /* Place key cards in landscape grid: language left, question right, quick questions full width */
        .language-section-card { grid-column: 1; }
        .main-input-card { grid-column: 2; }
        .quick-questions-card { grid-column: 1 / -1; }
        
        .question-input-wrapper {
            display: flex;
            flex-direction: column;
            gap: var(--spacing-lg);
        }

        /* Responsive two-column layout for language + input to fit one screen */
        .two-column {
            display: grid;
            grid-template-columns: 320px 1fr; /* revert horizontal width */
            gap: var(--spacing-xl);
            align-items: start;
        }

        @media (max-width: 1024px) {
            .two-column {
                grid-template-columns: 1fr;
            }
        }
        
        .input-field {
            position: relative;
        }
        
        .modern-textarea {
            width: 100%;
            padding: var(--spacing-md);
            border: 2px solid var(--border-color);
            border-radius: var(--radius-md);
            font-family: 'Noto Sans Devanagari', 'Inter', sans-serif;
            font-size: clamp(0.9rem, 0.85rem + 0.2vw, 1rem);
            line-height: 1.4;
            resize: vertical;
            transition: all 0.2s ease;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 110px; /* compact question box */
        }
        
        .modern-textarea:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }
        
        .input-counter {
            position: absolute;
            bottom: var(--spacing-sm);
            right: var(--spacing-sm);
            font-size: 0.75rem;
            color: var(--text-muted);
            background: var(--bg-primary);
            padding: 2px 6px;
            border-radius: var(--radius-sm);
        }
        
        .input-actions {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: var(--spacing-md);
            flex-wrap: wrap;
        }
        
        .tool-buttons {
            display: flex;
            gap: var(--spacing-sm);
            flex-wrap: wrap;
        }
        
        .tool-btn {
            display: flex;
            align-items: center;
            gap: var(--spacing-xs);
            padding: 6px 10px;
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            background: var(--bg-primary);
            color: var(--text-secondary);
            font-size: 0.82rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .tool-btn:hover {
            border-color: var(--border-hover);
            background: var(--bg-tertiary);
        }
        
        .tool-btn.clear-btn:hover {
            border-color: var(--accent-color);
            color: var(--accent-color);
        }
        
        .primary-btn {
            display: flex;
            align-items: center;
            gap: var(--spacing-sm);
            padding: 10px 16px;
            background: var(--primary-color);
            color: white;
            border: none;
            border-radius: var(--radius-md);
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: var(--shadow-sm);
        }
        
        .primary-btn:hover {
            background: var(--primary-hover);
            box-shadow: var(--shadow-md);
            transform: translateY(-1px);
        }
        
        .btn-icon {
            font-size: 1.125rem;
        }
        
        .btn-text {
            font-size: 1rem;
        }
        
        /* Quick Questions Styles */
        .quick-questions-section {
            display: grid;
            grid-template-columns: 1fr 1fr; /* two columns landscape */
            gap: 10px;
        }
        
        .questions-category {
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding: var(--spacing-sm);
            border-radius: var(--radius-lg);
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
        }
        
        .featured-category {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.05), rgba(37, 99, 235, 0.1));
            border: 2px solid var(--primary-color);
            position: relative;
            overflow: hidden;
        }
        
        .featured-category::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        }
        
        .nepali-category {
            background: linear-gradient(135deg, rgba(220, 38, 38, 0.05), rgba(220, 38, 38, 0.1));
            border: 2px solid var(--accent-color);
        }
        
        .category-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: var(--spacing-sm);
            margin: 0 0 6px 0;
            padding-bottom: 6px;
            border-bottom: 2px solid var(--border-color);
        }
        
        .featured-category .category-title {
            color: var(--primary-color);
        }
        
        .nepali-category .category-title {
            color: var(--accent-color);
        }
        
        .category-icon {
            font-size: 1.5rem;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
        }
        
        .questions-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(180px, 1fr)); /* revert horizontal density */
            gap: var(--spacing-sm);
        }
        
        .featured-grid {
            grid-template-columns: repeat(3, minmax(200px, 1fr));
        }
        
        .question-btn {
            display: flex;
            align-items: center;
            gap: var(--spacing-sm);
            padding: 10px 12px; /* compact */
            border: 1px solid var(--border-color);
            border-radius: 10px;
            background: var(--bg-primary);
            color: var(--text-primary);
            text-align: left;
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 0.8rem; /* smaller */
            font-weight: 500;
            box-shadow: var(--shadow-sm);
            position: relative;
            overflow: hidden;
            min-height: 38px;
        }
        
        .question-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
            transition: left 0.5s ease;
        }
        
        .question-btn:hover::before {
            left: 100%;
        }
        
        .question-btn:hover {
            border-color: var(--primary-color);
            background: rgba(37, 99, 235, 0.08);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }
        
        .featured-btn {
            border: 2px solid var(--primary-color);
            background: linear-gradient(135deg, var(--bg-primary), rgba(37, 99, 235, 0.05));
        }
        
        .featured-btn:hover {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.1), rgba(37, 99, 235, 0.15));
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(37, 99, 235, 0.2);
        }
        
        .nepali-category .question-btn:hover {
            border-color: var(--accent-color);
            background: rgba(220, 38, 38, 0.08);
        }
        
        .q-icon {
            font-size: 1.2rem;
            flex-shrink: 0;
            filter: drop-shadow(0 1px 2px rgba(0,0,0,0.08));
        }
        
        .q-content {
            display: flex;
            flex-direction: column;
            gap: 4px;
            flex: 1;
        }
        
        .q-text {
            font-weight: 600;
            font-size: 0.85rem;
            line-height: 1.2;
        }
        
        .q-desc {
            font-size: 0.72rem;
            color: var(--text-secondary);
            font-weight: 400;
            line-height: 1.1;
        }
        
        /* Enhanced animations */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .featured-category .category-icon {
            animation: pulse 2s ease-in-out infinite;
        }
        
        /* Results Container */
        .results-container {
            grid-column: 1 / -1; /* full width */
            margin-top: var(--spacing-sm);
            min-height: 60px; /* reduce empty space when no results */
        }
        
        /* Footer Styles */
        .app-footer {
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0;
            margin: 0;
            padding: 10px 16px; /* compact height */
            background: linear-gradient(135deg, var(--text-primary), #374151);
            color: white;
            border-radius: 12px 12px 0 0;
            box-shadow: 0 -6px 16px rgba(0,0,0,0.15);
            z-index: 50;
        }
        
        .footer-content {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            padding: 0 8px;
            margin: 0;
        }
        
        .footer-section { display:flex; align-items:center; gap:10px; }
        
        .footer-title { font-size: .95rem; font-weight: 700; margin:0; color:white; display:flex; align-items:center; gap:6px; }
        
        .footer-text { font-size: .8rem; line-height:1.2; opacity:.9; margin:0; }
        
        .footer-links { list-style:none; padding:0; margin:0; display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
        
        .footer-links li { font-size:.8rem; opacity:.85; transition:opacity .2s; cursor:pointer; }
        
        .footer-links li:hover { opacity:1; }
        
        .footer-bottom { display:none; }
        
        .footer-bottom p {
            font-size: 0.8rem;
            opacity: 0.7;
            margin: 0;
        }
        
        @media (max-width: 1024px) {
            .quick-questions-section { grid-template-columns: 1fr; }
            .questions-grid { grid-template-columns: repeat(3, minmax(150px, 1fr)); }
        }

        @media (max-width: 768px) {
            .container { 
                padding: var(--spacing-md); 
                gap: var(--spacing-lg);
            }
            
            .header-content {
                flex-direction: column;
                gap: var(--spacing-md);
                text-align: center;
            }
            
            .main-title {
                font-size: 2rem !important;
            }
            
            .lang-options {
                justify-content: center;
            }
            
            .input-actions {
                flex-direction: column;
                align-items: stretch;
            }
            
            .tool-buttons {
                justify-content: center;
            }
            
            .questions-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Modern Header with Clean Design -->
        <div class="modern-header">
            <div class="header-content">
                <div class="header-main">
                    <h1 class="main-title">
                        <span class="flag-icon">🇳🇵</span>
                        Nepal Government Q&A
                    </h1>
                    <p class="subtitle">Get instant answers from official government documents</p>
                </div>
                <div class="system-status">
                    <div class="status-indicator online"></div>
                    <span class="status-text">System Online</span>
                </div>
            </div>
        </div>

        <!-- Language Selection Card -->
        <div class="feature-card language-section-card">
            <div class="card-header">
                <h3 class="card-title">
                    <span class="icon">🌐</span>
                    Language Preference
                </h3>
            </div>
            <div class="card-content">
                <div class="language-selector">
                    <div class="lang-options">
                        <button class="lang-option active" data-lang="auto">
                            <span class="lang-icon">🔄</span>
                            <span class="lang-text">Auto Detect</span>
                        </button>
                        <button class="lang-option" data-lang="en">
                            <span class="lang-icon">🇺🇸</span>
                            <span class="lang-text">English</span>
                        </button>
                        <button class="lang-option" data-lang="ne">
                            <span class="lang-icon">🇳🇵</span>
                            <span class="lang-text">नेपाली</span>
                        </button>
                    </div>
                    <div id="nepali-help" class="help-text" style="display: none;">
                        <span class="help-icon">💡</span>
                        <strong>Nepali Typing:</strong> Select नेपाली above, then use your system's Nepali keyboard or the virtual keyboard below.
                    </div>
                </div>
            </div>
        </div>

        <!-- Question Input Card -->
        <div class="feature-card main-input-card">
            <div class="card-header">
                <h3 class="card-title">
                    <span class="icon">💬</span>
                    Ask Your Question
                </h3>
            </div>
            <div class="card-content">
                <div class="question-input-wrapper two-column">
                    <div class="input-field">
                        <textarea id="question" 
                                  placeholder="Type your question about government policies, services, or documents..."
                                  rows="4"
                                  class="modern-textarea devanagari-input nepali-text"
                                  autocapitalize="off" 
                                  autocomplete="off" 
                                  spellcheck="false"></textarea>
                        <div class="input-counter">
                            <span id="charCount">0</span>/500 characters
                        </div>
                    </div>
                    
                    <div class="input-actions">
                        <div class="tool-buttons">
                            <button type="button" class="tool-btn" onclick="toggleNepaliKeyboard()" title="Toggle Nepali Keyboard">
                                <span class="btn-icon">🇳🇵</span>
                                <span class="btn-text">नेपाली</span>
                            </button>
                            <button type="button" class="tool-btn" onclick="toggleTransliteration()" title="Toggle Roman→Nepali" id="translitBtn">
                                <span class="btn-icon">🔄</span>
                                <span class="btn-text">रोमन</span>
                            </button>
                            <button type="button" class="tool-btn" onclick="showTransliterationHelp()" title="Show help">
                                <span class="btn-icon">❓</span>
                                <span class="btn-text">Help</span>
                            </button>
                            <button type="button" class="tool-btn clear-btn" onclick="clearInput()" title="Clear input">
                                <span class="btn-icon">🗑️</span>
                                <span class="btn-text">Clear</span>
                            </button>
                        </div>
                        
                        <button type="button" class="primary-btn" onclick="askQuestion()" id="askBtn">
                            <span class="btn-icon">🔍</span>
                            <span class="btn-text">Get Answer</span>
                        </button>
                    </div>
                </div>
                
                <div id="nepali-keyboard" class="nepali-keyboard" style="display: none;">
                    <div class="keyboard-row">
                        <button class="key" onclick="insertText('ज्ञ')">ज्ञ</button>
                        <button class="key" onclick="insertText('१')">१</button>
                        <button class="key" onclick="insertText('२')">२</button>
                        <button class="key" onclick="insertText('३')">३</button>
                        <button class="key" onclick="insertText('४')">४</button>
                        <button class="key" onclick="insertText('५')">५</button>
                        <button class="key" onclick="insertText('६')">६</button>
                        <button class="key" onclick="insertText('७')">७</button>
                        <button class="key" onclick="insertText('८')">८</button>
                        <button class="key" onclick="insertText('९')">९</button>
                        <button class="key" onclick="insertText('०')">०</button>
                    </div>
                    <div class="keyboard-row">
                        <button class="key" onclick="insertText('त')">त</button>
                        <button class="key" onclick="insertText('थ')">थ</button>
                        <button class="key" onclick="insertText('द')">द</button>
                        <button class="key" onclick="insertText('ध')">ध</button>
                        <button class="key" onclick="insertText('न')">न</button>
                        <button class="key" onclick="insertText('प')">प</button>
                        <button class="key" onclick="insertText('फ')">फ</button>
                        <button class="key" onclick="insertText('ब')">ब</button>
                        <button class="key" onclick="insertText('भ')">भ</button>
                        <button class="key" onclick="insertText('म')">म</button>
                    </div>
                    <div class="keyboard-row">
                        <button class="key" onclick="insertText('य')">य</button>
                        <button class="key" onclick="insertText('र')">र</button>
                        <button class="key" onclick="insertText('ल')">ल</button>
                        <button class="key" onclick="insertText('व')">व</button>
                        <button class="key" onclick="insertText('श')">श</button>
                        <button class="key" onclick="insertText('ष')">ष</button>
                        <button class="key" onclick="insertText('स')">स</button>
                        <button class="key" onclick="insertText('ह')">ह</button>
                        <button class="key" onclick="insertText('क्ष')">क्ष</button>
                        <button class="key" onclick="insertText('त्र')">त्र</button>
                    </div>
                    <div class="keyboard-row">
                        <button class="key" onclick="insertText('अ')">अ</button>
                        <button class="key" onclick="insertText('आ')">आ</button>
                        <button class="key" onclick="insertText('इ')">इ</button>
                        <button class="key" onclick="insertText('ई')">ई</button>
                        <button class="key" onclick="insertText('उ')">उ</button>
                        <button class="key" onclick="insertText('ऊ')">ऊ</button>
                        <button class="key" onclick="insertText('ए')">ए</button>
                        <button class="key" onclick="insertText('ै')">ै</button>
                        <button class="key" onclick="insertText('ो')">ो</button>
                        <button class="key" onclick="insertText('औ')">औ</button>
                    </div>
                    <div class="keyboard-row">
                        <button class="key" onclick="insertText('क')">क</button>
                        <button class="key" onclick="insertText('ख')">ख</button>
                        <button class="key" onclick="insertText('ग')">ग</button>
                        <button class="key" onclick="insertText('घ')">घ</button>
                        <button class="key" onclick="insertText('ङ')">ङ</button>
                        <button class="key" onclick="insertText('च')">च</button>
                        <button class="key" onclick="insertText('छ')">छ</button>
                        <button class="key" onclick="insertText('ज')">ज</button>
                        <button class="key" onclick="insertText('झ')">झ</button>
                        <button class="key" onclick="insertText('ञ')">ञ</button>
                    </div>
                    <div class="keyboard-row">
                        <button class="key wide" onclick="insertText(' ')">Space</button>
                        <button class="key" onclick="insertText('।')">।</button>
                        <button class="key" onclick="insertText('?')">?</button>
                        <button class="key wide" onclick="backspace()">⌫ Delete</button>
                    </div>
                </div>
                <!-- Removed duplicate Get Answer button to avoid double actions -->
            </div>
        </div>

        <!-- Quick Questions Card (Collapsible) -->
        <div class="feature-card quick-questions-card" id="quickQuestionsCard">
            <div class="card-header" style="display:flex;align-items:center;justify-content:space-between;gap:8px;">
                <div style="display:flex;align-items:center;gap:10px;">
                    <h3 class="card-title" style="margin:0;">
                        <span class="icon">⚡</span>
                        Quick Questions
                    </h3>
                    <p class="card-subtitle" style="margin:0;">Tap a suggestion or collapse to save space</p>
                </div>
                <button class="tool-btn" id="quickToggleBtn" onclick="toggleQuickQuestions()" title="Collapse/Expand">
                    <span class="btn-icon">▾</span>
                    <span class="btn-text">Collapse</span>
                </button>
            </div>
            <div class="card-content" id="quickQuestionsContent">
                <div class="quick-questions-section">
                    <!-- Popular Questions -->
                    <div class="questions-category featured-category">
                        <h4 class="category-title">
                            <span class="category-icon">⭐</span>
                            Most Asked Questions
                        </h4>
                        <div class="questions-grid featured-grid">
                            <button class="question-btn featured-btn" onclick="setQuestion('What are the fundamental rights in Nepal?')">
                                <span class="q-icon">📜</span>
                                <div class="q-content">
                                    <span class="q-text">Fundamental Rights</span>
                                    <span class="q-desc">Constitutional rights of citizens</span>
                                </div>
                            </button>
                            <button class="question-btn featured-btn" onclick="setQuestion('What health services are available in Nepal?')">
                                <span class="q-icon">🏥</span>
                                <div class="q-content">
                                    <span class="q-text">Health Services</span>
                                    <span class="q-desc">Healthcare system and benefits</span>
                                </div>
                            </button>
                            <button class="question-btn featured-btn" onclick="setQuestion('What are the citizenship requirements in Nepal?')">
                                <span class="q-icon">🆔</span>
                                <div class="q-content">
                                    <span class="q-text">Citizenship Requirements</span>
                                    <span class="q-desc">How to obtain Nepali citizenship</span>
                                </div>
                            </button>
                        </div>
                    </div>
                    
                    <!-- Government & Legal -->
                    <div class="questions-category">
                        <h4 class="category-title">
                            <span class="category-icon">🏛️</span>
                            Government & Legal
                        </h4>
                        <div class="questions-grid">
                            <button class="question-btn" onclick="setQuestion('What are the policies relating to political and governance system of State?')">
                                <span class="q-icon">⚖️</span>
                                <span class="q-text">Governance Policies</span>
                            </button>
                            <button class="question-btn" onclick="setQuestion('What are the electoral laws in Nepal?')">
                                <span class="q-icon">🗳️</span>
                                <span class="q-text">Electoral Laws</span>
                            </button>
                            <button class="question-btn" onclick="setQuestion('What are the local government powers in Nepal?')">
                                <span class="q-icon">🏢</span>
                                <span class="q-text">Local Government</span>
                            </button>
                            <button class="question-btn" onclick="setQuestion('What is the judicial system structure in Nepal?')">
                                <span class="q-icon">👨‍⚖️</span>
                                <span class="q-text">Judicial System</span>
                            </button>
                        </div>
                    </div>
                    
                    <!-- Health & Social Services -->
                    <div class="questions-category">
                        <h4 class="category-title">
                            <span class="category-icon">🏥</span>
                            Health & Social Services
                        </h4>
                        <div class="questions-grid">
                            <button class="question-btn" onclick="setQuestion('PHC funding and allocation of resources')">
                                <span class="q-icon">💰</span>
                                <span class="q-text">PHC Funding</span>
                            </button>
                            <button class="question-btn" onclick="setQuestion('What social security schemes are available in Nepal?')">
                                <span class="q-icon">🛡️</span>
                                <span class="q-text">Social Security</span>
                            </button>
                            <button class="question-btn" onclick="setQuestion('What are the maternal health policies in Nepal?')">
                                <span class="q-icon">🤱</span>
                                <span class="q-text">Maternal Health</span>
                            </button>
                            <button class="question-btn" onclick="setQuestion('What disability benefits are available in Nepal?')">
                                <span class="q-icon">♿</span>
                                <span class="q-text">Disability Benefits</span>
                            </button>
                        </div>
                    </div>
                    
                    <!-- Education & Employment -->
                    <div class="questions-category">
                        <h4 class="category-title">
                            <span class="category-icon">🎓</span>
                            Education & Employment
                        </h4>
                        <div class="questions-grid">
                            <button class="question-btn" onclick="setQuestion('What are the education policies in Nepal?')">
                                <span class="q-icon">📚</span>
                                <span class="q-text">Education Policies</span>
                            </button>
                            <button class="question-btn" onclick="setQuestion('What employment opportunities are provided by the government?')">
                                <span class="q-icon">💼</span>
                                <span class="q-text">Government Jobs</span>
                            </button>
                            <button class="question-btn" onclick="setQuestion('What are the vocational training programs in Nepal?')">
                                <span class="q-icon">🔧</span>
                                <span class="q-text">Vocational Training</span>
                            </button>
                            <button class="question-btn" onclick="setQuestion('What scholarship programs are available for students?')">
                                <span class="q-icon">🎓</span>
                                <span class="q-text">Scholarships</span>
                            </button>
                        </div>
                    </div>
                    
                    <!-- नेपाली प्रश्नहरू -->
                    <div class="questions-category nepali-category">
                        <h4 class="category-title">
                            <span class="category-icon">🇳🇵</span>
                            नेपालीमा प्रश्नहरू
                        </h4>
                        <div class="questions-grid">
                            <button class="question-btn nepali-text" onclick="setQuestion('नेपालमा मौलिक अधिकारहरू के के छन्?')">
                                <span class="q-icon">📜</span>
                                <span class="q-text">मौलिक अधिकार</span>
                            </button>
                            <button class="question-btn nepali-text" onclick="setQuestion('स्वास्थ्य सेवाहरू के छन्?')">
                                <span class="q-icon">🏥</span>
                                <span class="q-text">स्वास्थ्य सेवा</span>
                            </button>
                            <button class="question-btn nepali-text" onclick="setQuestion('शासन व्यवस्था कस्तो छ?')">
                                <span class="q-icon">🏛️</span>
                                <span class="q-text">शासन व्यवस्था</span>
                            </button>
                            <button class="question-btn nepali-text" onclick="setQuestion('संविधानमा के लेखिएको छ?')">
                                <span class="q-icon">📋</span>
                                <span class="q-text">संविधान</span>
                            </button>
                            <button class="question-btn nepali-text" onclick="setQuestion('नागरिकताका लागि के के आवश्यक छ?')">
                                <span class="q-icon">🆔</span>
                                <span class="q-text">नागरिकता</span>
                            </button>
                            <button class="question-btn nepali-text" onclick="setQuestion('शिक्षा नीति के छ नेपालमा?')">
                                <span class="q-icon">📚</span>
                                <span class="q-text">शिक्षा नीति</span>
                            </button>
                            <button class="question-btn nepali-text" onclick="setQuestion('सामाजिक सुरक्षा योजना के छ?')">
                                <span class="q-icon">🛡️</span>
                                <span class="q-text">सामाजिक सुरक्षा</span>
                            </button>
                            <button class="question-btn nepali-text" onclick="setQuestion('स्थानीय सरकारका अधिकार के छन्?')">
                                <span class="q-icon">🏢</span>
                                <span class="q-text">स्थानीय सरकार</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Results Section -->
        <div id="result" class="results-container"></div>
        
        <!-- Footer -->
        <footer class="app-footer">
            <div class="footer-content">
                <div class="footer-section">
                    <h4 class="footer-title">🇳🇵 Nepal Government Q&A</h4>
                    <p class="footer-text">Your trusted source for official government information and services.</p>
                </div>
                <div class="footer-section">
                    <h4 class="footer-title">📚 Resources</h4>
                    <ul class="footer-links">
                        <li>Government Policies</li>
                        <li>Constitutional Rights</li>
                        <li>Public Services</li>
                        <li>Legal Information</li>
                    </ul>
                </div>
                <div class="footer-section">
                    <h4 class="footer-title">🌐 Languages</h4>
                    <ul class="footer-links">
                        <li>English Support</li>
                        <li>नेपाली समर्थन</li>
                        <li>Auto Detection</li>
                        <li>Bilingual Responses</li>
                    </ul>
                </div>
            </div>
            <div class="footer-bottom">
                <p>&copy; 2024 Nepal Government Q&A System. Powered by AI for better citizen services.</p>
            </div>
        </footer>
    </div>
    
    <script>
        let currentLanguage = 'auto';
        
        // Language selection
        document.querySelectorAll('.lang-option').forEach(option => {
            option.addEventListener('click', function() {
                document.querySelectorAll('.lang-option').forEach(opt => opt.classList.remove('active'));
                this.classList.add('active');
                currentLanguage = this.dataset.lang;
                
                // Update placeholder based on language
                const input = document.getElementById('question');
                const placeholders = {
                    'auto': 'Type your question in English or Nepali...',
                    'en': 'Type your question in English...',
                    'ne': 'नेपालीमा आफ्नो प्रश्न टाइप गर्नुहोस्...',
                    'bilingual': 'Ask in any language - get bilingual response...'
                };
                input.placeholder = placeholders[currentLanguage] || placeholders['auto'];

                // CRITICAL FIX: Force native keyboard IME for Nepali
                const helpDiv = document.getElementById('nepali-help');
                if (currentLanguage === 'ne') {
                    // Set up for Nepali input - FORCE IME activation
                    input.setAttribute('lang', 'ne');
                    input.setAttribute('dir', 'ltr');
                    input.style.imeMode = 'active';
                    input.style.fontFamily = "'Noto Sans Devanagari', 'Mangal', Arial, sans-serif";
                    input.style.fontSize = '18px';
                    input.classList.add('nepali-text');
                    
                    // Show help text
                    if (helpDiv) helpDiv.style.display = 'block';
                    
                    // Force focus to trigger IME
                    setTimeout(() => {
                        input.focus();
                        input.click();
                    }, 100);
                    
                } else {
                    // Set up for English input
                    input.setAttribute('lang', 'en');
                    input.style.imeMode = 'auto';
                    input.classList.remove('nepali-text');
                    input.style.fontFamily = "Arial, sans-serif";
                    input.style.fontSize = '16px';
                    
                    // Hide help text
                    if (helpDiv) helpDiv.style.display = 'none';
                }

                // Auto-enable transliteration helper when Nepali selected
                if (currentLanguage === 'ne' && !transliterationEnabled) {
                    toggleTransliteration();
                }
                if (currentLanguage !== 'ne' && transliterationEnabled) {
                    toggleTransliteration();
                }
            });
        });
        
        function setQuestion(q) {
            document.getElementById('question').value = q;
            updateCharCount();
        }
        
        // Character counter functionality
        function updateCharCount() {
            const textarea = document.getElementById('question');
            const counter = document.getElementById('charCount');
            if (textarea && counter) {
                const count = textarea.value.length;
                counter.textContent = count;
                
                // Change color based on character count
                if (count > 400) {
                    counter.style.color = 'var(--accent-color)';
                } else if (count > 300) {
                    counter.style.color = 'var(--warning-color)';
                } else {
                    counter.style.color = 'var(--text-muted)';
                }
            }
        }
        
        // Add event listener for character counting
        document.addEventListener('DOMContentLoaded', function() {
            const textarea = document.getElementById('question');
            if (textarea) {
                textarea.addEventListener('input', updateCharCount);
                updateCharCount(); // Initial count
            }

            // Initialize quick questions collapsed state from sessionStorage
            const saved = sessionStorage.getItem('quickCollapsed');
            if (saved === 'true') {
                setQuickCollapsed(true);
            }
        });

        function toggleQuickQuestions() {
            const currentlyCollapsed = document.getElementById('quickQuestionsContent').style.display === 'none';
            setQuickCollapsed(!currentlyCollapsed);
            sessionStorage.setItem('quickCollapsed', (!currentlyCollapsed).toString());
        }

        function setQuickCollapsed(collapsed) {
            const content = document.getElementById('quickQuestionsContent');
            const btn = document.getElementById('quickToggleBtn');
            if (!content || !btn) return;
            if (collapsed) {
                content.style.display = 'none';
                btn.querySelector('.btn-icon').textContent = '▸';
                btn.querySelector('.btn-text').textContent = 'Expand';
            } else {
                content.style.display = '';
                btn.querySelector('.btn-icon').textContent = '▾';
                btn.querySelector('.btn-text').textContent = 'Collapse';
            }
        }
        
        
        
        
        


        function clearQuestion() {
            document.getElementById('questionInput').value = '';
        }
        
        
        
        
        function askQuestion() {
            // Get question from the main input field
            const question = document.getElementById('question').value.trim();
            if (!question) {
                alert('Please enter a question!');
                return;
            }
            
            // CP11.5: Unicode normalization for consistent processing
            const normalizedQuestion = question.normalize('NFC');
            
            // Update loading message
            document.getElementById('result').innerHTML = '<div class="loading">🔍 Searching all government documents and processing your request...</div>';
            
            const requestData = {
                question: normalizedQuestion,
                language_preference: currentLanguage
            };
            
            fetch('/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json; charset=utf-8'},
                body: JSON.stringify(requestData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('result').innerHTML = 
                        '<div class="error">❌ Error: ' + data.error + '</div>';
                } else {
                    displayResult(data);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('result').innerHTML = 
                    '<div class="error">❌ Network error. Please check your connection and try again.</div>';
            });
        }
        
        function displayResult(data) {
            const confidence = data.confidence || 0;
            let confClass = confidence >= 0.7 ? 'conf-high' : confidence >= 0.4 ? 'conf-medium' : 'conf-low';
            let confText = confidence >= 0.7 ? 'High' : confidence >= 0.4 ? 'Medium' : 'Low';
            let confIcon = confidence >= 0.7 ? '✅' : confidence >= 0.4 ? '⚠️' : '❌';
            
            // Translation badge
            let translationBadge = '';
            let originalEnglishPanel = '';
            
            if (data.ui_translation_applied) {
                translationBadge = `
                    <span class="translation-badge" title="Translated from English: ${data.ui_translation_reason}">
                        🌐 ${data.ui_translation_badge || '[translated]'}
                    </span>
                `;
                
                if (data.original_english || data.original_nepali) {
                    const originalText = data.original_english || data.original_nepali;
                    const originalLang = data.original_english ? 'English' : 'Nepali';
                    
                    // Enhanced view with sentence alignment if available
                    let alignmentView = '';
                    if (data.sentence_alignment && data.sentence_alignment.length > 0) {
                        alignmentView = `
                            <div class="sentence-alignment" style="margin-top: 15px;">
                                <h4>🔍 Sentence-level Alignment (hover to see pairs)</h4>
                                <div class="alignment-pairs">
                                    ${data.sentence_alignment.map((pair, idx) => `
                                        <div class="alignment-pair" style="margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                                            <div class="translated-sentence" 
                                                 style="cursor: pointer; padding: 5px; border-radius: 3px;"
                                                 onmouseover="this.nextElementSibling.style.display='block'"
                                                 onmouseout="this.nextElementSibling.style.display='none'">
                                                <strong>Sentence ${idx + 1}:</strong> ${pair.translated}
                                            </div>
                                            <div class="original-sentence" 
                                                 style="display: none; background: #f0f0f0; padding: 5px; margin-top: 5px; border-radius: 3px; font-style: italic;">
                                                <strong>Original:</strong> ${pair.original}
                                            </div>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        `;
                    }
                    
                    originalEnglishPanel = `
                        <div class="original-text-panel" style="margin-top: 15px;">
                            <details>
                                <summary style="cursor: pointer; font-weight: 600; color: var(--primary-color);">
                                    📄 View Original ${originalLang}
                                </summary>
                                <div style="margin-top: 10px; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 3px solid #007bff;">
                                    <pre class="answer-content">${originalText}</pre>
                                </div>
                                ${alignmentView}
                            </details>
                        </div>
                    `;
                }
            }
            
            let sourcesHtml = '';
            if (data.sources && data.sources.length > 0) {
                sourcesHtml = `
                    <div class="sources-section">
                        <h4>📚 Sources & Citations</h4>
                        ${data.sources.map((source, index) => `
                            <div class="source-item">
                                <div class="source-title">${index + 1}. ${source.doc_title || 'Unknown Document'}</div>
                                <div class="source-meta">
                                    <span>📄 Page ${source.page_num || 'N/A'}</span>
                                    <span>🎯 Relevance: ${(source.similarity_score || 0).toFixed(1)}</span>
                                    <span>📁 Doc ID: ${source.doc_id || 'N/A'}</span>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                `;
            }
            
            let processingInfo = '';
            if (data.processing_info) {
                let langInfo = '';
                if (data.same_lang_citation_rate !== undefined) {
                    langInfo = `, Same-language citations: ${(data.same_lang_citation_rate * 100).toFixed(0)}%`;
                }
                
                processingInfo = `
                    <div style="margin-top: 15px; padding: 10px; background: #e9ecef; border-radius: 5px; font-size: 14px;">
                        <strong>Processing Info:</strong> 
                        Found ${data.processing_info.total_results} relevant chunks, 
                        Top relevance score: ${data.processing_info.top_score.toFixed(1)}, 
                        Detected language: ${data.detected_language === 'ne' ? '🇳🇵 Nepali' : '🇺🇸 English'}${langInfo}
                        ${data.ui_translation_backend ? `, Translation: ${data.ui_translation_backend}` : ''}
                    </div>
                `;
            }
            
            let answerSection = '';
            // SIMPLE: Just show the answer as-is without complex translation logic
            answerSection = `
                <div class="answer-section">
                    <div class="answer-header">
                        <h3>💡 Answer</h3>
                        <div>
                            <span class="confidence-badge ${confClass}">
                                ${confIcon} ${confText} Confidence (${(confidence*100).toFixed(0)}%)
                            </span>
                        </div>
                    </div>
                    <div class="answer-content nepali-text">${data.answer || 'No answer available'}</div>
                    ${processingInfo}
                    ${sourcesHtml}
                </div>
            `;

            let html = answerSection;
            
            document.getElementById('result').innerHTML = html;
        }
        
        // Nepali keyboard functions
        function toggleNepaliKeyboard() {
            const keyboard = document.getElementById('nepali-keyboard');
            const isVisible = keyboard.style.display !== 'none';
            keyboard.style.display = isVisible ? 'none' : 'block';
            
            // Update button text
            const btn = event.target;
            btn.innerHTML = isVisible ? '🇳🇵 नेपाली' : '❌ Close';
        }
        
        function insertText(text) {
            const input = document.getElementById('question');
            const start = input.selectionStart;
            const end = input.selectionEnd;
            const currentValue = input.value;
            
            input.value = currentValue.substring(0, start) + text + currentValue.substring(end);
            input.focus();
            input.setSelectionRange(start + text.length, start + text.length);
        }
        
        function backspace() {
            const input = document.getElementById('question');
            const start = input.selectionStart;
            const end = input.selectionEnd;
            const currentValue = input.value;
            
            if (start > 0) {
                if (start === end) {
                    // No selection, delete one character before cursor
                    input.value = currentValue.substring(0, start - 1) + currentValue.substring(start);
                    input.focus();
                    input.setSelectionRange(start - 1, start - 1);
                } else {
                    // Delete selected text
                    input.value = currentValue.substring(0, start) + currentValue.substring(end);
                    input.focus();
                    input.setSelectionRange(start, start);
                }
            }
        }
        
        function clearInput() {
            document.getElementById('question').value = '';
            document.getElementById('question').focus();
        }
        
        // Transliteration support
        let transliterationEnabled = false;
        
        function toggleTransliteration() {
            transliterationEnabled = !transliterationEnabled;
            const btn = document.getElementById('translitBtn');
            
            if (transliterationEnabled) {
                btn.classList.add('transliteration-active');
                btn.innerHTML = '✅ रोमन→नेपाली';
                btn.title = 'Transliteration ON - Type roman, get Nepali';
            } else {
                btn.classList.remove('transliteration-active');
                btn.innerHTML = '🔄 रोमन';
                btn.title = 'Toggle Roman→Nepali';
            }
        }
        
        function transliterateText(romanText) {
            return fetch('/transliterate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json; charset=utf-8'},
                body: JSON.stringify({text: romanText})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    return data.nepali;
                } else {
                    console.error('Transliteration failed:', data.error);
                    return romanText; // Return original on failure
                }
            })
            .catch(error => {
                console.error('Transliteration error:', error);
                return romanText; // Return original on failure
            });
        }
        
        // Auto-transliteration on space or enter
        let lastInputLength = 0;
        
        function handleTransliteration(event) {
            if (!transliterationEnabled) return;
            
            const input = document.getElementById('question');
            const currentValue = input.value;
            
            // Trigger transliteration on space or enter
            if (event.key === ' ' || event.key === 'Enter') {
                const words = currentValue.split(' ');
                const lastWord = words[words.length - 1].trim();
                
                if (lastWord && lastWord.match(/^[a-zA-Z]+$/)) {
                    // Only transliterate if it's roman characters
                    transliterateText(lastWord).then(nepaliWord => {
                        if (nepaliWord !== lastWord) {
                            words[words.length - 1] = nepaliWord;
                            input.value = words.join(' ');
                            
                            if (event.key === ' ') {
                                input.value += ' '; // Add the space back
                            }
                        }
                    });
                }
            }
        }
        
        function showTransliterationHelp() {
            const helpContent = `
                <div style="max-height: 300px; overflow-y: auto;">
                    <h4>🔄 Transliteration Guide</h4>
                    <p>Type Roman letters and they'll convert to Devanagari:</p>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 14px;">
                        <div><strong>Common Words:</strong></div>
                        <div><strong>Result:</strong></div>
                        <div>nepal</div><div>नेपाल</div>
                        <div>swasthya</div><div>स्वास्थ्य</div>
                        <div>sarkar</div><div>सरकार</div>
                        <div>samvidhan</div><div>संविधान</div>
                        <div>adhikar</div><div>अधिकार</div>
                        <div>seva</div><div>सेवा</div>
                        <div>mantralaya</div><div>मन्त्रालय</div>
                        <div>hospital</div><div>अस्पताल</div>
                        <div>doctor</div><div>डाक्टर</div>
                        <div>budget</div><div>बजेट</div>
                    </div>
                    <p style="margin-top: 15px;"><strong>How to use:</strong></p>
                    <ol style="margin-left: 20px;">
                        <li>Click the "🔄 रोमन" button to enable</li>
                        <li>Type Roman words and press Space</li>
                        <li>Words will auto-convert to Nepali</li>
                        <li>Use the virtual keyboard for direct input</li>
                    </ol>
                </div>
            `;
            
            // Create modal
            const modal = document.createElement('div');
            modal.style.cssText = `
                position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                background: rgba(0,0,0,0.5); z-index: 1000; display: flex; 
                align-items: center; justify-content: center;
            `;
            
            const content = document.createElement('div');
            content.style.cssText = `
                background: white; padding: 30px; border-radius: 15px; 
                max-width: 500px; margin: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            `;
            content.innerHTML = helpContent + `
                <div style="text-align: center; margin-top: 20px;">
                    <button onclick="this.closest('.modal').remove()" 
                            style="padding: 10px 20px; background: #DC143C; color: white; border: none; border-radius: 5px; cursor: pointer;">
                        Got it!
                    </button>
                </div>
            `;
            
            modal.className = 'modal';
            modal.appendChild(content);
            document.body.appendChild(modal);
            
            // Close on background click
            modal.addEventListener('click', (e) => {
                if (e.target === modal) modal.remove();
            });
        }
        
        // Allow Enter key to submit and handle transliteration
        document.getElementById('question').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                askQuestion();
            } else {
                handleTransliteration(e);
            }
        });
        
        // Handle transliteration on keyup for space
        document.getElementById('question').addEventListener('keyup', function(e) {
            if (e.key === ' ') {
                handleTransliteration(e);
            }
        });
        
        // Auto-resize text input
        document.getElementById('question').addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.max(this.scrollHeight, 60) + 'px';
        });
        
        // Auto-detect Nepali input and suggest keyboard
        document.getElementById('question').addEventListener('input', function() {
            const text = this.value;
            const hasDevanagari = /[\u0900-\u097F]/.test(text);
            const keyboard = document.getElementById('nepali-keyboard');
            
            // Auto-show keyboard if user starts typing Nepali
            if (hasDevanagari && keyboard.style.display === 'none') {
                // Optional: Auto-show keyboard when Nepali is detected
                // keyboard.style.display = 'block';
            }
        });
    </script>
</body>
</html>
            """
            
            self.wfile.write(html.encode('utf-8'))
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/ask':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                question = data.get('question', '').strip()
                language_preference = data.get('language_preference', 'auto')
                
                # CP11.5: Unicode normalization for consistent processing
                question = unicodedata.normalize('NFC', question)
                
                if not question:
                    raise ValueError("Question cannot be empty")
                
                if CORPUS_DF is None:
                    raise ValueError("Corpus not loaded")
                
                start_time = time.time()
                results = dynamic_search(CORPUS_DF, question, title_mapping=TITLE_MAPPING)
                answer_data = generate_answer(results, question, language_preference)
                processing_time = (time.time() - start_time) * 1000
                
                answer_data['processing_time_ms'] = round(processing_time, 1)
                answer_data['language_preference'] = language_preference
                
                # CP11.5: Add diagnostics metadata
                answer_data['cp11_5_diagnostics'] = {
                    'unicode_normalized': True,
                    'input_length': len(question),
                    'contains_devanagari': any('\u0900' <= char <= '\u097F' for char in question),
                    'font_support_enabled': True,
                    'translation_system_available': True
                }
                
                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(answer_data, ensure_ascii=False).encode('utf-8'))
                
            except Exception as e:
                error_response = {
                    'query': data.get('question', '') if 'data' in locals() else '',
                    'answer': f'Error processing request: {str(e)}',
                    'confidence': 0.0,
                    'sources': [],
                    'detected_language': 'en',
                    'ui_translation_applied': False
                }
                
                self.send_response(500)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(error_response, ensure_ascii=False).encode('utf-8'))
        
        elif self.path == '/transliterate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                roman_text = data.get('text', '').strip()
                
                # Import and use transliterator
                from translation import get_transliterator
                transliterator = get_transliterator()
                
                nepali_text = transliterator.transliterate(roman_text)
                
                response = {
                    'roman': roman_text,
                    'nepali': nepali_text,
                    'success': True
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response, ensure_ascii=False).encode('utf-8'))
                
            except Exception as e:
                error_response = {
                    'error': str(e),
                    'success': False
                }
                
                self.send_response(500)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(error_response, ensure_ascii=False).encode('utf-8'))
        
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress default logging for cleaner output
        pass

def start_server(port=8090):
    """Start the enhanced web server with auto-restart capability."""
    import socket
    
    # Check if port is available, if not, try next port
    original_port = port
    while True:
        try:
            # Test if port is available
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            test_socket.bind(('0.0.0.0', port))
            test_socket.close()
            break
        except OSError:
            print(f"⚠️  Port {port} is busy, trying {port + 1}...")
            port += 1
            if port > original_port + 10:
                print(f"❌ Could not find available port in range {original_port}-{port}")
                return
    
    try:
        server = HTTPServer(('0.0.0.0', port), EnhancedHandler)
        server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        print(f"🚀 Enhanced Nepal Government Q&A Server starting...")
        print(f"📍 Open your browser to: http://localhost:{port}")
        print(f"🎯 Features: Government Document Search, Language selection")
        print(f"🌐 Supports: English, नेपाली, Virtual Keyboard, and Bilingual responses")
        print(f"🔄 Auto-reload: File changes will restart server automatically")
        print(f"🛑 Press Ctrl+C to stop")
        print()
        
        # Serve with short TCP keepalive to allow fast reuse after restart
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped gracefully")
    except Exception as e:
        print(f"❌ Server error: {e}")
    finally:
        try:
            server.shutdown()
            server.server_close()
        except:
            pass

if __name__ == "__main__":
    if CORPUS_DF is not None:
        print(f"🔍 Enhanced system ready:")
        test_id = '9789240061262-eng'
        print(f"  Sample title: {TITLE_MAPPING.get(test_id, 'NOT FOUND')}")
        print(f"  Language detection: {detect_language('नेपालमा मौलिक अधिकार')}")
        print(f"  Corpus size: {len(CORPUS_DF)} chunks")
        print()
        # Start the server normally (auto-reload handled by launcher script)
        start_server(8093)
    else:
        print("❌ Cannot start server - corpus not loaded")
        print("🔧 Please ensure data/real_corpus.parquet exists")
