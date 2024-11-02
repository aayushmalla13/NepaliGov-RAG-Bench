#!/usr/bin/env python3
"""
🇳🇵 Dynamic Nepal Government Q&A System

A truly dynamic system that:
1. Works with ANY PDF without hardcoding
2. Finds relevant content based on semantic similarity
3. Doesn't prioritize any specific document type
4. Scales automatically when new PDFs are added
"""

import pandas as pd
import re
import time
from typing import List, Dict, Any
from pathlib import Path

def load_corpus():
    """Load corpus dynamically."""
    corpus_path = Path("data/real_corpus.parquet")
    if not corpus_path.exists():
        print("❌ Corpus not found!")
        return None
    
    df = pd.read_parquet(corpus_path)
    print(f"✅ Loaded {len(df)} chunks from {len(df['doc_id'].unique())} documents")
    return df

def generate_dynamic_title(doc_id: str, text_sample: str = "") -> str:
    """
    Generate document title dynamically based on content and filename.
    No hardcoding - works for any PDF.
    """
    # Clean the filename
    title = doc_id.replace('_', ' ').replace('-', ' ')
    title = re.sub(r'\.pdf$', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^[0-9]+\s*', '', title)  # Remove leading numbers
    
    # Capitalize properly
    title = ' '.join(word.capitalize() for word in title.split())
    
    # If we have text sample, try to extract a better title
    if text_sample:
        # Look for document titles in the text
        lines = text_sample.split('\n')[:5]  # First 5 lines
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 100:
                # Check if it looks like a title (no lowercase articles at start)
                if not line.lower().startswith(('the ', 'a ', 'an ', 'to ', 'of ')):
                    if any(keyword in line.lower() for keyword in ['constitution', 'act', 'policy', 'report', 'plan']):
                        return line.title()
    
    return title

def calculate_semantic_similarity(query: str, text: str) -> float:
    """
    Calculate semantic similarity between query and text.
    Uses multiple techniques without hardcoding document types.
    """
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Clean and tokenize
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    text_words = set(re.findall(r'\b\w+\b', text_lower))
    
    score = 0.0
    
    # 1. EXACT PHRASE MATCHING
    if query_lower in text_lower:
        score += 25.0
    
    # 2. MULTI-WORD PHRASE DETECTION
    query_tokens = query_lower.split()
    for i in range(len(query_tokens)):
        for j in range(i + 2, min(i + 6, len(query_tokens) + 1)):  # 2-5 word phrases
            phrase = ' '.join(query_tokens[i:j])
            if phrase in text_lower:
                phrase_length = j - i
                score += 5.0 + (phrase_length * 2.0)  # Longer phrases get higher scores
    
    # 3. WORD OVERLAP SCORING
    if query_words and text_words:
        intersection = query_words.intersection(text_words)
        union = query_words.union(text_words)
        
        # Jaccard similarity
        jaccard = len(intersection) / len(union) if union else 0
        score += jaccard * 15.0
        
        # Coverage (how much of query is covered)
        coverage = len(intersection) / len(query_words)
        score += coverage * 10.0
    
    # 4. CONTEXTUAL RELEVANCE
    # Look for related terms and concepts dynamically
    context_keywords = extract_context_keywords(query_lower)
    context_matches = sum(1 for keyword in context_keywords if keyword in text_lower)
    score += context_matches * 3.0
    
    # 5. STRUCTURAL CONTENT QUALITY
    # Prefer structured content (articles, sections, numbered lists)
    if re.search(r'\(\d+\)', text):  # Numbered clauses like (1), (2)
        score += 4.0
    if re.search(r'Article \d+|Section \d+|Chapter \d+', text, re.IGNORECASE):
        score += 3.0
    if re.search(r'[a-z]\)\s|[0-9]+\.\s', text):  # Lists like a) or 1.
        score += 2.0
    
    # 6. CONTENT LENGTH QUALITY
    if 100 <= len(text) <= 1000:  # Prefer medium-length, substantial content
        score += 2.0
    elif len(text) > 1000:
        score += 1.0
    
    return score

def extract_context_keywords(query: str) -> List[str]:
    """
    Extract related context keywords dynamically.
    No hardcoding of document types.
    """
    keywords = []
    
    # Government/political terms
    if any(word in query for word in ['government', 'political', 'governance', 'policy', 'policies']):
        keywords.extend(['administration', 'state', 'public', 'official', 'law', 'regulation'])
    
    # Rights/legal terms  
    if any(word in query for word in ['rights', 'duties', 'legal', 'law']):
        keywords.extend(['citizen', 'constitutional', 'fundamental', 'obligation', 'freedom'])
    
    # Health terms
    if any(word in query for word in ['health', 'medical', 'hospital', 'treatment']):
        keywords.extend(['healthcare', 'medicine', 'patient', 'clinic', 'therapy'])
    
    # Emergency/crisis terms
    if any(word in query for word in ['emergency', 'crisis', 'disaster', 'covid']):
        keywords.extend(['response', 'management', 'protocol', 'pandemic', 'preparedness'])
    
    return keywords

def dynamic_search(df: pd.DataFrame, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Perform dynamic search without hardcoding document types.
    Works for any content in any PDF.
    """
    print(f"🔍 Searching for: '{query}'")
    
    results = []
    
    for idx, row in df.iterrows():
        text = str(row['text'])
        doc_id = str(row['doc_id'])
        
        # Calculate semantic similarity
        similarity_score = calculate_semantic_similarity(query, text)
        
        # Only include results with meaningful similarity
        if similarity_score > 3.0:
            # Generate dynamic title
            doc_title = generate_dynamic_title(doc_id, text[:200])
            
            result = {
                'doc_id': doc_id,
                'doc_title': doc_title,
                'text': text,
                'similarity_score': round(similarity_score, 2),
                'page_num': int(row.get('page_num', 1)),
                'language': str(row.get('language', 'en')),
                'char_count': len(text)
            }
            results.append(result)
    
    # Sort by similarity score
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    print(f"✅ Found {len(results)} relevant results")
    return results[:max_results]

def generate_dynamic_answer(results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """
    Generate answer dynamically from any relevant content.
    No hardcoding or document type assumptions.
    """
    if not results:
        return {
            'query': query,
            'answer': f"I couldn't find specific information about '{query}' in the available documents. Try rephrasing your question or using different keywords.",
            'confidence': 0.0,
            'sources': [],
            'total_found': 0
        }
    
    # Use top results for answer
    top_results = results[:4]  # Top 4 most relevant
    
    answer_parts = []
    sources = []
    
    for i, result in enumerate(top_results):
        text = result['text']
        doc_title = result['doc_title']
        
        # Extract most relevant portion of text
        relevant_text = extract_relevant_portion(text, query)
        
        # Format answer section
        if i == 0:
            answer_parts.append(f"**According to {doc_title}:**\n{relevant_text}")
        else:
            answer_parts.append(f"**{doc_title} also mentions:**\n{relevant_text}")
        
        # Add to sources
        sources.append({
            'doc_title': doc_title,
            'page_num': result['page_num'],
            'similarity_score': result['similarity_score'],
            'language': result['language']
        })
    
    # Combine answer
    answer = "\n\n".join(answer_parts)
    
    # Add sources
    if sources:
        answer += "\n\n**📚 Sources:**"
        for i, source in enumerate(sources, 1):
            answer += f"\n{i}. {source['doc_title']} (Page {source['page_num']}, Relevance: {source['similarity_score']:.1f})"
    
    # Calculate confidence based on similarity scores
    avg_similarity = sum(r['similarity_score'] for r in top_results) / len(top_results)
    confidence = min(avg_similarity / 25.0, 1.0)  # Normalize to 0-1
    
    return {
        'query': query,
        'answer': answer,
        'confidence': round(confidence, 2),
        'sources': sources,
        'total_found': len(results)
    }

def extract_relevant_portion(text: str, query: str) -> str:
    """
    Extract the most relevant portion of text for the query.
    Dynamic extraction without hardcoding.
    """
    if len(text) <= 300:
        return text
    
    query_words = set(query.lower().split())
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Score each sentence
    sentence_scores = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue
        
        sentence_words = set(sentence.lower().split())
        overlap = len(query_words.intersection(sentence_words))
        score = overlap / len(query_words) if query_words else 0
        
        sentence_scores.append((sentence, score))
    
    # Get top sentences
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 2-3 sentences
    relevant_sentences = [s[0] for s in sentence_scores[:3] if s[1] > 0]
    
    if relevant_sentences:
        result = '. '.join(relevant_sentences) + '.'
        if len(result) > 400:
            result = result[:397] + "..."
        return result
    else:
        # Fallback to first part
        return text[:297] + "..."

def test_dynamic_system():
    """Test the dynamic system with various questions."""
    
    print("🇳🇵 Dynamic Nepal Government Q&A System")
    print("=" * 60)
    print("✨ Features:")
    print("• Works with ANY PDF without hardcoding")
    print("• Dynamic document title generation")
    print("• Semantic similarity matching")
    print("• Scalable for new documents")
    print()
    
    # Load corpus
    df = load_corpus()
    if df is None:
        return
    
    # Test questions
    test_questions = [
        "What are the policies relating to political and governance system of State?",
        "What are the fundamental rights in Nepal?",
        "What health services are available?",
        "How did Nepal respond to COVID-19?",
        "What emergency protocols exist?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"🔍 QUESTION {i}: {question}")
        print("-" * 60)
        
        start_time = time.time()
        
        # Dynamic search
        results = dynamic_search(df, question, max_results=8)
        
        # Generate answer
        answer_data = generate_dynamic_answer(results, question)
        
        processing_time = (time.time() - start_time) * 1000
        
        print(f"⚡ Processing: {processing_time:.0f}ms")
        print(f"📊 Results: {answer_data['total_found']} found")
        print(f"🎯 Confidence: {answer_data['confidence']:.1%}")
        print()
        
        print("📝 **ANSWER:**")
        print(answer_data['answer'][:800] + "..." if len(answer_data['answer']) > 800 else answer_data['answer'])
        print()
        
        if i < len(test_questions):
            input("Press Enter for next question...")

if __name__ == "__main__":
    test_dynamic_system()


