#!/usr/bin/env python3
"""
Working API for Nepal Government Q&A

Uses the real corpus we just created from all PDFs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

import pandas as pd
import json
import re
from typing import List, Dict, Any

# Load the real corpus
print("🔍 Loading real corpus...")
try:
    corpus_df = pd.read_parquet("data/real_corpus.parquet")
    print(f"✅ Loaded {len(corpus_df)} chunks from real PDFs")
except Exception as e:
    print(f"❌ Could not load corpus: {e}")
    sys.exit(1)

def search_corpus(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Search the real corpus for relevant chunks."""
    
    query_lower = query.lower()
    query_words = query_lower.split()
    
    results = []
    
    for idx, row in corpus_df.iterrows():
        text = str(row['text']).lower()
        doc_id = str(row['doc_id'])
        
        # Calculate relevance score
        score = 0.0
        
        # Exact phrase match (high score)
        if query_lower in text:
            score += 10.0
        
        # Word matches
        word_matches = 0
        for word in query_words:
            if word in text:
                word_matches += 1
                score += 1.0
        
        # Bonus for more word matches
        if len(query_words) > 1:
            coverage = word_matches / len(query_words)
            score += coverage * 5.0
        
        # Bonus for constitution-related content
        if any(keyword in text for keyword in ['constitution', 'fundamental', 'rights', 'duties']):
            if any(keyword in query_lower for keyword in ['constitution', 'fundamental', 'rights']):
                score += 3.0
        
        # Bonus for health-related content
        if any(keyword in text for keyword in ['health', 'medical', 'hospital', 'treatment']):
            if any(keyword in query_lower for keyword in ['health', 'medical', 'hospital', 'covid']):
                score += 3.0
        
        # Only include results with some relevance
        if score > 0:
            result = {
                'doc_id': doc_id,
                'text': str(row['text']),
                'score': score,
                'page_num': int(row.get('page_num', 1)),
                'chunk_id': str(row['chunk_id']),
                'language': str(row.get('language', 'en')),
                'char_count': len(str(row['text'])),
                'is_authoritative': True
            }
            results.append(result)
    
    # Sort by score and return top k
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:k]

def generate_answer(candidates: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """Generate an answer from the search results."""
    
    if not candidates:
        return {
            'query': query,
            'answer': "I couldn't find specific information about this topic in the Nepal government documents. Please try rephrasing your question or ask about topics like fundamental rights, health services, or government policies.",
            'sources': [],
            'is_refusal': True
        }
    
    # Generate answer from top candidates
    answer_parts = []
    sources = []
    
    for i, candidate in enumerate(candidates[:3]):  # Use top 3
        text = candidate['text']
        doc_id = candidate['doc_id']
        
        # Clean and summarize text if too long
        if len(text) > 300:
            # Try to find the most relevant sentence
            sentences = text.split('.')
            relevant_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                
                # Check if sentence contains query words
                if any(word.lower() in sentence.lower() for word in query.split()):
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                text = '. '.join(relevant_sentences[:2])
            else:
                text = text[:297] + "..."
        
        answer_parts.append(f"• {text}")
        sources.append({
            'doc_id': doc_id,
            'page_num': candidate['page_num'],
            'score': candidate['score']
        })
    
    # Combine answer
    if len(answer_parts) == 1:
        answer = f"According to the Nepal government documents:\n\n{answer_parts[0]}"
    else:
        answer = f"Based on the Nepal government documents:\n\n" + "\n\n".join(answer_parts)
    
    # Add source references
    if sources:
        answer += "\n\n**Sources:**"
        for i, source in enumerate(sources, 1):
            doc_name = source['doc_id'].replace('_', ' ').replace('-', ' ').title()
            answer += f"\n[{i}] {doc_name} (Page {source['page_num']})"
    
    return {
        'query': query,
        'answer': answer,
        'sources': sources,
        'is_refusal': False,
        'total_candidates': len(candidates)
    }

def main():
    """Interactive Q&A system."""
    
    print("\n🇳🇵 Nepal Government Q&A System")
    print("=" * 50)
    print("Ask questions about Nepal's constitution, health policies, government services, etc.")
    print("Type 'quit' to exit.")
    print()
    
    # Test queries
    test_queries = [
        "What are the fundamental rights in Nepal?",
        "What health services are available in Nepal?", 
        "What are the COVID-19 policies in Nepal?",
        "What is the structure of Nepal government?",
        "What are emergency health protocols?"
    ]
    
    print("📋 Try these example questions:")
    for i, q in enumerate(test_queries, 1):
        print(f"{i}. {q}")
    print()
    
    while True:
        try:
            query = input("❓ Your question: ").strip()
            
            if not query or query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            print(f"\n🔍 Searching for: '{query}'")
            
            # Search corpus
            candidates = search_corpus(query, k=10)
            
            if candidates:
                print(f"✅ Found {len(candidates)} relevant documents")
                
                # Show top candidate info
                top_candidate = candidates[0]
                print(f"📄 Best match: {top_candidate['doc_id']} (Score: {top_candidate['score']:.1f})")
            
            # Generate answer
            result = generate_answer(candidates, query)
            
            print(f"\n📝 **Answer:**")
            print(result['answer'])
            print()
            print("-" * 50)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()


