#!/usr/bin/env python3
"""
🇳🇵 Nepal Government Q&A API

Dynamic, robust, and efficient API that can answer questions from ANY PDF
in the Nepal government document collection.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import re
import time
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="🇳🇵 Nepal Government Q&A System",
    description="Dynamic Q&A system for Nepal government documents - Constitution, Health Policies, COVID-19, Emergency Services, and more",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
corpus_data = None
document_stats = {}

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    max_results: int = 5
    include_sources: bool = True
    language_preference: str = "auto"  # auto, en, ne

class SearchResult(BaseModel):
    doc_id: str
    doc_title: str
    page_num: int
    text: str
    relevance_score: float
    language: str
    char_count: int

class Answer(BaseModel):
    question: str
    answer: str
    confidence: float
    sources: List[SearchResult]
    total_sources_found: int
    processing_time_ms: float
    document_coverage: Dict[str, int]  # How many chunks from each document

class DocumentInfo(BaseModel):
    doc_id: str
    title: str
    total_chunks: int
    language: str
    topics: List[str]
    char_count: int

def load_corpus():
    """Load the real corpus with error handling."""
    global corpus_data, document_stats
    
    try:
        corpus_path = Path("data/real_corpus.parquet")
        if not corpus_path.exists():
            logger.error("Real corpus not found!")
            return False
        
        corpus_data = pd.read_parquet(corpus_path)
        logger.info(f"✅ Loaded {len(corpus_data)} chunks from real government PDFs")
        
        # Calculate document statistics
        document_stats = {}
        for doc_id in corpus_data['doc_id'].unique():
            doc_chunks = corpus_data[corpus_data['doc_id'] == doc_id]
            
            # Clean document title
            title = doc_id.replace('_', ' ').replace('-', ' ')
            title = re.sub(r'\.pdf$', '', title, flags=re.IGNORECASE)
            title = title.title()
            
            # Determine topics based on document name and content
            topics = []
            doc_name_lower = doc_id.lower()
            
            if 'constitution' in doc_name_lower:
                topics = ['Constitution', 'Fundamental Rights', 'Government Structure']
            elif 'health' in doc_name_lower or 'medical' in doc_name_lower:
                topics = ['Health Services', 'Medical Care', 'Public Health']
            elif 'covid' in doc_name_lower:
                topics = ['COVID-19', 'Pandemic Response', 'Health Emergency']
            elif 'heoc' in doc_name_lower:
                topics = ['Emergency Services', 'Health Operations', 'Crisis Management']
            elif 'blue' in doc_name_lower:
                topics = ['Budget', 'Government Policy', 'Economic Policy']
            else:
                topics = ['Government Policy', 'Public Services']
            
            document_stats[doc_id] = {
                'title': title,
                'total_chunks': len(doc_chunks),
                'language': doc_chunks['language'].mode().iloc[0] if len(doc_chunks) > 0 else 'en',
                'topics': topics,
                'char_count': doc_chunks['char_count'].sum() if 'char_count' in doc_chunks.columns else len(' '.join(doc_chunks['text'].astype(str)))
            }
        
        logger.info(f"📊 Processed {len(document_stats)} documents")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load corpus: {e}")
        return False

def intelligent_search(query: str, max_results: int = 5, language_preference: str = "auto") -> List[Dict[str, Any]]:
    """
    Intelligent search that can handle questions from ANY PDF.
    Uses multiple scoring strategies for maximum relevance.
    """
    
    if corpus_data is None:
        return []
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    results = []
    
    for idx, row in corpus_data.iterrows():
        text = str(row['text'])
        text_lower = text.lower()
        doc_id = str(row['doc_id'])
        
        # Initialize score
        score = 0.0
        
        # 1. EXACT PHRASE MATCHING (highest priority)
        if query_lower in text_lower:
            score += 20.0
        
        # 2. WORD COVERAGE SCORING
        word_matches = len(query_words.intersection(set(text_lower.split())))
        if len(query_words) > 0:
            coverage = word_matches / len(query_words)
            score += coverage * 15.0
        
        # 3. DOCUMENT TYPE RELEVANCE BOOST
        doc_lower = doc_id.lower()
        
        # Constitution-related queries
        if any(keyword in query_lower for keyword in ['constitution', 'fundamental', 'rights', 'duties', 'citizen', 'article']):
            if 'constitution' in doc_lower or 'fundamental' in doc_lower:
                score += 25.0
            elif any(phrase in text_lower for phrase in ['fundamental rights', 'constitution', 'article', 'right to']):
                score += 15.0
        
        # Health-related queries
        elif any(keyword in query_lower for keyword in ['health', 'medical', 'hospital', 'doctor', 'treatment', 'medicine']):
            if any(keyword in doc_lower for keyword in ['health', 'medical', 'hospital']):
                score += 25.0
            elif any(phrase in text_lower for phrase in ['health service', 'medical', 'hospital', 'treatment']):
                score += 15.0
        
        # COVID-related queries
        elif any(keyword in query_lower for keyword in ['covid', 'coronavirus', 'pandemic', 'vaccination', 'lockdown']):
            if 'covid' in doc_lower or 'coronavirus' in doc_lower:
                score += 25.0
            elif any(phrase in text_lower for phrase in ['covid', 'coronavirus', 'pandemic', 'vaccination']):
                score += 15.0
        
        # Emergency/HEOC queries
        elif any(keyword in query_lower for keyword in ['emergency', 'crisis', 'disaster', 'heoc']):
            if 'heoc' in doc_lower or 'emergency' in doc_lower:
                score += 25.0
            elif any(phrase in text_lower for phrase in ['emergency', 'crisis', 'disaster']):
                score += 15.0
        
        # Government/Policy queries
        elif any(keyword in query_lower for keyword in ['government', 'policy', 'ministry', 'budget', 'law']):
            if any(keyword in doc_lower for keyword in ['government', 'policy', 'ministry', 'blue']):
                score += 20.0
        
        # 4. CONTENT QUALITY SCORING
        # Prefer longer, more substantive content
        if len(text) > 500:
            score += 5.0
        elif len(text) > 200:
            score += 2.0
        
        # Prefer content with specific details (numbers, dates, specific terms)
        if re.search(r'\d{4}', text):  # Years
            score += 2.0
        if re.search(r'Article \d+|Section \d+|Chapter \d+', text, re.IGNORECASE):
            score += 3.0
        
        # 5. LANGUAGE PREFERENCE
        row_lang = str(row.get('language', 'en'))
        if language_preference != "auto":
            if row_lang == language_preference:
                score += 5.0
            else:
                score *= 0.8  # Slight penalty for different language
        
        # Only include results with meaningful relevance
        if score > 1.0:
            result = {
                'doc_id': doc_id,
                'doc_title': document_stats.get(doc_id, {}).get('title', doc_id),
                'page_num': int(row.get('page_num', 1)),
                'text': text,
                'relevance_score': round(score, 2),
                'language': row_lang,
                'char_count': len(text),
                'chunk_id': str(row.get('chunk_id', f"{doc_id}_chunk_{idx}"))
            }
            results.append(result)
    
    # Sort by relevance score and return top results
    results.sort(key=lambda x: x['relevance_score'], reverse=True)
    return results[:max_results * 2]  # Get more results for better answer generation

def generate_comprehensive_answer(search_results: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
    """
    Generate a comprehensive, well-structured answer from search results.
    """
    
    start_time = time.time()
    
    if not search_results:
        return {
            'question': question,
            'answer': """I couldn't find specific information about this topic in the Nepal government documents. 

**Suggestions:**
- Try rephrasing your question with different keywords
- Ask about topics like: fundamental rights, health services, COVID-19 policies, emergency services, government structure
- Be more specific about what aspect you're interested in

**Available Documents Include:**
- Constitution of Nepal (fundamental rights, government structure)
- Health Ministry reports and policies  
- COVID-19 response and vaccination policies
- Emergency health service protocols
- Government budget and policy documents""",
            'confidence': 0.0,
            'sources': [],
            'total_sources_found': 0,
            'processing_time_ms': round((time.time() - start_time) * 1000, 2),
            'document_coverage': {}
        }
    
    # Group results by document for better organization
    doc_groups = {}
    for result in search_results[:8]:  # Use top 8 results
        doc_id = result['doc_id']
        if doc_id not in doc_groups:
            doc_groups[doc_id] = []
        doc_groups[doc_id].append(result)
    
    # Generate answer sections
    answer_parts = []
    sources = []
    document_coverage = {}
    
    # Prioritize high-scoring results
    top_results = search_results[:5]
    
    for i, result in enumerate(top_results):
        text = result['text']
        doc_title = result['doc_title']
        
        # Clean and format text
        if len(text) > 400:
            # Extract most relevant sentences
            sentences = text.split('.')
            relevant_sentences = []
            
            question_words = set(question.lower().split())
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                
                sentence_words = set(sentence.lower().split())
                if len(question_words.intersection(sentence_words)) >= 2:
                    relevant_sentences.append(sentence)
                    if len(relevant_sentences) >= 2:
                        break
            
            if relevant_sentences:
                text = '. '.join(relevant_sentences) + '.'
            else:
                text = text[:397] + "..."
        
        # Format answer section
        if i == 0 and result['relevance_score'] > 15:
            answer_parts.append(f"**According to {doc_title}:**\n{text}")
        else:
            answer_parts.append(f"**{doc_title} also states:**\n{text}")
        
        # Add to sources
        sources.append({
            'doc_id': result['doc_id'],
            'doc_title': doc_title,
            'page_num': result['page_num'],
            'text': text,
            'relevance_score': result['relevance_score'],
            'language': result['language'],
            'char_count': result['char_count']
        })
        
        # Update document coverage
        doc_id = result['doc_id']
        document_coverage[doc_id] = document_coverage.get(doc_id, 0) + 1
    
    # Combine answer parts
    if len(answer_parts) == 1:
        answer = answer_parts[0]
    else:
        answer = "\n\n".join(answer_parts)
    
    # Add source references
    if sources:
        answer += "\n\n**📚 Sources:**"
        for i, source in enumerate(sources, 1):
            answer += f"\n{i}. {source['doc_title']} (Page {source['page_num']}, Relevance: {source['relevance_score']:.1f})"
    
    # Calculate confidence based on relevance scores and coverage
    avg_relevance = sum(r['relevance_score'] for r in top_results) / len(top_results)
    confidence = min(avg_relevance / 20.0, 1.0)  # Normalize to 0-1
    
    processing_time = round((time.time() - start_time) * 1000, 2)
    
    return {
        'question': question,
        'answer': answer,
        'confidence': round(confidence, 2),
        'sources': sources,
        'total_sources_found': len(search_results),
        'processing_time_ms': processing_time,
        'document_coverage': document_coverage
    }

@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup."""
    logger.info("🚀 Starting Nepal Government Q&A API...")
    if load_corpus():
        logger.info("✅ API ready to answer questions from all Nepal government documents!")
    else:
        logger.error("❌ Failed to load document corpus!")

@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "title": "🇳🇵 Nepal Government Q&A System",
        "version": "2.0.0",
        "status": "operational",
        "description": "Ask questions about Nepal's constitution, health policies, COVID-19 response, emergency services, and more",
        "total_documents": len(document_stats) if document_stats else 0,
        "total_chunks": len(corpus_data) if corpus_data is not None else 0,
        "available_topics": ["Constitution & Rights", "Health Services", "COVID-19 Policies", "Emergency Services", "Government Structure", "Budget & Policy"],
        "example_questions": [
            "What are the fundamental rights in Nepal?",
            "What health services are available in Nepal?",
            "What are Nepal's COVID-19 policies?",
            "How is the Nepal government structured?",
            "What emergency health protocols exist?"
        ]
    }

@app.get("/documents", response_model=List[DocumentInfo])
async def get_documents():
    """Get information about all available documents."""
    if not document_stats:
        raise HTTPException(status_code=503, detail="Document corpus not loaded")
    
    return [
        DocumentInfo(
            doc_id=doc_id,
            title=info['title'],
            total_chunks=info['total_chunks'],
            language=info['language'],
            topics=info['topics'],
            char_count=info['char_count']
        )
        for doc_id, info in document_stats.items()
    ]

@app.post("/ask", response_model=Answer)
async def ask_question(request: QueryRequest):
    """
    Ask any question about Nepal government documents.
    The system will intelligently search across ALL documents to find the best answer.
    """
    
    if corpus_data is None:
        raise HTTPException(status_code=503, detail="Document corpus not loaded")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Perform intelligent search
        search_results = intelligent_search(
            request.question, 
            max_results=request.max_results,
            language_preference=request.language_preference
        )
        
        # Generate comprehensive answer
        result = generate_comprehensive_answer(search_results, request.question)
        
        return Answer(**result)
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "corpus_loaded": corpus_data is not None,
        "total_documents": len(document_stats) if document_stats else 0,
        "total_chunks": len(corpus_data) if corpus_data is not None else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("nepal_gov_api:app", host="0.0.0.0", port=8000, reload=True)


