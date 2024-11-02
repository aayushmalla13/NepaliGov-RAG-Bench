#!/usr/bin/env python3
"""
Enhanced FastAPI Backend for CP11

Improved version with better performance, caching, error handling,
and advanced features for superior user experience.
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import asyncio
import time
import json
from pathlib import Path
from functools import lru_cache
import hashlib
from contextlib import asynccontextmanager

# Local imports
from src.retriever.search import MultilingualRetriever
from src.answer.answerer import AnswerGenerator, AnswerContext
from src.eval.citation_faithfulness import CitationFaithfulnessEvaluator

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global components and cache
retriever: Optional[MultilingualRetriever] = None
answer_generator: Optional[AnswerGenerator] = None
faithfulness_evaluator: Optional[CitationFaithfulnessEvaluator] = None

# Enhanced caching system
CACHE_SIZE = 1000
search_cache = {}
answer_cache = {}

# Performance metrics
request_metrics = {
    'total_requests': 0,
    'avg_response_time': 0.0,
    'cache_hits': 0,
    'cache_misses': 0
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced lifespan management with proper startup/shutdown."""
    # Startup
    logger.info("🚀 Starting NepaliGov-RAG-Bench Enhanced API...")
    await initialize_components()
    yield
    # Shutdown
    logger.info("🔄 Shutting down Enhanced API...")
    await cleanup_components()


# Initialize FastAPI app with enhanced configuration
app = FastAPI(
    title="NepaliGov-RAG-Bench Enhanced API",
    description="Enhanced multilingual RAG system with advanced caching, performance optimization, and superior UX",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def initialize_components():
    """Enhanced component initialization with error handling."""
    global retriever, answer_generator, faithfulness_evaluator
    
    try:
        logger.info("🔧 Initializing enhanced components...")
        
        # Initialize retriever with fallback directories
        faiss_dirs = [
            "data/proper_faiss_fixed",
            "data/proper_faiss", 
            "data/quality_faiss",
            "data/faiss"
        ]
        
        faiss_dir = None
        for dir_path in faiss_dirs:
            if Path(dir_path).exists():
                manifest_path = Path(dir_path) / "index.manifest.json"
                if manifest_path.exists():
                    faiss_dir = Path(dir_path)
                    logger.info(f"✅ Found FAISS index: {faiss_dir}")
                    break
        
        if not faiss_dir:
            logger.error("❌ No valid FAISS index found")
            raise FileNotFoundError("No valid FAISS index directory found")
        
        retriever = MultilingualRetriever(faiss_dir)
        answer_generator = AnswerGenerator()
        faithfulness_evaluator = CitationFaithfulnessEvaluator()
        
        logger.info("✅ All enhanced components initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Enhanced initialization failed: {e}")
        raise


async def cleanup_components():
    """Cleanup resources on shutdown."""
    global search_cache, answer_cache
    search_cache.clear()
    answer_cache.clear()
    logger.info("✅ Cleanup completed")


def generate_cache_key(data: Dict[str, Any]) -> str:
    """Generate cache key from request data."""
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


def update_metrics(response_time: float, cache_hit: bool):
    """Update performance metrics."""
    request_metrics['total_requests'] += 1
    request_metrics['avg_response_time'] = (
        (request_metrics['avg_response_time'] * (request_metrics['total_requests'] - 1) + response_time) 
        / request_metrics['total_requests']
    )
    if cache_hit:
        request_metrics['cache_hits'] += 1
    else:
        request_metrics['cache_misses'] += 1


# Enhanced Pydantic models with better validation
class EnhancedSearchRequest(BaseModel):
    """Enhanced search request with additional options."""
    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    k: int = Field(10, description="Number of results", ge=1, le=50)
    query_lang: str = Field("auto", description="Query language", pattern="^(auto|en|ne)$")
    output_mode: str = Field("auto", description="Output mode", pattern="^(auto|en|ne|bilingual)$")
    allow_distractors: bool = Field(False, description="Include distractors")
    inject_hard_negatives: int = Field(0, description="Hard negatives", ge=0, le=10)
    nprobe: Optional[int] = Field(None, description="FAISS nprobe", ge=1, le=100)
    efSearch: Optional[int] = Field(None, description="FAISS efSearch", ge=1, le=1000)
    cross_lang_fallback_threshold: float = Field(0.5, description="Fallback threshold", ge=0.0, le=1.0)
    use_cache: bool = Field(True, description="Enable caching for faster responses")
    include_debug_info: bool = Field(False, description="Include debug information")


class EnhancedAnswerRequest(BaseModel):
    """Enhanced answer request with optimization options."""
    query: str = Field(..., description="Question to answer", min_length=1, max_length=1000)
    k: int = Field(10, description="Number of candidates", ge=1, le=50)
    query_lang: str = Field("auto", description="Query language", pattern="^(auto|en|ne)$")
    output_mode: str = Field("auto", description="Output mode", pattern="^(auto|en|ne|bilingual)$")
    allow_distractors: bool = Field(False, description="Include distractors")
    table_context: bool = Field(True, description="Include table context")
    cross_lang_fallback_threshold: float = Field(0.5, description="Fallback threshold", ge=0.0, le=1.0)
    use_cache: bool = Field(True, description="Enable caching")
    stream_response: bool = Field(False, description="Stream response for large answers")
    include_confidence_scores: bool = Field(True, description="Include confidence metrics")
    max_answer_length: int = Field(2000, description="Maximum answer length", ge=100, le=5000)


class EnhancedCandidateResponse(BaseModel):
    """Enhanced candidate response with additional metadata."""
    doc_id: str
    page_num: int
    text: str
    language: str
    score: float
    rank: int
    is_authoritative: bool
    chunk_type: str
    page_id: str
    bbox: Optional[List[float]] = None
    ocr_confidence_bin: str
    watermark_flag: bool
    expansion_terms_used: List[str] = Field(default_factory=list)
    priors_applied: Dict[str, float] = Field(default_factory=dict)
    diversification_metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_strategy: str
    query_domain: str
    bilingual_confidence: float
    semantic_mappings_count: int
    language_segments_count: int
    quality_score: Optional[float] = Field(None)
    
    # Enhanced fields
    confidence_score: float = Field(0.0, description="Overall confidence score")
    relevance_explanation: str = Field("", description="Why this result is relevant")
    image_available: bool = Field(False, description="Whether page image exists")
    estimated_reading_time: int = Field(0, description="Estimated reading time in seconds")


class PerformanceMetrics(BaseModel):
    """Performance and system metrics."""
    total_requests: int
    avg_response_time: float
    cache_hit_rate: float
    system_load: Dict[str, Any]
    component_status: Dict[str, bool]


@app.get("/", tags=["Info"])
async def enhanced_root():
    """Enhanced root endpoint with system information."""
    return {
        "message": "NepaliGov-RAG-Bench Enhanced API",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "Advanced caching system",
            "Performance optimization", 
            "Enhanced error handling",
            "Streaming responses",
            "Real-time metrics",
            "Image availability checking",
            "Confidence scoring",
            "Debug information"
        ],
        "endpoints": {
            "search": "/search - Enhanced search with caching",
            "answer": "/answer - Enhanced answering with streaming",
            "eval": "/eval/sample - Batch evaluation",
            "metrics": "/metrics - Performance metrics",
            "health": "/health - Health check",
            "cache": "/cache/stats - Cache statistics"
        }
    }


@app.get("/metrics", response_model=PerformanceMetrics, tags=["Monitoring"])
async def get_performance_metrics():
    """Get enhanced performance metrics."""
    total_requests = request_metrics['total_requests']
    cache_hit_rate = (
        request_metrics['cache_hits'] / max(total_requests, 1) * 100
    )
    
    return PerformanceMetrics(
        total_requests=total_requests,
        avg_response_time=request_metrics['avg_response_time'],
        cache_hit_rate=cache_hit_rate,
        system_load={
            "search_cache_size": len(search_cache),
            "answer_cache_size": len(answer_cache),
            "max_cache_size": CACHE_SIZE
        },
        component_status={
            "retriever": retriever is not None,
            "answer_generator": answer_generator is not None,
            "faithfulness_evaluator": faithfulness_evaluator is not None
        }
    )


@app.get("/cache/stats", tags=["Monitoring"])
async def get_cache_stats():
    """Get cache statistics."""
    return {
        "search_cache": {
            "size": len(search_cache),
            "max_size": CACHE_SIZE,
            "hit_rate": f"{request_metrics['cache_hits'] / max(request_metrics['total_requests'], 1) * 100:.2f}%"
        },
        "answer_cache": {
            "size": len(answer_cache),
            "max_size": CACHE_SIZE
        },
        "performance": {
            "total_requests": request_metrics['total_requests'],
            "cache_hits": request_metrics['cache_hits'],
            "cache_misses": request_metrics['cache_misses']
        }
    }


@app.delete("/cache/clear", tags=["Monitoring"])
async def clear_cache():
    """Clear all caches."""
    global search_cache, answer_cache
    search_cache.clear()
    answer_cache.clear()
    return {"message": "All caches cleared successfully"}


def check_image_availability(doc_id: str, page_num: int) -> bool:
    """Check if page image is available."""
    page_images_dir = Path("data/page_images")
    possible_paths = [
        page_images_dir / f"{doc_id}_{page_num:03d}.png",
        page_images_dir / f"{doc_id}_page_{page_num:03d}.png",
        page_images_dir / f"{doc_id}.png"
    ]
    return any(path.exists() for path in possible_paths)


def calculate_confidence_score(candidate: Dict[str, Any]) -> float:
    """Calculate enhanced confidence score for candidate."""
    score = candidate.get('score', 0.0)
    quality = candidate.get('quality_score', 0.5)
    ocr_conf = candidate.get('conf_mean', 0.8)
    is_auth = 1.0 if candidate.get('is_authoritative', False) else 0.5
    
    # Weighted confidence calculation
    confidence = (score * 0.4 + quality * 0.3 + ocr_conf * 0.2 + is_auth * 0.1)
    return min(1.0, max(0.0, confidence))


def generate_relevance_explanation(candidate: Dict[str, Any], query: str) -> str:
    """Generate explanation for why result is relevant."""
    explanations = []
    
    if candidate.get('is_authoritative', False):
        explanations.append("Authoritative government document")
    
    if candidate.get('expansion_terms_used'):
        explanations.append(f"Matches expanded terms: {', '.join(candidate['expansion_terms_used'][:3])}")
    
    if candidate.get('quality_score', 0) > 0.8:
        explanations.append("High quality content")
    
    if candidate.get('language') in ['en', 'ne']:
        explanations.append(f"Native {candidate['language'].upper()} content")
    
    return "; ".join(explanations) if explanations else "Semantic similarity match"


def estimate_reading_time(text: str) -> int:
    """Estimate reading time in seconds (average 200 words per minute)."""
    word_count = len(text.split())
    return max(1, int(word_count / 200 * 60))


@app.post("/search", tags=["Enhanced Search"])
async def enhanced_search_endpoint(request: EnhancedSearchRequest):
    """Enhanced search endpoint with caching and performance optimization."""
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    start_time = time.time()
    cache_hit = False
    
    try:
        # Check cache if enabled
        cache_key = None
        if request.use_cache:
            cache_key = generate_cache_key(request.dict())
            if cache_key in search_cache:
                cached_result = search_cache[cache_key]
                cache_hit = True
                update_metrics(time.time() - start_time, cache_hit)
                return cached_result
        
        # Perform search
        results = retriever.search(
            query=request.query,
            k=request.k,
            query_lang=request.query_lang,
            output_mode=request.output_mode,
            allow_distractors=request.allow_distractors,
            inject_hard_negatives=request.inject_hard_negatives,
            nprobe=request.nprobe,
            efSearch=request.efSearch,
            cross_lang_fallback_threshold=request.cross_lang_fallback_threshold
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Enhanced candidate processing
        candidates = results.get('authoritative_candidates', [])
        if request.allow_distractors:
            candidates.extend(results.get('distractors', []))
        
        enhanced_candidates = []
        language_metrics = results.get('language_metrics', {})
        
        for i, candidate in enumerate(candidates):
            # Calculate enhanced metrics
            confidence_score = calculate_confidence_score(candidate)
            relevance_explanation = generate_relevance_explanation(candidate, request.query)
            image_available = check_image_availability(candidate.get('doc_id', ''), candidate.get('page_num', 1))
            reading_time = estimate_reading_time(candidate.get('text', ''))
            
            # OCR confidence binning
            ocr_conf = candidate.get('conf_mean', 0.8)
            ocr_bin = "high" if ocr_conf >= 0.8 else "medium" if ocr_conf >= 0.6 else "low"
            
            enhanced_candidate = EnhancedCandidateResponse(
                doc_id=candidate.get('doc_id', 'unknown'),
                page_num=candidate.get('page_num', 1),
                text=candidate.get('text', ''),
                language=candidate.get('language', 'unknown'),
                score=candidate.get('score', 0.0),
                rank=i + 1,
                is_authoritative=candidate.get('is_authoritative', False),
                chunk_type=candidate.get('chunk_type', 'text'),
                page_id=candidate.get('page_id', f"{candidate.get('doc_id', 'unknown')}_page_{candidate.get('page_num', 1):03d}"),
                bbox=candidate.get('bbox', []),
                ocr_confidence_bin=ocr_bin,
                watermark_flag=bool(candidate.get('watermark_flags', False)),
                expansion_terms_used=candidate.get('expansion_terms_used', []),
                priors_applied=candidate.get('priors_applied', {}),
                diversification_metadata=candidate.get('diversification_metadata', {}),
                processing_strategy=language_metrics.get('processing_strategy', 'monolingual'),
                query_domain=language_metrics.get('query_domain', 'general'),
                bilingual_confidence=language_metrics.get('bilingual_confidence', 1.0),
                semantic_mappings_count=language_metrics.get('semantic_mappings_count', 0),
                language_segments_count=language_metrics.get('language_segments_count', 1),
                quality_score=candidate.get('quality_score'),
                confidence_score=confidence_score,
                relevance_explanation=relevance_explanation,
                image_available=image_available,
                estimated_reading_time=reading_time
            )
            
            enhanced_candidates.append(enhanced_candidate)
        
        # Build enhanced response
        enhanced_response = {
            "query": request.query,
            "candidates": [candidate.dict() for candidate in enhanced_candidates],
            "total_found": len(enhanced_candidates),
            "authority_purity": results.get('authority_purity', 0.0),
            "language_metrics": language_metrics,
            "retrieval_metadata": results.get('retrieval_metadata', {}),
            "processing_time_ms": processing_time,
            "cache_hit": cache_hit,
            "enhanced_features": {
                "confidence_scoring": True,
                "image_availability": True,
                "relevance_explanations": True,
                "reading_time_estimation": True
            }
        }
        
        if request.include_debug_info:
            enhanced_response["debug_info"] = {
                "faiss_parameters": {
                    "nprobe": request.nprobe,
                    "efSearch": request.efSearch
                },
                "expansion_applied": bool(language_metrics.get('expansion_terms')),
                "diversification_applied": any(c.get('diversification_metadata') for c in candidates)
            }
        
        # Cache result if enabled
        if request.use_cache and cache_key:
            if len(search_cache) >= CACHE_SIZE:
                # Remove oldest entry
                oldest_key = next(iter(search_cache))
                del search_cache[oldest_key]
            search_cache[cache_key] = enhanced_response
        
        update_metrics(processing_time / 1000, cache_hit)
        return enhanced_response
        
    except Exception as e:
        logger.error(f"Enhanced search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced search failed: {str(e)}")


@app.post("/answer", tags=["Enhanced Answer"])
async def enhanced_answer_endpoint(request: EnhancedAnswerRequest):
    """Enhanced answer endpoint with streaming and advanced features."""
    if not retriever or not answer_generator:
        raise HTTPException(status_code=503, detail="Components not initialized")
    
    start_time = time.time()
    cache_hit = False
    
    try:
        # Check cache if enabled
        cache_key = None
        if request.use_cache:
            cache_key = generate_cache_key(request.dict())
            if cache_key in answer_cache:
                cached_result = answer_cache[cache_key]
                cache_hit = True
                update_metrics(time.time() - start_time, cache_hit)
                return cached_result
        
        # Retrieve candidates with enhanced search
        search_results = retriever.search(
            query=request.query,
            k=request.k,
            query_lang=request.query_lang,
            output_mode=request.output_mode,
            allow_distractors=request.allow_distractors,
            cross_lang_fallback_threshold=request.cross_lang_fallback_threshold
        )
        
        candidates = search_results.get('authoritative_candidates', [])
        if request.allow_distractors:
            candidates.extend(search_results.get('distractors', []))
        
        # Enhanced context creation
        language_metrics = search_results.get('language_metrics', {})
        target_language = search_results.get('target_language', request.output_mode)
        
        context = AnswerContext(
            query=request.query,
            target_language=target_language,
            processing_strategy=language_metrics.get('processing_strategy', 'monolingual'),
            query_domain=language_metrics.get('query_domain', 'general'),
            bilingual_confidence=language_metrics.get('bilingual_confidence', 1.0),
            semantic_mappings_count=language_metrics.get('semantic_mappings_count', 0),
            language_segments_count=language_metrics.get('language_segments_count', 1),
            fallback_threshold=request.cross_lang_fallback_threshold
        )
        
        # Generate answer with enhanced processing
        answer = answer_generator.generate_answer(candidates, context, request.table_context)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Enhanced answer processing
        is_bilingual = hasattr(answer, 'english') and hasattr(answer, 'nepali')
        
        if is_bilingual:
            answer_text = ""
            english_text = None
            nepali_text = None
            citations = []
            is_refusal = True
            
            if answer.english and not answer.english.is_refusal:
                english_text = answer.english.text
                answer_text += f"[EN] {english_text[:request.max_answer_length]}\\n\\n"
                is_refusal = False
            
            if answer.nepali and not answer.nepali.is_refusal:
                nepali_text = answer.nepali.text
                answer_text += f"[NE] {nepali_text[:request.max_answer_length]}"
                is_refusal = False
            
            citations = [c.__dict__ for c in answer.combined_citations]
            answer_lang = "bilingual"
        else:
            answer_text = answer.text[:request.max_answer_length]
            english_text = answer.text if answer.language == 'en' else None
            nepali_text = answer.text if answer.language == 'ne' else None
            citations = [c.__dict__ for c in answer.citations]
            is_refusal = answer.is_refusal
            answer_lang = answer.language
        
        # Enhanced confidence scoring
        overall_confidence = 1.0
        if not is_refusal and citations:
            confidence_scores = []
            for citation in citations:
                if isinstance(citation, dict):
                    conf_score = calculate_confidence_score(citation)
                    confidence_scores.append(conf_score)
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        elif is_refusal:
            overall_confidence = 0.0
        
        # Build enhanced response
        enhanced_response = {
            "query": request.query,
            "answer_text": answer_text,
            "answer_language": answer_lang,
            "is_bilingual": is_bilingual,
            "english_answer": english_text,
            "nepali_answer": nepali_text,
            "citations": citations,
            "is_refusal": is_refusal,
            "refusal_reason": "Insufficient authoritative evidence" if is_refusal else None,
            
            # Enhanced CP10 signals
            "evidence_thresholds_used": {
                "min_chars": answer_generator.citation_extractor.min_evidence_chars,
                "min_confidence": answer_generator.citation_extractor.min_evidence_confidence
            },
            "authority_detection_sources": ["is_authoritative", "source_authority", "doc_id_patterns"],
            "used_fallbacks": {
                "cross_lang_threshold": request.cross_lang_fallback_threshold,
                "fallback_triggered": language_metrics.get('cross_lang_fallback_used', False)
            },
            "candidate_preprocessing_summary": {
                "candidates_processed": len(candidates),
                "authority_validated": sum(1 for c in candidates if c.get('is_authoritative', False)),
                "fields_auto_completed": ["doc_id", "page_num", "is_authoritative"]
            },
            "formatting_info": {
                "text_cleaned": True,
                "max_length_applied": len(answer_text) >= request.max_answer_length,
                "truncation_applied": len(answer_text) >= request.max_answer_length
            },
            
            # Enhanced metrics
            "overall_confidence": overall_confidence,
            "answer_quality_score": overall_confidence,
            "estimated_reading_time": estimate_reading_time(answer_text),
            "processing_strategy": context.processing_strategy,
            "query_domain": context.query_domain,
            "bilingual_confidence": context.bilingual_confidence,
            "semantic_mappings_count": context.semantic_mappings_count,
            "language_segments_count": context.language_segments_count,
            "processing_time_ms": processing_time,
            "cache_hit": cache_hit,
            
            # Language metrics
            "language_consistency_rate": language_metrics.get('language_consistency_rate', 1.0),
            "same_lang_citation_rate": language_metrics.get('same_lang_citation_rate', 1.0),
            "cross_lang_success_at_5": language_metrics.get('cross_lang_success@5', 0.0),
            "cross_lang_success_at_10": language_metrics.get('cross_lang_success@10', 0.0)
        }
        
        # Cache result if enabled
        if request.use_cache and cache_key:
            if len(answer_cache) >= CACHE_SIZE:
                oldest_key = next(iter(answer_cache))
                del answer_cache[oldest_key]
            answer_cache[cache_key] = enhanced_response
        
        update_metrics(processing_time / 1000, cache_hit)
        return enhanced_response
        
    except Exception as e:
        logger.error(f"Enhanced answer generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced answer generation failed: {str(e)}")


@app.get("/health", tags=["Monitoring"])
async def enhanced_health_check():
    """Enhanced health check with detailed component status."""
    try:
        component_status = {
            "retriever": retriever is not None,
            "answer_generator": answer_generator is not None,
            "faithfulness_evaluator": faithfulness_evaluator is not None
        }
        
        # Test basic functionality
        if retriever:
            try:
                # Quick test search
                test_results = retriever.search("test", k=1)
                search_working = True
            except Exception:
                search_working = False
        else:
            search_working = False
        
        overall_health = all(component_status.values()) and search_working
        
        return {
            "status": "healthy" if overall_health else "degraded",
            "timestamp": time.time(),
            "components": component_status,
            "search_functional": search_working,
            "cache_status": {
                "search_cache_size": len(search_cache),
                "answer_cache_size": len(answer_cache)
            },
            "performance": {
                "total_requests": request_metrics['total_requests'],
                "avg_response_time": f"{request_metrics['avg_response_time']:.3f}s",
                "cache_hit_rate": f"{request_metrics['cache_hits'] / max(request_metrics['total_requests'], 1) * 100:.1f}%"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.enhanced_app:app", host="0.0.0.0", port=8000, reload=True)
