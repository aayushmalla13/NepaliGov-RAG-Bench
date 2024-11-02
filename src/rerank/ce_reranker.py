#!/usr/bin/env python3
"""
Cross-Encoder Reranker for NepaliGov-RAG-Bench

Implements cross-encoder reranking with bge-reranker-base or monoT5-small,
supporting AMP, caching, and fusion across expanded queries.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

import torch
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Try to import reranking models with fallbacks
try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class CEReranker:
    """Cross-encoder reranker with caching and AMP support."""
    
    def __init__(self, 
                 model_name: str = "BAAI/bge-reranker-base",
                 cache_dir: Optional[Path] = None,
                 use_amp: bool = True,
                 max_length: int = 512,
                 batch_size: int = 16):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Model name (bge-reranker-base, monoT5-small, etc.)
            cache_dir: Directory for caching rerank scores
            use_amp: Use automatic mixed precision
            max_length: Maximum sequence length
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.use_amp = use_amp
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Initialize cache
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = cache_dir / f"rerank_cache_{model_name.replace('/', '_')}.pkl"
            self._load_cache()
        else:
            self.cache = {}
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_cache(self):
        """Load reranking cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"Loaded rerank cache: {len(self.cache)} entries")
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
                self.cache = {}
        else:
            self.cache = {}
    
    def _save_cache(self):
        """Save reranking cache to disk."""
        if self.cache_dir and self.cache:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
            except Exception as e:
                print(f"Warning: Could not save cache: {e}")
    
    def _load_model(self):
        """Load cross-encoder model with fallbacks."""
        print(f"Loading reranker: {self.model_name}")
        
        # Try SentenceTransformers CrossEncoder first
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = CrossEncoder(
                    self.model_name, 
                    max_length=self.max_length,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                print(f"✅ Loaded with SentenceTransformers: {self.model_name}")
                return
            except Exception as e:
                print(f"SentenceTransformers failed: {e}")
        
        # Fallback to transformers
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                
                print(f"✅ Loaded with transformers: {self.model_name}")
                return
            except Exception as e:
                print(f"Transformers failed: {e}")
        
        # Final fallback - no reranking
        print("⚠️  No reranking libraries available, using identity reranker")
        self.model = None
        self.tokenizer = None
    
    def _get_cache_key(self, query: str, text: str) -> str:
        """Generate cache key for query-text pair."""
        content = f"{query}|||{text}|||{self.model_name}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _predict_batch(self, query_text_pairs: List[Tuple[str, str]]) -> List[float]:
        """Predict relevance scores for batch of query-text pairs."""
        if not self.model:
            # Identity fallback - return neutral scores
            return [0.5] * len(query_text_pairs)
        
        try:
            if hasattr(self.model, 'predict'):
                # SentenceTransformers CrossEncoder
                scores = self.model.predict(query_text_pairs)
                return scores.tolist() if hasattr(scores, 'tolist') else list(scores)
            
            elif self.tokenizer:
                # Manual transformers implementation
                inputs = []
                for query, text in query_text_pairs:
                    inputs.append(f"{query} [SEP] {text}")
                
                # Tokenize
                encoded = self.tokenizer(
                    inputs,
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                )
                
                if torch.cuda.is_available():
                    encoded = {k: v.cuda() for k, v in encoded.items()}
                
                # Inference with AMP if enabled
                with torch.no_grad():
                    if self.use_amp and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            outputs = self.model(**encoded)
                    else:
                        outputs = self.model(**encoded)
                
                # Extract scores
                logits = outputs.logits
                if logits.shape[1] == 1:
                    # Regression model
                    scores = logits.squeeze().cpu().numpy()
                else:
                    # Classification model - use softmax
                    scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                
                return scores.tolist()
            
        except Exception as e:
            print(f"Reranking error: {e}")
            # Fallback to neutral scores
            return [0.5] * len(query_text_pairs)
        
        return [0.5] * len(query_text_pairs)
    
    def rerank(self, 
               query: str, 
               candidates: List[Dict[str, Any]], 
               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: Search query
            candidates: List of candidate documents with 'text' field
            top_k: Number of top results to return (None = all)
        
        Returns:
            Reranked candidates with 'rerank_score' field
        """
        if not candidates:
            return []
        
        # Prepare query-text pairs
        query_text_pairs = []
        cache_keys = []
        cached_scores = []
        uncached_indices = []
        
        for i, candidate in enumerate(candidates):
            text = candidate.get('text', '')
            cache_key = self._get_cache_key(query, text)
            cache_keys.append(cache_key)
            
            if cache_key in self.cache:
                cached_scores.append(self.cache[cache_key])
            else:
                cached_scores.append(None)
                uncached_indices.append(i)
                query_text_pairs.append((query, text))
        
        # Compute scores for uncached pairs
        if query_text_pairs:
            # Process in batches
            new_scores = []
            for i in range(0, len(query_text_pairs), self.batch_size):
                batch = query_text_pairs[i:i + self.batch_size]
                batch_scores = self._predict_batch(batch)
                new_scores.extend(batch_scores)
            
            # Update cache
            for idx, score in zip(uncached_indices, new_scores):
                cache_key = cache_keys[idx]
                self.cache[cache_key] = score
                cached_scores[idx] = score
        
        # Add rerank scores to candidates
        reranked_candidates = []
        for candidate, score in zip(candidates, cached_scores):
            candidate_copy = candidate.copy()
            candidate_copy['rerank_score'] = score if score is not None else 0.5
            reranked_candidates.append(candidate_copy)
        
        # Sort by rerank score
        reranked_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Apply top_k limit
        if top_k:
            reranked_candidates = reranked_candidates[:top_k]
        
        # Save cache periodically
        if len(self.cache) % 100 == 0:
            self._save_cache()
        
        return reranked_candidates
    
    def fusion_rerank(self, 
                     queries: List[str], 
                     candidates: List[Dict[str, Any]], 
                     fusion_method: str = "max",
                     top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank with fusion across multiple queries (e.g., expanded queries).
        
        Args:
            queries: List of queries (original + expanded)
            candidates: Candidate documents
            fusion_method: How to combine scores ('max', 'mean', 'sum')
            top_k: Number of top results
        
        Returns:
            Fusion-reranked candidates
        """
        if not queries or not candidates:
            return []
        
        # Get rerank scores for each query
        all_scores = []
        for query in queries:
            reranked = self.rerank(query, candidates, top_k=None)
            scores = [c['rerank_score'] for c in reranked]
            all_scores.append(scores)
        
        # Fusion
        if fusion_method == "max":
            fusion_scores = [max(scores) for scores in zip(*all_scores)]
        elif fusion_method == "mean":
            fusion_scores = [sum(scores) / len(scores) for scores in zip(*all_scores)]
        elif fusion_method == "sum":
            fusion_scores = [sum(scores) for scores in zip(*all_scores)]
        else:
            fusion_scores = [max(scores) for scores in zip(*all_scores)]
        
        # Create fusion-reranked candidates
        fusion_candidates = []
        for candidate, fusion_score in zip(candidates, fusion_scores):
            candidate_copy = candidate.copy()
            candidate_copy['fusion_rerank_score'] = fusion_score
            candidate_copy['rerank_score'] = fusion_score  # For consistency
            fusion_candidates.append(candidate_copy)
        
        # Sort by fusion score
        fusion_candidates.sort(key=lambda x: x['fusion_rerank_score'], reverse=True)
        
        if top_k:
            fusion_candidates = fusion_candidates[:top_k]
        
        return fusion_candidates
    
    def __del__(self):
        """Save cache on destruction."""
        if hasattr(self, 'cache_dir') and self.cache_dir:
            self._save_cache()


def main():
    """CLI for testing cross-encoder reranker."""
    parser = argparse.ArgumentParser(description="Test cross-encoder reranker")
    parser.add_argument("--model", default="BAAI/bge-reranker-base", help="Reranker model")
    parser.add_argument("--query", required=True, help="Test query")
    parser.add_argument("--texts", nargs="+", required=True, help="Candidate texts")
    parser.add_argument("--cache-dir", type=Path, help="Cache directory")
    parser.add_argument("--top-k", type=int, default=5, help="Top K results")
    
    args = parser.parse_args()
    
    try:
        # Initialize reranker
        reranker = CEReranker(
            model_name=args.model,
            cache_dir=args.cache_dir
        )
        
        # Create candidates
        candidates = [{"text": text, "id": i} for i, text in enumerate(args.texts)]
        
        # Rerank
        reranked = reranker.rerank(args.query, candidates, top_k=args.top_k)
        
        print(f"Query: {args.query}")
        print(f"Reranked results:")
        for i, candidate in enumerate(reranked):
            print(f"  {i+1}. Score: {candidate['rerank_score']:.4f}")
            print(f"     Text: {candidate['text'][:100]}...")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



