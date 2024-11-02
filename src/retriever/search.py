#!/usr/bin/env python3
"""
Multilingual Retrieval & Search for NepaliGov-RAG-Bench

Authority-aware, quality-aware retrieval with distractor support and CP5 compatibility.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

import numpy as np
import pandas as pd

# Optional imports with fallbacks
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Search functionality limited.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available.")

# Local imports
from src.lang.query_language import QueryLanguageDetector
from src.lang.target_language import TargetLanguageSelector
from src.lang.advanced_bilingual import AdvancedBilingualProcessor

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class MultilingualRetriever:
    """CP5-aware multilingual retriever with authority and quality priors."""
    
    def __init__(self, 
                 faiss_dir: Path,
                 query_lang_detector: Optional[QueryLanguageDetector] = None,
                 target_lang_selector: Optional[TargetLanguageSelector] = None):
        """
        Initialize retriever.
        
        Args:
            faiss_dir: Directory containing FAISS index artifacts
            query_lang_detector: Optional language detector
            target_lang_selector: Optional target language selector
        """
        self.faiss_dir = Path(faiss_dir)
        self.query_lang_detector = query_lang_detector or QueryLanguageDetector()
        self.target_lang_selector = target_lang_selector or TargetLanguageSelector(self.query_lang_detector)
        self.bilingual_processor = AdvancedBilingualProcessor(self.query_lang_detector, self.target_lang_selector)
        
        # Load artifacts
        self.manifest = self._load_manifest()
        self.meta_df = self._load_metadata()
        self.indices = self._load_indices()
        self.model = self._load_embedding_model()
        
        # Configuration (tunable priors)
        self.same_lang_boost = 0.15  # Increased for better language preference
        self.authority_boost = 0.08  # Increased for stronger authority preference
        self.watermark_penalty = 0.05  # Increased penalty for watermarked content
        self.table_boost = 0.04  # Increased for tabular queries
        self.quality_tier_boosts = {
            'high': 0.06,
            'medium': 0.02,
            'low': -0.02,
            'recovery': -0.04
        }
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load index manifest."""
        manifest_path = self.faiss_dir / "index.manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load metadata DataFrame."""
        meta_path = self.faiss_dir / "meta.parquet"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        
        return pd.read_parquet(meta_path)
    
    def _load_indices(self) -> List[Any]:
        """Load FAISS indices (single or sharded)."""
        if not FAISS_AVAILABLE:
            return []
        
        indices = []
        
        # Check for sharded indices
        shard_files = list(self.faiss_dir.glob("shard_*.bin"))
        if shard_files:
            print(f"Loading {len(shard_files)} sharded indices...")
            for shard_file in sorted(shard_files):
                index = faiss.read_index(str(shard_file))
                indices.append(index)
        else:
            # Single index
            index_path = self.faiss_dir / "index.bin"
            if index_path.exists():
                index = faiss.read_index(str(index_path))
                indices.append(index)
            else:
                raise FileNotFoundError(f"No FAISS index found in {self.faiss_dir}")
        
        return indices
    
    def _load_embedding_model(self) -> Optional[Any]:
        """Load embedding model if available."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return None
        
        model_name = self.manifest.get('model_name', 'bge-m3')
        
        # Map model names
        model_map = {
            'bge-m3': 'BAAI/bge-m3',
            'e5-base': 'intfloat/e5-base-v2',
            'bge-base': 'BAAI/bge-base-en-v1.5'
        }
        
        model_id = model_map.get(model_name, model_name)
        
        try:
            return SentenceTransformer(model_id)
        except Exception as e:
            print(f"Warning: Could not load model {model_id}: {e}")
            return None
    
    def _is_tabular_query(self, query: str) -> bool:
        """Enhanced heuristic to detect tabular/numeric queries."""
        # Look for numeric patterns, table-related keywords
        numeric_ratio = len(re.findall(r'\d', query)) / max(len(query), 1)
        table_keywords = ['table', 'टेबल', 'सूची', 'list', 'data', 'statistics', 
                         'numbers', 'count', 'total', 'sum', 'chart', 'graph',
                         'संख्या', 'तालिका', 'डाटा', 'आँकडा', 'गणना']
        
        has_table_keywords = any(kw in query.lower() for kw in table_keywords)
        
        # Enhanced patterns
        has_comparison = any(word in query.lower() for word in ['compare', 'vs', 'versus', 'तुलना'])
        has_quantifiers = any(word in query.lower() for word in ['how many', 'कति', 'संख्या'])
        
        return numeric_ratio > 0.08 or has_table_keywords or has_comparison or has_quantifiers
    
    def _expand_query_terms(self, query: str, query_lang: str, target_langs: List[str] = None) -> Dict[str, List[str]]:
        """Enhanced language-aware query expansion with advanced bilingual processing."""
        if target_langs is None:
            if query_lang == 'mixed':
                target_langs = ['ne', 'en']
            else:
                target_langs = [query_lang]
        
        # Use advanced bilingual processing for complex queries
        try:
            bilingual_analysis = self.bilingual_processor.analyze_bilingual_query(query)
            
            # If we have semantic mappings or bilingual segments, use enhanced processing
            if (bilingual_analysis.semantic_mappings or 
                bilingual_analysis.processing_strategy in ['bilingual_segmented', 'semantic_expansion']):
                
                enhanced_expansions = self.bilingual_processor.generate_enhanced_expansions(bilingual_analysis)
                
                # Filter to target languages and add original query
                expanded_by_lang = {}
                for lang in target_langs:
                    if lang in enhanced_expansions:
                        expanded_by_lang[lang] = enhanced_expansions[lang][:4]  # Limit to 4
                    else:
                        expanded_by_lang[lang] = [query]
                
                return expanded_by_lang
        
        except Exception as e:
            # Fallback to original method on error
            print(f"Advanced bilingual processing failed: {e}, falling back to basic expansion")
        
        # Fallback: Original domain-specific expansions
        health_expansions = {
            'ne': {
                'स्वास्थ्य': ['चिकित्सा', 'उपचार', 'रोग'],
                'आपतकालीन': ['संकटकाल', 'विपद्', 'दुर्घटना'],
                'सरकार': ['राज्य', 'प्रशासन', 'मन्त्रालय'],
                'संविधान': ['कानून', 'ऐन', 'नियम']
            },
            'en': {
                'health': ['medical', 'healthcare', 'treatment', 'disease'],
                'emergency': ['crisis', 'disaster', 'urgent', 'critical'],
                'government': ['administration', 'ministry', 'department'],
                'constitution': ['law', 'act', 'regulation', 'legal']
            }
        }
        
        expanded_by_lang = {}
        
        for lang in target_langs:
            expanded_terms = [query]  # Always include original
            
            if lang in health_expansions:
                expansions = health_expansions[lang]
                words = query.lower().split()
                
                for word in words:
                    if word in expansions:
                        # Add one synonym to avoid over-expansion
                        if expansions[word]:
                            synonym = expansions[word][0]
                            expanded_query = query.lower().replace(word, synonym)
                            expanded_terms.append(expanded_query)
                            break  # Only expand one term to keep focused
            
            expanded_by_lang[lang] = expanded_terms[:3]  # Limit to avoid noise
        
        return expanded_by_lang
    
    def _diversify_results(self, candidates: List[Dict], max_results: int) -> List[Dict]:
        """Diversify results to avoid too many from the same page/document."""
        if not candidates:
            return candidates
        
        diversified = []
        page_counts = {}
        doc_counts = {}
        
        for candidate in candidates:
            page_id = candidate.get('page_id', '')
            doc_id = candidate.get('doc_id', '')
            
            # Limits: max 3 per page, max 5 per document
            page_limit = 3
            doc_limit = 5
            
            page_count = page_counts.get(page_id, 0)
            doc_count = doc_counts.get(doc_id, 0)
            
            # Allow if under limits or if it's a very high scoring result
            very_high_score = candidate.get('adjusted_score', 0) > 1.5
            
            if (page_count < page_limit and doc_count < doc_limit) or very_high_score:
                diversified.append(candidate)
                page_counts[page_id] = page_count + 1
                doc_counts[doc_id] = doc_count + 1
                
                if len(diversified) >= max_results:
                    break
        
        return diversified
    
    def _calculate_conf_penalty(self, conf_mean: Optional[float]) -> float:
        """Calculate OCR confidence penalty."""
        if conf_mean is None or pd.isna(conf_mean):
            return 0.0
        
        # Bin-based penalty: higher penalty for lower confidence
        if conf_mean < 30:
            return 0.08  # High penalty
        elif conf_mean < 50:
            return 0.05  # Medium penalty
        elif conf_mean < 70:
            return 0.02  # Low penalty
        else:
            return 0.0   # No penalty
    
    def _calculate_priors(self, 
                         query_lang: str,
                         is_tabular: bool,
                         row: pd.Series) -> float:
        """Calculate enhanced quality and authority priors for a result."""
        prior = 0.0
        
        # Same-language boost with confidence weighting
        if query_lang != 'mixed' and row.get('language') == query_lang:
            lang_conf = row.get('language_confidence', 1.0)
            prior += self.same_lang_boost * lang_conf
        
        # Cross-language penalty for mismatched languages
        elif query_lang != 'mixed' and row.get('language') != 'mixed' and row.get('language') != query_lang:
            prior -= 0.03  # Small penalty for language mismatch
        
        # Authority boost
        if row.get('source_authority') == 'authoritative':
            prior += self.authority_boost
        
        # Quality tier boost/penalty
        quality_tier = row.get('quality_tier', 'medium')
        prior += self.quality_tier_boosts.get(quality_tier, 0.0)
        
        # Watermark penalty
        watermark_flags = row.get('watermark_flags')
        # Handle different types: numpy array, list, string, None
        has_watermark = False
        if watermark_flags is not None:
            try:
                if hasattr(watermark_flags, 'size'):  # numpy array
                    has_watermark = watermark_flags.size > 0
                elif hasattr(watermark_flags, '__len__'):  # list, tuple, string
                    has_watermark = len(watermark_flags) > 0 and str(watermark_flags) != 'null'
                else:
                    has_watermark = bool(watermark_flags) and str(watermark_flags) != 'null'
            except (ValueError, TypeError):
                # Fallback for problematic array types
                has_watermark = False
        
        if has_watermark:
            prior -= self.watermark_penalty
        
        # OCR confidence penalty
        conf_penalty = self._calculate_conf_penalty(row.get('conf_mean'))
        prior -= conf_penalty
        
        # Table boost for tabular queries
        if is_tabular and row.get('chunk_type') == 'table':
            prior += self.table_boost
        
        # Reconstruction penalty (content that needed heavy OCR fixing)
        if row.get('is_reconstructed', False):
            reconstruction_score = row.get('reconstruction_score', 0.0)
            if reconstruction_score < 0.5:  # Poor reconstruction
                prior -= 0.02
        
        # Length-based quality signal (very short or very long chunks)
        token_count = row.get('token_count', 0)
        if token_count < 10:  # Very short chunks often low quality
            prior -= 0.02
        elif token_count > 800:  # Very long chunks may be less focused
            prior -= 0.01
        
        return prior
    
    def _calculate_dynamic_fallback_threshold(self, 
                                            query: str, 
                                            query_lang: str,
                                            base_threshold: float = 0.5) -> float:
        """Calculate dynamic fallback threshold based on query complexity and domain."""
        try:
            # Analyze query complexity
            bilingual_analysis = self.bilingual_processor.analyze_bilingual_query(query)
            
            # Start with base threshold
            threshold = base_threshold
            
            # Adjust based on domain
            domain_adjustments = {
                'constitution': -0.1,  # More aggressive fallback for legal terms
                'health': -0.05,       # Slightly more aggressive for health
                'emergency': -0.15,    # Very aggressive for emergency terms
                'covid': -0.1,         # More aggressive for COVID terms
                'general': 0.0         # No adjustment
            }
            
            threshold += domain_adjustments.get(bilingual_analysis.domain_context, 0.0)
            
            # Adjust based on semantic mappings
            if bilingual_analysis.semantic_mappings:
                # If we have strong semantic mappings, be more aggressive with fallback
                avg_semantic_weight = sum(m['semantic_weight'] for m in bilingual_analysis.semantic_mappings) / len(bilingual_analysis.semantic_mappings)
                threshold -= (avg_semantic_weight - 0.5) * 0.2
            
            # Adjust based on processing strategy
            strategy_adjustments = {
                'bilingual_segmented': -0.1,  # More aggressive for bilingual queries
                'semantic_expansion': -0.05,  # Slightly more aggressive for semantic queries
                'mixed_language': -0.1,       # More aggressive for mixed language
                'monolingual': 0.0            # No adjustment
            }
            
            threshold += strategy_adjustments.get(bilingual_analysis.processing_strategy, 0.0)
            
            # Ensure threshold stays within reasonable bounds
            threshold = max(0.1, min(0.8, threshold))
            
            return threshold
            
        except Exception as e:
            # Fallback to base threshold on error
            print(f"Dynamic threshold calculation failed: {e}, using base threshold")
            return base_threshold
    
    def _search_single_language(self, 
                               query: str,
                               k: int,
                               query_lang: str,
                               target_lang: str,
                               is_tabular: bool,
                               allow_distractors: bool,
                               inject_hard_negatives: int,
                               nprobe: Optional[int],
                               efSearch: Optional[int],
                               cross_lang_fallback_threshold: float) -> Dict[str, Any]:
        """Search with single target language and potential cross-language fallback."""
        # Generate language-aware expansions
        expanded_by_lang = self._expand_query_terms(query, query_lang, [target_lang])
        expanded_queries = expanded_by_lang.get(target_lang, [query])
        
        if len(expanded_queries) > 1:
            print(f"Query expansion: {query} -> {expanded_queries[1:]}")
        
        # Generate embeddings
        query_embeddings = []
        for q in expanded_queries:
            emb = self.model.encode([q]).astype(np.float32)
            if FAISS_AVAILABLE:
                faiss.normalize_L2(emb)
            query_embeddings.append(emb)
        
        # Perform search
        results = self._execute_search(
            query_embeddings, k, query_lang, is_tabular, 
            allow_distractors, inject_hard_negatives, nprobe, efSearch
        )
        
        # Check if cross-language fallback is needed
        auth_candidates = results.get('authoritative_candidates', [])
        authority_recall = len(auth_candidates) / max(k, 1)
        
        # Calculate dynamic fallback threshold
        dynamic_threshold = self._calculate_dynamic_fallback_threshold(
            query, query_lang, cross_lang_fallback_threshold
        )
        
        if authority_recall < dynamic_threshold and target_lang != 'mixed':
            # Try cross-language fallback
            fallback_lang = 'en' if target_lang == 'ne' else 'ne'
            fallback_expanded = self._expand_query_terms(query, query_lang, [fallback_lang])
            fallback_queries = fallback_expanded.get(fallback_lang, [query])
            
            fallback_embeddings = []
            for q in fallback_queries:
                emb = self.model.encode([q]).astype(np.float32)
                if FAISS_AVAILABLE:
                    faiss.normalize_L2(emb)
                fallback_embeddings.append(emb)
            
            fallback_results = self._execute_search(
                fallback_embeddings, k, query_lang, is_tabular,
                allow_distractors, inject_hard_negatives, nprobe, efSearch,
                cross_lang_penalty=0.2  # Penalty for cross-language results
            )
            
            # Merge results
            results = self._merge_search_results(results, fallback_results, target_lang, fallback_lang)
            results['cross_lang_fallback_used'] = True
        else:
            results['cross_lang_fallback_used'] = False
        
        # Add language control metadata
        results['target_language'] = target_lang
        results['output_mode'] = 'single'
        results['expansion_terms'] = {target_lang: expanded_queries[1:] if len(expanded_queries) > 1 else []}
        
        # Calculate language metrics
        lang_metrics = self.calculate_language_metrics(query, query_lang, target_lang, results)
        results['language_metrics'] = lang_metrics
        
        return results
    
    def _search_bilingual(self,
                         query: str,
                         k: int,
                         query_lang: str,
                         target_langs: Tuple[str, str],
                         is_tabular: bool,
                         allow_distractors: bool,
                         inject_hard_negatives: int,
                         nprobe: Optional[int],
                         efSearch: Optional[int]) -> Dict[str, Any]:
        """Search with bilingual mode - independent EN/NE expansions with stable dedup."""
        en_lang, ne_lang = target_langs
        
        # Independent expansions for each language
        expanded_by_lang = self._expand_query_terms(query, query_lang, [en_lang, ne_lang])
        
        # Search each language independently
        results_by_lang = {}
        
        for lang in [en_lang, ne_lang]:
            expanded_queries = expanded_by_lang.get(lang, [query])
            
            # Generate embeddings
            query_embeddings = []
            for q in expanded_queries:
                emb = self.model.encode([q]).astype(np.float32)
                if FAISS_AVAILABLE:
                    faiss.normalize_L2(emb)
                query_embeddings.append(emb)
            
            # Execute search for this language
            lang_results = self._execute_search(
                query_embeddings, k // 2, query_lang, is_tabular,
                allow_distractors, inject_hard_negatives, nprobe, efSearch
            )
            
            results_by_lang[lang] = lang_results
        
        # Merge with stable deduplication
        merged_results = self._merge_bilingual_results(results_by_lang, k)
        
        # Add bilingual metadata
        merged_results['target_languages'] = target_langs
        merged_results['output_mode'] = 'bilingual'
        merged_results['expansion_terms'] = {
            lang: expanded_by_lang.get(lang, [])[1:] if len(expanded_by_lang.get(lang, [])) > 1 else []
            for lang in [en_lang, ne_lang]
        }
        merged_results['per_language_evidence'] = results_by_lang
        
        # Calculate language metrics for bilingual mode
        lang_metrics = self.calculate_language_metrics(query, query_lang, target_langs, merged_results)
        merged_results['language_metrics'] = lang_metrics
        
        return merged_results
    
    def _execute_search(self,
                       query_embeddings: List[np.ndarray],
                       k: int,
                       query_lang: str,
                       is_tabular: bool,
                       allow_distractors: bool,
                       inject_hard_negatives: int,
                       nprobe: Optional[int],
                       efSearch: Optional[int],
                       cross_lang_penalty: float = 0.0) -> Dict[str, Any]:
        """Execute the actual FAISS search with given embeddings."""
        # Use original query as primary, expanded as secondary
        primary_embedding = query_embeddings[0]
        
        # Set search parameters
        if nprobe and hasattr(self.indices[0], 'nprobe'):
            self.indices[0].nprobe = nprobe
        if efSearch and hasattr(self.indices[0], 'hnsw'):
            self.indices[0].hnsw.efSearch = efSearch
        
        # Search FAISS index
        search_k = min(k * 3, len(self.meta_df))  # Over-retrieve for filtering
        
        if FAISS_AVAILABLE:
            distances, indices = self.indices[0].search(primary_embedding, search_k)
            distances = distances[0]
            indices = indices[0]
        else:
            # NumPy fallback
            all_embeddings = np.vstack([self.embeddings])  # Assume embeddings stored
            similarities = np.dot(all_embeddings, primary_embedding.T).flatten()
            top_indices = np.argsort(similarities)[-search_k:][::-1]
            indices = top_indices
            distances = 1.0 - similarities[indices]  # Convert to distances
        
        # Filter out invalid indices
        valid_mask = (indices >= 0) & (indices < len(self.meta_df))
        indices = indices[valid_mask]
        distances = distances[valid_mask]
        
        if len(indices) == 0:
            return self._empty_search_results()
        
        # Get metadata for results
        candidates = []
        for idx, distance in zip(indices, distances):
            row = self.meta_df.iloc[idx]
            
            # Calculate priors
            prior = self._calculate_priors(query_lang, is_tabular, row)
            
            # Apply cross-language penalty if specified
            candidate_lang = row.get('language', 'unknown')
            if cross_lang_penalty > 0.0 and candidate_lang != query_lang:
                prior -= cross_lang_penalty
            
            # Final score
            base_score = 1.0 - distance
            adjusted_score = base_score + prior
            
            candidate = {
                'chunk_id': row.get('chunk_id'),
                'text': row.get('text', ''),
                'doc_id': row.get('doc_id'),
                'page_id': row.get('page_id'),
                'language': candidate_lang,
                'source_authority': row.get('source_authority'),
                'chunk_type': row.get('chunk_type', 'text'),
                'base_score': base_score,
                'prior_adjustment': prior,
                'adjusted_score': adjusted_score,
                'bbox': row.get('bbox'),
                'source_page_is_ocr': row.get('source_page_is_ocr', False),
                'conf_mean': row.get('conf_mean', 0.0),
                'watermark_flags': row.get('watermark_flags'),
                'quality_tier': row.get('quality_tier', 'medium')
            }
            candidates.append(candidate)
        
        # Sort by adjusted score
        candidates.sort(key=lambda x: x['adjusted_score'], reverse=True)
        
        # Apply diversification
        candidates = self._diversify_results(candidates, k * 2)
        
        # Split into authoritative and distractors
        authoritative_candidates = [c for c in candidates if c['source_authority'] == 'authoritative'][:k]
        distractor_candidates = [c for c in candidates if c['source_authority'] != 'authoritative'][:inject_hard_negatives]
        
        # Calculate metrics
        total_candidates = len(authoritative_candidates) + len(distractor_candidates)
        authority_purity = len(authoritative_candidates) / total_candidates if total_candidates > 0 else 0.0
        
        return {
            'authoritative_candidates': authoritative_candidates,
            'distractor_candidates': distractor_candidates if allow_distractors else [],
            'authority_purity': authority_purity,
            'total_searched': len(self.meta_df),
            'candidates_considered': len(candidates)
        }
    
    def _merge_search_results(self, 
                             primary_results: Dict[str, Any],
                             fallback_results: Dict[str, Any],
                             target_lang: str,
                             fallback_lang: str) -> Dict[str, Any]:
        """Merge primary and fallback search results."""
        # Combine candidates
        primary_auth = primary_results.get('authoritative_candidates', [])
        fallback_auth = fallback_results.get('authoritative_candidates', [])
        
        # Dedup by chunk_id, preferring primary results
        seen_chunks = set()
        merged_auth = []
        
        # Add primary results first
        for candidate in primary_auth:
            chunk_id = candidate.get('chunk_id')
            if chunk_id not in seen_chunks:
                candidate['result_source'] = f'primary_{target_lang}'
                merged_auth.append(candidate)
                seen_chunks.add(chunk_id)
        
        # Add fallback results
        for candidate in fallback_auth:
            chunk_id = candidate.get('chunk_id')
            if chunk_id not in seen_chunks:
                candidate['result_source'] = f'fallback_{fallback_lang}'
                # Apply additional penalty for fallback
                candidate['adjusted_score'] -= 0.1
                merged_auth.append(candidate)
                seen_chunks.add(chunk_id)
        
        # Re-sort by adjusted score
        merged_auth.sort(key=lambda x: x['adjusted_score'], reverse=True)
        
        # Merge distractors similarly
        primary_dist = primary_results.get('distractor_candidates', [])
        fallback_dist = fallback_results.get('distractor_candidates', [])
        merged_dist = primary_dist + fallback_dist  # Simple concatenation for distractors
        
        # Calculate new authority purity
        total_candidates = len(merged_auth) + len(merged_dist)
        authority_purity = len(merged_auth) / total_candidates if total_candidates > 0 else 0.0
        
        return {
            'authoritative_candidates': merged_auth,
            'distractor_candidates': merged_dist,
            'authority_purity': authority_purity,
            'total_searched': primary_results.get('total_searched', 0),
            'candidates_considered': primary_results.get('candidates_considered', 0) + fallback_results.get('candidates_considered', 0)
        }
    
    def _merge_bilingual_results(self, 
                                results_by_lang: Dict[str, Dict[str, Any]], 
                                k: int) -> Dict[str, Any]:
        """Merge bilingual search results with stable deduplication."""
        all_auth_candidates = []
        all_dist_candidates = []
        
        # Collect all candidates
        for lang, results in results_by_lang.items():
            auth_candidates = results.get('authoritative_candidates', [])
            dist_candidates = results.get('distractor_candidates', [])
            
            # Tag with source language
            for candidate in auth_candidates:
                candidate['result_source_lang'] = lang
                all_auth_candidates.append(candidate)
            
            for candidate in dist_candidates:
                candidate['result_source_lang'] = lang
                all_dist_candidates.append(candidate)
        
        # Stable deduplication by (doc_id, page_id, span) with score-based preference
        def get_dedup_key(candidate):
            return (
                candidate.get('doc_id', ''),
                candidate.get('page_id', ''),
                str(candidate.get('bbox', ''))[:50]  # Rough span approximation
            )
        
        # Dedup authoritative candidates
        seen_keys = {}
        deduped_auth = []
        
        # Sort by adjusted score first for stable dedup
        all_auth_candidates.sort(key=lambda x: x['adjusted_score'], reverse=True)
        
        for candidate in all_auth_candidates:
            key = get_dedup_key(candidate)
            if key not in seen_keys:
                deduped_auth.append(candidate)
                seen_keys[key] = candidate['adjusted_score']
            elif candidate['adjusted_score'] > seen_keys[key]:
                # Replace with higher-scoring duplicate
                for i, existing in enumerate(deduped_auth):
                    if get_dedup_key(existing) == key:
                        deduped_auth[i] = candidate
                        seen_keys[key] = candidate['adjusted_score']
                        break
        
        # Limit to k results
        deduped_auth = deduped_auth[:k]
        
        # Simple dedup for distractors
        dist_seen = set()
        deduped_dist = []
        for candidate in all_dist_candidates:
            key = get_dedup_key(candidate)
            if key not in dist_seen:
                deduped_dist.append(candidate)
                dist_seen.add(key)
        
        # Calculate metrics
        total_candidates = len(deduped_auth) + len(deduped_dist)
        authority_purity = len(deduped_auth) / total_candidates if total_candidates > 0 else 0.0
        
        return {
            'authoritative_candidates': deduped_auth,
            'distractor_candidates': deduped_dist,
            'authority_purity': authority_purity,
            'total_searched': sum(r.get('total_searched', 0) for r in results_by_lang.values()),
            'candidates_considered': sum(r.get('candidates_considered', 0) for r in results_by_lang.values())
        }
    
    def _empty_search_results(self) -> Dict[str, Any]:
        """Return empty search results."""
        return {
            'authoritative_candidates': [],
            'distractor_candidates': [],
            'authority_purity': 0.0,
            'total_searched': len(self.meta_df) if hasattr(self, 'meta_df') else 0,
            'candidates_considered': 0
        }
    
    def search(self,
               query: str,
               k: int = 10,
               query_lang: str = 'auto',
               output_mode: str = 'auto',
               allow_distractors: bool = False,
               inject_hard_negatives: int = 0,
               nprobe: Optional[int] = None,
               efSearch: Optional[int] = None,
               cross_lang_fallback_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Perform multilingual search with language control.
        
        Args:
            query: Search query
            k: Number of results to return
            query_lang: Query language ('auto', 'ne', 'en', 'mixed')
            output_mode: Target language mode ('auto', 'en', 'ne', 'bilingual')
            allow_distractors: Whether to include distractor results
            inject_hard_negatives: Number of hard negatives to inject
            nprobe: IVF nprobe parameter (if applicable)
            efSearch: HNSW efSearch parameter (if applicable)
            cross_lang_fallback_threshold: Threshold for cross-language fallback
            
        Returns:
            Search results with authority purity, language metrics, and diagnostics
        """
        if not self.indices:
            raise RuntimeError("No FAISS indices loaded")
        
        if not self.model:
            raise RuntimeError("No embedding model loaded")
        
        # Detect query language
        if query_lang == 'auto':
            detected_lang, lang_conf = self.query_lang_detector.detect(query)
            query_lang = detected_lang
        else:
            lang_conf = 1.0
        
        # Determine target language(s)
        target_langs = self.target_lang_selector.choose_target_lang(query, output_mode)
        is_bilingual = isinstance(target_langs, tuple)
        
        # Check if query looks tabular
        is_tabular = self._is_tabular_query(query)
        
        if is_bilingual:
            # Bilingual mode: independent expansions for EN & NE
            return self._search_bilingual(
                query, k, query_lang, target_langs, is_tabular,
                allow_distractors, inject_hard_negatives, nprobe, efSearch
            )
        else:
            # Single language mode with potential cross-language fallback
            return self._search_single_language(
                query, k, query_lang, target_langs, is_tabular,
                allow_distractors, inject_hard_negatives, nprobe, efSearch,
                cross_lang_fallback_threshold
            )
    
    def calculate_language_metrics(self, 
                                  query: str,
                                  query_lang: str, 
                                  target_lang: Union[str, Tuple[str, str]],
                                  results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate language-specific metrics for retrieval results."""
        auth_candidates = results.get('authoritative_candidates', [])
        
        if not auth_candidates:
            return {
                'language_consistency_rate': 0.0,
                'same_lang_citation_rate': 0.0,
                'cross_lang_success@5': 0.0,
                'cross_lang_success@10': 0.0
            }
        
        # Language consistency: fraction of results matching target language(s)
        if isinstance(target_lang, tuple):
            # Bilingual mode
            target_langs = set(target_lang)
            consistent_count = sum(1 for c in auth_candidates 
                                 if c.get('language', '') in target_langs)
        else:
            # Single language mode
            consistent_count = sum(1 for c in auth_candidates 
                                 if c.get('language', '') == target_lang)
        
        language_consistency_rate = consistent_count / len(auth_candidates)
        
        # Same language citation: fraction matching query language
        same_lang_count = sum(1 for c in auth_candidates 
                            if c.get('language', '') == query_lang)
        same_lang_citation_rate = same_lang_count / len(auth_candidates)
        
        # Cross-language success: ability to find relevant results in different languages
        cross_lang_count_5 = sum(1 for c in auth_candidates[:5] 
                               if c.get('language', '') != query_lang)
        cross_lang_count_10 = sum(1 for c in auth_candidates[:10] 
                                if c.get('language', '') != query_lang)
        
        cross_lang_success_5 = cross_lang_count_5 / min(5, len(auth_candidates))
        cross_lang_success_10 = cross_lang_count_10 / min(10, len(auth_candidates))
        
        # Enhanced metrics
        enhanced_metrics = {
            'language_consistency_rate': language_consistency_rate,
            'same_lang_citation_rate': same_lang_citation_rate,
            'cross_lang_success@5': cross_lang_success_5,
            'cross_lang_success@10': cross_lang_success_10
        }
        
        # Add bilingual processing metrics if available
        try:
            bilingual_analysis = self.bilingual_processor.analyze_bilingual_query(query)
            enhanced_metrics.update({
                'query_domain': bilingual_analysis.domain_context,
                'processing_strategy': bilingual_analysis.processing_strategy,
                'bilingual_confidence': bilingual_analysis.confidence,
                'semantic_mappings_count': len(bilingual_analysis.semantic_mappings),
                'language_segments_count': len(bilingual_analysis.language_segments)
            })
            
            # Domain-specific success rates
            if bilingual_analysis.domain_context != 'general':
                domain_results = [c for c in auth_candidates if bilingual_analysis.domain_context.lower() in c.get('text', '').lower()]
                enhanced_metrics[f'{bilingual_analysis.domain_context}_success_rate'] = len(domain_results) / len(auth_candidates) if auth_candidates else 0.0
                
        except Exception as e:
            print(f"Enhanced metrics calculation failed: {e}")
        
        return enhanced_metrics


def main():
    """CLI for multilingual search."""
    # Add CLI code here if needed
    pass


if __name__ == "__main__":
    main()
