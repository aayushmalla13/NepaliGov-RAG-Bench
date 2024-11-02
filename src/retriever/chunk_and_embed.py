#!/usr/bin/env python3
"""
Chunking and Embedding for NepaliGov-RAG-Bench

Page-anchored chunking with table-aware segmentation and quality-aware scoring.
Builds FAISS index with comprehensive metadata preservation.
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
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None  # type: ignore
    print("Warning: faiss not available. Will fall back to NumPy index.")
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

# Optional language identification
try:
    import langid  # Lightweight language ID
    LANGID_AVAILABLE = True
except ImportError:
    LANGID_AVAILABLE = False
    print("Warning: langid not available. Install with: pip install langid")


class Chunk:
    """Container for text chunks with metadata."""
    
    def __init__(self,
                 chunk_id: str,
                 text: str,
                 chunk_type: str,  # 'text' or 'table'
                 doc_id: str,
                 page_id: str,
                 block_id: str,
                 language: str,
                 source_authority: str,
                 is_distractor: bool,
                 bbox: Optional[List[float]] = None,
                 source_page_is_ocr: bool = False,
                 conf_mean: Optional[float] = None,
                 watermark_flags: Optional[str] = None,
                 token_count: int = 0,
                 quality_score: float = 1.0,
                 language_confidence: Optional[float] = None,
                 quality_tier: str = "high",
                 is_reconstructed: bool = False,
                 reconstruction_score: float = 0.0):
        self.chunk_id = chunk_id
        self.text = text
        self.chunk_type = chunk_type
        self.doc_id = doc_id
        self.page_id = page_id
        self.block_id = block_id
        self.language = language
        self.source_authority = source_authority
        self.is_distractor = is_distractor
        self.bbox = bbox
        self.source_page_is_ocr = source_page_is_ocr
        self.conf_mean = conf_mean
        self.watermark_flags = watermark_flags
        self.token_count = token_count
        self.quality_score = quality_score
        self.language_confidence = language_confidence
        self.quality_tier = quality_tier
        self.is_reconstructed = is_reconstructed
        self.reconstruction_score = reconstruction_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for metadata storage."""
        return {
            'chunk_id': self.chunk_id,
            'text': self.text,
            'chunk_type': self.chunk_type,
            'doc_id': self.doc_id,
            'page_id': self.page_id,
            'block_id': self.block_id,
            'language': self.language,
            'source_authority': self.source_authority,
            'is_distractor': self.is_distractor,
            'bbox': json.dumps(self.bbox) if self.bbox else None,
            'source_page_is_ocr': self.source_page_is_ocr,
            'conf_mean': self.conf_mean,
            'watermark_flags': self.watermark_flags,
            'token_count': self.token_count,
            'quality_score': self.quality_score,
            'language_confidence': self.language_confidence,
            'quality_tier': self.quality_tier,
            'is_reconstructed': self.is_reconstructed,
            'reconstruction_score': self.reconstruction_score
        }


class ChunkingEngine:
    """Enhanced chunking engine with table-aware segmentation."""
    
    def __init__(self,
                 text_chunk_size: int = 1000,
                 text_overlap: int = 200,
                 table_max_length: int = 2000):
        """
        Initialize chunking engine.
        
        Args:
            text_chunk_size: Target size for text chunks (tokens)
            text_overlap: Overlap between adjacent text chunks (tokens)
            table_max_length: Maximum length for table chunks (characters)
        """
        self.text_chunk_size = text_chunk_size
        self.text_overlap = text_overlap
        self.table_max_length = table_max_length
    
    def _devanagari_ratio(self, text: str) -> float:
        total_alpha = sum(1 for c in text if c.isalpha())
        if total_alpha == 0:
            return 0.0
        dev = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
        return dev / total_alpha
    
    def _is_ocr_garbage(self, text: str) -> bool:
        """Detect OCR garbage/artifacts that should be filtered."""
        if not text or len(text) < 10:
            return True
        
        # Character composition analysis
        total_chars = len(text)
        alpha_chars = sum(1 for c in text if c.isalpha())
        digit_chars = sum(1 for c in text if c.isdigit())
        space_chars = sum(1 for c in text if c.isspace())
        symbol_chars = total_chars - alpha_chars - digit_chars - space_chars
        
        # OCR garbage indicators
        alpha_ratio = alpha_chars / total_chars
        symbol_ratio = symbol_chars / total_chars
        
        # Too few alphabetic characters (likely OCR noise)
        if alpha_ratio < 0.3:
            return True
        
        # Too many symbols/punctuation (OCR artifacts)
        if symbol_ratio > 0.4:
            return True
        
        # Check for excessive fragmentation (single chars with spaces)
        words = text.split()
        if len(words) > 5:
            single_char_words = sum(1 for w in words if len(w) == 1)
            if single_char_words / len(words) > 0.6:  # >60% single-char words
                return True
        
        # Check for excessive mixed scripts in short segments
        if len(text) < 100:  # Short text
            devanagari = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
            latin = sum(1 for c in text if c.isalpha() and ord(c) < 0x0900)
            if devanagari > 0 and latin > 0 and min(devanagari, latin) < 5:
                # Very few characters of minority script - likely OCR error
                return True
        
        return False
    
    def _calculate_text_coherence(self, text: str) -> float:
        """Calculate text coherence score (0-1, higher = more coherent)."""
        if not text:
            return 0.0
        
        words = text.split()
        if len(words) < 3:
            return 0.3  # Short text gets low coherence
        
        # Word length distribution (coherent text has reasonable word lengths)
        word_lengths = [len(w) for w in words]
        avg_word_len = sum(word_lengths) / len(word_lengths)
        
        coherence = 0.5  # Base score
        
        # Reasonable average word length (3-12 chars)
        if 3 <= avg_word_len <= 12:
            coherence += 0.2
        
        # Not too many very short words
        short_words = sum(1 for w in words if len(w) <= 2)
        if short_words / len(words) < 0.5:
            coherence += 0.2
        
        # Contains some longer words (indicates real language)
        long_words = sum(1 for w in words if len(w) >= 4)
        if long_words / len(words) > 0.3:
            coherence += 0.1
        
        return min(coherence, 1.0)
    
    def _attempt_text_reconstruction(self, text: str) -> Tuple[str, float]:
        """
        Attempt to reconstruct/improve OCR text quality.
        Returns (improved_text, improvement_score).
        """
        if not text:
            return text, 0.0
        
        original_text = text
        improved_text = text
        improvements = 0
        
        # 1. Fix common OCR character substitutions
        ocr_fixes = {
            # Common OCR errors
            r'\b0\b': 'O',  # Zero to O
            r'\b1\b': 'I',  # One to I  
            r'rn': 'm',     # rn to m
            r'vv': 'w',     # vv to w
            r'cl': 'd',     # cl to d
            # Devanagari OCR fixes
            r'०': '0',      # Devanagari zero to ASCII
            r'१': '1',      # Devanagari one to ASCII
        }
        
        for pattern, replacement in ocr_fixes.items():
            import re
            new_text = re.sub(pattern, replacement, improved_text)
            if new_text != improved_text:
                improvements += 1
                improved_text = new_text
        
        # 2. Remove excessive whitespace and normalize
        import re
        # Collapse multiple spaces
        new_text = re.sub(r'\s+', ' ', improved_text)
        if new_text != improved_text:
            improvements += 1
            improved_text = new_text
        
        # 3. Remove isolated single characters (likely OCR noise)
        words = improved_text.split()
        filtered_words = []
        for word in words:
            # Keep single chars if they're meaningful (numbers, common letters)
            if len(word) == 1:
                if word.isdigit() or word.lower() in 'aio':  # Keep meaningful single chars
                    filtered_words.append(word)
                # else: skip isolated noise characters
            else:
                filtered_words.append(word)
        
        new_text = ' '.join(filtered_words)
        if new_text != improved_text:
            improvements += 1
            improved_text = new_text
        
        # Calculate improvement score
        if not improved_text or improved_text == original_text:
            return original_text, 0.0
        
        # Score based on text length preservation and improvements made
        length_ratio = len(improved_text) / max(len(original_text), 1)
        improvement_score = min(1.0, (improvements * 0.3) + (length_ratio * 0.7))
        
        return improved_text, improvement_score
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Enhanced language detection with OCR recovery and quality improvement.
        Returns (lang, confidence) where lang in {'ne','en','mixed'}.
        """
        original_text = (text or "").strip()
        if not original_text:
            return ('mixed', 0.0)
        
        # Step 1: Check if it's OCR garbage that might be recoverable
        is_garbage = self._is_ocr_garbage(original_text)
        
        # Step 2: Attempt text reconstruction if it looks like OCR garbage
        if is_garbage:
            improved_text, improvement_score = self._attempt_text_reconstruction(original_text)
            
            # If reconstruction helped significantly, use improved text
            if improvement_score > 0.3:
                text = improved_text
                # Recheck if it's still garbage after improvement
                is_garbage = self._is_ocr_garbage(text)
            else:
                text = original_text
        else:
            text = original_text
            improvement_score = 0.0
        
        # Step 3: Calculate text coherence
        coherence = self._calculate_text_coherence(text)
        
        # Step 4: Language detection with recovery awareness
        dev_ratio = self._devanagari_ratio(text)
        
        # Enhanced script-based decision with coherence and improvement factors
        if dev_ratio > 0.7:
            confidence = min(1.0, 0.6 + 0.4 * dev_ratio + 0.2 * coherence + 0.1 * improvement_score)
            return ('ne', confidence)
        
        if dev_ratio < 0.05:
            confidence = min(1.0, 0.6 + 0.4 * (1.0 - dev_ratio) + 0.2 * coherence + 0.1 * improvement_score)
            return ('en', confidence)
        
        # Ambiguous range - use enhanced heuristics
        if LANGID_AVAILABLE:
            try:
                pred, raw_conf = langid.classify(text)
                # Convert langid confidence (can be negative log prob) to 0-1
                conf = max(0.0, min(1.0, abs(float(raw_conf)) / 100.0))
                
                # Enhanced mapping with coherence and improvement
                if pred in ('ne', 'hi') and dev_ratio >= 0.3:
                    final_conf = min(1.0, conf + 0.3 * coherence + 0.1 * improvement_score)
                    return ('ne', final_conf)
                
                if pred == 'en' and dev_ratio <= 0.3:
                    final_conf = min(1.0, conf + 0.3 * coherence + 0.1 * improvement_score)
                    return ('en', final_conf)
                
                # Mixed language - give recovery a chance
                if is_garbage and improvement_score < 0.2:
                    # Still garbage after attempted recovery - low confidence mixed
                    return ('mixed', 0.1)
                
                return ('mixed', conf * coherence + 0.1 * improvement_score)
            except Exception:
                pass
        
        # Fallback - don't completely discard, but mark as low confidence
        if is_garbage and improvement_score < 0.2:
            return ('mixed', 0.1)  # Low confidence but don't filter completely
        
        return ('mixed', 0.5 * coherence + 0.1 * improvement_score)
    
    def _assign_quality_tier(self, language_confidence: float, quality_score: float, 
                           is_reconstructed: bool, reconstruction_score: float) -> str:
        """
        Assign quality tier based on multiple factors.
        Returns 'high', 'medium', 'low', or 'recovery' for different tiers.
        """
        # High quality: good confidence, good quality score, not reconstructed
        if language_confidence >= 0.8 and quality_score >= 0.9 and not is_reconstructed:
            return 'high'
        
        # Medium quality: decent confidence and quality
        elif language_confidence >= 0.5 and quality_score >= 0.6:
            return 'medium'
        
        # Recovery tier: was garbage but successfully reconstructed
        elif is_reconstructed and reconstruction_score >= 0.3:
            return 'recovery'
        
        # Low quality: poor confidence or quality but still potentially useful
        else:
            return 'low'
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Simple tokenization for chunking.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Enhanced tokenization for multilingual text
        tokens = re.findall(r'\S+', text)
        return [token for token in tokens if len(token) > 0]
    
    def calculate_quality_score(self, row: pd.Series) -> float:
        """
        Calculate quality score for a corpus row.
        
        Args:
            row: Corpus row from DataFrame
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        score = 1.0
        
        # OCR confidence penalty
        if row.get('source_page_is_ocr') and row.get('conf_mean') is not None:
            conf_score = row['conf_mean'] / 100.0
            score *= (0.5 + 0.5 * conf_score)  # Scale from 0.5-1.0 based on confidence
        
        # Authority boost
        if row.get('source_authority') == 'authoritative':
            score *= 1.2
        
        # Distractor penalty
        if row.get('is_distractor'):
            score *= 0.7
        
        # Language preference (slight boost for clear language detection)
        if row.get('language') in ['ne', 'en']:
            score *= 1.1
        
        # Watermark penalty (if present)
        watermark_flags = row.get('watermark_flags')
        has_watermarks = False
        if watermark_flags is not None:
            try:
                if isinstance(watermark_flags, str) and watermark_flags != 'null':
                    has_watermarks = True
                elif isinstance(watermark_flags, (list, tuple)) and len(watermark_flags) > 0:
                    has_watermarks = True
                elif hasattr(watermark_flags, 'size') and watermark_flags.size > 0:
                    has_watermarks = True
            except:
                has_watermarks = False
        
        if has_watermarks:
            score *= 0.9
        
        # Ensure score is in valid range
        return min(max(score, 0.0), 1.0)
    
    def serialize_table(self, table_data: str) -> str:
        """
        Serialize table data to text format.
        
        Args:
            table_data: Raw table data (JSON string or text)
            
        Returns:
            Serialized table text
        """
        if not table_data or table_data == 'null':
            return "[TABLE]"
        
        try:
            # Try to parse as JSON (from table structure)
            table_info = json.loads(table_data)
            if isinstance(table_info, dict) and 'data' in table_info:
                # Convert table data to TSV format
                rows = table_info['data']
                if rows:
                    # Get headers from first row keys
                    headers = list(rows[0].keys()) if rows else []
                    lines = ['\t'.join(headers)]
                    
                    # Add data rows
                    for row in rows:
                        values = [str(row.get(col, '')) for col in headers]
                        lines.append('\t'.join(values))
                    
                    table_text = '\n'.join(lines)
                    
                    # Add table metadata
                    meta_info = f"[TABLE: {table_info.get('rows', 0)} rows × {table_info.get('cols', 0)} cols]"
                    return f"{meta_info}\n{table_text}"
            
        except (json.JSONDecodeError, KeyError):
            pass
        
        # Fallback: return as-is with table marker
        return f"[TABLE]\n{str(table_data)[:self.table_max_length]}"
    
    def chunk_text_block(self, text: str, base_chunk_id: str, metadata: Dict) -> List[Chunk]:
        """
        Chunk a text block with overlap.
        
        Args:
            text: Text to chunk
            base_chunk_id: Base ID for chunk naming
            metadata: Common metadata for all chunks
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        tokens = self.tokenize_text(text)
        if len(tokens) <= self.text_chunk_size:
            # Single chunk with recovery processing
            original_text = text
            
            # Check if this looks like OCR garbage that needs reconstruction
            is_garbage = self._is_ocr_garbage(text)
            is_reconstructed = False
            reconstruction_score = 0.0
            
            if is_garbage:
                improved_text, reconstruction_score = self._attempt_text_reconstruction(text)
                if reconstruction_score > 0.3:
                    text = improved_text
                    is_reconstructed = True
            
            lang, lconf = self.detect_language(text)
            quality = metadata.get('quality_score', 1.0)
            # Boost quality by language confidence and reconstruction
            quality = float(min(max(quality * (0.9 + 0.2 * (lconf or 0.0)), 0.0), 1.2))
            
            # Assign quality tier
            quality_tier = self._assign_quality_tier(lconf or 0.0, quality, is_reconstructed, reconstruction_score)
            
            chunk = Chunk(
                chunk_id=f"{base_chunk_id}_000",
                text=text,
                chunk_type='text',
                token_count=len(tokens),
                language=lang,
                quality_score=quality,
                language_confidence=lconf,
                quality_tier=quality_tier,
                is_reconstructed=is_reconstructed,
                reconstruction_score=reconstruction_score,
                # pass-through metadata
                doc_id=metadata['doc_id'],
                page_id=metadata['page_id'],
                block_id=metadata['block_id'],
                source_authority=metadata['source_authority'],
                is_distractor=metadata['is_distractor'],
                bbox=metadata['bbox'],
                source_page_is_ocr=metadata['source_page_is_ocr'],
                conf_mean=metadata['conf_mean'],
                watermark_flags=metadata['watermark_flags']
            )
            return [chunk]
        
        # Multiple chunks with overlap
        chunks = []
        chunk_counter = 0
        start_idx = 0
        
        while start_idx < len(tokens):
            end_idx = min(start_idx + self.text_chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = ' '.join(chunk_tokens)
            
            # Apply reconstruction to each chunk
            original_chunk_text = chunk_text
            is_garbage = self._is_ocr_garbage(chunk_text)
            is_reconstructed = False
            reconstruction_score = 0.0
            
            if is_garbage:
                improved_text, reconstruction_score = self._attempt_text_reconstruction(chunk_text)
                if reconstruction_score > 0.3:
                    chunk_text = improved_text
                    is_reconstructed = True
            
            lang, lconf = self.detect_language(chunk_text)
            quality = metadata.get('quality_score', 1.0)
            quality = float(min(max(quality * (0.9 + 0.2 * (lconf or 0.0)), 0.0), 1.2))
            
            # Penalize long mixed-language segments slightly
            if lang == 'mixed' and len(chunk_tokens) > 200:
                quality *= 0.97
            
            # Assign quality tier
            quality_tier = self._assign_quality_tier(lconf or 0.0, quality, is_reconstructed, reconstruction_score)
            
            chunk = Chunk(
                chunk_id=f"{base_chunk_id}_{chunk_counter:03d}",
                text=chunk_text,
                chunk_type='text',
                token_count=len(chunk_tokens),
                language=lang,
                quality_score=quality,
                language_confidence=lconf,
                quality_tier=quality_tier,
                is_reconstructed=is_reconstructed,
                reconstruction_score=reconstruction_score,
                doc_id=metadata['doc_id'],
                page_id=metadata['page_id'],
                block_id=metadata['block_id'],
                source_authority=metadata['source_authority'],
                is_distractor=metadata['is_distractor'],
                bbox=metadata['bbox'],
                source_page_is_ocr=metadata['source_page_is_ocr'],
                conf_mean=metadata['conf_mean'],
                watermark_flags=metadata['watermark_flags']
            )
            chunks.append(chunk)
            
            # Move start position with overlap
            if end_idx >= len(tokens):
                break
            start_idx = max(start_idx + self.text_chunk_size - self.text_overlap, start_idx + 1)
            chunk_counter += 1
        
        return chunks
    
    def chunk_table_block(self, table_data: str, base_chunk_id: str, metadata: Dict) -> List[Chunk]:
        """
        Process a table block into chunks.
        
        Args:
            table_data: Table data to process
            base_chunk_id: Base ID for chunk naming
            metadata: Common metadata
            
        Returns:
            List of table chunks
        """
        serialized_table = self.serialize_table(table_data)
        
        # Truncate if too long
        if len(serialized_table) > self.table_max_length:
            serialized_table = serialized_table[:self.table_max_length] + "...[TRUNCATED]"
        
        # Apply reconstruction to table data if needed
        is_garbage = self._is_ocr_garbage(serialized_table)
        is_reconstructed = False
        reconstruction_score = 0.0
        
        if is_garbage:
            improved_text, reconstruction_score = self._attempt_text_reconstruction(serialized_table)
            if reconstruction_score > 0.3:
                serialized_table = improved_text
                is_reconstructed = True
        
        lang, lconf = self.detect_language(serialized_table)
        quality = metadata.get('quality_score', 1.0)
        quality = float(min(max(quality * (0.9 + 0.2 * (lconf or 0.0)), 0.0), 1.2))
        
        # Assign quality tier
        quality_tier = self._assign_quality_tier(lconf or 0.0, quality, is_reconstructed, reconstruction_score)
        
        chunk = Chunk(
            chunk_id=f"{base_chunk_id}_table_000",
            text=serialized_table,
            chunk_type='table',
            token_count=len(self.tokenize_text(serialized_table)),
            language=lang,
            quality_score=quality,
            language_confidence=lconf,
            quality_tier=quality_tier,
            is_reconstructed=is_reconstructed,
            reconstruction_score=reconstruction_score,
            doc_id=metadata['doc_id'],
            page_id=metadata['page_id'],
            block_id=metadata['block_id'],
            source_authority=metadata['source_authority'],
            is_distractor=metadata['is_distractor'],
            bbox=metadata['bbox'],
            source_page_is_ocr=metadata['source_page_is_ocr'],
            conf_mean=metadata['conf_mean'],
            watermark_flags=metadata['watermark_flags']
        )
        
        return [chunk]
    
    def chunk_corpus(self, corpus_df: pd.DataFrame) -> List[Chunk]:
        """
        Chunk entire corpus with page-anchored constraints and quality filtering.
        
        Args:
            corpus_df: Corpus DataFrame from CP4
            
        Returns:
            List of all chunks
        """
        print("Chunking corpus with enhanced quality filtering...")
        
        all_chunks = []
        chunk_counter = 0
        filtered_count = 0
        
        # Group by page to ensure no cross-page chunking
        page_groups = corpus_df.groupby('page_id')
        
        for page_id, page_df in page_groups:
            print(f"  Processing page: {page_id} ({len(page_df)} blocks)")
            
            # Process each block in the page
            for _, row in page_df.iterrows():
                if row['block_type'] == 'empty' or not row['text'].strip():
                    continue
                
                # Common metadata for all chunks from this block
                metadata = {
                    'doc_id': row['doc_id'],
                    'page_id': row['page_id'],
                    'block_id': row['block_id'],
                    'language': row['language'],
                    'source_authority': row['source_authority'],
                    'is_distractor': row['is_distractor'],
                    'bbox': json.loads(row['bbox']) if isinstance(row['bbox'], str) and row['bbox'] != 'null' else (row['bbox'].tolist() if hasattr(row['bbox'], 'tolist') else row['bbox']) if row['bbox'] is not None else None,
                    'source_page_is_ocr': row['source_page_is_ocr'],
                    'conf_mean': row['conf_mean'] if pd.notna(row['conf_mean']) else None,
                    'watermark_flags': row['watermark_flags'],
                    'quality_score': self.calculate_quality_score(row)
                }
                
                base_chunk_id = f"chunk_{chunk_counter:06d}"
                
                if row['block_type'] == 'table':
                    # Table chunking
                    chunks = self.chunk_table_block(row['text'], base_chunk_id, metadata)
                else:
                    # Text chunking
                    chunks = self.chunk_text_block(row['text'], base_chunk_id, metadata)
                
                # Quality-aware inclusion instead of hard filtering
                quality_chunks = []
                for chunk in chunks:
                    # Only skip completely empty or malformed chunks
                    if not chunk.text.strip() or len(chunk.text.strip()) < 3:
                        filtered_count += 1
                        continue
                    
                    # Include all chunks but mark quality tiers for downstream use
                    quality_chunks.append(chunk)
                
                all_chunks.extend(quality_chunks)
                chunk_counter += len(chunks)  # Count all attempted chunks
        
        print(f"  Generated {len(all_chunks)} chunks from {len(corpus_df)} blocks")
        if filtered_count > 0:
            print(f"  Filtered out {filtered_count} low-quality chunks")
        return all_chunks


class EmbeddingEngine:
    """Embedding engine with multiple model support."""
    
    def __init__(self, model_name: str = 'bge-m3'):
        """
        Initialize embedding engine.
        
        Args:
            model_name: Name of the embedding model
        """
        self.model_name = model_name
        self.model = None
        self._tfidf = None
        self._svd = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        if SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer is not None:
            print(f"Loading embedding model: {self.model_name}")
            # Map model names to actual model identifiers
            model_map = {
                'bge-m3': 'BAAI/bge-m3',
                'e5-base': 'intfloat/e5-base-v2',
                'bge-base': 'BAAI/bge-base-en-v1.5'
            }
            model_id = model_map.get(self.model_name, self.model_name)
            try:
                self.model = SentenceTransformer(model_id)
                print(f"  Model loaded: {model_id}")
                print(f"  Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            except Exception as e:
                print(f"Error loading model {model_id}: {e}")
                print("Falling back to all-MiniLM-L6-v2...")
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.model_name = 'all-MiniLM-L6-v2'
        else:
            # Lightweight fallback: TF-IDF + SVD to fixed dimension
            print("Using TF-IDF + SVD fallback for embeddings")
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            self._tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
            self._svd = TruncatedSVD(n_components=384, random_state=42)
            self.model_name = 'tfidf-svd-384'
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self.model is not None:
            return self.model.get_sentence_embedding_dimension()
        # Fallback dimension
        return 384
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Embedding matrix (n_texts × embedding_dim)
        """
        if not texts:
            return np.array([])
        
        print(f"Generating embeddings for {len(texts)} texts...")
        if self.model is not None:
            # sentence-transformers path
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
                embeddings.append(batch_embeddings)
                if (i // batch_size + 1) % 10 == 0:
                    print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
            return np.vstack(embeddings)
        else:
            # TF-IDF + SVD path
            assert self._tfidf is not None and self._svd is not None
            # Fit on all texts at once for simplicity
            tfidf_matrix = self._tfidf.fit_transform(texts)
            reduced = self._svd.fit_transform(tfidf_matrix)
            return reduced.astype(np.float32)


class FAISSIndexBuilder:
    """FAISS index builder with metadata management."""
    
    def __init__(self, embedding_dim: int):
        """
        Initialize FAISS index builder.
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = None
    
    def build_index(self, embeddings: np.ndarray, index_type: str = 'HNSW'):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Embedding matrix
            index_type: Type of FAISS index ('HNSW', 'IVF', 'Flat')
            
        Returns:
            FAISS index
        """
        n_vectors, dim = embeddings.shape
        print(f"Building index for {n_vectors} vectors (dim={dim})")

        embeddings_f32 = embeddings.astype(np.float32)

        if FAISS_AVAILABLE:
            if index_type == 'HNSW':
                index = faiss.IndexHNSWFlat(dim, 32)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 100
            elif index_type == 'IVF':
                nlist = min(max(int(np.sqrt(n_vectors)), 10), 1000)
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFPQ(quantizer, dim, nlist, 8, 8)
                index.train(embeddings_f32)
            else:
                index = faiss.IndexFlatIP(dim)

            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings_f32)
            index.add(embeddings_f32)
            print(f"  FAISS index built: {index.ntotal} vectors")
            self.index = index
            return index
        else:
            # NumPy fallback: store normalized embeddings for cosine search
            norms = np.linalg.norm(embeddings_f32, axis=1, keepdims=True) + 1e-12
            normalized = embeddings_f32 / norms
            self.index = normalized  # type: ignore
            print("  NumPy index prepared (normalized embeddings)")
            return normalized
    
    def save_index(self, index, output_path: Path):
        """Save index to file (FAISS or NumPy fallback)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if FAISS_AVAILABLE and hasattr(faiss, 'write_index') and not isinstance(index, np.ndarray):
            print(f"Saving FAISS index to {output_path}")
            faiss.write_index(index, str(output_path))
        else:
            # Save as .npy
            npy_path = output_path.with_suffix('.npy')
            print(f"FAISS unavailable. Saving NumPy index to {npy_path}")
            np.save(npy_path, index)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chunking and embedding for NepaliGov-RAG-Bench"
    )
    parser.add_argument(
        "--in",
        dest="input_file",
        type=Path,
        required=True,
        help="Input corpus Parquet file"
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output FAISS index file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bge-m3",
        choices=["bge-m3", "e5-base", "bge-base"],
        help="Embedding model to use (default: bge-m3)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Target chunk size in tokens (default: 1000)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Overlap between chunks in tokens (default: 200)"
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="HNSW",
        choices=["HNSW", "IVF", "Flat"],
        help="FAISS index type (default: HNSW)"
    )
    
    args = parser.parse_args()
    
    # Check preconditions
    if not args.input_file.exists():
        print(f"❌ Input file not found: {args.input_file}")
        sys.exit(1)
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Warning: sentence-transformers not available. Using TF-IDF+SVD fallback.")
    
    try:
        # Load corpus
        print(f"Loading corpus from {args.input_file}...")
        corpus_df = pd.read_parquet(args.input_file)
        print(f"  Loaded {len(corpus_df)} corpus rows")
        
        # Initialize chunking engine
        chunker = ChunkingEngine(
            text_chunk_size=args.chunk_size,
            text_overlap=args.overlap
        )
        
        # Generate chunks
        chunks = chunker.chunk_corpus(corpus_df)
        
        if not chunks:
            print("❌ No chunks generated")
            sys.exit(1)
        
        # Initialize embedding engine
        embedder = EmbeddingEngine(model_name=args.model)
        
        # Generate embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = embedder.embed_texts(chunk_texts)
        
        # Build FAISS index
        index_builder = FAISSIndexBuilder(embedder.get_embedding_dim())
        index = index_builder.build_index(embeddings, args.index_type)
        
        # Create output directory
        args.out.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_builder.save_index(index, args.out)
        
        # Save metadata
        meta_path = args.out.parent / "meta.parquet"
        chunk_metadata = pd.DataFrame([chunk.to_dict() for chunk in chunks])
        chunk_metadata.to_parquet(meta_path, index=False)
        print(f"Saved metadata to {meta_path}")
        
        # Save index manifest
        manifest_path = args.out.parent / "index.manifest.json"
        manifest = {
            "model_name": embedder.model_name,
            "dim": embedder.get_embedding_dim(),
            "nvecs": len(chunks),
            "built_at": datetime.now().isoformat(),
            "tokenizer_info": {
                "chunk_size": args.chunk_size,
                "overlap": args.overlap,
                "table_max_length": chunker.table_max_length
            },
            "index_type": args.index_type,
            "corpus_source": str(args.input_file),
            "chunk_types": chunk_metadata['chunk_type'].value_counts().to_dict(),
            "languages": chunk_metadata['language'].value_counts().to_dict()
        }
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"Saved manifest to {manifest_path}")
        
        print(f"\n✅ Chunking and embedding complete:")
        print(f"   Chunks generated: {len(chunks)}")
        print(f"   Text chunks: {len([c for c in chunks if c.chunk_type == 'text'])}")
        print(f"   Table chunks: {len([c for c in chunks if c.chunk_type == 'table'])}")
        print(f"   Embedding model: {embedder.model_name}")
        print(f"   Embedding dimension: {embedder.get_embedding_dim()}")
        print(f"   FAISS index: {args.out}")
        print(f"   Metadata: {meta_path}")
        print(f"   Manifest: {manifest_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
