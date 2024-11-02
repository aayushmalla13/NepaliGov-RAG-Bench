#!/usr/bin/env python3
"""
Query Language Detection for NepaliGov-RAG-Bench

Lazy-cached language detection with langid/fastlid fallbacks.
"""

import argparse
import sys
from typing import Optional, Tuple
from pathlib import Path

# Optional imports with fallbacks
try:
    import langid
    LANGID_AVAILABLE = True
except ImportError:
    LANGID_AVAILABLE = False

try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False


class QueryLanguageDetector:
    """Lazy-cached language detector with multiple backends."""
    
    def __init__(self, backend: str = 'auto'):
        """
        Initialize language detector.
        
        Args:
            backend: 'auto', 'langid', 'fasttext', or 'heuristic'
        """
        self.backend = backend
        self._langid_model = None
        self._fasttext_model = None
        self._cache = {}
    
    def _load_langid(self):
        """Lazy load langid model."""
        if not LANGID_AVAILABLE:
            raise ImportError("langid not available")
        if self._langid_model is None:
            self._langid_model = langid
        return self._langid_model
    
    def _load_fasttext(self):
        """Lazy load fasttext model."""
        if not FASTTEXT_AVAILABLE:
            raise ImportError("fasttext not available")
        if self._fasttext_model is None:
            # Try to load a pretrained language identification model
            try:
                self._fasttext_model = fasttext.load_model('lid.176.bin')
            except Exception:
                # Fallback: no fasttext model available
                raise ImportError("fasttext model not available")
        return self._fasttext_model
    
    def _devanagari_ratio(self, text: str) -> float:
        """Calculate ratio of Devanagari characters."""
        if not text:
            return 0.0
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars == 0:
            return 0.0
        devanagari_chars = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
        return devanagari_chars / alpha_chars
    
    def _heuristic_detect(self, text: str) -> Tuple[str, float]:
        """Heuristic language detection based on script analysis."""
        if not text.strip():
            return ('mixed', 0.0)
        
        dev_ratio = self._devanagari_ratio(text)
        
        if dev_ratio > 0.7:
            return ('ne', min(1.0, 0.7 + 0.3 * dev_ratio))
        elif dev_ratio < 0.05:
            return ('en', min(1.0, 0.7 + 0.3 * (1.0 - dev_ratio)))
        else:
            return ('mixed', 0.5)
    
    def detect(self, query: str) -> Tuple[str, float]:
        """
        Detect query language.
        
        Args:
            query: Input query text
            
        Returns:
            Tuple of (language_code, confidence) where language_code in {'ne', 'en', 'mixed'}
        """
        # Check cache
        if query in self._cache:
            return self._cache[query]
        
        result = self._detect_uncached(query)
        self._cache[query] = result
        return result
    
    def _detect_uncached(self, query: str) -> Tuple[str, float]:
        """Internal detection without caching."""
        if not query.strip():
            return ('mixed', 0.0)
        
        if self.backend == 'heuristic':
            return self._heuristic_detect(query)
        
        elif self.backend == 'langid':
            try:
                model = self._load_langid()
                lang, conf = model.classify(query)
                # Map langid output to our schema
                if lang in ('ne', 'hi') and self._devanagari_ratio(query) > 0.3:
                    return ('ne', max(0.0, min(1.0, conf)))
                elif lang == 'en' and self._devanagari_ratio(query) < 0.1:
                    return ('en', max(0.0, min(1.0, conf)))
                else:
                    return ('mixed', max(0.0, min(1.0, conf * 0.8)))
            except Exception:
                return self._heuristic_detect(query)
        
        elif self.backend == 'fasttext':
            try:
                model = self._load_fasttext()
                predictions = model.predict(query, k=1)
                lang = predictions[0][0].replace('__label__', '')
                conf = predictions[1][0]
                # Map fasttext output to our schema
                if lang in ('ne', 'hi') and self._devanagari_ratio(query) > 0.3:
                    return ('ne', max(0.0, min(1.0, conf)))
                elif lang == 'en' and self._devanagari_ratio(query) < 0.1:
                    return ('en', max(0.0, min(1.0, conf)))
                else:
                    return ('mixed', max(0.0, min(1.0, conf * 0.8)))
            except Exception:
                return self._heuristic_detect(query)
        
        else:  # auto
            # Try backends in order of preference
            if LANGID_AVAILABLE:
                try:
                    return self._detect_uncached(query.replace(self.backend, 'langid'))
                except Exception:
                    pass
            
            if FASTTEXT_AVAILABLE:
                try:
                    return self._detect_uncached(query.replace(self.backend, 'fasttext'))
                except Exception:
                    pass
            
            # Fallback to heuristic
            return self._heuristic_detect(query)


def main():
    """CLI demo for query language detection."""
    parser = argparse.ArgumentParser(
        description="Detect query language for NepaliGov-RAG-Bench"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "langid", "fasttext", "heuristic"],
        help="Detection backend to use"
    )
    parser.add_argument(
        "queries",
        nargs="*",
        help="Query strings to detect (or read from stdin)"
    )
    
    args = parser.parse_args()
    
    detector = QueryLanguageDetector(backend=args.backend)
    
    # Get queries from args or stdin
    if args.queries:
        queries = args.queries
    else:
        queries = [line.strip() for line in sys.stdin if line.strip()]
    
    if not queries:
        print("No queries provided", file=sys.stderr)
        sys.exit(1)
    
    # Detect languages
    for query in queries:
        lang, conf = detector.detect(query)
        print(f"{lang}\t{conf:.3f}\t{query}")


if __name__ == "__main__":
    main()



