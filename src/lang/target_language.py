#!/usr/bin/env python3
"""
Target Language Selection for NepaliGov-RAG-Bench

Implements intelligent target language selection for multilingual retrieval,
supporting auto-detection, explicit language control, and bilingual modes.
"""

import argparse
import re
from typing import Union, Tuple, List, Dict, Any
from pathlib import Path

from .query_language import QueryLanguageDetector


class TargetLanguageSelector:
    """Intelligent target language selection for multilingual retrieval."""
    
    def __init__(self, lang_detector: QueryLanguageDetector = None):
        """
        Initialize target language selector.
        
        Args:
            lang_detector: Query language detector instance
        """
        self.lang_detector = lang_detector or QueryLanguageDetector()
        
        # Language confidence thresholds
        self.high_confidence_threshold = 0.8
        self.medium_confidence_threshold = 0.6
        
        # Devanagari script detection
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]')
        
        # Common English/Nepali patterns
        self.english_patterns = [
            r'\b(the|and|or|of|in|to|for|with|by|from|up|about|into|through|during)\b',
            r'\b(health|ministry|government|constitution|emergency|operations|center)\b'
        ]
        
        self.nepali_patterns = [
            r'\b(को|का|की|ले|लाई|मा|बाट|सँग|द्वारा)\b',
            r'\b(स्वास्थ्य|मन्त्रालय|सरकार|संविधान|आपतकालीन)\b'
        ]
    
    def _analyze_script_content(self, text: str) -> Dict[str, float]:
        """Analyze script content ratios in text."""
        if not text:
            return {'devanagari': 0.0, 'latin': 0.0, 'other': 0.0}
        
        total_chars = len(text)
        devanagari_count = len(self.devanagari_pattern.findall(text))
        latin_count = len(re.findall(r'[a-zA-Z]', text))
        other_count = total_chars - devanagari_count - latin_count
        
        return {
            'devanagari': devanagari_count / total_chars if total_chars > 0 else 0.0,
            'latin': latin_count / total_chars if total_chars > 0 else 0.0,
            'other': other_count / total_chars if total_chars > 0 else 0.0
        }
    
    def _detect_language_patterns(self, text: str) -> Dict[str, int]:
        """Detect language-specific patterns in text."""
        text_lower = text.lower()
        
        english_matches = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) 
                            for pattern in self.english_patterns)
        nepali_matches = sum(len(re.findall(pattern, text_lower, re.IGNORECASE)) 
                           for pattern in self.nepali_patterns)
        
        return {
            'english_patterns': english_matches,
            'nepali_patterns': nepali_matches
        }
    
    def choose_target_lang(self, 
                          query_text: str, 
                          output_mode: str = "auto") -> Union[str, Tuple[str, str]]:
        """
        Choose target language(s) for retrieval based on query and mode.
        
        Args:
            query_text: Input query text
            output_mode: One of "auto", "en", "ne", "bilingual"
        
        Returns:
            Single language code ("en" or "ne") or tuple ("en", "ne") for bilingual
        """
        # Handle explicit modes first
        if output_mode == "en":
            return "en"
        elif output_mode == "ne":
            return "ne"
        elif output_mode == "bilingual":
            return ("en", "ne")
        elif output_mode != "auto":
            raise ValueError(f"Invalid output_mode: {output_mode}. Must be 'auto', 'en', 'ne', or 'bilingual'")
        
        # Auto-detection logic
        if not query_text or not query_text.strip():
            return "en"  # Default to English for empty queries
        
        # Get language detection result
        detected_lang, confidence = self.lang_detector.detect(query_text)
        
        # Analyze script content
        script_analysis = self._analyze_script_content(query_text)
        
        # Detect language patterns
        pattern_analysis = self._detect_language_patterns(query_text)
        
        # Decision logic
        if confidence >= self.high_confidence_threshold:
            # High confidence detection
            if detected_lang == "ne":
                return "ne"
            elif detected_lang == "en":
                return "en"
        
        # Medium confidence - use additional signals
        if confidence >= self.medium_confidence_threshold:
            # Strong Devanagari presence suggests Nepali
            if script_analysis['devanagari'] > 0.3:
                return "ne"
            # Strong Latin with English patterns suggests English
            elif script_analysis['latin'] > 0.7 and pattern_analysis['english_patterns'] > 0:
                return "en"
        
        # Low confidence - fall back to script analysis
        if script_analysis['devanagari'] > script_analysis['latin']:
            return "ne"
        elif script_analysis['latin'] > script_analysis['devanagari']:
            return "en"
        
        # Mixed or unclear - check for language patterns
        if pattern_analysis['nepali_patterns'] > pattern_analysis['english_patterns']:
            return "ne"
        elif pattern_analysis['english_patterns'] > pattern_analysis['nepali_patterns']:
            return "en"
        
        # Final fallback based on detected language
        if detected_lang in ["ne", "en"]:
            return detected_lang
        
        # Ultimate fallback
        return "en"
    
    def get_target_analysis(self, 
                           query_text: str, 
                           output_mode: str = "auto") -> Dict[str, Any]:
        """
        Get detailed analysis of target language selection.
        
        Args:
            query_text: Input query text
            output_mode: Output mode
        
        Returns:
            Dictionary with analysis details
        """
        detected_lang, confidence = self.lang_detector.detect(query_text)
        script_analysis = self._analyze_script_content(query_text)
        pattern_analysis = self._detect_language_patterns(query_text)
        target_lang = self.choose_target_lang(query_text, output_mode)
        
        return {
            'query_text': query_text,
            'output_mode': output_mode,
            'target_language': target_lang,
            'detected_language': detected_lang,
            'detection_confidence': confidence,
            'script_analysis': script_analysis,
            'pattern_analysis': pattern_analysis,
            'decision_factors': {
                'high_confidence': confidence >= self.high_confidence_threshold,
                'medium_confidence': confidence >= self.medium_confidence_threshold,
                'dominant_script': 'devanagari' if script_analysis['devanagari'] > script_analysis['latin'] else 'latin',
                'dominant_patterns': 'nepali' if pattern_analysis['nepali_patterns'] > pattern_analysis['english_patterns'] else 'english'
            }
        }


def main():
    """CLI demo for target language selection."""
    parser = argparse.ArgumentParser(description="Target language selection demo")
    parser.add_argument("--query", required=True, help="Query text to analyze")
    parser.add_argument("--mode", choices=["auto", "en", "ne", "bilingual"], 
                       default="auto", help="Output mode")
    parser.add_argument("--lang-backend", default="auto", 
                       help="Language detection backend")
    parser.add_argument("--detailed", action="store_true", 
                       help="Show detailed analysis")
    
    args = parser.parse_args()
    
    try:
        # Initialize components
        lang_detector = QueryLanguageDetector(backend=args.lang_backend)
        selector = TargetLanguageSelector(lang_detector)
        
        if args.detailed:
            # Detailed analysis
            analysis = selector.get_target_analysis(args.query, args.mode)
            
            print(f"Query: {args.query}")
            print(f"Mode: {args.mode}")
            print(f"Target Language: {analysis['target_language']}")
            print(f"Detected: {analysis['detected_language']} (confidence: {analysis['detection_confidence']:.3f})")
            print()
            
            print("Script Analysis:")
            for script, ratio in analysis['script_analysis'].items():
                print(f"  {script}: {ratio:.3f}")
            print()
            
            print("Pattern Analysis:")
            for pattern_type, count in analysis['pattern_analysis'].items():
                print(f"  {pattern_type}: {count}")
            print()
            
            print("Decision Factors:")
            for factor, value in analysis['decision_factors'].items():
                print(f"  {factor}: {value}")
        
        else:
            # Simple output
            target_lang = selector.choose_target_lang(args.query, args.mode)
            print(f"Query: {args.query}")
            print(f"Target Language: {target_lang}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
