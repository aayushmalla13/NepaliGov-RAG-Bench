#!/usr/bin/env python3
"""
Advanced Bilingual Query Processing for NepaliGov-RAG-Bench

Implements sophisticated bilingual query understanding, semantic cross-language mapping,
and domain-aware language processing for enhanced multilingual retrieval.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .query_language import QueryLanguageDetector
from .target_language import TargetLanguageSelector


@dataclass
class BilingualQueryAnalysis:
    """Analysis result for bilingual queries."""
    original_query: str
    language_segments: List[Dict[str, Any]]
    semantic_mappings: List[Dict[str, Any]]
    domain_context: str
    processing_strategy: str
    confidence: float


class SemanticCrossLanguageMapper:
    """Maps concepts between Nepali and English with domain awareness."""
    
    def __init__(self):
        # Domain-specific concept mappings
        self.concept_mappings = {
            'constitution': {
                'ne_terms': ['संविधान', 'कानून', 'ऐन', 'नियम', 'विधान'],
                'en_terms': ['constitution', 'law', 'act', 'regulation', 'statute'],
                'semantic_weight': 1.0
            },
            'health': {
                'ne_terms': ['स्वास्थ्य', 'चिकित्सा', 'उपचार', 'रोग', 'आरोग्य'],
                'en_terms': ['health', 'medical', 'treatment', 'disease', 'healthcare'],
                'semantic_weight': 0.9
            },
            'government': {
                'ne_terms': ['सरकार', 'राज्य', 'प्रशासन', 'मन्त्रालय', 'निकाय'],
                'en_terms': ['government', 'state', 'administration', 'ministry', 'agency'],
                'semantic_weight': 0.85
            },
            'emergency': {
                'ne_terms': ['आपातकाल', 'संकटकाल', 'विपद्', 'दुर्घटना', 'आकस्मिक'],
                'en_terms': ['emergency', 'crisis', 'disaster', 'accident', 'urgent'],
                'semantic_weight': 0.8
            },
            'rights': {
                'ne_terms': ['अधिकार', 'हक', 'स्वतन्त्रता', 'न्याय', 'समानता'],
                'en_terms': ['rights', 'freedom', 'liberty', 'justice', 'equality'],
                'semantic_weight': 0.9
            }
        }
        
        # Grammatical patterns for better segmentation
        self.nepali_patterns = [
            r'[\u0900-\u097F]+',  # Devanagari script
            r'(?:को|का|की|ले|लाई|मा|बाट|सँग|द्वारा)',  # Nepali particles
        ]
        
        self.english_patterns = [
            r'\b[a-zA-Z]+\b',  # English words
            r'(?:the|and|or|of|in|to|for|with|by|from)\b',  # English function words
        ]
    
    def find_semantic_mappings(self, query: str) -> List[Dict[str, Any]]:
        """Find semantic concept mappings in the query."""
        mappings = []
        query_lower = query.lower()
        
        for concept, data in self.concept_mappings.items():
            ne_matches = []
            en_matches = []
            
            # Find Nepali terms
            for term in data['ne_terms']:
                if term in query:
                    ne_matches.append(term)
            
            # Find English terms
            for term in data['en_terms']:
                if term in query_lower:
                    en_matches.append(term)
            
            if ne_matches or en_matches:
                mappings.append({
                    'concept': concept,
                    'nepali_terms': ne_matches,
                    'english_terms': en_matches,
                    'semantic_weight': data['semantic_weight'],
                    'cross_language_potential': len(ne_matches) > 0 and len(en_matches) > 0
                })
        
        return mappings
    
    def generate_cross_language_expansions(self, query: str, mappings: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Generate cross-language query expansions based on semantic mappings."""
        expansions = {'ne': [], 'en': []}
        
        for mapping in mappings:
            concept = mapping['concept']
            ne_terms = mapping['nepali_terms']
            en_terms = mapping['english_terms']
            
            # If query has English terms, add Nepali equivalents
            if en_terms and not ne_terms:
                for en_term in en_terms:
                    for ne_equivalent in self.concept_mappings[concept]['ne_terms'][:2]:  # Top 2
                        expanded = query.replace(en_term, ne_equivalent)
                        if expanded != query:
                            expansions['ne'].append(expanded)
            
            # If query has Nepali terms, add English equivalents
            if ne_terms and not en_terms:
                for ne_term in ne_terms:
                    for en_equivalent in self.concept_mappings[concept]['en_terms'][:2]:  # Top 2
                        expanded = query.replace(ne_term, en_equivalent)
                        if expanded != query:
                            expansions['en'].append(expanded)
        
        return expansions


class AdvancedBilingualProcessor:
    """Advanced bilingual query processor with semantic understanding."""
    
    def __init__(self, 
                 lang_detector: Optional[QueryLanguageDetector] = None,
                 target_selector: Optional[TargetLanguageSelector] = None):
        """
        Initialize advanced bilingual processor.
        
        Args:
            lang_detector: Language detector instance
            target_selector: Target language selector instance
        """
        self.lang_detector = lang_detector or QueryLanguageDetector()
        self.target_selector = target_selector or TargetLanguageSelector(self.lang_detector)
        self.semantic_mapper = SemanticCrossLanguageMapper()
        
        # Domain detection patterns
        self.domain_patterns = {
            'constitution': r'(?:संविधान|constitution|कानून|law|अधिकार|rights|न्याय|justice)',
            'health': r'(?:स्वास्थ्य|health|चिकित्सा|medical|रोग|disease|उपचार|treatment)',
            'emergency': r'(?:आपातकाल|emergency|संकट|crisis|विपद्|disaster|HEOC)',
            'government': r'(?:सरकार|government|मन्त्रालय|ministry|प्रशासन|administration)',
            'covid': r'(?:कोभिड|COVID|कोरोना|corona|महामारी|pandemic|भ्याक्सिन|vaccine)'
        }
    
    def detect_domain_context(self, query: str) -> str:
        """Detect the domain context of the query."""
        query_combined = query.lower()
        
        domain_scores = {}
        for domain, pattern in self.domain_patterns.items():
            matches = len(re.findall(pattern, query_combined, re.IGNORECASE))
            if matches > 0:
                domain_scores[domain] = matches
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return 'general'
    
    def segment_bilingual_query(self, query: str) -> List[Dict[str, Any]]:
        """Segment bilingual query into language-specific parts."""
        segments = []
        
        # Find Devanagari (Nepali) segments
        nepali_matches = re.finditer(r'[\u0900-\u097F\s]+', query)
        for match in nepali_matches:
            if match.group().strip():
                segments.append({
                    'text': match.group().strip(),
                    'language': 'ne',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9
                })
        
        # Find Latin (English) segments
        english_matches = re.finditer(r'[a-zA-Z\s\-]+', query)
        for match in english_matches:
            text = match.group().strip()
            if text and len(text) > 2:  # Avoid single chars
                # Check if it overlaps with Nepali segments
                overlaps = any(
                    match.start() < seg['end'] and match.end() > seg['start']
                    for seg in segments
                )
                if not overlaps:
                    segments.append({
                        'text': text,
                        'language': 'en',
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.8
                    })
        
        # Sort by position
        segments.sort(key=lambda x: x['start'])
        return segments
    
    def analyze_bilingual_query(self, query: str) -> BilingualQueryAnalysis:
        """Perform comprehensive bilingual query analysis."""
        # Basic language detection
        detected_lang, base_confidence = self.lang_detector.detect(query)
        
        # Segment the query
        segments = self.segment_bilingual_query(query)
        
        # Find semantic mappings
        semantic_mappings = self.semantic_mapper.find_semantic_mappings(query)
        
        # Detect domain context
        domain_context = self.detect_domain_context(query)
        
        # Determine processing strategy
        if len(segments) > 1:
            processing_strategy = 'bilingual_segmented'
            confidence = 0.8
        elif semantic_mappings and any(m['cross_language_potential'] for m in semantic_mappings):
            processing_strategy = 'semantic_expansion'
            confidence = 0.7
        elif detected_lang == 'mixed':
            processing_strategy = 'mixed_language'
            confidence = base_confidence
        else:
            processing_strategy = 'monolingual'
            confidence = base_confidence
        
        return BilingualQueryAnalysis(
            original_query=query,
            language_segments=segments,
            semantic_mappings=semantic_mappings,
            domain_context=domain_context,
            processing_strategy=processing_strategy,
            confidence=confidence
        )
    
    def generate_enhanced_expansions(self, analysis: BilingualQueryAnalysis) -> Dict[str, List[str]]:
        """Generate enhanced query expansions based on bilingual analysis."""
        expansions = {'ne': [analysis.original_query], 'en': [analysis.original_query]}
        
        # Strategy-specific expansions
        if analysis.processing_strategy == 'bilingual_segmented':
            # Create language-specific versions
            ne_parts = [seg['text'] for seg in analysis.language_segments if seg['language'] == 'ne']
            en_parts = [seg['text'] for seg in analysis.language_segments if seg['language'] == 'en']
            
            if ne_parts:
                expansions['ne'].append(' '.join(ne_parts))
            if en_parts:
                expansions['en'].append(' '.join(en_parts))
        
        elif analysis.processing_strategy == 'semantic_expansion':
            # Use semantic mappings for cross-language expansion
            semantic_expansions = self.semantic_mapper.generate_cross_language_expansions(
                analysis.original_query, analysis.semantic_mappings
            )
            expansions['ne'].extend(semantic_expansions['ne'])
            expansions['en'].extend(semantic_expansions['en'])
        
        # Domain-specific enhancements
        if analysis.domain_context != 'general':
            domain_expansions = self._get_domain_expansions(
                analysis.original_query, analysis.domain_context
            )
            for lang in ['ne', 'en']:
                expansions[lang].extend(domain_expansions.get(lang, []))
        
        # Remove duplicates and limit
        for lang in ['ne', 'en']:
            expansions[lang] = list(dict.fromkeys(expansions[lang]))[:4]  # Max 4 per language
        
        return expansions
    
    def _get_domain_expansions(self, query: str, domain: str) -> Dict[str, List[str]]:
        """Get domain-specific query expansions."""
        domain_templates = {
            'constitution': {
                'ne': ['संविधानमा {query}', '{query} कानूनी व्यवस्था'],
                'en': ['constitutional {query}', '{query} legal framework']
            },
            'health': {
                'ne': ['स्वास्थ्य क्षेत्रमा {query}', '{query} चिकित्सा सेवा'],
                'en': ['health sector {query}', '{query} medical services']
            },
            'emergency': {
                'ne': ['आपातकालीन {query}', '{query} संकट व्यवस्थापन'],
                'en': ['emergency {query}', '{query} crisis management']
            }
        }
        
        expansions = {'ne': [], 'en': []}
        if domain in domain_templates:
            templates = domain_templates[domain]
            for lang in ['ne', 'en']:
                for template in templates[lang]:
                    expanded = template.replace('{query}', query)
                    if expanded != query:
                        expansions[lang].append(expanded)
        
        return expansions


def main():
    """CLI demo for advanced bilingual processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced bilingual query processing demo")
    parser.add_argument("--query", required=True, help="Query to analyze")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    
    args = parser.parse_args()
    
    try:
        processor = AdvancedBilingualProcessor()
        analysis = processor.analyze_bilingual_query(args.query)
        
        print(f"Query: {args.query}")
        print(f"Domain: {analysis.domain_context}")
        print(f"Strategy: {analysis.processing_strategy}")
        print(f"Confidence: {analysis.confidence:.3f}")
        
        if args.detailed:
            print(f"\nLanguage Segments:")
            for seg in analysis.language_segments:
                print(f"  {seg['language']}: '{seg['text']}' (conf: {seg['confidence']:.3f})")
            
            print(f"\nSemantic Mappings:")
            for mapping in analysis.semantic_mappings:
                print(f"  {mapping['concept']}: NE={mapping['nepali_terms']}, EN={mapping['english_terms']}")
            
            expansions = processor.generate_enhanced_expansions(analysis)
            print(f"\nEnhanced Expansions:")
            for lang, terms in expansions.items():
                print(f"  {lang}: {terms}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



