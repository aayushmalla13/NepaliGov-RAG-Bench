#!/usr/bin/env python3
"""
Cite-Constrained Answerer with Advanced Bilingual Intelligence

Produces faithful answers that cite only authoritative spans, fully integrating
CP9 bilingual intelligence and CP6 expansion/diversification features.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CitationSpan:
    """Represents a citation span with precise location information."""
    doc_id: str
    page_num: int
    start_char: int
    end_char: int
    text: str
    bbox: Optional[List[float]] = None
    language: str = 'unknown'
    is_authoritative: bool = True
    
    def to_citation_token(self) -> str:
        """Generate citation token in format [[doc:ID|page:P|span:S:E]]."""
        return f"[[doc:{self.doc_id}|page:{self.page_num}|span:{self.start_char}:{self.end_char}]]"


@dataclass 
class AnswerContext:
    """Context information for answer generation."""
    query: str
    target_language: Union[str, Tuple[str, str]]
    processing_strategy: str
    query_domain: str
    bilingual_confidence: float
    semantic_mappings_count: int
    language_segments_count: int
    domain_success_rate: float = 0.0
    fallback_threshold: float = 0.5


@dataclass
class Answer:
    """Generated answer with citations and metadata."""
    text: str
    citations: List[CitationSpan] = field(default_factory=list)
    language: str = 'en'
    is_refusal: bool = False
    evidence_count: int = 0
    same_lang_evidence_count: int = 0
    cross_lang_evidence_count: int = 0
    table_context_used: bool = False


@dataclass
class BilingualAnswer:
    """Bilingual answer with separate EN and NE sections."""
    english: Optional[Answer] = None
    nepali: Optional[Answer] = None
    combined_citations: List[CitationSpan] = field(default_factory=list)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)


class CitationExtractor:
    """Extracts and validates citation spans from candidates."""
    
    def __init__(self):
        # Enhanced minimum evidence thresholds (more lenient)
        self.min_evidence_chars = 30  # Reduced from 50 for better coverage
        self.min_evidence_confidence = 0.2  # Reduced from 0.3 for more inclusion
    
    def extract_authoritative_spans(self, 
                                  candidates: List[Dict[str, Any]], 
                                  target_lang: str = None) -> List[CitationSpan]:
        """Extract authoritative citation spans from candidates."""
        spans = []
        
        for candidate in candidates:
            # Enhanced authority check with fallbacks
            is_auth = candidate.get('is_authoritative', False)
            if not is_auth:
                # Check alternative authority indicators
                source_auth = candidate.get('source_authority', '')
                doc_id = candidate.get('doc_id', '')
                if (source_auth == 'authoritative' or 
                    any(prefix in doc_id for prefix in ['constitution', 'health', 'covid', 'heoc', 'ministry'])):
                    is_auth = True
            
            if not is_auth:
                continue
            
            # Extract basic information
            doc_id = candidate.get('doc_id', 'unknown')
            page_num = candidate.get('page_num', 0)
            text = candidate.get('text', '')
            language = candidate.get('language', 'unknown')
            
            # Skip if insufficient text
            if len(text.strip()) < self.min_evidence_chars:
                continue
            
            # Language filtering
            if target_lang and target_lang != 'mixed' and language != target_lang:
                continue
            
            # Extract character spans
            char_spans = candidate.get('char_spans', [])
            if char_spans:
                for span_info in char_spans:
                    if isinstance(span_info, dict):
                        start_char = span_info.get('start', 0)
                        end_char = span_info.get('end', len(text))
                        span_text = text[start_char:end_char] if end_char > start_char else text
                    else:
                        # Fallback: use entire text
                        start_char = 0
                        end_char = len(text)
                        span_text = text
                    
                    # Create citation span
                    span = CitationSpan(
                        doc_id=doc_id,
                        page_num=page_num,
                        start_char=start_char,
                        end_char=end_char,
                        text=span_text,
                        bbox=candidate.get('bbox'),
                        language=language,
                        is_authoritative=True
                    )
                    spans.append(span)
            else:
                # No specific spans, use entire text
                span = CitationSpan(
                    doc_id=doc_id,
                    page_num=page_num,
                    start_char=0,
                    end_char=len(text),
                    text=text,
                    bbox=candidate.get('bbox'),
                    language=language,
                    is_authoritative=True
                )
                spans.append(span)
        
        return spans
    
    def filter_spans_by_language(self, 
                                spans: List[CitationSpan], 
                                preferred_lang: str,
                                fallback_threshold: float = 0.5) -> Tuple[List[CitationSpan], bool]:
        """Filter spans by language preference with fallback logic."""
        if not spans:
            return [], False
        
        # Separate by language
        same_lang_spans = [s for s in spans if s.language == preferred_lang]
        cross_lang_spans = [s for s in spans if s.language != preferred_lang and s.language != 'unknown']
        
        # Calculate coverage
        same_lang_coverage = len(same_lang_spans) / len(spans) if spans else 0
        
        # Apply fallback threshold
        fallback_used = False
        if same_lang_coverage < fallback_threshold and cross_lang_spans:
            # Use both same-language and cross-language spans
            filtered_spans = same_lang_spans + cross_lang_spans[:2]  # Limit cross-lang
            fallback_used = True
        else:
            # Use only same-language spans
            filtered_spans = same_lang_spans
        
        return filtered_spans, fallback_used


class EvidenceSelector:
    """Selects and organizes evidence for answer generation."""
    
    def __init__(self):
        self.max_evidence_per_answer = 5
        self.max_same_page_evidence = 2
        self.table_keywords = ['table', 'टेबल', 'तालिका', 'chart', 'चार्ट']
    
    def select_evidence(self, 
                       spans: List[CitationSpan], 
                       context: AnswerContext,
                       table_context: bool = True) -> List[CitationSpan]:
        """Select best evidence spans for answer generation."""
        if not spans:
            return []
        
        # Apply diversification (avoid over-concentration)
        diversified_spans = self._diversify_spans(spans)
        
        # Apply domain-specific selection
        domain_filtered = self._apply_domain_filtering(diversified_spans, context.query_domain)
        
        # Apply table-aware selection
        if table_context and self._is_table_query(context.query):
            table_enhanced = self._enhance_table_context(domain_filtered)
        else:
            table_enhanced = domain_filtered
        
        # Limit to max evidence
        selected = table_enhanced[:self.max_evidence_per_answer]
        
        return selected
    
    def _diversify_spans(self, spans: List[CitationSpan]) -> List[CitationSpan]:
        """Apply diversification to avoid over-concentration on single page/doc."""
        if len(spans) <= self.max_evidence_per_answer:
            return spans
        
        # Group by page
        page_groups = {}
        for span in spans:
            page_key = f"{span.doc_id}:{span.page_num}"
            if page_key not in page_groups:
                page_groups[page_key] = []
            page_groups[page_key].append(span)
        
        # Select diverse spans
        diversified = []
        page_keys = list(page_groups.keys())
        
        # Round-robin selection from different pages
        for i in range(self.max_evidence_per_answer):
            if not page_keys:
                break
            
            page_key = page_keys[i % len(page_keys)]
            page_spans = page_groups[page_key]
            
            if page_spans:
                # Take best span from this page
                best_span = max(page_spans, key=lambda s: len(s.text))
                diversified.append(best_span)
                page_spans.remove(best_span)
                
                # Remove empty groups
                if not page_spans:
                    page_keys.remove(page_key)
        
        return diversified
    
    def _apply_domain_filtering(self, 
                               spans: List[CitationSpan], 
                               domain: str) -> List[CitationSpan]:
        """Apply domain-specific filtering and ranking."""
        if domain == 'general':
            return spans
        
        # Domain keywords for relevance scoring
        domain_keywords = {
            'constitution': ['संविधान', 'constitution', 'कानून', 'law', 'अधिकार', 'rights'],
            'health': ['स्वास्थ्य', 'health', 'चिकित्सा', 'medical', 'उपचार', 'treatment'],
            'emergency': ['आपातकाल', 'emergency', 'संकट', 'crisis', 'विपद्', 'disaster'],
            'government': ['सरकार', 'government', 'मन्त्रालय', 'ministry', 'प्रशासन', 'administration'],
            'covid': ['कोभिड', 'COVID', 'कोरोना', 'corona', 'महामारी', 'pandemic']
        }
        
        keywords = domain_keywords.get(domain, [])
        if not keywords:
            return spans
        
        # Score spans by domain relevance
        scored_spans = []
        for span in spans:
            score = 0
            text_lower = span.text.lower()
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1
            scored_spans.append((span, score))
        
        # Sort by relevance score (descending)
        scored_spans.sort(key=lambda x: x[1], reverse=True)
        
        return [span for span, score in scored_spans]
    
    def _is_table_query(self, query: str) -> bool:
        """Check if query is table-related."""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.table_keywords)
    
    def _enhance_table_context(self, spans: List[CitationSpan]) -> List[CitationSpan]:
        """Enhance spans with table context information."""
        # For now, just return spans as-is
        # In a full implementation, this would add table cell context
        for span in spans:
            if any(keyword in span.text.lower() for keyword in self.table_keywords):
                span.table_context_used = True
        
        return spans


class AnswerGenerator:
    """Generates faithful answers with proper citations."""
    
    def __init__(self):
        self.citation_extractor = CitationExtractor()
        self.evidence_selector = EvidenceSelector()
        
        # Enhanced refusal messages
        self.refusal_messages = {
            'en': "I don't have sufficient authoritative evidence in these documents to provide a reliable answer to your question.",
            'ne': "यी कागजातहरूमा तपाईंको प्रश्नको भरपर्दो जवाफ दिनका लागि पर्याप्त आधिकारिक प्रमाण छैन।"
        }
    
    def generate_answer(self, 
                       candidates: List[Dict[str, Any]], 
                       context: AnswerContext,
                       table_context: bool = True) -> Union[Answer, BilingualAnswer]:
        """Generate answer from candidates with bilingual intelligence."""
        
        # Handle bilingual mode
        if isinstance(context.target_language, tuple):
            return self._generate_bilingual_answer(candidates, context, table_context)
        else:
            return self._generate_single_language_answer(candidates, context, table_context)
    
    def _generate_single_language_answer(self, 
                                       candidates: List[Dict[str, Any]], 
                                       context: AnswerContext,
                                       table_context: bool) -> Answer:
        """Generate single-language answer."""
        target_lang = context.target_language
        
        # Extract authoritative spans
        all_spans = self.citation_extractor.extract_authoritative_spans(candidates)
        
        if not all_spans:
            return Answer(
                text=self.refusal_messages.get(target_lang, self.refusal_messages['en']),
                language=target_lang,
                is_refusal=True
            )
        
        # Apply language filtering with dynamic fallback
        filtered_spans, fallback_used = self.citation_extractor.filter_spans_by_language(
            all_spans, target_lang, context.fallback_threshold
        )
        
        if not filtered_spans:
            return Answer(
                text=self.refusal_messages.get(target_lang, self.refusal_messages['en']),
                language=target_lang,
                is_refusal=True
            )
        
        # Select evidence
        evidence_spans = self.evidence_selector.select_evidence(filtered_spans, context, table_context)
        
        if not evidence_spans:
            return Answer(
                text=self.refusal_messages.get(target_lang, self.refusal_messages['en']),
                language=target_lang,
                is_refusal=True
            )
        
        # Generate answer text
        answer_text = self._synthesize_answer(evidence_spans, context.query, target_lang)
        
        # Count evidence types
        same_lang_count = sum(1 for s in evidence_spans if s.language == target_lang)
        cross_lang_count = len(evidence_spans) - same_lang_count
        
        return Answer(
            text=answer_text,
            citations=evidence_spans,
            language=target_lang,
            is_refusal=False,
            evidence_count=len(evidence_spans),
            same_lang_evidence_count=same_lang_count,
            cross_lang_evidence_count=cross_lang_count,
            table_context_used=any(getattr(s, 'table_context_used', False) for s in evidence_spans)
        )
    
    def _generate_bilingual_answer(self, 
                                 candidates: List[Dict[str, Any]], 
                                 context: AnswerContext,
                                 table_context: bool) -> BilingualAnswer:
        """Generate bilingual answer with separate EN and NE sections."""
        en_lang, ne_lang = context.target_language
        
        # Extract all authoritative spans
        all_spans = self.citation_extractor.extract_authoritative_spans(candidates)
        
        if not all_spans:
            return BilingualAnswer(
                english=Answer(
                    text=self.refusal_messages['en'],
                    language='en',
                    is_refusal=True
                ),
                nepali=Answer(
                    text=self.refusal_messages['ne'],
                    language='ne',
                    is_refusal=True
                )
            )
        
        # Generate EN section
        en_context = AnswerContext(
            query=context.query,
            target_language='en',
            processing_strategy=context.processing_strategy,
            query_domain=context.query_domain,
            bilingual_confidence=context.bilingual_confidence,
            semantic_mappings_count=context.semantic_mappings_count,
            language_segments_count=context.language_segments_count,
            fallback_threshold=context.fallback_threshold
        )
        en_answer = self._generate_single_language_answer(candidates, en_context, table_context)
        
        # Generate NE section
        ne_context = AnswerContext(
            query=context.query,
            target_language='ne',
            processing_strategy=context.processing_strategy,
            query_domain=context.query_domain,
            bilingual_confidence=context.bilingual_confidence,
            semantic_mappings_count=context.semantic_mappings_count,
            language_segments_count=context.language_segments_count,
            fallback_threshold=context.fallback_threshold
        )
        ne_answer = self._generate_single_language_answer(candidates, ne_context, table_context)
        
        # Combine citations
        combined_citations = []
        if en_answer and en_answer.citations:
            combined_citations.extend(en_answer.citations)
        if ne_answer and ne_answer.citations:
            combined_citations.extend(ne_answer.citations)
        
        # Remove duplicates while preserving order
        seen_citations = set()
        unique_citations = []
        for citation in combined_citations:
            citation_key = (citation.doc_id, citation.page_num, citation.start_char, citation.end_char)
            if citation_key not in seen_citations:
                seen_citations.add(citation_key)
                unique_citations.append(citation)
        
        return BilingualAnswer(
            english=en_answer,
            nepali=ne_answer,
            combined_citations=unique_citations,
            processing_metadata={
                'processing_strategy': context.processing_strategy,
                'query_domain': context.query_domain,
                'bilingual_confidence': context.bilingual_confidence,
                'semantic_mappings_count': context.semantic_mappings_count,
                'language_segments_count': context.language_segments_count,
                'fallback_threshold_used': context.fallback_threshold
            }
        )
    
    def _synthesize_answer(self, 
                          evidence_spans: List[CitationSpan], 
                          query: str, 
                          target_lang: str) -> str:
        """Synthesize answer text from evidence spans with citations."""
        if not evidence_spans:
            return self.refusal_messages.get(target_lang, self.refusal_messages['en'])
        
        # Create answer based on evidence
        answer_parts = []
        
        # Add introduction based on language
        if target_lang == 'ne':
            intro = "कागजातका आधारमा:"
        else:
            intro = "Based on the documents:"
        
        answer_parts.append(intro)
        
        # Add evidence with citations
        for i, span in enumerate(evidence_spans):
            # Clean and truncate evidence text
            evidence_text = span.text.strip()
            if len(evidence_text) > 200:
                evidence_text = evidence_text[:197] + "..."
            
            # Add citation token
            citation_token = span.to_citation_token()
            
            # Format evidence
            if target_lang == 'ne':
                evidence_line = f"• {evidence_text} {citation_token}"
            else:
                evidence_line = f"• {evidence_text} {citation_token}"
            
            answer_parts.append(evidence_line)
        
        return "\n".join(answer_parts)


def main():
    """CLI demo for cite-constrained answerer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cite-constrained answerer demo")
    parser.add_argument("--query", required=True, help="Query to answer")
    parser.add_argument("--candidates-file", required=True, help="JSON file with candidates")
    parser.add_argument("--target-lang", default="auto", help="Target language")
    parser.add_argument("--table-context", action="store_true", help="Enable table context")
    
    args = parser.parse_args()
    
    try:
        # Load candidates
        with open(args.candidates_file, 'r', encoding='utf-8') as f:
            candidates = json.load(f)
        
        # Create context
        context = AnswerContext(
            query=args.query,
            target_language=args.target_lang,
            processing_strategy='monolingual',
            query_domain='general',
            bilingual_confidence=1.0,
            semantic_mappings_count=0,
            language_segments_count=1
        )
        
        # Generate answer
        generator = AnswerGenerator()
        answer = generator.generate_answer(candidates, context, args.table_context)
        
        # Display result
        if isinstance(answer, BilingualAnswer):
            print("=== BILINGUAL ANSWER ===")
            if answer.english:
                print(f"English: {answer.english.text}")
                print(f"EN Citations: {len(answer.english.citations)}")
            if answer.nepali:
                print(f"Nepali: {answer.nepali.text}")
                print(f"NE Citations: {len(answer.nepali.citations)}")
        else:
            print("=== ANSWER ===")
            print(f"Text: {answer.text}")
            print(f"Language: {answer.language}")
            print(f"Citations: {len(answer.citations)}")
            print(f"Is Refusal: {answer.is_refusal}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
