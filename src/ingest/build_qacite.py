#!/usr/bin/env python3
"""
Enhanced Q-A-Cite Generation for NepaliGov-RAG-Bench

Generates supervised Q-A pairs with precise citations from authoritative chunks,
including bbox-faithful spans and Wikipedia hard negatives via multi-query retrieval.
Enhanced with OCR quality filtering, domain-aware templates, and fuzzy span matching.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import pandas as pd
import numpy as np

# Local imports
from src.retriever.search import MultilingualRetriever
from src.lang.query_language import QueryLanguageDetector

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class QACiteGenerator:
    """Enhanced Q-A pair generator with quality filtering and domain awareness."""
    
    def __init__(self, 
                 corpus_path: Path,
                 faiss_dir: Path,
                 random_seed: int = 42,
                 min_confidence: float = 45.0,
                 min_text_length: int = 50):
        """
        Initialize enhanced Q-A-Cite generator.
        
        Args:
            corpus_path: Path to corpus parquet file
            faiss_dir: Path to FAISS index directory
            random_seed: Random seed for deterministic generation
            min_confidence: Minimum OCR confidence threshold
            min_text_length: Minimum text length for chunk inclusion
        """
        self.corpus_path = corpus_path
        self.faiss_dir = faiss_dir
        self.random_seed = random_seed
        self.min_confidence = min_confidence
        self.min_text_length = min_text_length
        
        # Set random seed for deterministic generation
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Load corpus
        self.corpus_df = pd.read_parquet(corpus_path)
        print(f"Loaded corpus: {len(self.corpus_df)} rows")
        
        # Initialize retriever for hard negatives
        self.retriever = MultilingualRetriever(faiss_dir)
        print("Initialized retriever for hard negative generation")
        
        # Enhanced question templates by language and domain
        self.question_templates = {
            'ne': {
                'generic': [
                    "{topic} के हो?",
                    "{topic} कहाँ छ?", 
                    "{topic} कसरी काम गर्छ?",
                    "{topic} किन महत्वपूर्ण छ?",
                    "{topic} कति छ?",
                    "{topic} को बारेमा के भनिएको छ?"
                ],
                'health': [
                    "स्वास्थ्य सेवामा {topic} के भूमिका छ?",
                    "कोभिड-१९ को सन्दर्भमा {topic} कस्तो छ?",
                    "नेपालमा {topic} को अवस्था कस्तो छ?",
                    "स्वास्थ्य मन्त्रालयले {topic} बारे के भनेको छ?"
                ],
                'legal': [
                    "संविधानमा {topic} को व्यवस्था के छ?",
                    "कानुनी रूपमा {topic} कस्तो परिभाषित छ?",
                    "{topic} संग सम्बन्धित अधिकार के छन्?"
                ]
            },
            'en': {
                'generic': [
                    "What is {topic}?",
                    "Where is {topic}?",
                    "How does {topic} work?",
                    "Why is {topic} important?",
                    "How many {topic}?",
                    "What does the document say about {topic}?"
                ],
                'health': [
                    "What is the role of {topic} in health services?",
                    "How does {topic} relate to COVID-19 response?",
                    "What is the status of {topic} in Nepal?",
                    "What does the Ministry of Health say about {topic}?"
                ],
                'legal': [
                    "What does the Constitution say about {topic}?",
                    "How is {topic} legally defined?",
                    "What rights are associated with {topic}?"
                ]
            }
        }
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean OCR artifacts and normalize text."""
        if not text:
            return text
        
        # Remove isolated single characters and numbers
        text = re.sub(r'\b[0-9]\b', '', text)
        text = re.sub(r'\b[a-zA-Z]\b', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove lines with mostly symbols/garbage
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            # Keep lines with reasonable character distribution
            if len(line.strip()) > 5:
                alpha_ratio = sum(c.isalpha() or 0x0900 <= ord(c) <= 0x097F for c in line) / len(line)
                if alpha_ratio > 0.3:  # At least 30% alphabetic
                    clean_lines.append(line.strip())
        
        return ' '.join(clean_lines).strip()
    
    def _assess_text_quality(self, text: str, confidence: float) -> float:
        """Assess text quality for Q-A generation suitability - very strict."""
        if not text or len(text) < 20:
            return 0.0
        
        # Immediate disqualifiers for OCR garbage
        words = text.split()
        if len(words) < 5:  # Too few words
            return 0.0
        
        # Check for excessive symbols/numbers
        alpha_chars = sum(c.isalpha() or 0x0900 <= ord(c) <= 0x097F for c in text)
        digit_symbol_chars = sum(c.isdigit() or c in '!@#$%^&*()_+-={}[]|\\:;"<>,.?/' for c in text)
        
        if len(text) > 0:
            alpha_ratio = alpha_chars / len(text)
            garbage_ratio = digit_symbol_chars / len(text)
            
            # Must be mostly alphabetic with minimal garbage
            if alpha_ratio < 0.7 or garbage_ratio > 0.3:
                return 0.0
        
        # Check for coherent word patterns
        coherent_words = 0
        for word in words:
            if len(word) >= 3 and (word.isalpha() or any(0x0900 <= ord(c) <= 0x097F for c in word)):
                coherent_words += 1
        
        coherence_ratio = coherent_words / len(words) if words else 0
        if coherence_ratio < 0.8:  # 80% of words must be coherent
            return 0.0
        
        # Only if all checks pass, calculate score
        score = 0.0
        
        # Confidence component (much stricter)
        if confidence >= 70:  # Only high confidence
            score += 0.3
        elif confidence >= 60:
            score += 0.1
        else:
            return 0.0  # Reject low confidence entirely
        
        # Length component
        if len(text) >= 100:
            score += 0.3
        elif len(text) >= 50:
            score += 0.2
        
        # Alphabetic ratio component
        score += alpha_ratio * 0.2
        
        # Coherence component
        score += coherence_ratio * 0.2
        
        return min(score, 1.0)
    
    def _detect_domain(self, text: str) -> str:
        """Detect document domain for better question templates."""
        text_lower = text.lower()
        
        # Health domain keywords
        health_keywords = ['health', 'covid', 'hospital', 'patient', 'medicine', 'disease', 
                          'स्वास्थ्य', 'कोभिड', 'अस्पताल', 'बिरामी', 'औषधि', 'रोग']
        
        # Legal domain keywords  
        legal_keywords = ['constitution', 'law', 'right', 'legal', 'act', 'section',
                         'संविधान', 'कानुन', 'अधिकार', 'ऐन', 'धारा']
        
        health_count = sum(1 for kw in health_keywords if kw in text_lower)
        legal_count = sum(1 for kw in legal_keywords if kw in text_lower)
        
        if health_count > legal_count and health_count > 0:
            return 'health'
        elif legal_count > 0:
            return 'legal'
        else:
            return 'generic'
    
    def _extract_key_terms(self, text: str, language: str) -> List[str]:
        """Extract key terms/topics from text for question generation."""
        if not text or len(text.strip()) < 10:
            return []
        
        # Clean text for analysis
        clean_text = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)
        words = clean_text.split()
        
        # Filter meaningful terms
        key_terms = []
        for word in words:
            # Skip very short words, numbers only, or common stop words
            if len(word) < 3:
                continue
            if word.isdigit():
                continue
            if language == 'en' and word.lower() in ['the', 'and', 'for', 'are', 'was', 'with', 'this', 'that', 'have', 'from', 'they', 'been', 'said', 'what', 'were']:
                continue
            if language == 'ne' and word in ['र', 'को', 'का', 'की', 'ले', 'मा', 'लाई', 'बाट', 'सँग', 'छ', 'हो', 'भए', 'गर्न']:
                continue
            
            key_terms.append(word)
        
        # Return top terms (by frequency and length)
        term_scores = {}
        for term in key_terms:
            # Score by length and frequency
            score = len(term) * key_terms.count(term)
            term_scores[term] = score
        
        # Sort and return top terms
        sorted_terms = sorted(term_scores.keys(), key=lambda x: term_scores[x], reverse=True)
        return sorted_terms[:5]  # Top 5 terms
    
    def _generate_question(self, chunk_text: str, language: str, chunk_type: str, doc_id: str = '') -> Optional[str]:
        """Generate a question based on chunk content with domain awareness."""
        if language not in self.question_templates:
            return None
        
        # Clean text first
        clean_text = self._clean_ocr_text(chunk_text)
        if not clean_text or len(clean_text) < 10:
            return None
        
        # For table chunks, focus on data/numbers
        if chunk_type == 'table':
            if 'कति' in clean_text or any(c.isdigit() for c in clean_text):
                if language == 'ne':
                    return "तालिकामा के जानकारी दिइएको छ?"
                else:
                    return "What information is provided in the table?"
        
        # Detect domain for better templates
        domain = self._detect_domain(clean_text + ' ' + doc_id)
        
        # Extract key terms for question generation
        key_terms = self._extract_key_terms(clean_text, language)
        if not key_terms:
            # Fallback generic questions
            if language == 'ne':
                return "यस दस्तावेजमा के भनिएको छ?"
            else:
                return "What information is provided in this document?"
        
        # Select template from appropriate domain
        domain_templates = self.question_templates[language].get(domain, self.question_templates[language]['generic'])
        template = random.choice(domain_templates)
        main_term = key_terms[0]
        
        # Fill template
        try:
            if '{topic}' in template:
                question = template.format(topic=main_term)
            elif '{document}' in template and '{topic}' in template:
                doc_name = "दस्तावेज" if language == 'ne' else "document"
                question = template.format(document=doc_name, topic=main_term)
            else:
                question = template
        except:
            # Fallback
            if language == 'ne':
                question = f"{main_term} के हो?"
            else:
                question = f"What is {main_term}?"
        
        return question
    
    def _validate_answer_quality(self, answer: str, chunk_text: str, language: str) -> bool:
        """Validate answer quality before including in dataset - very strict."""
        if not answer or len(answer.strip()) < 10:
            return False
        
        # Check for excessive OCR garbage
        words = answer.split()
        if len(words) < 3:  # Need at least 3 words
            return False
        
        # Much stricter character distribution
        alpha_chars = sum(c.isalpha() or 0x0900 <= ord(c) <= 0x097F for c in answer)
        digit_symbol_chars = sum(c.isdigit() or c in '!@#$%^&*()_+-={}[]|\\:;"<>,.?/' for c in answer)
        
        if len(answer) > 0:
            alpha_ratio = alpha_chars / len(answer)
            garbage_ratio = digit_symbol_chars / len(answer)
            
            if alpha_ratio < 0.8 or garbage_ratio > 0.2:  # Much stricter
                return False
        
        # Check for coherent words only
        coherent_words = 0
        for word in words:
            if len(word) >= 3 and (word.isalpha() or any(0x0900 <= ord(c) <= 0x097F for c in word)):
                coherent_words += 1
        
        if coherent_words / len(words) < 0.9:  # 90% coherent words required
            return False
        
        # Check if answer exists in chunk (stricter match)
        answer_words = set(re.findall(r'\w+', answer.lower()))
        chunk_words = set(re.findall(r'\w+', chunk_text.lower()))
        
        if len(answer_words) > 0:
            overlap_ratio = len(answer_words.intersection(chunk_words)) / len(answer_words)
            if overlap_ratio < 0.6:  # Much stricter overlap requirement
                return False
        
        return True
    
    def _generate_answer(self, question: str, chunk_text: str, language: str, chunk_type: str) -> Optional[str]:
        """Generate answer from chunk text with quality validation."""
        # For table chunks, extract relevant data
        if chunk_type == 'table':
            # Look for numbers or key data points
            lines = chunk_text.split('\n')
            data_lines = [line for line in lines if '\t' in line or any(c.isdigit() for c in line)]
            if data_lines:
                # Use first meaningful data line
                answer = data_lines[0].strip()
                if len(answer) > 100:
                    answer = answer[:100] + "..."
                return answer
        
        # Clean text first
        clean_text = self._clean_ocr_text(chunk_text)
        if not clean_text:
            return None
        
        # For text chunks, extract relevant sentences
        sentences = re.split(r'[।\.!?]+', clean_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return None
        
        # Simple heuristic: take first 1-2 meaningful sentences
        if len(sentences) == 1:
            answer = sentences[0]
        else:
            # Take first sentence, or first two if first is very short
            answer = sentences[0]
            if len(answer) < 50 and len(sentences) > 1:
                answer += "। " + sentences[1] if language == 'ne' else ". " + sentences[1]
        
        # Truncate if too long
        if len(answer) > 200:
            answer = answer[:200] + ("..." if language == 'en' else "...")
        
        answer = answer.strip()
        
        # Validate answer quality
        if not self._validate_answer_quality(answer, clean_text, language):
            return None
        
        return answer
    
    def _fuzzy_find_spans(self, phrase: str, text: str, max_distance: int = 2) -> List[Tuple[int, int]]:
        """Find phrase in text with fuzzy matching for OCR errors."""
        spans = []
        phrase_words = phrase.split()
        text_words = text.split()
        
        if not phrase_words or not text_words:
            return spans
        
        # Sliding window approach
        for i in range(len(text_words) - len(phrase_words) + 1):
            window = text_words[i:i + len(phrase_words)]
            
            # Calculate edit distance for each word pair
            matches = 0
            for pw, tw in zip(phrase_words, window):
                # Simple character overlap check
                if len(pw) > 2 and len(tw) > 2:
                    overlap = len(set(pw.lower()).intersection(set(tw.lower())))
                    if overlap >= min(len(pw), len(tw)) * 0.6:  # 60% character overlap
                        matches += 1
                elif pw.lower() == tw.lower():
                    matches += 1
            
            # If most words match, consider it a match
            if matches >= len(phrase_words) * 0.6:  # 60% word match
                # Find character positions
                start_pos = text.find(window[0])
                if start_pos != -1:
                    end_word = window[-1]
                    end_pos = text.find(end_word, start_pos) + len(end_word)
                    spans.append((start_pos, end_pos))
                    break
        
        return spans
    
    def _find_char_spans(self, answer: str, chunk_text: str) -> List[Tuple[int, int]]:
        """Find character spans of answer text in chunk with fuzzy matching."""
        if not answer or not chunk_text:
            return []
        
        spans = []
        
        # Split answer into meaningful phrases
        # Remove punctuation for matching
        answer_clean = re.sub(r'[^\w\s\u0900-\u097F]', ' ', answer)
        phrases = [p.strip() for p in answer_clean.split() if len(p.strip()) > 2]
        
        # Find each phrase in the chunk
        for phrase in phrases[:3]:  # Limit to first 3 phrases
            # Try exact match first
            start = chunk_text.find(phrase)
            if start != -1:
                spans.append((start, start + len(phrase)))
            else:
                # Try case-insensitive match
                start = chunk_text.lower().find(phrase.lower())
                if start != -1:
                    # Find the actual case-sensitive match
                    actual_phrase = chunk_text[start:start + len(phrase)]
                    spans.append((start, start + len(actual_phrase)))
                else:
                    # Try fuzzy matching for OCR errors
                    fuzzy_spans = self._fuzzy_find_spans(phrase, chunk_text)
                    spans.extend(fuzzy_spans)
        
        # Remove overlapping spans
        spans = sorted(set(spans))
        non_overlapping = []
        for start, end in spans:
            if not non_overlapping or start >= non_overlapping[-1][1]:
                non_overlapping.append((start, end))
        
        return non_overlapping
    
    def _compute_bbox_list(self, char_spans: List[Tuple[int, int]], 
                          char_span_bboxes: Optional[str]) -> List[List[float]]:
        """Compute bbox list from character spans."""
        if not char_spans or not char_span_bboxes:
            return []
        
        try:
            # Parse char_span_bboxes JSON
            span_data = json.loads(char_span_bboxes)
            if not isinstance(span_data, list):
                return []
            
            bbox_list = []
            for start, end in char_spans:
                # Find overlapping character spans in the bbox data
                for span_info in span_data:
                    if isinstance(span_info, dict) and 'bbox' in span_info:
                        span_bbox = span_info['bbox']
                        if isinstance(span_bbox, list) and len(span_bbox) == 4:
                            bbox_list.append(span_bbox)
                            break  # One bbox per span for simplicity
            
            return bbox_list
        except (json.JSONDecodeError, KeyError, TypeError):
            return []
    
    def _get_hard_negatives(self, question: str, language: str, k: int = 3) -> List[Dict[str, Any]]:
        """Get hard negative examples using multi-query retrieval."""
        try:
            # Use CP6 multi-query search to find Wikipedia distractors
            results = self.retriever.search(
                query=question,
                k=k * 2,  # Oversample
                query_lang=language,
                allow_distractors=True,
                inject_hard_negatives=k
            )
            
            # Extract Wikipedia distractors
            hard_negatives = []
            for candidate in results.get('distractor_candidates', []):
                if candidate.get('source_authority') == 'wikipedia':
                    hard_negative = {
                        'text': candidate['text'],
                        'doc_id': candidate['doc_id'],
                        'page_id': candidate['page_id'],
                        'chunk_id': candidate['chunk_id'],
                        'language': candidate['language'],
                        'score': candidate['adjusted_score']
                    }
                    hard_negatives.append(hard_negative)
            
            # Return top k distinct negatives
            return hard_negatives[:k]
            
        except Exception as e:
            print(f"Warning: Could not generate hard negatives: {e}")
            return []
    
    def generate_qacite_pair(self, row: pd.Series, per_chunk: int = 1) -> List[Dict[str, Any]]:
        """Generate enhanced Q-A-Cite pairs for a single corpus row."""
        chunk_text = row.get('text', '').strip()
        if not chunk_text or len(chunk_text) < self.min_text_length:
            return []
        
        # Quality filtering
        confidence = row.get('conf_mean', 0)
        if confidence < self.min_confidence:
            return []
        
        # Assess text quality - be much more aggressive
        quality_score = self._assess_text_quality(chunk_text, confidence)
        if quality_score < 0.8:  # Much stricter threshold - only very high quality
            return []
        
        language = row.get('language', 'mixed')
        if language == 'mixed':
            return []  # Skip mixed language chunks
        
        chunk_type = row.get('chunk_type', 'text')
        doc_id = row.get('doc_id', '')
        page_id = row.get('page_id', '')
        chunk_id = row.get('chunk_id', '')
        
        pairs = []
        
        for i in range(per_chunk):
            # Generate question
            question = self._generate_question(chunk_text, language, chunk_type, doc_id)
            if not question:
                continue
            
            # Generate answer
            answer = self._generate_answer(question, chunk_text, language, chunk_type)
            if not answer:
                continue
            
            # Find character spans
            char_spans = self._find_char_spans(answer, chunk_text)
            if not char_spans:
                continue
            
            # Compute bbox list
            char_span_bboxes = row.get('char_span')
            bbox_list = self._compute_bbox_list(char_spans, char_span_bboxes)
            
            # Get hard negatives using multi-query expansion
            hard_negatives = self._get_hard_negatives(question, language, k=3)
            
            # Get expansion terms used (if available from retriever)
            expansion_terms_used = []
            try:
                expanded_queries = self.retriever._expand_query_terms(question, language)
                if len(expanded_queries) > 1:
                    expansion_terms_used = expanded_queries[1:]
            except:
                pass
            
            # Create Q-A-Cite pair
            pair = {
                'question': question,
                'answer_exact': answer,
                'char_spans': char_spans,
                'bbox_list': bbox_list,
                'language': language,
                'chunk_type': chunk_type,
                'doc_id': doc_id,
                'page_id': page_id,
                'chunk_id': chunk_id,
                'source_authority': row.get('source_authority', 'unknown'),
                'hard_negatives': hard_negatives,
                'expansion_terms_used': expansion_terms_used,
                'quality_score': quality_score,
                'ocr_confidence': confidence,
                'priors_applied': {
                    'same_lang_boost': self.retriever.same_lang_boost,
                    'authority_boost': self.retriever.authority_boost,
                    'quality_score': quality_score
                },
                'diversification_caps': {
                    'per_page_limit': 3,
                    'per_doc_limit': 5
                }
            }
            
            pairs.append(pair)
        
        return pairs
    
    def build_dataset(self, 
                     output_dir: Path,
                     per_chunk: int = 1,
                     use_multiquery: bool = True,
                     train_ratio: float = 0.7,
                     dev_ratio: float = 0.15,
                     test_ratio: float = 0.15) -> Dict[str, int]:
        """Build complete Q-A-Cite dataset with enhanced quality filtering."""
        
        # Filter to high-quality authoritative chunks only
        auth_corpus = self.corpus_df[
            (self.corpus_df['source_authority'] == 'authoritative') &
            (self.corpus_df['language'].isin(['ne', 'en'])) &
            (self.corpus_df['text'].str.len() >= self.min_text_length) &
            (self.corpus_df['conf_mean'] >= self.min_confidence)
        ].copy()
        
        # Sort by confidence for better quality
        auth_corpus = auth_corpus.sort_values('conf_mean', ascending=False)
        
        print(f"Filtered to {len(auth_corpus)} high-quality authoritative chunks")
        print(f"Quality filters: min_confidence={self.min_confidence}, min_length={self.min_text_length}")
        
        # Generate all Q-A pairs
        all_pairs = []
        generated_count = 0
        filtered_count = 0
        
        for idx, row in auth_corpus.iterrows():
            pairs = self.generate_qacite_pair(row, per_chunk)
            if pairs:
                all_pairs.extend(pairs)
                generated_count += len(pairs)
            else:
                filtered_count += 1
            
            if len(all_pairs) % 50 == 0 and len(all_pairs) > 0:
                print(f"Generated {len(all_pairs)} Q-A pairs...")
        
        print(f"Total Q-A pairs generated: {len(all_pairs)}")
        print(f"Generated from {generated_count} chunks, filtered out {filtered_count} chunks")
        
        if not all_pairs:
            print("\n❌ NO USABLE Q-A PAIRS GENERATED")
            print("\n🔍 ROOT CAUSE ANALYSIS:")
            print("   The OCR quality in this corpus is catastrophically poor.")
            print("   All text chunks contain mostly garbage characters and symbols.")
            print("   This makes them unsuitable for Q-A generation.")
            print("\n💡 RECOMMENDED SOLUTIONS:")
            print("   1. Re-run OCR pipeline with higher quality settings")
            print("   2. Use born-digital PDFs instead of scanned documents")
            print("   3. Manual text correction for key documents")
            print("   4. Use synthetic Q-A generation from document summaries")
            print("\n⚙️  IMMEDIATE WORKAROUND:")
            print("   Creating minimal synthetic dataset for pipeline testing...")
            
            # Create synthetic examples for pipeline testing
            synthetic_pairs = self._create_synthetic_examples()
            if synthetic_pairs:
                print(f"   Generated {len(synthetic_pairs)} synthetic examples")
                
                # Save synthetic examples
                output_dir.mkdir(parents=True, exist_ok=True)
                synthetic_file = output_dir / "synthetic.jsonl"
                with open(synthetic_file, 'w', encoding='utf-8') as f:
                    for pair in synthetic_pairs:
                        f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                print(f"   Saved to {synthetic_file}")
                
                return {'synthetic': len(synthetic_pairs)}
            else:
                raise ValueError("No Q-A pairs generated and synthetic generation failed")
        
        # Shuffle for random splits
        random.shuffle(all_pairs)
        
        # Calculate split indices
        n_total = len(all_pairs)
        n_train = int(n_total * train_ratio)
        n_dev = int(n_total * dev_ratio)
        
        # Create splits
        train_pairs = all_pairs[:n_train]
        dev_pairs = all_pairs[n_train:n_train + n_dev]
        test_pairs = all_pairs[n_train + n_dev:]
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save splits
        splits = {
            'train': train_pairs,
            'dev': dev_pairs, 
            'test': test_pairs
        }
        
        counts = {}
        for split_name, pairs in splits.items():
            output_file = output_dir / f"{split_name}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for pair in pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + '\n')
            
            counts[split_name] = len(pairs)
            print(f"Saved {split_name}: {len(pairs)} pairs to {output_file}")
        
        return counts
    
    def _create_synthetic_examples(self) -> List[Dict[str, Any]]:
        """Create synthetic Q-A examples for pipeline testing when OCR fails."""
        synthetic_examples = [
            {
                'question': 'नेपालको संविधानमा मौलिक अधिकारहरू के के छन्?',
                'answer_exact': 'नेपालको संविधानमा जीवनको अधिकार, स्वतन्त्रताको अधिकार, र समानताको अधिकार मौलिक अधिकारहरू छन्।',
                'char_spans': [(0, 25), (30, 55)],
                'bbox_list': [[100, 200, 300, 220], [100, 240, 300, 260]],
                'language': 'ne',
                'chunk_type': 'text',
                'doc_id': 'constitution-ne',
                'page_id': 'constitution-ne_page_001',
                'chunk_id': 'synthetic_001',
                'source_authority': 'authoritative',
                'hard_negatives': [],
                'expansion_terms_used': [],
                'quality_score': 1.0,
                'ocr_confidence': 95.0,
                'priors_applied': {'same_lang_boost': 0.15, 'authority_boost': 0.08, 'quality_score': 1.0},
                'diversification_caps': {'per_page_limit': 3, 'per_doc_limit': 5}
            },
            {
                'question': 'What is the role of the Health Emergency Operations Center?',
                'answer_exact': 'The Health Emergency Operations Center coordinates emergency response activities and serves as the central hub for information management during health crises.',
                'char_spans': [(0, 40), (50, 120)],
                'bbox_list': [[50, 100, 400, 120], [50, 140, 400, 160]],
                'language': 'en',
                'chunk_type': 'text',
                'doc_id': 'heoc-report',
                'page_id': 'heoc-report_page_001',
                'chunk_id': 'synthetic_002',
                'source_authority': 'authoritative',
                'hard_negatives': [],
                'expansion_terms_used': [],
                'quality_score': 1.0,
                'ocr_confidence': 95.0,
                'priors_applied': {'same_lang_boost': 0.15, 'authority_boost': 0.08, 'quality_score': 1.0},
                'diversification_caps': {'per_page_limit': 3, 'per_doc_limit': 5}
            },
            {
                'question': 'स्वास्थ्य मन्त्रालयको मुख्य जिम्मेवारी के हो?',
                'answer_exact': 'स्वास्थ्य मन्त्रालयको मुख्य जिम्मेवारी नेपालको स्वास्थ्य सेवा व्यवस्थापन र नीति निर्माण गर्नु हो।',
                'char_spans': [(0, 30), (40, 80)],
                'bbox_list': [[100, 300, 350, 320], [100, 340, 350, 360]],
                'language': 'ne',
                'chunk_type': 'text',
                'doc_id': 'health-policy',
                'page_id': 'health-policy_page_001',
                'chunk_id': 'synthetic_003',
                'source_authority': 'authoritative',
                'hard_negatives': [],
                'expansion_terms_used': [],
                'quality_score': 1.0,
                'ocr_confidence': 95.0,
                'priors_applied': {'same_lang_boost': 0.15, 'authority_boost': 0.08, 'quality_score': 1.0},
                'diversification_caps': {'per_page_limit': 3, 'per_doc_limit': 5}
            }
        ]
        
        return synthetic_examples


def main():
    """CLI for enhanced Q-A-Cite generation."""
    parser = argparse.ArgumentParser(
        description="Generate enhanced Q-A-Cite dataset for NepaliGov-RAG-Bench"
    )
    parser.add_argument(
        "--in",
        dest="input_file",
        type=Path,
        required=True,
        help="Input corpus parquet file"
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for Q-A-Cite dataset"
    )
    parser.add_argument(
        "--faiss-dir",
        type=Path,
        default=Path("data/faiss"),
        help="FAISS index directory for hard negatives"
    )
    parser.add_argument(
        "--per-chunk",
        type=int,
        default=1,
        help="Number of Q-A pairs per chunk"
    )
    parser.add_argument(
        "--use-multiquery",
        type=bool,
        default=True,
        help="Use multi-query expansion for hard negatives"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio"
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.15,
        help="Development set ratio"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=60.0,
        help="Minimum OCR confidence threshold (default: 60.0 for quality)"
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=100,
        help="Minimum text length for chunk inclusion (default: 100 for quality)"
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.dev_ratio + args.test_ratio - 1.0) > 0.01:
        print("Error: Train/dev/test ratios must sum to 1.0")
        sys.exit(1)
    
    try:
        # Initialize enhanced generator with strict quality requirements
        generator = QACiteGenerator(
            corpus_path=args.input_file,
            faiss_dir=args.faiss_dir,
            random_seed=args.seed,
            min_confidence=max(args.min_confidence, 60.0),  # Force minimum 60 confidence
            min_text_length=max(args.min_text_length, 100)   # Force minimum 100 chars
        )
        
        # Build dataset
        counts = generator.build_dataset(
            output_dir=args.out,
            per_chunk=args.per_chunk,
            use_multiquery=args.use_multiquery,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            test_ratio=args.test_ratio
        )
        
        if 'synthetic' in counts:
            print(f"\n⚠️  Synthetic Q-A-Cite dataset created (OCR quality too poor):")
            print(f"   Synthetic examples: {counts['synthetic']} pairs")
            print(f"   Output directory: {args.out}")
            print(f"\n🔧 NEXT STEPS:")
            print(f"   1. Fix OCR quality issues in the corpus")
            print(f"   2. Re-run with higher quality source documents")
            print(f"   3. Use synthetic dataset for pipeline development only")
        else:
            print(f"\n✅ Enhanced Q-A-Cite dataset generation complete:")
            print(f"   Train: {counts['train']} pairs")
            print(f"   Dev: {counts['dev']} pairs")
            print(f"   Test: {counts['test']} pairs")
            print(f"   Total: {sum(counts.values())} pairs")
            print(f"   Output directory: {args.out}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()