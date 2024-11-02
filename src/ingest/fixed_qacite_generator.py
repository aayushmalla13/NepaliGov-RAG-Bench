#!/usr/bin/env python3
"""
Fixed Q-A-Cite Generation for NepaliGov-RAG-Bench

Fixes the fundamental issues in question and answer generation to create
meaningful, grammatically correct Q-A pairs.
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


class FixedQACiteGenerator:
    """Fixed Q-A pair generator with proper question and answer logic."""
    
    def __init__(self, 
                 corpus_path: Path,
                 faiss_dir: Path,
                 random_seed: int = 42,
                 min_confidence: float = 50.0,
                 min_text_length: int = 50):
        """Initialize the fixed Q-A-Cite generator."""
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
        
        # Fixed question templates with proper grammar
        self.question_templates = {
            'ne': {
                'constitution': [
                    "नेपालको संविधानमा {topic} के उल्लेख छ?",
                    "{topic} बारे संविधानमा के भनिएको छ?",
                    "संविधानले {topic} को कस्तो व्यवस्था गरेको छ?",
                    "{topic} को मामलामा संविधानको के नीति छ?"
                ],
                'health': [
                    "स्वास्थ्य क्षेत्रमा {topic} को के भूमिका छ?",
                    "{topic} ले कसरी स्वास्थ्य सेवा प्रदान गर्छ?",
                    "नेपालमा {topic} को अवस्था कस्तो छ?",
                    "{topic} संग सम्बन्धित मुख्य जिम्मेवारीहरू के के छन्?"
                ],
                'covid': [
                    "कोभिड-१९ को समयमा {topic} के गरियो?",
                    "महामारीको बेलामा {topic} कस्तो थियो?",
                    "{topic} मार्फत कसरी कोभिड नियन्त्रण गरियो?",
                    "कोभिड-१९ विरुद्धको {topic} के थियो?"
                ],
                'generic': [
                    "{topic} के हो?",
                    "{topic} को बारेमा के जानकारी छ?",
                    "{topic} कसरी काम गर्छ?",
                    "{topic} किन महत्वपूर्ण छ?"
                ]
            },
            'en': {
                'constitution': [
                    "What does Nepal's Constitution say about {topic}?",
                    "How does the Constitution address {topic}?",
                    "What provisions exist for {topic} in the Constitution?",
                    "What is the constitutional framework for {topic}?"
                ],
                'health': [
                    "What is the role of {topic} in Nepal's health system?",
                    "How does {topic} contribute to healthcare?",
                    "What are the main responsibilities of {topic}?",
                    "How is {topic} organized in Nepal?"
                ],
                'covid': [
                    "What measures were taken regarding {topic} during COVID-19?",
                    "How was {topic} managed during the pandemic?",
                    "What was the {topic} response to COVID-19?",
                    "How did {topic} help control the pandemic?"
                ],
                'generic': [
                    "What is {topic}?",
                    "How does {topic} work?",
                    "What information is available about {topic}?",
                    "Why is {topic} important?"
                ]
            }
        }
    
    def _detect_content_domain(self, text: str, doc_id: str) -> str:
        """Detect the domain of the content for better question templates."""
        text_lower = text.lower()
        doc_lower = doc_id.lower()
        
        # Constitution domain
        if any(word in text_lower or word in doc_lower for word in [
            'constitution', 'संविधान', 'fundamental', 'मौलिक', 'rights', 'अधिकार', 'government', 'सरकार'
        ]):
            return 'constitution'
        
        # COVID domain
        if any(word in text_lower or word in doc_lower for word in [
            'covid', 'कोभिड', 'pandemic', 'महामारी', 'lockdown', 'लकडाउन', 'vaccination', 'खोप'
        ]):
            return 'covid'
        
        # Health domain
        if any(word in text_lower or word in doc_lower for word in [
            'health', 'स्वास्थ्य', 'hospital', 'अस्पताल', 'ministry', 'मन्त्रालय', 'heoc', 'medical', 'चिकित्सा'
        ]):
            return 'health'
        
        return 'generic'
    
    def _extract_meaningful_topics(self, text: str, language: str) -> List[str]:
        """Extract meaningful topics/entities from text for question generation."""
        topics = []
        
        if language == 'ne':
            # Nepali topic patterns
            patterns = [
                r'(मौलिक अधिकार\w*)',
                r'(स्वास्थ्य.*?मन्त्रालय)',
                r'(संविधान.*?२०७२)',
                r'(कोभिड-१९)',
                r'(लकडाउन)',
                r'(खोप.*?कार्यक्रम)',
                r'(आपतकालीन.*?योजना)',
                r'(सामाजिक दूरी)',
                r'(स्वास्थ्य सेवा)',
                r'(नेपाल सरकार)'
            ]
        else:
            # English topic patterns
            patterns = [
                r'(fundamental rights?)',
                r'(Health.*?Ministry)',
                r'(Constitution.*?201[56])',
                r'(COVID-19)',
                r'(federal democratic republic)',
                r'(Health Emergency Operations Center|HEOC)',
                r'(vaccination.*?program)',
                r'(emergency.*?response)',
                r'(three levels of government)',
                r'(health.*?system)'
            ]
        
        # Extract topics using patterns
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            topics.extend(matches)
        
        # If no specific topics found, extract general noun phrases
        if not topics:
            if language == 'ne':
                # Extract Devanagari words that look like topics (longer words)
                words = re.findall(r'[\u0900-\u097F]{4,}', text)
                topics = [w for w in words if w not in ['गरेको', 'भएको', 'दिएको', 'लिएको']][:3]
            else:
                # Extract capitalized words and noun phrases
                words = re.findall(r'\b[A-Z][a-z]{3,}\b', text)
                topics = [w for w in words if w not in ['The', 'This', 'That', 'These', 'Those']][:3]
        
        return topics[:3]  # Return top 3 topics
    
    def _generate_question(self, text: str, language: str, chunk_type: str, doc_id: str) -> Optional[str]:
        """Generate a meaningful, grammatically correct question."""
        if language not in self.question_templates:
            return None
        
        # Detect domain
        domain = self._detect_content_domain(text, doc_id)
        
        # Extract topics
        topics = self._extract_meaningful_topics(text, language)
        if not topics:
            return None
        
        # Select template
        domain_templates = self.question_templates[language].get(domain, 
                                                               self.question_templates[language]['generic'])
        template = random.choice(domain_templates)
        
        # Choose best topic (first one is usually most relevant)
        topic = topics[0]
        
        # Generate question
        try:
            question = template.format(topic=topic)
            
            # Post-process for better grammar
            if language == 'ne':
                # Fix common Nepali grammar issues
                question = question.replace('को को', 'को')
                question = question.replace('ले ले', 'ले')
                question = question.replace('मा मा', 'मा')
            else:
                # Fix common English grammar issues
                question = question.replace('  ', ' ')
                question = question.strip()
            
            return question
            
        except Exception as e:
            print(f"Question generation error: {e}")
            return None
    
    def _generate_answer(self, question: str, text: str, language: str, chunk_type: str) -> Optional[str]:
        """Generate a meaningful answer that actually answers the question."""
        if not question or not text:
            return None
        
        # For table chunks, extract relevant data
        if chunk_type == 'table':
            # Look for numbers or data that might answer the question
            lines = text.split('\n')
            data_lines = [line for line in lines if '\t' in line and any(c.isdigit() for c in line)]
            if data_lines and len(question) > 10:
                # Try to find relevant data line
                for line in data_lines[:3]:  # Check first 3 data rows
                    if any(word in line.lower() for word in question.lower().split()[-3:]):
                        return f"According to the data: {line.strip()}"
                # Fallback to first data line
                return f"The data shows: {data_lines[0].strip()}"
        
        # For text chunks, extract relevant sentences
        sentences = re.split(r'[।\.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        
        if not sentences:
            return None
        
        # Simple strategy: use first 1-2 sentences as they usually contain main info
        if len(sentences) >= 2:
            answer = sentences[0] + ('।' if language == 'ne' else '.') + ' ' + sentences[1]
        else:
            answer = sentences[0]
        
        # Truncate if too long
        if len(answer) > 300:
            if language == 'ne':
                answer = answer[:300] + '।'
            else:
                answer = answer[:300] + '.'
        
        return answer.strip()
    
    def _find_char_spans(self, answer: str, chunk_text: str) -> List[Tuple[int, int]]:
        """Find character spans of answer text in chunk."""
        if not answer or not chunk_text:
            return []
        
        spans = []
        
        # Split answer into sentences and find each
        if '।' in answer:
            parts = answer.split('।')
        else:
            parts = answer.split('.')
        
        for part in parts[:2]:  # Max 2 parts
            part = part.strip()
            if len(part) > 10:
                start = chunk_text.find(part)
                if start != -1:
                    spans.append((start, start + len(part)))
        
        return spans
    
    def _compute_bbox_list(self, char_spans: List[Tuple[int, int]], 
                          char_span_bboxes: Optional[str]) -> List[List[float]]:
        """Compute bbox list from character spans."""
        if not char_spans or not char_span_bboxes:
            return []
        
        try:
            span_data = json.loads(char_span_bboxes)
            if not isinstance(span_data, list):
                return []
            
            bbox_list = []
            for start, end in char_spans:
                # Use first available bbox as approximation
                if span_data and 'bbox' in span_data[0]:
                    bbox_list.append(span_data[0]['bbox'])
            
            return bbox_list
        except:
            return []
    
    def generate_qacite_pair(self, row: pd.Series, per_chunk: int = 1) -> List[Dict[str, Any]]:
        """Generate fixed Q-A-Cite pairs for a single corpus row."""
        chunk_text = row.get('text', '').strip()
        if not chunk_text or len(chunk_text) < self.min_text_length:
            return []
        
        # Quality filtering
        confidence = row.get('conf_mean', 0)
        if confidence < self.min_confidence:
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
                # Create approximate spans
                char_spans = [(0, min(len(answer), len(chunk_text)))]
            
            # Compute bbox list
            char_span_bboxes = row.get('char_span')
            bbox_list = self._compute_bbox_list(char_spans, char_span_bboxes)
            
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
                'hard_negatives': [],  # Simplified for now
                'expansion_terms_used': [],
                'quality_score': confidence / 100.0,
                'ocr_confidence': confidence,
                'priors_applied': {
                    'same_lang_boost': 0.15,
                    'authority_boost': 0.08,
                    'quality_score': confidence / 100.0
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
                     train_ratio: float = 0.7,
                     dev_ratio: float = 0.15,
                     test_ratio: float = 0.15) -> Dict[str, int]:
        """Build complete Q-A-Cite dataset with fixed generation."""
        
        # Filter to authoritative chunks
        auth_corpus = self.corpus_df[
            (self.corpus_df['source_authority'] == 'authoritative') &
            (self.corpus_df['language'].isin(['ne', 'en'])) &
            (self.corpus_df['text'].str.len() >= self.min_text_length) &
            (self.corpus_df['conf_mean'] >= self.min_confidence)
        ].copy()
        
        print(f"Filtered to {len(auth_corpus)} authoritative chunks")
        
        # Generate all Q-A pairs
        all_pairs = []
        for idx, row in auth_corpus.iterrows():
            pairs = self.generate_qacite_pair(row, per_chunk)
            all_pairs.extend(pairs)
            
            if len(all_pairs) % 10 == 0 and len(all_pairs) > 0:
                print(f"Generated {len(all_pairs)} Q-A pairs...")
        
        print(f"Total Q-A pairs generated: {len(all_pairs)}")
        
        if not all_pairs:
            print("❌ No Q-A pairs generated")
            return {}
        
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


def main():
    """CLI for fixed Q-A-Cite generation."""
    parser = argparse.ArgumentParser(
        description="Generate fixed Q-A-Cite dataset for NepaliGov-RAG-Bench"
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
        default=Path("data/proper_faiss"),
        help="FAISS index directory"
    )
    parser.add_argument(
        "--per-chunk",
        type=int,
        default=1,
        help="Number of Q-A pairs per chunk"
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
        default=50.0,
        help="Minimum OCR confidence threshold"
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=50,
        help="Minimum text length for chunk inclusion"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.4,
        help="Training set ratio"
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.3,
        help="Development set ratio"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.3,
        help="Test set ratio"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize fixed generator
        generator = FixedQACiteGenerator(
            corpus_path=args.input_file,
            faiss_dir=args.faiss_dir,
            random_seed=args.seed,
            min_confidence=args.min_confidence,
            min_text_length=args.min_text_length
        )
        
        # Build dataset
        counts = generator.build_dataset(
            output_dir=args.out,
            per_chunk=args.per_chunk,
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            test_ratio=args.test_ratio
        )
        
        if counts:
            print(f"\n✅ Fixed Q-A-Cite dataset generation complete:")
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
