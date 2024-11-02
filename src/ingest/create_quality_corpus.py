#!/usr/bin/env python3
"""
Quality Corpus Generator for NepaliGov-RAG-Bench

Creates a high-quality synthetic corpus to replace poor OCR data,
enabling proper Q-A-Cite generation and system development.
"""

import argparse
import json
import uuid
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime


class QualityCorpusGenerator:
    """Generates high-quality synthetic corpus data for system development."""
    
    def __init__(self):
        """Initialize the quality corpus generator."""
        self.generated_chunks = []
        
    def create_constitution_chunks(self) -> List[Dict[str, Any]]:
        """Create high-quality chunks from Nepal Constitution content."""
        chunks = [
            {
                'doc_id': 'constitution-ne-2072',
                'page_id': 'constitution-ne-2072_page_001',
                'block_id': 'block_001',
                'block_type': 'text',
                'text': 'नेपालको संविधान २०७२ ले नेपालीहरूको मौलिक अधिकारहरूको ग्यारेन्टी दिन्छ। यसमा जीवनको अधिकार, स्वतन्त्रताको अधिकार, समानताको अधिकार, गोपनीयताको अधिकार, धार्मिक स्वतन्त्रताको अधिकार, सूचनाको अधिकार, र शिक्षाको अधिकार समावेश छन्। यी अधिकारहरू सबै नेपाली नागरिकहरूका लागि समान रूपमा लागू हुन्छन्।',
                'language': 'ne',
                'char_span': '[{"text": "मौलिक अधिकारहरूको ग्यारेन्टी", "bbox": [100, 200, 400, 220]}, {"text": "जीवनको अधिकार", "bbox": [100, 240, 250, 260]}]',
                'bbox': [100, 200, 500, 300],
                'ocr_engine': 'synthetic',
                'conf_mean': 95.0,
                'conf_min': 95.0,
                'conf_max': 95.0,
                'tokens': 45,
                'source_authority': 'authoritative',
                'is_distractor': False,
                'source_page_is_ocr': False,
                'watermark_flags': [],
                'font_stats': '{"primary_font": "Devanagari", "font_size_mean": 12}',
                'pdf_meta': '{"title": "Constitution of Nepal 2072", "language_hint": "ne"}'
            },
            {
                'doc_id': 'constitution-en-2015',
                'page_id': 'constitution-en-2015_page_001',
                'block_id': 'block_001',
                'block_type': 'text',
                'text': 'The Constitution of Nepal 2015 guarantees fundamental rights to all Nepali citizens. These include the right to life, right to liberty, right to equality, right to privacy, right to religious freedom, right to information, and right to education. These rights are universally applicable to all citizens without discrimination based on caste, ethnicity, gender, religion, or economic status.',
                'language': 'en',
                'char_span': '[{"text": "fundamental rights", "bbox": [150, 200, 300, 220]}, {"text": "right to life", "bbox": [150, 240, 250, 260]}]',
                'bbox': [150, 200, 550, 300],
                'ocr_engine': 'synthetic',
                'conf_mean': 95.0,
                'conf_min': 95.0,
                'conf_max': 95.0,
                'tokens': 52,
                'source_authority': 'authoritative',
                'is_distractor': False,
                'source_page_is_ocr': False,
                'watermark_flags': [],
                'font_stats': '{"primary_font": "Arial", "font_size_mean": 12}',
                'pdf_meta': '{"title": "Constitution of Nepal 2015", "language_hint": "en"}'
            }
        ]
        return chunks
    
    def create_health_chunks(self) -> List[Dict[str, Any]]:
        """Create high-quality chunks from health policy content."""
        chunks = [
            {
                'doc_id': 'health-policy-ne-2076',
                'page_id': 'health-policy-ne-2076_page_001',
                'block_id': 'block_001',
                'block_type': 'text',
                'text': 'स्वास्थ्य तथा जनसंख्या मन्त्रालयले नेपालको स्वास्थ्य सेवा व्यवस्थापन र नीति निर्माणको जिम्मेवारी लिएको छ। मन्त्रालयका मुख्य उद्देश्यहरूमा गुणस्तरीय स्वास्थ्य सेवा प्रदान गर्नु, रोग नियन्त्रण कार्यक्रम सञ्चालन गर्नु, र स्वास्थ्य शिक्षाको प्रवर्द्धन गर्नु समावेश छ। मन्त्रालयले विशेष गरी मातृत्व स्वास्थ्य, बाल स्वास्थ्य, र संक्रामक रोग नियन्त्रणमा जोड दिएको छ।',
                'language': 'ne',
                'char_span': '[{"text": "स्वास्थ्य तथा जनसंख्या मन्त्रालय", "bbox": [100, 200, 350, 220]}, {"text": "गुणस्तरीय स्वास्थ्य सेवा", "bbox": [100, 240, 280, 260]}]',
                'bbox': [100, 200, 550, 320],
                'ocr_engine': 'synthetic',
                'conf_mean': 95.0,
                'conf_min': 95.0,
                'conf_max': 95.0,
                'tokens': 48,
                'source_authority': 'authoritative',
                'is_distractor': False,
                'source_page_is_ocr': False,
                'watermark_flags': [],
                'font_stats': '{"primary_font": "Devanagari", "font_size_mean": 12}',
                'pdf_meta': '{"title": "Health Policy Nepal 2076", "language_hint": "ne"}'
            },
            {
                'doc_id': 'heoc-report-en-2020',
                'page_id': 'heoc-report-en-2020_page_001',
                'block_id': 'block_001',
                'block_type': 'text',
                'text': 'The Health Emergency Operations Center (HEOC) serves as the central coordination hub for emergency health responses in Nepal. Established under the Ministry of Health and Population, HEOC coordinates with various stakeholders including WHO, UNICEF, and international partners during health emergencies. The center played a crucial role during the COVID-19 pandemic by managing information flow, coordinating response activities, and facilitating resource allocation to affected areas.',
                'language': 'en',
                'char_span': '[{"text": "Health Emergency Operations Center", "bbox": [150, 200, 400, 220]}, {"text": "central coordination hub", "bbox": [150, 240, 320, 260]}]',
                'bbox': [150, 200, 600, 320],
                'ocr_engine': 'synthetic',
                'conf_mean': 95.0,
                'conf_min': 95.0,
                'conf_max': 95.0,
                'tokens': 58,
                'source_authority': 'authoritative',
                'is_distractor': False,
                'source_page_is_ocr': False,
                'watermark_flags': [],
                'font_stats': '{"primary_font": "Arial", "font_size_mean": 12}',
                'pdf_meta': '{"title": "HEOC Report 2020", "language_hint": "en"}'
            }
        ]
        return chunks
    
    def create_covid_chunks(self) -> List[Dict[str, Any]]:
        """Create high-quality chunks from COVID-19 response content."""
        chunks = [
            {
                'doc_id': 'covid-response-ne-2077',
                'page_id': 'covid-response-ne-2077_page_001',
                'block_id': 'block_001',
                'block_type': 'text',
                'text': 'नेपालमा कोभिड-१९ महामारीको प्रतिक्रियास्वरूप सरकारले राष्ट्रिय आपतकालीन कार्य योजना तयार गर्यो। यस योजनामा स्वास्थ्य सेवा क्षमता विस्तार, परीक्षण सुविधा वृद्धि, र जनस्वास्थ्य सुरक्षा उपायहरू समावेश थिए। सरकारले लकडाउन, सामाजिक दूरी, र मास्क प्रयोगलाई अनिवार्य बनायो। खोप कार्यक्रम सुरु गरी जनतालाई निःशुल्क खोप उपलब्ध गराइयो।',
                'language': 'ne',
                'char_span': '[{"text": "कोभिड-१९ महामारी", "bbox": [100, 200, 250, 220]}, {"text": "राष्ट्रिय आपतकालीन कार्य योजना", "bbox": [100, 240, 350, 260]}]',
                'bbox': [100, 200, 550, 320],
                'ocr_engine': 'synthetic',
                'conf_mean': 95.0,
                'conf_min': 95.0,
                'conf_max': 95.0,
                'tokens': 44,
                'source_authority': 'authoritative',
                'is_distractor': False,
                'source_page_is_ocr': False,
                'watermark_flags': [],
                'font_stats': '{"primary_font": "Devanagari", "font_size_mean": 12}',
                'pdf_meta': '{"title": "COVID-19 Response Nepal 2077", "language_hint": "ne"}'
            },
            {
                'doc_id': 'covid-vaccination-en-2021',
                'page_id': 'covid-vaccination-en-2021_page_001',
                'block_id': 'block_001',
                'block_type': 'table',
                'text': '[TABLE: COVID-19 Vaccination Statistics]\nDistrict\tFirst Dose\tSecond Dose\tBooster\nKathmandu\t850000\t780000\t450000\nPokhara\t320000\t295000\t180000\nBiratnagar\t280000\t260000\t155000\nBhairahawa\t195000\t175000\t95000',
                'language': 'en',
                'char_span': '[{"text": "Kathmandu\\t850000", "bbox": [150, 240, 300, 260]}, {"text": "Pokhara\\t320000", "bbox": [150, 280, 280, 300]}]',
                'bbox': [150, 200, 500, 350],
                'ocr_engine': 'synthetic',
                'conf_mean': 95.0,
                'conf_min': 95.0,
                'conf_max': 95.0,
                'tokens': 32,
                'source_authority': 'authoritative',
                'is_distractor': False,
                'source_page_is_ocr': False,
                'watermark_flags': [],
                'font_stats': '{"primary_font": "Arial", "font_size_mean": 10}',
                'pdf_meta': '{"title": "COVID-19 Vaccination Nepal 2021", "language_hint": "en"}'
            }
        ]
        return chunks
    
    def create_wikipedia_distractors(self) -> List[Dict[str, Any]]:
        """Create Wikipedia distractor chunks for hard negatives."""
        chunks = [
            {
                'doc_id': 'nepal-wiki-en',
                'page_id': 'nepal-wiki-en_page_001',
                'block_id': 'block_001',
                'block_type': 'text',
                'text': 'Nepal is a landlocked country in South Asia, located mainly in the Himalayas. It borders China to the north and India to the south, east, and west. The capital and largest city is Kathmandu. Nepal has a diverse geography, including the Himalayas in the north, hills in the middle, and the Terai plains in the south. Mount Everest, the world\'s highest peak, is located on the Nepal-China border.',
                'language': 'en',
                'char_span': '[{"text": "landlocked country", "bbox": [150, 200, 280, 220]}, {"text": "Mount Everest", "bbox": [150, 280, 250, 300]}]',
                'bbox': [150, 200, 550, 320],
                'ocr_engine': 'synthetic',
                'conf_mean': 95.0,
                'conf_min': 95.0,
                'conf_max': 95.0,
                'tokens': 56,
                'source_authority': 'wikipedia',
                'is_distractor': True,
                'source_page_is_ocr': False,
                'watermark_flags': [],
                'font_stats': '{"primary_font": "Arial", "font_size_mean": 12}',
                'pdf_meta': '{"title": "Nepal - Wikipedia", "language_hint": "en"}'
            },
            {
                'doc_id': 'health-wiki-ne',
                'page_id': 'health-wiki-ne_page_001',
                'block_id': 'block_001',
                'block_type': 'text',
                'text': 'स्वास्थ्य भनेको शारीरिक, मानसिक र सामाजिक कल्याणको पूर्ण अवस्था हो। यो केवल रोग वा दुर्बलताको अभाव मात्र होइन। विश्व स्वास्थ्य संगठनको परिभाषा अनुसार स्वास्थ्यलाई व्यापक रूपमा बुझिन्छ। स्वस्थ जीवनशैली, नियमित व्यायाम, सन्तुलित आहार, र तनाव व्यवस्थापनले राम्रो स्वास्थ्य कायम राख्न मद्दत गर्छ।',
                'language': 'ne',
                'char_span': '[{"text": "शारीरिक, मानसिक र सामाजिक कल्याण", "bbox": [100, 200, 400, 220]}, {"text": "विश्व स्वास्थ्य संगठन", "bbox": [100, 240, 280, 260]}]',
                'bbox': [100, 200, 550, 320],
                'ocr_engine': 'synthetic',
                'conf_mean': 95.0,
                'conf_min': 95.0,
                'conf_max': 95.0,
                'tokens': 42,
                'source_authority': 'wikipedia',
                'is_distractor': True,
                'source_page_is_ocr': False,
                'watermark_flags': [],
                'font_stats': '{"primary_font": "Devanagari", "font_size_mean": 12}',
                'pdf_meta': '{"title": "स्वास्थ्य - Wikipedia", "language_hint": "ne"}'
            }
        ]
        return chunks
    
    def generate_quality_corpus(self) -> pd.DataFrame:
        """Generate complete high-quality corpus."""
        all_chunks = []
        
        # Add authoritative content
        all_chunks.extend(self.create_constitution_chunks())
        all_chunks.extend(self.create_health_chunks())
        all_chunks.extend(self.create_covid_chunks())
        
        # Add Wikipedia distractors
        all_chunks.extend(self.create_wikipedia_distractors())
        
        # Add unique chunk IDs
        for i, chunk in enumerate(all_chunks):
            chunk['chunk_id'] = f"quality_chunk_{i+1:03d}"
        
        return pd.DataFrame(all_chunks)
    
    def save_corpus(self, output_path: Path) -> None:
        """Save the quality corpus to parquet file."""
        corpus_df = self.generate_quality_corpus()
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        corpus_df.to_parquet(output_path, index=False)
        
        print(f"✅ Quality corpus saved: {len(corpus_df)} chunks")
        print(f"   Authoritative chunks: {len(corpus_df[corpus_df['source_authority'] == 'authoritative'])}")
        print(f"   Wikipedia distractors: {len(corpus_df[corpus_df['source_authority'] == 'wikipedia'])}")
        print(f"   Languages: {corpus_df['language'].value_counts().to_dict()}")
        print(f"   Output: {output_path}")


def main():
    """CLI for quality corpus generation."""
    parser = argparse.ArgumentParser(
        description="Generate high-quality synthetic corpus for NepaliGov-RAG-Bench"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/quality_corpus.parquet"),
        help="Output corpus parquet file"
    )
    
    args = parser.parse_args()
    
    try:
        generator = QualityCorpusGenerator()
        generator.save_corpus(args.out)
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Run chunking: python -m src.retriever.chunk_and_embed --in {args.out}")
        print(f"   2. Generate Q-A: python -m src.ingest.build_qacite --in {args.out}")
        print(f"   3. Test retrieval: python -m src.retriever.search --q 'मौलिक अधिकार के हो?'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



