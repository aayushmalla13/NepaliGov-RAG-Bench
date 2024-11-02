#!/usr/bin/env python3
"""
Proper Quality Corpus Generator for NepaliGov-RAG-Bench

Creates meaningful, well-structured content that makes sense in both languages
and generates coherent Q-A pairs.
"""

import argparse
import json
import uuid
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime


class ProperCorpusGenerator:
    """Generates meaningful, coherent corpus data for proper Q-A generation."""
    
    def __init__(self):
        """Initialize the proper corpus generator."""
        self.generated_chunks = []
        
    def create_constitution_chunks(self) -> List[Dict[str, Any]]:
        """Create meaningful chunks from Nepal Constitution content."""
        chunks = [
            {
                'doc_id': 'constitution-ne-2072',
                'page_id': 'constitution-ne-2072_page_001',
                'block_id': 'block_001',
                'block_type': 'text',
                'text': 'नेपालको संविधान २०७२ ले सबै नेपाली नागरिकहरूलाई मौलिक अधिकारको ग्यारेन्टी दिएको छ। यी मौलिक अधिकारहरूमा जीवनको अधिकार, स्वतन्त्रताको अधिकार, समानताको अधिकार, गोपनीयताको अधिकार, धार्मिक स्वतन्त्रताको अधिकार, सूचनाको अधिकार र शिक्षाको अधिकार पर्दछन्। यी अधिकारहरू जाति, धर्म, लिङ्ग, वा आर्थिक स्थितिको आधारमा कुनै भेदभाव बिना सबै नागरिकहरूमा समान रूपमा लागू हुन्छन्।',
                'language': 'ne',
                'char_span': '[{"text": "मौलिक अधिकार", "bbox": [100, 200, 200, 220]}, {"text": "जीवनको अधिकार", "bbox": [100, 240, 220, 260]}]',
                'bbox': [100, 200, 500, 300],
                'ocr_engine': 'synthetic',
                'conf_mean': 98.0,
                'conf_min': 98.0,
                'conf_max': 98.0,
                'tokens': 58,
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
                'text': 'The Constitution of Nepal 2015 establishes Nepal as a federal democratic republic. It guarantees fundamental rights to all citizens including the right to equality, freedom of speech and expression, right to information, and right to constitutional remedies. The Constitution also establishes three levels of government: federal, provincial, and local. It recognizes Nepal as a multi-ethnic, multi-lingual, multi-religious, and multi-cultural nation.',
                'language': 'en',
                'char_span': '[{"text": "fundamental rights", "bbox": [150, 200, 300, 220]}, {"text": "federal democratic republic", "bbox": [150, 240, 350, 260]}]',
                'bbox': [150, 200, 550, 300],
                'ocr_engine': 'synthetic',
                'conf_mean': 98.0,
                'conf_min': 98.0,
                'conf_max': 98.0,
                'tokens': 62,
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
        """Create meaningful chunks from health policy content."""
        chunks = [
            {
                'doc_id': 'health-ministry-ne-2076',
                'page_id': 'health-ministry-ne-2076_page_001',
                'block_id': 'block_001',
                'block_type': 'text',
                'text': 'स्वास्थ्य तथा जनसंख्या मन्त्रालय नेपाल सरकारको एक महत्वपूर्ण मन्त्रालय हो। यसको मुख्य जिम्मेवारी देशभरका नागरिकहरूलाई गुणस्तरीय स्वास्थ्य सेवा प्रदान गर्नु हो। मन्त्रालयले स्वास्थ्य नीति निर्माण, अस्पताल व्यवस्थापन, रोग नियन्त्रण कार्यक्रम, र स्वास्थ्य शिक्षाको काम गर्दछ। यसले विशेष गरी मातृत्व स्वास्थ्य, बाल स्वास्थ्य, र संक्रामक रोगहरूको रोकथाम र उपचारमा जोड दिन्छ।',
                'language': 'ne',
                'char_span': '[{"text": "स्वास्थ्य तथा जनसंख्या मन्त्रालय", "bbox": [100, 200, 350, 220]}, {"text": "गुणस्तरीय स्वास्थ्य सेवा", "bbox": [100, 240, 280, 260]}]',
                'bbox': [100, 200, 550, 320],
                'ocr_engine': 'synthetic',
                'conf_mean': 98.0,
                'conf_min': 98.0,
                'conf_max': 98.0,
                'tokens': 52,
                'source_authority': 'authoritative',
                'is_distractor': False,
                'source_page_is_ocr': False,
                'watermark_flags': [],
                'font_stats': '{"primary_font": "Devanagari", "font_size_mean": 12}',
                'pdf_meta': '{"title": "Health Ministry Nepal 2076", "language_hint": "ne"}'
            },
            {
                'doc_id': 'heoc-report-en-2020',
                'page_id': 'heoc-report-en-2020_page_001',
                'block_id': 'block_001',
                'block_type': 'text',
                'text': 'The Health Emergency Operations Center (HEOC) is Nepal\'s central coordination hub for health emergency responses. Established under the Ministry of Health and Population, HEOC coordinates emergency preparedness, response activities, and recovery efforts during health crises. The center works closely with WHO, UNICEF, and other international partners. During the COVID-19 pandemic, HEOC played a crucial role in coordinating the national response, managing information systems, and facilitating resource distribution across the country.',
                'language': 'en',
                'char_span': '[{"text": "Health Emergency Operations Center", "bbox": [150, 200, 400, 220]}, {"text": "central coordination hub", "bbox": [150, 240, 320, 260]}]',
                'bbox': [150, 200, 600, 320],
                'ocr_engine': 'synthetic',
                'conf_mean': 98.0,
                'conf_min': 98.0,
                'conf_max': 98.0,
                'tokens': 68,
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
        """Create meaningful chunks from COVID-19 response content."""
        chunks = [
            {
                'doc_id': 'covid-response-ne-2077',
                'page_id': 'covid-response-ne-2077_page_001',
                'block_id': 'block_001',
                'block_type': 'text',
                'text': 'नेपालमा कोभिड-१९ महामारीको समयमा सरकारले व्यापक स्वास्थ्य सुरक्षा उपायहरू लागू गर्यो। यसमा देशव्यापी लकडाउन, सामाजिक दूरी कायम राख्ने नीति, र अनिवार्य मास्क प्रयोग समावेश थियो। सरकारले स्वास्थ्य सेवाको क्षमता बढाउन अस्पतालहरूमा आइसोलेसन वार्ड र आईसीयू बेडहरूको संख्या वृद्धि गर्यो। साथै, निःशुल्क परीक्षण र उपचारको व्यवस्था गरियो।',
                'language': 'ne',
                'char_span': '[{"text": "कोभिड-१९ महामारी", "bbox": [100, 200, 250, 220]}, {"text": "स्वास्थ्य सुरक्षा उपायहरू", "bbox": [100, 240, 300, 260]}]',
                'bbox': [100, 200, 550, 320],
                'ocr_engine': 'synthetic',
                'conf_mean': 98.0,
                'conf_min': 98.0,
                'conf_max': 98.0,
                'tokens': 48,
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
                'text': '[TABLE: COVID-19 Vaccination Progress by District]\nDistrict\tPopulation\tFirst Dose\tSecond Dose\tBooster Dose\nKathmandu\t1200000\t950000\t820000\t450000\nPokhara\t450000\t380000\t320000\t180000\nBiratnagar\t350000\t290000\t250000\t140000\nBhairahawa\t280000\t230000\t195000\t95000',
                'language': 'en',
                'char_span': '[{"text": "Kathmandu\\t1200000\\t950000", "bbox": [150, 240, 350, 260]}, {"text": "Pokhara\\t450000\\t380000", "bbox": [150, 280, 320, 300]}]',
                'bbox': [150, 200, 500, 350],
                'ocr_engine': 'synthetic',
                'conf_mean': 98.0,
                'conf_min': 98.0,
                'conf_max': 98.0,
                'tokens': 42,
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
        """Create meaningful Wikipedia distractor chunks."""
        chunks = [
            {
                'doc_id': 'nepal-geography-wiki-en',
                'page_id': 'nepal-geography-wiki-en_page_001',
                'block_id': 'block_001',
                'block_type': 'text',
                'text': 'Nepal is a landlocked country located in South Asia, bordered by China to the north and India to the south, east, and west. The country has diverse geography ranging from the Terai plains in the south to the Himalayan peaks in the north. Mount Everest, the world\'s highest mountain at 8,848 meters, lies on the Nepal-China border. Nepal covers an area of 147,516 square kilometers and has a population of approximately 30 million people.',
                'language': 'en',
                'char_span': '[{"text": "landlocked country", "bbox": [150, 200, 280, 220]}, {"text": "Mount Everest", "bbox": [150, 280, 250, 300]}]',
                'bbox': [150, 200, 550, 320],
                'ocr_engine': 'synthetic',
                'conf_mean': 98.0,
                'conf_min': 98.0,
                'conf_max': 98.0,
                'tokens': 64,
                'source_authority': 'wikipedia',
                'is_distractor': True,
                'source_page_is_ocr': False,
                'watermark_flags': [],
                'font_stats': '{"primary_font": "Arial", "font_size_mean": 12}',
                'pdf_meta': '{"title": "Nepal Geography - Wikipedia", "language_hint": "en"}'
            },
            {
                'doc_id': 'health-definition-wiki-ne',
                'page_id': 'health-definition-wiki-ne_page_001',
                'block_id': 'block_001',
                'block_type': 'text',
                'text': 'स्वास्थ्य भनेको शारीरिक, मानसिक र सामाजिक कल्याणको पूर्ण अवस्था हो। यो केवल रोग वा अशक्तताको अनुपस्थिति मात्र होइन। विश्व स्वास्थ्य संगठनको परिभाषा अनुसार स्वास्थ्यलाई व्यापक रूपमा बुझिनुपर्छ। स्वस्थ जीवनशैली, नियमित शारीरिक व्यायाम, सन्तुलित आहार, पर्याप्त निद्रा र तनाव व्यवस्थापनले राम्रो स्वास्थ्य कायम राख्न सहायता गर्छ।',
                'language': 'ne',
                'char_span': '[{"text": "शारीरिक, मानसिक र सामाजिक कल्याण", "bbox": [100, 200, 400, 220]}, {"text": "विश्व स्वास्थ्य संगठन", "bbox": [100, 240, 280, 260]}]',
                'bbox': [100, 200, 550, 320],
                'ocr_engine': 'synthetic',
                'conf_mean': 98.0,
                'conf_min': 98.0,
                'conf_max': 98.0,
                'tokens': 46,
                'source_authority': 'wikipedia',
                'is_distractor': True,
                'source_page_is_ocr': False,
                'watermark_flags': [],
                'font_stats': '{"primary_font": "Devanagari", "font_size_mean": 12}',
                'pdf_meta': '{"title": "स्वास्थ्य - Wikipedia", "language_hint": "ne"}'
            }
        ]
        return chunks
    
    def generate_proper_corpus(self) -> pd.DataFrame:
        """Generate complete meaningful corpus."""
        all_chunks = []
        
        # Add authoritative content
        all_chunks.extend(self.create_constitution_chunks())
        all_chunks.extend(self.create_health_chunks())
        all_chunks.extend(self.create_covid_chunks())
        
        # Add Wikipedia distractors
        all_chunks.extend(self.create_wikipedia_distractors())
        
        # Add unique chunk IDs
        for i, chunk in enumerate(all_chunks):
            chunk['chunk_id'] = f"proper_chunk_{i+1:03d}"
        
        return pd.DataFrame(all_chunks)
    
    def save_corpus(self, output_path: Path) -> None:
        """Save the proper corpus to parquet file."""
        corpus_df = self.generate_proper_corpus()
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet
        corpus_df.to_parquet(output_path, index=False)
        
        print(f"✅ Proper corpus saved: {len(corpus_df)} chunks")
        print(f"   Authoritative chunks: {len(corpus_df[corpus_df['source_authority'] == 'authoritative'])}")
        print(f"   Wikipedia distractors: {len(corpus_df[corpus_df['source_authority'] == 'wikipedia'])}")
        print(f"   Languages: {corpus_df['language'].value_counts().to_dict()}")
        print(f"   Average text length: {corpus_df['text'].str.len().mean():.0f} chars")
        print(f"   Output: {output_path}")
        
        # Show sample content
        print(f"\n📝 Sample content:")
        for i, row in corpus_df.head(2).iterrows():
            print(f"   {row['language']}: {row['text'][:80]}...")


def main():
    """CLI for proper corpus generation."""
    parser = argparse.ArgumentParser(
        description="Generate proper meaningful corpus for NepaliGov-RAG-Bench"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/proper_corpus.parquet"),
        help="Output corpus parquet file"
    )
    
    args = parser.parse_args()
    
    try:
        generator = ProperCorpusGenerator()
        generator.save_corpus(args.out)
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Run chunking: python -m src.retriever.chunk_and_embed --in {args.out}")
        print(f"   2. Generate Q-A: python -m src.ingest.build_qacite --in {args.out}")
        print(f"   3. Test retrieval: python -m src.retriever.search --q 'नेपालको संविधानमा के अधिकारहरू छन्?'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



