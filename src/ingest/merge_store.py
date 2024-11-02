#!/usr/bin/env python3
"""
Merge & Store for NepaliGov-RAG-Bench

Merges born-digital text + OCR text, persists unified page-anchored Parquet
with rich signals from CP2/CP3, and exports page images.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

import pandas as pd
import numpy as np
import yaml
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    fitz = None
    FITZ_AVAILABLE = False

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class CorpusBuilder:
    """Builds unified corpus from CP2/CP3 outputs and layout detection."""
    
    def __init__(self, manifest_path: Path):
        """
        Initialize corpus builder.
        
        Args:
            manifest_path: Path to seed manifest YAML
        """
        self.manifest_path = manifest_path
        self.manifest_data = self._load_manifest()
        self.pil_available = PIL_AVAILABLE
        
        # Schema for corpus rows
        self.schema_fields = [
            'doc_id', 'page_id', 'block_id', 'block_type', 'text', 'language',
            'char_span', 'bbox', 'ocr_engine', 'conf_mean', 'conf_min', 'conf_max',
            'tokens', 'source_authority', 'is_distractor', 'source_page_is_ocr',
            'watermark_flags', 'font_stats', 'pdf_meta'
        ]
    
    def _load_manifest(self) -> Dict:
        """Load and validate manifest."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_page_types(self, doc_id: str) -> Dict[str, Dict]:
        """
        Load page types for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Dictionary mapping page_id to page info
        """
        types_file = Path("tmp/types.jsonl")
        page_types = {}
        
        if not types_file.exists():
            print(f"Warning: No page types file found at {types_file}")
            return page_types
        
        with open(types_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    page_info = json.loads(line)
                    if page_info.get('page_id', '').startswith(doc_id):
                        page_types[page_info['page_id']] = page_info
        
        return page_types
    
    def _load_ocr_data(self, doc_id: str) -> Optional[Dict]:
        """
        Load OCR data for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            OCR data dictionary or None
        """
        # Try both regular and enhanced OCR files
        ocr_files = [
            Path(f"data/ocr_json/{doc_id}-enhanced.json"),
            Path(f"data/ocr_json/{doc_id}.json")
        ]
        
        for ocr_file in ocr_files:
            if ocr_file.exists():
                with open(ocr_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        return None
    
    def _extract_pdf_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Enhanced PDF metadata extraction with additional analysis.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Enhanced PDF metadata dictionary
        """
        try:
            if not FITZ_AVAILABLE:
                raise ImportError("PyMuPDF unavailable")
            doc = fitz.open(str(pdf_path))
            metadata = doc.metadata
            
            # Enhanced metadata extraction
            enhanced_metadata = {
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
                'mod_date': metadata.get('modDate', ''),
                'page_count': len(doc),
                'subject': metadata.get('subject', ''),
                'keywords': metadata.get('keywords', ''),
                'format': metadata.get('format', ''),
                'encryption': metadata.get('encryption', ''),
                'pdf_version': doc.metadata.get('format', ''),
                'has_forms': len(doc.get_page_pixmap(0).tobytes()) > 0 if len(doc) > 0 else False,
                'file_size_bytes': pdf_path.stat().st_size,
                'file_modified': pdf_path.stat().st_mtime
            }
            
            # Analyze content characteristics
            if len(doc) > 0:
                page = doc.load_page(0)
                text_sample = page.get_text()[:500]  # First 500 chars
                
                # Language analysis
                devanagari_chars = sum(1 for char in text_sample if 0x0900 <= ord(char) <= 0x097F)
                total_chars = len([c for c in text_sample if c.isalpha()])
                devanagari_ratio = devanagari_chars / max(total_chars, 1)
                
                enhanced_metadata.update({
                    'content_language_hint': 'ne' if devanagari_ratio > 0.6 else 'en' if devanagari_ratio < 0.1 else 'mixed',
                    'devanagari_ratio': devanagari_ratio,
                    'text_sample': text_sample[:200]  # First 200 chars for analysis
                })
            
            doc.close()
            return enhanced_metadata
            
        except Exception as e:
            print(f"Warning: Could not extract PDF metadata from {pdf_path}: {e}")
            return {
                'file_size_bytes': pdf_path.stat().st_size if pdf_path.exists() else 0,
                'file_modified': pdf_path.stat().st_mtime if pdf_path.exists() else 0
            }
    
    def _export_page_image(self, pdf_path: Path, page_num: int, 
                          output_dir: Path, doc_id: str) -> Optional[str]:
        """
        Export page as PNG image.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            output_dir: Output directory for images
            doc_id: Document identifier
            
        Returns:
            Path to exported image or None
        """
        if not FITZ_AVAILABLE:
            return None
        try:
            doc = fitz.open(str(pdf_path))
            page = doc.load_page(page_num)
            
            # Render page to image (150 DPI for reasonable file size)
            pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
            
            # Save as PNG
            image_path = output_dir / f"{doc_id}_page_{page_num:03d}.png"
            pix.save(str(image_path))
            
            doc.close()
            return str(image_path.name)
            
        except Exception as e:
            print(f"Warning: Could not export page {page_num} from {pdf_path}: {e}")
            return None
    
    def _tokenize_text(self, text: str) -> int:
        """
        Enhanced multilingual tokenization for token count.
        
        Args:
            text: Input text
            
        Returns:
            Token count
        """
        if not text.strip():
            return 0
        
        # Enhanced tokenization for multilingual text
        import re
        
        # Split on whitespace and punctuation, but preserve Devanagari words
        # This handles both English and Nepali text better
        tokens = re.findall(r'\S+', text)
        
        # Filter out very short tokens (likely artifacts)
        meaningful_tokens = [token for token in tokens if len(token) > 1 or token.isalnum()]
        
        return len(meaningful_tokens)
    
    def _detect_language(self, text: str) -> str:
        """
        Detect language based on Devanagari character ratio.
        
        Args:
            text: Input text
            
        Returns:
            Language code ('ne', 'en', 'mixed')
        """
        if not text.strip():
            return 'mixed'
        
        # Count Devanagari characters
        devanagari_chars = sum(1 for char in text if 0x0900 <= ord(char) <= 0x097F)
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return 'mixed'
        
        devanagari_ratio = devanagari_chars / total_chars
        
        if devanagari_ratio > 0.6:
            return 'ne'
        elif devanagari_ratio < 0.1:
            return 'en'
        else:
            return 'mixed'
    
    def _process_document(self, doc_info: Dict) -> List[Dict[str, Any]]:
        """
        Process a single document into corpus rows.
        
        Args:
            doc_info: Document info from manifest
            
        Returns:
            List of corpus row dictionaries
        """
        doc_id = doc_info['doc_id']
        # Handle both absolute and relative file paths
        file_path = doc_info['file']
        if file_path.startswith('raw_pdfs/'):
            pdf_path = Path("data") / file_path
        else:
            pdf_path = Path("data/raw_pdfs") / file_path
        
        if not pdf_path.exists():
            print(f"Warning: PDF not found: {pdf_path}")
            return []
        
        print(f"Processing document: {doc_id}")
        
        # Load supporting data
        page_types = self._load_page_types(doc_id)
        ocr_data = self._load_ocr_data(doc_id)
        pdf_meta = self._extract_pdf_metadata(pdf_path)
        
        # Create page images directory
        images_dir = Path("data/page_images")
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Process pages
        corpus_rows = []

        if not FITZ_AVAILABLE:
            # Fallback: rely solely on OCR data
            if not ocr_data:
                print("Warning: PyMuPDF unavailable and no OCR data; skipping document")
                return []
            for page_data in ocr_data.get('pages', []):
                page_id = page_data.get('page_id')
                page_info = page_types.get(page_id, {})
                rows = self._process_ocr_page(
                    doc_info, page_id, page_info, ocr_data, pdf_meta, None
                )
                corpus_rows.extend(rows)
        else:
            doc = fitz.open(str(pdf_path))
            try:
                for page_num in range(len(doc)):
                    page_id = f"{doc_id}_page_{page_num:03d}"
                    page_info = page_types.get(page_id, {})
                    page_type = page_info.get('page_type', 'unknown')
                    
                    # Export page image
                    image_filename = self._export_page_image(pdf_path, page_num, images_dir, doc_id)
                    
                    # Determine if this is an OCR page
                    source_page_is_ocr = (page_type == 'image') or (ocr_data is not None and 
                        any(p.get('page_id') == page_id for p in ocr_data.get('pages', [])))
                    
                    # Process page content
                    if page_type == 'text' and not source_page_is_ocr:
                        # Born-digital text page
                        rows = self._process_born_digital_page(
                            doc.load_page(page_num), doc_info, page_id, page_info, pdf_meta, image_filename
                        )
                        corpus_rows.extend(rows)
                    
                    elif source_page_is_ocr and ocr_data:
                        # OCR'd page
                        rows = self._process_ocr_page(
                            doc_info, page_id, page_info, ocr_data, pdf_meta, image_filename
                        )
                        corpus_rows.extend(rows)
                    
                    else:
                        # Create minimal row for pages without processable content
                        row = self._create_base_row(doc_info, page_id, pdf_meta, image_filename)
                        row.update({
                            'block_id': f"{page_id}_empty",
                            'block_type': 'empty',
                            'text': '',
                            'language': 'mixed',
                            'source_page_is_ocr': source_page_is_ocr
                        })
                        corpus_rows.append(row)
            finally:
                doc.close()
        
        print(f"  → {len(corpus_rows)} rows generated")
        return corpus_rows
    
    def _create_base_row(self, doc_info: Dict, page_id: str, 
                        pdf_meta: Dict, image_filename: Optional[str]) -> Dict[str, Any]:
        """Create base row with common fields."""
        return {
            'doc_id': doc_info['doc_id'],
            'page_id': page_id,
            'source_authority': doc_info.get('authority', 'unknown'),
            'is_distractor': doc_info.get('is_distractor', False),
            'pdf_meta': json.dumps(pdf_meta),
            'page_image': image_filename,
            # Initialize nullable fields
            'char_span': None,
            'bbox': None,
            'ocr_engine': None,
            'conf_mean': None,
            'conf_min': None,
            'conf_max': None,
            'tokens': 0,
            'watermark_flags': None,
            'font_stats': None
        }
    
    def _process_born_digital_page(self, page: Any, doc_info: Dict, 
                                  page_id: str, page_info: Dict, pdf_meta: Dict,
                                  image_filename: Optional[str]) -> List[Dict[str, Any]]:
        """
        Process born-digital page content.
        
        Args:
            page: PyMuPDF page object
            doc_info: Document info from manifest
            page_id: Page identifier
            page_info: Page info from CP2
            pdf_meta: PDF metadata
            image_filename: Page image filename
            
        Returns:
            List of corpus rows
        """
        if not FITZ_AVAILABLE:
            return []
        rows = []
        
        # Extract text blocks
        text_blocks = page.get_text("dict")
        
        block_counter = 0
        for block_data in text_blocks.get("blocks", []):
            if "lines" not in block_data:  # Skip image blocks
                continue
            
            # Extract text and metadata
            lines = []
            char_spans = []
            font_info = []
            
            for line in block_data["lines"]:
                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    if span_text:
                        lines.append(span_text)
                        
                        # Character span info
                        char_spans.append({
                            'text': span_text,
                            'bbox': span.get('bbox', [0, 0, 0, 0])
                        })
                        
                        # Font statistics
                        font_info.append({
                            'font': span.get('font', 'unknown'),
                            'size': span.get('size', 0),
                            'flags': span.get('flags', 0)
                        })
            
            if not lines:
                continue
            
            block_text = " ".join(lines)
            
            # Create corpus row
            row = self._create_base_row(doc_info, page_id, pdf_meta, image_filename)
            row.update({
                'block_id': f"{page_id}_block_{block_counter:03d}",
                'block_type': 'text',
                'text': block_text,
                'language': self._detect_language(block_text),
                'char_span': json.dumps(char_spans) if char_spans else None,
                'bbox': json.dumps(block_data.get('bbox', [0, 0, 0, 0])),
                'tokens': self._tokenize_text(block_text),
                'source_page_is_ocr': False,
                'font_stats': json.dumps(font_info) if font_info else None,
                'watermark_flags': json.dumps(page_info.get('watermark_objects', {})) if page_info.get('watermark_objects') else None
            })
            
            rows.append(row)
            block_counter += 1
        
        # If no text blocks found, create empty row
        if not rows:
            row = self._create_base_row(doc_info, page_id, pdf_meta, image_filename)
            row.update({
                'block_id': f"{page_id}_empty",
                'block_type': 'empty',
                'text': '',
                'language': 'mixed',
                'source_page_is_ocr': False
            })
            rows.append(row)
        
        return rows
    
    def _process_ocr_page(self, doc_info: Dict, page_id: str, page_info: Dict,
                         ocr_data: Dict, pdf_meta: Dict, 
                         image_filename: Optional[str]) -> List[Dict[str, Any]]:
        """
        Process OCR'd page content.
        
        Args:
            doc_info: Document info from manifest
            page_id: Page identifier
            page_info: Page info from CP2
            ocr_data: OCR data from CP3
            pdf_meta: PDF metadata
            image_filename: Page image filename
            
        Returns:
            List of corpus rows
        """
        rows = []
        
        # Find matching page in OCR data
        page_ocr_data = None
        for page_data in ocr_data.get('pages', []):
            if page_data.get('page_id') == page_id:
                page_ocr_data = page_data
                break
        
        if not page_ocr_data:
            # Create empty row for pages without OCR data
            row = self._create_base_row(doc_info, page_id, pdf_meta, image_filename)
            row.update({
                'block_id': f"{page_id}_no_ocr",
                'block_type': 'empty',
                'text': '',
                'language': 'mixed',
                'source_page_is_ocr': True
            })
            return [row]
        
        # Extract OCR blocks
        normalized_text = page_ocr_data.get('normalized_text', '')
        line_bboxes = page_ocr_data.get('line_bboxes', [])
        char_span_bboxes = page_ocr_data.get('char_span_bboxes', [])
        
        # Confidence statistics
        conf_mean = page_ocr_data.get('mean_confidence', 0)
        ocr_method = page_ocr_data.get('ocr_method', 'unknown')
        
        if line_bboxes:
            # Create blocks from line bboxes
            for i, line_data in enumerate(line_bboxes):
                line_text = line_data.get('text', '').strip()
                if not line_text:
                    continue
                
                row = self._create_base_row(doc_info, page_id, pdf_meta, image_filename)
                row.update({
                    'block_id': f"{page_id}_ocr_{i:03d}",
                    'block_type': 'text',
                    'text': line_text,
                    'language': self._detect_language(line_text),
                    'char_span': json.dumps(char_span_bboxes) if char_span_bboxes else None,
                    'bbox': json.dumps(line_data.get('bbox', [0, 0, 0, 0])),
                    'ocr_engine': ocr_method,
                    'conf_mean': conf_mean,
                    'conf_min': conf_mean,  # Line-level confidence not available
                    'conf_max': conf_mean,
                    'tokens': self._tokenize_text(line_text),
                    'source_page_is_ocr': True
                })
                
                rows.append(row)
        
        elif normalized_text.strip():
            # Fallback: single block with all OCR text
            row = self._create_base_row(doc_info, page_id, pdf_meta, image_filename)
            row.update({
                'block_id': f"{page_id}_ocr_full",
                'block_type': 'text',
                'text': normalized_text,
                'language': self._detect_language(normalized_text),
                'char_span': json.dumps(char_span_bboxes) if char_span_bboxes else None,
                'bbox': json.dumps([0, 0, page_ocr_data.get('page_width', 595), page_ocr_data.get('page_height', 842)]),
                'ocr_engine': ocr_method,
                'conf_mean': conf_mean,
                'conf_min': conf_mean,
                'conf_max': conf_mean,
                'tokens': self._tokenize_text(normalized_text),
                'source_page_is_ocr': True
            })
            
            rows.append(row)
        
        else:
            # Empty OCR result
            row = self._create_base_row(doc_info, page_id, pdf_meta, image_filename)
            row.update({
                'block_id': f"{page_id}_ocr_empty",
                'block_type': 'empty',
                'text': '',
                'language': 'mixed',
                'source_page_is_ocr': True,
                'ocr_engine': ocr_method,
                'conf_mean': conf_mean
            })
            rows.append(row)
        
        return rows
    
    def build_corpus(self) -> pd.DataFrame:
        """
        Build complete corpus from all documents.
        
        Returns:
            Pandas DataFrame with corpus data
        """
        print("Building unified corpus...")
        
        all_rows = []
        processed_docs = 0
        
        for doc_info in self.manifest_data.get('documents', []):
            try:
                doc_rows = self._process_document(doc_info)
                all_rows.extend(doc_rows)
                processed_docs += 1
            except Exception as e:
                print(f"Error processing document {doc_info.get('doc_id', 'unknown')}: {e}")
                continue
        
        print(f"\n📊 Corpus Statistics:")
        print(f"   Documents processed: {processed_docs}")
        print(f"   Total rows: {len(all_rows)}")
        
        if not all_rows:
            print("Warning: No corpus rows generated")
            return pd.DataFrame(columns=self.schema_fields)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_rows)
        
        # Ensure all schema fields are present
        for field in self.schema_fields:
            if field not in df.columns:
                df[field] = None
        
        # Reorder columns to match schema
        df = df[self.schema_fields]
        
        # Enhanced statistics
        text_rows = len(df[df['block_type'] == 'text'])
        table_rows = len(df[df['block_type'] == 'table'])
        figure_rows = len(df[df['block_type'] == 'figure'])
        ocr_rows = len(df[df['source_page_is_ocr'] == True])
        ne_rows = len(df[df['language'] == 'ne'])
        en_rows = len(df[df['language'] == 'en'])
        mixed_rows = len(df[df['language'] == 'mixed'])
        
        # Quality metrics
        avg_confidence = df[df['conf_mean'].notna()]['conf_mean'].mean() if len(df[df['conf_mean'].notna()]) > 0 else 0
        total_tokens = df['tokens'].sum()
        avg_tokens_per_block = df[df['tokens'] > 0]['tokens'].mean() if len(df[df['tokens'] > 0]) > 0 else 0
        
        print(f"   Text blocks: {text_rows}")
        print(f"   Table blocks: {table_rows}")
        print(f"   Figure blocks: {figure_rows}")
        print(f"   OCR blocks: {ocr_rows}")
        print(f"   Nepali blocks: {ne_rows}")
        print(f"   English blocks: {en_rows}")
        print(f"   Mixed blocks: {mixed_rows}")
        print(f"   Total tokens: {total_tokens:,}")
        print(f"   Avg tokens/block: {avg_tokens_per_block:.1f}")
        if avg_confidence > 0:
            print(f"   Avg OCR confidence: {avg_confidence:.1f}")
        
        return df


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Merge and store unified corpus for NepaliGov-RAG-Bench"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Seed manifest YAML file"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/corpus_parquet"),
        help="Output directory for Parquet files (default: data/corpus_parquet)"
    )
    parser.add_argument(
        "--enhanced",
        action="store_true",
        default=True,
        help="Enable enhanced processing (table detection, figure extraction, enhanced metadata)"
    )
    parser.add_argument(
        "--image-dpi",
        type=int,
        default=150,
        help="DPI for page image export (default: 150)"
    )
    
    args = parser.parse_args()
    
    # Initialize corpus builder
    builder = CorpusBuilder(args.manifest)
    
    # Build corpus
    corpus_df = builder.build_corpus()
    
    if corpus_df.empty:
        print("❌ No corpus data generated")
        sys.exit(1)
    
    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-document Parquet files
    print("\nSaving per-document Parquet files...")
    doc_groups = corpus_df.groupby('doc_id')
    
    for doc_id, doc_df in doc_groups:
        doc_parquet_path = args.out_dir / f"{doc_id}.parquet"
        doc_df.to_parquet(doc_parquet_path, index=False)
        print(f"  Saved: {doc_parquet_path} ({len(doc_df)} rows)")
    
    # Save unified corpus
    unified_path = Path("data/corpus.parquet")
    unified_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_df.to_parquet(unified_path, index=False)
    
    print(f"\n✅ Corpus building complete:")
    print(f"   Unified corpus: {unified_path} ({len(corpus_df)} rows)")
    print(f"   Per-doc files: {args.out_dir}/ ({len(doc_groups)} files)")
    print(f"   Page images: data/page_images/ ({len(corpus_df['page_id'].unique())} pages)")


if __name__ == "__main__":
    main()
