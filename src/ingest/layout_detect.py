#!/usr/bin/env python3
"""
Layout Detection for NepaliGov-RAG-Bench

Detects text, table, and figure blocks from both born-digital and OCR'd pages.
Integrates with CP2 page types and CP3 OCR results.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import fitz  # PyMuPDF
import pandas as pd
import numpy as np

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class LayoutBlock:
    """Container for detected layout blocks."""
    
    def __init__(self, 
                 block_id: str,
                 block_type: str,  # text|table|figure
                 text: str,
                 bbox: List[float],  # [x1, y1, x2, y2]
                 language: str = 'mixed',
                 char_spans: List[Dict] = None,
                 table_structure: Dict = None,
                 confidence_stats: Dict = None):
        self.block_id = block_id
        self.block_type = block_type
        self.text = text
        self.bbox = bbox
        self.language = language
        self.char_spans = char_spans or []
        self.table_structure = table_structure
        self.confidence_stats = confidence_stats or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'block_id': self.block_id,
            'block_type': self.block_type,
            'text': self.text,
            'bbox': self.bbox,
            'language': self.language,
            'char_spans': self.char_spans,
            'table_structure': self.table_structure,
            'confidence_stats': self.confidence_stats
        }


class LayoutDetector:
    """Layout detection for PDF pages using CP2/CP3 integration."""
    
    def __init__(self):
        self.camelot_available = CAMELOT_AVAILABLE
        self.pil_available = PIL_AVAILABLE
        
        if not self.camelot_available:
            print("Warning: Camelot not available. Table detection will be limited.")
    
    def detect_language(self, text: str) -> str:
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
    
    def extract_born_digital_blocks(self, page: fitz.Page, page_id: str) -> List[LayoutBlock]:
        """
        Enhanced extraction of text blocks and figures from born-digital pages.
        
        Args:
            page: PyMuPDF page object
            page_id: Page identifier
            
        Returns:
            List of detected layout blocks
        """
        blocks = []
        
        # Get text blocks with detailed information
        text_blocks = page.get_text("dict")
        
        block_counter = 0
        for block_data in text_blocks.get("blocks", []):
            if "lines" not in block_data:  # This is likely an image/figure block
                # Extract figure information
                if "bbox" in block_data:
                    figure_block = LayoutBlock(
                        block_id=f"{page_id}_figure_{block_counter:03d}",
                        block_type='figure',
                        text='[FIGURE]',
                        bbox=block_data['bbox'],
                        language='mixed',
                        char_spans=[]
                    )
                    blocks.append(figure_block)
                    block_counter += 1
                continue
            
            # Extract text and bbox
            lines = []
            char_spans = []
            bbox_coords = []
            
            for line in block_data["lines"]:
                line_text = ""
                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    if span_text:
                        line_text += span_text + " "
                        
                        # Record character spans
                        char_spans.append({
                            'text': span_text,
                            'bbox': span.get('bbox', [0, 0, 0, 0]),
                            'font': span.get('font', 'unknown'),
                            'size': span.get('size', 0),
                            'flags': span.get('flags', 0)
                        })
                        
                        # Collect bbox coordinates
                        if 'bbox' in span:
                            bbox_coords.extend(span['bbox'])
                
                if line_text.strip():
                    lines.append(line_text.strip())
            
            if not lines:
                continue
            
            # Combine lines into block text
            block_text = " ".join(lines)
            
            # Calculate block bbox
            if bbox_coords:
                # Group coordinates into [x1, y1, x2, y2] sets
                coords = [bbox_coords[i:i+4] for i in range(0, len(bbox_coords), 4)]
                if coords:
                    x_coords = [coord[0] for coord in coords] + [coord[2] for coord in coords]
                    y_coords = [coord[1] for coord in coords] + [coord[3] for coord in coords]
                    block_bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                else:
                    block_bbox = block_data.get('bbox', [0, 0, 0, 0])
            else:
                block_bbox = block_data.get('bbox', [0, 0, 0, 0])
            
            # Detect language
            language = self.detect_language(block_text)
            
            # Create layout block
            block_id = f"{page_id}_block_{block_counter:03d}"
            block = LayoutBlock(
                block_id=block_id,
                block_type='text',  # Default to text, table detection below
                text=block_text,
                bbox=block_bbox,
                language=language,
                char_spans=char_spans
            )
            
            blocks.append(block)
            block_counter += 1
        
        return blocks
    
    def detect_tables_camelot(self, pdf_path: Path, page_num: int) -> List[Dict]:
        """
        Enhanced table detection using Camelot with multiple strategies.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed for Camelot)
            
        Returns:
            List of table structures
        """
        if not self.camelot_available:
            return []
        
        table_structures = []
        
        # Try multiple Camelot flavors for better detection
        flavors = ['lattice', 'stream']
        
        for flavor in flavors:
            try:
                tables = camelot.read_pdf(str(pdf_path), pages=str(page_num + 1), flavor=flavor)
                
                for i, table in enumerate(tables):
                    if table.df.empty:
                        continue
                    
                    # Enhanced table structure extraction
                    structure = {
                        'table_id': f"table_{flavor}_{i}",
                        'flavor': flavor,
                        'rows': table.df.shape[0],
                        'cols': table.df.shape[1],
                        'data': table.df.to_dict('records'),
                        'bbox': table._bbox if hasattr(table, '_bbox') else None,
                        'confidence': getattr(table, 'accuracy', 0.0),
                        'has_headers': self._detect_table_headers(table.df),
                        'cell_count': table.df.shape[0] * table.df.shape[1],
                        'non_empty_cells': table.df.notna().sum().sum()
                    }
                    
                    # Only add if it's a substantial table
                    if structure['rows'] >= 2 and structure['cols'] >= 2:
                        table_structures.append(structure)
                
                # If we found tables with this flavor, don't try others
                if table_structures:
                    break
                    
            except Exception as e:
                print(f"Camelot {flavor} detection failed for page {page_num}: {e}")
                continue
        
        return table_structures
    
    def _detect_table_headers(self, df: pd.DataFrame) -> bool:
        """
        Detect if table has headers (first row different from others).
        
        Args:
            df: Table DataFrame
            
        Returns:
            True if likely has headers
        """
        if len(df) < 2:
            return False
        
        # Check if first row has different characteristics
        first_row = df.iloc[0]
        other_rows = df.iloc[1:]
        
        # Simple heuristic: if first row has more text-like content
        first_row_text_ratio = first_row.astype(str).str.len().mean()
        other_rows_text_ratio = other_rows.astype(str).str.len().mean()
        
        return first_row_text_ratio > other_rows_text_ratio * 1.2
    
    def extract_ocr_blocks(self, ocr_data: Dict, page_id: str) -> List[LayoutBlock]:
        """
        Extract blocks from OCR data (CP3 output).
        
        Args:
            ocr_data: OCR data from CP3
            page_id: Page identifier
            
        Returns:
            List of detected layout blocks
        """
        blocks = []
        
        if not ocr_data.get('pages'):
            return blocks
        
        # Find matching page data
        page_data = None
        for page in ocr_data['pages']:
            if page.get('page_id') == page_id:
                page_data = page
                break
        
        if not page_data:
            return blocks
        
        # Extract text blocks from OCR data
        normalized_text = page_data.get('normalized_text', '')
        line_bboxes = page_data.get('line_bboxes', [])
        char_span_bboxes = page_data.get('char_span_bboxes', [])
        
        if not normalized_text.strip():
            return blocks
        
        # Create blocks from line bboxes (group nearby lines)
        if line_bboxes:
            # Simple grouping: each line becomes a block for now
            # TODO: Could implement more sophisticated grouping based on proximity
            for i, line_data in enumerate(line_bboxes):
                block_id = f"{page_id}_ocr_block_{i:03d}"
                
                # Extract confidence stats
                confidence_stats = {
                    'mean_confidence': page_data.get('mean_confidence', 0),
                    'ocr_method': page_data.get('ocr_method', 'unknown'),
                    'quality_score': page_data.get('quality_score', 0)
                }
                
                # Detect language
                language = self.detect_language(line_data.get('text', ''))
                
                block = LayoutBlock(
                    block_id=block_id,
                    block_type='text',
                    text=line_data.get('text', ''),
                    bbox=line_data.get('bbox', [0, 0, 0, 0]),
                    language=language,
                    char_spans=char_span_bboxes,
                    confidence_stats=confidence_stats
                )
                
                blocks.append(block)
        else:
            # Fallback: create single block with all text
            block_id = f"{page_id}_ocr_block_000"
            
            confidence_stats = {
                'mean_confidence': page_data.get('mean_confidence', 0),
                'ocr_method': page_data.get('ocr_method', 'unknown'),
                'quality_score': page_data.get('quality_score', 0)
            }
            
            language = self.detect_language(normalized_text)
            
            # Estimate bbox from page dimensions
            page_width = page_data.get('page_width', 595)  # Default A4 width
            page_height = page_data.get('page_height', 842)  # Default A4 height
            estimated_bbox = [0, 0, page_width, page_height]
            
            block = LayoutBlock(
                block_id=block_id,
                block_type='text',
                text=normalized_text,
                bbox=estimated_bbox,
                language=language,
                char_spans=char_span_bboxes,
                confidence_stats=confidence_stats
            )
            
            blocks.append(block)
        
        return blocks
    
    def process_page(self, pdf_path: Path, page_num: int, page_type: str, 
                    ocr_data: Dict = None) -> Tuple[List[LayoutBlock], Dict]:
        """
        Process a single page for layout detection.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            page_type: Page type from CP2 ('text' or 'image')
            ocr_data: OCR data from CP3 (if available)
            
        Returns:
            Tuple of (layout blocks, page metadata)
        """
        doc = fitz.open(str(pdf_path))
        page = doc.load_page(page_num)
        doc_id = pdf_path.stem
        page_id = f"{doc_id}_page_{page_num:03d}"
        
        # Page metadata
        page_meta = {
            'page_id': page_id,
            'page_number': page_num,
            'page_type': page_type,
            'source_page_is_ocr': page_type == 'image' or (ocr_data is not None),
            'page_width': page.rect.width,
            'page_height': page.rect.height
        }
        
        blocks = []
        
        try:
            if page_type == 'text':
                # Born-digital page - extract using PyMuPDF
                blocks = self.extract_born_digital_blocks(page, page_id)
                
                # Try table detection with Camelot
                tables = self.detect_tables_camelot(pdf_path, page_num)
                if tables:
                    page_meta['tables_detected'] = len(tables)
                    # TODO: Mark text blocks that are actually tables
                
            elif page_type == 'image' and ocr_data:
                # OCR'd page - extract from CP3 data
                blocks = self.extract_ocr_blocks(ocr_data, page_id)
                
                # TODO: Add image-based table detection for OCR pages
                
            else:
                # No processable content
                print(f"Warning: No processable content for {page_id} (type={page_type})")
            
        except Exception as e:
            print(f"Error processing page {page_num}: {e}")
        
        finally:
            doc.close()
        
        return blocks, page_meta


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Layout detection for NepaliGov-RAG-Bench"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Input PDF file"
    )
    parser.add_argument(
        "--types",
        type=Path,
        required=True,
        help="Page types JSONL file from CP2"
    )
    parser.add_argument(
        "--ocr",
        type=Path,
        help="OCR JSON file from CP3 (optional)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON file for layout blocks"
    )
    
    args = parser.parse_args()
    
    # Load page types
    page_types = {}
    if args.types.exists():
        with open(args.types, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    page_info = json.loads(line)
                    page_types[page_info['page_id']] = page_info
    
    # Load OCR data if available
    ocr_data = None
    if args.ocr and args.ocr.exists():
        with open(args.ocr, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
    
    # Initialize detector
    detector = LayoutDetector()
    
    # Get PDF info
    doc = fitz.open(str(args.pdf))
    total_pages = len(doc)
    doc_id = args.pdf.stem
    doc.close()
    
    print(f"Processing layout detection for {total_pages} pages from {args.pdf.name}...")
    
    # Process pages
    all_blocks = []
    page_metadata = []
    
    for page_num in range(total_pages):
        page_id = f"{doc_id}_page_{page_num:03d}"
        page_info = page_types.get(page_id, {})
        page_type = page_info.get('page_type', 'image')  # Default to image if unknown
        
        print(f"  Page {page_num:3d}: type={page_type}", end="")
        
        # Process page
        blocks, meta = detector.process_page(args.pdf, page_num, page_type, ocr_data)
        
        all_blocks.extend(blocks)
        page_metadata.append(meta)
        
        print(f" → {len(blocks)} blocks detected")
    
    # Save results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'pdf_file': args.pdf.name,
        'doc_id': doc_id,
        'total_pages': total_pages,
        'total_blocks': len(all_blocks),
        'blocks': [block.to_dict() for block in all_blocks],
        'page_metadata': page_metadata,
        'processing_stats': {
            'camelot_available': detector.camelot_available,
            'pil_available': detector.pil_available
        }
    }
    
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Layout detection complete:")
    print(f"   Total blocks: {len(all_blocks)}")
    print(f"   Text blocks: {sum(1 for b in all_blocks if b.block_type == 'text')}")
    print(f"   Table blocks: {sum(1 for b in all_blocks if b.block_type == 'table')}")
    print(f"   Output saved to: {args.out}")


if __name__ == "__main__":
    main()
