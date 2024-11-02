#!/usr/bin/env python3
"""
PDF Object Filter for Watermark Removal.

Filters out watermark-like objects from born-digital PDFs using heuristics:
- Rotation angle |θ| > 10°
- Alpha transparency < 0.5
- Bounding box area > 25% of page
- Object appears on ≥ 30% of pages

Produces debug SVG overlays showing kept vs dropped objects.
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import fitz  # PyMuPDF


class PDFObjectAnalyzer:
    """Analyzes PDF objects to identify watermark-like patterns with enhanced detection."""
    
    def __init__(self, 
                 rotation_threshold: float = 10.0,
                 alpha_threshold: float = 0.5,
                 area_threshold: float = 0.25,
                 frequency_threshold: float = 0.3,
                 font_size_threshold: float = 24.0):
        """
        Initialize analyzer with enhanced watermark detection thresholds.
        
        Args:
            rotation_threshold: Rotation angle threshold in degrees
            alpha_threshold: Alpha transparency threshold (0-1)
            area_threshold: Area threshold as fraction of page area
            frequency_threshold: Frequency threshold as fraction of total pages
            font_size_threshold: Font size threshold for watermark text
        """
        self.rotation_threshold = rotation_threshold
        self.alpha_threshold = alpha_threshold
        self.area_threshold = area_threshold
        self.frequency_threshold = frequency_threshold
        self.font_size_threshold = font_size_threshold
        
        # Track objects across pages for frequency analysis
        self.object_signatures = defaultdict(list)  # signature -> [page_nums]
        self.page_objects = {}  # page_num -> [objects]
        
        # Common watermark text patterns
        self.watermark_patterns = [
            r'(?i)\b(confidential|draft|copy|duplicate|sample|watermark|preview)\b',
            r'(?i)\b(not for distribution|internal use|preliminary)\b',
            r'(?i)\b(copyright|©|\(c\))\b',
            r'(?i)\b(proprietary|restricted|classified)\b'
        ]
        
    def get_object_signature(self, obj: Dict) -> str:
        """
        Create a signature for an object to identify similar objects across pages.
        
        Args:
            obj: Object dictionary with properties
            
        Returns:
            String signature for the object
        """
        # Use text content, font, size, and approximate position
        text = obj.get('text', '').strip()
        font = obj.get('font', 'unknown')
        size = round(obj.get('size', 0), 1)
        
        # Normalize position to detect objects in similar relative positions
        bbox = obj.get('bbox', [0, 0, 0, 0])
        rel_x = round(bbox[0] / max(obj.get('page_width', 1), 1), 2)
        rel_y = round(bbox[1] / max(obj.get('page_height', 1), 1), 2)
        
        return f"{text}|{font}|{size}|{rel_x},{rel_y}"
    
    def extract_transformation_matrix(self, span: Dict) -> Tuple[float, float]:
        """
        Extract rotation and skew from text transformation matrix (if available).
        
        Args:
            span: Text span dictionary
            
        Returns:
            Tuple of (rotation_degrees, alpha_estimate)
        """
        # PyMuPDF doesn't easily expose transformation matrices for text spans
        # This is a simplified approach - in practice you'd need to examine
        # the PDF's content stream or use more advanced PDF parsing
        
        flags = span.get('flags', 0)
        
        # Estimate rotation from flags (very rough approximation)
        # Real implementation would parse the transformation matrix
        rotation = 0.0
        
        # Check for italic/oblique flags which might indicate skewed text
        if flags & (1 << 6):  # Italic flag
            rotation = 15.0  # Rough estimate
        
        # Estimate alpha from color intensity (rough approximation)
        color = span.get('color', 0)
        if color == 0:  # Black text
            alpha = 1.0
        else:
            # Convert color to grayscale and estimate alpha
            # This is very rough - real implementation would need proper color analysis
            alpha = min(1.0, color / 16777215.0)  # Normalize RGB to 0-1
        
        return rotation, alpha
    
    def matches_watermark_pattern(self, text: str) -> bool:
        """
        Check if text matches common watermark patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if text matches watermark patterns
        """
        for pattern in self.watermark_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def analyze_font_characteristics(self, span: Dict) -> Dict[str, any]:
        """
        Analyze font characteristics that might indicate watermarks.
        
        Args:
            span: Text span dictionary
            
        Returns:
            Dictionary with font analysis results
        """
        font = span.get('font', '')
        size = span.get('size', 0)
        flags = span.get('flags', 0)
        
        # Font name analysis
        is_bold = 'bold' in font.lower() or (flags & (1 << 4))
        is_italic = 'italic' in font.lower() or (flags & (1 << 6))
        is_decorative = any(term in font.lower() for term in ['script', 'display', 'decorative'])
        
        # Size analysis
        is_large_font = size > self.font_size_threshold
        is_very_small_font = size < 6.0
        
        return {
            'is_bold': is_bold,
            'is_italic': is_italic,
            'is_decorative': is_decorative,
            'is_large_font': is_large_font,
            'is_very_small_font': is_very_small_font,
            'font_name': font,
            'size': size
        }
    
    def extract_page_objects(self, page: fitz.Page, page_num: int) -> List[Dict]:
        """
        Extract objects from a PDF page with detailed properties.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number
            
        Returns:
            List of object dictionaries
        """
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        
        objects = []
        
        # Extract text objects with detailed information
        text_dict = page.get_text("dict")
        
        for block in text_dict.get("blocks", []):
            if "lines" not in block:  # Skip image blocks for now
                continue
                
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    
                    bbox = span["bbox"]
                    obj_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    
                    # Extract enhanced transformation and style information
                    flags = span.get("flags", 0)
                    font = span.get("font", "")
                    size = span.get("size", 0)
                    color = span.get("color", 0)
                    
                    # Enhanced rotation and alpha detection
                    rotation, alpha = self.extract_transformation_matrix(span)
                    
                    # Font characteristics analysis
                    font_analysis = self.analyze_font_characteristics(span)
                    
                    # Text pattern analysis
                    matches_watermark_pattern = self.matches_watermark_pattern(text)
                    
                    obj = {
                        'type': 'text',
                        'text': text,
                        'bbox': bbox,
                        'font': font,
                        'size': size,
                        'color': color,
                        'flags': flags,
                        'rotation': rotation,
                        'alpha': alpha,
                        'area': obj_area,
                        'area_ratio': obj_area / page_area if page_area > 0 else 0,
                        'page_width': page_rect.width,
                        'page_height': page_rect.height,
                        'page_num': page_num,
                        'font_analysis': font_analysis,
                        'matches_watermark_pattern': matches_watermark_pattern
                    }
                    
                    objects.append(obj)
        
        # Extract image objects
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            try:
                # Get image bbox (this is a simplified approach)
                img_bbox = page.get_image_bbox(img[7])  # img[7] is the xref
                img_area = (img_bbox[2] - img_bbox[0]) * (img_bbox[3] - img_bbox[1])
                
                obj = {
                    'type': 'image',
                    'text': f'image_{img_index}',
                    'bbox': list(img_bbox),
                    'font': 'image',
                    'size': 0,
                    'color': 0,
                    'flags': 0,
                    'rotation': 0.0,  # Would need matrix analysis
                    'alpha': 1.0,     # Would need image analysis
                    'area': img_area,
                    'area_ratio': img_area / page_area if page_area > 0 else 0,
                    'page_width': page_rect.width,
                    'page_height': page_rect.height,
                    'page_num': page_num,
                    'img_info': img
                }
                
                objects.append(obj)
                
            except Exception as e:
                print(f"Warning: Could not process image {img_index} on page {page_num}: {e}")
        
        return objects
    
    def is_watermark_like(self, obj: Dict, total_pages: int) -> Tuple[bool, List[str]]:
        """
        Determine if an object is watermark-like based on enhanced heuristics.
        
        Args:
            obj: Object dictionary
            total_pages: Total number of pages in document
            
        Returns:
            Tuple of (is_watermark, reasons)
        """
        reasons = []
        
        # Original heuristics
        # Check rotation threshold
        if abs(obj.get('rotation', 0)) > self.rotation_threshold:
            reasons.append(f"rotation_{obj.get('rotation', 0):.1f}deg")
        
        # Check alpha threshold
        if obj.get('alpha', 1.0) < self.alpha_threshold:
            reasons.append(f"alpha_{obj.get('alpha', 1.0):.2f}")
        
        # Check area threshold
        if obj.get('area_ratio', 0) > self.area_threshold:
            reasons.append(f"large_area_{obj.get('area_ratio', 0):.2f}")
        
        # Check frequency threshold
        signature = self.get_object_signature(obj)
        pages_with_signature = len(self.object_signatures.get(signature, []))
        frequency = pages_with_signature / total_pages if total_pages > 0 else 0
        
        if frequency >= self.frequency_threshold:
            reasons.append(f"frequent_{frequency:.2f}")
        
        # Enhanced heuristics
        # Check watermark text patterns
        if obj.get('matches_watermark_pattern', False):
            reasons.append("watermark_text_pattern")
        
        # Check font characteristics
        font_analysis = obj.get('font_analysis', {})
        if font_analysis.get('is_large_font', False) and font_analysis.get('is_bold', False):
            reasons.append("large_bold_font")
        
        if font_analysis.get('is_decorative', False):
            reasons.append("decorative_font")
        
        if font_analysis.get('is_very_small_font', False):
            reasons.append("micro_text")
        
        # Check positioning (center, corners, edges often indicate watermarks)
        bbox = obj.get('bbox', [0, 0, 0, 0])
        page_width = obj.get('page_width', 1)
        page_height = obj.get('page_height', 1)
        
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        rel_center_x = center_x / page_width if page_width > 0 else 0.5
        rel_center_y = center_y / page_height if page_height > 0 else 0.5
        
        # Center positioning (common for watermarks)
        if 0.3 < rel_center_x < 0.7 and 0.3 < rel_center_y < 0.7:
            if obj.get('size', 0) > self.font_size_threshold:
                reasons.append("center_positioned_large_text")
        
        # Edge positioning (headers/footers that might be watermarks)
        if rel_center_y < 0.1 or rel_center_y > 0.9:
            if frequency >= self.frequency_threshold * 0.5:  # Lower threshold for headers/footers
                reasons.append("edge_positioned_frequent")
        
        is_watermark = len(reasons) > 0
        return is_watermark, reasons
    
    def extract_pdf_metadata(self, doc: fitz.Document) -> Dict[str, any]:
        """
        Extract PDF metadata for enhanced classification.
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            Dictionary with PDF metadata
        """
        try:
            metadata = doc.metadata
            return {
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'subject': metadata.get('subject', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
                'modification_date': metadata.get('modDate', ''),
                'page_count': len(doc),
                'is_encrypted': doc.needs_pass,
                'is_pdf_a': 'PDF/A' in metadata.get('format', ''),
                'has_forms': doc.has_acro_form or doc.has_xfa_form
            }
        except Exception as e:
            print(f"Warning: Could not extract PDF metadata: {e}")
            return {}
    
    def analyze_document(self, pdf_path: Path) -> Dict:
        """
        Analyze entire document to identify watermark patterns with metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Analysis results dictionary
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise ValueError(f"Failed to open PDF {pdf_path}: {e}")
        
        print(f"Analyzing objects in {len(doc)} pages of {pdf_path.name}...")
        
        # Extract PDF metadata
        pdf_metadata = self.extract_pdf_metadata(doc)
        
        # First pass: extract all objects and build signatures
        all_objects = []
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                objects = self.extract_page_objects(page, page_num)
                
                self.page_objects[page_num] = objects
                all_objects.extend(objects)
                
                # Build object signatures for frequency analysis
                for obj in objects:
                    signature = self.get_object_signature(obj)
                    self.object_signatures[signature].append(page_num)
                
                print(f"  Page {page_num:3d}: {len(objects):3d} objects extracted")
                
            except Exception as e:
                print(f"  Warning: Failed to analyze page {page_num}: {e}")
        
        # Second pass: classify objects as watermarks or content
        kept_objects = []
        dropped_objects = []
        
        for obj in all_objects:
            is_watermark, reasons = self.is_watermark_like(obj, len(doc))
            
            if is_watermark:
                obj['drop_reasons'] = reasons
                dropped_objects.append(obj)
            else:
                kept_objects.append(obj)
        
        analysis_result = {
            'pdf_file': str(pdf_path.name),
            'total_pages': len(doc),
            'total_objects': len(all_objects),
            'kept_objects': len(kept_objects),
            'dropped_objects': len(dropped_objects),
            'kept_objects_list': kept_objects,
            'dropped_objects_list': dropped_objects,
            'object_signatures': dict(self.object_signatures),
            'pdf_metadata': pdf_metadata,
            'thresholds': {
                'rotation': self.rotation_threshold,
                'alpha': self.alpha_threshold,
                'area': self.area_threshold,
                'frequency': self.frequency_threshold,
                'font_size': self.font_size_threshold
            }
        }
        
        doc.close()
        return analysis_result


def create_debug_svg(page_objects: List[Dict], page_width: float, page_height: float, 
                    page_num: int) -> str:
    """
    Create SVG debug overlay showing kept vs dropped objects.
    
    Args:
        page_objects: List of objects for this page
        page_width: Page width
        page_height: Page height
        page_num: Page number
        
    Returns:
        SVG string
    """
    svg_lines = [
        f'<svg width="{page_width}" height="{page_height}" xmlns="http://www.w3.org/2000/svg">',
        f'<title>Page {page_num} Object Analysis</title>',
        '<defs>',
        '<style>',
        '.kept { fill: rgba(0, 255, 0, 0.3); stroke: green; stroke-width: 1; }',
        '.dropped { fill: rgba(255, 0, 0, 0.3); stroke: red; stroke-width: 2; }',
        '.text { font-family: Arial, sans-serif; font-size: 10px; }',
        '</style>',
        '</defs>',
        f'<rect width="{page_width}" height="{page_height}" fill="white" stroke="black" stroke-width="1"/>',
    ]
    
    # Group objects by status
    kept_objects = [obj for obj in page_objects if 'drop_reasons' not in obj]
    dropped_objects = [obj for obj in page_objects if 'drop_reasons' in obj]
    
    # Draw kept objects (green)
    for obj in kept_objects:
        bbox = obj['bbox']
        x, y, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        width = x2 - x
        height = y2 - y
        
        text_preview = obj.get('text', '')[:50].replace('"', '&quot;')
        svg_lines.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'class="kept" title="KEPT: {text_preview}"/>'
        )
    
    # Draw dropped objects (red)
    for obj in dropped_objects:
        bbox = obj['bbox']
        x, y, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        width = x2 - x
        height = y2 - y
        
        reasons = ', '.join(obj.get('drop_reasons', []))
        text_preview = obj.get('text', '')[:50].replace('"', '&quot;')
        svg_lines.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'class="dropped" title="DROPPED: {reasons} | {text_preview}"/>'
        )
    
    # Add legend
    legend_y = 20
    svg_lines.extend([
        f'<rect x="10" y="{legend_y}" width="15" height="15" class="kept"/>',
        f'<text x="30" y="{legend_y + 12}" class="text">Kept Objects ({len(kept_objects)})</text>',
        f'<rect x="10" y="{legend_y + 20}" width="15" height="15" class="dropped"/>',
        f'<text x="30" y="{legend_y + 32}" class="text">Dropped Objects ({len(dropped_objects)})</text>',
    ])
    
    svg_lines.append('</svg>')
    return '\n'.join(svg_lines)


def save_debug_svgs(analysis_result: Dict, page_objects_by_page: Dict, 
                   output_dir: Path) -> None:
    """
    Save debug SVG files for each page.
    
    Args:
        analysis_result: Analysis result from PDFObjectAnalyzer
        page_objects_by_page: Objects grouped by page number
        output_dir: Output directory for SVG files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_name = Path(analysis_result['pdf_file']).stem
    
    for page_num, page_objects in page_objects_by_page.items():
        if not page_objects:
            continue
            
        # Get page dimensions from first object
        page_width = page_objects[0].get('page_width', 600)
        page_height = page_objects[0].get('page_height', 800)
        
        # Create SVG
        svg_content = create_debug_svg(page_objects, page_width, page_height, page_num)
        
        # Save to file
        svg_file = output_dir / f"{pdf_name}_page_{page_num:03d}.svg"
        with open(svg_file, 'w', encoding='utf-8') as f:
            f.write(svg_content)
    
    print(f"✅ Saved {len(page_objects_by_page)} debug SVG files to {output_dir}")


def print_analysis_summary(analysis_result: Dict) -> None:
    """Print summary of object filtering analysis."""
    total_objects = analysis_result['total_objects']
    kept_objects = analysis_result['kept_objects']
    dropped_objects = analysis_result['dropped_objects']
    
    print(f"\n📊 Object Filtering Summary:")
    print(f"=" * 40)
    print(f"Total pages:        {analysis_result['total_pages']:4d}")
    print(f"Total objects:      {total_objects:4d}")
    print(f"Kept objects:       {kept_objects:4d} ({100*kept_objects/total_objects:.1f}%)")
    print(f"Dropped objects:    {dropped_objects:4d} ({100*dropped_objects/total_objects:.1f}%)")
    
    # Analyze drop reasons
    drop_reasons = defaultdict(int)
    for obj in analysis_result['dropped_objects_list']:
        for reason in obj.get('drop_reasons', []):
            drop_reasons[reason] += 1
    
    if drop_reasons:
        print(f"\nDrop reasons:")
        for reason, count in sorted(drop_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason:20s}: {count:3d} objects")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Filter watermark-like objects from PDF files"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Input PDF file path"
    )
    parser.add_argument(
        "--debug-out",
        type=Path,
        required=True,
        help="Output directory for debug SVG files"
    )
    parser.add_argument(
        "--rotation-threshold",
        type=float,
        default=10.0,
        help="Rotation angle threshold in degrees (default: 10.0)"
    )
    parser.add_argument(
        "--alpha-threshold",
        type=float,
        default=0.5,
        help="Alpha transparency threshold (default: 0.5)"
    )
    parser.add_argument(
        "--area-threshold",
        type=float,
        default=0.25,
        help="Area threshold as fraction of page (default: 0.25)"
    )
    parser.add_argument(
        "--frequency-threshold",
        type=float,
        default=0.3,
        help="Frequency threshold as fraction of pages (default: 0.3)"
    )
    parser.add_argument(
        "--font-size-threshold",
        type=float,
        default=24.0,
        help="Font size threshold for watermark detection (default: 24.0)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = PDFObjectAnalyzer(
            rotation_threshold=args.rotation_threshold,
            alpha_threshold=args.alpha_threshold,
            area_threshold=args.area_threshold,
            frequency_threshold=args.frequency_threshold,
            font_size_threshold=args.font_size_threshold
        )
        
        # Analyze document
        analysis_result = analyzer.analyze_document(args.pdf)
        
        # Save debug SVGs
        save_debug_svgs(analysis_result, analyzer.page_objects, args.debug_out)
        
        # Print summary
        print_analysis_summary(analysis_result)
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
