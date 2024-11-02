#!/usr/bin/env python3
"""
Page Type Detection for PDFs.

Classifies pages as "text" vs "image" based on text density analysis using PyMuPDF.
Outputs results to JSONL format for downstream processing.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF


def detect_page_creation_method(page: fitz.Page) -> str:
    """
    Detect if page is born-digital or OCR-scanned based on characteristics.
    
    Args:
        page: PyMuPDF page object
        
    Returns:
        'born_digital', 'ocr_scanned', or 'mixed'
    """
    # Get images on the page
    images = page.get_images()
    
    # Get text blocks
    text_dict = page.get_text("dict")
    text_blocks = [block for block in text_dict.get("blocks", []) if "lines" in block]
    
    # Heuristics for detection
    has_large_images = any(
        page.get_image_bbox(img[7]).width * page.get_image_bbox(img[7]).height > 
        page.rect.width * page.rect.height * 0.5
        for img in images
    )
    
    # Check font consistency (OCR often has inconsistent fonts)
    fonts = set()
    for block in text_blocks:
        for line in block["lines"]:
            for span in line["spans"]:
                fonts.add(span.get("font", ""))
    
    font_diversity = len(fonts)
    
    # Check for searchable text vs image ratio
    page_area = page.rect.width * page.rect.height
    text_coverage = sum(
        (span["bbox"][2] - span["bbox"][0]) * (span["bbox"][3] - span["bbox"][1])
        for block in text_blocks
        for line in block["lines"]
        for span in line["spans"]
    ) / page_area if page_area > 0 else 0
    
    # Classification logic
    if has_large_images and text_coverage < 0.1:
        return 'ocr_scanned'
    elif font_diversity > 10 and text_coverage > 0.3:
        return 'mixed'  # Likely OCR with some born-digital elements
    else:
        return 'born_digital'


def analyze_page_text_density(page: fitz.Page) -> Dict[str, float]:
    """
    Analyze text density and characteristics of a PDF page with enhanced metrics.
    
    Args:
        page: PyMuPDF page object
        
    Returns:
        Dictionary with comprehensive text analysis metrics
    """
    # Get page dimensions
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height
    
    if page_area == 0:
        return {
            'text_density': 0.0,
            'char_count': 0,
            'block_count': 0,
            'line_count': 0,
            'word_count': 0,
            'coverage_ratio': 0.0,
            'avg_font_size': 0.0
        }
    
    # Extract text blocks with detailed information
    text_dict = page.get_text("dict")
    
    char_count = 0
    block_count = 0
    line_count = 0
    word_count = 0
    text_area = 0.0
    font_sizes = []
    
    for block in text_dict.get("blocks", []):
        if "lines" not in block:  # Skip image blocks
            continue
            
        block_count += 1
        block_bbox = fitz.Rect(block["bbox"])
        
        for line in block["lines"]:
            line_count += 1
            line_bbox = fitz.Rect(line["bbox"])
            
            for span in line["spans"]:
                span_text = span.get("text", "")
                char_count += len(span_text)
                word_count += len(span_text.split())
                
                # Track font sizes
                font_size = span.get("size", 0)
                if font_size > 0:
                    font_sizes.append(font_size)
                
                # Estimate text area (rough approximation)
                span_bbox = fitz.Rect(span["bbox"])
                text_area += span_bbox.width * span_bbox.height
    
    # Calculate metrics
    text_density = char_count / page_area if page_area > 0 else 0.0
    coverage_ratio = text_area / page_area if page_area > 0 else 0.0
    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0.0
    
    # Additional metrics for better classification
    font_diversity = len(set(font_sizes)) if font_sizes else 0
    min_font_size = min(font_sizes) if font_sizes else 0
    max_font_size = max(font_sizes) if font_sizes else 0
    
    # Note: creation_method detection would need page object, 
    # but we don't have it in this context. Will be added at higher level.
    creation_method = 'unknown'
    
    return {
        'text_density': text_density,
        'char_count': char_count,
        'block_count': block_count,
        'line_count': line_count,
        'word_count': word_count,
        'coverage_ratio': coverage_ratio,
        'avg_font_size': avg_font_size,
        'font_diversity': font_diversity,
        'min_font_size': min_font_size,
        'max_font_size': max_font_size,
        'creation_method': creation_method
    }


def classify_page_type(
    page_metrics: Dict[str, float],
    text_density_threshold: float = 0.001,
    char_count_threshold: int = 50,
    coverage_threshold: float = 0.01
) -> str:
    """
    Classify page as "text" or "image" based on text metrics.
    
    Args:
        page_metrics: Text analysis metrics from analyze_page_text_density
        text_density_threshold: Minimum text density for "text" classification
        char_count_threshold: Minimum character count for "text" classification
        coverage_threshold: Minimum text coverage ratio for "text" classification
        
    Returns:
        Page type: "text" or "image"
    """
    # Multiple criteria for robust classification
    has_sufficient_density = page_metrics['text_density'] >= text_density_threshold
    has_sufficient_chars = page_metrics['char_count'] >= char_count_threshold
    has_sufficient_coverage = page_metrics['coverage_ratio'] >= coverage_threshold
    has_text_blocks = page_metrics['block_count'] > 0
    
    # Page is "text" if it meets multiple criteria
    text_criteria_met = sum([
        has_sufficient_density,
        has_sufficient_chars, 
        has_sufficient_coverage,
        has_text_blocks
    ])
    
    return "text" if text_criteria_met >= 2 else "image"


def detect_page_types(
    pdf_path: Path,
    text_density_threshold: float = 0.001,
    char_count_threshold: int = 50,
    coverage_threshold: float = 0.01
) -> List[Dict[str, any]]:
    """
    Detect page types for all pages in a PDF.
    
    Args:
        pdf_path: Path to PDF file
        text_density_threshold: Threshold for text density classification
        char_count_threshold: Threshold for character count classification
        coverage_threshold: Threshold for coverage ratio classification
        
    Returns:
        List of page type records
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        raise ValueError(f"Failed to open PDF {pdf_path}: {e}")
    
    page_records = []
    
    print(f"Analyzing {len(doc)} pages in {pdf_path.name}...")
    
    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            
            # Analyze page text characteristics
            metrics = analyze_page_text_density(page)
            
            # Detect creation method
            creation_method = detect_page_creation_method(page)
            metrics['creation_method'] = creation_method
            
            # Classify page type
            page_type = classify_page_type(
                metrics, 
                text_density_threshold,
                char_count_threshold, 
                coverage_threshold
            )
            
            # Create page record
            page_record = {
                'page_id': f"{pdf_path.stem}_page_{page_num:03d}",
                'page_number': page_num,
                'page_type': page_type,
                'metrics': metrics,
                'pdf_file': str(pdf_path.name)
            }
            
            page_records.append(page_record)
            
            # Progress info
            print(f"  Page {page_num:3d}: {page_type:5s} "
                  f"(chars={metrics['char_count']:4d}, "
                  f"density={metrics['text_density']:.6f}, "
                  f"coverage={metrics['coverage_ratio']:.4f})")
            
        except Exception as e:
            print(f"  Warning: Failed to analyze page {page_num}: {e}")
            # Create fallback record
            page_record = {
                'page_id': f"{pdf_path.stem}_page_{page_num:03d}",
                'page_number': page_num,
                'page_type': 'image',  # Conservative fallback
                'metrics': {'error': str(e)},
                'pdf_file': str(pdf_path.name)
            }
            page_records.append(page_record)
    
    doc.close()
    return page_records


def save_page_types_jsonl(page_records: List[Dict[str, any]], output_path: Path) -> None:
    """
    Save page type records to JSONL format.
    
    Args:
        page_records: List of page type records
        output_path: Output JSONL file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in page_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(page_records)} page type records to {output_path}")


def print_summary_statistics(page_records: List[Dict[str, any]]) -> None:
    """Print summary statistics of page type detection."""
    if not page_records:
        print("No pages analyzed.")
        return
    
    # Count page types
    type_counts = {'text': 0, 'image': 0}
    total_chars = 0
    valid_metrics = []
    
    for record in page_records:
        page_type = record.get('page_type', 'unknown')
        type_counts[page_type] = type_counts.get(page_type, 0) + 1
        
        metrics = record.get('metrics', {})
        if isinstance(metrics, dict) and 'char_count' in metrics:
            total_chars += metrics['char_count']
            valid_metrics.append(metrics)
    
    total_pages = len(page_records)
    text_pages = type_counts.get('text', 0)
    image_pages = type_counts.get('image', 0)
    
    print(f"\n📊 Page Type Detection Summary:")
    print(f"=" * 40)
    print(f"Total pages:        {total_pages:4d}")
    print(f"Text pages:         {text_pages:4d} ({100*text_pages/total_pages:.1f}%)")
    print(f"Image pages:        {image_pages:4d} ({100*image_pages/total_pages:.1f}%)")
    print(f"Total characters:   {total_chars:6d}")
    
    if valid_metrics:
        avg_density = sum(m['text_density'] for m in valid_metrics) / len(valid_metrics)
        avg_coverage = sum(m['coverage_ratio'] for m in valid_metrics) / len(valid_metrics)
        print(f"Avg text density:   {avg_density:.6f}")
        print(f"Avg coverage ratio: {avg_coverage:.4f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Detect page types (text vs image) in PDF files"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Input PDF file path"
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--text-density-threshold",
        type=float,
        default=0.001,
        help="Text density threshold for classification (default: 0.001)"
    )
    parser.add_argument(
        "--char-count-threshold",
        type=int,
        default=50,
        help="Character count threshold for classification (default: 50)"
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.01,
        help="Coverage ratio threshold for classification (default: 0.01)"
    )
    
    args = parser.parse_args()
    
    try:
        # Detect page types
        page_records = detect_page_types(
            args.pdf,
            args.text_density_threshold,
            args.char_count_threshold,
            args.coverage_threshold
        )
        
        # Save results
        save_page_types_jsonl(page_records, args.out)
        
        # Print summary
        print_summary_statistics(page_records)
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
