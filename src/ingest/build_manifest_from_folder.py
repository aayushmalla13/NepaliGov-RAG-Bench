#!/usr/bin/env python3
"""
Build YAML manifest from existing PDFs with robust authority/distractor flags + SHA-256.

This module implements bulletproof pattern logic for distinguishing between authoritative
government sources and distractors (especially Wikipedia), with comprehensive heuristics
for metadata extraction.
"""

import argparse
import hashlib
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm


# Compiled regex patterns for performance and consistency  
PATTERN_AUTH = re.compile(r'(?i)(^|[_\W])(gov\.?_?np|govnp|mohp|dohs|heoc|parliament|who(\.int)?)([_\W]|$)')

# Wikipedia detection patterns
P_WP1 = re.compile(r'(?i)(^|[_\W])(wikipedia|wikidata|wikisource|wikibooks|wikimedia)([_\W]|$)')
P_WP2 = re.compile(r'(?i)(^|[_\W])(en|ne|hi|np)?wiki(pedia)?([_\W]|$)')  # Allow lang prefix
P_WP3 = re.compile(r'(?i)(^|[_\W])(en|ne|hi|np)?wp([_\W]|$)')

# Wikipedia negative guards
N_WP = re.compile(r'(?i)(whitepaper|wordpress|wp-content|wpadmin|workplan|work-package)')

# Devanagari character range
DEVANAGARI_RANGE = re.compile(r'[\u0900-\u097F]')
LETTER_CHARS = re.compile(r'[a-zA-Z\u0900-\u097F]')

# Year extraction
YEAR_PATTERN = re.compile(r'\b(19[5-9][0-9]|20[0-5][0-9])\b')

# Stopwords for title cleaning
STOPWORDS = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def is_wikipedia_like(name: str) -> Tuple[bool, Dict[str, bool]]:
    """
    Determine if filename suggests Wikipedia using bulletproof pattern logic.
    
    Args:
        name: Filename to check
        
    Returns:
        Tuple of (is_wikipedia, pattern_matches_dict)
    """
    # Check positive patterns
    wp1_match = bool(P_WP1.search(name))
    wp2_match = bool(P_WP2.search(name))
    wp3_match = bool(P_WP3.search(name))
    
    # Check negative guards
    guard_match = bool(N_WP.search(name))
    
    # Decision logic
    positive_match = wp1_match or wp2_match or wp3_match
    is_wikipedia = positive_match and not guard_match
    
    pattern_matches = {
        'wp1_explicit': wp1_match,
        'wp2_wiki': wp2_match,
        'wp3_wp_token': wp3_match,
        'guard_hit': guard_match,
        'positive_match': positive_match,
        'is_wikipedia': is_wikipedia
    }
    
    return is_wikipedia, pattern_matches


def compute_language_guess(filename: str, devanagari_thresh: float = 0.25) -> str:
    """
    Guess language based on Devanagari character ratio.
    
    Args:
        filename: The filename to analyze
        devanagari_thresh: Threshold for Devanagari ratio to classify as Nepali
        
    Returns:
        Language code: 'ne' for Nepali, 'en' for English
    """
    devanagari_chars = len(DEVANAGARI_RANGE.findall(filename))
    total_letters = len(LETTER_CHARS.findall(filename))
    
    if total_letters == 0:
        return 'en'  # Default to English if no letters
    
    devanagari_ratio = devanagari_chars / total_letters
    return 'ne' if devanagari_ratio >= devanagari_thresh else 'en'


def extract_year_guess(filename: str, year_min: int = 1950, year_max: int = 2050) -> Optional[int]:
    """
    Extract the highest valid year from filename.
    
    Args:
        filename: The filename to analyze
        year_min: Minimum valid year
        year_max: Maximum valid year
        
    Returns:
        Year as integer, or None if not found
    """
    years = [int(match) for match in YEAR_PATTERN.findall(filename)]
    valid_years = [year for year in years if year_min <= year <= year_max]
    return max(valid_years) if valid_years else None


def clean_title_from_filename(filename: str) -> str:
    """
    Generate a clean title from filename using specified logic.
    
    Args:
        filename: The filename to process
        
    Returns:
        Cleaned title string
    """
    # Remove extension
    title = Path(filename).stem
    
    # Replace underscores and hyphens with spaces
    title = re.sub(r'[_-]+', ' ', title)
    
    # Collapse multiple spaces
    title = re.sub(r'\s+', ' ', title).strip()
    
    # Title case except stopwords (only for Latin script)
    if re.match(r'^[a-zA-Z0-9\s\-_.()]+$', title):
        words = title.split()
        title_words = []
        for i, word in enumerate(words):
            if i == 0 or word.lower() not in STOPWORDS:
                title_words.append(word.capitalize())
            else:
                title_words.append(word.lower())
        title = ' '.join(title_words)
    
    return title or filename  # Fallback to original filename


def determine_source_domain(filename: str, is_wikipedia: bool, is_authoritative: bool) -> str:
    """
    Determine source domain based on patterns and flags.
    
    Args:
        filename: The filename to analyze
        is_wikipedia: Whether file is Wikipedia-like
        is_authoritative: Whether file is from authoritative source
        
    Returns:
        Source domain string
    """
    if is_wikipedia:
        return 'wikipedia.org'
    
    if is_authoritative:
        auth_match = PATTERN_AUTH.search(filename)
        if auth_match:
            token = auth_match.group(2).lower()
            if 'who' in token:
                return 'who.int'
            elif 'gov' in token:
                return 'gov.np'
            else:
                return 'gov.np'  # Default for other official sources
    
    return 'unknown'


def process_single_file(
    pdf_file: Path,
    devanagari_thresh: float = 0.25,
    year_min: int = 1950,
    year_max: int = 2050
) -> Tuple[Dict[str, any], Dict[str, bool]]:
    """
    Process a single PDF file and extract metadata.
    
    Uses conservative classification approach due to filename unreliability.
    
    Args:
        pdf_file: Path to the PDF file
        devanagari_thresh: Threshold for Devanagari language detection
        year_min: Minimum valid year
        year_max: Maximum valid year
        
    Returns:
        Tuple of (document_entry, pattern_matches)
    """
    filename = pdf_file.name
    
    # Wikipedia detection (most reliable pattern-based classification)
    is_wikipedia, pattern_matches = is_wikipedia_like(filename)
    
    # Authority decision - more conservative approach
    filename_suggests_authority = bool(PATTERN_AUTH.search(filename))
    
    # Conservative classification logic
    if is_wikipedia:
        # High confidence: explicit Wikipedia patterns
        authority = 'non_authoritative'
        is_distractor = True
        confidence = 'high'
    elif filename_suggests_authority:
        # Medium confidence: filename suggests authority but could be spoofed
        authority = 'authoritative' 
        is_distractor = False
        confidence = 'medium'
    else:
        # Low confidence: default to non-authoritative when uncertain
        # Better to be conservative than to trust potentially fake documents
        authority = 'non_authoritative'
        is_distractor = False
        confidence = 'low'
    
    # Generate document entry
    doc_entry = {
        'doc_id': pdf_file.stem,
        'title_guess': clean_title_from_filename(filename),
        'file': str(pdf_file.relative_to(pdf_file.parent.parent)),  # Relative to project root
        'source_domain_guess': determine_source_domain(filename, is_wikipedia, filename_suggests_authority),
        'language_guess': compute_language_guess(filename, devanagari_thresh),
        'year_guess': extract_year_guess(filename, year_min, year_max),
        'license': 'TBD',
        'authority': authority,
        'is_distractor': is_distractor,
        'classification_confidence': confidence,  # Track confidence level
        'classification_method': 'filename_heuristics',  # Method used
        'requires_content_verification': confidence != 'high',  # Flag for future content analysis
        'sha256': compute_sha256(pdf_file),
        'retrieved_at': datetime.now().isoformat()
    }
    
    return doc_entry, pattern_matches


def print_confusion_table(documents: List[Dict[str, any]]) -> None:
    """Print confusion table: authority × is_distractor."""
    # Count combinations
    counts = {
        ('authoritative', True): 0,
        ('authoritative', False): 0,
        ('non_authoritative', True): 0,
        ('non_authoritative', False): 0
    }
    
    for doc in documents:
        key = (doc['authority'], doc['is_distractor'])
        counts[key] += 1
    
    print("\n📊 Confusion Table (Authority × Distractor):")
    print("=" * 50)
    print(f"{'Authority':<18} | {'Distractor=False':<15} | {'Distractor=True':<15}")
    print("-" * 50)
    print(f"{'authoritative':<18} | {counts[('authoritative', False)]:<15} | {counts[('authoritative', True)]:<15}")
    print(f"{'non_authoritative':<18} | {counts[('non_authoritative', False)]:<15} | {counts[('non_authoritative', True)]:<15}")


def print_pattern_statistics(all_pattern_matches: List[Dict[str, bool]]) -> None:
    """Print token-hit statistics for Wikipedia patterns."""
    # Count pattern hits
    pattern_counts = {
        'wp1_explicit': 0,
        'wp2_wiki': 0,
        'wp3_wp_token': 0,
        'guard_hit': 0,
        'positive_match': 0,
        'is_wikipedia': 0
    }
    
    wp3_only_files = []
    
    for i, matches in enumerate(all_pattern_matches):
        for pattern, hit in matches.items():
            if hit:
                pattern_counts[pattern] += 1
        
        # Check for WP3-only matches (for warning)
        if matches['wp3_wp_token'] and not matches['wp1_explicit'] and not matches['wp2_wiki']:
            wp3_only_files.append(i)
    
    print("\n🔍 Wikipedia Pattern Hit Statistics:")
    print("=" * 50)
    print(f"P_WP1 (explicit wiki*):     {pattern_counts['wp1_explicit']:3d}")
    print(f"P_WP2 (wiki/wikipedia):     {pattern_counts['wp2_wiki']:3d}")
    print(f"P_WP3 (wp token):           {pattern_counts['wp3_wp_token']:3d}")
    print(f"Guard patterns:             {pattern_counts['guard_hit']:3d}")
    print(f"Positive matches:           {pattern_counts['positive_match']:3d}")
    print(f"Final Wikipedia classified: {pattern_counts['is_wikipedia']:3d}")
    
    # Print warnings for WP3-only matches
    if wp3_only_files:
        print(f"\n⚠️  WARNING: {len(wp3_only_files)} file(s) matched P_WP3 ONLY:")
        for file_idx in wp3_only_files:
            print(f"    File #{file_idx + 1} - Guard patterns evaluated but no other Wikipedia patterns matched")


def build_manifest_from_folder(
    input_folder: Path,
    output_file: Path,
    devanagari_thresh: float = 0.25,
    year_min: int = 1950,
    year_max: int = 2050
) -> Dict[str, any]:
    """
    Build manifest from PDF files in a folder using robust heuristics.
    
    Args:
        input_folder: Path to folder containing PDFs
        output_file: Path where to save the manifest YAML
        devanagari_thresh: Threshold for Devanagari language detection
        year_min: Minimum valid year
        year_max: Maximum valid year
        
    Returns:
        Dictionary containing the manifest data
    """
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    
    # Find all PDF files
    pdf_files = list(input_folder.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {input_folder}")
    
    print(f"Found {len(pdf_files)} PDF files in {input_folder}")
    
    # Process files
    documents = []
    all_pattern_matches = []
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            doc_entry, pattern_matches = process_single_file(
                pdf_file, devanagari_thresh, year_min, year_max
            )
            documents.append(doc_entry)
            all_pattern_matches.append(pattern_matches)
            
        except Exception as e:
            print(f"Warning: Failed to process {pdf_file.name}: {e}")
            continue
    
    # Create manifest structure
    manifest = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'source_folder': str(input_folder),
            'total_documents': len(documents),
            'description': 'Seed manifest with robust authority/distractor heuristics',
            'parameters': {
                'devanagari_thresh': devanagari_thresh,
                'year_min': year_min,
                'year_max': year_max
            }
        },
        'documents': documents
    }
    
    # Save manifest
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(manifest, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"✅ Manifest saved to {output_file}")
    
    # Print statistics
    print_summary_tables(manifest)
    print_confusion_table(documents)
    print_pattern_statistics(all_pattern_matches)
    print_confidence_analysis(documents)
    
    return manifest


def print_confidence_analysis(documents: List[Dict[str, any]]) -> None:
    """Print analysis of classification confidence and recommendations."""
    confidence_counts = {'high': 0, 'medium': 0, 'low': 0}
    needs_verification = 0
    
    high_conf_docs = []
    medium_conf_docs = []
    low_conf_docs = []
    
    for doc in documents:
        conf = doc.get('classification_confidence', 'unknown')
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        
        if doc.get('requires_content_verification', False):
            needs_verification += 1
            
        if conf == 'high':
            high_conf_docs.append(doc['doc_id'])
        elif conf == 'medium':
            medium_conf_docs.append(doc['doc_id'])
        elif conf == 'low':
            low_conf_docs.append(doc['doc_id'])
    
    print("\n🎯 Classification Confidence Analysis:")
    print("=" * 50)
    print(f"High confidence (Wikipedia patterns):    {confidence_counts.get('high', 0):3d} documents")
    print(f"Medium confidence (Authority patterns):  {confidence_counts.get('medium', 0):3d} documents")
    print(f"Low confidence (Default classification): {confidence_counts.get('low', 0):3d} documents")
    print(f"Requires content verification:           {needs_verification:3d} documents")
    
    if medium_conf_docs:
        print(f"\n⚠️  Medium confidence documents (filename-based authority):")
        for doc_id in medium_conf_docs[:5]:  # Show first 5
            print(f"   - {doc_id}")
        if len(medium_conf_docs) > 5:
            print(f"   ... and {len(medium_conf_docs) - 5} more")
    
    if low_conf_docs:
        print(f"\n❓ Low confidence documents (uncertain classification):")
        for doc_id in low_conf_docs[:5]:  # Show first 5
            print(f"   - {doc_id}")
        if len(low_conf_docs) > 5:
            print(f"   ... and {len(low_conf_docs) - 5} more")
    
    print(f"\n💡 Recommendations:")
    print(f"   1. Implement PDF content analysis for {needs_verification} uncertain documents")
    print(f"   2. Extract PDF metadata (author, creator, subject) for verification")
    print(f"   3. Look for official signatures, letterheads, and formatting patterns")
    print(f"   4. Consider manual review for medium/low confidence authoritative claims")


def print_summary_tables(manifest: Dict[str, any]) -> None:
    """Print summary tables for the manifest."""
    documents = manifest['documents']
    
    # Count by authority
    authority_counts = {}
    for doc in documents:
        authority = doc['authority']
        authority_counts[authority] = authority_counts.get(authority, 0) + 1
    
    print("\n📊 Summary by Authority:")
    print("=" * 40)
    for authority, count in sorted(authority_counts.items()):
        print(f"{authority:20} | {count:3d} documents")
    
    # Count by language
    language_counts = {}
    for doc in documents:
        language = doc['language_guess']
        language_counts[language] = language_counts.get(language, 0) + 1
    
    print("\n🌐 Summary by Language:")
    print("=" * 40)
    for language, count in sorted(language_counts.items()):
        lang_name = {'en': 'English', 'ne': 'Nepali'}.get(language, language)
        print(f"{lang_name:20} | {count:3d} documents")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build YAML manifest from PDF files with robust authority/distractor detection"
    )
    parser.add_argument(
        "--in", 
        dest="input_folder",
        type=Path,
        required=True,
        help="Input folder containing PDF files"
    )
    parser.add_argument(
        "--out",
        dest="output_file", 
        type=Path,
        required=True,
        help="Output YAML manifest file"
    )
    parser.add_argument(
        "--devanagari-thresh",
        type=float,
        default=0.25,
        help="Devanagari character ratio threshold for Nepali classification (default: 0.25)"
    )
    parser.add_argument(
        "--year-min",
        type=int,
        default=1950,
        help="Minimum valid year (default: 1950)"
    )
    parser.add_argument(
        "--year-max",
        type=int,
        default=2050,
        help="Maximum valid year (default: 2050)"
    )
    
    args = parser.parse_args()
    
    try:
        # Build manifest
        manifest = build_manifest_from_folder(
            args.input_folder, 
            args.output_file,
            args.devanagari_thresh,
            args.year_min,
            args.year_max
        )
        
        print(f"\n✅ Processed {manifest['metadata']['total_documents']} documents")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()