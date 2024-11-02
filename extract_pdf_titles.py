#!/usr/bin/env python3
"""
🇳🇵 PDF Title Extractor

Extracts proper titles from PDF content instead of using filenames.
"""

import pandas as pd
import re
from pathlib import Path
from typing import Dict, Optional

def extract_title_from_content(text: str, doc_id: str) -> str:
    """
    Extract the actual document title from PDF content.
    Uses multiple heuristics to find the real title.
    """
    
    # Clean the text
    lines = text.split('\n')
    clean_lines = [line.strip() for line in lines if line.strip()]
    
    # Method 1: Look for title patterns in first few lines
    for i, line in enumerate(clean_lines[:10]):  # Check first 10 lines
        line = line.strip()
        
        # Skip very short lines or lines with mostly numbers/symbols
        if len(line) < 10 or len(re.findall(r'[a-zA-Z]', line)) < 5:
            continue
            
        # Skip lines that look like headers/footers
        if any(skip in line.lower() for skip in ['page', 'chapter', 'section', 'article', 'www.', 'http', '@']):
            continue
            
        # Look for title-like patterns
        if any(keyword in line.lower() for keyword in [
            'constitution', 'act', 'policy', 'report', 'plan', 'strategy',
            'framework', 'guideline', 'manual', 'handbook', 'protocol'
        ]):
            # Clean up the title
            title = re.sub(r'\s+', ' ', line)
            title = re.sub(r'[^\w\s\-\(\),]', '', title)  # Remove special chars except basic punctuation
            if 15 <= len(title) <= 100:  # Reasonable title length
                return title.title()
    
    # Method 2: Look for document metadata patterns
    full_text = ' '.join(clean_lines[:20])  # First 20 lines
    
    # Constitution patterns
    if 'constitution' in full_text.lower():
        if 'nepal' in full_text.lower():
            if '2072' in full_text or '2015' in full_text:
                return "Constitution of Nepal 2015"
            else:
                return "Constitution of Nepal"
    
    # Health report patterns  
    if any(term in full_text.lower() for term in ['health report', 'annual health']):
        if 'nepal' in full_text.lower():
            # Look for year
            years = re.findall(r'20\d{2}', full_text)
            if years:
                return f"Annual Health Report Nepal {years[0]}"
            else:
                return "Annual Health Report Nepal"
    
    # WHO/Health patterns
    if 'who' in full_text.lower() or 'world health' in full_text.lower():
        if 'nepal' in full_text.lower():
            if 'special initiative' in full_text.lower():
                return "WHO Special Initiative Country Report - Nepal"
            elif 'covid' in full_text.lower() or 'pandemic' in full_text.lower():
                return "WHO COVID-19 Response Report - Nepal"
            else:
                return "WHO Health Report - Nepal"
    
    # COVID patterns
    if 'covid' in full_text.lower() or 'coronavirus' in full_text.lower():
        if 'nepal' in full_text.lower():
            if 'pandemic' in full_text.lower():
                return "COVID-19 Pandemic Response in Nepal"
            elif 'vaccination' in full_text.lower():
                return "COVID-19 Vaccination Program in Nepal"
            else:
                return "COVID-19 Response in Nepal"
    
    # Emergency/HEOC patterns
    if any(term in full_text.lower() for term in ['heoc', 'emergency', 'disaster']):
        if 'nepal' in full_text.lower():
            return "Health Emergency Operations Center Report - Nepal"
    
    # Ministry patterns
    if 'ministry' in full_text.lower():
        if 'health' in full_text.lower():
            return "Ministry of Health and Population - Nepal"
        elif 'government' in full_text.lower():
            return "Government Ministry Report - Nepal"
    
    # Treatment/Service patterns
    if any(term in full_text.lower() for term in ['treatment protocol', 'service package', 'health service']):
        return "Health Service Treatment Protocol - Nepal"
    
    # Blue Book pattern
    if 'blue book' in full_text.lower():
        years = re.findall(r'20\d{2}', full_text)
        if years:
            return f"Blue Book Nepal {years[0]}"
        else:
            return "Blue Book Nepal"
    
    # Act patterns
    if ' act' in full_text.lower():
        if 'public health' in full_text.lower():
            return "Public Health Service Act - Nepal"
        elif 'disaster' in full_text.lower() or 'drm' in full_text.lower():
            return "Disaster Risk Management Act - Nepal"
    
    # Method 3: Try to extract from first substantial line
    for line in clean_lines[:15]:
        line = line.strip()
        if 20 <= len(line) <= 80:  # Good title length
            # Remove common prefixes/suffixes
            line = re.sub(r'^(the|a|an)\s+', '', line, flags=re.IGNORECASE)
            line = re.sub(r'\s+(report|document|manual)$', '', line, flags=re.IGNORECASE)
            
            if len(line) >= 15:
                return line.title()
    
    # Method 4: Fallback to cleaned filename
    return generate_fallback_title(doc_id)

def generate_fallback_title(doc_id: str) -> str:
    """Generate a readable title from filename as fallback."""
    title = doc_id.replace('_', ' ').replace('-', ' ')
    title = re.sub(r'\.pdf$', '', title, flags=re.IGNORECASE)
    title = re.sub(r'^[0-9]+\s*', '', title)  # Remove leading numbers
    title = re.sub(r'\s+', ' ', title)  # Clean whitespace
    
    # Capitalize properly
    words = title.split()
    capitalized_words = []
    for word in words:
        if word.lower() in ['of', 'and', 'the', 'in', 'for', 'to', 'a', 'an']:
            capitalized_words.append(word.lower())
        else:
            capitalized_words.append(word.capitalize())
    
    result = ' '.join(capitalized_words)
    
    # Add "Nepal" if not present and it looks like a government document
    if 'nepal' not in result.lower() and any(keyword in result.lower() for keyword in [
        'constitution', 'health', 'government', 'ministry', 'act', 'policy'
    ]):
        result += " - Nepal"
    
    return result

def create_title_mapping(corpus_df: pd.DataFrame) -> Dict[str, str]:
    """Create a mapping of doc_id to proper titles."""
    
    print("🔍 Extracting titles from PDF content...")
    
    title_mapping = {}
    doc_groups = corpus_df.groupby('doc_id')
    
    for doc_id, group in doc_groups:
        print(f"📄 Processing: {doc_id}")
        
        # Get the first chunk with substantial text (likely contains title)
        best_chunk = None
        for _, row in group.iterrows():
            text = str(row['text'])
            if len(text) > 100:  # Substantial content
                best_chunk = text
                break
        
        if best_chunk:
            title = extract_title_from_content(best_chunk, doc_id)
            title_mapping[doc_id] = title
            print(f"✅ Title: {title}")
        else:
            title = generate_fallback_title(doc_id)
            title_mapping[doc_id] = title
            print(f"⚠️  Fallback: {title}")
    
    return title_mapping

def save_title_mapping(title_mapping: Dict[str, str], output_path: str = "data/pdf_titles.json"):
    """Save title mapping to JSON file."""
    import json
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(title_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved title mapping to: {output_path}")

def load_title_mapping(file_path: str = "data/pdf_titles.json") -> Dict[str, str]:
    """Load title mapping from JSON file."""
    import json
    
    if not Path(file_path).exists():
        return {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def test_title_extraction():
    """Test the title extraction on our corpus."""
    
    print("🇳🇵 PDF Title Extraction Test")
    print("=" * 50)
    
    # Load corpus
    corpus_path = Path("data/real_corpus.parquet")
    if not corpus_path.exists():
        print("❌ Corpus not found!")
        return
    
    df = pd.read_parquet(corpus_path)
    print(f"✅ Loaded {len(df)} chunks from {len(df['doc_id'].unique())} documents")
    print()
    
    # Extract titles
    title_mapping = create_title_mapping(df)
    
    print("\n" + "=" * 50)
    print("📚 EXTRACTED TITLES:")
    print("=" * 50)
    
    for doc_id, title in title_mapping.items():
        print(f"📄 {doc_id}")
        print(f"   → {title}")
        print()
    
    # Save mapping
    save_title_mapping(title_mapping)
    
    print("✅ Title extraction complete!")
    return title_mapping

if __name__ == "__main__":
    test_title_extraction()


