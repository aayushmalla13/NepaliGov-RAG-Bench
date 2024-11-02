#!/usr/bin/env python3
"""
Create a REAL corpus from all the Nepal government PDFs

This will properly extract content from all PDFs and create meaningful chunks
that can actually answer questions about Nepal's constitution, health policies, etc.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

import pandas as pd
import fitz  # PyMuPDF
import re
from typing import List, Dict, Any
import json

def extract_text_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract text from PDF with proper chunking."""
    
    try:
        doc = fitz.open(pdf_path)
        chunks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if not text.strip():
                continue
            
            # Clean text
            text = re.sub(r'\s+', ' ', text.strip())
            
            # Skip very short pages (likely headers/footers only)
            if len(text) < 50:
                continue
            
            # Create meaningful chunks (paragraphs or sections)
            # Split by double newlines, periods followed by newlines, or section headers
            potential_chunks = re.split(r'(?:\n\s*\n|\. *\n|(?:\n(?=\d+\.|\n[A-Z][A-Z\s]+\n)))', text)
            
            for i, chunk_text in enumerate(potential_chunks):
                chunk_text = chunk_text.strip()
                
                # Skip very short chunks
                if len(chunk_text) < 100:
                    continue
                
                # Skip chunks that are mostly numbers/dates/references
                if len(re.findall(r'\d', chunk_text)) / len(chunk_text) > 0.3:
                    continue
                
                chunk = {
                    'doc_id': pdf_path.stem,
                    'page_num': page_num + 1,
                    'chunk_id': f"{pdf_path.stem}_p{page_num+1}_c{i+1}",
                    'text': chunk_text,
                    'char_count': len(chunk_text),
                    'is_authoritative': True,  # All government PDFs are authoritative
                    'source_type': 'government_pdf',
                    'language': detect_language(chunk_text)
                }
                chunks.append(chunk)
        
        doc.close()
        return chunks
        
    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")
        return []

def detect_language(text: str) -> str:
    """Simple language detection for English vs Nepali."""
    
    # Count Devanagari characters (Nepali)
    devanagari_count = len(re.findall(r'[\u0900-\u097F]', text))
    
    # Count English letters
    english_count = len(re.findall(r'[a-zA-Z]', text))
    
    if devanagari_count > english_count:
        return 'ne'
    elif english_count > 0:
        return 'en'
    else:
        return 'unknown'

def create_real_corpus():
    """Create a real corpus from all PDFs."""
    
    print("🏛️ Creating REAL Nepal Government Corpus")
    print("=" * 60)
    
    pdf_dir = Path("data/raw_pdfs")
    if not pdf_dir.exists():
        print("❌ PDF directory not found!")
        return False
    
    all_chunks = []
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    print(f"📚 Found {len(pdf_files)} PDF files")
    print()
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"🔍 Processing {i}/{len(pdf_files)}: {pdf_path.name}")
        
        # Skip the unconfirmed download
        if 'Unconfirmed' in pdf_path.name:
            print("   ⏭️ Skipping unconfirmed download")
            continue
        
        chunks = extract_text_from_pdf(pdf_path)
        
        if chunks:
            print(f"   ✅ Extracted {len(chunks)} chunks")
            
            # Show sample for important documents
            if any(keyword in pdf_path.stem.lower() for keyword in ['constitution', 'fundamental', 'rights']):
                print("   📄 Sample content:")
                for chunk in chunks[:2]:
                    print(f"      {chunk['text'][:150]}...")
                print()
            
            all_chunks.extend(chunks)
        else:
            print(f"   ❌ No content extracted")
    
    if not all_chunks:
        print("❌ No chunks created!")
        return False
    
    print(f"✅ Total chunks created: {len(all_chunks)}")
    print()
    
    # Create DataFrame
    df = pd.DataFrame(all_chunks)
    
    # Add required columns for compatibility with existing system
    df['page_id'] = df['doc_id'] + '_page_' + df['page_num'].astype(str).str.zfill(3)
    df['block_id'] = df['chunk_id']
    df['block_type'] = 'text'
    df['char_span'] = df.apply(lambda x: [0, x['char_count']], axis=1)
    df['bbox'] = [[0, 0, 100, 100]] * len(df)  # Dummy bbox
    df['ocr_engine'] = 'pymupdf'
    df['conf_mean'] = 0.95
    df['conf_min'] = 0.9
    df['conf_max'] = 1.0
    df['tokens'] = df['text'].str.split().str.len()
    df['source_authority'] = df['doc_id']
    df['is_distractor'] = False
    df['source_page_is_ocr'] = False
    df['watermark_flags'] = False
    df['font_stats'] = [{'dummy': 0}] * len(df)  # Add dummy field for parquet
    df['pdf_meta'] = [{'dummy': 0}] * len(df)  # Add dummy field for parquet
    
    # Statistics
    print("📊 Corpus Statistics:")
    print(f"   Total chunks: {len(df)}")
    print(f"   English chunks: {len(df[df['language'] == 'en'])}")
    print(f"   Nepali chunks: {len(df[df['language'] == 'ne'])}")
    print(f"   Average chunk length: {df['char_count'].mean():.0f} characters")
    print()
    
    # Show content by document type
    print("📄 Content by Document:")
    doc_stats = df.groupby('doc_id').agg({
        'chunk_id': 'count',
        'char_count': 'sum',
        'language': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown'
    }).round(0)
    
    for doc_id, stats in doc_stats.iterrows():
        print(f"   {doc_id}: {stats['chunk_id']} chunks, {stats['char_count']:,.0f} chars, {stats['language']}")
    
    print()
    
    # Save corpus
    output_path = Path("data/real_corpus.parquet")
    df.to_parquet(output_path, index=False)
    print(f"✅ Saved corpus to: {output_path}")
    
    # Test search
    print("\n🔍 Testing search on new corpus...")
    test_queries = [
        "fundamental rights Nepal constitution",
        "health services Nepal",
        "COVID-19 response policy",
        "emergency health services"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        # Simple text search
        query_lower = query.lower()
        matches = df[df['text'].str.lower().str.contains('|'.join(query_lower.split()), na=False)]
        
        print(f"   Found {len(matches)} matching chunks")
        
        if len(matches) > 0:
            # Show best match
            best_match = matches.iloc[0]
            print(f"   Best match from: {best_match['doc_id']}")
            print(f"   Text: {best_match['text'][:200]}...")
    
    print(f"\n🎉 Real corpus created successfully!")
    print(f"📍 Location: {output_path}")
    print(f"📊 Total: {len(df)} chunks from {len(df['doc_id'].unique())} documents")
    
    return True

if __name__ == "__main__":
    
    # Check if PyMuPDF is available
    try:
        import fitz
    except ImportError:
        print("❌ PyMuPDF (fitz) not available. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "PyMuPDF"], check=True)
        import fitz
    
    success = create_real_corpus()
    
    if success:
        print("\n✅ SUCCESS: Real corpus created with actual PDF content!")
        print("🚀 Now the system can answer real questions about Nepal's constitution and policies!")
    else:
        print("\n❌ FAILED: Could not create corpus")
