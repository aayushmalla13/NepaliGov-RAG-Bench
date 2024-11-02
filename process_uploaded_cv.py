#!/usr/bin/env python3
"""
Simple script to process uploaded CV and add it to the corpus
"""

import pandas as pd
import PyPDF2
import re
from pathlib import Path

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
    except Exception as e:
        print(f"Error extracting text: {e}")
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        chunks.append(chunk_text)
    
    return chunks

def process_cv(cv_path, doc_id):
    """Process CV and return DataFrame."""
    print(f"Processing: {cv_path}")
    
    # Extract text
    text = extract_text_from_pdf(cv_path)
    if not text.strip():
        print("No text extracted from PDF")
        return None
    
    # Chunk text
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")
    
    # Create DataFrame
    data = []
    for i, chunk in enumerate(chunks):
        data.append({
            'doc_id': doc_id,
            'chunk_id': f"{doc_id}_chunk_{i}",
            'text': chunk,
            'page_num': i + 1,  # Approximate page
            'chunk_index': i
        })
    
    return pd.DataFrame(data)

def main():
    # Process the latest uploaded CV
    cv_path = Path("data/raw_pdfs/user_upload_1758209162256.pdf")
    doc_id = "user_upload_1758209162256"
    
    if not cv_path.exists():
        print(f"CV file not found: {cv_path}")
        return
    
    # Process CV
    cv_df = process_cv(cv_path, doc_id)
    if cv_df is None:
        return
    
    # Load existing corpus
    corpus_path = Path("data/real_corpus.parquet")
    if corpus_path.exists():
        print("Loading existing corpus...")
        existing_df = pd.read_parquet(corpus_path)
        print(f"Existing corpus: {len(existing_df)} chunks")
        
        # Combine with new CV data
        combined_df = pd.concat([existing_df, cv_df], ignore_index=True)
        print(f"Combined corpus: {len(combined_df)} chunks")
    else:
        print("No existing corpus found, creating new one...")
        combined_df = cv_df
    
    # Save updated corpus
    combined_df.to_parquet(corpus_path, index=False)
    print(f"✅ Updated corpus saved to {corpus_path}")
    
    # Check if CV is now in corpus
    cv_chunks = combined_df[combined_df['doc_id'] == doc_id]
    print(f"✅ CV chunks in corpus: {len(cv_chunks)}")

if __name__ == "__main__":
    main()
