#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import sys
import os

# Add src directory to path for ingest module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from ingest.pdf_to_text import process_pdf_to_chunks
    print('✅ Using project PDF processor')
except ImportError:
    print('❌ Project PDF processor not available')
    sys.exit(1)

# Process the latest document
latest_doc_path = Path('data/raw_pdfs/user_upload_1758214803373.pdf')
corpus_path = Path('data/real_corpus.parquet')

if latest_doc_path.exists():
    print(f'Processing: {latest_doc_path}')
    chunks = process_pdf_to_chunks(latest_doc_path)
    
    if chunks:
        print(f'Created {len(chunks)} chunks')
        new_df = pd.DataFrame(chunks)
        
        # Load existing corpus
        if corpus_path.exists():
            existing_df = pd.read_parquet(corpus_path)
            print(f'Existing corpus: {len(existing_df)} chunks')
            # Remove old versions of this document
            existing_df = existing_df[~existing_df['doc_id'].str.contains('user_upload_175821', na=False)]
            # Append new chunks
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            updated_df = new_df
        
        # Save updated corpus
        updated_df.to_parquet(corpus_path, index=False)
        print(f'✅ Updated corpus saved with {len(updated_df)} total chunks')
        print(f'✅ Latest CV chunks in corpus: {len(new_df)}')
    else:
        print('❌ No chunks generated from latest CV.')
else:
    print(f'❌ Latest document not found: {latest_doc_path}')
