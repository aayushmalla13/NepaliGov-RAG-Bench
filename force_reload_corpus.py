#!/usr/bin/env python3
"""
Force reload the corpus to ensure latest documents are loaded
"""
import pandas as pd
import sys
import os

def reload_corpus():
    print("🔄 Force reloading corpus...")
    
    # Load the corpus
    corpus_path = "data/real_corpus.parquet"
    if os.path.exists(corpus_path):
        df = pd.read_parquet(corpus_path)
        print(f"✅ Loaded corpus with {len(df)} chunks")
        
        # Check for the target document
        target_doc = "user_upload_1758214803373"
        chunks = df[df['doc_id'] == target_doc]
        print(f"📄 Document {target_doc}: {len(chunks)} chunks")
        
        if len(chunks) > 0:
            print("✅ Document is in corpus")
            print("Sample content:", chunks.iloc[0]['text'][:100] + "...")
        else:
            print("❌ Document not found in corpus")
            
        # Show all user uploads
        user_uploads = df[df['doc_id'].str.contains('user_upload', na=False)]
        print(f"📋 All user uploads: {len(user_uploads)} total chunks")
        for doc_id in user_uploads['doc_id'].unique():
            count = len(user_uploads[user_uploads['doc_id'] == doc_id])
            print(f"  {doc_id}: {count} chunks")
    else:
        print(f"❌ Corpus file not found: {corpus_path}")

if __name__ == "__main__":
    reload_corpus()
