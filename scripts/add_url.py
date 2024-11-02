#!/usr/bin/env python3
"""
URL-based document ingestion for NepaliGov-RAG-Bench

Downloads documents from URLs and adds them to the manifest with proper metadata.
Implements idempotent operations and SHA-256 based change detection.
"""

import argparse
import hashlib
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import urllib.request
import urllib.error
import yaml
import shutil


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def download_pdf(url: str, output_path: Path) -> bool:
    """
    Download PDF from URL to output path.
    
    Args:
        url: URL to download from
        output_path: Local path to save to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading from {url}...")
        
        # Create a temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_path = Path(temp_file.name)
        
        # Download to temporary file
        urllib.request.urlretrieve(url, temp_path)
        
        # Verify it's a PDF (basic check)
        with open(temp_path, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                print(f"❌ Downloaded file is not a valid PDF")
                temp_path.unlink()
                return False
        
        # Move to final location
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temp_path), str(output_path))
        print(f"✅ Downloaded to {output_path}")
        return True
        
    except urllib.error.URLError as e:
        print(f"❌ Download failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during download: {e}")
        return False


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load existing manifest or create new one."""
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        # Create new manifest structure
        return {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'source_folder': 'data/raw_pdfs',
                'total_documents': 0,
                'description': 'Seed manifest with robust authority/distractor heuristics',
                'parameters': {
                    'devanagari_thresh': 0.25,
                    'year_min': 1950,
                    'year_max': 2050
                }
            },
            'documents': []
        }


def save_manifest(manifest: Dict[str, Any], manifest_path: Path) -> None:
    """Save manifest to file."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        yaml.dump(manifest, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def find_document_by_id(manifest: Dict[str, Any], doc_id: str) -> Optional[Dict[str, Any]]:
    """Find document in manifest by doc_id."""
    for doc in manifest.get('documents', []):
        if doc['doc_id'] == doc_id:
            return doc
    return None


def add_url_to_manifest(
    url: str,
    doc_id: str,
    title: str,
    language: str,
    authority: str,
    is_distractor: bool,
    manifest_path: Path,
    raw_pdfs_dir: Path
) -> bool:
    """
    Add URL-based document to manifest with idempotent operations.
    
    Args:
        url: Source URL
        doc_id: Document identifier
        title: Human-readable title
        language: Language code (en|ne)
        authority: Authority level (authoritative|non_authoritative)
        is_distractor: Whether this is a distractor document
        manifest_path: Path to manifest file
        raw_pdfs_dir: Directory for raw PDF files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load existing manifest
        manifest = load_manifest(manifest_path)
        
        # Check if document already exists
        existing_doc = find_document_by_id(manifest, doc_id)
        
        # Define target file path
        pdf_filename = f"{doc_id}.pdf"
        pdf_path = raw_pdfs_dir / pdf_filename
        
        # Download the PDF to a temporary location first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_path = Path(temp_file.name)
        
        if not download_pdf(url, temp_path):
            return False
        
        # Compute SHA-256 of downloaded file
        new_sha256 = compute_sha256(temp_path)
        
        # Check if we need to update
        should_update = True
        if existing_doc:
            existing_sha256 = existing_doc.get('sha256')
            if existing_sha256 == new_sha256:
                print(f"✅ Document {doc_id} is up to date (SHA-256 matches)")
                temp_path.unlink()  # Clean up temp file
                return True
            else:
                print(f"📝 Document {doc_id} has changed (SHA-256: {existing_sha256} → {new_sha256})")
        else:
            print(f"🆕 Adding new document {doc_id}")
        
        # Check if target file exists and has different SHA
        if pdf_path.exists():
            existing_file_sha = compute_sha256(pdf_path)
            if existing_file_sha != new_sha256:
                # Create backup with timestamp
                backup_path = pdf_path.with_suffix(f'.{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf.bak')
                shutil.copy2(pdf_path, backup_path)
                print(f"📁 Backed up existing file to {backup_path}")
        
        # Move downloaded file to final location
        shutil.move(str(temp_path), str(pdf_path))
        
        # Create document entry
        doc_entry = {
            'doc_id': doc_id,
            'title_guess': title,
            'file': f"raw_pdfs/{pdf_filename}",
            'source_domain_guess': 'url_provided',
            'language_guess': language,
            'year_guess': None,  # Could be extracted from content later
            'license': 'TBD',
            'authority': authority,
            'is_distractor': is_distractor,
            'classification_confidence': 'high',  # User-provided metadata is high confidence
            'classification_method': 'user_provided',
            'requires_content_verification': False,
            'sha256': new_sha256,
            'retrieved_at': datetime.now().isoformat(),
            'source_url': url
        }
        
        # Update manifest
        if existing_doc:
            # Update existing document
            for i, doc in enumerate(manifest['documents']):
                if doc['doc_id'] == doc_id:
                    manifest['documents'][i] = doc_entry
                    break
        else:
            # Add new document
            manifest['documents'].append(doc_entry)
            manifest['metadata']['total_documents'] = len(manifest['documents'])
        
        # Update manifest metadata
        manifest['metadata']['last_updated'] = datetime.now().isoformat()
        
        # Save updated manifest
        save_manifest(manifest, manifest_path)
        
        print(f"✅ Successfully added/updated {doc_id} in manifest")
        return True
        
    except Exception as e:
        print(f"❌ Error adding URL to manifest: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Add documents by URL to NepaliGov-RAG-Bench manifest"
    )
    parser.add_argument(
        "--url",
        required=True,
        help="URL to download PDF from"
    )
    parser.add_argument(
        "--doc-id",
        required=True,
        help="Document identifier (used as filename without extension)"
    )
    parser.add_argument(
        "--title",
        required=True,
        help="Human-readable document title"
    )
    parser.add_argument(
        "--language",
        choices=['en', 'ne'],
        required=True,
        help="Document language (en|ne)"
    )
    parser.add_argument(
        "--authority",
        choices=['authoritative', 'non_authoritative'],
        required=True,
        help="Authority level"
    )
    parser.add_argument(
        "--is-distractor",
        type=str,
        choices=['true', 'false'],
        required=True,
        help="Whether this is a distractor document"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/seed_manifest.yaml"),
        help="Path to manifest file (default: data/seed_manifest.yaml)"
    )
    parser.add_argument(
        "--raw-pdfs-dir",
        type=Path,
        default=Path("data/raw_pdfs"),
        help="Directory for raw PDF files (default: data/raw_pdfs)"
    )
    
    args = parser.parse_args()
    
    # Convert string boolean to actual boolean
    is_distractor = args.is_distractor.lower() == 'true'
    
    # Validate inputs
    if not args.url.startswith(('http://', 'https://')):
        print("❌ URL must start with http:// or https://")
        sys.exit(1)
    
    # Ensure raw PDFs directory exists
    args.raw_pdfs_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        success = add_url_to_manifest(
            url=args.url,
            doc_id=args.doc_id,
            title=args.title,
            language=args.language,
            authority=args.authority,
            is_distractor=is_distractor,
            manifest_path=args.manifest,
            raw_pdfs_dir=args.raw_pdfs_dir
        )
        
        if success:
            print(f"\n✅ Document {args.doc_id} successfully added to manifest")
            sys.exit(0)
        else:
            print(f"\n❌ Failed to add document {args.doc_id}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
