#!/usr/bin/env python3
"""
Lineage Index Management for NepaliGov-RAG-Bench

Maintains file→doc_id→shard lineage tracking and configuration checksums
for incremental re-indexing operations.
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple


class LineageIndex:
    """Manages document lineage and configuration tracking."""
    
    def __init__(self, index_path: Path = Path("ops/lineage_index.json")):
        """
        Initialize lineage index.
        
        Args:
            index_path: Path to lineage index JSON file
        """
        self.index_path = index_path
        self.data = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """Load existing index or create new one."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load lineage index: {e}")
                print("Creating new lineage index")
        
        # Create new index structure
        return {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            },
            "documents": {},  # doc_id -> {file_path, sha256, shard_ids, last_processed, status}
            "shards": {},     # shard_id -> {doc_ids, nvecs, last_built, index_path}
            "config_checksums": {
                "cp6_synonyms.json": None,
                "cp9_mappings.json": None, 
                "cp10_thresholds.yaml": None,
                "cp11_5_ui_translation.yaml": None
            },
            "faiss_index": {
                "type": "single",  # "single" or "sharded"
                "shards": ["default"],  # List of shard IDs
                "total_nvecs": 0,
                "last_built": None,
                "index_manifest_path": None
            }
        }
    
    def save_index(self) -> None:
        """Save index to file."""
        self.data["metadata"]["last_updated"] = datetime.now().isoformat()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def compute_file_checksum(self, file_path: Path) -> Optional[str]:
        """Compute SHA-256 checksum of a file."""
        if not file_path.exists():
            return None
        
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except IOError:
            return None
    
    def update_document(
        self, 
        doc_id: str, 
        file_path: Path, 
        sha256: str, 
        shard_id: str = "default",
        status: str = "pending"
    ) -> None:
        """
        Update document information in lineage index.
        
        Args:
            doc_id: Document identifier
            file_path: Path to source PDF file
            sha256: SHA-256 hash of the file
            shard_id: Shard identifier (default: "default")
            status: Processing status ("pending", "processing", "completed", "failed")
        """
        self.data["documents"][doc_id] = {
            "file_path": str(file_path),
            "sha256": sha256,
            "shard_ids": [shard_id],
            "last_processed": datetime.now().isoformat() if status == "completed" else None,
            "status": status,
            "added_at": self.data["documents"].get(doc_id, {}).get("added_at", datetime.now().isoformat())
        }
        
        # Update shard information
        if shard_id not in self.data["shards"]:
            self.data["shards"][shard_id] = {
                "doc_ids": [],
                "nvecs": 0,
                "last_built": None,
                "index_path": None
            }
        
        if doc_id not in self.data["shards"][shard_id]["doc_ids"]:
            self.data["shards"][shard_id]["doc_ids"].append(doc_id)
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document information."""
        return self.data["documents"].get(doc_id)
    
    def get_shard(self, shard_id: str) -> Optional[Dict[str, Any]]:
        """Get shard information."""
        return self.data["shards"].get(shard_id)
    
    def get_changed_documents(self, manifest_data: Dict[str, Any]) -> List[Tuple[str, str, str]]:
        """
        Find documents that have changed since last processing.
        
        Args:
            manifest_data: Loaded manifest data
            
        Returns:
            List of (doc_id, old_sha256, new_sha256) tuples for changed documents
        """
        changed = []
        
        for doc in manifest_data.get("documents", []):
            doc_id = doc["doc_id"]
            new_sha256 = doc["sha256"]
            
            existing_doc = self.get_document(doc_id)
            if existing_doc is None:
                # New document
                changed.append((doc_id, None, new_sha256))
            elif existing_doc["sha256"] != new_sha256:
                # Changed document
                changed.append((doc_id, existing_doc["sha256"], new_sha256))
        
        return changed
    
    def get_documents_by_shard(self, shard_id: str) -> List[str]:
        """Get list of document IDs in a shard."""
        shard = self.get_shard(shard_id)
        return shard["doc_ids"] if shard else []
    
    def get_shards_for_document(self, doc_id: str) -> List[str]:
        """Get list of shard IDs containing a document."""
        doc = self.get_document(doc_id)
        return doc["shard_ids"] if doc else []
    
    def update_shard_info(
        self, 
        shard_id: str, 
        nvecs: int, 
        index_path: Optional[Path] = None
    ) -> None:
        """
        Update shard build information.
        
        Args:
            shard_id: Shard identifier
            nvecs: Number of vectors in shard
            index_path: Path to shard index file
        """
        if shard_id not in self.data["shards"]:
            self.data["shards"][shard_id] = {
                "doc_ids": [],
                "nvecs": 0,
                "last_built": None,
                "index_path": None
            }
        
        self.data["shards"][shard_id]["nvecs"] = nvecs
        self.data["shards"][shard_id]["last_built"] = datetime.now().isoformat()
        if index_path:
            self.data["shards"][shard_id]["index_path"] = str(index_path)
    
    def update_config_checksum(self, config_name: str, checksum: str) -> bool:
        """
        Update configuration file checksum.
        
        Args:
            config_name: Name of config file
            checksum: New checksum
            
        Returns:
            True if checksum changed, False if same
        """
        old_checksum = self.data["config_checksums"].get(config_name)
        self.data["config_checksums"][config_name] = checksum
        return old_checksum != checksum
    
    def get_config_checksum(self, config_name: str) -> Optional[str]:
        """Get stored checksum for a config file."""
        return self.data["config_checksums"].get(config_name)
    
    def detect_config_changes(self, config_paths: Dict[str, Path]) -> List[Tuple[str, str, str]]:
        """
        Detect configuration file changes.
        
        Args:
            config_paths: Mapping of config_name -> file_path
            
        Returns:
            List of (config_name, old_checksum, new_checksum) tuples
        """
        changes = []
        
        for config_name, config_path in config_paths.items():
            new_checksum = self.compute_file_checksum(config_path)
            old_checksum = self.get_config_checksum(config_name)
            
            if new_checksum != old_checksum:
                changes.append((config_name, old_checksum, new_checksum))
                self.update_config_checksum(config_name, new_checksum)
        
        return changes
    
    def update_faiss_index_info(
        self, 
        index_type: str, 
        shards: List[str], 
        total_nvecs: int, 
        manifest_path: Path
    ) -> None:
        """
        Update FAISS index information.
        
        Args:
            index_type: "single" or "sharded"
            shards: List of shard IDs
            total_nvecs: Total number of vectors
            manifest_path: Path to index manifest
        """
        self.data["faiss_index"] = {
            "type": index_type,
            "shards": shards,
            "total_nvecs": total_nvecs,
            "last_built": datetime.now().isoformat(),
            "index_manifest_path": str(manifest_path)
        }
    
    def get_affected_shards(self, changed_doc_ids: List[str]) -> Set[str]:
        """
        Get shards affected by document changes.
        
        Args:
            changed_doc_ids: List of changed document IDs
            
        Returns:
            Set of affected shard IDs
        """
        affected_shards = set()
        
        for doc_id in changed_doc_ids:
            shard_ids = self.get_shards_for_document(doc_id)
            affected_shards.update(shard_ids)
        
        return affected_shards
    
    def mark_documents_status(self, doc_ids: List[str], status: str) -> None:
        """Mark documents with given status."""
        for doc_id in doc_ids:
            if doc_id in self.data["documents"]:
                self.data["documents"][doc_id]["status"] = status
                if status == "completed":
                    self.data["documents"][doc_id]["last_processed"] = datetime.now().isoformat()
    
    def get_pending_documents(self) -> List[str]:
        """Get list of documents with pending status."""
        return [
            doc_id for doc_id, doc_info in self.data["documents"].items()
            if doc_info["status"] in ["pending", "failed"]
        ]
    
    def create_backup(self, backup_path: Optional[Path] = None) -> Path:
        """
        Create backup of current lineage index.
        
        Args:
            backup_path: Optional custom backup path
            
        Returns:
            Path to backup file
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.index_path.with_suffix(f".{timestamp}.bak")
        
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        
        return backup_path
    
    def restore_from_backup(self, backup_path: Path) -> bool:
        """
        Restore lineage index from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if successful
        """
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            self.save_index()
            return True
        except Exception as e:
            print(f"Error restoring from backup: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get lineage index statistics."""
        total_docs = len(self.data["documents"])
        completed_docs = len([
            doc for doc in self.data["documents"].values()
            if doc["status"] == "completed"
        ])
        pending_docs = len([
            doc for doc in self.data["documents"].values()
            if doc["status"] in ["pending", "failed"]
        ])
        
        total_shards = len(self.data["shards"])
        total_nvecs = sum(shard["nvecs"] for shard in self.data["shards"].values())
        
        return {
            "total_documents": total_docs,
            "completed_documents": completed_docs,
            "pending_documents": pending_docs,
            "total_shards": total_shards,
            "total_vectors": total_nvecs,
            "faiss_index_type": self.data["faiss_index"]["type"],
            "last_updated": self.data["metadata"]["last_updated"]
        }


def main():
    """CLI for lineage index management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Lineage Index Management")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show lineage statistics")
    stats_parser.add_argument("--index", type=Path, default=Path("ops/lineage_index.json"))
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create backup")
    backup_parser.add_argument("--index", type=Path, default=Path("ops/lineage_index.json"))
    backup_parser.add_argument("--output", type=Path, help="Backup file path")
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore from backup")
    restore_parser.add_argument("--index", type=Path, default=Path("ops/lineage_index.json"))
    restore_parser.add_argument("--backup", type=Path, required=True, help="Backup file to restore")
    
    args = parser.parse_args()
    
    if args.command == "stats":
        lineage = LineageIndex(args.index)
        stats = lineage.get_statistics()
        
        print("📊 Lineage Index Statistics:")
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Completed documents: {stats['completed_documents']}")
        print(f"  Pending documents: {stats['pending_documents']}")
        print(f"  Total shards: {stats['total_shards']}")
        print(f"  Total vectors: {stats['total_vectors']}")
        print(f"  FAISS index type: {stats['faiss_index_type']}")
        print(f"  Last updated: {stats['last_updated']}")
        
    elif args.command == "backup":
        lineage = LineageIndex(args.index)
        backup_path = lineage.create_backup(args.output)
        print(f"✅ Backup created: {backup_path}")
        
    elif args.command == "restore":
        lineage = LineageIndex(args.index)
        if lineage.restore_from_backup(args.backup):
            print(f"✅ Restored from backup: {args.backup}")
        else:
            print(f"❌ Failed to restore from backup")
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
