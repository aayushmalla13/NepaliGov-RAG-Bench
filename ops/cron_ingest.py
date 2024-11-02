#!/usr/bin/env python3
"""
Incremental Re-indexing for NepaliGov-RAG-Bench

Detects changed PDFs, rebuilds only affected shards, handles configuration changes,
and runs quick evaluation slices with rollback on regression.
"""

import argparse
import json
import subprocess
import sys
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import pandas as pd
import yaml

# Import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from ops.lineage_index import LineageIndex


class IncrementalIngester:
    """Manages incremental ingestion and re-indexing operations."""
    
    def __init__(
        self,
        manifest_path: Path = Path("data/seed_manifest.yaml"),
        lineage_path: Path = Path("ops/lineage_index.json"),
        corpus_dir: Path = Path("data/corpus_parquet"),
        faiss_dir: Path = Path("data/faiss"),
        changelog_path: Path = Path("ops/changelog.md")
    ):
        """
        Initialize incremental ingester.
        
        Args:
            manifest_path: Path to document manifest
            lineage_path: Path to lineage index
            corpus_dir: Directory for individual document parquet files
            faiss_dir: Directory for FAISS index files
            changelog_path: Path to changelog file
        """
        self.manifest_path = manifest_path
        self.lineage_path = lineage_path
        self.corpus_dir = corpus_dir
        self.faiss_dir = faiss_dir
        self.changelog_path = changelog_path
        
        self.lineage = LineageIndex(lineage_path)
        self.config_paths = {
            "cp6_synonyms.json": Path("config/cp6_synonyms.json"),
            "cp9_mappings.json": Path("config/cp9_mappings.json"),
            "cp10_thresholds.yaml": Path("config/cp10_thresholds.yaml"),
            "cp11_5_ui_translation.yaml": Path("config/cp11_5_ui_translation.yaml")
        }
        # cache latest detected changes for manifest annotation
        self._last_config_changes: List[Tuple[str, Optional[str], str]] = []
        
        # Create directories
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)
        self.changelog_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_manifest(self) -> Dict[str, Any]:
        """Load document manifest."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def detect_document_changes(self) -> List[Tuple[str, Optional[str], str]]:
        """
        Detect changed documents by comparing manifest SHA-256 with lineage.
        
        Returns:
            List of (doc_id, old_sha256, new_sha256) tuples
        """
        print("🔍 Detecting document changes...")
        
        manifest = self.load_manifest()
        changed = self.lineage.get_changed_documents(manifest)
        
        if changed:
            print(f"  Found {len(changed)} changed documents:")
            for doc_id, old_sha, new_sha in changed:
                if old_sha is None:
                    print(f"    🆕 {doc_id}: NEW")
                else:
                    print(f"    📝 {doc_id}: {old_sha[:8]} → {new_sha[:8]}")
        else:
            print("  No document changes detected")
        
        return changed
    
    def detect_config_changes(self) -> List[Tuple[str, Optional[str], str]]:
        """
        Detect configuration file changes.
        
        Returns:
            List of (config_name, old_checksum, new_checksum) tuples
        """
        print("🔧 Detecting configuration changes...")
        
        # Filter to only existing config files
        existing_configs = {
            name: path for name, path in self.config_paths.items()
            if path.exists()
        }
        
        changes = self.lineage.detect_config_changes(existing_configs)
        
        if changes:
            print(f"  Found {len(changes)} configuration changes:")
            for config_name, old_checksum, new_checksum in changes:
                old_display = old_checksum[:8] if old_checksum else "NEW"
                new_display = new_checksum[:8] if new_checksum else "DELETED"
                print(f"    🔧 {config_name}: {old_display} → {new_display}")
        else:
            print("  No configuration changes detected")
        
        # cache for later use (e.g., writing into index.manifest.json)
        self._last_config_changes = changes
        return changes
    
    def process_changed_documents(self, changed_docs: List[Tuple[str, Optional[str], str]]) -> bool:
        """
        Process changed documents through the ingestion pipeline.
        
        Args:
            changed_docs: List of changed document tuples
            
        Returns:
            True if successful, False otherwise
        """
        if not changed_docs:
            return True
        
        print(f"📄 Processing {len(changed_docs)} changed documents...")
        
        manifest = self.load_manifest()
        doc_map = {doc["doc_id"]: doc for doc in manifest["documents"]}
        
        success_count = 0
        
        for doc_id, old_sha, new_sha in changed_docs:
            if doc_id not in doc_map:
                print(f"  ❌ Document {doc_id} not found in manifest")
                continue
            
            doc_info = doc_map[doc_id]
            pdf_path = Path(doc_info["file"])
            
            if not pdf_path.exists():
                print(f"  ❌ PDF file not found: {pdf_path}")
                continue
            
            try:
                print(f"  🔄 Processing {doc_id}...")
                
                # Update lineage with new document info
                self.lineage.update_document(
                    doc_id=doc_id,
                    file_path=pdf_path,
                    sha256=new_sha,
                    shard_id="default",  # For now, use single shard
                    status="processing"
                )
                
                # Run document processing pipeline (CP2-CP5)
                success = self._process_single_document(doc_id, pdf_path, doc_info)
                
                if success:
                    self.lineage.mark_documents_status([doc_id], "completed")
                    success_count += 1
                    print(f"    ✅ Processed {doc_id}")
                else:
                    self.lineage.mark_documents_status([doc_id], "failed")
                    print(f"    ❌ Failed to process {doc_id}")
                
            except Exception as e:
                print(f"  ❌ Error processing {doc_id}: {e}")
                self.lineage.mark_documents_status([doc_id], "failed")
        
        print(f"  📊 Processed {success_count}/{len(changed_docs)} documents successfully")
        return success_count == len(changed_docs)
    
    def _process_single_document(self, doc_id: str, pdf_path: Path, doc_info: Dict[str, Any]) -> bool:
        """
        Process a single document through the ingestion pipeline.
        
        Args:
            doc_id: Document identifier
            pdf_path: Path to PDF file
            doc_info: Document metadata from manifest
            
        Returns:
            True if successful
        """
        try:
            # Step 1: Page type detection (CP2)
            print(f"    🔍 Page type detection...")
            page_types_path = Path(f"data/page_types/{doc_id}.jsonl")
            page_types_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Run page type detection
            cmd = [
                sys.executable, "-m", "src.ingest.page_type_detector",
                "--input", str(pdf_path),
                "--output", str(page_types_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"      ❌ Page type detection failed: {result.stderr}")
                return False
            
            # Step 2: OCR pipeline (CP3)
            print(f"    📝 OCR processing...")
            ocr_path = Path(f"data/ocr_json/{doc_id}.json")
            ocr_path.parent.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                sys.executable, "-m", "src.ingest.ocr_pipeline",
                "--input", str(pdf_path),
                "--page-types", str(page_types_path),
                "--output", str(ocr_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"      ❌ OCR processing failed: {result.stderr}")
                return False
            
            # Step 3: Text extraction and corpus building (CP4)
            print(f"    📚 Corpus extraction...")
            corpus_path = self.corpus_dir / f"{doc_id}.parquet"
            
            cmd = [
                sys.executable, "-m", "src.ingest.text_extractor",
                "--input", str(pdf_path),
                "--ocr", str(ocr_path),
                "--output", str(corpus_path),
                "--doc-id", doc_id,
                "--authority", doc_info["authority"],
                "--is-distractor", str(doc_info["is_distractor"]).lower()
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"      ❌ Corpus extraction failed: {result.stderr}")
                return False
            
            return True
            
        except Exception as e:
            print(f"    ❌ Document processing error: {e}")
            return False
    
    def rebuild_corpus(self) -> bool:
        """
        Rebuild the unified corpus from individual document parquets.
        
        Returns:
            True if successful
        """
        print("📚 Rebuilding unified corpus...")
        
        try:
            # Find all document parquet files
            parquet_files = list(self.corpus_dir.glob("*.parquet"))
            if not parquet_files:
                print("  ❌ No document parquet files found")
                return False
            
            print(f"  Found {len(parquet_files)} document files")
            
            # Load and combine all parquets
            dfs = []
            for parquet_file in parquet_files:
                try:
                    df = pd.read_parquet(parquet_file)
                    dfs.append(df)
                except Exception as e:
                    print(f"  ⚠️ Could not load {parquet_file}: {e}")
            
            if not dfs:
                print("  ❌ No valid parquet files loaded")
                return False
            
            # Combine all dataframes
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Save unified corpus
            unified_corpus_path = Path("data/corpus.parquet")
            combined_df.to_parquet(unified_corpus_path, index=False)
            
            print(f"  ✅ Unified corpus saved: {len(combined_df)} rows → {unified_corpus_path}")
            return True
            
        except Exception as e:
            print(f"  ❌ Error rebuilding corpus: {e}")
            return False
    
    def rebuild_faiss_index(self, force_full_rebuild: bool = False) -> bool:
        """
        Rebuild FAISS index incrementally or fully.
        
        Args:
            force_full_rebuild: Force full index rebuild
            
        Returns:
            True if successful
        """
        print("🔍 Rebuilding FAISS index...")
        rebuild_start_time = time.time()
        
        try:
            # For now, always do full rebuild (incremental is complex)
            # In a production system, this would be more sophisticated
            
            corpus_path = Path("data/corpus.parquet")
            if not corpus_path.exists():
                print("  ❌ Unified corpus not found")
                return False
            
            # Run chunking and embedding
            cmd = [
                sys.executable, "-m", "src.retriever.chunk_and_embed",
                "--in", str(corpus_path),
                "--out", str(self.faiss_dir / "index.bin"),
                "--model", "bge-m3",
                "--index-type", "HNSW"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ❌ FAISS indexing failed: {result.stderr}")
                return False
            
            # Update lineage with index info
            manifest_path = self.faiss_dir / "index.manifest.json"
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest_data = json.load(f)
                # attach config checksum info/deltas for provenance
                current_checksums = {}
                for name, path in self.config_paths.items():
                    if path.exists():
                        cs = self.lineage.compute_file_checksum(path)
                        current_checksums[name] = cs
                # format deltas with enhanced metadata
                config_deltas = []
                for name, old_cs, new_cs in getattr(self, "_last_config_changes", []) or []:
                    config_deltas.append({
                        "file": name,
                        "hash_old": old_cs,
                        "hash_new": new_cs,
                        "changed_at": datetime.now().isoformat(),
                        "change_type": "new" if old_cs is None else "modified"
                    })
                
                # Enhanced manifest metadata
                manifest_data["config_checksums"] = current_checksums
                manifest_data["config_last_checked"] = datetime.now().isoformat()
                if config_deltas:
                    manifest_data["config_deltas"] = config_deltas
                
                # Add build provenance with timing
                rebuild_time = time.time() - rebuild_start_time
                manifest_data["build_provenance"] = {
                    "incremental_build": True,
                    "config_changes_detected": len(config_deltas),
                    "lineage_version": "1.0",
                    "cp12_enhanced": True,
                    "rebuild_time_seconds": round(rebuild_time, 2),
                    "build_timestamp": datetime.now().isoformat()
                }
                
                # persist enriched manifest deterministically
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    json.dump(manifest_data, f, indent=2, ensure_ascii=False)

                self.lineage.update_faiss_index_info(
                    index_type="single",
                    shards=["default"],
                    total_nvecs=manifest_data.get("nvecs", 0),
                    manifest_path=manifest_path
                )
            
            print(f"  ✅ FAISS index rebuilt")
            return True
            
        except Exception as e:
            print(f"  ❌ Error rebuilding FAISS index: {e}")
            return False
    
    def run_quick_eval(self) -> Dict[str, float]:
        """
        Run quick evaluation slice to check for regressions.
        
        Returns:
            Dictionary of metric scores
        """
        print("📊 Running quick evaluation slice...")
        
        try:
            # Check if evaluation datasets exist
            dev_path = Path("data/qacite/dev.jsonl")
            if not dev_path.exists():
                print("  ⚠️ No evaluation dataset found, skipping eval")
                return {}
            
            # Run retrieval evaluation on subset
            print("  🔍 Running retrieval evaluation...")
            retrieval_cmd = [
                sys.executable, "-m", "src.retriever.run_retrieval_eval",
                "--data", str(dev_path),
                "--index", str(self.faiss_dir),
                "--output", "data/metrics/quick_retrieval_eval.json",
                "--limit", "50"  # Quick eval on subset
            ]
            
            result = subprocess.run(retrieval_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"    ⚠️ Retrieval eval failed: {result.stderr}")
            
            # Run QA evaluation on subset
            print("  💬 Running QA evaluation...")
            qa_cmd = [
                sys.executable, "-m", "src.answer.run_qa_eval",
                "--data", str(dev_path),
                "--index", str(self.faiss_dir),
                "--output", "data/metrics/quick_qa_eval.json",
                "--limit", "30"  # Quick eval on subset
            ]
            
            result = subprocess.run(qa_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"    ⚠️ QA eval failed: {result.stderr}")
            
            # Parse results
            metrics = {}
            
            # Load retrieval metrics
            retrieval_results_path = Path("data/metrics/quick_retrieval_eval.json")
            if retrieval_results_path.exists():
                with open(retrieval_results_path, 'r') as f:
                    retrieval_data = json.load(f)
                    metrics.update({
                        "ndcg@10": retrieval_data.get("ndcg@10", 0.0),
                        "recall@10": retrieval_data.get("recall@10", 0.0),
                        "authority_purity": retrieval_data.get("authority_purity", 0.0)
                    })
            
            # Load QA metrics
            qa_results_path = Path("data/metrics/quick_qa_eval.json")
            if qa_results_path.exists():
                with open(qa_results_path, 'r') as f:
                    qa_data = json.load(f)
                    metrics.update({
                        "citation_precision": qa_data.get("citation_precision", 0.0),
                        "citation_recall": qa_data.get("citation_recall", 0.0),
                        "same_lang_citation_rate": qa_data.get("same_lang_citation_rate", 0.0),
                        "cross_lang_success@k": qa_data.get("cross_lang_success@k", 0.0)
                    })
            
            print(f"  📈 Quick evaluation completed: {len(metrics)} metrics")
            return metrics
            
        except Exception as e:
            print(f"  ❌ Error in quick evaluation: {e}")
            return {}
    
    def check_regression(self, current_metrics: Dict[str, float], tolerance: float = 0.05) -> bool:
        """
        Check if current metrics show regression compared to baseline.
        
        Args:
            current_metrics: Current evaluation metrics
            tolerance: Regression tolerance (5% by default)
            
        Returns:
            True if regression detected, False otherwise
        """
        print("🚨 Checking for metric regression...")
        
        # Load baseline metrics (if available)
        baseline_path = Path("data/metrics/baseline_metrics.json")
        if not baseline_path.exists():
            print("  ℹ️ No baseline metrics found, skipping regression check")
            return False
        
        try:
            with open(baseline_path, 'r') as f:
                baseline_metrics = json.load(f)
            
            # Check critical metrics for regression
            critical_metrics = ["authority_purity", "citation_precision", "citation_recall"]
            
            regressions = []
            for metric in critical_metrics:
                if metric in current_metrics and metric in baseline_metrics:
                    current = current_metrics[metric]
                    baseline = baseline_metrics[metric]
                    
                    if current < baseline * (1 - tolerance):
                        regression_pct = ((baseline - current) / baseline) * 100
                        regressions.append(f"{metric}: {baseline:.3f} → {current:.3f} (-{regression_pct:.1f}%)")
            
            if regressions:
                print(f"  🚨 REGRESSION DETECTED:")
                for regression in regressions:
                    print(f"    📉 {regression}")
                return True
            else:
                print("  ✅ No significant regression detected")
                return False
                
        except Exception as e:
            print(f"  ❌ Error checking regression: {e}")
            return False
    
    def rollback_index(self) -> bool:
        """
        Rollback to last known good index state.
        
        Returns:
            True if successful
        """
        print("🔄 Rolling back to last known good index...")
        
        try:
            # Create backup of current state
            backup_path = self.lineage.create_backup()
            print(f"  📁 Current state backed up to {backup_path}")
            
            # Look for previous backup to restore
            backup_pattern = self.lineage.index_path.with_suffix("*.bak")
            backup_files = list(self.lineage.index_path.parent.glob(f"{self.lineage.index_path.stem}.*.bak"))
            
            if backup_files:
                # Restore from most recent backup (excluding the one we just created)
                backup_files.sort(reverse=True)
                restore_backup = backup_files[1] if len(backup_files) > 1 else backup_files[0]
                
                if self.lineage.restore_from_backup(restore_backup):
                    print(f"  ✅ Restored lineage from {restore_backup}")
                    
                    # TODO: Also restore FAISS index files
                    # This would require keeping versioned FAISS backups
                    print("  ⚠️ Manual FAISS index restoration may be required")
                    return True
                else:
                    print(f"  ❌ Failed to restore from backup")
                    return False
            else:
                print("  ⚠️ No backup files found for rollback")
                return False
                
        except Exception as e:
            print(f"  ❌ Error during rollback: {e}")
            return False
    
    def write_changelog(
        self, 
        changed_docs: List[Tuple[str, Optional[str], str]],
        config_changes: List[Tuple[str, Optional[str], str]],
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float],
        success: bool
    ) -> None:
        """Write changelog entry."""
        print("📝 Writing changelog...")
        
        timestamp = datetime.now().isoformat()
        
        changelog_entry = f"""
## Incremental Ingestion - {timestamp}

### Status: {'✅ SUCCESS' if success else '❌ FAILED'}

### Documents Changed ({len(changed_docs)})
"""
        
        for doc_id, old_sha, new_sha in changed_docs:
            if old_sha is None:
                changelog_entry += f"- 🆕 **{doc_id}**: NEW document\n"
            else:
                changelog_entry += f"- 📝 **{doc_id}**: {old_sha[:8]} → {new_sha[:8]}\n"
        
        if config_changes:
            changelog_entry += f"\n### Configuration Changes ({len(config_changes)})\n"
            for config_name, old_checksum, new_checksum in config_changes:
                old_display = old_checksum[:8] if old_checksum else "NEW"
                new_display = new_checksum[:8] if new_checksum else "DELETED"
                changelog_entry += f"- 🔧 **{config_name}**: {old_display} → {new_display}\n"
        
        if metrics_before or metrics_after:
            changelog_entry += "\n### Metrics Comparison\n"
            changelog_entry += "| Metric | Before | After | Change |\n"
            changelog_entry += "|--------|--------|-------|--------|\n"
            
            all_metrics = set(metrics_before.keys()) | set(metrics_after.keys())
            for metric in sorted(all_metrics):
                before = metrics_before.get(metric, 0.0)
                after = metrics_after.get(metric, 0.0)
                change = after - before
                change_str = f"{change:+.3f}" if change != 0 else "0.000"
                changelog_entry += f"| {metric} | {before:.3f} | {after:.3f} | {change_str} |\n"
        
        # Append to changelog file
        try:
            with open(self.changelog_path, 'a', encoding='utf-8') as f:
                f.write(changelog_entry)
            print(f"  ✅ Changelog updated: {self.changelog_path}")
        except Exception as e:
            print(f"  ❌ Error writing changelog: {e}")
    
    def run_incremental_ingest(self, refresh_expansion: bool = True) -> bool:
        """
        Run full incremental ingestion process.
        
        Args:
            refresh_expansion: Whether to refresh query expansion configs
            
        Returns:
            True if successful
        """
        print("🚀 Starting incremental ingestion process...")
        print(f"  Manifest: {self.manifest_path}")
        print(f"  Lineage: {self.lineage_path}")
        print(f"  Corpus dir: {self.corpus_dir}")
        print(f"  FAISS dir: {self.faiss_dir}")
        
        try:
            # Create backup before starting
            backup_path = self.lineage.create_backup()
            print(f"📁 Created backup: {backup_path}")
            
            # Step 1: Detect changes
            changed_docs = self.detect_document_changes()
            config_changes = self.detect_config_changes()
            
            if not changed_docs and not config_changes:
                print("✅ No changes detected, nothing to do")
                return True
            
            # Step 2: Get baseline metrics
            print("📊 Getting baseline metrics...")
            metrics_before = self.run_quick_eval()
            
            # Step 3: Process changed documents
            if changed_docs:
                if not self.process_changed_documents(changed_docs):
                    print("❌ Document processing failed")
                    return False
                
                # Rebuild unified corpus
                if not self.rebuild_corpus():
                    print("❌ Corpus rebuild failed")
                    return False
            
            # Step 4: Rebuild FAISS index
            force_rebuild = bool(changed_docs) or bool(config_changes)
            if force_rebuild:
                if not self.rebuild_faiss_index(force_full_rebuild=True):
                    print("❌ FAISS index rebuild failed")
                    return False
            
            # Step 5: Run evaluation and check for regression
            print("📊 Running post-change evaluation...")
            metrics_after = self.run_quick_eval()
            
            regression_detected = self.check_regression(metrics_after)
            
            if regression_detected:
                print("🚨 Regression detected - rolling back...")
                if self.rollback_index():
                    print("✅ Rollback completed")
                    self.write_changelog(changed_docs, config_changes, metrics_before, metrics_after, False)
                    return False
                else:
                    print("❌ Rollback failed - manual intervention required")
                    return False
            
            # Step 6: Save successful state
            self.lineage.save_index()
            
            # Step 7: Write changelog
            self.write_changelog(changed_docs, config_changes, metrics_before, metrics_after, True)
            
            print("✅ Incremental ingestion completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Incremental ingestion failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Incremental re-indexing for NepaliGov-RAG-Bench"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/seed_manifest.yaml"),
        help="Path to document manifest"
    )
    parser.add_argument(
        "--lineage",
        type=Path,
        default=Path("ops/lineage_index.json"),
        help="Path to lineage index"
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/corpus_parquet"),
        help="Directory for document parquet files"
    )
    parser.add_argument(
        "--faiss-dir",
        type=Path,
        default=Path("data/faiss"),
        help="Directory for FAISS index files"
    )
    parser.add_argument(
        "--refresh-expansion",
        type=bool,
        default=True,
        help="Refresh query expansion configs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    try:
        ingester = IncrementalIngester(
            manifest_path=args.manifest,
            lineage_path=args.lineage,
            corpus_dir=args.corpus_dir,
            faiss_dir=args.faiss_dir
        )
        
        if args.dry_run:
            print("🔍 DRY RUN MODE - Detecting changes only...")
            changed_docs = ingester.detect_document_changes()
            config_changes = ingester.detect_config_changes()
            
            if changed_docs or config_changes:
                print(f"Would process {len(changed_docs)} documents and {len(config_changes)} config changes")
            else:
                print("No changes detected")
            
            return
        
        success = ingester.run_incremental_ingest(
            refresh_expansion=args.refresh_expansion
        )
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
