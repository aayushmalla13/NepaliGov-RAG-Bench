#!/usr/bin/env python3
"""
Retrieval Metrics for NepaliGov-RAG-Bench

Implements comprehensive retrieval evaluation including core metrics (nDCG, Recall),
diversity metrics, and slice-based analysis with MLflow logging.
"""

import argparse
import json
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Try to import MLflow with fallback
try:
    import mlflow
    import mlflow.metrics
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class RetrievalMetrics:
    """Comprehensive retrieval evaluation with core, diversity, and slice metrics."""
    
    def __init__(self, log_to_mlflow: bool = True, csv_output: Optional[Path] = None):
        """
        Initialize metrics calculator.
        
        Args:
            log_to_mlflow: Whether to log to MLflow
            csv_output: Optional CSV output path
        """
        self.log_to_mlflow = log_to_mlflow and MLFLOW_AVAILABLE
        self.csv_output = csv_output
        self.metrics_cache = {}
    
    def calculate_ndcg(self, relevance_scores: List[float], k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k."""
        if not relevance_scores:
            return 0.0
        
        # Truncate to k
        relevance_scores = relevance_scores[:k]
        
        # DCG
        dcg = 0.0
        for i, rel in enumerate(relevance_scores):
            dcg += rel / math.log2(i + 2)  # i+2 because positions start at 1
        
        # IDCG (ideal DCG)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_scores):
            idcg += rel / math.log2(i + 2)
        
        # nDCG
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_recall(self, relevance_scores: List[float], k: int = 10) -> float:
        """Calculate Recall at k."""
        if not relevance_scores:
            return 0.0
        
        relevant_retrieved = sum(1 for score in relevance_scores[:k] if score > 0)
        total_relevant = sum(1 for score in relevance_scores if score > 0)
        
        return relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
    
    def calculate_authority_purity(self, candidates: List[Dict[str, Any]], k: int = 10) -> float:
        """Calculate authority purity - fraction of authoritative results in top k."""
        if not candidates:
            return 0.0
        
        top_k = candidates[:k]
        authoritative_count = sum(1 for c in top_k if c.get('source_authority') == 'authoritative')
        
        return authoritative_count / len(top_k)
    
    def calculate_diversity_metrics(self, candidates: List[Dict[str, Any]], k: int = 10) -> Dict[str, float]:
        """Calculate diversity metrics: unique docs, pages, coverage."""
        if not candidates:
            return {"unique_docs@k": 0.0, "unique_pages@k": 0.0, "coverage@k": 0.0}
        
        top_k = candidates[:k]
        
        # Unique documents
        unique_docs = len(set(c.get('doc_id', '') for c in top_k))
        
        # Unique pages
        unique_pages = len(set(c.get('page_id', '') for c in top_k))
        
        # Coverage (fraction of total available docs/pages represented)
        total_docs = len(set(c.get('doc_id', '') for c in candidates))
        total_pages = len(set(c.get('page_id', '') for c in candidates))
        
        doc_coverage = unique_docs / total_docs if total_docs > 0 else 0.0
        page_coverage = unique_pages / total_pages if total_pages > 0 else 0.0
        coverage = (doc_coverage + page_coverage) / 2
        
        return {
            "unique_docs@k": unique_docs,
            "unique_pages@k": unique_pages,
            "coverage@k": coverage
        }
    
    def calculate_slice_metrics(self, 
                               query_results: List[Dict[str, Any]], 
                               slice_field: str,
                               base_metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for different slices of data."""
        slice_metrics = defaultdict(dict)
        
        # Group by slice values
        slices = defaultdict(list)
        for result in query_results:
            slice_value = result.get(slice_field, 'unknown')
            if slice_field == 'conf_mean' and isinstance(slice_value, (int, float)):
                # Bin confidence scores
                if slice_value >= 80:
                    slice_value = 'high_conf'
                elif slice_value >= 60:
                    slice_value = 'med_conf'
                else:
                    slice_value = 'low_conf'
            elif slice_field == 'watermark_flags':
                # Convert to boolean
                slice_value = 'has_watermark' if slice_value else 'no_watermark'
            
            slices[str(slice_value)].append(result)
        
        # Calculate metrics for each slice
        for slice_value, slice_results in slices.items():
            if len(slice_results) >= 3:  # Minimum for meaningful metrics
                # Create relevance scores (simplified: authoritative=1, wikipedia=0.5, other=0)
                relevance_scores = []
                for result in slice_results:
                    if result.get('source_authority') == 'authoritative':
                        relevance_scores.append(1.0)
                    elif result.get('source_authority') == 'wikipedia':
                        relevance_scores.append(0.5)
                    else:
                        relevance_scores.append(0.0)
                
                slice_metrics[slice_value] = {
                    'ndcg@10': self.calculate_ndcg(relevance_scores, k=10),
                    'recall@10': self.calculate_recall(relevance_scores, k=10),
                    'authority_purity': self.calculate_authority_purity(slice_results, k=10),
                    'count': len(slice_results)
                }
        
        return dict(slice_metrics)
    
    def evaluate_query(self, 
                      query: str,
                      candidates: List[Dict[str, Any]], 
                      ground_truth: Optional[Dict[str, Any]] = None,
                      config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate a single query with comprehensive metrics.
        
        Args:
            query: Search query
            candidates: Retrieved candidates with metadata
            ground_truth: Optional ground truth for relevance
            config: Retrieval configuration used
        
        Returns:
            Dictionary of metrics
        """
        if not candidates:
            return self._empty_metrics()
        
        # Create relevance scores based on authority and ground truth
        relevance_scores = []
        for candidate in candidates:
            if ground_truth and candidate.get('chunk_id') == ground_truth.get('chunk_id'):
                relevance_scores.append(1.0)  # Perfect match
            elif candidate.get('source_authority') == 'authoritative':
                relevance_scores.append(0.8)  # High relevance
            elif candidate.get('source_authority') == 'wikipedia':
                relevance_scores.append(0.3)  # Low relevance (distractor)
            else:
                relevance_scores.append(0.0)  # No relevance
        
        # Core metrics
        metrics = {
            'query': query,
            'num_candidates': len(candidates),
            'ndcg@10': self.calculate_ndcg(relevance_scores, k=10),
            'recall@10': self.calculate_recall(relevance_scores, k=10),
            'authority_purity': self.calculate_authority_purity(candidates, k=10)
        }
        
        # Diversity metrics
        diversity_metrics = self.calculate_diversity_metrics(candidates, k=10)
        metrics.update(diversity_metrics)
        
        # Slice metrics
        slice_fields = ['source_page_is_ocr', 'conf_mean', 'watermark_flags', 'chunk_type']
        for field in slice_fields:
            if any(field in c for c in candidates):
                slice_metrics = self.calculate_slice_metrics(candidates, field, metrics)
                metrics[f'slices_{field}'] = slice_metrics
        
        # Configuration impact
        if config:
            metrics['config'] = config
            
            # Flag-based metrics
            if config.get('expand_query', False):
                metrics['expansion_used'] = True
            if config.get('diversify', False):
                metrics['diversification_used'] = True
        
        return metrics
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics for failed queries."""
        return {
            'num_candidates': 0,
            'ndcg@10': 0.0,
            'recall@10': 0.0,
            'authority_purity': 0.0,
            'unique_docs@k': 0.0,
            'unique_pages@k': 0.0,
            'coverage@k': 0.0
        }
    
    def evaluate_dataset(self, 
                        queries_and_results: List[Dict[str, Any]], 
                        config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate entire dataset with aggregated metrics.
        
        Args:
            queries_and_results: List of {query, candidates, ground_truth} dicts
            config: Overall configuration
        
        Returns:
            Aggregated metrics
        """
        all_metrics = []
        
        for item in queries_and_results:
            query = item['query']
            candidates = item.get('candidates', [])
            ground_truth = item.get('ground_truth')
            
            query_metrics = self.evaluate_query(query, candidates, ground_truth, config)
            all_metrics.append(query_metrics)
        
        # Aggregate metrics
        aggregated = self._aggregate_metrics(all_metrics)
        
        # Add dataset-level info
        aggregated['num_queries'] = len(queries_and_results)
        aggregated['config'] = config or {}
        
        # Log to MLflow if enabled
        if self.log_to_mlflow:
            self._log_to_mlflow(aggregated)
        
        # Save to CSV if requested
        if self.csv_output:
            self._save_to_csv(all_metrics, aggregated)
        
        return aggregated
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across queries."""
        if not metrics_list:
            return {}
        
        # Core metrics - take mean
        core_metrics = ['ndcg@10', 'recall@10', 'authority_purity', 'unique_docs@k', 'unique_pages@k', 'coverage@k']
        aggregated = {}
        
        for metric in core_metrics:
            values = [m.get(metric, 0.0) for m in metrics_list if metric in m]
            if values:
                aggregated[f'mean_{metric}'] = np.mean(values)
                aggregated[f'std_{metric}'] = np.std(values)
        
        # Count metrics
        aggregated['total_candidates'] = sum(m.get('num_candidates', 0) for m in metrics_list)
        
        # Configuration flags
        expansion_count = sum(1 for m in metrics_list if m.get('expansion_used', False))
        diversification_count = sum(1 for m in metrics_list if m.get('diversification_used', False))
        
        aggregated['expansion_usage_rate'] = expansion_count / len(metrics_list)
        aggregated['diversification_usage_rate'] = diversification_count / len(metrics_list)
        
        return aggregated
    
    def _log_to_mlflow(self, metrics: Dict[str, Any]):
        """Log metrics to MLflow."""
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            # Start MLflow run if not already active
            if not mlflow.active_run():
                mlflow.start_run()
            
            # Log metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not math.isnan(value):
                    mlflow.log_metric(key, value)
                elif isinstance(value, dict):
                    # Log nested metrics with prefix
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float)) and not math.isnan(subvalue):
                            mlflow.log_metric(f"{key}_{subkey}", subvalue)
            
            # Log config as parameters
            config = metrics.get('config', {})
            for key, value in config.items():
                mlflow.log_param(key, value)
            
            print("✅ Metrics logged to MLflow")
            
        except Exception as e:
            print(f"Warning: MLflow logging failed: {e}")
    
    def _save_to_csv(self, query_metrics: List[Dict[str, Any]], aggregated: Dict[str, Any]):
        """Save metrics to CSV file."""
        try:
            # Flatten query metrics for CSV
            flattened_metrics = []
            for metrics in query_metrics:
                flat = {}
                for key, value in metrics.items():
                    if isinstance(value, dict):
                        # Skip complex nested structures
                        continue
                    elif isinstance(value, (list, tuple)):
                        flat[key] = str(value)
                    else:
                        flat[key] = value
                flattened_metrics.append(flat)
            
            # Create DataFrame and save
            df = pd.DataFrame(flattened_metrics)
            self.csv_output.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.csv_output, index=False)
            
            # Save aggregated metrics
            agg_path = self.csv_output.parent / f"aggregated_{self.csv_output.name}"
            pd.DataFrame([aggregated]).to_csv(agg_path, index=False)
            
            print(f"✅ Metrics saved to CSV: {self.csv_output}")
            
        except Exception as e:
            print(f"Warning: CSV saving failed: {e}")


def main():
    """CLI for testing retrieval metrics."""
    parser = argparse.ArgumentParser(description="Test retrieval metrics")
    parser.add_argument("--query", required=True, help="Test query")
    parser.add_argument("--candidates-file", type=Path, help="JSON file with candidates")
    parser.add_argument("--csv-output", type=Path, help="CSV output path")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    
    args = parser.parse_args()
    
    try:
        # Initialize metrics
        metrics_calc = RetrievalMetrics(
            log_to_mlflow=not args.no_mlflow,
            csv_output=args.csv_output
        )
        
        # Load candidates
        if args.candidates_file and args.candidates_file.exists():
            with open(args.candidates_file) as f:
                candidates = json.load(f)
        else:
            # Create dummy candidates for testing
            candidates = [
                {"text": "Test document 1", "source_authority": "authoritative", "doc_id": "doc1", "page_id": "page1"},
                {"text": "Test document 2", "source_authority": "wikipedia", "doc_id": "doc2", "page_id": "page2"},
                {"text": "Test document 3", "source_authority": "authoritative", "doc_id": "doc1", "page_id": "page2"}
            ]
        
        # Evaluate
        config = {"expand_query": True, "diversify": True}
        results = metrics_calc.evaluate_query(args.query, candidates, config=config)
        
        print(f"Query: {args.query}")
        print(f"Results:")
        for key, value in results.items():
            if not isinstance(value, dict):
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



