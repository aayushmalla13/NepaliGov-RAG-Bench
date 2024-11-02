#!/usr/bin/env python3
"""
Retrieval Evaluation Orchestrator for NepaliGov-RAG-Bench

Orchestrates comprehensive retrieval evaluation with reranking, expansion,
diversification, and detailed metrics analysis.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

import pandas as pd

# Local imports
from src.retriever.search import MultilingualRetriever
from src.rerank.ce_reranker import CEReranker
from src.eval.retrieval_metrics import RetrievalMetrics

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class RetrievalEvaluator:
    """Orchestrates comprehensive retrieval evaluation."""
    
    def __init__(self, 
                 faiss_dir: Path,
                 qacite_dir: Path,
                 reranker_model: str = "BAAI/bge-reranker-base",
                 cache_dir: Optional[Path] = None):
        """
        Initialize retrieval evaluator.
        
        Args:
            faiss_dir: FAISS index directory
            qacite_dir: Q-A-Cite dataset directory
            reranker_model: Reranker model name
            cache_dir: Cache directory for reranker
        """
        self.faiss_dir = faiss_dir
        self.qacite_dir = qacite_dir
        
        # Initialize retriever
        print("Loading retriever...")
        self.retriever = MultilingualRetriever(faiss_dir)
        
        # Initialize reranker
        print(f"Loading reranker: {reranker_model}")
        self.reranker = CEReranker(
            model_name=reranker_model,
            cache_dir=cache_dir
        )
        
        # Initialize metrics calculator
        self.metrics_calc = RetrievalMetrics()
        
        print("✅ Retrieval evaluator initialized")
    
    def load_qacite_dataset(self, split: str = "dev") -> List[Dict[str, Any]]:
        """Load Q-A-Cite dataset for evaluation."""
        qacite_file = self.qacite_dir / f"{split}.jsonl"
        
        if not qacite_file.exists():
            print(f"Warning: {qacite_file} not found, trying other splits...")
            # Try other files
            for alt_split in ["test", "train"]:
                alt_file = self.qacite_dir / f"{alt_split}.jsonl"
                if alt_file.exists() and alt_file.stat().st_size > 0:
                    print(f"Using {alt_split} split instead")
                    qacite_file = alt_file
                    break
            else:
                raise FileNotFoundError(f"No usable Q-A-Cite files found in {self.qacite_dir}")
        
        dataset = []
        with open(qacite_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    dataset.append(item)
        
        print(f"Loaded {len(dataset)} Q-A pairs from {qacite_file.name}")
        return dataset
    
    def run_retrieval_experiment(self, 
                                config: Dict[str, Any],
                                dataset: List[Dict[str, Any]],
                                max_queries: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Run retrieval experiment with given configuration.
        
        Args:
            config: Retrieval configuration
            dataset: Q-A-Cite dataset
            max_queries: Maximum number of queries to evaluate
        
        Returns:
            List of query results with candidates
        """
        results = []
        
        # Limit queries if specified
        if max_queries:
            dataset = dataset[:max_queries]
        
        print(f"Running retrieval on {len(dataset)} queries...")
        
        for i, qa_item in enumerate(dataset):
            query = qa_item['question']
            ground_truth = {
                'chunk_id': qa_item.get('chunk_id'),
                'doc_id': qa_item.get('doc_id'),
                'page_id': qa_item.get('page_id')
            }
            
            try:
                # Retrieve candidates (using correct API parameters)
                search_results = self.retriever.search(
                    query=query,
                    k=config.get('k', 20),
                    query_lang=config.get('query_lang', 'auto'),
                    allow_distractors=config.get('allow_distractors', True),
                    inject_hard_negatives=config.get('inject_hard_negatives', 5),
                    nprobe=config.get('nprobe', 10),
                    efSearch=config.get('efSearch', 40)
                )
                
                # Extract candidates
                candidates = search_results.get('authoritative_candidates', [])
                if config.get('allow_distractors', True):
                    candidates.extend(search_results.get('distractor_candidates', []))
                
                # Apply reranking if enabled
                if config.get('with_rerank', False):
                    if config.get('fusion_rerank', False) and 'expansion_terms' in search_results:
                        # Fusion reranking with expanded queries
                        original_query = query
                        expanded_queries = [original_query] + search_results.get('expansion_terms', [])
                        candidates = self.reranker.fusion_rerank(
                            queries=expanded_queries,
                            candidates=candidates,
                            fusion_method=config.get('fusion_method', 'max'),
                            top_k=config.get('rerank_top_k', 20)
                        )
                    else:
                        # Standard reranking
                        candidates = self.reranker.rerank(
                            query=query,
                            candidates=candidates,
                            top_k=config.get('rerank_top_k', 20)
                        )
                
                # Store results
                results.append({
                    'query': query,
                    'candidates': candidates,
                    'ground_truth': ground_truth,
                    'search_results': search_results,
                    'config': config
                })
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(dataset)} queries")
            
            except Exception as e:
                print(f"Error processing query {i}: {e}")
                # Add empty result to maintain alignment
                results.append({
                    'query': query,
                    'candidates': [],
                    'ground_truth': ground_truth,
                    'search_results': {},
                    'config': config,
                    'error': str(e)
                })
        
        return results
    
    def evaluate_configuration(self, 
                              config: Dict[str, Any],
                              dataset: List[Dict[str, Any]],
                              output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Evaluate a single configuration.
        
        Args:
            config: Configuration to evaluate
            dataset: Q-A-Cite dataset
            output_dir: Optional output directory for detailed results
        
        Returns:
            Evaluation metrics
        """
        print(f"\n🔬 Evaluating configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Run experiment
        start_time = time.time()
        results = self.run_retrieval_experiment(config, dataset)
        eval_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self.metrics_calc.evaluate_dataset(results, config)
        metrics['evaluation_time'] = eval_time
        
        # Save detailed results if requested
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            config_name = "_".join([f"{k}={v}" for k, v in config.items() if k != 'description'])[:100]
            results_file = output_dir / f"results_{config_name}.json"
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_for_json(obj):
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif hasattr(obj, 'item'):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'config': convert_for_json(config),
                    'metrics': convert_for_json(metrics),
                    'results': convert_for_json(results[:5])  # Save first 5 for inspection
                }, f, ensure_ascii=False, indent=2)
            
            print(f"  Detailed results saved to {results_file}")
        
        return metrics
    
    def run_ablation_study(self, 
                          dataset: List[Dict[str, Any]],
                          output_dir: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
        """
        Run ablation study with different configurations.
        
        Args:
            dataset: Q-A-Cite dataset
            output_dir: Output directory for results
        
        Returns:
            Dictionary of configuration name to metrics
        """
        # Define configurations for ablation study
        configs = {
            'baseline': {
                'description': 'Baseline retrieval',
                'expand_query': False,
                'diversify': False,
                'with_rerank': False,
                'k': 20
            },
            'expansion': {
                'description': 'With query expansion',
                'expand_query': True,
                'diversify': False,
                'with_rerank': False,
                'k': 20
            },
            'diversification': {
                'description': 'With diversification',
                'expand_query': False,
                'diversify': True,
                'with_rerank': False,
                'k': 20
            },
            'expansion_diversification': {
                'description': 'Expansion + diversification',
                'expand_query': True,
                'diversify': True,
                'with_rerank': False,
                'k': 20
            },
            'reranking': {
                'description': 'With reranking',
                'expand_query': False,
                'diversify': False,
                'with_rerank': True,
                'k': 20
            },
            'full_system': {
                'description': 'Full system (expansion + diversification + reranking)',
                'expand_query': True,
                'diversify': True,
                'with_rerank': True,
                'fusion_rerank': True,
                'k': 20
            }
        }
        
        print(f"\n🧪 Running ablation study with {len(configs)} configurations")
        
        all_results = {}
        for config_name, config in configs.items():
            print(f"\n--- Configuration: {config_name} ---")
            metrics = self.evaluate_configuration(config, dataset, output_dir)
            all_results[config_name] = metrics
            
            # Print key metrics
            print(f"  nDCG@10: {metrics.get('mean_ndcg@10', 0.0):.4f}")
            print(f"  Recall@10: {metrics.get('mean_recall@10', 0.0):.4f}")
            print(f"  Authority Purity: {metrics.get('mean_authority_purity', 0.0):.4f}")
            print(f"  Coverage@10: {metrics.get('mean_coverage@k', 0.0):.4f}")
        
        return all_results
    
    def print_comparison_table(self, results: Dict[str, Dict[str, Any]]):
        """Print comparison table of results."""
        print(f"\n📊 RETRIEVAL EVALUATION COMPARISON")
        print("=" * 80)
        
        # Headers
        headers = ["Config", "nDCG@10", "Recall@10", "Auth.Purity", "Coverage@10", "Time(s)"]
        print(f"{headers[0]:<20} {headers[1]:<10} {headers[2]:<10} {headers[3]:<12} {headers[4]:<12} {headers[5]:<8}")
        print("-" * 80)
        
        # Results
        for config_name, metrics in results.items():
            ndcg = metrics.get('mean_ndcg@10', 0.0)
            recall = metrics.get('mean_recall@10', 0.0)
            purity = metrics.get('mean_authority_purity', 0.0)
            coverage = metrics.get('mean_coverage@k', 0.0)
            eval_time = metrics.get('evaluation_time', 0.0)
            
            print(f"{config_name:<20} {ndcg:<10.4f} {recall:<10.4f} {purity:<12.4f} {coverage:<12.4f} {eval_time:<8.1f}")
        
        print("=" * 80)


def main():
    """CLI for retrieval evaluation."""
    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    
    # Required arguments
    parser.add_argument("--faiss-dir", type=Path, required=True, help="FAISS index directory")
    parser.add_argument("--qacite-dir", type=Path, required=True, help="Q-A-Cite dataset directory")
    
    # Configuration arguments
    parser.add_argument("--with-rerank", action="store_true", help="Enable reranking")
    parser.add_argument("--with-expansion", type=bool, default=True, help="Enable query expansion")
    parser.add_argument("--diversify", type=bool, default=True, help="Enable diversification")
    parser.add_argument("--same-lang-boost", type=float, default=0.15, help="Same language boost")
    parser.add_argument("--authority-boost", type=float, default=0.08, help="Authority boost")
    parser.add_argument("--ocr-penalty", type=float, default=0.1, help="OCR penalty")
    parser.add_argument("--tabular-boost", type=float, default=0.05, help="Tabular boost")
    parser.add_argument("--per-doc-cap", type=int, default=5, help="Per-document result cap")
    parser.add_argument("--per-page-cap", type=int, default=3, help="Per-page result cap")
    parser.add_argument("--nprobe", type=int, default=10, help="FAISS nprobe")
    parser.add_argument("--efSearch", type=int, default=40, help="FAISS efSearch")
    parser.add_argument("--k", type=int, default=20, help="Number of results to retrieve")
    
    # Output arguments
    parser.add_argument("--output-dir", type=Path, help="Output directory for results")
    parser.add_argument("--csv-output", type=Path, help="CSV output file")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--max-queries", type=int, help="Maximum number of queries to evaluate")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base", help="Reranker model")
    parser.add_argument("--cache-dir", type=Path, help="Cache directory")
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = RetrievalEvaluator(
            faiss_dir=args.faiss_dir,
            qacite_dir=args.qacite_dir,
            reranker_model=args.reranker_model,
            cache_dir=args.cache_dir
        )
        
        # Load dataset
        dataset = evaluator.load_qacite_dataset()
        
        if args.ablation:
            # Run ablation study
            results = evaluator.run_ablation_study(dataset, args.output_dir)
            evaluator.print_comparison_table(results)
        else:
            # Single configuration evaluation
            config = {
                'with_rerank': args.with_rerank,
                'expand_query': args.with_expansion,
                'diversify': args.diversify,
                'same_lang_boost': args.same_lang_boost,
                'authority_boost': args.authority_boost,
                'k': args.k,
                'nprobe': args.nprobe,
                'efSearch': args.efSearch
            }
            
            metrics = evaluator.evaluate_configuration(config, dataset, args.output_dir)
            
            print(f"\n📊 EVALUATION RESULTS")
            print("=" * 50)
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not key.startswith('config'):
                    print(f"  {key}: {value:.4f}")
        
        print(f"\n✅ Evaluation complete")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
