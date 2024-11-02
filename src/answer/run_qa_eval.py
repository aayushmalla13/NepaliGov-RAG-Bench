#!/usr/bin/env python3
"""
Q&A Evaluation Runner with Advanced Bilingual Features

Orchestrates end-to-end Q&A evaluation with CP6 retrieval features (expansion, 
diversification, priors) and CP9 bilingual intelligence integration.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from dataclasses import asdict

# Local imports
from src.retriever.search import MultilingualRetriever
from src.answer.answerer import AnswerGenerator, AnswerContext
from src.eval.citation_faithfulness import CitationFaithfulnessEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QAEvaluationRunner:
    """Runs comprehensive Q&A evaluation with bilingual intelligence."""
    
    def __init__(self, 
                 faiss_dir: Path,
                 qa_dataset_path: Path,
                 output_dir: Path):
        """
        Initialize Q&A evaluation runner.
        
        Args:
            faiss_dir: Path to FAISS index directory
            qa_dataset_path: Path to Q&A dataset (JSONL)
            output_dir: Output directory for results
        """
        self.faiss_dir = Path(faiss_dir)
        self.qa_dataset_path = Path(qa_dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.retriever = MultilingualRetriever(self.faiss_dir)
        self.answer_generator = AnswerGenerator()
        self.faithfulness_evaluator = CitationFaithfulnessEvaluator()
        
        # Default configurations
        self.fallback_threshold_profiles = {
            'conservative': 0.7,
            'balanced': 0.5,
            'aggressive': 0.3,
            'adaptive': 'dynamic'  # Use CP9 dynamic calculation
        }
    
    def load_qa_dataset(self) -> List[Dict[str, Any]]:
        """Load Q&A dataset from JSONL file."""
        qa_samples = []
        
        with open(self.qa_dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    qa_samples.append(json.loads(line))
        
        logger.info(f"Loaded {len(qa_samples)} Q&A samples from {self.qa_dataset_path}")
        return qa_samples
    
    def run_evaluation(self,
                      k: int = 10,
                      allow_distractors: bool = False,
                      with_expansion: bool = True,
                      diversify: bool = True,
                      fallback_threshold_profile: str = 'adaptive',
                      table_context: bool = True,
                      output_mode: str = 'auto',
                      sample_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run comprehensive Q&A evaluation.
        
        Args:
            k: Number of candidates to retrieve
            allow_distractors: Whether to allow distractor candidates
            with_expansion: Whether to use query expansion
            diversify: Whether to apply result diversification
            fallback_threshold_profile: Fallback threshold profile
            table_context: Whether to include table context
            output_mode: Output language mode (auto/en/ne/bilingual)
            sample_limit: Limit number of samples for testing
            
        Returns:
            Evaluation results dictionary
        """
        
        # Load dataset
        qa_samples = self.load_qa_dataset()
        if sample_limit:
            qa_samples = qa_samples[:sample_limit]
            logger.info(f"Limited to {len(qa_samples)} samples for evaluation")
        
        # Process samples
        results = []
        retrieval_metadata = []
        
        logger.info(f"Processing {len(qa_samples)} Q&A samples...")
        
        for i, qa_sample in enumerate(qa_samples):
            if i % 10 == 0:
                logger.info(f"Processing sample {i+1}/{len(qa_samples)}")
            
            try:
                result = self._process_qa_sample(
                    qa_sample, k, allow_distractors, with_expansion, diversify,
                    fallback_threshold_profile, table_context, output_mode
                )
                results.append(result['answer_result'])
                retrieval_metadata.append(result['retrieval_metadata'])
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                # Add empty result to maintain alignment
                results.append({
                    'query': qa_sample.get('query', ''),
                    'answer_text': 'ERROR: Processing failed',
                    'is_error': True,
                    'error_message': str(e)
                })
                retrieval_metadata.append({})
        
        # Evaluate results
        logger.info("Evaluating results...")
        evaluation_result = self.faithfulness_evaluator.evaluate(
            results, retrieval_metadata=retrieval_metadata
        )
        
        # Compile final results
        final_results = {
            'evaluation_config': {
                'k': k,
                'allow_distractors': allow_distractors,
                'with_expansion': with_expansion,
                'diversify': diversify,
                'fallback_threshold_profile': fallback_threshold_profile,
                'table_context': table_context,
                'output_mode': output_mode,
                'sample_count': len(qa_samples)
            },
            'citation_metrics': asdict(evaluation_result.citation_metrics),
            'faithfulness_metrics': asdict(evaluation_result.faithfulness_metrics),
            'language_metrics': asdict(evaluation_result.language_metrics),
            'evaluation_metadata': evaluation_result.evaluation_metadata,
            'sample_results': results,
            'retrieval_metadata': retrieval_metadata
        }
        
        return final_results
    
    def _process_qa_sample(self,
                          qa_sample: Dict[str, Any],
                          k: int,
                          allow_distractors: bool,
                          with_expansion: bool,
                          diversify: bool,
                          fallback_threshold_profile: str,
                          table_context: bool,
                          output_mode: str) -> Dict[str, Any]:
        """Process a single Q&A sample."""
        
        query = qa_sample['query']
        expected_answer = qa_sample.get('answer', '')
        query_lang = qa_sample.get('query_lang', 'auto')
        
        # Determine fallback threshold
        if fallback_threshold_profile == 'adaptive':
            # Use CP9 dynamic threshold (will be calculated by retriever)
            cross_lang_fallback_threshold = 0.5  # Base threshold
        else:
            cross_lang_fallback_threshold = self.fallback_threshold_profiles.get(
                fallback_threshold_profile, 0.5
            )
        
        # Retrieve candidates
        retrieval_results = self.retriever.search(
            query=query,
            k=k,
            query_lang=query_lang,
            output_mode=output_mode,
            allow_distractors=allow_distractors,
            inject_hard_negatives=0,  # No hard negatives for evaluation
            cross_lang_fallback_threshold=cross_lang_fallback_threshold
        )
        
        # Extract candidates and metadata
        candidates = retrieval_results.get('authoritative_candidates', [])
        if allow_distractors:
            candidates.extend(retrieval_results.get('distractors', []))
        
        # Create answer context from retrieval metadata
        language_metrics = retrieval_results.get('language_metrics', {})
        target_language = retrieval_results.get('target_language', 
                                               retrieval_results.get('target_languages', 'auto'))
        
        answer_context = AnswerContext(
            query=query,
            target_language=target_language,
            processing_strategy=language_metrics.get('processing_strategy', 'monolingual'),
            query_domain=language_metrics.get('query_domain', 'general'),
            bilingual_confidence=language_metrics.get('bilingual_confidence', 1.0),
            semantic_mappings_count=language_metrics.get('semantic_mappings_count', 0),
            language_segments_count=language_metrics.get('language_segments_count', 1),
            domain_success_rate=language_metrics.get(f"{language_metrics.get('query_domain', 'general')}_success_rate", 0.0),
            fallback_threshold=cross_lang_fallback_threshold
        )
        
        # Generate answer
        answer = self.answer_generator.generate_answer(
            candidates, answer_context, table_context
        )
        
        # Format result
        if hasattr(answer, 'english') and hasattr(answer, 'nepali'):
            # Bilingual answer
            answer_result = {
                'query': query,
                'query_language': query_lang,
                'target_language': target_language,
                'answer_text': self._format_bilingual_answer(answer),
                'answer_language': 'bilingual',
                'is_refusal': (answer.english and answer.english.is_refusal) or 
                             (answer.nepali and answer.nepali.is_refusal),
                'citations': [asdict(c) for c in answer.combined_citations],
                'candidates': candidates,
                'evidence_count': len(answer.combined_citations),
                'same_lang_evidence_count': sum(
                    (answer.english.same_lang_evidence_count if answer.english else 0),
                    (answer.nepali.same_lang_evidence_count if answer.nepali else 0)
                ),
                'cross_lang_evidence_count': sum(
                    (answer.english.cross_lang_evidence_count if answer.english else 0),
                    (answer.nepali.cross_lang_evidence_count if answer.nepali else 0)
                ),
                'table_context_used': any([
                    answer.english.table_context_used if answer.english else False,
                    answer.nepali.table_context_used if answer.nepali else False
                ]),
                'processing_metadata': answer.processing_metadata
            }
        else:
            # Single language answer
            answer_result = {
                'query': query,
                'query_language': query_lang,
                'target_language': target_language,
                'answer_text': answer.text,
                'answer_language': answer.language,
                'is_refusal': answer.is_refusal,
                'citations': [asdict(c) for c in answer.citations],
                'candidates': candidates,
                'evidence_count': answer.evidence_count,
                'same_lang_evidence_count': answer.same_lang_evidence_count,
                'cross_lang_evidence_count': answer.cross_lang_evidence_count,
                'table_context_used': answer.table_context_used
            }
        
        return {
            'answer_result': answer_result,
            'retrieval_metadata': {
                'retrieval_results': retrieval_results,
                'language_metrics': language_metrics,
                'authority_purity': retrieval_results.get('authority_purity', 0.0),
                'expansion_terms': retrieval_results.get('expansion_terms', {}),
                'cross_lang_fallback_used': retrieval_results.get('cross_lang_fallback_used', False)
            }
        }
    
    def _format_bilingual_answer(self, bilingual_answer) -> str:
        """Format bilingual answer for display."""
        parts = []
        
        if bilingual_answer.english and bilingual_answer.english.text:
            parts.append(f"[EN] {bilingual_answer.english.text}")
        
        if bilingual_answer.nepali and bilingual_answer.nepali.text:
            parts.append(f"[NE] {bilingual_answer.nepali.text}")
        
        return "\n\n".join(parts) if parts else "No answer generated"
    
    def save_results(self, results: Dict[str, Any], filename: str = "qa_evaluation_results.json"):
        """Save evaluation results to file."""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
        return output_path


def main():
    """CLI for Q&A evaluation runner."""
    parser = argparse.ArgumentParser(description="Q&A Evaluation Runner with Advanced Features")
    
    # Required arguments
    parser.add_argument("--faiss-dir", required=True, help="Path to FAISS index directory")
    parser.add_argument("--qa-dataset", required=True, help="Path to Q&A dataset (JSONL)")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    
    # Retrieval parameters
    parser.add_argument("--k", type=int, default=10, help="Number of candidates to retrieve")
    parser.add_argument("--allow-distractors", action="store_true", 
                       help="Allow distractor candidates in retrieval")
    parser.add_argument("--with-expansion", action="store_true", default=True,
                       help="Use query expansion")
    parser.add_argument("--diversify", action="store_true", default=True,
                       help="Apply result diversification")
    
    # Advanced parameters
    parser.add_argument("--fallback-thresholds", 
                       choices=['conservative', 'balanced', 'aggressive', 'adaptive'],
                       default='adaptive',
                       help="Fallback threshold profile")
    parser.add_argument("--table-context", 
                       choices=['on', 'off'], default='on',
                       help="Include table context")
    parser.add_argument("--output-mode", 
                       choices=['auto', 'en', 'ne', 'bilingual'],
                       default='auto',
                       help="Output language mode")
    
    # Evaluation parameters
    parser.add_argument("--sample-limit", type=int, help="Limit number of samples for testing")
    parser.add_argument("--output-file", default="qa_evaluation_results.json",
                       help="Output filename")
    
    args = parser.parse_args()
    
    try:
        # Initialize runner
        runner = QAEvaluationRunner(
            faiss_dir=Path(args.faiss_dir),
            qa_dataset_path=Path(args.qa_dataset),
            output_dir=Path(args.output_dir)
        )
        
        # Run evaluation
        results = runner.run_evaluation(
            k=args.k,
            allow_distractors=args.allow_distractors,
            with_expansion=args.with_expansion,
            diversify=args.diversify,
            fallback_threshold_profile=args.fallback_thresholds,
            table_context=(args.table_context == 'on'),
            output_mode=args.output_mode,
            sample_limit=args.sample_limit
        )
        
        # Save results
        output_path = runner.save_results(results, args.output_file)
        
        # Display summary
        print("\n=== Q&A EVALUATION SUMMARY ===")
        print(f"Dataset: {args.qa_dataset}")
        print(f"Samples processed: {results['evaluation_config']['sample_count']}")
        print(f"Output mode: {args.output_mode}")
        print(f"Fallback profile: {args.fallback_thresholds}")
        
        print("\n=== CITATION METRICS ===")
        cm = results['citation_metrics']
        print(f"Precision: {cm['precision']:.3f}")
        print(f"Recall: {cm['recall']:.3f}")
        print(f"F1: {cm['f1']:.3f}")
        print(f"Char IoU: {cm['char_iou']:.3f}")
        print(f"Bbox IoU: {cm['bbox_iou']:.3f}")
        
        print("\n=== FAITHFULNESS METRICS ===")
        fm = results['faithfulness_metrics']
        print(f"Faithfulness: {fm['faithfulness_score']:.3f}")
        print(f"Distractor Attack Rate: {fm['distractor_attack_rate']:.3f}")
        print(f"Refusal Rate: {fm['refusal_rate']:.3f}")
        
        print("\n=== LANGUAGE METRICS ===")
        lm = results['language_metrics']
        print(f"Language Consistency: {lm['language_consistency_rate']:.3f}")
        print(f"Same Lang Citation Rate: {lm['same_lang_citation_rate']:.3f}")
        print(f"Cross Lang Success@5: {lm['cross_lang_success_at_5']:.3f}")
        print(f"Cross Lang Success@10: {lm['cross_lang_success_at_10']:.3f}")
        
        print("\n=== CP9 LANGUAGE ANALYTICS ===")
        print(f"Processing Strategies: {lm['processing_strategy_distribution']}")
        print(f"Query Domains: {lm['query_domain_distribution']}")
        print(f"Domain Success Rates: {lm['domain_specific_success_rates']}")
        
        print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


