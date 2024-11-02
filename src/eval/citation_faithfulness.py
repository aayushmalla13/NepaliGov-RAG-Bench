#!/usr/bin/env python3
"""
Citation Faithfulness Evaluation with Language Analytics

Evaluates citation precision/recall, faithfulness, and distractor attack resistance
with comprehensive language analytics passthrough from CP9 bilingual intelligence.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CitationMetrics:
    """Citation evaluation metrics."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    char_iou: float = 0.0
    bbox_iou: float = 0.0
    valid_citations: int = 0
    total_citations: int = 0
    authoritative_citations: int = 0
    distractor_citations: int = 0


@dataclass
class FaithfulnessMetrics:
    """Faithfulness evaluation metrics."""
    faithfulness_score: float = 0.0
    supported_claims: int = 0
    total_claims: int = 0
    distractor_attack_rate: float = 0.0
    refusal_rate: float = 0.0
    refusal_at_k: Dict[int, float] = field(default_factory=dict)


@dataclass
class LanguageMetrics:
    """Language-specific evaluation metrics."""
    language_consistency_rate: float = 0.0
    same_lang_citation_rate: float = 0.0
    cross_lang_success_at_5: float = 0.0
    cross_lang_success_at_10: float = 0.0
    
    # CP9 Language Analytics Passthrough
    processing_strategy_distribution: Dict[str, float] = field(default_factory=dict)
    query_domain_distribution: Dict[str, float] = field(default_factory=dict)
    bilingual_confidence_histogram: Dict[str, int] = field(default_factory=dict)
    semantic_mappings_count_stats: Dict[str, float] = field(default_factory=dict)
    language_segments_count_stats: Dict[str, float] = field(default_factory=dict)
    domain_specific_success_rates: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    citation_metrics: CitationMetrics
    faithfulness_metrics: FaithfulnessMetrics
    language_metrics: LanguageMetrics
    sample_count: int = 0
    evaluation_metadata: Dict[str, Any] = field(default_factory=dict)


class CitationValidator:
    """Validates citation spans and calculates IoU metrics."""
    
    def __init__(self):
        self.citation_pattern = re.compile(r'\[\[doc:([^|]+)\|page:(\d+)\|span:(\d+):(\d+)\]\]')
    
    def extract_citations_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract citation tokens from answer text."""
        citations = []
        matches = self.citation_pattern.findall(text)
        
        for match in matches:
            doc_id, page_num, start_char, end_char = match
            citations.append({
                'doc_id': doc_id,
                'page_num': int(page_num),
                'start_char': int(start_char),
                'end_char': int(end_char)
            })
        
        return citations
    
    def calculate_char_iou(self, 
                          predicted_span: Dict[str, Any], 
                          ground_truth_spans: List[Dict[str, Any]]) -> float:
        """Calculate character-level IoU between predicted and ground truth spans."""
        if not ground_truth_spans:
            return 0.0
        
        pred_start = predicted_span['start_char']
        pred_end = predicted_span['end_char']
        
        max_iou = 0.0
        for gt_span in ground_truth_spans:
            # Check if same document and page
            if (predicted_span['doc_id'] != gt_span.get('doc_id') or 
                predicted_span['page_num'] != gt_span.get('page_num')):
                continue
            
            gt_start = gt_span.get('start_char', 0)
            gt_end = gt_span.get('end_char', 0)
            
            # Calculate IoU
            intersection_start = max(pred_start, gt_start)
            intersection_end = min(pred_end, gt_end)
            intersection = max(0, intersection_end - intersection_start)
            
            union_start = min(pred_start, gt_start)
            union_end = max(pred_end, gt_end)
            union = union_end - union_start
            
            iou = intersection / union if union > 0 else 0.0
            max_iou = max(max_iou, iou)
        
        return max_iou
    
    def calculate_bbox_iou(self, 
                          predicted_bbox: Optional[List[float]], 
                          ground_truth_bboxes: List[Optional[List[float]]]) -> float:
        """Calculate bounding box IoU."""
        if not predicted_bbox or not ground_truth_bboxes:
            return 0.0
        
        max_iou = 0.0
        for gt_bbox in ground_truth_bboxes:
            if not gt_bbox or len(gt_bbox) < 4 or len(predicted_bbox) < 4:
                continue
            
            # Calculate IoU for bounding boxes [x1, y1, x2, y2]
            pred_x1, pred_y1, pred_x2, pred_y2 = predicted_bbox[:4]
            gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox[:4]
            
            # Calculate intersection
            intersection_x1 = max(pred_x1, gt_x1)
            intersection_y1 = max(pred_y1, gt_y1)
            intersection_x2 = min(pred_x2, gt_x2)
            intersection_y2 = min(pred_y2, gt_y2)
            
            intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
            
            # Calculate union
            pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
            gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
            union_area = pred_area + gt_area - intersection_area
            
            iou = intersection_area / union_area if union_area > 0 else 0.0
            max_iou = max(max_iou, iou)
        
        return max_iou
    
    def validate_citation_authority(self, 
                                   citation: Dict[str, Any], 
                                   candidates: List[Dict[str, Any]]) -> bool:
        """Validate that citation points to authoritative content."""
        for candidate in candidates:
            if (candidate.get('doc_id') == citation['doc_id'] and
                candidate.get('page_num') == citation['page_num'] and
                candidate.get('is_authoritative', False)):
                return True
        return False


class FaithfulnessEvaluator:
    """Evaluates answer faithfulness and distractor resistance."""
    
    def __init__(self):
        self.citation_validator = CitationValidator()
        self.refusal_patterns = {
            'en': [
                "don't have enough authoritative evidence",
                "insufficient evidence",
                "cannot answer",
                "not enough information"
            ],
            'ne': [
                "पर्याप्त आधिकारिक प्रमाण नहुँदा",
                "जवाफ दिन असमर्थ",
                "पर्याप्त जानकारी छैन"
            ]
        }
    
    def evaluate_faithfulness(self, 
                            answer_text: str, 
                            candidates: List[Dict[str, Any]],
                            ground_truth: Optional[Dict[str, Any]] = None) -> FaithfulnessMetrics:
        """Evaluate answer faithfulness."""
        
        # Check if answer is a refusal
        is_refusal = self._is_refusal(answer_text)
        
        if is_refusal:
            return FaithfulnessMetrics(
                faithfulness_score=1.0,  # Refusal is perfectly faithful
                refusal_rate=1.0
            )
        
        # Extract citations
        citations = self.citation_validator.extract_citations_from_text(answer_text)
        
        if not citations:
            # No citations = not faithful
            return FaithfulnessMetrics(
                faithfulness_score=0.0,
                total_claims=1,
                distractor_attack_rate=1.0  # Uncited claims are distractor attacks
            )
        
        # Validate citations
        authoritative_citations = 0
        distractor_citations = 0
        
        for citation in citations:
            if self.citation_validator.validate_citation_authority(citation, candidates):
                authoritative_citations += 1
            else:
                distractor_citations += 1
        
        total_citations = len(citations)
        
        # Calculate faithfulness score
        if total_citations > 0:
            faithfulness_score = authoritative_citations / total_citations
            distractor_attack_rate = distractor_citations / total_citations
        else:
            faithfulness_score = 0.0
            distractor_attack_rate = 1.0
        
        # Estimate claims (simple heuristic: sentences)
        sentences = len([s for s in answer_text.split('.') if s.strip()])
        supported_claims = min(sentences, authoritative_citations)
        
        return FaithfulnessMetrics(
            faithfulness_score=faithfulness_score,
            supported_claims=supported_claims,
            total_claims=sentences,
            distractor_attack_rate=distractor_attack_rate,
            refusal_rate=0.0
        )
    
    def _is_refusal(self, text: str) -> bool:
        """Check if text is a refusal."""
        text_lower = text.lower()
        
        # Check English patterns
        for pattern in self.refusal_patterns['en']:
            if pattern.lower() in text_lower:
                return True
        
        # Check Nepali patterns
        for pattern in self.refusal_patterns['ne']:
            if pattern in text:
                return True
        
        return False
    
    def calculate_refusal_at_k(self, 
                              results: List[Dict[str, Any]], 
                              k_values: List[int] = [1, 3, 5, 10]) -> Dict[int, float]:
        """Calculate refusal rate at different k values."""
        refusal_at_k = {}
        
        for k in k_values:
            if not results:
                refusal_at_k[k] = 0.0
                continue
            
            refusal_count = 0
            for result in results[:k]:
                answer_text = result.get('answer_text', '')
                if self._is_refusal(answer_text):
                    refusal_count += 1
            
            refusal_at_k[k] = refusal_count / min(k, len(results))
        
        return refusal_at_k


class LanguageAnalytics:
    """Analyzes language-specific metrics and CP9 analytics passthrough."""
    
    def __init__(self):
        pass
    
    def analyze_language_metrics(self, 
                                results: List[Dict[str, Any]],
                                retrieval_metadata: List[Dict[str, Any]] = None) -> LanguageMetrics:
        """Analyze language metrics with CP9 analytics passthrough."""
        
        if not results:
            return LanguageMetrics()
        
        # Basic language metrics
        total_results = len(results)
        consistent_language = 0
        same_lang_citations = 0
        cross_lang_success_5 = 0
        cross_lang_success_10 = 0
        
        # CP9 Analytics aggregation
        processing_strategies = []
        query_domains = []
        bilingual_confidences = []
        semantic_mappings_counts = []
        language_segments_counts = []
        domain_success_rates = {}
        
        for i, result in enumerate(results):
            # Basic metrics
            query_lang = result.get('query_language', 'unknown')
            answer_lang = result.get('answer_language', 'unknown')
            
            if query_lang == answer_lang:
                consistent_language += 1
            
            # Citation analysis
            citations = result.get('citations', [])
            same_lang_count = sum(1 for c in citations if c.get('language') == query_lang)
            if citations:
                same_lang_citations += same_lang_count / len(citations)
            
            # Cross-language success
            cross_lang_count_5 = sum(1 for c in citations[:5] if c.get('language') != query_lang)
            cross_lang_count_10 = sum(1 for c in citations[:10] if c.get('language') != query_lang)
            
            if len(citations) >= 5:
                cross_lang_success_5 += cross_lang_count_5 / 5
            if len(citations) >= 10:
                cross_lang_success_10 += cross_lang_count_10 / 10
            
            # CP9 Analytics passthrough
            if retrieval_metadata and i < len(retrieval_metadata):
                metadata = retrieval_metadata[i]
                lang_metrics = metadata.get('language_metrics', {})
                
                # Collect analytics data
                processing_strategies.append(lang_metrics.get('processing_strategy', 'unknown'))
                query_domains.append(lang_metrics.get('query_domain', 'general'))
                bilingual_confidences.append(lang_metrics.get('bilingual_confidence', 0.0))
                semantic_mappings_counts.append(lang_metrics.get('semantic_mappings_count', 0))
                language_segments_counts.append(lang_metrics.get('language_segments_count', 1))
                
                # Domain-specific success rates
                domain = lang_metrics.get('query_domain', 'general')
                domain_success_rate = lang_metrics.get(f'{domain}_success_rate', 0.0)
                if domain not in domain_success_rates:
                    domain_success_rates[domain] = []
                domain_success_rates[domain].append(domain_success_rate)
        
        # Calculate averages
        language_consistency_rate = consistent_language / total_results
        same_lang_citation_rate = same_lang_citations / total_results
        cross_lang_success_at_5 = cross_lang_success_5 / total_results
        cross_lang_success_at_10 = cross_lang_success_10 / total_results
        
        # Process CP9 analytics
        processing_strategy_dist = self._calculate_distribution(processing_strategies)
        query_domain_dist = self._calculate_distribution(query_domains)
        bilingual_confidence_hist = self._calculate_histogram(bilingual_confidences)
        semantic_mappings_stats = self._calculate_stats(semantic_mappings_counts)
        language_segments_stats = self._calculate_stats(language_segments_counts)
        domain_specific_rates = {
            domain: np.mean(rates) for domain, rates in domain_success_rates.items()
        }
        
        return LanguageMetrics(
            language_consistency_rate=language_consistency_rate,
            same_lang_citation_rate=same_lang_citation_rate,
            cross_lang_success_at_5=cross_lang_success_at_5,
            cross_lang_success_at_10=cross_lang_success_at_10,
            processing_strategy_distribution=processing_strategy_dist,
            query_domain_distribution=query_domain_dist,
            bilingual_confidence_histogram=bilingual_confidence_hist,
            semantic_mappings_count_stats=semantic_mappings_stats,
            language_segments_count_stats=language_segments_stats,
            domain_specific_success_rates=domain_specific_rates
        )
    
    def _calculate_distribution(self, values: List[str]) -> Dict[str, float]:
        """Calculate distribution of categorical values."""
        if not values:
            return {}
        
        from collections import Counter
        counts = Counter(values)
        total = len(values)
        
        return {k: v / total for k, v in counts.items()}
    
    def _calculate_histogram(self, values: List[float], bins: int = 5) -> Dict[str, int]:
        """Calculate histogram of continuous values."""
        if not values:
            return {}
        
        # Create bins
        min_val = min(values)
        max_val = max(values)
        if min_val == max_val:
            return {f"{min_val:.2f}": len(values)}
        
        bin_width = (max_val - min_val) / bins
        histogram = {}
        
        for i in range(bins):
            bin_start = min_val + i * bin_width
            bin_end = min_val + (i + 1) * bin_width
            bin_label = f"{bin_start:.2f}-{bin_end:.2f}"
            
            count = sum(1 for v in values if bin_start <= v < bin_end)
            if i == bins - 1:  # Include max value in last bin
                count = sum(1 for v in values if bin_start <= v <= bin_end)
            
            histogram[bin_label] = count
        
        return histogram
    
    def _calculate_stats(self, values: List[Union[int, float]]) -> Dict[str, float]:
        """Calculate basic statistics for numerical values."""
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }


class CitationFaithfulnessEvaluator:
    """Main evaluator for citation faithfulness with language analytics."""
    
    def __init__(self):
        self.citation_validator = CitationValidator()
        self.faithfulness_evaluator = FaithfulnessEvaluator()
        self.language_analytics = LanguageAnalytics()
    
    def evaluate(self, 
                results: List[Dict[str, Any]],
                ground_truth: Optional[List[Dict[str, Any]]] = None,
                retrieval_metadata: Optional[List[Dict[str, Any]]] = None,
                iou_threshold: float = 0.5) -> EvaluationResult:
        """
        Evaluate citation faithfulness with comprehensive metrics.
        
        Args:
            results: List of answer results with citations
            ground_truth: Optional ground truth for validation
            retrieval_metadata: CP9 retrieval metadata for analytics
            iou_threshold: IoU threshold for citation validation
        
        Returns:
            Complete evaluation result
        """
        
        if not results:
            return EvaluationResult(
                citation_metrics=CitationMetrics(),
                faithfulness_metrics=FaithfulnessMetrics(),
                language_metrics=LanguageMetrics()
            )
        
        # Evaluate citations
        citation_metrics = self._evaluate_citations(results, ground_truth, iou_threshold)
        
        # Evaluate faithfulness
        faithfulness_metrics = self._evaluate_faithfulness(results)
        
        # Analyze language metrics
        language_metrics = self.language_analytics.analyze_language_metrics(
            results, retrieval_metadata
        )
        
        return EvaluationResult(
            citation_metrics=citation_metrics,
            faithfulness_metrics=faithfulness_metrics,
            language_metrics=language_metrics,
            sample_count=len(results),
            evaluation_metadata={
                'iou_threshold': iou_threshold,
                'has_ground_truth': ground_truth is not None,
                'has_retrieval_metadata': retrieval_metadata is not None
            }
        )
    
    def _evaluate_citations(self, 
                           results: List[Dict[str, Any]], 
                           ground_truth: Optional[List[Dict[str, Any]]], 
                           iou_threshold: float) -> CitationMetrics:
        """Evaluate citation precision and recall."""
        
        total_citations = 0
        valid_citations = 0
        authoritative_citations = 0
        distractor_citations = 0
        total_char_iou = 0.0
        total_bbox_iou = 0.0
        citation_count = 0
        
        for i, result in enumerate(results):
            answer_text = result.get('answer_text', '')
            candidates = result.get('candidates', [])
            
            # Extract citations from answer
            citations = self.citation_validator.extract_citations_from_text(answer_text)
            total_citations += len(citations)
            
            for citation in citations:
                citation_count += 1
                
                # Validate authority
                is_authoritative = self.citation_validator.validate_citation_authority(
                    citation, candidates
                )
                
                if is_authoritative:
                    authoritative_citations += 1
                    valid_citations += 1
                else:
                    distractor_citations += 1
                
                # Calculate IoU if ground truth available
                if ground_truth and i < len(ground_truth):
                    gt_spans = ground_truth[i].get('spans', [])
                    
                    # Character IoU
                    char_iou = self.citation_validator.calculate_char_iou(citation, gt_spans)
                    total_char_iou += char_iou
                    
                    # Bbox IoU (if available)
                    citation_bbox = None
                    for candidate in candidates:
                        if (candidate.get('doc_id') == citation['doc_id'] and
                            candidate.get('page_num') == citation['page_num']):
                            citation_bbox = candidate.get('bbox')
                            break
                    
                    if citation_bbox:
                        gt_bboxes = [span.get('bbox') for span in gt_spans if span.get('bbox')]
                        bbox_iou = self.citation_validator.calculate_bbox_iou(citation_bbox, gt_bboxes)
                        total_bbox_iou += bbox_iou
        
        # Calculate metrics
        precision = authoritative_citations / total_citations if total_citations > 0 else 0.0
        recall = valid_citations / max(1, citation_count)  # Simplified recall calculation
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        avg_char_iou = total_char_iou / citation_count if citation_count > 0 else 0.0
        avg_bbox_iou = total_bbox_iou / citation_count if citation_count > 0 else 0.0
        
        return CitationMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            char_iou=avg_char_iou,
            bbox_iou=avg_bbox_iou,
            valid_citations=valid_citations,
            total_citations=total_citations,
            authoritative_citations=authoritative_citations,
            distractor_citations=distractor_citations
        )
    
    def _evaluate_faithfulness(self, results: List[Dict[str, Any]]) -> FaithfulnessMetrics:
        """Evaluate overall faithfulness metrics."""
        
        total_faithfulness = 0.0
        total_supported_claims = 0
        total_claims = 0
        total_distractor_attacks = 0.0
        refusal_count = 0
        
        for result in results:
            answer_text = result.get('answer_text', '')
            candidates = result.get('candidates', [])
            
            faithfulness = self.faithfulness_evaluator.evaluate_faithfulness(
                answer_text, candidates
            )
            
            total_faithfulness += faithfulness.faithfulness_score
            total_supported_claims += faithfulness.supported_claims
            total_claims += faithfulness.total_claims
            total_distractor_attacks += faithfulness.distractor_attack_rate
            
            if faithfulness.refusal_rate > 0:
                refusal_count += 1
        
        sample_count = len(results)
        
        # Calculate refusal@k
        refusal_at_k = self.faithfulness_evaluator.calculate_refusal_at_k(results)
        
        return FaithfulnessMetrics(
            faithfulness_score=total_faithfulness / sample_count,
            supported_claims=total_supported_claims,
            total_claims=total_claims,
            distractor_attack_rate=total_distractor_attacks / sample_count,
            refusal_rate=refusal_count / sample_count,
            refusal_at_k=refusal_at_k
        )


def main():
    """CLI for citation faithfulness evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Citation faithfulness evaluation")
    parser.add_argument("--results-file", required=True, help="JSON file with answer results")
    parser.add_argument("--ground-truth", help="Optional ground truth file")
    parser.add_argument("--retrieval-metadata", help="Optional retrieval metadata file")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    try:
        # Load results
        with open(args.results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Load optional files
        ground_truth = None
        if args.ground_truth:
            with open(args.ground_truth, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
        
        retrieval_metadata = None
        if args.retrieval_metadata:
            with open(args.retrieval_metadata, 'r', encoding='utf-8') as f:
                retrieval_metadata = json.load(f)
        
        # Evaluate
        evaluator = CitationFaithfulnessEvaluator()
        evaluation_result = evaluator.evaluate(
            results, ground_truth, retrieval_metadata, args.iou_threshold
        )
        
        # Display results
        print("=== CITATION METRICS ===")
        print(f"Precision: {evaluation_result.citation_metrics.precision:.3f}")
        print(f"Recall: {evaluation_result.citation_metrics.recall:.3f}")
        print(f"F1: {evaluation_result.citation_metrics.f1:.3f}")
        print(f"Char IoU: {evaluation_result.citation_metrics.char_iou:.3f}")
        print(f"Bbox IoU: {evaluation_result.citation_metrics.bbox_iou:.3f}")
        
        print("\n=== FAITHFULNESS METRICS ===")
        print(f"Faithfulness: {evaluation_result.faithfulness_metrics.faithfulness_score:.3f}")
        print(f"Distractor Attack Rate: {evaluation_result.faithfulness_metrics.distractor_attack_rate:.3f}")
        print(f"Refusal Rate: {evaluation_result.faithfulness_metrics.refusal_rate:.3f}")
        
        print("\n=== LANGUAGE METRICS ===")
        print(f"Language Consistency: {evaluation_result.language_metrics.language_consistency_rate:.3f}")
        print(f"Same Lang Citation Rate: {evaluation_result.language_metrics.same_lang_citation_rate:.3f}")
        print(f"Cross Lang Success@5: {evaluation_result.language_metrics.cross_lang_success_at_5:.3f}")
        print(f"Cross Lang Success@10: {evaluation_result.language_metrics.cross_lang_success_at_10:.3f}")
        
        # CP9 Analytics
        print("\n=== CP9 LANGUAGE ANALYTICS ===")
        print(f"Processing Strategies: {evaluation_result.language_metrics.processing_strategy_distribution}")
        print(f"Query Domains: {evaluation_result.language_metrics.query_domain_distribution}")
        print(f"Domain Success Rates: {evaluation_result.language_metrics.domain_specific_success_rates}")
        
        # Save results if requested
        if args.output:
            output_data = {
                'citation_metrics': evaluation_result.citation_metrics.__dict__,
                'faithfulness_metrics': evaluation_result.faithfulness_metrics.__dict__,
                'language_metrics': evaluation_result.language_metrics.__dict__,
                'evaluation_metadata': evaluation_result.evaluation_metadata
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"\nResults saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


