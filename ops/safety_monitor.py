#!/usr/bin/env python3
"""
Safety Monitor for NepaliGov-RAG-Bench Incremental Operations

Implements safety checks, regression detection, and automated rollback
mechanisms for incremental re-indexing operations.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import shutil
import yaml


class SafetyMonitor:
    """Monitors system safety and handles rollbacks."""
    
    def __init__(
        self,
        baseline_metrics_path: Path = Path("data/metrics/baseline_metrics.json"),
        safety_config_path: Path = Path("config/safety_config.yaml")
    ):
        """
        Initialize safety monitor.
        
        Args:
            baseline_metrics_path: Path to baseline metrics file
            safety_config_path: Path to safety configuration
        """
        self.baseline_metrics_path = baseline_metrics_path
        self.safety_config_path = safety_config_path
        self.safety_config = self._load_safety_config()
    
    def _load_safety_config(self) -> Dict[str, Any]:
        """Load safety configuration or create default."""
        if self.safety_config_path.exists():
            with open(self.safety_config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # Default safety configuration
            default_config = {
                "regression_thresholds": {
                    "authority_purity": 0.05,      # 5% regression tolerance
                    "citation_precision": 0.05,
                    "citation_recall": 0.05,
                    "ndcg@10": 0.10,              # 10% tolerance for retrieval
                    "recall@10": 0.10,
                    "same_lang_citation_rate": 0.05,
                    "cross_lang_success@k": 0.05
                },
                "critical_metrics": [
                    "authority_purity",
                    "citation_precision", 
                    "citation_recall"
                ],
                "rollback_policy": {
                    "auto_rollback_on_critical": True,
                    "require_approval_threshold": 0.10,  # 10% regression needs approval
                    "max_rollback_attempts": 3
                },
                "backup_retention": {
                    "keep_last_n_backups": 10,
                    "max_age_days": 30
                }
            }
            
            # Save default config
            self.safety_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.safety_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            return default_config
    
    def load_baseline_metrics(self) -> Optional[Dict[str, float]]:
        """Load baseline metrics for comparison."""
        if not self.baseline_metrics_path.exists():
            return None
        
        try:
            with open(self.baseline_metrics_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load baseline metrics: {e}")
            return None
    
    def save_baseline_metrics(self, metrics: Dict[str, float]) -> None:
        """Save new baseline metrics."""
        self.baseline_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.baseline_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
    
    def detect_regressions(self, current_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Detect metric regressions compared to baseline.
        
        Args:
            current_metrics: Current evaluation metrics
            
        Returns:
            List of regression details
        """
        baseline_metrics = self.load_baseline_metrics()
        if baseline_metrics is None:
            print("No baseline metrics available for regression detection")
            return []
        
        regressions = []
        thresholds = self.safety_config["regression_thresholds"]
        
        for metric_name, threshold in thresholds.items():
            if metric_name in current_metrics and metric_name in baseline_metrics:
                current = current_metrics[metric_name]
                baseline = baseline_metrics[metric_name]
                
                # Calculate regression (negative change is bad)
                if baseline > 0:
                    regression_ratio = (baseline - current) / baseline
                    
                    if regression_ratio > threshold:
                        severity = self._classify_regression_severity(
                            metric_name, regression_ratio, threshold
                        )
                        
                        regressions.append({
                            "metric": metric_name,
                            "baseline": baseline,
                            "current": current,
                            "regression_ratio": regression_ratio,
                            "regression_percent": regression_ratio * 100,
                            "threshold": threshold,
                            "severity": severity,
                            "is_critical": metric_name in self.safety_config["critical_metrics"]
                        })
        
        return regressions
    
    def _classify_regression_severity(
        self, 
        metric_name: str, 
        regression_ratio: float, 
        threshold: float
    ) -> str:
        """Classify regression severity."""
        approval_threshold = self.safety_config["rollback_policy"]["require_approval_threshold"]
        
        if regression_ratio > approval_threshold:
            return "severe"
        elif regression_ratio > threshold * 2:
            return "moderate"
        else:
            return "minor"
    
    def should_auto_rollback(self, regressions: List[Dict[str, Any]]) -> bool:
        """
        Determine if automatic rollback should be triggered.
        
        Args:
            regressions: List of detected regressions
            
        Returns:
            True if auto rollback should be triggered
        """
        if not regressions:
            return False
        
        policy = self.safety_config["rollback_policy"]
        
        if not policy["auto_rollback_on_critical"]:
            return False
        
        # Check for critical metric regressions
        critical_regressions = [
            r for r in regressions 
            if r["is_critical"] and r["severity"] in ["moderate", "severe"]
        ]
        
        return len(critical_regressions) > 0
    
    def requires_approval(self, regressions: List[Dict[str, Any]]) -> bool:
        """
        Check if regressions require manual approval to proceed.
        
        Args:
            regressions: List of detected regressions
            
        Returns:
            True if manual approval is required
        """
        approval_threshold = self.safety_config["rollback_policy"]["require_approval_threshold"]
        
        severe_regressions = [
            r for r in regressions
            if r["regression_ratio"] > approval_threshold
        ]
        
        return len(severe_regressions) > 0
    
    def create_safety_report(
        self, 
        regressions: List[Dict[str, Any]],
        current_metrics: Dict[str, float],
        baseline_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive safety report.
        
        Args:
            regressions: Detected regressions
            current_metrics: Current metrics
            baseline_metrics: Baseline metrics for comparison
            
        Returns:
            Safety report dictionary
        """
        if baseline_metrics is None:
            baseline_metrics = self.load_baseline_metrics() or {}
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_regressions": len(regressions),
                "critical_regressions": len([r for r in regressions if r["is_critical"]]),
                "severe_regressions": len([r for r in regressions if r["severity"] == "severe"]),
                "auto_rollback_triggered": self.should_auto_rollback(regressions),
                "requires_approval": self.requires_approval(regressions)
            },
            "regressions": regressions,
            "metrics_comparison": {
                "current": current_metrics,
                "baseline": baseline_metrics,
                "changes": {
                    metric: {
                        "absolute": current_metrics.get(metric, 0) - baseline_metrics.get(metric, 0),
                        "relative": ((current_metrics.get(metric, 0) - baseline_metrics.get(metric, 0)) / 
                                   max(baseline_metrics.get(metric, 1), 0.001)) * 100
                    }
                    for metric in set(current_metrics.keys()) | set(baseline_metrics.keys())
                }
            },
            "recommendations": self._generate_recommendations(regressions)
        }
        
        return report
    
    def _generate_recommendations(self, regressions: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on regressions."""
        recommendations = []
        
        if not regressions:
            recommendations.append("✅ No regressions detected. Safe to proceed.")
            return recommendations
        
        critical_regressions = [r for r in regressions if r["is_critical"]]
        severe_regressions = [r for r in regressions if r["severity"] == "severe"]
        
        if critical_regressions:
            recommendations.append(
                f"🚨 {len(critical_regressions)} critical metric(s) regressed. "
                "Consider rolling back or investigating root cause."
            )
        
        if severe_regressions:
            recommendations.append(
                f"⚠️ {len(severe_regressions)} metric(s) show severe regression. "
                "Manual review recommended before proceeding."
            )
        
        # Specific recommendations by metric
        for regression in regressions:
            metric = regression["metric"]
            severity = regression["severity"]
            
            if metric == "authority_purity" and severity in ["moderate", "severe"]:
                recommendations.append(
                    "🔍 Authority purity regression detected. Check if distractor documents "
                    "are being ranked higher than authoritative sources."
                )
            elif metric == "citation_precision" and severity in ["moderate", "severe"]:
                recommendations.append(
                    "📝 Citation precision regression. Verify citation extraction and "
                    "bbox mapping accuracy."
                )
            elif metric == "citation_recall" and severity in ["moderate", "severe"]:
                recommendations.append(
                    "🔎 Citation recall regression. Check if relevant citations are "
                    "being missed or filtered out."
                )
        
        if self.should_auto_rollback(regressions):
            recommendations.append("🔄 Auto-rollback will be triggered due to critical regressions.")
        elif self.requires_approval(regressions):
            recommendations.append("✋ Manual approval required before proceeding.")
        
        return recommendations
    
    def save_safety_report(self, report: Dict[str, Any], report_path: Optional[Path] = None) -> Path:
        """Save safety report to file."""
        if report_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = Path(f"ops/safety_reports/safety_report_{timestamp}.json")
        
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report_path
    
    def cleanup_old_backups(self) -> None:
        """Clean up old backup files according to retention policy."""
        retention = self.safety_config["backup_retention"]
        max_age = timedelta(days=retention["max_age_days"])
        keep_last_n = retention["keep_last_n_backups"]
        
        # Find backup files
        backup_patterns = [
            "ops/lineage_index.*.bak",
            "data/faiss/index.*.bak",
            "data/metrics/baseline_metrics.*.bak"
        ]
        
        all_backups = []
        for pattern in backup_patterns:
            all_backups.extend(Path(".").glob(pattern))
        
        if not all_backups:
            return
        
        # Sort by modification time (newest first)
        all_backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Keep the most recent N backups
        to_keep = all_backups[:keep_last_n]
        to_check = all_backups[keep_last_n:]
        
        # Remove old backups
        now = datetime.now()
        removed_count = 0
        
        for backup_path in to_check:
            try:
                backup_time = datetime.fromtimestamp(backup_path.stat().st_mtime)
                if now - backup_time > max_age:
                    backup_path.unlink()
                    removed_count += 1
            except Exception as e:
                print(f"Warning: Could not remove backup {backup_path}: {e}")
        
        if removed_count > 0:
            print(f"🧹 Cleaned up {removed_count} old backup files")


def main():
    """CLI for safety monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Safety Monitor for Incremental Operations")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Check command
    check_parser = subparsers.add_parser("check", help="Check for regressions")
    check_parser.add_argument("--metrics", type=Path, required=True, help="Current metrics file")
    check_parser.add_argument("--baseline", type=Path, help="Baseline metrics file")
    check_parser.add_argument("--report", type=Path, help="Output safety report path")
    
    # Update baseline command  
    baseline_parser = subparsers.add_parser("update-baseline", help="Update baseline metrics")
    baseline_parser.add_argument("--metrics", type=Path, required=True, help="New baseline metrics file")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old backups")
    
    args = parser.parse_args()
    
    monitor = SafetyMonitor()
    
    if args.command == "check":
        # Load current metrics
        with open(args.metrics, 'r') as f:
            current_metrics = json.load(f)
        
        # Load baseline if specified
        baseline_metrics = None
        if args.baseline:
            with open(args.baseline, 'r') as f:
                baseline_metrics = json.load(f)
        
        # Detect regressions
        regressions = monitor.detect_regressions(current_metrics)
        
        # Create safety report
        report = monitor.create_safety_report(regressions, current_metrics, baseline_metrics)
        
        # Save report
        report_path = monitor.save_safety_report(report, args.report)
        print(f"Safety report saved: {report_path}")
        
        # Print summary
        print(f"\n📊 Safety Check Summary:")
        print(f"  Total regressions: {report['summary']['total_regressions']}")
        print(f"  Critical regressions: {report['summary']['critical_regressions']}")
        print(f"  Auto-rollback triggered: {report['summary']['auto_rollback_triggered']}")
        print(f"  Requires approval: {report['summary']['requires_approval']}")
        
        # Print recommendations
        print(f"\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"  {rec}")
        
        # Exit with appropriate code
        if report['summary']['auto_rollback_triggered']:
            sys.exit(2)  # Rollback needed
        elif report['summary']['requires_approval']:
            sys.exit(1)  # Approval needed
        else:
            sys.exit(0)  # Safe to proceed
    
    elif args.command == "update-baseline":
        with open(args.metrics, 'r') as f:
            new_metrics = json.load(f)
        
        monitor.save_baseline_metrics(new_metrics)
        print(f"✅ Baseline metrics updated from {args.metrics}")
    
    elif args.command == "cleanup":
        monitor.cleanup_old_backups()
        print("✅ Backup cleanup completed")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
