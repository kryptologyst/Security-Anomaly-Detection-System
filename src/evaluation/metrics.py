"""
Security-Focused Evaluation Metrics

Implements comprehensive evaluation metrics specifically designed for
anomaly detection in security applications, including AUCPR, precision@K,
FPR at target TPR, and operational metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class SecurityMetricsConfig:
    """Configuration for security metrics evaluation."""
    
    # Thresholds for operational metrics
    target_tpr: float = 0.95  # Target True Positive Rate
    target_fpr: float = 0.01  # Target False Positive Rate
    precision_k_values: List[int] = None  # K values for precision@K
    
    # Alert workload metrics
    max_alerts_per_day: int = 100
    investigation_time_minutes: int = 30
    
    def __post_init__(self):
        if self.precision_k_values is None:
            self.precision_k_values = [10, 50, 100, 500]


class SecurityMetrics:
    """
    Comprehensive evaluation metrics for security anomaly detection.
    
    Focuses on metrics relevant to security operations including
    precision@K, FPR at target TPR, and alert workload analysis.
    """
    
    def __init__(self, config: Optional[SecurityMetricsConfig] = None):
        """
        Initialize security metrics evaluator.
        
        Args:
            config: Configuration for metrics evaluation
        """
        self.config = config or SecurityMetricsConfig()
        self.metrics_history = []
    
    def compute_all(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all security-relevant metrics.
        
        Args:
            y_true: True binary labels (0=normal, 1=anomaly)
            y_pred: Predicted binary labels
            y_scores: Anomaly scores (optional)
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics.update(self._compute_basic_metrics(y_true, y_pred))
        
        # Precision@K metrics
        if y_scores is not None:
            metrics.update(self._compute_precision_at_k(y_true, y_scores))
        
        # ROC and PR curve metrics
        if y_scores is not None:
            metrics.update(self._compute_curve_metrics(y_true, y_scores))
        
        # Operational metrics
        metrics.update(self._compute_operational_metrics(y_true, y_pred, y_scores))
        
        # Alert workload analysis
        metrics.update(self._compute_workload_metrics(y_true, y_pred))
        
        # Store metrics history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _compute_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute basic classification metrics."""
        metrics = {}
        
        # Handle edge cases
        if len(np.unique(y_true)) == 1:
            logger.warning("Only one class present in y_true")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'specificity': 0.0,
                'accuracy': float(np.mean(y_pred == y_true))
            }
        
        if len(np.unique(y_pred)) == 1:
            logger.warning("Only one class present in y_pred")
        
        # Basic metrics
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['accuracy'] = np.mean(y_pred == y_true)
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Additional metrics
        metrics['true_positives'] = int(tp)
        metrics['false_positives'] = int(fp)
        metrics['true_negatives'] = int(tn)
        metrics['false_negatives'] = int(fn)
        
        return metrics
    
    def _compute_precision_at_k(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray
    ) -> Dict[str, float]:
        """Compute precision@K metrics."""
        metrics = {}
        
        # Sort by scores (descending)
        sorted_indices = np.argsort(y_scores)[::-1]
        sorted_labels = y_true[sorted_indices]
        
        for k in self.config.precision_k_values:
            if k > len(sorted_labels):
                k = len(sorted_labels)
            
            # Precision@K = TP / (TP + FP) in top K
            top_k_labels = sorted_labels[:k]
            precision_k = np.sum(top_k_labels) / k if k > 0 else 0.0
            metrics[f'precision_at_{k}'] = precision_k
        
        return metrics
    
    def _compute_curve_metrics(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray
    ) -> Dict[str, float]:
        """Compute ROC and PR curve metrics."""
        metrics = {}
        
        try:
            # ROC AUC
            if len(np.unique(y_true)) > 1:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
                
                # FPR at target TPR
                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                target_tpr_idx = np.argmin(np.abs(tpr - self.config.target_tpr))
                metrics[f'fpr_at_tpr_{self.config.target_tpr}'] = fpr[target_tpr_idx]
                
                # TPR at target FPR
                target_fpr_idx = np.argmin(np.abs(fpr - self.config.target_fpr))
                metrics[f'tpr_at_fpr_{self.config.target_fpr}'] = tpr[target_fpr_idx]
            else:
                metrics['roc_auc'] = 0.0
                metrics[f'fpr_at_tpr_{self.config.target_tpr}'] = 0.0
                metrics[f'tpr_at_fpr_{self.config.target_fpr}'] = 0.0
            
            # PR AUC (Average Precision)
            metrics['pr_auc'] = average_precision_score(y_true, y_scores)
            
            # Precision-Recall curve metrics
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
            
            # Find threshold for target precision
            target_precision = 0.9
            precision_idx = np.where(precision >= target_precision)[0]
            if len(precision_idx) > 0:
                metrics[f'recall_at_precision_{target_precision}'] = recall[precision_idx[0]]
            else:
                metrics[f'recall_at_precision_{target_precision}'] = 0.0
            
        except Exception as e:
            logger.warning(f"Error computing curve metrics: {e}")
            metrics.update({
                'roc_auc': 0.0,
                'pr_auc': 0.0,
                f'fpr_at_tpr_{self.config.target_tpr}': 0.0,
                f'tpr_at_fpr_{self.config.target_fpr}': 0.0,
                f'recall_at_precision_{target_precision}': 0.0
            })
        
        return metrics
    
    def _compute_operational_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute operational security metrics."""
        metrics = {}
        
        # Detection rate (recall)
        total_anomalies = np.sum(y_true)
        detected_anomalies = np.sum((y_true == 1) & (y_pred == 1))
        metrics['detection_rate'] = detected_anomalies / total_anomalies if total_anomalies > 0 else 0.0
        
        # False alarm rate
        total_normal = np.sum(y_true == 0)
        false_alarms = np.sum((y_true == 0) & (y_pred == 1))
        metrics['false_alarm_rate'] = false_alarms / total_normal if total_normal > 0 else 0.0
        
        # Alert efficiency (precision)
        total_alerts = np.sum(y_pred)
        true_alerts = np.sum((y_true == 1) & (y_pred == 1))
        metrics['alert_efficiency'] = true_alerts / total_alerts if total_alerts > 0 else 0.0
        
        # Miss rate (1 - recall)
        metrics['miss_rate'] = 1.0 - metrics['detection_rate']
        
        # Positive predictive value (precision)
        metrics['positive_predictive_value'] = metrics['alert_efficiency']
        
        # Negative predictive value
        true_negatives = np.sum((y_true == 0) & (y_pred == 0))
        total_negatives = np.sum(y_pred == 0)
        metrics['negative_predictive_value'] = true_negatives / total_negatives if total_negatives > 0 else 0.0
        
        return metrics
    
    def _compute_workload_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute alert workload and operational metrics."""
        metrics = {}
        
        # Total alerts generated
        total_alerts = np.sum(y_pred)
        metrics['total_alerts'] = int(total_alerts)
        
        # Alert volume per day (assuming hourly data)
        alerts_per_day = total_alerts * 24 / len(y_pred) if len(y_pred) > 0 else 0
        metrics['alerts_per_day'] = alerts_per_day
        
        # Investigation workload
        investigation_hours = total_alerts * self.config.investigation_time_minutes / 60
        metrics['investigation_hours'] = investigation_hours
        
        # Workload efficiency
        true_alerts = np.sum((y_true == 1) & (y_pred == 1))
        workload_efficiency = true_alerts / investigation_hours if investigation_hours > 0 else 0.0
        metrics['workload_efficiency'] = workload_efficiency
        
        # Alert burden (alerts per analyst per day)
        analysts_per_day = 8  # Assuming 8-hour work day
        alert_burden = alerts_per_day / analysts_per_day
        metrics['alert_burden_per_analyst'] = alert_burden
        
        # Overload indicator
        metrics['overload_indicator'] = 1.0 if alert_burden > 10 else 0.0
        
        return metrics
    
    def compute_threshold_analysis(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Analyze performance across different thresholds.
        
        Args:
            y_true: True binary labels
            y_scores: Anomaly scores
            thresholds: Threshold values to analyze (optional)
            
        Returns:
            DataFrame with metrics for each threshold
        """
        if thresholds is None:
            thresholds = np.linspace(np.min(y_scores), np.max(y_scores), 100)
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            metrics = self._compute_basic_metrics(y_true, y_pred)
            metrics['threshold'] = threshold
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def plot_metrics_curves(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot ROC and PR curves.
        
        Args:
            y_true: True binary labels
            y_scores: Anomaly scores
            save_path: Path to save plots (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        
        axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend(loc="lower right")
        axes[0].grid(True)
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        
        axes[1].plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].legend(loc="lower left")
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics curves saved to {save_path}")
        
        plt.show()
    
    def generate_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> str:
        """
        Generate a comprehensive security evaluation report.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_scores: Anomaly scores (optional)
            feature_names: Names of features (optional)
            
        Returns:
            Formatted report string
        """
        metrics = self.compute_all(y_true, y_pred, y_scores)
        
        report = []
        report.append("=" * 60)
        report.append("SECURITY ANOMALY DETECTION EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Basic Performance
        report.append("BASIC PERFORMANCE METRICS:")
        report.append("-" * 30)
        report.append(f"Accuracy:           {metrics['accuracy']:.4f}")
        report.append(f"Precision:          {metrics['precision']:.4f}")
        report.append(f"Recall:             {metrics['recall']:.4f}")
        report.append(f"F1-Score:           {metrics['f1_score']:.4f}")
        report.append(f"Specificity:        {metrics['specificity']:.4f}")
        report.append("")
        
        # Security-Specific Metrics
        report.append("SECURITY-SPECIFIC METRICS:")
        report.append("-" * 30)
        report.append(f"Detection Rate:     {metrics['detection_rate']:.4f}")
        report.append(f"False Alarm Rate:   {metrics['false_alarm_rate']:.4f}")
        report.append(f"Alert Efficiency:   {metrics['alert_efficiency']:.4f}")
        report.append(f"Miss Rate:          {metrics['miss_rate']:.4f}")
        report.append("")
        
        # Precision@K Metrics
        if any(k.startswith('precision_at_') for k in metrics.keys()):
            report.append("PRECISION@K METRICS:")
            report.append("-" * 30)
            for k, v in metrics.items():
                if k.startswith('precision_at_'):
                    report.append(f"{k.replace('precision_at_', 'Precision@')}: {v:.4f}")
            report.append("")
        
        # Operational Metrics
        report.append("OPERATIONAL METRICS:")
        report.append("-" * 30)
        report.append(f"Total Alerts:       {metrics['total_alerts']}")
        report.append(f"Alerts per Day:     {metrics['alerts_per_day']:.2f}")
        report.append(f"Investigation Hours: {metrics['investigation_hours']:.2f}")
        report.append(f"Workload Efficiency: {metrics['workload_efficiency']:.4f}")
        report.append(f"Alert Burden:       {metrics['alert_burden_per_analyst']:.2f}")
        report.append("")
        
        # Confusion Matrix
        report.append("CONFUSION MATRIX:")
        report.append("-" * 30)
        report.append(f"True Positives:     {metrics['true_positives']}")
        report.append(f"False Positives:    {metrics['false_positives']}")
        report.append(f"True Negatives:     {metrics['true_negatives']}")
        report.append(f"False Negatives:    {metrics['false_negatives']}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 30)
        
        if metrics['false_alarm_rate'] > 0.05:
            report.append("⚠️  High false alarm rate - consider adjusting threshold")
        
        if metrics['detection_rate'] < 0.8:
            report.append("⚠️  Low detection rate - model may need retraining")
        
        if metrics['alert_burden_per_analyst'] > 10:
            report.append("⚠️  High alert burden - consider alert prioritization")
        
        if metrics['workload_efficiency'] < 0.1:
            report.append("⚠️  Low workload efficiency - many false alerts")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def save_metrics(self, filepath: str) -> None:
        """Save metrics history to file."""
        if self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            df.to_csv(filepath, index=False)
            logger.info(f"Metrics saved to {filepath}")
        else:
            logger.warning("No metrics history to save")
    
    def load_metrics(self, filepath: str) -> None:
        """Load metrics history from file."""
        df = pd.read_csv(filepath)
        self.metrics_history = df.to_dict('records')
        logger.info(f"Metrics loaded from {filepath}")
