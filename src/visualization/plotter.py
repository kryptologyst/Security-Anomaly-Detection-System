"""
Anomaly Detection Visualization and Explainability

Provides comprehensive visualization tools for anomaly detection results
and explainability features using SHAP and other interpretability methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class AnomalyPlotter:
    """
    Comprehensive visualization tools for anomaly detection results.
    
    Provides various plots for understanding anomaly patterns, model performance,
    and feature importance in security contexts.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the anomaly plotter.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        plt.style.use(style)
        self.figsize = figsize
        self.colors = {
            'normal': '#2E8B57',      # Sea Green
            'anomaly': '#DC143C',     # Crimson
            'predicted_normal': '#87CEEB',  # Sky Blue
            'predicted_anomaly': '#FF6347',  # Tomato
            'true_positive': '#32CD32',       # Lime Green
            'false_positive': '#FFD700',      # Gold
            'false_negative': '#FF4500',     # Orange Red
            'true_negative': '#4169E1'       # Royal Blue
        }
    
    def plot_anomalies(
        self, 
        data: Dict[str, Any], 
        predictions: np.ndarray,
        scores: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create comprehensive anomaly visualization.
        
        Args:
            data: Data dictionary with features and labels
            predictions: Predicted anomaly labels
            scores: Anomaly scores (optional)
            save_path: Path to save plots (optional)
        """
        df = data['data']
        true_labels = data['labels']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Security Anomaly Detection Results', fontsize=16, fontweight='bold')
        
        # 1. CPU vs Memory scatter plot
        self._plot_cpu_memory_scatter(df, true_labels, predictions, axes[0, 0])
        
        # 2. Feature distribution comparison
        self._plot_feature_distributions(df, true_labels, predictions, axes[0, 1])
        
        # 3. Time series of anomalies
        self._plot_time_series(df, true_labels, predictions, axes[1, 0])
        
        # 4. Confusion matrix
        self._plot_confusion_matrix(true_labels, predictions, axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Anomaly plots saved to {save_path}")
        
        plt.show()
    
    def _plot_cpu_memory_scatter(
        self, 
        df: pd.DataFrame, 
        true_labels: np.ndarray, 
        predictions: np.ndarray, 
        ax: plt.Axes
    ) -> None:
        """Plot CPU vs Memory usage with anomaly highlighting."""
        # Create color mapping
        colors = []
        for i, (true_label, pred_label) in enumerate(zip(true_labels, predictions)):
            if true_label == 1 and pred_label == 1:
                colors.append(self.colors['true_positive'])
            elif true_label == 0 and pred_label == 1:
                colors.append(self.colors['false_positive'])
            elif true_label == 1 and pred_label == 0:
                colors.append(self.colors['false_negative'])
            else:
                colors.append(self.colors['true_negative'])
        
        scatter = ax.scatter(
            df['cpu_usage'], 
            df['memory_usage'], 
            c=colors, 
            alpha=0.7, 
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
        
        ax.set_xlabel('CPU Usage (%)')
        ax.set_ylabel('Memory Usage (%)')
        ax.set_title('CPU vs Memory Usage\n(Anomaly Detection Results)')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['true_positive'], 
                      markersize=8, label='True Positive'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['false_positive'], 
                      markersize=8, label='False Positive'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['false_negative'], 
                      markersize=8, label='False Negative'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['true_negative'], 
                      markersize=8, label='True Negative')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    def _plot_feature_distributions(
        self, 
        df: pd.DataFrame, 
        true_labels: np.ndarray, 
        predictions: np.ndarray, 
        ax: plt.Axes
    ) -> None:
        """Plot feature distributions for normal vs anomalous data."""
        # Select key features for visualization
        key_features = ['cpu_usage', 'memory_usage', 'network_activity', 'resource_pressure']
        
        # Create violin plots
        normal_data = df[true_labels == 0][key_features]
        anomaly_data = df[true_labels == 1][key_features]
        
        # Combine data for plotting
        plot_data = []
        for feature in key_features:
            plot_data.extend([
                {'Feature': feature, 'Value': val, 'Type': 'Normal'} 
                for val in normal_data[feature]
            ])
            plot_data.extend([
                {'Feature': feature, 'Value': val, 'Type': 'Anomaly'} 
                for val in anomaly_data[feature]
            ])
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create violin plot
        sns.violinplot(
            data=plot_df, 
            x='Feature', 
            y='Value', 
            hue='Type',
            ax=ax,
            palette=[self.colors['normal'], self.colors['anomaly']]
        )
        
        ax.set_title('Feature Distributions: Normal vs Anomaly')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_time_series(
        self, 
        df: pd.DataFrame, 
        true_labels: np.ndarray, 
        predictions: np.ndarray, 
        ax: plt.Axes
    ) -> None:
        """Plot time series of anomalies."""
        if 'timestamp' not in df.columns:
            ax.text(0.5, 0.5, 'No timestamp data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Time Series Analysis')
            return
        
        # Convert timestamps to numeric for plotting
        timestamps = pd.to_datetime(df['timestamp'])
        time_numeric = (timestamps - timestamps.min()).dt.total_seconds() / 3600  # hours
        
        # Plot normal points
        normal_mask = true_labels == 0
        ax.scatter(time_numeric[normal_mask], df.loc[normal_mask, 'cpu_usage'], 
                  c=self.colors['normal'], alpha=0.6, s=20, label='Normal')
        
        # Plot anomaly points
        anomaly_mask = true_labels == 1
        ax.scatter(time_numeric[anomaly_mask], df.loc[anomaly_mask, 'cpu_usage'], 
                  c=self.colors['anomaly'], alpha=0.8, s=30, label='Anomaly')
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('CPU Usage (%)')
        ax.set_title('Anomaly Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_confusion_matrix(
        self, 
        true_labels: np.ndarray, 
        predictions: np.ndarray, 
        ax: plt.Axes
    ) -> None:
        """Plot confusion matrix."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(true_labels, predictions)
        
        # Create heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            ax=ax,
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly']
        )
        
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    def plot_feature_importance(
        self, 
        feature_names: List[str], 
        importance_scores: np.ndarray,
        method: str = "Feature Importance",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot feature importance scores.
        
        Args:
            feature_names: Names of features
            importance_scores: Importance scores for each feature
            method: Method name for title
            save_path: Path to save plot (optional)
        """
        # Sort features by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_scores = importance_scores[sorted_indices]
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        bars = ax.barh(range(len(sorted_features)), sorted_scores, 
                      color='steelblue', alpha=0.7)
        
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'{method} - Feature Importance')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_anomaly_scores_distribution(
        self, 
        scores: np.ndarray, 
        true_labels: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot distribution of anomaly scores.
        
        Args:
            scores: Anomaly scores
            true_labels: True binary labels
            save_path: Path to save plot (optional)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot histograms for normal and anomaly scores
        normal_scores = scores[true_labels == 0]
        anomaly_scores = scores[true_labels == 1]
        
        ax.hist(normal_scores, bins=50, alpha=0.7, color=self.colors['normal'], 
               label='Normal', density=True)
        ax.hist(anomaly_scores, bins=50, alpha=0.7, color=self.colors['anomaly'], 
               label='Anomaly', density=True)
        
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Anomaly Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Score distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_dimensionality_reduction(
        self, 
        data: Dict[str, Any], 
        predictions: np.ndarray,
        method: str = 'PCA',
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot data in reduced dimensions.
        
        Args:
            data: Data dictionary with features
            predictions: Predicted anomaly labels
            method: Dimensionality reduction method ('PCA' or 't-SNE')
            save_path: Path to save plot (optional)
        """
        df = data['data']
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'anomaly_label']]
        X = df[feature_cols].values
        
        # Apply dimensionality reduction
        if method == 'PCA':
            reducer = PCA(n_components=2, random_state=42)
            X_reduced = reducer.fit_transform(X)
            title_suffix = f' (Explained Variance: {reducer.explained_variance_ratio_.sum():.2%})'
        elif method == 't-SNE':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            X_reduced = reducer.fit_transform(X)
            title_suffix = ''
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot normal points
        normal_mask = predictions == 0
        ax.scatter(X_reduced[normal_mask, 0], X_reduced[normal_mask, 1], 
                  c=self.colors['normal'], alpha=0.6, s=20, label='Normal')
        
        # Plot anomaly points
        anomaly_mask = predictions == 1
        ax.scatter(X_reduced[anomaly_mask, 0], X_reduced[anomaly_mask, 1], 
                  c=self.colors['anomaly'], alpha=0.8, s=30, label='Anomaly')
        
        ax.set_xlabel(f'{method} Component 1')
        ax.set_ylabel(f'{method} Component 2')
        ax.set_title(f'Anomaly Detection Results - {method}{title_suffix}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dimensionality reduction plot saved to {save_path}")
        
        plt.show()


class AnomalyExplainer:
    """
    Explainability tools for anomaly detection models.
    
    Provides SHAP-based explanations and other interpretability methods
    for understanding why specific instances are flagged as anomalies.
    """
    
    def __init__(self, model: Any, feature_names: List[str]):
        """
        Initialize the explainer.
        
        Args:
            model: Trained anomaly detection model
            feature_names: Names of features
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def fit_explainer(self, X: np.ndarray, sample_size: int = 100) -> None:
        """
        Fit SHAP explainer to the model.
        
        Args:
            X: Training data
            sample_size: Size of background sample for SHAP
        """
        try:
            # Sample background data
            background_indices = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
            background = X[background_indices]
            
            # Create SHAP explainer
            self.explainer = shap.Explainer(self.model, background)
            logger.info("SHAP explainer fitted successfully")
            
        except Exception as e:
            logger.warning(f"Could not fit SHAP explainer: {e}")
            self.explainer = None
    
    def explain_anomalies(
        self, 
        X: np.ndarray, 
        anomaly_indices: List[int],
        max_features: int = 10
    ) -> Dict[int, Dict[str, Any]]:
        """
        Explain specific anomaly instances.
        
        Args:
            X: Feature matrix
            anomaly_indices: Indices of anomalies to explain
            max_features: Maximum number of features to show
            
        Returns:
            Dictionary with explanations for each anomaly
        """
        explanations = {}
        
        if self.explainer is None:
            logger.warning("SHAP explainer not fitted, using simple feature analysis")
            return self._simple_explanation(X, anomaly_indices, max_features)
        
        try:
            # Get SHAP values for anomalies
            shap_values = self.explainer(X[anomaly_indices])
            
            for i, idx in enumerate(anomaly_indices):
                # Get top contributing features
                feature_contributions = shap_values.values[i]
                feature_importance = np.abs(feature_contributions)
                
                # Sort by importance
                sorted_indices = np.argsort(feature_importance)[::-1][:max_features]
                
                explanation = {
                    'instance_index': idx,
                    'top_features': [
                        {
                            'feature': self.feature_names[j],
                            'contribution': float(feature_contributions[j]),
                            'importance': float(feature_importance[j])
                        }
                        for j in sorted_indices
                    ],
                    'prediction_score': float(np.sum(feature_contributions)),
                    'explanation_method': 'SHAP'
                }
                
                explanations[idx] = explanation
        
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return self._simple_explanation(X, anomaly_indices, max_features)
        
        return explanations
    
    def _simple_explanation(
        self, 
        X: np.ndarray, 
        anomaly_indices: List[int],
        max_features: int
    ) -> Dict[int, Dict[str, Any]]:
        """Simple explanation based on feature values."""
        explanations = {}
        
        # Calculate feature statistics for normal data
        normal_mask = np.ones(len(X), dtype=bool)
        normal_mask[anomaly_indices] = False
        normal_data = X[normal_mask]
        
        feature_means = np.mean(normal_data, axis=0)
        feature_stds = np.std(normal_data, axis=0)
        
        for idx in anomaly_indices:
            instance = X[idx]
            
            # Calculate z-scores
            z_scores = np.abs((instance - feature_means) / (feature_stds + 1e-8))
            
            # Get top anomalous features
            sorted_indices = np.argsort(z_scores)[::-1][:max_features]
            
            explanation = {
                'instance_index': idx,
                'top_features': [
                    {
                        'feature': self.feature_names[j],
                        'value': float(instance[j]),
                        'z_score': float(z_scores[j]),
                        'normal_mean': float(feature_means[j]),
                        'normal_std': float(feature_stds[j])
                    }
                    for j in sorted_indices
                ],
                'explanation_method': 'Z-Score'
            }
            
            explanations[idx] = explanation
        
        return explanations
    
    def plot_explanations(
        self, 
        explanations: Dict[int, Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot explanations for anomalies.
        
        Args:
            explanations: Explanations dictionary
            save_path: Path to save plots (optional)
        """
        n_anomalies = len(explanations)
        if n_anomalies == 0:
            logger.warning("No explanations to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(
            (n_anomalies + 1) // 2, 2, 
            figsize=(16, 4 * ((n_anomalies + 1) // 2))
        )
        
        if n_anomalies == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (idx, explanation) in enumerate(explanations.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Extract data for plotting
            features = [f['feature'] for f in explanation['top_features']]
            
            if explanation['explanation_method'] == 'SHAP':
                contributions = [f['contribution'] for f in explanation['top_features']]
                colors = ['red' if c < 0 else 'blue' for c in contributions]
                ylabel = 'SHAP Contribution'
            else:
                contributions = [f['z_score'] for f in explanation['top_features']]
                colors = 'steelblue'
                ylabel = 'Z-Score'
            
            # Create bar plot
            bars = ax.barh(features, contributions, color=colors, alpha=0.7)
            
            ax.set_xlabel(ylabel)
            ax.set_title(f'Anomaly {idx} Explanation')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for j, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{width:.2f}', ha='left', va='center')
        
        # Hide unused subplots
        for i in range(n_anomalies, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Explanation plots saved to {save_path}")
        
        plt.show()
    
    def generate_explanation_report(
        self, 
        explanations: Dict[int, Dict[str, Any]]
    ) -> str:
        """
        Generate a text report of explanations.
        
        Args:
            explanations: Explanations dictionary
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("ANOMALY EXPLANATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        for idx, explanation in explanations.items():
            report.append(f"ANOMALY INSTANCE {idx}:")
            report.append("-" * 30)
            report.append(f"Explanation Method: {explanation['explanation_method']}")
            
            if 'prediction_score' in explanation:
                report.append(f"Prediction Score: {explanation['prediction_score']:.4f}")
            
            report.append("")
            report.append("Top Contributing Features:")
            
            for i, feature_info in enumerate(explanation['top_features'], 1):
                report.append(f"  {i}. {feature_info['feature']}")
                
                if explanation['explanation_method'] == 'SHAP':
                    report.append(f"     Contribution: {feature_info['contribution']:.4f}")
                    report.append(f"     Importance: {feature_info['importance']:.4f}")
                else:
                    report.append(f"     Value: {feature_info['value']:.4f}")
                    report.append(f"     Z-Score: {feature_info['z_score']:.4f}")
                    report.append(f"     Normal Mean: {feature_info['normal_mean']:.4f}")
                    report.append(f"     Normal Std: {feature_info['normal_std']:.4f}")
                
                report.append("")
            
            report.append("-" * 60)
            report.append("")
        
        return "\n".join(report)
