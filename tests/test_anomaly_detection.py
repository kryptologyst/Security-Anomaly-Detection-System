"""
Unit tests for the Security Anomaly Detection System.

Tests cover data generation, anomaly detection models, evaluation metrics,
and visualization components.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.generator import SecurityLogGenerator, SecurityLogConfig
from src.models.anomaly_detector import (
    AnomalyDetector, IsolationForestDetector, AutoencoderDetector,
    LSTMAutoencoderDetector, EnsembleAnomalyDetector
)
from src.evaluation.metrics import SecurityMetrics, SecurityMetricsConfig
from src.visualization.plotter import AnomalyPlotter, AnomalyExplainer
from src.utils.config import ConfigManager


class TestSecurityLogGenerator:
    """Test cases for SecurityLogGenerator."""
    
    def test_init(self):
        """Test generator initialization."""
        generator = SecurityLogGenerator(random_state=42)
        assert generator.random_state == 42
        assert generator.config is not None
        assert len(generator.anomaly_patterns) > 0
    
    def test_generate_logs(self):
        """Test log generation."""
        generator = SecurityLogGenerator(random_state=42)
        data = generator.generate_logs(n_samples=100, anomaly_ratio=0.1)
        
        assert 'data' in data
        assert 'features' in data
        assert 'labels' in data
        assert 'timestamps' in data
        assert 'metadata' in data
        
        assert len(data['data']) == 100
        assert len(data['labels']) == 100
        assert data['labels'].sum() == 10  # 10% of 100
        
        # Check data types
        assert isinstance(data['data'], pd.DataFrame)
        assert isinstance(data['labels'], np.ndarray)
        assert isinstance(data['timestamps'], np.ndarray)
    
    def test_generate_logs_anomaly_ratio(self):
        """Test different anomaly ratios."""
        generator = SecurityLogGenerator(random_state=42)
        
        # Test 5% anomalies
        data_5 = generator.generate_logs(n_samples=100, anomaly_ratio=0.05)
        assert data_5['labels'].sum() == 5
        
        # Test 20% anomalies
        data_20 = generator.generate_logs(n_samples=100, anomaly_ratio=0.2)
        assert data_20['labels'].sum() == 20
    
    def test_data_features(self):
        """Test that generated data has expected features."""
        generator = SecurityLogGenerator(random_state=42)
        data = generator.generate_logs(n_samples=50, anomaly_ratio=0.1)
        
        df = data['data']
        expected_features = [
            'cpu_usage', 'memory_usage', 'disk_usage',
            'bytes_sent', 'bytes_received', 'active_connections',
            'process_count', 'packet_size_variance', 'connection_duration',
            'failed_login_attempts', 'privilege_escalation_attempts',
            'suspicious_file_access', 'user_id', 'source_ip', 'user_agent',
            'cpu_memory_ratio', 'network_activity', 'resource_pressure'
        ]
        
        for feature in expected_features:
            assert feature in df.columns
    
    def test_privacy_protection(self):
        """Test that PII is properly protected."""
        generator = SecurityLogGenerator(random_state=42)
        data = generator.generate_logs(n_samples=50, anomaly_ratio=0.1)
        
        df = data['data']
        
        # Check that user IDs are hashed
        user_ids = df['user_id'].unique()
        for user_id in user_ids:
            assert user_id.startswith('usr_')
            assert len(user_id) == 12  # usr_ + 8 chars
        
        # Check that IP addresses are hashed
        ip_addresses = df['source_ip'].unique()
        for ip in ip_addresses:
            assert ip.startswith('192.168.')
            parts = ip.split('.')
            assert len(parts) == 4
    
    def test_save_load_data(self):
        """Test saving and loading data."""
        generator = SecurityLogGenerator(random_state=42)
        data = generator.generate_logs(n_samples=50, anomaly_ratio=0.1)
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save data
            generator.save_data(data, tmp_path)
            assert os.path.exists(tmp_path)
            
            # Load data
            loaded_data = generator.load_data(tmp_path)
            assert len(loaded_data['data']) == 50
            assert loaded_data['labels'].sum() == 5
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestAnomalyDetector:
    """Test cases for AnomalyDetector."""
    
    def test_isolation_forest_detector(self):
        """Test Isolation Forest detector."""
        detector = IsolationForestDetector(random_state=42)
        
        # Generate test data
        generator = SecurityLogGenerator(random_state=42)
        data = generator.generate_logs(n_samples=100, anomaly_ratio=0.1)
        
        # Fit and predict
        predictions = detector.fit_predict(data)
        scores = detector.decision_function(data)
        
        assert len(predictions) == 100
        assert len(scores) == 100
        assert detector.is_fitted
        
        # Check prediction values
        assert set(predictions).issubset({0, 1})
        assert isinstance(scores, np.ndarray)
    
    def test_autoencoder_detector(self):
        """Test Autoencoder detector."""
        detector = AutoencoderDetector(
            hidden_dims=[32, 16],
            epochs=5,  # Reduced for testing
            random_state=42
        )
        
        # Generate test data
        generator = SecurityLogGenerator(random_state=42)
        data = generator.generate_logs(n_samples=100, anomaly_ratio=0.1)
        
        # Fit and predict
        predictions = detector.fit_predict(data)
        scores = detector.decision_function(data)
        
        assert len(predictions) == 100
        assert len(scores) == 100
        assert detector.is_fitted
        
        # Check prediction values
        assert set(predictions).issubset({0, 1})
        assert isinstance(scores, np.ndarray)
    
    def test_ensemble_detector(self):
        """Test Ensemble detector."""
        detector = EnsembleAnomalyDetector(random_state=42)
        
        # Generate test data
        generator = SecurityLogGenerator(random_state=42)
        data = generator.generate_logs(n_samples=100, anomaly_ratio=0.1)
        
        # Fit and predict
        predictions = detector.fit_predict(data)
        scores = detector.decision_function(data)
        
        assert len(predictions) == 100
        assert len(scores) == 100
        assert detector.is_fitted
        
        # Check prediction values
        assert set(predictions).issubset({0, 1})
        assert isinstance(scores, np.ndarray)
    
    def test_anomaly_detector_wrapper(self):
        """Test AnomalyDetector wrapper class."""
        # Test isolation forest
        detector = AnomalyDetector(method='isolation_forest', random_state=42)
        
        generator = SecurityLogGenerator(random_state=42)
        data = generator.generate_logs(n_samples=100, anomaly_ratio=0.1)
        
        predictions = detector.fit_predict(data)
        scores = detector.decision_function(data)
        
        assert len(predictions) == 100
        assert len(scores) == 100
    
    def test_invalid_method(self):
        """Test invalid method raises error."""
        with pytest.raises(ValueError):
            AnomalyDetector(method='invalid_method', random_state=42)


class TestSecurityMetrics:
    """Test cases for SecurityMetrics."""
    
    def test_init(self):
        """Test metrics initialization."""
        metrics = SecurityMetrics()
        assert metrics.config is not None
        assert len(metrics.metrics_history) == 0
    
    def test_compute_all(self):
        """Test computing all metrics."""
        metrics = SecurityMetrics()
        
        # Generate test data
        generator = SecurityLogGenerator(random_state=42)
        data = generator.generate_logs(n_samples=100, anomaly_ratio=0.1)
        
        # Create mock predictions
        y_true = data['labels']
        y_pred = np.random.choice([0, 1], size=100, p=[0.9, 0.1])
        y_scores = np.random.random(100)
        
        results = metrics.compute_all(y_true, y_pred, y_scores)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'specificity',
            'detection_rate', 'false_alarm_rate', 'alert_efficiency',
            'total_alerts', 'alerts_per_day', 'investigation_hours'
        ]
        
        for metric in expected_metrics:
            assert metric in results
        
        # Check metrics history
        assert len(metrics.metrics_history) == 1
    
    def test_compute_basic_metrics(self):
        """Test basic metrics computation."""
        metrics = SecurityMetrics()
        
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        
        results = metrics._compute_basic_metrics(y_true, y_pred)
        
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results
        assert 'specificity' in results
        
        # Check that all values are between 0 and 1
        for key, value in results.items():
            if key not in ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']:
                assert 0 <= value <= 1
    
    def test_compute_precision_at_k(self):
        """Test precision@K computation."""
        metrics = SecurityMetrics()
        
        y_true = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        
        results = metrics._compute_precision_at_k(y_true, y_scores)
        
        assert 'precision_at_10' in results
        assert 'precision_at_50' in results
        assert 'precision_at_100' in results
        assert 'precision_at_500' in results
        
        # Check that precision@K values are between 0 and 1
        for key, value in results.items():
            assert 0 <= value <= 1
    
    def test_compute_operational_metrics(self):
        """Test operational metrics computation."""
        metrics = SecurityMetrics()
        
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        
        results = metrics._compute_operational_metrics(y_true, y_pred)
        
        assert 'detection_rate' in results
        assert 'false_alarm_rate' in results
        assert 'alert_efficiency' in results
        assert 'miss_rate' in results
        
        # Check that rates are between 0 and 1
        for key, value in results.items():
            assert 0 <= value <= 1
    
    def test_generate_report(self):
        """Test report generation."""
        metrics = SecurityMetrics()
        
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
        y_scores = np.random.random(10)
        
        report = metrics.generate_report(y_true, y_pred, y_scores)
        
        assert isinstance(report, str)
        assert "SECURITY ANOMALY DETECTION EVALUATION REPORT" in report
        assert "BASIC PERFORMANCE METRICS" in report
        assert "SECURITY-SPECIFIC METRICS" in report
        assert "OPERATIONAL METRICS" in report


class TestAnomalyPlotter:
    """Test cases for AnomalyPlotter."""
    
    def test_init(self):
        """Test plotter initialization."""
        plotter = AnomalyPlotter()
        assert plotter.figsize == (12, 8)
        assert 'normal' in plotter.colors
        assert 'anomaly' in plotter.colors
    
    def test_plot_anomalies(self):
        """Test anomaly plotting."""
        plotter = AnomalyPlotter()
        
        # Generate test data
        generator = SecurityLogGenerator(random_state=42)
        data = generator.generate_logs(n_samples=50, anomaly_ratio=0.1)
        
        predictions = np.random.choice([0, 1], size=50, p=[0.9, 0.1])
        scores = np.random.random(50)
        
        # Test that plotting doesn't raise errors
        try:
            plotter.plot_anomalies(data, predictions, scores)
        except Exception as e:
            pytest.fail(f"plot_anomalies raised an exception: {e}")
    
    def test_plot_feature_importance(self):
        """Test feature importance plotting."""
        plotter = AnomalyPlotter()
        
        feature_names = ['cpu_usage', 'memory_usage', 'network_activity']
        importance_scores = np.array([0.5, 0.3, 0.2])
        
        # Test that plotting doesn't raise errors
        try:
            plotter.plot_feature_importance(feature_names, importance_scores)
        except Exception as e:
            pytest.fail(f"plot_feature_importance raised an exception: {e}")
    
    def test_plot_anomaly_scores_distribution(self):
        """Test anomaly scores distribution plotting."""
        plotter = AnomalyPlotter()
        
        scores = np.random.random(100)
        true_labels = np.random.choice([0, 1], size=100, p=[0.9, 0.1])
        
        # Test that plotting doesn't raise errors
        try:
            plotter.plot_anomaly_scores_distribution(scores, true_labels)
        except Exception as e:
            pytest.fail(f"plot_anomaly_scores_distribution raised an exception: {e}")


class TestAnomalyExplainer:
    """Test cases for AnomalyExplainer."""
    
    def test_init(self):
        """Test explainer initialization."""
        mock_model = Mock()
        feature_names = ['cpu_usage', 'memory_usage', 'network_activity']
        
        explainer = AnomalyExplainer(mock_model, feature_names)
        assert explainer.model == mock_model
        assert explainer.feature_names == feature_names
        assert explainer.explainer is None
    
    def test_simple_explanation(self):
        """Test simple explanation generation."""
        mock_model = Mock()
        feature_names = ['cpu_usage', 'memory_usage', 'network_activity']
        
        explainer = AnomalyExplainer(mock_model, feature_names)
        
        X = np.random.random((100, 3))
        anomaly_indices = [10, 25, 50]
        
        explanations = explainer._simple_explanation(X, anomaly_indices, max_features=3)
        
        assert len(explanations) == 3
        assert 10 in explanations
        assert 25 in explanations
        assert 50 in explanations
        
        # Check explanation structure
        for idx, explanation in explanations.items():
            assert 'instance_index' in explanation
            assert 'top_features' in explanation
            assert 'explanation_method' in explanation
            assert explanation['explanation_method'] == 'Z-Score'
            assert len(explanation['top_features']) <= 3
    
    def test_explain_anomalies(self):
        """Test anomaly explanation."""
        mock_model = Mock()
        feature_names = ['cpu_usage', 'memory_usage', 'network_activity']
        
        explainer = AnomalyExplainer(mock_model, feature_names)
        
        X = np.random.random((100, 3))
        anomaly_indices = [10, 25]
        
        explanations = explainer.explain_anomalies(X, anomaly_indices)
        
        assert len(explanations) == 2
        assert 10 in explanations
        assert 25 in explanations
    
    def test_generate_explanation_report(self):
        """Test explanation report generation."""
        mock_model = Mock()
        feature_names = ['cpu_usage', 'memory_usage']
        
        explainer = AnomalyExplainer(mock_model, feature_names)
        
        # Create mock explanations
        explanations = {
            10: {
                'instance_index': 10,
                'top_features': [
                    {'feature': 'cpu_usage', 'value': 0.8, 'z_score': 2.5},
                    {'feature': 'memory_usage', 'value': 0.6, 'z_score': 1.8}
                ],
                'explanation_method': 'Z-Score'
            }
        }
        
        report = explainer.generate_explanation_report(explanations)
        
        assert isinstance(report, str)
        assert "ANOMALY EXPLANATION REPORT" in report
        assert "ANOMALY INSTANCE 10" in report
        assert "cpu_usage" in report
        assert "memory_usage" in report


class TestConfigManager:
    """Test cases for ConfigManager."""
    
    def test_init(self):
        """Test config manager initialization."""
        config_manager = ConfigManager()
        assert config_manager.config is not None
        assert isinstance(config_manager.config, dict)
    
    def test_get_set(self):
        """Test get and set methods."""
        config_manager = ConfigManager()
        
        # Test setting and getting values
        config_manager.set('test.value', 42)
        assert config_manager.get('test.value') == 42
        
        # Test default value
        assert config_manager.get('nonexistent.key', 'default') == 'default'
    
    def test_get_data_config(self):
        """Test getting data configuration."""
        config_manager = ConfigManager()
        data_config = config_manager.get_data_config()
        
        assert isinstance(data_config, dict)
        assert 'n_samples' in data_config
        assert 'anomaly_ratio' in data_config
    
    def test_get_model_config(self):
        """Test getting model configuration."""
        config_manager = ConfigManager()
        model_config = config_manager.get_model_config()
        
        assert isinstance(model_config, dict)
        assert 'method' in model_config
    
    def test_validate_config(self):
        """Test configuration validation."""
        config_manager = ConfigManager()
        
        # Should pass validation with default config
        assert config_manager.validate_config()
    
    def test_create_directories(self):
        """Test directory creation."""
        config_manager = ConfigManager()
        
        # Test that directory creation doesn't raise errors
        try:
            config_manager.create_directories()
        except Exception as e:
            pytest.fail(f"create_directories raised an exception: {e}")
    
    def test_get_device_config(self):
        """Test device configuration."""
        config_manager = ConfigManager()
        device = config_manager.get_device_config()
        
        assert device in ['cpu', 'cuda', 'mps']
    
    def test_get_random_state(self):
        """Test random state retrieval."""
        config_manager = ConfigManager()
        random_state = config_manager.get_random_state()
        
        assert isinstance(random_state, int)
        assert random_state >= 0


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Generate data
        generator = SecurityLogGenerator(random_state=42)
        data = generator.generate_logs(n_samples=100, anomaly_ratio=0.1)
        
        # Run detection
        detector = AnomalyDetector(method='isolation_forest', random_state=42)
        predictions = detector.fit_predict(data)
        scores = detector.decision_function(data)
        
        # Evaluate performance
        metrics = SecurityMetrics()
        results = metrics.compute_all(data['labels'], predictions, scores)
        
        # Generate report
        report = metrics.generate_report(data['labels'], predictions, scores)
        
        # Check that everything worked
        assert len(predictions) == 100
        assert len(scores) == 100
        assert 'accuracy' in results
        assert isinstance(report, str)
    
    def test_multiple_models_comparison(self):
        """Test comparing multiple models."""
        generator = SecurityLogGenerator(random_state=42)
        data = generator.generate_logs(n_samples=100, anomaly_ratio=0.1)
        
        methods = ['isolation_forest', 'autoencoder']
        results = {}
        
        for method in methods:
            detector = AnomalyDetector(method=method, random_state=42)
            predictions = detector.fit_predict(data)
            
            metrics = SecurityMetrics()
            model_results = metrics.compute_all(data['labels'], predictions)
            results[method] = model_results['accuracy']
        
        # Check that both models produced results
        assert len(results) == 2
        assert 'isolation_forest' in results
        assert 'autoencoder' in results
        
        # Check that accuracies are reasonable
        for method, accuracy in results.items():
            assert 0 <= accuracy <= 1


if __name__ == "__main__":
    pytest.main([__file__])
