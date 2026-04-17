"""
Security Anomaly Detection System

A comprehensive anomaly detection framework for identifying security threats
in system logs and network traffic. This implementation focuses on defensive
security research and education purposes only.

DISCLAIMER: This is a research/educational demonstration tool. It is not
intended for production security operations and may not accurately detect
real-world threats. Use only for learning and defensive security research.
"""

__version__ = "1.0.0"
__author__ = "Security Research Team"
__email__ = "research@example.com"
__description__ = "A comprehensive anomaly detection framework for security research and education"

# Import main components
from .data.generator import SecurityLogGenerator, SecurityLogConfig
from .models.anomaly_detector import AnomalyDetector
from .evaluation.metrics import SecurityMetrics, SecurityMetricsConfig
from .visualization.plotter import AnomalyPlotter, AnomalyExplainer
from .utils.config import ConfigManager, get_config_manager
from .utils.privacy import PrivacyManager, get_privacy_manager

__all__ = [
    # Data generation
    'SecurityLogGenerator',
    'SecurityLogConfig',
    
    # Anomaly detection
    'AnomalyDetector',
    
    # Evaluation
    'SecurityMetrics',
    'SecurityMetricsConfig',
    
    # Visualization
    'AnomalyPlotter',
    'AnomalyExplainer',
    
    # Configuration
    'ConfigManager',
    'get_config_manager',
    
    # Privacy and safety
    'PrivacyManager',
    'get_privacy_manager',
    
    # Version info
    '__version__',
    '__author__',
    '__email__',
    '__description__'
]
