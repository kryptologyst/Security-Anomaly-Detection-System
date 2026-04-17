"""
Security Anomaly Detection System

A comprehensive anomaly detection framework for identifying security threats
in system logs and network traffic. This implementation focuses on defensive
security research and education purposes only.

DISCLAIMER: This is a research/educational demonstration tool. It is not
intended for production security operations and may not accurately detect
real-world threats. Use only for learning and defensive security research.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

# Set up deterministic behavior
import random
import numpy as np
import torch

# Set seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main() -> None:
    """Main entry point for the anomaly detection demo."""
    logger.info("Starting Security Anomaly Detection System")
    
    # Import here to ensure proper initialization
    from src.data.generator import SecurityLogGenerator
    from src.models.anomaly_detector import AnomalyDetector
    from src.evaluation.metrics import SecurityMetrics
    from src.visualization.plotter import AnomalyPlotter
    
    # Generate synthetic security log data
    generator = SecurityLogGenerator(random_state=RANDOM_SEED)
    data = generator.generate_logs(n_samples=1000, anomaly_ratio=0.1)
    
    # Initialize anomaly detector
    detector = AnomalyDetector(random_state=RANDOM_SEED)
    
    # Train and predict
    predictions = detector.fit_predict(data)
    
    # Evaluate performance
    metrics = SecurityMetrics()
    results = metrics.compute_all(data['labels'], predictions)
    
    # Display results
    logger.info("Anomaly Detection Results:")
    for metric, value in results.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Create visualizations
    plotter = AnomalyPlotter()
    plotter.plot_anomalies(data, predictions)
    
    logger.info("Anomaly detection demo completed successfully")

if __name__ == "__main__":
    main()

