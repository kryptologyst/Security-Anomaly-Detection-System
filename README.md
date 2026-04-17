# Security Anomaly Detection System

A comprehensive anomaly detection framework for identifying security threats in system logs and network traffic. This implementation focuses on defensive security research and education purposes only.

## ⚠️ Security Research Disclaimer

**This is a research and educational demonstration tool only.**

- **NOT intended for production security operations**
- **May not accurately detect real-world threats**
- **Use only for learning and defensive security research**
- **All data is synthetic and contains no real PII**

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/km15588/Security-Anomaly-Detection.git
   cd Security-Anomaly-Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the demo:**
   ```bash
   streamlit run demo/streamlit_demo.py
   ```

4. **Access the web interface:**
   Open your browser to `http://localhost:8501`

## Features

### Core Capabilities

- **Multiple Detection Methods**: Isolation Forest, Autoencoders, LSTM, and Ensemble methods
- **Comprehensive Evaluation**: Security-focused metrics including AUCPR, precision@K, FPR at target TPR
- **Interactive Visualization**: Real-time anomaly detection with Streamlit interface
- **Explainability**: SHAP-based explanations for anomaly detection decisions
- **Synthetic Data Generation**: Privacy-safe synthetic security log data

### Security-Focused Metrics

- **Detection Rate**: Percentage of true anomalies detected
- **False Alarm Rate**: Rate of false positive alerts
- **Alert Efficiency**: Precision of generated alerts
- **Precision@K**: Performance on top-K most suspicious events
- **Operational Metrics**: Alert workload, investigation time, analyst burden

### Advanced Features

- **Real-time Simulation**: Live anomaly detection simulation
- **Threshold Analysis**: Performance analysis across different thresholds
- **Feature Importance**: Understanding which features contribute to anomalies
- **Dimensionality Reduction**: PCA and t-SNE visualization of anomaly patterns
- **Configuration Management**: YAML-based configuration with OmegaConf

## Project Structure

```
security-anomaly-detection/
├── src/                          # Source code
│   ├── data/                     # Data generation and processing
│   │   └── generator.py         # Synthetic security log generator
│   ├── models/                   # Anomaly detection models
│   │   └── anomaly_detector.py  # Detection algorithms
│   ├── evaluation/               # Evaluation metrics
│   │   └── metrics.py           # Security-focused metrics
│   ├── visualization/            # Visualization tools
│   │   └── plotter.py           # Plotting and explainability
│   └── utils/                    # Utility functions
│       └── config.py            # Configuration management
├── configs/                      # Configuration files
│   └── default.yaml            # Default configuration
├── demo/                        # Interactive demos
│   └── streamlit_demo.py       # Streamlit web interface
├── tests/                       # Unit tests
├── assets/                      # Generated plots and results
├── data/                        # Data storage
├── models/                      # Saved models
├── logs/                        # Log files
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## Configuration

The system uses YAML configuration files for easy customization. Key configuration sections:

### Data Generation
```yaml
data:
  n_samples: 1000
  anomaly_ratio: 0.1
  system_metrics:
    cpu_mean: 25.0
    cpu_std: 15.0
    memory_mean: 45.0
    memory_std: 20.0
```

### Model Configuration
```yaml
model:
  method: "isolation_forest"  # Options: isolation_forest, autoencoder, lstm, ensemble
  isolation_forest:
    contamination: 0.1
    n_estimators: 100
```

### Evaluation Settings
```yaml
evaluation:
  security_metrics:
    target_tpr: 0.95
    target_fpr: 0.01
    precision_k_values: [10, 50, 100, 500]
```

## Usage Examples

### Basic Usage

```python
from src.data.generator import SecurityLogGenerator
from src.models.anomaly_detector import AnomalyDetector
from src.evaluation.metrics import SecurityMetrics

# Generate synthetic data
generator = SecurityLogGenerator(random_state=42)
data = generator.generate_logs(n_samples=1000, anomaly_ratio=0.1)

# Run anomaly detection
detector = AnomalyDetector(method='isolation_forest', random_state=42)
predictions = detector.fit_predict(data)

# Evaluate performance
metrics = SecurityMetrics()
results = metrics.compute_all(data['labels'], predictions)
print(f"Detection Rate: {results['detection_rate']:.3f}")
print(f"False Alarm Rate: {results['false_alarm_rate']:.3f}")
```

### Advanced Usage with Explainability

```python
from src.visualization.plotter import AnomalyExplainer

# Initialize explainer
explainer = AnomalyExplainer(detector.detector, data['features'])
explainer.fit_explainer(X, sample_size=100)

# Explain specific anomalies
anomaly_indices = [10, 25, 50]  # Example anomaly indices
explanations = explainer.explain_anomalies(X, anomaly_indices)

# Generate explanation report
report = explainer.generate_explanation_report(explanations)
print(report)
```

### Configuration Management

```python
from src.utils.config import get_config_manager

# Load configuration
config = get_config_manager()

# Access configuration values
n_samples = config.get_sample_size()
model_method = config.get_model_method()
contamination = config.get_contamination()

# Update configuration
config.set('data.n_samples', 2000)
config.set('model.method', 'autoencoder')
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_anomaly_detector.py
```

## Demo Screenshots

### Main Dashboard
The main dashboard provides an overview of system status, key metrics, and real-time anomaly detection results.

### Data Explorer
Interactive data exploration with filtering, visualization, and detailed analysis of security log features.

### Model Performance
Comprehensive evaluation metrics including ROC curves, confusion matrices, and operational security metrics.

### Explainability
SHAP-based explanations showing which features contribute to anomaly detection decisions.

## Privacy and Security

### Data Privacy
- **Synthetic Data Only**: All data is generated synthetically
- **PII Protection**: User IDs and IP addresses are hashed
- **No Real Logs**: No real security logs or sensitive data
- **Anonymization**: Timestamps and identifiers are anonymized

### Security Measures
- **Input Validation**: All inputs are validated and sanitized
- **Output Sanitization**: Sensitive data is redacted from outputs
- **Audit Trail**: All operations are logged for security auditing
- **Access Control**: Demo interface includes security warnings

## Performance

### Optimization Features
- **Device Fallback**: Automatic CUDA → MPS → CPU device selection
- **Memory Management**: Configurable memory limits and batch processing
- **Parallel Processing**: Multi-threaded data processing and evaluation
- **Caching**: Intelligent caching of intermediate results

### Scalability
- **Batch Processing**: Efficient batch processing for large datasets
- **Streaming Support**: Real-time anomaly detection capabilities
- **Memory Efficient**: Optimized memory usage for large-scale data
- **Distributed Ready**: Architecture supports distributed processing

## Contributing

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Install development dependencies**: `pip install -r requirements.txt`
4. **Run tests**: `pytest`
5. **Format code**: `black src/` and `ruff check src/`
6. **Commit changes**: `git commit -m "Add feature"`
7. **Push to branch**: `git push origin feature-name`
8. **Create Pull Request**

### Code Quality

- **Type Hints**: All functions include type annotations
- **Documentation**: Comprehensive docstrings for all modules
- **Testing**: Unit tests for all major components
- **Formatting**: Black code formatting and Ruff linting
- **Security**: Security-focused code review process

## Documentation

### API Reference
- [Data Generator API](docs/api/data_generator.md)
- [Anomaly Detector API](docs/api/anomaly_detector.md)
- [Metrics API](docs/api/metrics.md)
- [Visualization API](docs/api/visualization.md)

### Tutorials
- [Getting Started](docs/tutorials/getting_started.md)
- [Custom Models](docs/tutorials/custom_models.md)
- [Evaluation Metrics](docs/tutorials/evaluation.md)
- [Explainability](docs/tutorials/explainability.md)

### Examples
- [Basic Anomaly Detection](examples/basic_detection.py)
- [Advanced Evaluation](examples/advanced_evaluation.py)
- [Custom Configuration](examples/custom_config.py)
- [Batch Processing](examples/batch_processing.py)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **CUDA Issues**: Check CUDA installation and PyTorch compatibility
3. **Memory Errors**: Reduce batch size or sample size
4. **Configuration Errors**: Validate YAML configuration syntax

### Getting Help

- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the comprehensive documentation
- **Examples**: Review the example scripts

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Scikit-learn**: Machine learning algorithms and utilities
- **PyTorch**: Deep learning framework
- **Streamlit**: Interactive web application framework
- **SHAP**: Model explainability library
- **Plotly**: Interactive visualization library

## Contact

For questions, suggestions, or collaboration opportunities:

- **Email**: research@example.com
- **GitHub**: [Repository Issues](https://github.com/km15588/Security-Anomaly-Detection/issues)
- **Documentation**: [Project Wiki](https://github.com/km15588/Security-Anomaly-Detection/wiki)

---

**Remember**: This tool is for educational and research purposes only. Always follow responsible disclosure practices and applicable laws when working with security-related tools.
# Security-Anomaly-Detection-System
