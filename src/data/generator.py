"""
Security Log Data Generator

Generates synthetic security log data for anomaly detection research.
Includes system metrics, network traffic, and user behavior patterns.
All data is synthetic and contains no real PII or sensitive information.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class SecurityLogConfig:
    """Configuration for security log generation."""
    
    # System metrics
    cpu_mean: float = 25.0
    cpu_std: float = 15.0
    memory_mean: float = 45.0
    memory_std: float = 20.0
    disk_mean: float = 30.0
    disk_std: float = 10.0
    
    # Network metrics
    bytes_sent_mean: float = 1000.0
    bytes_sent_std: float = 500.0
    bytes_received_mean: float = 2000.0
    bytes_received_std: float = 1000.0
    connections_mean: float = 5.0
    connections_std: float = 3.0
    
    # Process metrics
    processes_mean: float = 60.0
    processes_std: float = 15.0
    
    # Time patterns
    time_window_hours: int = 24
    anomaly_ratio: float = 0.1


class SecurityLogGenerator:
    """
    Generates synthetic security log data for anomaly detection.
    
    Creates realistic system metrics with configurable anomaly patterns
    for research and educational purposes.
    """
    
    def __init__(self, config: Optional[SecurityLogConfig] = None, random_state: int = 42):
        """
        Initialize the security log generator.
        
        Args:
            config: Configuration for data generation
            random_state: Random seed for reproducibility
        """
        self.config = config or SecurityLogConfig()
        self.rng = np.random.RandomState(random_state)
        self.random_state = random_state
        
        # Anomaly patterns
        self.anomaly_patterns = {
            'cpu_spike': {'cpu_multiplier': 3.0, 'memory_multiplier': 1.5},
            'memory_leak': {'memory_multiplier': 2.5, 'cpu_multiplier': 1.2},
            'network_flood': {'bytes_multiplier': 5.0, 'connections_multiplier': 4.0},
            'process_explosion': {'processes_multiplier': 3.0, 'cpu_multiplier': 2.0},
            'disk_full': {'disk_multiplier': 2.0, 'cpu_multiplier': 1.3},
        }
    
    def _generate_normal_metrics(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Generate normal system metrics."""
        metrics = {
            'cpu_usage': self.rng.normal(
                self.config.cpu_mean, 
                self.config.cpu_std, 
                n_samples
            ),
            'memory_usage': self.rng.normal(
                self.config.memory_mean, 
                self.config.memory_std, 
                n_samples
            ),
            'disk_usage': self.rng.normal(
                self.config.disk_mean, 
                self.config.disk_std, 
                n_samples
            ),
            'bytes_sent': self.rng.lognormal(
                np.log(self.config.bytes_sent_mean), 
                0.5, 
                n_samples
            ),
            'bytes_received': self.rng.lognormal(
                np.log(self.config.bytes_received_mean), 
                0.5, 
                n_samples
            ),
            'active_connections': self.rng.poisson(
                self.config.connections_mean, 
                n_samples
            ),
            'process_count': self.rng.normal(
                self.config.processes_mean, 
                self.config.processes_std, 
                n_samples
            ),
        }
        
        # Ensure non-negative values
        for key, values in metrics.items():
            metrics[key] = np.maximum(values, 0)
            
        return metrics
    
    def _generate_anomaly_metrics(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Generate anomalous system metrics."""
        metrics = self._generate_normal_metrics(n_samples)
        
        # Select random anomaly patterns
        pattern_names = list(self.anomaly_patterns.keys())
        selected_patterns = self.rng.choice(
            pattern_names, 
            size=n_samples, 
            replace=True
        )
        
        for i, pattern in enumerate(selected_patterns):
            pattern_config = self.anomaly_patterns[pattern]
            
            # Apply pattern-specific modifications
            if 'cpu_multiplier' in pattern_config:
                metrics['cpu_usage'][i] *= pattern_config['cpu_multiplier']
            if 'memory_multiplier' in pattern_config:
                metrics['memory_usage'][i] *= pattern_config['memory_multiplier']
            if 'disk_multiplier' in pattern_config:
                metrics['disk_usage'][i] *= pattern_config['disk_multiplier']
            if 'bytes_multiplier' in pattern_config:
                metrics['bytes_sent'][i] *= pattern_config['bytes_multiplier']
                metrics['bytes_received'][i] *= pattern_config['bytes_multiplier']
            if 'connections_multiplier' in pattern_config:
                metrics['active_connections'][i] *= pattern_config['connections_multiplier']
            if 'processes_multiplier' in pattern_config:
                metrics['process_count'][i] *= pattern_config['processes_multiplier']
        
        # Ensure values stay within reasonable bounds
        metrics['cpu_usage'] = np.minimum(metrics['cpu_usage'], 100)
        metrics['memory_usage'] = np.minimum(metrics['memory_usage'], 100)
        metrics['disk_usage'] = np.minimum(metrics['disk_usage'], 100)
        
        return metrics
    
    def _generate_timestamps(self, n_samples: int) -> List[datetime]:
        """Generate realistic timestamps."""
        start_time = datetime.now() - timedelta(hours=self.config.time_window_hours)
        timestamps = []
        
        for i in range(n_samples):
            # Add some randomness to timestamps
            offset_minutes = self.rng.uniform(0, self.config.time_window_hours * 60)
            timestamp = start_time + timedelta(minutes=offset_minutes)
            timestamps.append(timestamp)
        
        return sorted(timestamps)
    
    def _generate_network_features(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Generate additional network security features."""
        features = {
            'packet_size_variance': self.rng.exponential(1000, n_samples),
            'connection_duration': self.rng.exponential(300, n_samples),  # seconds
            'failed_login_attempts': self.rng.poisson(0.5, n_samples),
            'privilege_escalation_attempts': self.rng.poisson(0.1, n_samples),
            'suspicious_file_access': self.rng.poisson(0.2, n_samples),
        }
        
        return features
    
    def _generate_user_features(self, n_samples: int) -> Dict[str, List[str]]:
        """Generate user-related features with privacy protection."""
        # Generate hashed user IDs to protect privacy
        user_ids = []
        for i in range(n_samples):
            user_hash = hashlib.sha256(f"user_{i}".encode()).hexdigest()[:8]
            user_ids.append(f"usr_{user_hash}")
        
        # Generate hashed IP addresses
        ip_addresses = []
        for i in range(n_samples):
            ip_hash = hashlib.sha256(f"ip_{i}".encode()).hexdigest()[:8]
            ip_addresses.append(f"192.168.{int(ip_hash[:2], 16)}.{int(ip_hash[2:4], 16)}")
        
        # Generate user agents (simplified)
        user_agents = self.rng.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "curl/7.68.0",
            "python-requests/2.25.1"
        ], n_samples)
        
        return {
            'user_id': user_ids,
            'source_ip': ip_addresses,
            'user_agent': user_agents.tolist(),
        }
    
    def generate_logs(self, n_samples: int = 1000, anomaly_ratio: float = 0.1) -> Dict[str, Any]:
        """
        Generate synthetic security log data.
        
        Args:
            n_samples: Total number of log entries to generate
            anomaly_ratio: Proportion of anomalous entries
            
        Returns:
            Dictionary containing features, labels, and metadata
        """
        logger.info(f"Generating {n_samples} security log entries with {anomaly_ratio:.1%} anomalies")
        
        # Calculate sample sizes
        n_anomalies = int(n_samples * anomaly_ratio)
        n_normal = n_samples - n_anomalies
        
        # Generate normal and anomalous data
        normal_metrics = self._generate_normal_metrics(n_normal)
        anomaly_metrics = self._generate_anomaly_metrics(n_anomalies)
        
        # Combine data
        all_metrics = {}
        for key in normal_metrics.keys():
            all_metrics[key] = np.concatenate([
                normal_metrics[key], 
                anomaly_metrics[key]
            ])
        
        # Generate additional features
        network_features = self._generate_network_features(n_samples)
        user_features = self._generate_user_features(n_samples)
        
        # Create labels
        labels = np.concatenate([
            np.zeros(n_normal, dtype=int),  # Normal = 0
            np.ones(n_anomalies, dtype=int)  # Anomaly = 1
        ])
        
        # Generate timestamps
        timestamps = self._generate_timestamps(n_samples)
        
        # Shuffle data to mix normal and anomalous entries
        indices = np.arange(n_samples)
        self.rng.shuffle(indices)
        
        # Create DataFrame
        data_dict = {**all_metrics, **network_features, **user_features}
        data_dict['timestamp'] = timestamps
        data_dict['anomaly_label'] = labels
        
        df = pd.DataFrame(data_dict)
        df = df.iloc[indices].reset_index(drop=True)
        
        # Add derived features
        df['cpu_memory_ratio'] = df['cpu_usage'] / (df['memory_usage'] + 1e-6)
        df['network_activity'] = df['bytes_sent'] + df['bytes_received']
        df['resource_pressure'] = (df['cpu_usage'] + df['memory_usage'] + df['disk_usage']) / 3
        
        logger.info(f"Generated dataset with {len(df)} samples, {df['anomaly_label'].sum()} anomalies")
        
        return {
            'data': df,
            'features': [col for col in df.columns if col not in ['timestamp', 'anomaly_label']],
            'labels': df['anomaly_label'].values,
            'timestamps': df['timestamp'].values,
            'metadata': {
                'n_samples': n_samples,
                'anomaly_ratio': anomaly_ratio,
                'feature_names': [col for col in df.columns if col not in ['timestamp', 'anomaly_label']],
                'generated_at': datetime.now().isoformat(),
            }
        }
    
    def save_data(self, data: Dict[str, Any], filepath: str) -> None:
        """Save generated data to file."""
        data['data'].to_parquet(filepath, index=False)
        logger.info(f"Saved data to {filepath}")
    
    def load_data(self, filepath: str) -> Dict[str, Any]:
        """Load data from file."""
        df = pd.read_parquet(filepath)
        
        return {
            'data': df,
            'features': [col for col in df.columns if col not in ['timestamp', 'anomaly_label']],
            'labels': df['anomaly_label'].values,
            'timestamps': df['timestamp'].values,
            'metadata': {
                'loaded_from': filepath,
                'loaded_at': datetime.now().isoformat(),
            }
        }
