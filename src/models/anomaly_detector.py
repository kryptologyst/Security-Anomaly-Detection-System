"""
Advanced Anomaly Detection Models

Implements multiple anomaly detection algorithms including Isolation Forest,
Autoencoders, LSTM-based models, and ensemble methods for security applications.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.lstm import LSTM
from pyod.models.ensemble import IsolationForest as PyODIsolationForest
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from abc import ABC, abstractmethod
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseAnomalyDetector(ABC):
    """Abstract base class for anomaly detectors."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the detector."""
        self.random_state = random_state
        self.is_fitted = False
        self.scaler = StandardScaler()
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseAnomalyDetector':
        """Fit the anomaly detector."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        pass
    
    @abstractmethod
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        pass
    
    def fit_predict(self, data: Dict[str, Any]) -> np.ndarray:
        """Fit and predict in one step."""
        X = self._prepare_features(data)
        self.fit(X)
        return self.predict(X)
    
    def _prepare_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Prepare features from data dictionary."""
        if isinstance(data, dict) and 'data' in data:
            df = data['data']
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'anomaly_label']]
            return df[feature_cols].values
        elif isinstance(data, pd.DataFrame):
            return data.values
        else:
            return np.array(data)


class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest anomaly detector."""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        super().__init__(random_state)
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
    
    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """Fit the Isolation Forest model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        logger.info("Isolation Forest fitted successfully")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 = anomaly, 0 = normal)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        # Convert from -1/1 to 0/1
        return (predictions == -1).astype(int)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        scores = self.model.decision_function(X_scaled)
        return -scores  # Lower scores = more anomalous


class AutoencoderDetector(BaseAnomalyDetector):
    """Autoencoder-based anomaly detector."""
    
    def __init__(
        self,
        hidden_dims: List[int] = [64, 32, 16],
        contamination: float = 0.1,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        random_state: int = 42
    ):
        super().__init__(random_state)
        self.hidden_dims = hidden_dims
        self.contamination = contamination
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.threshold = None
    
    def _build_model(self, input_dim: int) -> nn.Module:
        """Build the autoencoder model."""
        layers = []
        dims = [input_dim] + self.hidden_dims + self.hidden_dims[::-1] + [input_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # Don't add activation after output layer
                layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
    
    def fit(self, X: np.ndarray) -> 'AutoencoderDetector':
        """Fit the autoencoder model."""
        X_scaled = self.scaler.fit_transform(X)
        
        # Build model
        input_dim = X_scaled.shape[1]
        self.model = self._build_model(input_dim).to(self.device)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                reconstructed = self.model(batch[0])
                loss = criterion(reconstructed, batch[0])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.6f}")
        
        # Calculate threshold
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            self.threshold = torch.quantile(reconstruction_errors, 1 - self.contamination)
        
        self.is_fitted = True
        logger.info(f"Autoencoder fitted successfully, threshold: {self.threshold:.6f}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            predictions = (reconstruction_errors > self.threshold).cpu().numpy()
        
        return predictions.astype(int)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
        
        return reconstruction_errors.cpu().numpy()


class LSTMAutoencoderDetector(BaseAnomalyDetector):
    """LSTM-based autoencoder for time series anomaly detection."""
    
    def __init__(
        self,
        sequence_length: int = 10,
        hidden_dim: int = 64,
        num_layers: int = 2,
        contamination: float = 0.1,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        random_state: int = 42
    ):
        super().__init__(random_state)
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.contamination = contamination
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.threshold = None
    
    def _build_model(self, input_dim: int) -> nn.Module:
        """Build the LSTM autoencoder model."""
        class LSTMAutoencoder(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, sequence_length):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                self.sequence_length = sequence_length
                
                # Encoder
                self.encoder = nn.LSTM(
                    input_dim, hidden_dim, num_layers,
                    batch_first=True, dropout=0.1
                )
                
                # Decoder
                self.decoder = nn.LSTM(
                    hidden_dim, input_dim, num_layers,
                    batch_first=True, dropout=0.1
                )
            
            def forward(self, x):
                # Encode
                encoded, (hidden, cell) = self.encoder(x)
                
                # Decode
                decoded, _ = self.decoder(encoded, (hidden, cell))
                
                return decoded
        
        return LSTMAutoencoder(input_dim, self.hidden_dim, self.num_layers, self.sequence_length)
    
    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """Create sequences from time series data."""
        sequences = []
        for i in range(len(X) - self.sequence_length + 1):
            sequences.append(X[i:i + self.sequence_length])
        return np.array(sequences)
    
    def fit(self, X: np.ndarray) -> 'LSTMAutoencoderDetector':
        """Fit the LSTM autoencoder model."""
        X_scaled = self.scaler.fit_transform(X)
        X_sequences = self._create_sequences(X_scaled)
        
        if len(X_sequences) == 0:
            raise ValueError("Not enough data to create sequences")
        
        # Build model
        input_dim = X_scaled.shape[1]
        self.model = self._build_model(input_dim).to(self.device)
        
        # Prepare data
        X_tensor = torch.FloatTensor(X_sequences).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                reconstructed = self.model(batch[0])
                loss = criterion(reconstructed, batch[0])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.6f}")
        
        # Calculate threshold
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=(1, 2))
            self.threshold = torch.quantile(reconstruction_errors, 1 - self.contamination)
        
        self.is_fitted = True
        logger.info(f"LSTM Autoencoder fitted successfully, threshold: {self.threshold:.6f}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_sequences = self._create_sequences(X_scaled)
        
        if len(X_sequences) == 0:
            return np.zeros(len(X), dtype=int)
        
        X_tensor = torch.FloatTensor(X_sequences).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=(1, 2))
            predictions = (reconstruction_errors > self.threshold).cpu().numpy()
        
        # Pad predictions to match original length
        full_predictions = np.zeros(len(X), dtype=int)
        full_predictions[:len(predictions)] = predictions
        
        return full_predictions
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        X_sequences = self._create_sequences(X_scaled)
        
        if len(X_sequences) == 0:
            return np.zeros(len(X))
        
        X_tensor = torch.FloatTensor(X_sequences).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=(1, 2))
        
        # Pad scores to match original length
        full_scores = np.zeros(len(X))
        full_scores[:len(reconstruction_errors)] = reconstruction_errors.cpu().numpy()
        
        return full_scores


class EnsembleAnomalyDetector(BaseAnomalyDetector):
    """Ensemble of multiple anomaly detection methods."""
    
    def __init__(
        self,
        detectors: Optional[List[BaseAnomalyDetector]] = None,
        voting_method: str = 'soft',
        random_state: int = 42
    ):
        super().__init__(random_state)
        self.detectors = detectors or [
            IsolationForestDetector(random_state=random_state),
            AutoencoderDetector(random_state=random_state),
        ]
        self.voting_method = voting_method
        self.scores_history = []
    
    def fit(self, X: np.ndarray) -> 'EnsembleAnomalyDetector':
        """Fit all detectors."""
        for i, detector in enumerate(self.detectors):
            logger.info(f"Fitting detector {i+1}/{len(self.detectors)}")
            detector.fit(X)
        
        self.is_fitted = True
        logger.info("Ensemble fitted successfully")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using ensemble voting."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        scores = []
        
        for detector in self.detectors:
            pred = detector.predict(X)
            score = detector.decision_function(X)
            predictions.append(pred)
            scores.append(score)
        
        predictions = np.array(predictions)
        scores = np.array(scores)
        
        if self.voting_method == 'hard':
            # Majority voting
            ensemble_pred = (np.mean(predictions, axis=0) > 0.5).astype(int)
        else:
            # Soft voting based on average scores
            avg_scores = np.mean(scores, axis=0)
            threshold = np.quantile(avg_scores, 1 - 0.1)  # 10% contamination
            ensemble_pred = (avg_scores > threshold).astype(int)
        
        return ensemble_pred
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble anomaly scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        scores = []
        for detector in self.detectors:
            score = detector.decision_function(X)
            scores.append(score)
        
        # Average scores across detectors
        return np.mean(scores, axis=0)


class AnomalyDetector:
    """Main anomaly detector class that wraps different methods."""
    
    def __init__(
        self,
        method: str = 'isolation_forest',
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize the anomaly detector.
        
        Args:
            method: Detection method ('isolation_forest', 'autoencoder', 'lstm', 'ensemble')
            random_state: Random seed
            **kwargs: Additional parameters for specific methods
        """
        self.method = method
        self.random_state = random_state
        
        if method == 'isolation_forest':
            self.detector = IsolationForestDetector(random_state=random_state, **kwargs)
        elif method == 'autoencoder':
            self.detector = AutoencoderDetector(random_state=random_state, **kwargs)
        elif method == 'lstm':
            self.detector = LSTMAutoencoderDetector(random_state=random_state, **kwargs)
        elif method == 'ensemble':
            self.detector = EnsembleAnomalyDetector(random_state=random_state, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Initialized {method} anomaly detector")
    
    def fit_predict(self, data: Dict[str, Any]) -> np.ndarray:
        """Fit and predict anomalies."""
        return self.detector.fit_predict(data)
    
    def predict(self, data: Dict[str, Any]) -> np.ndarray:
        """Predict anomalies."""
        X = self.detector._prepare_features(data)
        return self.detector.predict(X)
    
    def decision_function(self, data: Dict[str, Any]) -> np.ndarray:
        """Get anomaly scores."""
        X = self.detector._prepare_features(data)
        return self.detector.decision_function(X)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.detector.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'method': self.method,
            'random_state': self.random_state,
            'detector': self.detector,
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        model_data = joblib.load(filepath)
        self.method = model_data['method']
        self.random_state = model_data['random_state']
        self.detector = model_data['detector']
        logger.info(f"Model loaded from {filepath}")
