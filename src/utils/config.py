"""
Configuration Management System

Handles loading and validation of configuration files using OmegaConf.
Provides a centralized way to manage all system settings.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from omegaconf import OmegaConf, DictConfig
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConfigManager:
    """
    Configuration manager for the Security Anomaly Detection System.
    
    Handles loading, validation, and access to configuration settings
    using OmegaConf for structured configuration management.
    """
    
    config: DictConfig = field(default_factory=lambda: OmegaConf.create({}))
    config_path: Optional[str] = None
    overrides: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize the configuration manager."""
        if self.config_path is None:
            self.config_path = self._find_default_config()
        
        if self.config_path and Path(self.config_path).exists():
            self.load_config(self.config_path)
    
    def _find_default_config(self) -> str:
        """Find the default configuration file."""
        # Look for config files in order of preference
        config_paths = [
            "configs/default.yaml",
            "config/default.yaml",
            "config.yaml",
            "configs/config.yaml"
        ]
        
        for path in config_paths:
            if Path(path).exists():
                return path
        
        logger.warning("No configuration file found, using defaults")
        return None
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            self.config = OmegaConf.load(config_path)
            self.config_path = config_path
            logger.info(f"Configuration loaded from {config_path}")
            
            # Apply overrides if any
            if self.overrides:
                self.apply_overrides(self.overrides)
                
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    def save_config(self, config_path: str) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration
        """
        try:
            OmegaConf.save(self.config, config_path)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise
    
    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """
        Apply configuration overrides.
        
        Args:
            overrides: Dictionary of configuration overrides
        """
        try:
            self.config = OmegaConf.merge(self.config, OmegaConf.create(overrides))
            self.overrides.update(overrides)
            logger.info(f"Applied configuration overrides: {list(overrides.keys())}")
        except Exception as e:
            logger.error(f"Failed to apply overrides: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            return OmegaConf.select(self.config, key, default=default)
        except Exception as e:
            logger.warning(f"Failed to get config key '{key}': {e}")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        try:
            OmegaConf.set(self.config, key, value)
            logger.debug(f"Set config key '{key}' to {value}")
        except Exception as e:
            logger.error(f"Failed to set config key '{key}': {e}")
            raise
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data generation configuration."""
        return OmegaConf.to_container(self.config.data, resolve=True)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return OmegaConf.to_container(self.config.model, resolve=True)
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return OmegaConf.to_container(self.config.evaluation, resolve=True)
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return OmegaConf.to_container(self.config.visualization, resolve=True)
    
    def get_explainability_config(self) -> Dict[str, Any]:
        """Get explainability configuration."""
        return OmegaConf.to_container(self.config.explainability, resolve=True)
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return OmegaConf.to_container(self.config.logging, resolve=True)
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return OmegaConf.to_container(self.config.paths, resolve=True)
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return OmegaConf.to_container(self.config.security, resolve=True)
    
    def get_demo_config(self) -> Dict[str, Any]:
        """Get demo configuration."""
        return OmegaConf.to_container(self.config.demo, resolve=True)
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return OmegaConf.to_container(self.config.performance, resolve=True)
    
    def validate_config(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate required sections
            required_sections = [
                'data', 'model', 'evaluation', 'visualization',
                'explainability', 'logging', 'paths', 'security'
            ]
            
            for section in required_sections:
                if not OmegaConf.select(self.config, section):
                    logger.error(f"Missing required configuration section: {section}")
                    return False
            
            # Validate data configuration
            data_config = self.get_data_config()
            if data_config['n_samples'] <= 0:
                logger.error("n_samples must be positive")
                return False
            
            if not 0 <= data_config['anomaly_ratio'] <= 1:
                logger.error("anomaly_ratio must be between 0 and 1")
                return False
            
            # Validate model configuration
            model_config = self.get_model_config()
            valid_methods = ['isolation_forest', 'autoencoder', 'lstm', 'ensemble']
            if model_config['method'] not in valid_methods:
                logger.error(f"Invalid model method: {model_config['method']}")
                return False
            
            # Validate paths
            paths_config = self.get_paths_config()
            for path_key, path_value in paths_config.items():
                if isinstance(path_value, str) and path_key.endswith('_dir'):
                    Path(path_value).mkdir(parents=True, exist_ok=True)
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def create_directories(self) -> None:
        """Create necessary directories based on configuration."""
        paths_config = self.get_paths_config()
        
        directories = [
            paths_config['data_dir'],
            paths_config['models_dir'],
            paths_config['results_dir'],
            paths_config['plots_dir'],
            paths_config['logs_dir']
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    def get_device_config(self) -> str:
        """Get device configuration with fallback logic."""
        device_config = self.get('performance.device', 'auto')
        
        if device_config == 'auto':
            import torch
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        
        return device_config
    
    def get_random_state(self) -> int:
        """Get random state from configuration."""
        return self.get('data.random_state', 42)
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.get('development.debug', False)
    
    def is_verbose_mode(self) -> bool:
        """Check if verbose mode is enabled."""
        return self.get('development.verbose', False)
    
    def should_save_plots(self) -> bool:
        """Check if plots should be saved."""
        return self.get('visualization.save_plots', True)
    
    def should_save_explanations(self) -> bool:
        """Check if explanations should be saved."""
        return self.get('explainability.report.save_explanations', True)
    
    def get_max_anomalies_to_explain(self) -> int:
        """Get maximum number of anomalies to explain."""
        return self.get('explainability.report.max_anomalies_to_explain', 10)
    
    def get_sample_size(self) -> int:
        """Get sample size for data generation."""
        return self.get('data.n_samples', 1000)
    
    def get_anomaly_ratio(self) -> float:
        """Get anomaly ratio for data generation."""
        return self.get('data.anomaly_ratio', 0.1)
    
    def get_model_method(self) -> str:
        """Get model method."""
        return self.get('model.method', 'isolation_forest')
    
    def get_contamination(self) -> float:
        """Get contamination parameter."""
        method = self.get_model_method()
        return self.get(f'model.{method}.contamination', 0.1)
    
    def get_epochs(self) -> int:
        """Get number of epochs for neural network models."""
        method = self.get_model_method()
        return self.get(f'model.{method}.epochs', 100)
    
    def get_batch_size(self) -> int:
        """Get batch size for neural network models."""
        method = self.get_model_method()
        return self.get(f'model.{method}.batch_size', 32)
    
    def get_learning_rate(self) -> float:
        """Get learning rate for neural network models."""
        method = self.get_model_method()
        return self.get(f'model.{method}.learning_rate', 0.001)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return OmegaConf.to_container(self.config, resolve=True)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return OmegaConf.to_yaml(self.config)
    
    def __repr__(self) -> str:
        """Representation of configuration manager."""
        return f"ConfigManager(config_path='{self.config_path}')"


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        Global ConfigManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    return _config_manager


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> ConfigManager:
    """
    Load configuration and return manager instance.
    
    Args:
        config_path: Path to configuration file
        overrides: Optional configuration overrides
        
    Returns:
        ConfigManager instance
    """
    manager = ConfigManager()
    manager.load_config(config_path)
    
    if overrides:
        manager.apply_overrides(overrides)
    
    return manager


def create_config_from_dict(config_dict: Dict[str, Any]) -> ConfigManager:
    """
    Create configuration manager from dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        ConfigManager instance
    """
    manager = ConfigManager()
    manager.config = OmegaConf.create(config_dict)
    return manager


# Convenience functions for common configuration access
def get_config() -> ConfigManager:
    """Get the global configuration manager."""
    return get_config_manager()


def get_data_config() -> Dict[str, Any]:
    """Get data configuration."""
    return get_config_manager().get_data_config()


def get_model_config() -> Dict[str, Any]:
    """Get model configuration."""
    return get_config_manager().get_model_config()


def get_evaluation_config() -> Dict[str, Any]:
    """Get evaluation configuration."""
    return get_config_manager().get_evaluation_config()


def get_visualization_config() -> Dict[str, Any]:
    """Get visualization configuration."""
    return get_config_manager().get_visualization_config()


def get_explainability_config() -> Dict[str, Any]:
    """Get explainability configuration."""
    return get_config_manager().get_explainability_config()


def get_security_config() -> Dict[str, Any]:
    """Get security configuration."""
    return get_config_manager().get_security_config()


def get_demo_config() -> Dict[str, Any]:
    """Get demo configuration."""
    return get_config_manager().get_demo_config()
