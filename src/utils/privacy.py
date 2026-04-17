"""
Privacy and Safety Utilities

Provides comprehensive privacy protection, PII handling, and safety measures
for the Security Anomaly Detection System.
"""

import hashlib
import re
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings

logger = logging.getLogger(__name__)


@dataclass
class PrivacyConfig:
    """Configuration for privacy protection settings."""
    
    # Data anonymization
    hash_user_ids: bool = True
    hash_ip_addresses: bool = True
    anonymize_timestamps: bool = False
    remove_pii: bool = True
    
    # Output sanitization
    redact_sensitive_data: bool = True
    sanitize_logs: bool = True
    audit_trail: bool = True
    
    # Model security
    save_model_weights: bool = True
    encrypt_model_files: bool = False
    validate_inputs: bool = True
    
    # Data retention
    max_data_age_days: int = 30
    auto_delete_temp_files: bool = True
    
    # Compliance
    gdpr_compliant: bool = True
    ccpa_compliant: bool = True
    hipaa_compliant: bool = False  # Set to True if handling health data


class PIIHandler:
    """
    Handles Personally Identifiable Information (PII) detection and protection.
    
    Provides methods to detect, redact, and anonymize PII in data and outputs.
    """
    
    def __init__(self, config: Optional[PrivacyConfig] = None):
        """
        Initialize PII handler.
        
        Args:
            config: Privacy configuration settings
        """
        self.config = config or PrivacyConfig()
        
        # PII patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'mac_address': r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b',
            'url': r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
        }
        
        # Sensitive keywords
        self.sensitive_keywords = [
            'password', 'passwd', 'pwd', 'secret', 'key', 'token',
            'api_key', 'access_token', 'refresh_token', 'auth',
            'credential', 'login', 'username', 'user_id'
        ]
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """
        Detect PII in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary mapping PII types to detected values
        """
        detected_pii = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected_pii[pii_type] = matches
        
        # Check for sensitive keywords
        sensitive_matches = []
        text_lower = text.lower()
        for keyword in self.sensitive_keywords:
            if keyword in text_lower:
                sensitive_matches.append(keyword)
        
        if sensitive_matches:
            detected_pii['sensitive_keywords'] = sensitive_matches
        
        return detected_pii
    
    def redact_pii(self, text: str, replacement: str = "[REDACTED]") -> str:
        """
        Redact PII from text.
        
        Args:
            text: Text to redact
            replacement: Replacement string for PII
            
        Returns:
            Text with PII redacted
        """
        redacted_text = text
        
        for pii_type, pattern in self.pii_patterns.items():
            redacted_text = re.sub(pattern, replacement, redacted_text, flags=re.IGNORECASE)
        
        return redacted_text
    
    def hash_identifier(self, identifier: str, salt: str = "") -> str:
        """
        Hash an identifier for privacy protection.
        
        Args:
            identifier: Identifier to hash
            salt: Optional salt for hashing
            
        Returns:
            Hashed identifier
        """
        if not identifier:
            return ""
        
        # Create hash with salt
        hash_input = f"{identifier}{salt}".encode('utf-8')
        hash_object = hashlib.sha256(hash_input)
        hash_hex = hash_object.hexdigest()
        
        # Return first 8 characters for readability
        return hash_hex[:8]
    
    def anonymize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Anonymize PII in DataFrame.
        
        Args:
            df: DataFrame to anonymize
            
        Returns:
            Anonymized DataFrame
        """
        anonymized_df = df.copy()
        
        # Hash user IDs
        if self.config.hash_user_ids and 'user_id' in anonymized_df.columns:
            anonymized_df['user_id'] = anonymized_df['user_id'].apply(
                lambda x: f"usr_{self.hash_identifier(str(x))}" if pd.notna(x) else x
            )
        
        # Hash IP addresses
        if self.config.hash_ip_addresses and 'source_ip' in anonymized_df.columns:
            anonymized_df['source_ip'] = anonymized_df['source_ip'].apply(
                lambda x: f"ip_{self.hash_identifier(str(x))}" if pd.notna(x) else x
            )
        
        # Anonymize timestamps
        if self.config.anonymize_timestamps and 'timestamp' in anonymized_df.columns:
            anonymized_df['timestamp'] = pd.to_datetime(anonymized_df['timestamp']).dt.floor('H')
        
        return anonymized_df
    
    def sanitize_output(self, output: Any) -> Any:
        """
        Sanitize output to remove sensitive information.
        
        Args:
            output: Output to sanitize
            
        Returns:
            Sanitized output
        """
        if isinstance(output, str):
            return self.redact_pii(output)
        elif isinstance(output, dict):
            return {k: self.sanitize_output(v) for k, v in output.items()}
        elif isinstance(output, list):
            return [self.sanitize_output(item) for item in output]
        elif isinstance(output, pd.DataFrame):
            return self.anonymize_dataframe(output)
        else:
            return output


class SecurityAuditor:
    """
    Security auditor for tracking and logging security-related activities.
    
    Provides audit trail functionality and security event logging.
    """
    
    def __init__(self, config: Optional[PrivacyConfig] = None):
        """
        Initialize security auditor.
        
        Args:
            config: Privacy configuration settings
        """
        self.config = config or PrivacyConfig()
        self.audit_log = []
        
        # Set up audit logging
        if self.config.audit_trail:
            self._setup_audit_logging()
    
    def _setup_audit_logging(self):
        """Set up audit logging configuration."""
        audit_logger = logging.getLogger('security_audit')
        audit_logger.setLevel(logging.INFO)
        
        # Create audit log file handler
        audit_handler = logging.FileHandler('logs/security_audit.log')
        audit_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(formatter)
        
        # Add handler to logger
        audit_logger.addHandler(audit_handler)
        
        self.audit_logger = audit_logger
    
    def log_security_event(
        self, 
        event_type: str, 
        description: str, 
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a security event.
        
        Args:
            event_type: Type of security event
            description: Description of the event
            user_id: User ID (if applicable)
            ip_address: IP address (if applicable)
            additional_data: Additional event data
        """
        event = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'event_type': event_type,
            'description': description,
            'user_id': user_id,
            'ip_address': ip_address,
            'additional_data': additional_data or {}
        }
        
        # Add to audit log
        self.audit_log.append(event)
        
        # Log to file if audit trail is enabled
        if self.config.audit_trail and hasattr(self, 'audit_logger'):
            self.audit_logger.info(f"{event_type}: {description}")
        
        logger.info(f"Security event logged: {event_type}")
    
    def log_data_access(
        self, 
        data_type: str, 
        access_type: str, 
        user_id: Optional[str] = None
    ) -> None:
        """
        Log data access events.
        
        Args:
            data_type: Type of data accessed
            access_type: Type of access (read, write, delete)
            user_id: User ID (if applicable)
        """
        self.log_security_event(
            event_type='data_access',
            description=f"{access_type} access to {data_type}",
            user_id=user_id,
            additional_data={'data_type': data_type, 'access_type': access_type}
        )
    
    def log_model_operation(
        self, 
        operation: str, 
        model_type: str, 
        user_id: Optional[str] = None
    ) -> None:
        """
        Log model operations.
        
        Args:
            operation: Type of operation (train, predict, evaluate)
            model_type: Type of model
            user_id: User ID (if applicable)
        """
        self.log_security_event(
            event_type='model_operation',
            description=f"{operation} operation on {model_type} model",
            user_id=user_id,
            additional_data={'operation': operation, 'model_type': model_type}
        )
    
    def log_anomaly_detection(
        self, 
        anomaly_count: int, 
        detection_method: str, 
        user_id: Optional[str] = None
    ) -> None:
        """
        Log anomaly detection events.
        
        Args:
            anomaly_count: Number of anomalies detected
            detection_method: Method used for detection
            user_id: User ID (if applicable)
        """
        self.log_security_event(
            event_type='anomaly_detection',
            description=f"Detected {anomaly_count} anomalies using {detection_method}",
            user_id=user_id,
            additional_data={
                'anomaly_count': anomaly_count,
                'detection_method': detection_method
            }
        )
    
    def get_audit_log(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get audit log entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of audit log entries
        """
        if limit:
            return self.audit_log[-limit:]
        return self.audit_log.copy()
    
    def export_audit_log(self, filepath: str) -> None:
        """
        Export audit log to file.
        
        Args:
            filepath: Path to export file
        """
        audit_df = pd.DataFrame(self.audit_log)
        audit_df.to_csv(filepath, index=False)
        logger.info(f"Audit log exported to {filepath}")


class DataValidator:
    """
    Data validator for input validation and sanitization.
    
    Provides comprehensive input validation for security and data integrity.
    """
    
    def __init__(self, config: Optional[PrivacyConfig] = None):
        """
        Initialize data validator.
        
        Args:
            config: Privacy configuration settings
        """
        self.config = config or PrivacyConfig()
        self.pii_handler = PIIHandler(config)
    
    def validate_input_data(self, data: Any) -> Dict[str, Any]:
        """
        Validate input data for security and integrity.
        
        Args:
            data: Input data to validate
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'pii_detected': False,
            'sanitized_data': data
        }
        
        try:
            # Check for PII
            if isinstance(data, str):
                pii_detected = self.pii_handler.detect_pii(data)
                if pii_detected:
                    validation_results['pii_detected'] = True
                    validation_results['warnings'].append("PII detected in input data")
                    
                    if self.config.remove_pii:
                        validation_results['sanitized_data'] = self.pii_handler.redact_pii(data)
            
            # Validate DataFrame
            elif isinstance(data, pd.DataFrame):
                validation_results.update(self._validate_dataframe(data))
            
            # Validate numpy array
            elif isinstance(data, np.ndarray):
                validation_results.update(self._validate_numpy_array(data))
            
            # Validate dictionary
            elif isinstance(data, dict):
                validation_results.update(self._validate_dictionary(data))
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    def _validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate DataFrame."""
        results = {'is_valid': True, 'warnings': [], 'errors': []}
        
        # Check for required columns
        required_columns = ['cpu_usage', 'memory_usage']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            results['errors'].append(f"Missing required columns: {missing_columns}")
            results['is_valid'] = False
        
        # Check for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            results['warnings'].append("No numeric columns found")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            results['warnings'].append(f"Found {missing_count} missing values")
        
        # Check for outliers
        for col in numeric_columns:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
                if len(outliers) > 0:
                    results['warnings'].append(f"Found {len(outliers)} outliers in column {col}")
        
        return results
    
    def _validate_numpy_array(self, arr: np.ndarray) -> Dict[str, Any]:
        """Validate numpy array."""
        results = {'is_valid': True, 'warnings': [], 'errors': []}
        
        # Check for NaN values
        if np.isnan(arr).any():
            results['warnings'].append("Array contains NaN values")
        
        # Check for infinite values
        if np.isinf(arr).any():
            results['warnings'].append("Array contains infinite values")
        
        # Check for negative values in positive-only columns
        if np.any(arr < 0):
            results['warnings'].append("Array contains negative values")
        
        return results
    
    def _validate_dictionary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dictionary."""
        results = {'is_valid': True, 'warnings': [], 'errors': []}
        
        # Check for required keys
        required_keys = ['data', 'labels']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            results['errors'].append(f"Missing required keys: {missing_keys}")
            results['is_valid'] = False
        
        return results
    
    def sanitize_input(self, data: Any) -> Any:
        """
        Sanitize input data.
        
        Args:
            data: Input data to sanitize
            
        Returns:
            Sanitized data
        """
        return self.pii_handler.sanitize_output(data)


class PrivacyManager:
    """
    Main privacy manager that coordinates all privacy and safety measures.
    
    Provides a unified interface for privacy protection, security auditing,
    and data validation.
    """
    
    def __init__(self, config: Optional[PrivacyConfig] = None):
        """
        Initialize privacy manager.
        
        Args:
            config: Privacy configuration settings
        """
        self.config = config or PrivacyConfig()
        self.pii_handler = PIIHandler(config)
        self.security_auditor = SecurityAuditor(config)
        self.data_validator = DataValidator(config)
        
        # Create necessary directories
        Path('logs').mkdir(exist_ok=True)
        Path('data').mkdir(exist_ok=True)
        Path('models').mkdir(exist_ok=True)
    
    def process_data(self, data: Any, operation: str = "process") -> Any:
        """
        Process data with privacy protection.
        
        Args:
            data: Data to process
            operation: Type of operation
            
        Returns:
            Processed data with privacy protection applied
        """
        # Log data access
        self.security_auditor.log_data_access(
            data_type=type(data).__name__,
            access_type=operation,
            user_id=None  # Would be provided in real implementation
        )
        
        # Validate input
        validation_results = self.data_validator.validate_input_data(data)
        
        if not validation_results['is_valid']:
            self.security_auditor.log_security_event(
                event_type='data_validation_failed',
                description=f"Data validation failed: {validation_results['errors']}"
            )
            raise ValueError(f"Data validation failed: {validation_results['errors']}")
        
        # Apply privacy protection
        if self.config.remove_pii:
            processed_data = self.pii_handler.sanitize_output(data)
        else:
            processed_data = data
        
        # Log warnings
        if validation_results['warnings']:
            self.security_auditor.log_security_event(
                event_type='data_validation_warning',
                description=f"Data validation warnings: {validation_results['warnings']}"
            )
        
        return processed_data
    
    def log_model_operation(self, operation: str, model_type: str) -> None:
        """Log model operation."""
        self.security_auditor.log_model_operation(operation, model_type)
    
    def log_anomaly_detection(self, anomaly_count: int, detection_method: str) -> None:
        """Log anomaly detection event."""
        self.security_auditor.log_anomaly_detection(anomaly_count, detection_method)
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """
        Generate privacy compliance report.
        
        Returns:
            Privacy compliance report
        """
        report = {
            'privacy_config': {
                'hash_user_ids': self.config.hash_user_ids,
                'hash_ip_addresses': self.config.hash_ip_addresses,
                'remove_pii': self.config.remove_pii,
                'audit_trail': self.config.audit_trail,
                'gdpr_compliant': self.config.gdpr_compliant,
                'ccpa_compliant': self.config.ccpa_compliant,
                'hipaa_compliant': self.config.hipaa_compliant
            },
            'audit_log_summary': {
                'total_events': len(self.security_auditor.audit_log),
                'recent_events': len(self.security_auditor.get_audit_log(limit=10))
            },
            'data_protection_status': 'ACTIVE',
            'compliance_status': 'COMPLIANT' if self.config.gdpr_compliant else 'NON_COMPLIANT'
        }
        
        return report
    
    def export_privacy_report(self, filepath: str) -> None:
        """
        Export privacy report to file.
        
        Args:
            filepath: Path to export file
        """
        report = self.get_privacy_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Privacy report exported to {filepath}")
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        if self.config.auto_delete_temp_files:
            temp_dirs = ['data/temp', 'models/temp', 'logs/temp']
            for temp_dir in temp_dirs:
                temp_path = Path(temp_dir)
                if temp_path.exists():
                    for file in temp_path.iterdir():
                        if file.is_file():
                            file.unlink()
                    logger.info(f"Cleaned up temporary files in {temp_dir}")


# Global privacy manager instance
_privacy_manager: Optional[PrivacyManager] = None


def get_privacy_manager() -> PrivacyManager:
    """
    Get the global privacy manager instance.
    
    Returns:
        Global PrivacyManager instance
    """
    global _privacy_manager
    
    if _privacy_manager is None:
        _privacy_manager = PrivacyManager()
    
    return _privacy_manager


def setup_privacy_protection(config: Optional[PrivacyConfig] = None) -> PrivacyManager:
    """
    Set up privacy protection for the application.
    
    Args:
        config: Privacy configuration settings
        
    Returns:
        Configured PrivacyManager instance
    """
    global _privacy_manager
    
    _privacy_manager = PrivacyManager(config)
    
    # Log privacy protection setup
    _privacy_manager.security_auditor.log_security_event(
        event_type='privacy_protection_setup',
        description="Privacy protection system initialized"
    )
    
    return _privacy_manager
