#!/usr/bin/env python3
"""
Example script demonstrating the Security Anomaly Detection System.

This script shows how to use the system for basic anomaly detection
with different models and evaluation metrics.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data.generator import SecurityLogGenerator
from src.models.anomaly_detector import AnomalyDetector
from src.evaluation.metrics import SecurityMetrics
from src.visualization.plotter import AnomalyPlotter
from src.utils.config import get_config_manager
from src.utils.privacy import get_privacy_manager


def main():
    """Main example function."""
    print("🔒 Security Anomaly Detection System - Example")
    print("=" * 50)
    
    # Initialize privacy protection
    privacy_manager = get_privacy_manager()
    print("✅ Privacy protection initialized")
    
    # Load configuration
    config_manager = get_config_manager()
    print("✅ Configuration loaded")
    
    # Generate synthetic security log data
    print("\n📊 Generating synthetic security log data...")
    generator = SecurityLogGenerator(random_state=42)
    data = generator.generate_logs(n_samples=1000, anomaly_ratio=0.1)
    
    print(f"✅ Generated {len(data['data'])} samples with {data['labels'].sum()} anomalies")
    
    # Test different anomaly detection methods
    methods = ['isolation_forest', 'autoencoder', 'ensemble']
    results = {}
    
    for method in methods:
        print(f"\n🤖 Testing {method} anomaly detection...")
        
        try:
            # Initialize detector
            detector = AnomalyDetector(method=method, random_state=42)
            
            # Run detection
            predictions = detector.fit_predict(data)
            scores = detector.decision_function(data)
            
            # Evaluate performance
            metrics = SecurityMetrics()
            evaluation_results = metrics.compute_all(data['labels'], predictions, scores)
            
            results[method] = {
                'predictions': predictions,
                'scores': scores,
                'metrics': evaluation_results
            }
            
            print(f"✅ {method} completed - Accuracy: {evaluation_results['accuracy']:.3f}")
            
            # Log model operation
            privacy_manager.log_model_operation('predict', method)
            
        except Exception as e:
            print(f"❌ {method} failed: {e}")
            continue
    
    # Compare results
    print("\n📈 Performance Comparison:")
    print("-" * 40)
    
    comparison_data = []
    for method, result in results.items():
        metrics = result['metrics']
        comparison_data.append({
            'Method': method,
            'Accuracy': f"{metrics['accuracy']:.3f}",
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}",
            'F1-Score': f"{metrics['f1_score']:.3f}",
            'Detection Rate': f"{metrics['detection_rate']:.3f}",
            'False Alarm Rate': f"{metrics['false_alarm_rate']:.3f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Generate comprehensive report for best method
    if results:
        best_method = max(results.keys(), key=lambda x: results[x]['metrics']['accuracy'])
        best_result = results[best_method]
        
        print(f"\n📋 Detailed Report for {best_method}:")
        print("-" * 40)
        
        report = metrics.generate_report(
            data['labels'], 
            best_result['predictions'], 
            best_result['scores']
        )
        print(report)
        
        # Log anomaly detection
        privacy_manager.log_anomaly_detection(
            anomaly_count=best_result['predictions'].sum(),
            detection_method=best_method
        )
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    if results:
        plotter = AnomalyPlotter()
        
        # Plot for best method
        best_result = results[best_method]
        
        try:
            plotter.plot_anomalies(data, best_result['predictions'], best_result['scores'])
            print("✅ Anomaly visualization created")
        except Exception as e:
            print(f"⚠️  Visualization failed: {e}")
        
        try:
            plotter.plot_anomaly_scores_distribution(best_result['scores'], data['labels'])
            print("✅ Score distribution plot created")
        except Exception as e:
            print(f"⚠️  Score distribution plot failed: {e}")
    
    # Privacy report
    print("\n🔒 Privacy and Security Report:")
    print("-" * 40)
    
    privacy_report = privacy_manager.get_privacy_report()
    print(f"Privacy Protection: {privacy_report['data_protection_status']}")
    print(f"Compliance Status: {privacy_report['compliance_status']}")
    print(f"Total Audit Events: {privacy_report['audit_log_summary']['total_events']}")
    
    # Export results
    print("\n💾 Exporting results...")
    
    try:
        # Export privacy report
        privacy_manager.export_privacy_report('privacy_report.json')
        print("✅ Privacy report exported")
        
        # Export audit log
        privacy_manager.security_auditor.export_audit_log('audit_log.csv')
        print("✅ Audit log exported")
        
        # Export comparison results
        comparison_df.to_csv('model_comparison.csv', index=False)
        print("✅ Model comparison exported")
        
    except Exception as e:
        print(f"⚠️  Export failed: {e}")
    
    # Cleanup
    privacy_manager.cleanup_temp_files()
    print("✅ Temporary files cleaned up")
    
    print("\n🎉 Example completed successfully!")
    print("\n⚠️  REMINDER: This is a research/educational tool only.")
    print("   Not intended for production security operations.")


if __name__ == "__main__":
    main()
