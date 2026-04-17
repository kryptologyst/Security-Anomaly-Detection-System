"""
Interactive Streamlit Demo for Security Anomaly Detection

Provides a comprehensive web interface for exploring anomaly detection
results, model performance, and explainability features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.generator import SecurityLogGenerator
from src.models.anomaly_detector import AnomalyDetector
from src.evaluation.metrics import SecurityMetrics
from src.visualization.plotter import AnomalyPlotter, AnomalyExplainer
from src.utils.config import get_config_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Security Anomaly Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .anomaly-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .normal-status {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)


class SecurityAnomalyDemo:
    """Main demo class for the Security Anomaly Detection system."""
    
    def __init__(self):
        """Initialize the demo."""
        self.config_manager = get_config_manager()
        self.data_generator = None
        self.detector = None
        self.metrics = None
        self.plotter = None
        self.explainer = None
        
        # Initialize session state
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None
        if 'scores' not in st.session_state:
            st.session_state.scores = None
        if 'metrics_results' not in st.session_state:
            st.session_state.metrics_results = None
    
    def run(self):
        """Run the main demo application."""
        # Header
        st.markdown('<h1 class="main-header">🔒 Security Anomaly Detection System</h1>', 
                   unsafe_allow_html=True)
        
        # Security disclaimer
        self._show_security_disclaimer()
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "🔍 Data Explorer", 
            "🤖 Model Performance", 
            "💡 Explainability", 
            "⚙️ Configuration"
        ])
        
        with tab1:
            self._render_overview_tab()
        
        with tab2:
            self._render_data_explorer_tab()
        
        with tab3:
            self._render_model_performance_tab()
        
        with tab4:
            self._render_explainability_tab()
        
        with tab5:
            self._render_configuration_tab()
    
    def _show_security_disclaimer(self):
        """Show security disclaimer."""
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Security Research Disclaimer</h4>
            <p><strong>This is a research and educational demonstration tool only.</strong></p>
            <ul>
                <li>Not intended for production security operations</li>
                <li>May not accurately detect real-world threats</li>
                <li>Use only for learning and defensive security research</li>
                <li>All data is synthetic and contains no real PII</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render the sidebar with controls."""
        st.sidebar.header("🎛️ Controls")
        
        # Data generation controls
        st.sidebar.subheader("📊 Data Generation")
        
        n_samples = st.sidebar.slider(
            "Number of Samples", 
            min_value=100, 
            max_value=5000, 
            value=self.config_manager.get_sample_size(),
            step=100
        )
        
        anomaly_ratio = st.sidebar.slider(
            "Anomaly Ratio", 
            min_value=0.01, 
            max_value=0.5, 
            value=self.config_manager.get_anomaly_ratio(),
            step=0.01,
            format="%.2f"
        )
        
        # Model selection
        st.sidebar.subheader("🤖 Model Selection")
        
        model_method = st.sidebar.selectbox(
            "Detection Method",
            ["isolation_forest", "autoencoder", "lstm", "ensemble"],
            index=["isolation_forest", "autoencoder", "lstm", "ensemble"].index(
                self.config_manager.get_model_method()
            )
        )
        
        contamination = st.sidebar.slider(
            "Contamination Rate",
            min_value=0.01,
            max_value=0.3,
            value=self.config_manager.get_contamination(),
            step=0.01,
            format="%.2f"
        )
        
        # Action buttons
        st.sidebar.subheader("🚀 Actions")
        
        if st.sidebar.button("🔄 Generate New Data", type="primary"):
            self._generate_data(n_samples, anomaly_ratio)
        
        if st.sidebar.button("🎯 Run Detection"):
            if st.session_state.data is not None:
                self._run_detection(model_method, contamination)
            else:
                st.sidebar.error("Please generate data first!")
        
        if st.sidebar.button("📈 Evaluate Performance"):
            if st.session_state.predictions is not None:
                self._evaluate_performance()
            else:
                st.sidebar.error("Please run detection first!")
        
        # Real-time simulation
        st.sidebar.subheader("⏱️ Real-time Simulation")
        
        if st.sidebar.checkbox("Enable Real-time Updates"):
            if st.sidebar.button("▶️ Start Simulation"):
                self._start_real_time_simulation()
    
    def _generate_data(self, n_samples: int, anomaly_ratio: float):
        """Generate new synthetic data."""
        with st.spinner("Generating synthetic security log data..."):
            try:
                self.data_generator = SecurityLogGenerator(random_state=42)
                st.session_state.data = self.data_generator.generate_logs(
                    n_samples=n_samples, 
                    anomaly_ratio=anomaly_ratio
                )
                
                st.success(f"✅ Generated {n_samples} samples with {anomaly_ratio:.1%} anomalies")
                
            except Exception as e:
                st.error(f"❌ Error generating data: {e}")
    
    def _run_detection(self, model_method: str, contamination: float):
        """Run anomaly detection."""
        with st.spinner(f"Running {model_method} anomaly detection..."):
            try:
                # Update configuration
                self.config_manager.set('model.method', model_method)
                self.config_manager.set(f'model.{model_method}.contamination', contamination)
                
                # Initialize detector
                self.detector = AnomalyDetector(
                    method=model_method,
                    random_state=42,
                    contamination=contamination
                )
                
                # Run detection
                st.session_state.predictions = self.detector.fit_predict(st.session_state.data)
                st.session_state.scores = self.detector.decision_function(st.session_state.data)
                
                st.success(f"✅ Detection completed using {model_method}")
                
            except Exception as e:
                st.error(f"❌ Error running detection: {e}")
    
    def _evaluate_performance(self):
        """Evaluate model performance."""
        with st.spinner("Evaluating model performance..."):
            try:
                self.metrics = SecurityMetrics()
                st.session_state.metrics_results = self.metrics.compute_all(
                    st.session_state.data['labels'],
                    st.session_state.predictions,
                    st.session_state.scores
                )
                
                st.success("✅ Performance evaluation completed")
                
            except Exception as e:
                st.error(f"❌ Error evaluating performance: {e}")
    
    def _render_overview_tab(self):
        """Render the overview tab."""
        st.header("📊 System Overview")
        
        if st.session_state.data is None:
            st.info("👆 Please generate data using the sidebar controls to get started.")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Samples",
                len(st.session_state.data['data']),
                delta=None
            )
        
        with col2:
            anomalies = st.session_state.data['labels'].sum()
            st.metric(
                "True Anomalies",
                anomalies,
                delta=f"{anomalies/len(st.session_state.data['data']):.1%}"
            )
        
        with col3:
            if st.session_state.predictions is not None:
                detected = st.session_state.predictions.sum()
                st.metric(
                    "Detected Anomalies",
                    detected,
                    delta=f"{detected/len(st.session_state.data['data']):.1%}"
                )
            else:
                st.metric("Detected Anomalies", "N/A")
        
        with col4:
            if st.session_state.metrics_results is not None:
                accuracy = st.session_state.metrics_results.get('accuracy', 0)
                st.metric(
                    "Accuracy",
                    f"{accuracy:.3f}",
                    delta=f"{accuracy*100:.1f}%"
                )
            else:
                st.metric("Accuracy", "N/A")
        
        # Data summary
        st.subheader("📋 Data Summary")
        
        df = st.session_state.data['data']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Feature Statistics:**")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        with col2:
            st.write("**Anomaly Distribution:**")
            anomaly_counts = df['anomaly_label'].value_counts()
            
            fig = px.pie(
                values=anomaly_counts.values,
                names=['Normal', 'Anomaly'],
                title="Data Distribution",
                color_discrete_map={'Normal': '#2E8B57', 'Anomaly': '#DC143C'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Real-time status
        if st.session_state.predictions is not None:
            st.subheader("🚨 Current Status")
            
            # Calculate current metrics
            recent_data = df.tail(100)  # Last 100 samples
            recent_predictions = st.session_state.predictions[-100:]
            recent_labels = st.session_state.data['labels'][-100:]
            
            recent_anomalies = recent_predictions.sum()
            recent_true_anomalies = recent_labels.sum()
            
            if recent_anomalies > 0:
                st.markdown(f"""
                <div class="anomaly-alert">
                    <h4>⚠️ Recent Activity Alert</h4>
                    <p><strong>{recent_anomalies}</strong> anomalies detected in the last 100 samples</p>
                    <p>True anomaly rate: <strong>{recent_true_anomalies/100:.1%}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="normal-status">
                    <h4>✅ System Status Normal</h4>
                    <p>No anomalies detected in recent activity</p>
                </div>
                """, unsafe_allow_html=True)
    
    def _render_data_explorer_tab(self):
        """Render the data explorer tab."""
        st.header("🔍 Data Explorer")
        
        if st.session_state.data is None:
            st.info("👆 Please generate data first.")
            return
        
        df = st.session_state.data['data']
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_anomalies_only = st.checkbox("Show anomalies only")
        
        with col2:
            feature_filter = st.selectbox(
                "Filter by feature",
                ["All"] + list(df.columns)
            )
        
        with col3:
            if st.session_state.predictions is not None:
                show_predictions = st.checkbox("Show predictions")
            else:
                show_predictions = False
        
        # Apply filters
        filtered_df = df.copy()
        
        if show_anomalies_only:
            filtered_df = filtered_df[filtered_df['anomaly_label'] == 1]
        
        if feature_filter != "All":
            filtered_df = filtered_df[[feature_filter, 'anomaly_label']]
        
        # Display data
        st.subheader("📊 Data Table")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Interactive plots
        st.subheader("📈 Interactive Visualizations")
        
        plot_type = st.selectbox(
            "Select plot type",
            ["CPU vs Memory", "Feature Distributions", "Time Series", "Network Activity"]
        )
        
        if plot_type == "CPU vs Memory":
            self._plot_cpu_memory_interactive(df)
        
        elif plot_type == "Feature Distributions":
            self._plot_feature_distributions_interactive(df)
        
        elif plot_type == "Time Series":
            self._plot_time_series_interactive(df)
        
        elif plot_type == "Network Activity":
            self._plot_network_activity_interactive(df)
    
    def _plot_cpu_memory_interactive(self, df):
        """Plot interactive CPU vs Memory scatter plot."""
        fig = px.scatter(
            df,
            x='cpu_usage',
            y='memory_usage',
            color='anomaly_label',
            size='resource_pressure',
            hover_data=['timestamp', 'user_id', 'source_ip'],
            title="CPU vs Memory Usage",
            color_discrete_map={0: '#2E8B57', 1: '#DC143C'},
            labels={'anomaly_label': 'Anomaly Status'}
        )
        
        fig.update_layout(
            xaxis_title="CPU Usage (%)",
            yaxis_title="Memory Usage (%)",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_feature_distributions_interactive(self, df):
        """Plot interactive feature distributions."""
        feature = st.selectbox(
            "Select feature for distribution",
            ['cpu_usage', 'memory_usage', 'disk_usage', 'network_activity', 'resource_pressure']
        )
        
        fig = px.histogram(
            df,
            x=feature,
            color='anomaly_label',
            nbins=50,
            title=f"Distribution of {feature}",
            color_discrete_map={0: '#2E8B57', 1: '#DC143C'},
            labels={'anomaly_label': 'Anomaly Status'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_time_series_interactive(self, df):
        """Plot interactive time series."""
        if 'timestamp' not in df.columns:
            st.warning("No timestamp data available for time series plot.")
            return
        
        metric = st.selectbox(
            "Select metric for time series",
            ['cpu_usage', 'memory_usage', 'network_activity', 'resource_pressure']
        )
        
        fig = px.line(
            df,
            x='timestamp',
            y=metric,
            color='anomaly_label',
            title=f"{metric} Over Time",
            color_discrete_map={0: '#2E8B57', 1: '#DC143C'},
            labels={'anomaly_label': 'Anomaly Status'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_network_activity_interactive(self, df):
        """Plot interactive network activity."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Bytes Sent', 'Bytes Received', 'Active Connections', 'Packet Size Variance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Bytes sent
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bytes_sent'], mode='markers',
                      marker=dict(color=df['anomaly_label'], colorscale='RdYlGn_r'),
                      name='Bytes Sent'),
            row=1, col=1
        )
        
        # Bytes received
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bytes_received'], mode='markers',
                      marker=dict(color=df['anomaly_label'], colorscale='RdYlGn_r'),
                      name='Bytes Received'),
            row=1, col=2
        )
        
        # Active connections
        fig.add_trace(
            go.Scatter(x=df.index, y=df['active_connections'], mode='markers',
                      marker=dict(color=df['anomaly_label'], colorscale='RdYlGn_r'),
                      name='Active Connections'),
            row=2, col=1
        )
        
        # Packet size variance
        fig.add_trace(
            go.Scatter(x=df.index, y=df['packet_size_variance'], mode='markers',
                      marker=dict(color=df['anomaly_label'], colorscale='RdYlGn_r'),
                      name='Packet Size Variance'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title="Network Activity Analysis")
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_model_performance_tab(self):
        """Render the model performance tab."""
        st.header("🤖 Model Performance")
        
        if st.session_state.metrics_results is None:
            st.info("👆 Please run detection and evaluation first.")
            return
        
        metrics = st.session_state.metrics_results
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Classification Metrics")
            
            metric_data = {
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity'],
                'Value': [
                    metrics.get('accuracy', 0),
                    metrics.get('precision', 0),
                    metrics.get('recall', 0),
                    metrics.get('f1_score', 0),
                    metrics.get('specificity', 0)
                ]
            }
            
            metric_df = pd.DataFrame(metric_data)
            st.dataframe(metric_df, use_container_width=True)
        
        with col2:
            st.subheader("🚨 Security Metrics")
            
            security_data = {
                'Metric': ['Detection Rate', 'False Alarm Rate', 'Alert Efficiency', 'Miss Rate'],
                'Value': [
                    metrics.get('detection_rate', 0),
                    metrics.get('false_alarm_rate', 0),
                    metrics.get('alert_efficiency', 0),
                    metrics.get('miss_rate', 0)
                ]
            }
            
            security_df = pd.DataFrame(security_data)
            st.dataframe(security_df, use_container_width=True)
        
        # ROC and PR curves
        if st.session_state.scores is not None:
            st.subheader("📈 Performance Curves")
            
            from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(st.session_state.data['labels'], st.session_state.scores)
            roc_auc = roc_auc_score(st.session_state.data['labels'], st.session_state.scores)
            
            # PR Curve
            precision, recall, _ = precision_recall_curve(st.session_state.data['labels'], st.session_state.scores)
            pr_auc = average_precision_score(st.session_state.data['labels'], st.session_state.scores)
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(f'ROC Curve (AUC = {roc_auc:.3f})', f'PR Curve (AUC = {pr_auc:.3f})')
            )
            
            # ROC curve
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve',
                          line=dict(color='darkorange', width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                          line=dict(color='navy', width=2, dash='dash')),
                row=1, col=1
            )
            
            # PR curve
            fig.add_trace(
                go.Scatter(x=recall, y=precision, mode='lines', name='PR Curve',
                          line=dict(color='darkorange', width=2)),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=True)
            fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
            fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
            fig.update_xaxes(title_text="Recall", row=1, col=2)
            fig.update_yaxes(title_text="Precision", row=1, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion matrix
        st.subheader("🔢 Confusion Matrix")
        
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(st.session_state.data['labels'], st.session_state.predictions)
        
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
            x=['Normal', 'Anomaly'],
            y=['Normal', 'Anomaly']
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Operational metrics
        st.subheader("⚙️ Operational Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Alerts", metrics.get('total_alerts', 0))
        
        with col2:
            st.metric("Alerts per Day", f"{metrics.get('alerts_per_day', 0):.1f}")
        
        with col3:
            st.metric("Investigation Hours", f"{metrics.get('investigation_hours', 0):.1f}")
        
        with col4:
            st.metric("Workload Efficiency", f"{metrics.get('workload_efficiency', 0):.3f}")
    
    def _render_explainability_tab(self):
        """Render the explainability tab."""
        st.header("💡 Model Explainability")
        
        if st.session_state.predictions is None:
            st.info("👆 Please run detection first.")
            return
        
        # Find anomalies to explain
        anomaly_indices = np.where(st.session_state.predictions == 1)[0]
        
        if len(anomaly_indices) == 0:
            st.info("No anomalies detected to explain.")
            return
        
        # Select anomaly to explain
        selected_idx = st.selectbox(
            "Select anomaly to explain",
            anomaly_indices[:10],  # Limit to first 10
            format_func=lambda x: f"Anomaly {x} (Score: {st.session_state.scores[x]:.3f})"
        )
        
        # Generate explanation
        if st.button("🔍 Explain Anomaly"):
            with st.spinner("Generating explanation..."):
                try:
                    # Initialize explainer
                    feature_names = st.session_state.data['features']
                    X = st.session_state.data['data'][feature_names].values
                    
                    self.explainer = AnomalyExplainer(self.detector.detector, feature_names)
                    self.explainer.fit_explainer(X, sample_size=100)
                    
                    # Get explanation
                    explanations = self.explainer.explain_anomalies(X, [selected_idx])
                    
                    if selected_idx in explanations:
                        explanation = explanations[selected_idx]
                        
                        st.subheader(f"🔍 Explanation for Anomaly {selected_idx}")
                        
                        # Show top contributing features
                        st.write("**Top Contributing Features:**")
                        
                        for i, feature_info in enumerate(explanation['top_features'], 1):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write(f"**{i}. {feature_info['feature']}**")
                            
                            with col2:
                                if explanation['explanation_method'] == 'SHAP':
                                    st.write(f"Contribution: {feature_info['contribution']:.4f}")
                                else:
                                    st.write(f"Value: {feature_info['value']:.4f}")
                            
                            with col3:
                                if explanation['explanation_method'] == 'SHAP':
                                    st.write(f"Importance: {feature_info['importance']:.4f}")
                                else:
                                    st.write(f"Z-Score: {feature_info['z_score']:.4f}")
                        
                        # Show feature values comparison
                        st.subheader("📊 Feature Value Analysis")
                        
                        instance_data = st.session_state.data['data'].iloc[selected_idx]
                        
                        # Create comparison plot
                        feature_values = []
                        feature_names_list = []
                        colors = []
                        
                        for feature_info in explanation['top_features'][:5]:  # Top 5 features
                            feature_name = feature_info['feature']
                            feature_values.append(instance_data[feature_name])
                            feature_names_list.append(feature_name)
                            
                            if explanation['explanation_method'] == 'SHAP':
                                colors.append('red' if feature_info['contribution'] < 0 else 'blue')
                            else:
                                colors.append('red' if feature_info['z_score'] > 2 else 'blue')
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=feature_names_list,
                                y=feature_values,
                                marker_color=colors,
                                text=feature_values,
                                textposition='auto'
                            )
                        ])
                        
                        fig.update_layout(
                            title=f"Feature Values for Anomaly {selected_idx}",
                            xaxis_title="Features",
                            yaxis_title="Values",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show detailed report
                        st.subheader("📋 Detailed Report")
                        report = self.explainer.generate_explanation_report(explanations)
                        st.text(report)
                
                except Exception as e:
                    st.error(f"❌ Error generating explanation: {e}")
    
    def _render_configuration_tab(self):
        """Render the configuration tab."""
        st.header("⚙️ Configuration")
        
        # Current configuration
        st.subheader("📋 Current Configuration")
        
        config_dict = self.config_manager.to_dict()
        
        # Display configuration sections
        sections = ['data', 'model', 'evaluation', 'visualization', 'explainability', 'security']
        
        for section in sections:
            if section in config_dict:
                with st.expander(f"🔧 {section.title()} Configuration"):
                    st.json(config_dict[section])
        
        # Configuration editor
        st.subheader("✏️ Configuration Editor")
        
        st.info("Configuration changes will take effect on the next data generation or model run.")
        
        # Model parameters
        st.write("**Model Parameters:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_method = st.selectbox(
                "Detection Method",
                ["isolation_forest", "autoencoder", "lstm", "ensemble"],
                index=["isolation_forest", "autoencoder", "lstm", "ensemble"].index(
                    self.config_manager.get_model_method()
                )
            )
        
        with col2:
            new_contamination = st.slider(
                "Contamination Rate",
                min_value=0.01,
                max_value=0.3,
                value=self.config_manager.get_contamination(),
                step=0.01,
                format="%.2f"
            )
        
        # Data parameters
        st.write("**Data Parameters:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_samples = st.number_input(
                "Number of Samples",
                min_value=100,
                max_value=5000,
                value=self.config_manager.get_sample_size(),
                step=100
            )
        
        with col2:
            new_anomaly_ratio = st.slider(
                "Anomaly Ratio",
                min_value=0.01,
                max_value=0.5,
                value=self.config_manager.get_anomaly_ratio(),
                step=0.01,
                format="%.2f"
            )
        
        # Apply configuration
        if st.button("💾 Apply Configuration"):
            try:
                self.config_manager.set('model.method', new_method)
                self.config_manager.set(f'model.{new_method}.contamination', new_contamination)
                self.config_manager.set('data.n_samples', new_samples)
                self.config_manager.set('data.anomaly_ratio', new_anomaly_ratio)
                
                st.success("✅ Configuration updated successfully!")
                
            except Exception as e:
                st.error(f"❌ Error updating configuration: {e}")
        
        # Export/Import configuration
        st.subheader("📤 Export/Import Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Configuration"):
                config_json = self.config_manager.to_dict()
                st.download_button(
                    label="Download Configuration",
                    data=str(config_json),
                    file_name="anomaly_detection_config.json",
                    mime="application/json"
                )
        
        with col2:
            uploaded_file = st.file_uploader(
                "Upload Configuration",
                type=['json', 'yaml'],
                help="Upload a configuration file to load settings"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.json'):
                        import json
                        config_data = json.load(uploaded_file)
                    else:
                        import yaml
                        config_data = yaml.safe_load(uploaded_file)
                    
                    self.config_manager.apply_overrides(config_data)
                    st.success("✅ Configuration loaded successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error loading configuration: {e}")
    
    def _start_real_time_simulation(self):
        """Start real-time simulation."""
        st.info("🔄 Starting real-time simulation...")
        
        # Create a placeholder for real-time updates
        placeholder = st.empty()
        
        for i in range(10):  # Simulate 10 updates
            with placeholder.container():
                st.write(f"🔄 Simulation step {i+1}/10")
                
                # Simulate new data
                new_data = self.data_generator.generate_logs(n_samples=10, anomaly_ratio=0.1)
                
                # Update session state
                if st.session_state.data is not None:
                    # Append new data
                    st.session_state.data['data'] = pd.concat([
                        st.session_state.data['data'], 
                        new_data['data']
                    ], ignore_index=True)
                    
                    st.session_state.data['labels'] = np.concatenate([
                        st.session_state.data['labels'],
                        new_data['labels']
                    ])
                
                # Show current status
                if st.session_state.predictions is not None:
                    recent_predictions = st.session_state.predictions[-10:]
                    recent_anomalies = recent_predictions.sum()
                    
                    if recent_anomalies > 0:
                        st.warning(f"⚠️ {recent_anomalies} anomalies detected in recent activity")
                    else:
                        st.success("✅ No anomalies in recent activity")
                
                time.sleep(1)  # Simulate real-time delay
        
        st.success("✅ Real-time simulation completed!")


def main():
    """Main function to run the demo."""
    demo = SecurityAnomalyDemo()
    demo.run()


if __name__ == "__main__":
    main()
