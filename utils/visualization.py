"""
Visualization Utilities for Real-Time Fraud Detection Dashboard
Provides charts, graphs, and visual components for the fraud detection system
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

class FraudVisualization:
    """Main class for fraud detection visualizations"""
    
    def __init__(self):
        self.color_scheme = {
            'fraud': '#dc3545',
            'legitimate': '#28a745',
            'warning': '#ffc107',
            'info': '#17a2b8',
            'primary': '#007bff',
            'secondary': '#6c757d'
        }
        
        self.risk_colors = {
            'low': '#28a745',
            'medium': '#ffc107', 
            'high': '#fd7e14',
            'critical': '#dc3545'
        }
    
    def create_fraud_overview_chart(self, stats: Dict) -> go.Figure:
        """Create overview chart showing fraud detection statistics"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Total Transactions', 'Fraud Rate', 'Processing Performance', 'Alert Distribution'],
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Total transactions indicator
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=stats.get('total_processed', 0),
                title={"text": "Total Processed"},
                number={'font': {'size': 30}},
                domain={'x': [0, 0.5], 'y': [0.5, 1]}
            ),
            row=1, col=1
        )
        
        # Fraud rate indicator
        fraud_rate = stats.get('fraud_rate_percent', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=fraud_rate,
                domain={'x': [0.5, 1], 'y': [0.5, 1]},
                title={'text': "Fraud Rate (%)"},
                gauge={
                    'axis': {'range': [None, 10]},
                    'bar': {'color': self.color_scheme['fraud'] if fraud_rate > 5 else self.color_scheme['legitimate']},
                    'steps': [
                        {'range': [0, 2], 'color': "lightgray"},
                        {'range': [2, 5], 'color': "yellow"},
                        {'range': [5, 10], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 5
                    }
                }
            ),
            row=1, col=2
        )
        
        # Processing performance
        processing_time = stats.get('avg_processing_time_ms', 0)
        throughput = stats.get('throughput_per_second', 0)
        
        fig.add_trace(
            go.Bar(
                x=['Avg Processing Time (ms)', 'Throughput (tx/sec)'],
                y=[processing_time, throughput],
                marker_color=[self.color_scheme['info'], self.color_scheme['primary']],
                text=[f"{processing_time:.2f}ms", f"{throughput:.1f}"],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # Alert distribution
        alert_stats = stats.get('alerts_by_risk_level', {})
        if alert_stats:
            labels = list(alert_stats.keys())
            values = list(alert_stats.values())
            colors = [self.risk_colors.get(risk, self.color_scheme['secondary']) for risk in labels]
            
            fig.add_trace(
                go.Pie(
                    labels=labels,
                    values=values,
                    marker_colors=colors,
                    hole=0.3
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Fraud Detection System Overview",
            title_x=0.5
        )
        
        return fig
    
    def create_transaction_timeline(self, transactions: List[Dict], window_hours: int = 24) -> go.Figure:
        """Create timeline chart of transactions and fraud detection"""
        if not transactions:
            return self._create_empty_chart("No transaction data available")
        
        df = pd.DataFrame(transactions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter to time window
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        df = df[df['timestamp'] >= cutoff_time]
        
        if df.empty:
            return self._create_empty_chart("No transactions in the selected time window")
        
        # Group by time intervals
        df['time_bin'] = df['timestamp'].dt.floor('H')
        timeline_data = df.groupby(['time_bin', 'is_fraud']).size().unstack(fill_value=0)
        
        # Create traces
        fig = go.Figure()
        
        if 0 in timeline_data.columns:
            fig.add_trace(go.Scatter(
                x=timeline_data.index,
                y=timeline_data[0],
                mode='lines+markers',
                name='Legitimate',
                line=dict(color=self.color_scheme['legitimate'], width=3),
                marker=dict(size=8)
            ))
        
        if 1 in timeline_data.columns:
            fig.add_trace(go.Scatter(
                x=timeline_data.index,
                y=timeline_data[1],
                mode='lines+markers',
                name='Fraud',
                line=dict(color=self.color_scheme['fraud'], width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=f"Transaction Timeline - Last {window_hours} Hours",
            xaxis_title="Time",
            yaxis_title="Number of Transactions",
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def create_amount_distribution(self, transactions: List[Dict]) -> go.Figure:
        """Create amount distribution chart"""
        if not transactions:
            return self._create_empty_chart("No transaction data available")
        
        df = pd.DataFrame(transactions)
        
        fig = go.Figure()
        
        # Legitimate transactions
        legitimate_amounts = df[df['is_fraud'] == 0]['amount']
        if not legitimate_amounts.empty:
            fig.add_trace(go.Histogram(
                x=legitimate_amounts,
                name='Legitimate',
                opacity=0.7,
                marker_color=self.color_scheme['legitimate'],
                nbinsx=50
            ))
        
        # Fraud transactions
        fraud_amounts = df[df['is_fraud'] == 1]['amount']
        if not fraud_amounts.empty:
            fig.add_trace(go.Histogram(
                x=fraud_amounts,
                name='Fraud',
                opacity=0.7,
                marker_color=self.color_scheme['fraud'],
                nbinsx=50
            ))
        
        fig.update_layout(
            title="Transaction Amount Distribution",
            xaxis_title="Amount ($)",
            yaxis_title="Frequency",
            barmode='overlay',
            height=400
        )
        
        return fig
    
    def create_risk_heatmap(self, transactions: List[Dict]) -> go.Figure:
        """Create risk level heatmap"""
        if not transactions:
            return self._create_empty_chart("No transaction data available")
        
        df = pd.DataFrame(transactions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create time and risk level bins
        df['hour'] = df['timestamp'].dt.hour
        df['risk_level'] = df.get('predicted_fraud', df['is_fraud'])
        
        # Create pivot table
        heatmap_data = df.groupby(['hour', 'risk_level']).size().unstack(fill_value=0)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlGn_r',
            hoverongaps=False,
            text=heatmap_data.values,
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Risk Level Heatmap by Hour",
            xaxis_title="Risk Level",
            yaxis_title="Hour of Day",
            height=400
        )
        
        return fig
    
    def create_user_behavior_chart(self, user_data: Dict) -> go.Figure:
        """Create user behavior analysis chart"""
        if not user_data:
            return self._create_empty_chart("No user behavior data available")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Transaction Frequency', 'Amount Patterns'],
            specs=[[{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Transaction frequency
        users = list(user_data.keys())[:10]  # Top 10 users
        frequencies = [user_data[user].get('transaction_count', 0) for user in users]
        
        fig.add_trace(
            go.Bar(
                x=users,
                y=frequencies,
                name='Transaction Count',
                marker_color=self.color_scheme['primary']
            ),
            row=1, col=1
        )
        
        # Amount patterns
        amounts = []
        for user in users:
            user_amounts = user_data[user].get('amounts', [])
            if user_amounts:
                amounts.extend(user_amounts)
        
        if amounts:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(amounts))),
                    y=amounts,
                    mode='markers',
                    name='Transaction Amounts',
                    marker=dict(color=self.color_scheme['info'], size=6)
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            height=400,
            title_text="User Behavior Analysis",
            showlegend=False
        )
        
        return fig
    
    def create_model_performance_chart(self, model_stats: Dict) -> go.Figure:
        """Create model performance comparison chart"""
        if not model_stats:
            return self._create_empty_chart("No model performance data available")
        
        models = list(model_stats.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [model_stats[model].get(metric, 0) for model in models]
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=models,
                y=values,
                text=[f"{v:.3f}" for v in values],
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode='group',
            height=500
        )
        
        return fig
    
    def create_real_time_dashboard(self, data: Dict) -> None:
        """Create real-time dashboard using Streamlit"""
        st.set_page_config(
            page_title="Real-Time Fraud Detection Dashboard",
            page_icon="⚠️",
            layout="wide"
        )
        
        # Header
        st.title("[ALERT] Real-Time Fraud Detection Dashboard")
        st.markdown("---")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Transactions",
                value=data.get('total_processed', 0),
                delta=f"+{data.get('recent_transactions', 0)}"
            )
        
        with col2:
            st.metric(
                label="Fraud Detected",
                value=data.get('fraud_detected', 0),
                delta=f"{data.get('fraud_rate_percent', 0):.2f}%"
            )
        
        with col3:
            st.metric(
                label="Avg Processing Time",
                value=f"{data.get('avg_processing_time_ms', 0):.2f}ms",
                delta=f"{data.get('throughput_per_second', 0):.1f} tx/sec"
            )
        
        with col4:
            st.metric(
                label="Active Alerts",
                value=data.get('recent_alerts_count', 0),
                delta=f"+{data.get('alerts_last_hour', 0)}"
            )
        
        st.markdown("---")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Timeline")
            timeline_fig = self.create_transaction_timeline(data.get('recent_transactions', []))
            st.plotly_chart(timeline_fig, width='stretch')
        
        with col2:
            st.subheader("Amount Distribution")
            amount_fig = self.create_amount_distribution(data.get('recent_transactions', []))
            st.plotly_chart(amount_fig, width='stretch')
        
        # Overview chart
        st.subheader("System Overview")
        overview_fig = self.create_fraud_overview_chart(data)
        st.plotly_chart(overview_fig, width='stretch')
        
        # Recent alerts
        if data.get('recent_alerts'):
            st.subheader("Recent Fraud Alerts")
            alerts_df = pd.DataFrame([alert.to_dict() for alert in data['recent_alerts']])
            
            # Color code by risk level
            def color_risk_level(val):
                colors = {
                    'low': 'background-color: #28a745; color: white',
                    'medium': 'background-color: #ffc107; color: black',
                    'high': 'background-color: #fd7e14; color: white',
                    'critical': 'background-color: #dc3545; color: white'
                }
                return colors.get(val, '')
            
            styled_df = alerts_df.style.applymap(
                color_risk_level, subset=['risk_level']
            )
            
            st.dataframe(styled_df, width='stretch')
    
    def create_alert_notification(self, alert: Dict) -> str:
        """Create HTML notification for fraud alert"""
        risk_color = self.risk_colors.get(alert.get('risk_level', 'medium'), self.color_scheme['warning'])
        
        html = f"""
        <div style="
            border-left: 5px solid {risk_color};
            padding: 15px;
            margin: 10px 0;
            background-color: #f8f9fa;
            border-radius: 5px;
        ">
            <h4 style="color: {risk_color}; margin: 0 0 10px 0;">
                [ALERT] Fraud Alert - {alert.get('risk_level', 'unknown').upper()}
            </h4>
            <p><strong>Transaction ID:</strong> {alert.get('transaction_id', 'N/A')}</p>
            <p><strong>User ID:</strong> {alert.get('user_id', 'N/A')}</p>
            <p><strong>Amount:</strong> ${alert.get('amount', 0):,.2f}</p>
            <p><strong>Confidence:</strong> {alert.get('confidence_score', 0):.3f}</p>
            <p><strong>Time:</strong> {alert.get('timestamp', 'N/A')}</p>
            <p><strong>Risk Factors:</strong></p>
            <ul>
                {''.join([f'<li>{factor}</li>' for factor in alert.get('risk_factors', [])])}
            </ul>
        </div>
        """
        return html
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis={'visible': False},
            yaxis={'visible': False},
            height=300
        )
        return fig
    
    def export_chart_data(self, fig: go.Figure, format: str = 'json') -> str:
        """Export chart data in specified format"""
        if format == 'json':
            return json.dumps(fig.to_dict(), indent=2)
        elif format == 'csv':
            # Extract data from figure and convert to CSV
            data = []
            for trace in fig.data:
                if hasattr(trace, 'x') and hasattr(trace, 'y'):
                    for x, y in zip(trace.x, trace.y):
                        data.append({'x': x, 'y': y, 'name': trace.name})
            return pd.DataFrame(data).to_csv(index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

# Example usage and testing
def test_visualization():
    """Test visualization functions"""
    print("Testing Visualization Functions...")
    
    viz = FraudVisualization()
    
    # Sample data
    sample_stats = {
        'total_processed': 1500,
        'fraud_detected': 45,
        'fraud_rate_percent': 3.0,
        'avg_processing_time_ms': 25.5,
        'throughput_per_second': 39.2,
        'alerts_by_risk_level': {
            'low': 10,
            'medium': 20,
            'high': 12,
            'critical': 3
        }
    }
    
    sample_transactions = [
        {
            'transaction_id': f'tx_{i}',
            'user_id': f'user_{i % 10}',
            'amount': np.random.lognormal(3, 1.5),
            'timestamp': datetime.now() - timedelta(minutes=i*5),
            'is_fraud': np.random.choice([0, 1], p=[0.95, 0.05])
        }
        for i in range(100)
    ]
    
    # Test overview chart
    overview_fig = viz.create_fraud_overview_chart(sample_stats)
    print("[SUCCESS] Overview chart created")
    
    # Test timeline chart
    timeline_fig = viz.create_transaction_timeline(sample_transactions)
    print("[SUCCESS] Timeline chart created")
    
    # Test amount distribution
    amount_fig = viz.create_amount_distribution(sample_transactions)
    print("[SUCCESS] Amount distribution chart created")
    
    print("🎉 Visualization testing completed!")

if __name__ == "__main__":
    test_visualization()
