"""
Real-Time Fraud Detection Engine
Core engine for processing transactions and detecting fraud in real-time
"""

import pandas as pd
import numpy as np
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass
from enum import Enum

class FraudRiskLevel(Enum):
    """Fraud risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class FraudAlert:
    """Fraud alert data structure"""
    transaction_id: str
    user_id: str
    amount: float
    risk_level: FraudRiskLevel
    confidence_score: float
    fraud_probability: float
    risk_factors: List[str]
    timestamp: datetime
    processing_time_ms: float

class FraudDetectionEngine:
    """Main fraud detection engine for real-time processing"""
    
    def __init__(self, model, preprocessor, config=None):
        self.model = model
        self.preprocessor = preprocessor
        self.config = config or self._default_config()
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'fraud_detected': 0,
            'false_positives': 0,
            'true_positives': 0,
            'processing_times': [],
            'alerts_generated': 0
        }
        
        # Real-time monitoring
        self.recent_transactions = []
        self.user_behavior_history = {}
        self.alert_history = []
        
        # Risk thresholds
        self.risk_thresholds = {
            FraudRiskLevel.LOW: 0.3,
            FraudRiskLevel.MEDIUM: 0.5,
            FraudRiskLevel.HIGH: 0.7,
            FraudRiskLevel.CRITICAL: 0.9
        }
    
    def _default_config(self):
        """Default configuration for fraud detection"""
        return {
            'enable_behavioral_analysis': True,
            'enable_velocity_checking': True,
            'enable_geolocation_checking': True,
            'enable_amount_anomaly_detection': True,
            'max_processing_time_ms': 100,
            'history_window_minutes': 60,
            'velocity_threshold_per_minute': 10,
            'amount_anomaly_threshold': 5.0,
            'geolocation_radius_km': 100
        }
    
    def analyze_transaction(self, transaction: Dict) -> Tuple[bool, float, List[str]]:
        """
        Analyze a single transaction for fraud indicators
        
        Returns:
            Tuple of (is_fraud, confidence_score, risk_factors)
        """
        risk_factors = []
        confidence_score = 0.0
        
        try:
            # ML Model Prediction
            processed_features = self.preprocessor.preprocess_realtime_data(
                pd.DataFrame([transaction])
            )
            
            feature_columns = self.preprocessor.get_feature_columns(processed_features)
            features = processed_features[feature_columns].values
            
            prediction, probability = self.model.predict_fraud(features)
            ml_confidence = float(probability[0])
            
            # Rule-based analysis
            rule_based_score, rule_factors = self._rule_based_analysis(transaction)
            risk_factors.extend(rule_factors)
            
            # Behavioral analysis
            if self.config['enable_behavioral_analysis']:
                behavioral_score, behavioral_factors = self._behavioral_analysis(transaction)
                risk_factors.extend(behavioral_factors)
                confidence_score += behavioral_score * 0.3
            
            # Combine ML and rule-based scores
            confidence_score = (ml_confidence * 0.7) + (rule_based_score * 0.3)
            
            # Determine fraud based on threshold
            is_fraud = confidence_score > self.config.get('fraud_threshold', 0.5)
            
            return is_fraud, confidence_score, risk_factors
            
        except Exception as e:
            print(f"Error analyzing transaction: {e}")
            return False, 0.0, [f"Analysis error: {str(e)}"]
    
    def _rule_based_analysis(self, transaction: Dict) -> Tuple[float, List[str]]:
        """Perform rule-based fraud analysis"""
        risk_score = 0.0
        risk_factors = []
        
        amount = transaction.get('amount', 0)
        transaction_type = transaction.get('transaction_type', '')
        user_id = transaction.get('user_id', '')
        
        # Amount-based rules
        if self.config['enable_amount_anomaly_detection']:
            if amount > 10000:
                risk_score += 0.3
                risk_factors.append("High transaction amount")
            elif amount < 0.01:
                risk_score += 0.2
                risk_factors.append("Unusually low amount")
        
        # Transaction type rules
        if transaction_type == 'withdrawal':
            risk_score += 0.1
            risk_factors.append("Withdrawal transaction")
        
        # Time-based rules
        timestamp = pd.to_datetime(transaction.get('timestamp', datetime.now()))
        hour = timestamp.hour
        
        if hour < 6 or hour > 22:
            risk_score += 0.15
            risk_factors.append("Unusual transaction time")
        
        return min(risk_score, 1.0), risk_factors
    
    def _behavioral_analysis(self, transaction: Dict) -> Tuple[float, List[str]]:
        """Analyze user behavior patterns"""
        risk_score = 0.0
        risk_factors = []
        
        user_id = transaction.get('user_id', '')
        amount = transaction.get('amount', 0)
        timestamp = pd.to_datetime(transaction.get('timestamp', datetime.now()))
        
        if user_id not in self.user_behavior_history:
            self.user_behavior_history[user_id] = []
        
        # Update user history
        self.user_behavior_history[user_id].append({
            'timestamp': timestamp,
            'amount': amount,
            'transaction_id': transaction.get('transaction_id', '')
        })
        
        # Keep only recent history
        cutoff_time = timestamp - timedelta(minutes=self.config['history_window_minutes'])
        self.user_behavior_history[user_id] = [
            tx for tx in self.user_behavior_history[user_id]
            if tx['timestamp'] > cutoff_time
        ]
        
        # Velocity checking
        if self.config['enable_velocity_checking']:
            recent_count = len(self.user_behavior_history[user_id])
            if recent_count > self.config['velocity_threshold_per_minute']:
                risk_score += 0.4
                risk_factors.append(f"High transaction velocity: {recent_count} transactions")
        
        # Amount pattern analysis
        if len(self.user_behavior_history[user_id]) > 1:
            amounts = [tx['amount'] for tx in self.user_behavior_history[user_id]]
            avg_amount = np.mean(amounts)
            
            if amount > avg_amount * self.config['amount_anomaly_threshold']:
                risk_score += 0.3
                risk_factors.append(f"Amount significantly higher than average: {amount:.2f} vs {avg_amount:.2f}")
        
        return min(risk_score, 1.0), risk_factors
    
    def _geolocation_analysis(self, transaction: Dict) -> Tuple[float, List[str]]:
        """Analyze geolocation patterns"""
        risk_score = 0.0
        risk_factors = []
        
        latitude = transaction.get('latitude')
        longitude = transaction.get('longitude')
        
        if latitude and longitude:
            # Simple geolocation analysis
            # In a real system, you'd check against user's usual locations
            user_id = transaction.get('user_id', '')
            
            # For demo purposes, flag transactions far from "home" (NYC)
            home_lat, home_lon = 40.7128, -74.0060
            
            # Calculate distance (simplified)
            distance = np.sqrt((latitude - home_lat)**2 + (longitude - home_lon)**2)
            
            if distance > 0.5:  # Roughly 50+ km
                risk_score += 0.2
                risk_factors.append(f"Transaction far from usual location: {distance:.3f} degrees")
        
        return min(risk_score, 1.0), risk_factors
    
    def determine_risk_level(self, confidence_score: float) -> FraudRiskLevel:
        """Determine fraud risk level based on confidence score"""
        if confidence_score >= self.risk_thresholds[FraudRiskLevel.CRITICAL]:
            return FraudRiskLevel.CRITICAL
        elif confidence_score >= self.risk_thresholds[FraudRiskLevel.HIGH]:
            return FraudRiskLevel.HIGH
        elif confidence_score >= self.risk_thresholds[FraudRiskLevel.MEDIUM]:
            return FraudRiskLevel.MEDIUM
        else:
            return FraudRiskLevel.LOW
    
    def process_transaction(self, transaction: Dict) -> Optional[FraudAlert]:
        """
        Process a transaction and return fraud alert if detected
        
        Returns:
            FraudAlert if fraud detected, None otherwise
        """
        start_time = time.time()
        
        try:
            # Analyze transaction
            is_fraud, confidence_score, risk_factors = self.analyze_transaction(transaction)
            
            # Update statistics
            self.stats['total_processed'] += 1
            processing_time = (time.time() - start_time) * 1000
            self.stats['processing_times'].append(processing_time)
            
            # Keep only recent processing times
            if len(self.stats['processing_times']) > 1000:
                self.stats['processing_times'] = self.stats['processing_times'][-1000:]
            
            # Store recent transaction
            self.recent_transactions.append({
                'transaction': transaction,
                'timestamp': datetime.now(),
                'is_fraud': is_fraud,
                'confidence_score': confidence_score
            })
            
            # Keep only recent transactions
            cutoff_time = datetime.now() - timedelta(minutes=self.config['history_window_minutes'])
            self.recent_transactions = [
                tx for tx in self.recent_transactions
                if tx['timestamp'] > cutoff_time
            ]
            
            # Generate alert if fraud detected
            if is_fraud:
                self.stats['fraud_detected'] += 1
                self.stats['alerts_generated'] += 1
                
                risk_level = self.determine_risk_level(confidence_score)
                
                alert = FraudAlert(
                    transaction_id=transaction.get('transaction_id', ''),
                    user_id=transaction.get('user_id', ''),
                    amount=transaction.get('amount', 0),
                    risk_level=risk_level,
                    confidence_score=confidence_score,
                    fraud_probability=confidence_score,
                    risk_factors=risk_factors,
                    timestamp=datetime.now(),
                    processing_time_ms=processing_time
                )
                
                # Store alert
                self.alert_history.append(alert)
                
                # Keep only recent alerts
                if len(self.alert_history) > 1000:
                    self.alert_history = self.alert_history[-1000:]
                
                return alert
            
            return None
            
        except Exception as e:
            print(f"Error processing transaction: {e}")
            return None
    
    async def process_transaction_async(self, transaction: Dict) -> Optional[FraudAlert]:
        """Async version of process_transaction"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_transaction, transaction)
    
    def get_statistics(self) -> Dict:
        """Get engine statistics"""
        avg_processing_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
        
        return {
            'total_processed': self.stats['total_processed'],
            'fraud_detected': self.stats['fraud_detected'],
            'alerts_generated': self.stats['alerts_generated'],
            'fraud_rate_percent': round(
                (self.stats['fraud_detected'] / max(self.stats['total_processed'], 1)) * 100, 2
            ),
            'avg_processing_time_ms': round(avg_processing_time, 2),
            'throughput_per_second': round(1000 / avg_processing_time, 2) if avg_processing_time > 0 else 0,
            'recent_transactions_count': len(self.recent_transactions),
            'active_users_count': len(self.user_behavior_history),
            'recent_alerts_count': len(self.alert_history)
        }
    
    def get_recent_alerts(self, limit=10) -> List[FraudAlert]:
        """Get recent fraud alerts"""
        return self.alert_history[-limit:] if self.alert_history else []
    
    def get_recent_transactions(self, limit=50) -> List[Dict]:
        """Get recent transactions"""
        return [
            tx['transaction'] for tx in self.recent_transactions[-limit:]
        ]
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.stats = {
            'total_processed': 0,
            'fraud_detected': 0,
            'false_positives': 0,
            'true_positives': 0,
            'processing_times': [],
            'alerts_generated': 0
        }
        self.recent_transactions = []
        self.user_behavior_history = {}
        self.alert_history = []
    
    def update_config(self, new_config: Dict):
        """Update engine configuration"""
        self.config.update(new_config)
        print(f"[SUCCESS] Configuration updated: {new_config}")

class BatchProcessor:
    """Process multiple transactions in batch mode"""
    
    def __init__(self, engine: FraudDetectionEngine):
        self.engine = engine
    
    def process_batch(self, transactions: List[Dict]) -> List[Optional[FraudAlert]]:
        """
        Process a batch of transactions
        
        Returns:
            List of FraudAlert objects (None for non-fraud transactions)
        """
        alerts = []
        
        for transaction in transactions:
            alert = self.engine.process_transaction(transaction)
            alerts.append(alert)
        
        return alerts
    
    async def process_batch_async(self, transactions: List[Dict]) -> List[Optional[FraudAlert]]:
        """Async version of batch processing"""
        tasks = [
            self.engine.process_transaction_async(transaction)
            for transaction in transactions
        ]
        
        alerts = await asyncio.gather(*tasks)
        return alerts

# Example usage and testing
def test_fraud_detection_engine():
    """Test the fraud detection engine"""
    print("Testing Fraud Detection Engine...")
    
    # This would normally use a trained model and preprocessor
    # For testing, we'll create mock objects
    
    class MockModel:
        def predict_fraud(self, features):
            # Mock prediction
            prediction = np.random.choice([0, 1], p=[0.8, 0.2])
            probability = np.random.uniform(0, 1)
            return np.array([prediction]), np.array([probability])
    
    class MockPreprocessor:
        def preprocess_realtime_data(self, df):
            # Mock preprocessing - just return the dataframe with some features
            df['feature_1'] = np.random.randn(len(df))
            df['feature_2'] = np.random.randn(len(df))
            return df
        
        def get_feature_columns(self, df):
            return ['feature_1', 'feature_2']
    
    # Create engine
    model = MockModel()
    preprocessor = MockPreprocessor()
    engine = FraudDetectionEngine(model, preprocessor)
    
    # Test transaction
    test_transaction = {
        'transaction_id': 'tx_001',
        'user_id': 'user_001',
        'amount': 1500.0,
        'timestamp': datetime.now().isoformat(),
        'transaction_type': 'payment',
        'latitude': 40.7128,
        'longitude': -74.0060
    }
    
    # Process transaction
    alert = engine.process_transaction(test_transaction)
    
    if alert:
        print(f"[ALERT] Fraud Alert Generated!")
        print(f"  Transaction ID: {alert.transaction_id}")
        print(f"  Risk Level: {alert.risk_level.value}")
        print(f"  Confidence: {alert.confidence_score:.3f}")
        print(f"  Risk Factors: {alert.risk_factors}")
    else:
        print("✅ Transaction approved")
    
    # Get statistics
    stats = engine.get_statistics()
    print(f"\nEngine Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_fraud_detection_engine()
