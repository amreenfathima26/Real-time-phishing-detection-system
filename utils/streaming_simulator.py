"""
Real-Time Data Streaming Simulator for Fraud Detection
Simulates live transaction data stream for testing and demonstration
"""

import pandas as pd
import numpy as np
import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, AsyncGenerator
import json

class TransactionStreamSimulator:
    """Simulates real-time transaction data stream"""
    
    def __init__(self, base_data_path=None, fraud_rate=0.05):
        self.base_data_path = base_data_path
        self.fraud_rate = fraud_rate
        self.transaction_id_counter = 1000000
        self.users = []
        self.merchants = []
        self.locations = []
        self.transaction_types = ['payment', 'transfer', 'withdrawal', 'deposit']
        self.is_running = False
        
        # Load or generate base data
        if base_data_path:
            self.load_base_data()
        else:
            self.generate_base_data()
    
    def load_base_data(self):
        """Load base data from file"""
        try:
            df = pd.read_csv(self.base_data_path)
            
            # Extract unique users, merchants, and locations
            self.users = df['user_id'].unique().tolist()
            self.merchants = df['merchant_id'].unique().tolist() if 'merchant_id' in df.columns else []
            self.locations = list(zip(df['latitude'].tolist(), df['longitude'].tolist()))
            
            print(f"[SUCCESS] Base data loaded: {len(self.users)} users, {len(self.merchants)} merchants")
        except Exception as e:
            print(f"Warning: Could not load base data: {e}")
            self.generate_base_data()
    
    def generate_base_data(self):
        """Generate synthetic base data"""
        print("Generating synthetic base data...")
        
        # Generate users
        self.users = [f"user_{i:06d}" for i in range(1, 1001)]
        
        # Generate merchants
        self.merchants = [f"merchant_{i:04d}" for i in range(1, 101)]
        
        # Generate locations (US coordinates)
        self.locations = [
            (40.7128, -74.0060),  # New York
            (34.0522, -118.2437),  # Los Angeles
            (41.8781, -87.6298),   # Chicago
            (29.7604, -95.3698),   # Houston
            (33.4484, -112.0740),  # Phoenix
            (39.7392, -104.9903),  # Denver
            (39.9526, -75.1652),   # Philadelphia
            (29.4241, -98.4936),   # San Antonio
            (32.7157, -117.1611),  # San Diego
            (32.7767, -96.7970),   # Dallas
        ]
        
        print(f"[SUCCESS] Synthetic data generated: {len(self.users)} users, {len(self.merchants)} merchants")
    
    def generate_transaction(self, timestamp=None) -> Dict:
        """Generate a single transaction"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Increment transaction ID
        self.transaction_id_counter += 1
        
        # Random selection
        user_id = random.choice(self.users)
        merchant_id = random.choice(self.merchants) if self.merchants else f"merchant_{random.randint(1, 100):04d}"
        transaction_type = random.choice(self.transaction_types)
        location = random.choice(self.locations)
        
        # Generate amount (log-normal distribution for realistic amounts)
        base_amount = np.random.lognormal(mean=3, sigma=1.5)
        amount = round(base_amount, 2)
        
        # Determine if fraud based on fraud rate
        is_fraud = random.random() < self.fraud_rate
        
        # Modify transaction for fraud cases
        if is_fraud:
            # Fraud patterns
            fraud_patterns = [
                lambda: amount * random.uniform(5, 20),  # Unusually high amount
                lambda: random.choice([0.01, 0.05, 0.1]),  # Unusually low amount
                lambda: amount  # Normal amount but other suspicious features
            ]
            
            amount = random.choice(fraud_patterns)()
            amount = round(amount, 2)
        
        # Create transaction
        transaction = {
            'transaction_id': f"tx_{self.transaction_id_counter}",
            'user_id': user_id,
            'merchant_id': merchant_id,
            'amount': amount,
            'timestamp': timestamp.isoformat(),
            'transaction_type': transaction_type,
            'latitude': location[0] + random.uniform(-0.1, 0.1),
            'longitude': location[1] + random.uniform(-0.1, 0.1),
            'is_fraud': int(is_fraud)
        }
        
        return transaction
    
    async def stream_transactions(self, interval=1.0, max_transactions=None) -> AsyncGenerator[Dict, None]:
        """
        Stream transactions asynchronously
        
        Args:
            interval: Time between transactions in seconds
            max_transactions: Maximum number of transactions to generate
        """
        self.is_running = True
        transaction_count = 0
        
        print(f"🚀 Starting transaction stream (interval: {interval}s)")
        
        while self.is_running:
            try:
                # Generate transaction
                transaction = self.generate_transaction()
                
                # Yield transaction
                yield transaction
                
                transaction_count += 1
                
                # Check if max transactions reached
                if max_transactions and transaction_count >= max_transactions:
                    break
                
                # Wait for next transaction
                await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"Error generating transaction: {e}")
                await asyncio.sleep(interval)
        
        print(f"🛑 Transaction stream stopped. Total transactions: {transaction_count}")
    
    def stop_stream(self):
        """Stop the transaction stream"""
        self.is_running = False
    
    def generate_batch(self, count=100) -> List[Dict]:
        """Generate a batch of transactions"""
        transactions = []
        
        for _ in range(count):
            transaction = self.generate_transaction()
            transactions.append(transaction)
        
        return transactions
    
    def save_stream_data(self, transactions: List[Dict], filepath: str):
        """Save stream data to CSV file"""
        df = pd.DataFrame(transactions)
        df.to_csv(filepath, index=False)
        print(f"[SUCCESS] Stream data saved to {filepath}")

class RealTimeDataProcessor:
    """Processes real-time transaction data for fraud detection"""
    
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
        self.processed_count = 0
        self.fraud_detected = 0
        self.processing_times = []
    
    async def process_transaction(self, transaction: Dict) -> Dict:
        """
        Process a single transaction for fraud detection
        
        Returns:
            Dict with original transaction + prediction results
        """
        start_time = time.time()
        
        try:
            # Preprocess transaction
            processed_features = self.preprocessor.preprocess_realtime_data(
                pd.DataFrame([transaction])
            )
            
            # Get feature columns
            feature_columns = self.preprocessor.get_feature_columns(processed_features)
            features = processed_features[feature_columns].values
            
            # Make prediction
            prediction, probability = self.model.predict_fraud(features)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Update counters
            self.processed_count += 1
            if prediction[0] == 1:
                self.fraud_detected += 1
            
            # Add prediction results to transaction
            result = transaction.copy()
            result['predicted_fraud'] = int(prediction[0])
            result['fraud_probability'] = float(probability[0])
            result['processing_time_ms'] = round(processing_time * 1000, 2)
            result['processing_timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            print(f"Error processing transaction: {e}")
            # Return original transaction with error flag
            result = transaction.copy()
            result['predicted_fraud'] = 0
            result['fraud_probability'] = 0.0
            result['processing_time_ms'] = 0.0
            result['error'] = str(e)
            return result
    
    async def process_stream(self, transaction_stream: AsyncGenerator) -> AsyncGenerator[Dict, None]:
        """
        Process a stream of transactions
        
        Args:
            transaction_stream: Async generator of transactions
            
        Yields:
            Processed transactions with fraud predictions
        """
        print("🔍 Starting fraud detection processing...")
        
        async for transaction in transaction_stream:
            processed_transaction = await self.process_transaction(transaction)
            yield processed_transaction
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        fraud_rate = (self.fraud_detected / self.processed_count * 100) if self.processed_count > 0 else 0
        
        return {
            'total_processed': self.processed_count,
            'fraud_detected': self.fraud_detected,
            'fraud_rate_percent': round(fraud_rate, 2),
            'avg_processing_time_ms': round(avg_processing_time * 1000, 2),
            'throughput_per_second': round(1 / avg_processing_time, 2) if avg_processing_time > 0 else 0
        }

class StreamDataGenerator:
    """Generates realistic streaming data for testing"""
    
    @staticmethod
    def create_historical_dataset(num_transactions=10000, fraud_rate=0.05, save_path="data/transactions.csv"):
        """Create a historical dataset for training"""
        print(f"Creating historical dataset with {num_transactions} transactions...")
        
        simulator = TransactionStreamSimulator(fraud_rate=fraud_rate)
        transactions = simulator.generate_batch(num_transactions)
        
        # Save to CSV
        df = pd.DataFrame(transactions)
        df.to_csv(save_path, index=False)
        
        print(f"[SUCCESS] Historical dataset created: {save_path}")
        print(f"  - Total transactions: {len(transactions)}")
        print(f"  - Fraud rate: {df['is_fraud'].mean() * 100:.2f}%")
        
        return df
    
    @staticmethod
    def create_streaming_dataset(num_transactions=1000, save_path="data/streaming_data.csv"):
        """Create a streaming dataset for real-time simulation"""
        print(f"Creating streaming dataset with {num_transactions} transactions...")
        
        simulator = TransactionStreamSimulator(fraud_rate=0.03)  # Lower fraud rate for streaming
        transactions = simulator.generate_batch(num_transactions)
        
        # Add timestamps with realistic intervals
        base_time = datetime.now() - timedelta(hours=1)
        for i, transaction in enumerate(transactions):
            # Random interval between transactions (1-30 seconds)
            interval = random.uniform(1, 30)
            transaction['timestamp'] = (base_time + timedelta(seconds=i * interval)).isoformat()
        
        # Save to CSV
        df = pd.DataFrame(transactions)
        df.to_csv(save_path, index=False)
        
        print(f"[SUCCESS] Streaming dataset created: {save_path}")
        print(f"  - Total transactions: {len(transactions)}")
        print(f"  - Time span: 1 hour")
        
        return df

# Example usage and testing
async def test_streaming():
    """Test the streaming functionality"""
    print("Testing transaction streaming...")
    
    # Create simulator
    simulator = TransactionStreamSimulator()
    
    # Stream some transactions
    transactions = []
    async for transaction in simulator.stream_transactions(interval=0.5, max_transactions=5):
        transactions.append(transaction)
        print(f"Generated: {transaction['transaction_id']} - ${transaction['amount']} - Fraud: {transaction['is_fraud']}")
    
    print(f"\nGenerated {len(transactions)} transactions")
    
    # Save test data
    simulator.save_stream_data(transactions, "data/test_stream.csv")

if __name__ == "__main__":
    # Create sample datasets
    print("Creating sample datasets...")
    
    # Historical data for training
    StreamDataGenerator.create_historical_dataset(
        num_transactions=5000,
        save_path="data/transactions.csv"
    )
    
    # Streaming data for real-time simulation
    StreamDataGenerator.create_streaming_dataset(
        num_transactions=500,
        save_path="data/streaming_data.csv"
    )
    
    print("\n🎉 Sample datasets created successfully!")
    
    # Test streaming
    print("\nTesting streaming functionality...")
    asyncio.run(test_streaming())
