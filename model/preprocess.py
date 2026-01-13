"""
Data Preprocessing Module for Real-Time Fraud Detection
Handles feature engineering, data cleaning, and preprocessing for ML models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TransactionPreprocessor:
    """Handles preprocessing of transaction data for fraud detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
        
    def create_features(self, df):
        """
        Create engineered features from raw transaction data
        """
        df = df.copy()
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Transaction amount features
        if 'amount' in df.columns:
            df['amount_log'] = np.log1p(df['amount'])
            df['amount_squared'] = df['amount'] ** 2
            df['is_high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        
        # User behavior features (requires historical data)
        if 'user_id' in df.columns:
            user_stats = df.groupby('user_id').agg({
                'amount': ['mean', 'std', 'count'],
                'transaction_id': 'count'
            }).reset_index()
            user_stats.columns = ['user_id', 'user_avg_amount', 'user_std_amount', 'user_tx_count', 'user_total_tx']
            
            df = df.merge(user_stats, on='user_id', how='left')
            df['amount_vs_user_avg'] = df['amount'] / (df['user_avg_amount'] + 1e-6)
            df['user_tx_frequency'] = df['user_total_tx']
        
        # Merchant features
        if 'merchant_id' in df.columns:
            merchant_stats = df.groupby('merchant_id').agg({
                'amount': ['mean', 'std'],
                'transaction_id': 'count'
            }).reset_index()
            merchant_stats.columns = ['merchant_id', 'merchant_avg_amount', 'merchant_std_amount', 'merchant_tx_count']
            
            df = df.merge(merchant_stats, on='merchant_id', how='left')
            df['amount_vs_merchant_avg'] = df['amount'] / (df['merchant_avg_amount'] + 1e-6)
        
        # Location features
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df['distance_from_home'] = np.sqrt(
                (df['latitude'] - df['latitude'].median())**2 + 
                (df['longitude'] - df['longitude'].median())**2
            )
        
        # Time-based features
        if 'timestamp' in df.columns:
            df['time_since_last_tx'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 3600
            df['time_since_last_tx'] = df['time_since_last_tx'].fillna(24)  # Default 24 hours
        
        # Transaction type features
        if 'transaction_type' in df.columns:
            df['is_transfer'] = (df['transaction_type'] == 'transfer').astype(int)
            df['is_payment'] = (df['transaction_type'] == 'payment').astype(int)
            df['is_withdrawal'] = (df['transaction_type'] == 'withdrawal').astype(int)
        
        # Risk score features
        df['risk_score'] = self.calculate_risk_score(df)
        
        return df
    
    def calculate_risk_score(self, df):
        """
        Calculate a simple risk score based on multiple factors
        """
        risk_factors = []
        
        # Amount risk
        if 'amount' in df.columns:
            amount_risk = np.where(df['amount'] > df['amount'].quantile(0.9), 0.3, 0)
            risk_factors.append(amount_risk)
        
        # Time risk (night transactions)
        if 'is_night' in df.columns:
            time_risk = df['is_night'] * 0.2
            risk_factors.append(time_risk)
        
        # Weekend risk
        if 'is_weekend' in df.columns:
            weekend_risk = df['is_weekend'] * 0.1
            risk_factors.append(weekend_risk)
        
        # High amount risk
        if 'is_high_amount' in df.columns:
            high_amount_risk = df['is_high_amount'] * 0.4
            risk_factors.append(high_amount_risk)
        
        # Combine risk factors
        if risk_factors:
            return np.sum(risk_factors, axis=0)
        else:
            return np.zeros(len(df))
    
    def clean_data(self, df):
        """
        Clean and prepare data for ML models
        """
        df = df.copy()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Handle categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna('Unknown')
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """
        Encode categorical features using label encoding
        """
        df = df.copy()
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in ['transaction_id', 'user_id', 'merchant_id']:  # Skip ID columns
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        unique_values = set(df[col].astype(str))
                        known_values = set(self.label_encoders[col].classes_)
                        new_values = unique_values - known_values
                        
                        if new_values:
                            # Add new categories to encoder
                            all_values = list(known_values) + list(new_values)
                            self.label_encoders[col].classes_ = np.array(all_values)
                        
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def scale_features(self, df, fit=True):
        """
        Scale numerical features using StandardScaler
        """
        df = df.copy()
        
        # Select only numerical columns for scaling
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        
        if fit:
            df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        else:
            df[numerical_columns] = self.scaler.transform(df[numerical_columns])
        
        return df
    
    def preprocess_training_data(self, df):
        """
        Complete preprocessing pipeline for training data
        """
        print("Starting data preprocessing...")
        
        # Create features
        df = self.create_features(df)
        print("[SUCCESS] Feature engineering completed")
        
        # Clean data
        df = self.clean_data(df)
        print("[SUCCESS] Data cleaning completed")
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=True)
        print("[SUCCESS] Categorical encoding completed")
        
        # Scale features
        df = self.scale_features(df, fit=True)
        print("[SUCCESS] Feature scaling completed")
        
        self.is_fitted = True
        print("[SUCCESS] Preprocessing pipeline fitted successfully")
        
        return df
    
    def preprocess_realtime_data(self, df):
        """
        Preprocess real-time transaction data using fitted pipeline
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted on training data first")
        
        # Create features
        df = self.create_features(df)
        
        # Clean data
        df = self.clean_data(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=False)
        
        # Scale features
        df = self.scale_features(df, fit=False)
        
        return df
    
    def get_feature_columns(self, df):
        """
        Get list of feature columns (excluding target and ID columns)
        """
        exclude_columns = ['is_fraud', 'transaction_id', 'user_id', 'merchant_id', 'timestamp']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        return feature_columns
    
    def save_preprocessor(self, filepath):
        """
        Save the fitted preprocessor
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before saving")
        
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'imputer': self.imputer,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath):
        """
        Load a fitted preprocessor
        """
        preprocessor_data = joblib.load(filepath)
        
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.imputer = preprocessor_data['imputer']
        self.is_fitted = preprocessor_data['is_fitted']
        
        print(f"Preprocessor loaded from {filepath}")

def preprocess_single_transaction(transaction_dict, preprocessor):
    """
    Preprocess a single transaction for real-time prediction
    """
    # Convert to DataFrame
    df = pd.DataFrame([transaction_dict])
    
    # Preprocess using fitted preprocessor
    processed_df = preprocessor.preprocess_realtime_data(df)
    
    # Get feature columns
    feature_columns = preprocessor.get_feature_columns(processed_df)
    
    # Return features as numpy array
    return processed_df[feature_columns].values

if __name__ == "__main__":
    # Test the preprocessor
    print("Testing TransactionPreprocessor...")
    
    # Create sample data
    sample_data = {
        'transaction_id': ['tx1', 'tx2', 'tx3'],
        'user_id': ['user1', 'user2', 'user1'],
        'amount': [100.0, 500.0, 50.0],
        'timestamp': ['2023-01-01 10:00:00', '2023-01-01 15:00:00', '2023-01-01 20:00:00'],
        'merchant_id': ['merchant1', 'merchant2', 'merchant1'],
        'transaction_type': ['payment', 'transfer', 'payment'],
        'latitude': [40.7128, 40.7589, 40.7128],
        'longitude': [-74.0060, -73.9851, -74.0060],
        'is_fraud': [0, 1, 0]
    }
    
    df = pd.DataFrame(sample_data)
    preprocessor = TransactionPreprocessor()
    
    # Test preprocessing
    processed_df = preprocessor.preprocess_training_data(df)
    print(f"Original shape: {df.shape}")
    print(f"Processed shape: {processed_df.shape}")
    print(f"Feature columns: {preprocessor.get_feature_columns(processed_df)}")
