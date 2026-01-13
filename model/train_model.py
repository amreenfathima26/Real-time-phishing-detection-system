"""
Model Training Module for Real-Time Fraud Detection
Implements multiple ML algorithms for fraud detection with real-time capabilities
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available, skipping LightGBM models")
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available, skipping deep learning models")
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModel:
    """Main class for training and managing fraud detection models"""
    
    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = None
        self.is_trained = False
    
    def prepare_data(self, df, target_column='is_fraud', test_size=0.2):
        """
        Prepare data for training and testing
        """
        # Ensure data types are correct
        print("Preparing data with proper type conversion...")
        
        # Convert all numeric columns to float
        numeric_columns = ['amount', 'old_balance_orig', 'new_balance_orig', 
                          'old_balance_dest', 'new_balance_dest', 'hour', 'day_of_week']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
        
        # Convert categorical columns to string
        categorical_columns = ['type', 'name_orig', 'name_dest']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Convert target column to int
        if target_column in df.columns:
            df[target_column] = pd.to_numeric(df[target_column], errors='coerce').fillna(0).astype(int)
        
        # Separate features and target
        X = df.drop([target_column], axis=1)
        y = df[target_column]
        
        # Encode categorical variables
        print("Encoding categorical variables...")
        
        # Create label encoders for categorical columns
        self.label_encoders = {}
        for col in categorical_columns:
            if col in X.columns:
                # Use pandas categorical encoding for better handling
                X[col] = X[col].astype('category')
                # Store the category mapping
                self.label_encoders[col] = dict(enumerate(X[col].cat.categories))
                # Convert to codes
                X[col] = X[col].cat.codes
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train):
        """
        Train Random Forest model
        """
        print("Training Random Forest...")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.models['RandomForest'] = grid_search.best_estimator_
        print(f"[SUCCESS] Random Forest trained with best params: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def train_xgboost(self, X_train, y_train):
        """
        Train XGBoost model
        """
        print("Training XGBoost...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.models['XGBoost'] = grid_search.best_estimator_
        print(f"[SUCCESS] XGBoost trained with best params: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def train_lightgbm(self, X_train, y_train):
        """
        Train LightGBM model
        """
        if not LIGHTGBM_AVAILABLE:
            print("Skipping LightGBM - not available")
            return None
            
        print("Training LightGBM...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [31, 50]
        }
        
        lgb_model = lgb.LGBMClassifier(random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(lgb_model, param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.models['LightGBM'] = grid_search.best_estimator_
        print(f"[SUCCESS] LightGBM trained with best params: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def train_neural_network_sklearn(self, X_train, y_train):
        """
        Train Neural Network using scikit-learn
        """
        print("Training Neural Network (sklearn)...")
        
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.001, 0.01]
        }
        
        nn = MLPClassifier(random_state=42, max_iter=1000)
        grid_search = GridSearchCV(nn, param_grid, cv=3, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.models['NeuralNetwork'] = grid_search.best_estimator_
        print(f"[SUCCESS] Neural Network trained with best params: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def train_neural_network_tensorflow(self, X_train, y_train, X_val, y_val):
        """
        Train Deep Neural Network using TensorFlow/Keras
        """
        if not TENSORFLOW_AVAILABLE:
            print("Skipping Deep Neural Network - TensorFlow not available")
            return None
            
        print("Training Deep Neural Network (TensorFlow)...")
        
        # Build model
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.models['DeepNeuralNetwork'] = model
        print("[SUCCESS] Deep Neural Network trained successfully")
        
        return model
    
    def train_logistic_regression(self, X_train, y_train):
        """
        Train Logistic Regression model
        """
        print("Training Logistic Regression...")
        
        lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        lr.fit(X_train, y_train)
        
        self.models['LogisticRegression'] = lr
        print("[SUCCESS] Logistic Regression trained successfully")
        
        return lr
    
    def train_svm(self, X_train, y_train):
        """
        Train Support Vector Machine model
        """
        print("Training SVM...")
        
        # Use smaller sample for SVM due to computational complexity
        if len(X_train) > 10000:
            sample_indices = np.random.choice(len(X_train), 10000, replace=False)
            X_train_sample = X_train.iloc[sample_indices]
            y_train_sample = y_train.iloc[sample_indices]
        else:
            X_train_sample = X_train
            y_train_sample = y_train
        
        svm = SVC(random_state=42, class_weight='balanced', probability=True)
        svm.fit(X_train_sample, y_train_sample)
        
        self.models['SVM'] = svm
        print("[SUCCESS] SVM trained successfully")
        
        return svm
    
    def train_isolation_forest(self, X_train):
        """
        Train Isolation Forest for anomaly detection
        """
        print("Training Isolation Forest...")
        
        iso_forest = IsolationForest(
            contamination=0.1,  # Assuming 10% fraud rate
            random_state=42
        )
        iso_forest.fit(X_train)
        
        self.models['IsolationForest'] = iso_forest
        print("[SUCCESS] Isolation Forest trained successfully")
        
        return iso_forest
    
    def evaluate_model(self, model, model_name, X_test, y_test):
        """
        Evaluate model performance
        """
        # Predictions
        if model_name == 'DeepNeuralNetwork':
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
        elif model_name == 'IsolationForest':
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == -1, 1, 0)  # Convert -1/1 to 0/1
            y_pred_proba = model.decision_function(X_test)
            y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.0
        
        performance = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
        
        self.model_performance[model_name] = performance
        
        print(f"\n{model_name} Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
        return performance
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train all models and evaluate their performance
        """
        print("=" * 50)
        print("TRAINING ALL FRAUD DETECTION MODELS")
        print("=" * 50)
        
        # Ensure data types are correct for all datasets
        print("Converting data types for training...")
        
        # Convert numeric columns to float
        numeric_columns = ['amount', 'old_balance_orig', 'new_balance_orig', 
                          'old_balance_dest', 'new_balance_dest', 'hour', 'day_of_week']
        
        for col in numeric_columns:
            if col in X_train.columns:
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0).astype(float)
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0).astype(float)
        
        # Convert categorical columns to string and encode them
        categorical_columns = ['type', 'name_orig', 'name_dest']
        for col in categorical_columns:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype(str)
                X_test[col] = X_test[col].astype(str)
        
        # Encode categorical variables
        # Create label encoders for categorical columns
        if not hasattr(self, 'label_encoders'):
            self.label_encoders = {}
        
        for col in categorical_columns:
            if col in X_train.columns:
                if col not in self.label_encoders:
                    # Use pandas categorical encoding for better handling
                    combined_data = pd.concat([X_train[col], X_test[col]])
                    combined_data = combined_data.astype('category')
                    # Store the category mapping
                    self.label_encoders[col] = dict(enumerate(combined_data.cat.categories))
                
                # Apply encoding to both train and test data
                X_train[col] = X_train[col].astype('category')
                X_train[col] = X_train[col].cat.set_categories(list(self.label_encoders[col].values()))
                X_train[col] = X_train[col].cat.codes
                
                X_test[col] = X_test[col].astype('category')
                X_test[col] = X_test[col].cat.set_categories(list(self.label_encoders[col].values()))
                X_test[col] = X_test[col].cat.codes
        
        # Convert target to int
        y_train = pd.to_numeric(y_train, errors='coerce').fillna(0).astype(int)
        y_test = pd.to_numeric(y_test, errors='coerce').fillna(0).astype(int)
        
        # Split training data for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Train models
        models_to_train = [
            self.train_logistic_regression,
            self.train_random_forest,
            self.train_xgboost,
            self.train_lightgbm,
            self.train_neural_network_sklearn,
            self.train_svm,
            self.train_isolation_forest
        ]
        
        # Train TensorFlow model separately
        self.train_neural_network_tensorflow(X_train_split, y_train_split, X_val, y_val)
        
        # Train other models
        for train_func in models_to_train:
            try:
                if train_func == self.train_isolation_forest:
                    train_func(X_train_split)
                else:
                    train_func(X_train_split, y_train_split)
            except Exception as e:
                print(f"Error training model: {e}")
                continue
        
        # Evaluate all models
        print("\n" + "=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)
        
        for model_name, model in self.models.items():
            try:
                self.evaluate_model(model, model_name, X_test, y_test)
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                continue
        
        # Select best model
        self.select_best_model()
        
        self.is_trained = True
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED")
        print("=" * 50)
    
    def select_best_model(self):
        """
        Select the best performing model based on F1-score
        """
        best_f1 = 0
        best_model_name = None
        
        for model_name, performance in self.model_performance.items():
            if performance['f1_score'] > best_f1:
                best_f1 = performance['f1_score']
                best_model_name = model_name
        
        if best_model_name:
            self.best_model = self.models[best_model_name]
            self.best_model_name = best_model_name
            print(f"\n[BEST] Best Model: {best_model_name} (F1-Score: {best_f1:.4f})")
        else:
            print("\n[ERROR] No models were successfully trained")
    
    def predict_fraud(self, X):
        """
        Predict fraud using the best model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.best_model is None:
            raise ValueError("No best model available")
        
        if self.best_model_name == 'DeepNeuralNetwork':
            probabilities = self.best_model.predict(X)
            predictions = (probabilities > 0.5).astype(int)
        elif self.best_model_name == 'IsolationForest':
            predictions = self.best_model.predict(X)
            predictions = np.where(predictions == -1, 1, 0)
            probabilities = self.best_model.decision_function(X)
            probabilities = (probabilities - probabilities.min()) / (probabilities.max() - probabilities.min())
        else:
            predictions = self.best_model.predict(X)
            probabilities = self.best_model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def save_models(self, filepath_prefix="model/"):
        """
        Save all trained models
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before saving")
        
        for model_name, model in self.models.items():
            if model_name == 'DeepNeuralNetwork':
                model.save(f"{filepath_prefix}deep_neural_network.h5")
            else:
                joblib.dump(model, f"{filepath_prefix}{model_name.lower()}_model.pkl")
        
        # Save model metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'model_performance': self.model_performance,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        
        joblib.dump(metadata, f"{filepath_prefix}model_metadata.pkl")
        print(f"[SUCCESS] Models saved to {filepath_prefix}")
    
    def load_models(self, filepath_prefix="model/"):
        """
        Load trained models
        """
        import os
        
        # Load model metadata
        metadata = joblib.load(f"{filepath_prefix}model_metadata.pkl")
        self.best_model_name = metadata['best_model_name']
        self.model_performance = metadata['model_performance']
        self.feature_columns = metadata['feature_columns']
        self.is_trained = metadata['is_trained']
        
        # Load individual models
        model_files = {
            'RandomForest': 'randomforest_model.pkl',
            'XGBoost': 'xgboost_model.pkl',
            'LightGBM': 'lightgbm_model.pkl',
            'NeuralNetwork': 'neuralnetwork_model.pkl',
            'LogisticRegression': 'logisticregression_model.pkl',
            'SVM': 'svm_model.pkl',
            'IsolationForest': 'isolationforest_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            filepath = f"{filepath_prefix}{filename}"
            if os.path.exists(filepath):
                self.models[model_name] = joblib.load(filepath)
        
        # Load TensorFlow model
        tf_model_path = f"{filepath_prefix}deep_neural_network.h5"
        if os.path.exists(tf_model_path):
            self.models['DeepNeuralNetwork'] = tf.keras.models.load_model(tf_model_path)
        
        # Set best model
        if self.best_model_name and self.best_model_name in self.models:
            self.best_model = self.models[self.best_model_name]
        
        print(f"[SUCCESS] Models loaded from {filepath_prefix}")

def train_fraud_detection_model(data_path="data/transactions.csv", save_path="model/"):
    """
    Main function to train fraud detection models
    """
    print("Starting Fraud Detection Model Training...")
    
    # Load data
    try:
        df = pd.read_csv(data_path)
        print(f"[SUCCESS] Data loaded: {df.shape}")
    except FileNotFoundError:
        print(f"[ERROR] Data file not found: {data_path}")
        return None
    
    # Initialize model trainer
    trainer = FraudDetectionModel()
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    print(f"[SUCCESS] Data prepared - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train all models
    trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Save models
    trainer.save_models(save_path)
    
    return trainer

if __name__ == "__main__":
    # Train models
    trainer = train_fraud_detection_model()
    
    if trainer:
        print("\n🎉 Model training completed successfully!")
        print(f"Best model: {trainer.best_model_name}")
        print(f"Best F1-score: {trainer.model_performance[trainer.best_model_name]['f1_score']:.4f}")
