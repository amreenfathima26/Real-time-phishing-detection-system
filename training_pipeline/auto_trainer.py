"""
Auto-Training Pipeline
Incremental learning from CSV + Database data
Combines historical CSV data with real-time feedback
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os

# Ensure pandas handles NaN properly
pd.options.mode.chained_assignment = None

sys.path.append(str(Path(__file__).parent.parent))

from database.database import get_db, Database
from ml_engine.nlp_model import NLPModel
from ml_engine.cnn_model import CNNModel
from ml_engine.gnn_model import GNNModel
from ml_engine.adversarial_detector import AdversarialDetector

class AutoTrainer:
    """Automatic training pipeline for continuous learning"""
    
    def __init__(self, db: Database):
        self.db = db
        # Get absolute paths to CSV files
        base_dir = Path(__file__).parent.parent
        self.data_dir = base_dir / "data"
        self.csv_data_paths = [
            # str(self.data_dir / "complete_training_dataset.csv"), # Skip noisy legacy data
            # str(self.data_dir / "transactions.csv"),
            # str(self.data_dir / "fraudulent_testing_dataset.csv"),
            # str(self.data_dir / "legitimate_testing_dataset.csv"),
            str(self.data_dir / "urls_training.csv"), 
            str(self.data_dir / "phishing_messages.csv")
        ]
        print(f"[AUTO TRAINER] CSV paths: {self.csv_data_paths}")
    
    def prepare_training_data(self, model_type: str = 'nlp') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from CSV + Database
        """
        print(f"[AUTO TRAINER] Preparing training data for {model_type}...")
        
        # Load CSV data
        csv_data = []
        for csv_path in self.csv_data_paths:
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if len(df) > 0:
                        # Check relevance
                        is_relevant = False
                        cols = df.columns.tolist()
                        
                        if model_type in ['nlp', 'adversarial']:
                            if any(c in cols for c in ['content', 'subject', 'url', 'text']):
                                is_relevant = True
                        elif model_type in ['cnn', 'gnn']:
                            if 'url' in cols or 'raw_url' in cols:
                                is_relevant = True
                        
                        if is_relevant:
                            csv_data.append(df)
                            print(f"  [OK] Loaded {len(df)} relevant samples from {os.path.basename(csv_path)}")
                except Exception as e:
                    print(f"  [ERROR] Error loading {os.path.basename(csv_path)}: {e}")
        
        # Load database data
        db_data = self._load_database_training_data(model_type)
        all_items = self._load_all_messages_for_training(model_type)
        if all_items:
            db_data.extend(all_items)
        
        # Combine data
        combined_df = pd.DataFrame()
        if csv_data:
            combined_df = pd.concat(csv_data, ignore_index=True, sort=False)
        
        if db_data:
            db_df = pd.DataFrame(db_data)
            if not combined_df.empty:
                combined_df = pd.concat([combined_df, db_df], ignore_index=True, sort=False)
            else:
                combined_df = db_df
        
        if combined_df.empty:
            raise ValueError(f"No training data available for {model_type}")
        
        X, y = self._extract_features(combined_df, model_type)
        return X, y
    
    def _load_database_training_data(self, model_type: str = 'nlp') -> List[Dict]:
        """Load training data from database (messages/urls with feedback)"""
        print(f"  [DB] Loading {model_type} data from database...")
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if model_type == 'nlp' or model_type == 'adversarial':
                cursor.execute("""
                    SELECT m.*, f.correct_label
                    FROM messages m
                    JOIN feedback_logs f ON m.id = f.item_id AND f.item_type = 'message'
                    WHERE f.used_for_training = 0
                    LIMIT 1000
                """)
                rows = cursor.fetchall()
                data = []
                for row in rows:
                    r = dict(row)
                    data.append({
                        'content': r['content'],
                        'subject': r.get('subject', ''),
                        'is_phishing': r['correct_label'],
                        'source': 'database'
                    })
            else: # cnn, gnn
                cursor.execute("""
                    SELECT u.*, f.correct_label
                    FROM urls u
                    JOIN feedback_logs f ON u.id = f.item_id AND f.item_type = 'url'
                    WHERE f.used_for_training = 0
                    LIMIT 1000
                """)
                rows = cursor.fetchall()
                data = []
                for row in rows:
                    r = dict(row)
                    data.append({
                        'url': r['raw_url'],
                        'is_phishing': r['correct_label'],
                        'source': 'database'
                    })
        
        print(f"  [OK] Loaded {len(data)} samples from database (with feedback)")
        return data
    
    def _load_all_messages_for_training(self, model_type: str = 'nlp') -> List[Dict]:
        """Load all items from database for training (use detected_label)"""
        print(f"  [DB] Loading all items for {model_type} from database...")
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if model_type == 'nlp' or model_type == 'adversarial':
                cursor.execute("""
                    SELECT id, content, subject, detected_label
                    FROM messages
                    WHERE content IS NOT NULL AND content != ''
                    AND detected_label IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT 500
                """)
                rows = cursor.fetchall()
                data = []
                for row in rows:
                    r = dict(row)
                    data.append({
                        'content': r['content'],
                        'subject': r.get('subject', ''),
                        'is_phishing': r.get('detected_label', 0),
                        'source': 'database_messages'
                    })
            else: # cnn, gnn
                cursor.execute("""
                    SELECT id, raw_url as url, is_phishing
                    FROM urls
                    WHERE raw_url IS NOT NULL
                    AND is_phishing IS NOT NULL
                    ORDER BY updated_at DESC
                    LIMIT 500
                """)
                rows = cursor.fetchall()
                data = []
                for row in rows:
                    r = dict(row)
                    data.append({
                        'url': r['url'],
                        'is_phishing': r['is_phishing'],
                        'source': 'database_urls'
                    })
        
        print(f"  [OK] Loaded {len(data)} items from database")
        return data

    def _extract_features(self, df: pd.DataFrame, model_type: str = 'nlp') -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features and labels from the combined dataframe"""
        print(f"  [FEATURES] Extracting features for {model_type}...")
        
        nlp_model = NLPModel()
        cnn_model = CNNModel()
        gnn_model = GNNModel()
        
        features = []
        labels = []
        processed_count = 0
        
        cols = {
            'nlp': ['phishing_probability', 'urgency_score', 'suspicious_keywords', 'trust_score', 'num_urls'],
            'cnn': ['phishing_probability', 'suspicious_url', 'fake_login_form', 'brand_impersonation', 'lookalike_domain', 'ssl_issues'],
            'gnn': ['phishing_probability', 'suspicious_domain', 'cluster_risk'],
            'adversarial': ['phishing_probability', 'urgency_score', 'suspicious_keywords', 'trust_score', 'num_urls']
        }
        
        for idx, row in df.iterrows():
            try:
                r = row
                feature_vector = []
                label = None
                
                if 'is_phishing' in r and pd.notnull(r['is_phishing']): 
                    label = int(r['is_phishing'])
                elif 'is_fraud' in r and pd.notnull(r['is_fraud']): 
                    label = int(r['is_fraud'])
                
                if label is None: continue
                
                if model_type == 'nlp' or model_type == 'adversarial':
                    content = r.get('content')
                    if pd.isnull(content) or str(content).lower() == 'nan':
                        content = r.get('text')
                    if pd.isnull(content) or str(content).lower() == 'nan':
                        content = r.get('url')
                    
                    content = str(content) if pd.notnull(content) else ""
                    subject = str(r.get('subject', '')) if pd.notnull(r.get('subject')) else ""
                    
                    if not content or content.lower() == 'nan': continue
                    
                    fs = nlp_model.analyze_text(content, subject)
                    
                    feature_vector = [
                        fs.get('phishing_probability', 0.0),
                        fs.get('urgency_score', 0.0),
                        fs.get('suspicious_keywords', 0.0),
                        fs.get('trust_score', 0.0),
                        float(fs.get('num_urls', 0))
                    ]
                elif model_type == 'cnn':
                    url = str(r.get('url', r.get('raw_url', '')))
                    if not url or len(url) < 4: continue
                    fs = cnn_model.analyze_webpage(url)
                    feature_vector = [
                        fs.get('phishing_probability', 0.0),
                        fs.get('suspicious_url', 0.0),
                        fs.get('fake_login_form', 0.0),
                        fs.get('brand_impersonation', 0.0),
                        fs.get('lookalike_domain', 0.0),
                        fs.get('ssl_issues', 0.0)
                    ]
                elif model_type == 'gnn':
                    url = str(r.get('url', r.get('raw_url', '')))
                    if not url or len(url) < 4: continue
                    fs = gnn_model.analyze_domain(url)
                    feature_vector = [
                        fs.get('phishing_probability', 0.0),
                        fs.get('suspicious_domain', 0.0),
                        fs.get('cluster_risk', 0.0)
                    ]
                
                if feature_vector:
                    features.append(feature_vector)
                    labels.append(label)
                    processed_count += 1
                    if processed_count % 500 == 0:
                        print(f"    Processed {processed_count} samples...")
            except: continue
            
        if not features:
            raise ValueError(f"No features extracted for {model_type}")
            
        X = pd.DataFrame(features, columns=cols[model_type])
        y = pd.Series(labels)
        
        # Rule-based accuracy check for debugging
        preds = (X['phishing_probability'] > 0.5).astype(int)
        rule_acc = (preds == y).mean()
        print(f"  [DEBUG] Rule-based Accuracy: {rule_acc:.4f}")
        
        print(f"  [OK] Extracted features: {X.shape}, Phishing: {y.sum()}")
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'nlp') -> Dict[str, float]:
        """Train models with K-Fold Cross-Validation"""
        print(f"[TRAIN] Training {model_type} model with CV...")
        
        from sklearn.model_selection import StratifiedKFold
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        import joblib
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        # Hyperparameters for improved accuracy
        best_model = RandomForestClassifier(
            n_estimators=300, 
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        
        # Use simple split for fast initial training if data is huge, but CV for quality
        if len(X) < 5000:
            for train_idx, test_idx in skf.split(X, y):
                X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
                y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
                
                temp_model = RandomForestClassifier(n_estimators=100, random_state=42)
                temp_model.fit(X_train_cv, y_train_cv)
                cv_scores.append(accuracy_score(y_test_cv, temp_model.predict(X_test_cv)))
            
            print(f"  [CV] Average Accuracy: {np.mean(cv_scores):.4f}")

        # Final Training
        best_model.fit(X, y)
        
        y_pred = best_model.predict(X)
        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred, zero_division=0)),
            'recall': float(recall_score(y, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y, y_pred, zero_division=0)),
            'training_samples': int(len(X))
        }
        
        print(f"  [OK] Training completed! Final Training Accuracy: {metrics['accuracy']:.4f}")
        
        model_path = f"model/{model_type}_auto_trained.pkl"
        os.makedirs("model", exist_ok=True)
        joblib.dump(best_model, model_path)
        print(f"  [SAVE] Model saved to {model_path}")
        
        return metrics
    
    def run_auto_training(self, model_type: str = 'nlp') -> Dict[str, any]:
        """Run complete auto-training pipeline"""
        print("=" * 60)
        print(f"🔄 AUTO-TRAINING PIPELINE: {model_type.upper()}")
        print("=" * 60)
        
        batch_id = self.db.create_training_batch(batch_type='incremental', model_type=model_type, status='training')
        try:
            X, y = self.prepare_training_data(model_type)
            metrics = self.train_models(X, y, model_type)
            
            version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.db.create_model_version(
                model_type=model_type, version=version, accuracy=metrics['accuracy'],
                precision=metrics['precision'], recall=metrics['recall'], f1_score=metrics['f1_score'],
                training_samples=metrics['training_samples'], model_path=f"model/{model_type}_auto_trained.pkl",
                is_active=True
            )
            
            self.db.update_training_batch(batch_id=batch_id, status='completed', accuracy_after=metrics['accuracy'])
            print("[AUTO-TRAINING] COMPLETED SUCCESSFULLY")
            return {'success': True, 'metrics': metrics}
        except Exception as e:
            print(f"[ERROR] Auto-training failed: {e}")
            self.db.update_training_batch(batch_id=batch_id, status='failed', error_message=str(e))
            return {'success': False, 'error': str(e)}

def main():
    db = get_db()
    trainer = AutoTrainer(db)
    trainer.run_auto_training(model_type='nlp')

if __name__ == "__main__":
    main()
