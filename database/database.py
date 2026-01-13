"""
Database connection and operations - SQLite Implementation
"""
import os
import sqlite3
import json
import urllib.parse as urlparse
from contextlib import contextmanager
from typing import Optional, Dict, List, Any
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

class Database:
    """Database manager - Supports SQLite, PostgreSQL, and MySQL"""
    
    def __init__(self):
        """Initialize database connection"""
        self.db_url = os.getenv("DATABASE_URL")
        self.db_type = "sqlite"
        
        if self.db_url:
            if self.db_url.startswith("postgres://") or self.db_url.startswith("postgresql://"):
                self.db_type = "postgresql"
            elif self.db_url.startswith("mysql://"):
                self.db_type = "mysql"
        
        if self.db_type == "sqlite":
            self.db_path = Path(__file__).parent.parent / "phishing_detection.db"
            self.placeholder = "?"
            self._ensure_database_exists()
        else:
            self.placeholder = "%s"
            print(f"✅ Using cloud {self.db_type} database.")

    def _ensure_database_exists(self):
        """Create database and tables if they don't exist (Only for SQLite)"""
        if self.db_type != "sqlite":
            return
            
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if users table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
                if not cursor.fetchone():
                    # Read and execute schema
                    schema_path = Path(__file__).parent / "schema_sqlite.sql"
                    if schema_path.exists():
                        with open(schema_path, 'r', encoding='utf-8') as f:
                            schema = f.read()
                        
                        # Execute schema script
                        cursor.executescript(schema)
                        print("✅ SQLite database initialized successfully.")
                    else:
                        print("❌ Schema file not found!")
        except Exception as e:
            print(f"❌ Database setup error: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Get database connection based on type"""
        if self.db_type == "sqlite":
            conn = sqlite3.connect(
                str(self.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
            )
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()
        elif self.db_type == "postgresql":
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            # Handle elephantSQL/Render/Neon URLs which might use postgres://
            url = self.db_url.replace("postgres://", "postgresql://")
            conn = psycopg2.connect(url, cursor_factory=RealDictCursor)
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()
        elif self.db_type == "mysql":
            import mysql.connector
            
            url = urlparse.urlparse(self.db_url)
            conn = mysql.connector.connect(
                user=url.username,
                password=url.password,
                host=url.hostname,
                port=url.port or 3306,
                database=url.path[1:]
            )
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()
    
    # Helper to convert Row to dict
    def _row_to_dict(self, row: Optional[sqlite3.Row]) -> Optional[Dict]:
        if row is None:
            return None
        return dict(row)
    
    def _rows_to_dict_list(self, rows: List[sqlite3.Row]) -> List[Dict]:
        return [dict(row) for row in rows]

    # User Operations
    def create_user(self, name: str, email: str, password_hash: str, role: str = 'user') -> int:
        """Create a new user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"INSERT INTO users (name, email, password_hash, role) VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder})",
                (name, email, password_hash, role)
            )
            return cursor.lastrowid
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Get user by email"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM users WHERE email = {self.placeholder}", (email,))
            return self._row_to_dict(cursor.fetchone())
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM users WHERE id = {self.placeholder}", (user_id,))
            return self._row_to_dict(cursor.fetchone())
    
    def update_last_login(self, user_id: int):
        """Update user last login"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = {self.placeholder}", (user_id,))
    
    # Message Operations
    def create_message(self, user_id: Optional[int], channel: str, content: str,
                      subject: Optional[str] = None, sender: Optional[str] = None,
                      recipient: Optional[str] = None) -> int:
        """Create a new message record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""INSERT INTO messages (user_id, channel, content, subject, sender, recipient)
                   VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder})""",
                (user_id, channel, content, subject, sender, recipient)
            )
            return cursor.lastrowid
    
    def update_message_prediction(self, message_id: int, detected_label: int,
                                 confidence_score: float, nlp_score: Optional[float] = None,
                                 adversarial_score: Optional[float] = None,
                                 risk_factors: Optional[List[str]] = None,
                                 explainable_reasons: Optional[Dict] = None):
        """Update message with prediction results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""UPDATE messages SET detected_label = {self.placeholder}, confidence_score = {self.placeholder},
                   nlp_score = {self.placeholder}, adversarial_score = {self.placeholder}, risk_factors = {self.placeholder}, explainable_reasons = {self.placeholder}
                   WHERE id = {self.placeholder}""",
                (detected_label, confidence_score, nlp_score, adversarial_score,
                 json.dumps(risk_factors) if risk_factors else None,
                 json.dumps(explainable_reasons) if explainable_reasons else None,
                 message_id)
            )
    
    # URL Operations
    def create_url(self, raw_url: str, expanded_url: Optional[str] = None,
                  final_url: Optional[str] = None, domain_id: Optional[int] = None) -> int:
        """Create a new URL record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""INSERT INTO urls (raw_url, expanded_url, final_url, domain_id)
                   VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder})""",
                (raw_url, expanded_url, final_url, domain_id)
            )
            return cursor.lastrowid
    
    def update_url_prediction(self, url_id: int, risk_score: float, is_phishing: int,
                            gnn_score: Optional[float] = None, cnn_score: Optional[float] = None,
                            redirect_depth: int = 0, redirect_chain: Optional[List[str]] = None):
        """Update URL with prediction results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""UPDATE urls SET risk_score = {self.placeholder}, is_phishing = {self.placeholder}, gnn_score = {self.placeholder},
                   cnn_score = {self.placeholder}, redirect_depth = {self.placeholder}, redirect_chain = {self.placeholder}, updated_at = CURRENT_TIMESTAMP
                   WHERE id = {self.placeholder}""",
                (risk_score, is_phishing, gnn_score, cnn_score, redirect_depth,
                 json.dumps(redirect_chain) if redirect_chain else None, url_id)
            )
    
    # Get User Scans
    def get_user_messages(self, user_id: int, limit: int = 50) -> List[Dict]:
        """Get user's message scans"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""SELECT id, channel, content, subject, sender, detected_label, 
                   confidence_score, created_at 
                   FROM messages 
                   WHERE user_id = {self.placeholder} 
                   ORDER BY created_at DESC 
                   LIMIT {self.placeholder}""",
                (user_id, limit)
            )
            results = self._rows_to_dict_list(cursor.fetchall())
            # Parse JSON fields
            for result in results:
                if result.get('risk_factors'):
                    try:
                        result['risk_factors'] = json.loads(result['risk_factors'])
                    except:
                        result['risk_factors'] = []
                if result.get('explainable_reasons'):
                    try:
                        result['explainable_reasons'] = json.loads(result['explainable_reasons'])
                    except:
                        result['explainable_reasons'] = {}
            return results
    
    def get_user_urls(self, user_id: Optional[int] = None, limit: int = 50) -> List[Dict]:
        """Get URL scans (all users or specific user)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if user_id:
                # For now, URLs don't have user_id in params, but schema might not link it properly in all ops
                # Assuming query logic is same as before
                cursor.execute(
                    f"""SELECT id, raw_url, risk_score, is_phishing, gnn_score, 
                       cnn_score, redirect_depth, created_at 
                       FROM urls 
                       ORDER BY created_at DESC 
                       LIMIT {self.placeholder}""",
                    (limit,)
                )
            else:
                cursor.execute(
                    f"""SELECT id, raw_url, risk_score, is_phishing, gnn_score, 
                       cnn_score, redirect_depth, created_at 
                       FROM urls 
                       ORDER BY created_at DESC 
                       LIMIT {self.placeholder}""",
                    (limit,)
                )
            results = self._rows_to_dict_list(cursor.fetchall())
            # Parse JSON fields
            for result in results:
                if result.get('redirect_chain'):
                    try:
                        result['redirect_chain'] = json.loads(result['redirect_chain'])
                    except:
                        result['redirect_chain'] = []
            return results
    
    # Training Operations
    def create_training_batch(self, batch_type: str, model_type: str, status: str = 'pending') -> int:
        """Create a new training batch"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""INSERT INTO training_batches (batch_type, model_type, status, started_at)
                   VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, CURRENT_TIMESTAMP)""",
                (batch_type, model_type, status)
            )
            return cursor.lastrowid
    
    def update_training_batch(self, batch_id: int, status: Optional[str] = None,
                             samples_count: Optional[int] = None,
                             csv_samples: Optional[int] = None,
                             db_samples: Optional[int] = None,
                             accuracy_before: Optional[float] = None,
                             accuracy_after: Optional[float] = None,
                             error_message: Optional[str] = None):
        """Update training batch"""
        updates = []
        params = []
        
        if status:
            updates.append(f"status = {self.placeholder}")
            params.append(status)
        if samples_count is not None:
            updates.append(f"samples_count = {self.placeholder}")
            params.append(samples_count)
        if csv_samples is not None:
            updates.append(f"csv_samples = {self.placeholder}")
            params.append(csv_samples)
        if db_samples is not None:
            updates.append(f"db_samples = {self.placeholder}")
            params.append(db_samples)
        if accuracy_before is not None:
            updates.append(f"accuracy_before = {self.placeholder}")
            params.append(accuracy_before)
        if accuracy_after is not None:
            updates.append(f"accuracy_after = {self.placeholder}")
            params.append(accuracy_after)
            updates.append("completed_at = CURRENT_TIMESTAMP")
        if error_message:
            updates.append(f"error_message = {self.placeholder}")
            params.append(error_message)
        
        if updates:
            params.append(batch_id)
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"UPDATE training_batches SET {', '.join(updates)} WHERE id = {self.placeholder}",
                    params
                )
    
    def create_model_version(self, model_type: str, version: str, accuracy: float,
                            precision: float, recall: float, f1_score: float,
                            training_samples: int, model_path: str, is_active: bool = False) -> int:
        """Create a new model version"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Deactivate other versions of same type
            if is_active:
                cursor.execute(
                    f"UPDATE model_versions SET is_active = 0 WHERE model_type = {self.placeholder}",
                    (model_type,)
                )
            cursor.execute(
                f"""INSERT INTO model_versions (model_type, version, accuracy, "precision", 
                   recall, f1_score, training_samples, model_path, is_active, deployed_at)
                   VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder}, CURRENT_TIMESTAMP)""",
                (model_type, version, accuracy, precision, recall, f1_score,
                 training_samples, model_path, 1 if is_active else 0)
            )
            return cursor.lastrowid
    
    def get_active_model_version(self, model_type: str) -> Optional[Dict]:
        """Get active model version"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM model_versions WHERE model_type = {self.placeholder} AND is_active = 1 ORDER BY deployed_at DESC LIMIT 1",
                (model_type,)
            )
            return self._row_to_dict(cursor.fetchone())
    
    # Statistics
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            stats = {}
            
            cursor.execute("SELECT COUNT(*) as count FROM messages")
            result = cursor.fetchone()
            stats['total_messages'] = result['count'] if result else 0
            
            cursor.execute("SELECT COUNT(*) as count FROM messages WHERE detected_label = 1")
            result = cursor.fetchone()
            stats['phishing_messages'] = result['count'] if result else 0
            
            cursor.execute("SELECT COUNT(*) as count FROM urls")
            result = cursor.fetchone()
            stats['total_urls'] = result['count'] if result else 0
            
            cursor.execute("SELECT COUNT(*) as count FROM urls WHERE is_phishing = 1")
            result = cursor.fetchone()
            stats['phishing_urls'] = result['count'] if result else 0
            
            return stats

# Global database instance
_db_instance: Optional[Database] = None

def get_db() -> Database:
    """Get database instance (singleton)"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance

