import sqlite3
import psycopg2
import os
from pathlib import Path
from dotenv import load_dotenv
import json

# Load env
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

SQLITE_DB = Path(__file__).parent / "phishing_detection.db"
POSTGRES_URL = os.getenv("DATABASE_URL")

def migrate():
    print(f"🚀 Starting migration from {SQLITE_DB} to Neon...")
    
    if not SQLITE_DB.exists():
        print("❌ SQLite database not found!")
        return

    # Connect to SQLite
    sqlite_conn = sqlite3.connect(SQLITE_DB)
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cursor = sqlite_conn.cursor()

    # Connect to Postgres
    pg_conn = psycopg2.connect(POSTGRES_URL)
    pg_cursor = pg_conn.cursor()

    # Tables to migrate in order (to respect foreign keys)
    tables = [
        "users",
        "domains",
        "messages",
        "urls",
        "domain_relationships",
        "feedback_logs",
        "model_versions",
        "training_batches",
        "predictions_log",
        "threat_intelligence"
    ]

    try:
        # First, ensure schema exists on PG
        # Note: We'll use the MySQL schema as a reference but adapt for PG if needed.
        # However, Neon/PG handles many things automatically if we just create tables.
        # Let's run the schema_sqlite.sql equivalent for PG first if possible, 
        # but tables might already exist if Database class initialized them (though it only does for SQLite).
        
        # Actually, let's create the tables manually if they don't exist to be safe.
        # We'll use a simplified PG-compatible schema version.
        
        # 1. users
        pg_cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                is_active INTEGER DEFAULT 1,
                last_login TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 2. messages
        pg_cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                channel TEXT NOT NULL,
                content TEXT NOT NULL,
                subject TEXT,
                sender TEXT,
                recipient TEXT,
                detected_label INTEGER DEFAULT 0,
                confidence_score FLOAT DEFAULT 0.0,
                nlp_score FLOAT,
                adversarial_score FLOAT,
                risk_factors TEXT,
                explainable_reasons TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 3. domains
        pg_cursor.execute("""
            CREATE TABLE IF NOT EXISTS domains (
                id SERIAL PRIMARY KEY,
                domain_name TEXT UNIQUE NOT NULL,
                tld TEXT,
                registration_date DATE,
                expiration_date DATE,
                registrar TEXT,
                whois_data TEXT,
                ssl_fingerprint TEXT,
                ssl_valid BOOLEAN DEFAULT FALSE,
                trust_score FLOAT DEFAULT 0.0,
                gnn_cluster_id INTEGER,
                is_phishing INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 4. urls
        pg_cursor.execute("""
            CREATE TABLE IF NOT EXISTS urls (
                id SERIAL PRIMARY KEY,
                raw_url TEXT NOT NULL,
                expanded_url TEXT,
                final_url TEXT,
                domain_id INTEGER REFERENCES domains(id) ON DELETE SET NULL,
                risk_score FLOAT DEFAULT 0.0,
                redirect_depth INTEGER DEFAULT 0,
                redirect_chain TEXT,
                gnn_score FLOAT,
                cnn_score FLOAT,
                is_phishing INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 5. domain_relationships
        pg_cursor.execute("""
            CREATE TABLE IF NOT EXISTS domain_relationships (
                id SERIAL PRIMARY KEY,
                source_domain_id INTEGER REFERENCES domains(id) ON DELETE CASCADE,
                target_domain_id INTEGER REFERENCES domains(id) ON DELETE CASCADE,
                relationship_type TEXT,
                weight FLOAT DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 6. feedback_logs
        pg_cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_logs (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                item_type TEXT NOT NULL,
                item_id INTEGER NOT NULL,
                feedback_type TEXT NOT NULL,
                correct_label INTEGER,
                used_for_training INTEGER DEFAULT 0,
                training_batch_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 7. model_versions
        pg_cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id SERIAL PRIMARY KEY,
                model_type TEXT NOT NULL,
                version TEXT NOT NULL,
                accuracy FLOAT,
                "precision" FLOAT,
                recall FLOAT,
                f1_score FLOAT,
                training_samples INTEGER,
                deployed_at TIMESTAMP,
                is_active INTEGER DEFAULT 0,
                model_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 8. training_batches
        pg_cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_batches (
                id SERIAL PRIMARY KEY,
                batch_type TEXT NOT NULL,
                model_type TEXT NOT NULL,
                samples_count INTEGER,
                csv_samples INTEGER DEFAULT 0,
                db_samples INTEGER DEFAULT 0,
                accuracy_before FLOAT,
                accuracy_after FLOAT,
                status TEXT DEFAULT 'pending',
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 9. predictions_log
        pg_cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions_log (
                id SERIAL PRIMARY KEY,
                item_type TEXT NOT NULL,
                item_id INTEGER NOT NULL,
                model_type TEXT,
                prediction INTEGER,
                probability FLOAT,
                processing_time_ms FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 10. threat_intelligence
        pg_cursor.execute("""
            CREATE TABLE IF NOT EXISTS threat_intelligence (
                id SERIAL PRIMARY KEY,
                ioc_type TEXT NOT NULL,
                ioc_value TEXT NOT NULL,
                threat_type TEXT,
                severity TEXT,
                source TEXT,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        pg_conn.commit()
        print("✅ PostgreSQL Schema verified.")

        # Data Migration
        for table in tables:
            print(f"📦 Migrating table: {table}...")
            sqlite_cursor.execute(f"SELECT * FROM {table}")
            rows = sqlite_cursor.fetchall()
            
            if not rows:
                print(f"  - No data in {table}")
                continue

            # Prepare insert query
            columns = rows[0].keys()
            placeholders = ",".join(["%s"] * len(columns))
            # Handle quoted column names like "precision" for PG
            col_names = ",".join([f'"{c}"' if c == "precision" else c for c in columns])
            insert_query = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"

            batch_data = []
            for row in rows:
                batch_data.append(list(row))
            
            pg_cursor.executemany(insert_query, batch_data)
            print(f"  - Migrated {len(rows)} rows to {table}")

        pg_conn.commit()
        print("🎉 Migration completed 100000% successfully!")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        pg_conn.rollback()
    finally:
        sqlite_conn.close()
        pg_conn.close()

if __name__ == "__main__":
    migrate()
