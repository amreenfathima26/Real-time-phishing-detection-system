-- Real-Time Phishing Detection System - SQLite Database Schema

-- Users Table
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'user',
    is_active INTEGER DEFAULT 1,
    last_login DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Messages Table
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    channel TEXT NOT NULL,
    content TEXT NOT NULL,
    subject TEXT,
    sender TEXT,
    recipient TEXT,
    detected_label INTEGER DEFAULT 0,
    confidence_score REAL DEFAULT 0.0,
    nlp_score REAL,
    adversarial_score REAL,
    risk_factors TEXT,
    explainable_reasons TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Domains Table
CREATE TABLE IF NOT EXISTS domains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain_name TEXT UNIQUE NOT NULL,
    tld TEXT,
    registration_date DATE,
    expiration_date DATE,
    registrar TEXT,
    whois_data TEXT,
    ssl_fingerprint TEXT,
    ssl_valid BOOLEAN DEFAULT 0,
    trust_score REAL DEFAULT 0.0,
    gnn_cluster_id INTEGER,
    is_phishing INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- URLs Table
CREATE TABLE IF NOT EXISTS urls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_url TEXT NOT NULL,
    expanded_url TEXT,
    final_url TEXT,
    domain_id INTEGER,
    risk_score REAL DEFAULT 0.0,
    redirect_depth INTEGER DEFAULT 0,
    redirect_chain TEXT,
    gnn_score REAL,
    cnn_score REAL,
    is_phishing INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (domain_id) REFERENCES domains(id) ON DELETE SET NULL
);

-- Domain Relationships
CREATE TABLE IF NOT EXISTS domain_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_domain_id INTEGER NOT NULL,
    target_domain_id INTEGER NOT NULL,
    relationship_type TEXT,
    weight REAL DEFAULT 1.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_domain_id) REFERENCES domains(id) ON DELETE CASCADE,
    FOREIGN KEY (target_domain_id) REFERENCES domains(id) ON DELETE CASCADE
);

-- Feedback Logs
CREATE TABLE IF NOT EXISTS feedback_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    item_type TEXT NOT NULL,
    item_id INTEGER NOT NULL,
    feedback_type TEXT NOT NULL,
    correct_label INTEGER,
    used_for_training INTEGER DEFAULT 0,
    training_batch_id INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Model Versions
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type TEXT NOT NULL,
    version TEXT NOT NULL,
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1_score REAL,
    training_samples INTEGER,
    deployed_at DATETIME,
    is_active INTEGER DEFAULT 0,
    model_path TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_type, version)
);

-- Training Batches
CREATE TABLE IF NOT EXISTS training_batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_type TEXT NOT NULL,
    model_type TEXT NOT NULL,
    samples_count INTEGER,
    csv_samples INTEGER DEFAULT 0,
    db_samples INTEGER DEFAULT 0,
    accuracy_before REAL,
    accuracy_after REAL,
    status TEXT DEFAULT 'pending',
    started_at DATETIME,
    completed_at DATETIME,
    error_message TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Predictions Log
CREATE TABLE IF NOT EXISTS predictions_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_type TEXT NOT NULL,
    item_id INTEGER NOT NULL,
    model_type TEXT,
    prediction INTEGER,
    probability REAL,
    processing_time_ms REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Threat Intelligence
CREATE TABLE IF NOT EXISTS threat_intelligence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ioc_type TEXT NOT NULL,
    ioc_value TEXT NOT NULL,
    threat_type TEXT,
    severity TEXT,
    source TEXT,
    first_seen DATETIME,
    last_seen DATETIME,
    is_active INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ioc_type, ioc_value)
);
