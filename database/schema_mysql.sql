-- Real-Time Phishing Detection System - MySQL Database Schema
-- Supports: Users, Messages, URLs, Domains, Feedback, Model Versions, Auto-Training

-- Create Database (run this first)
-- CREATE DATABASE IF NOT EXISTS phishing_detection CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
-- USE phishing_detection;

-- Users Table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,  -- Hashed password
    role VARCHAR(50) DEFAULT 'user',  -- 'user', 'admin', 'soc_analyst'
    is_active INT DEFAULT 1,  -- 1=active, 0=inactive
    last_login TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_email (email),
    INDEX idx_email_password (email, password_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Messages Table (Emails, SMS, Chat)
CREATE TABLE IF NOT EXISTS messages (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    channel VARCHAR(50) NOT NULL,  -- 'email', 'sms', 'chat', 'browser'
    content TEXT NOT NULL,
    subject VARCHAR(500),
    sender VARCHAR(255),
    recipient VARCHAR(255),
    detected_label INT DEFAULT 0,  -- 0=legitimate, 1=phishing
    confidence_score FLOAT DEFAULT 0.0,
    nlp_score FLOAT,
    adversarial_score FLOAT,
    risk_factors TEXT,  -- JSON array of risk factors
    explainable_reasons TEXT,  -- JSON object with explanations
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_user_id (user_id),
    INDEX idx_detected_label (detected_label),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Domains Table (must be created before URLs due to foreign key)
CREATE TABLE IF NOT EXISTS domains (
    id INT AUTO_INCREMENT PRIMARY KEY,
    domain_name VARCHAR(255) UNIQUE NOT NULL,
    tld VARCHAR(50),
    registration_date DATE,
    expiration_date DATE,
    registrar VARCHAR(255),
    whois_data TEXT,  -- JSON object
    ssl_fingerprint VARCHAR(255),
    ssl_valid BOOLEAN DEFAULT FALSE,
    trust_score FLOAT DEFAULT 0.0,
    gnn_cluster_id INT,
    is_phishing INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_domain_name (domain_name),
    INDEX idx_is_phishing (is_phishing)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- URLs Table
CREATE TABLE IF NOT EXISTS urls (
    id INT AUTO_INCREMENT PRIMARY KEY,
    raw_url TEXT NOT NULL,
    expanded_url TEXT,
    final_url TEXT,
    domain_id INT,
    risk_score FLOAT DEFAULT 0.0,
    redirect_depth INT DEFAULT 0,
    redirect_chain TEXT,  -- JSON array of redirects
    gnn_score FLOAT,
    cnn_score FLOAT,
    is_phishing INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (domain_id) REFERENCES domains(id) ON DELETE SET NULL,
    INDEX idx_domain_id (domain_id),
    INDEX idx_is_phishing (is_phishing)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Domain Relationships (Graph Edges)
CREATE TABLE IF NOT EXISTS domain_relationships (
    id INT AUTO_INCREMENT PRIMARY KEY,
    source_domain_id INT NOT NULL,
    target_domain_id INT NOT NULL,
    relationship_type VARCHAR(50),  -- 'redirect', 'subdomain', 'similar', 'ip_shared'
    weight FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_domain_id) REFERENCES domains(id) ON DELETE CASCADE,
    FOREIGN KEY (target_domain_id) REFERENCES domains(id) ON DELETE CASCADE,
    UNIQUE KEY unique_relationship (source_domain_id, target_domain_id, relationship_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Feedback Logs (For Auto-Training)
CREATE TABLE IF NOT EXISTS feedback_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    item_type VARCHAR(50) NOT NULL,  -- 'message', 'url', 'domain'
    item_id INT NOT NULL,
    feedback_type VARCHAR(50) NOT NULL,  -- 'true_positive', 'false_positive', 'true_negative', 'false_negative'
    correct_label INT,  -- 0=legitimate, 1=phishing
    used_for_training INT DEFAULT 0,  -- 0=not used, 1=used
    training_batch_id INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_used_for_training (used_for_training),
    INDEX idx_item (item_type, item_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Model Versions (For Model Management)
CREATE TABLE IF NOT EXISTS model_versions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,  -- 'nlp', 'cnn', 'gnn', 'adversarial', 'ensemble'
    version VARCHAR(50) NOT NULL,
    accuracy FLOAT,
    `precision` FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    training_samples INT,
    deployed_at TIMESTAMP NULL,
    is_active INT DEFAULT 0,
    model_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_model_version (model_type, version),
    INDEX idx_is_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Training Batches (For Auto-Training Tracking)
CREATE TABLE IF NOT EXISTS training_batches (
    id INT AUTO_INCREMENT PRIMARY KEY,
    batch_type VARCHAR(50) NOT NULL,  -- 'initial', 'incremental', 'retrain'
    model_type VARCHAR(50) NOT NULL,
    samples_count INT,
    csv_samples INT DEFAULT 0,
    db_samples INT DEFAULT 0,
    accuracy_before FLOAT,
    accuracy_after FLOAT,
    status VARCHAR(50) DEFAULT 'pending',  -- 'pending', 'training', 'completed', 'failed'
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Predictions Log (All Predictions Stored)
CREATE TABLE IF NOT EXISTS predictions_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    item_type VARCHAR(50) NOT NULL,  -- 'message', 'url', 'domain'
    item_id INT NOT NULL,
    model_type VARCHAR(50),  -- 'nlp', 'cnn', 'gnn', 'ensemble'
    prediction INT,  -- 0=legitimate, 1=phishing
    probability FLOAT,
    processing_time_ms FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_item (item_type, item_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Threat Intelligence (IOCs)
CREATE TABLE IF NOT EXISTS threat_intelligence (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ioc_type VARCHAR(50) NOT NULL,  -- 'url', 'domain', 'ip', 'hash'
    ioc_value TEXT NOT NULL,
    threat_type VARCHAR(100),
    severity VARCHAR(50),  -- 'low', 'medium', 'high', 'critical'
    source VARCHAR(100),  -- 'user_report', 'sandbox', 'ti_feed'
    first_seen TIMESTAMP NULL,
    last_seen TIMESTAMP NULL,
    is_active INT DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_ioc (ioc_type, ioc_value(255))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

