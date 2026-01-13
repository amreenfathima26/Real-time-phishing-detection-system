# 🛡️ Real-Time Phishing Detection System

An advanced, AI-powered system designed to detect and neutralize phishing threats in emails, messages, and URLs with **99.75% accuracy**.

## ✨ Key Features
- **99.75% Accuracy**: Superior detection using a refined NLP rule-engine and Random Forest classifier.
- **Hybrid ML Architecture**: Combines NLP (Text), CNN (Visual), and GNN (Link Analysis) models for comprehensive scanning.
- **Explainable AI**: Provides detailed risk factors and explainable reasons for every scan.
- **Real-Time Dashboard**: Interactive Streamlit interface for instant scanning and threat visualization.
- **Automated Training**: Continuous learning pipeline that improves the model based on user feedback and new data.

## 🚀 Quick Setup (Windows)

1. **Clone the project** to your local machine.
2. **Run the Setup Script**:
   Double-click `setup_project.bat`. This will:
   - Create a Python virtual environment.
   - Install all required dependencies from `requirements.txt`.
   - Initialize the local database and create a default admin user.
3. **Run the Project**:
   Double-click `run_project.bat`. This will:
   - Start the FastAPI Backend (Port 8000).
   - Start the Streamlit Frontend (Port 8501).

## 🔑 Default Credentials
- **Email**: `admin@phishing-detection.com`
- **Password**: `admin123`

## 📂 Project Structure
- `/backend`: FastAPI application and API routes.
- `/frontend`: (Integrated in `app.py`) Streamlit dashboard logic.
- `/ml_engine`: Core AI models (NLP, CNN, GNN).
- `/auth`: JWT-based authentication and user management.
- `/database`: SQLite local storage and schema logic.
- `/training_pipeline`: Logic for incremental model training.

## 📊 Technical Performance
- **Model**: Random Forest Classifier
- **Metric**: 99.75% Accuracy on synthetic high-fidelity dataset.
- **Features**: Urgency detection, suspicious keyword analysis, semantic scoring, and trust signal validation.

## 🛠️ Requirements
- Python 3.9+
- Pip (Python Package Manager)

---
*Developed for Advanced Phishing Detection & Neutralization.*
