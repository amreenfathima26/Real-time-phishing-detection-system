"""
FastAPI Backend - CLEAN REBUILD
Real-Time Phishing Detection API
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from starlette.requests import Request
from datetime import datetime
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from database.database import get_db, Database
from auth.auth_manager import AuthManager
from auth.jwt_handler import verify_token, get_current_user
from ml_engine.phishing_detector import PhishingDetector

# Initialize FastAPI
app = FastAPI(
    title="Real-Time Phishing Detection API",
    description="AI/ML-Based Phishing Detection System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize ML Engine (lazy loading)
_phishing_detector: Optional[PhishingDetector] = None

def get_phishing_detector() -> PhishingDetector:
    """Get or initialize phishing detector"""
    global _phishing_detector
    if _phishing_detector is None:
        _phishing_detector = PhishingDetector()
    return _phishing_detector

# Request Models
class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str = "user"

class LoginRequest(BaseModel):
    email: str
    password: str

class MessageRequest(BaseModel):
    content: str
    subject: Optional[str] = None
    sender: Optional[str] = None
    recipient: Optional[str] = None
    channel: str = "email"
    user_id: Optional[int] = None

class URLRequest(BaseModel):
    url: str
    user_id: Optional[int] = None

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Real-Time Phishing Detection API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/auth/register")
async def register(request: RegisterRequest):
    """Register a new user"""
    try:
        db = get_db()
        auth_manager = AuthManager(db)
        result = auth_manager.register_user(
            name=request.name,
            email=request.email,
            password=request.password,
            role=request.role if request.role in ['user', 'admin'] else 'user'
        )
        
        if result.get('success'):
            return {
                "success": True,
                "user_id": result.get('user_id'),
                "message": result.get('message')
            }
        else:
            raise HTTPException(status_code=400, detail=result.get('error', 'Registration failed'))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """User login"""
    try:
        db = get_db()
        auth_manager = AuthManager(db)
        result = auth_manager.login_user(request.email, request.password)
        
        if result.get('success'):
            return {
                "success": True,
                "user": result.get('user'),
                "token": result.get('token'),
                "message": result.get('message')
            }
        else:
            raise HTTPException(status_code=401, detail=result.get('error', 'Login failed'))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@app.get("/api/auth/me")
async def get_current_user_info(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user info"""
    token = credentials.credentials
    user = get_current_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

@app.get("/api/scans/messages")
async def get_user_messages(user_id: Optional[int] = None, limit: int = 50):
    """Get user's message scans"""
    try:
        db = get_db()
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        messages = db.get_user_messages(user_id, limit)
        return {"success": True, "messages": messages, "count": len(messages)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get messages: {str(e)}")

@app.get("/api/scans/urls")
async def get_user_urls(user_id: Optional[int] = None, limit: int = 50):
    """Get URL scans"""
    try:
        db = get_db()
        urls = db.get_user_urls(user_id, limit)
        return {"success": True, "urls": urls, "count": len(urls)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get URLs: {str(e)}")

@app.post("/api/train/start")
async def start_training(http_request: Request):
    """Start auto-training"""
    try:
        from training_pipeline.auto_trainer import AutoTrainer
        db = get_db()
        trainer = AutoTrainer(db)
        
        # Get model type from request body
        try:
            body = await http_request.json()
            model_type = body.get('model_type', 'nlp') if body else 'nlp'
        except:
            model_type = 'nlp'  # default
        
        # Start training
        if model_type == 'all':
            result = trainer.train_all_models()
        else:
            result = trainer.run_auto_training(model_type=model_type)
        
        return {
            "success": True,
            "message": "Training completed",
            "result": result
        }
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}\n{error_detail}")

@app.get("/api/train/status")
async def get_training_status():
    """Get training batch status"""
    try:
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, batch_type, model_type, status, samples_count, 
                       csv_samples, db_samples, accuracy_before, accuracy_after,
                       started_at, completed_at, created_at
                FROM training_batches
                ORDER BY created_at DESC
                LIMIT 50
            """)
            rows = cursor.fetchall()
            batches = [dict(row) for row in rows]
            
            # Convert timestamps to strings
            for batch in batches:
                for key in ['started_at', 'completed_at', 'created_at']:
                    if batch.get(key):
                        batch[key] = str(batch[key])
            
            return {"success": True, "batches": batches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {str(e)}")

@app.get("/api/train/models")
async def get_model_versions():
    """Get model versions"""
    try:
        db = get_db()
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, model_type, version, accuracy, "precision", recall, f1_score,
                       training_samples, model_path, is_active, deployed_at, created_at
                FROM model_versions
                ORDER BY created_at DESC
            """)
            rows = cursor.fetchall()
            versions = [dict(row) for row in rows]
            
            # Convert timestamps to strings
            for version in versions:
                for key in ['deployed_at', 'created_at']:
                    if version.get(key):
                        version[key] = str(version[key])
            
            return {"success": True, "versions": versions}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to get model versions: {str(e)}")

@app.get("/api/statistics")
async def get_statistics():
    """Get system statistics"""
    try:
        db = get_db()
        stats = db.get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@app.post("/api/scan/message")
async def scan_message(request: MessageRequest):
    """Scan message for phishing - WITH ML MODELS"""
    start_time = time.time()
    try:
        db = get_db()
        detector = get_phishing_detector()
        
        # Create message record
        message_id = db.create_message(
            user_id=request.user_id,
            channel=request.channel,
            content=request.content,
            subject=request.subject,
            sender=request.sender,
            recipient=request.recipient
        )
        
        # Run ML detection
        detection_result = detector.detect_message_phishing(
            content=request.content,
            subject=request.subject,
            sender=request.sender
        )
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Update message with prediction results
        db.update_message_prediction(
            message_id=message_id,
            detected_label=1 if detection_result['is_phishing'] else 0,
            confidence_score=detection_result['confidence_score'],
            nlp_score=detection_result.get('nlp_score'),
            adversarial_score=detection_result.get('adversarial_score'),
            risk_factors=detection_result.get('risk_factors', []),
            explainable_reasons=detection_result.get('explainable_reasons', {})
        )
        
        # Return detailed results
        return {
            "success": True,
            "message_id": message_id,
            "is_phishing": detection_result['is_phishing'],
            "confidence_score": round(detection_result['confidence_score'], 4),
            "risk_score": round(detection_result.get('risk_score', detection_result['confidence_score']), 4),
            "nlp_score": round(detection_result.get('nlp_score', 0), 4),
            "adversarial_score": round(detection_result.get('adversarial_score', 0), 4),
            "risk_factors": detection_result.get('risk_factors', []),
            "explainable_reasons": detection_result.get('explainable_reasons', {}),
            "processing_time_ms": round(processing_time_ms, 2),
            "message": "Scan completed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

@app.post("/api/scan/url")
async def scan_url(request: URLRequest):
    """Scan URL for phishing - WITH ML MODELS"""
    start_time = time.time()
    try:
        db = get_db()
        detector = get_phishing_detector()
        
        # Create URL record
        url_id = db.create_url(raw_url=request.url)
        
        # Run ML detection
        detection_result = detector.detect_url_phishing(url=request.url)
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Update URL with prediction results
        db.update_url_prediction(
            url_id=url_id,
            risk_score=detection_result['risk_score'],
            is_phishing=1 if detection_result['is_phishing'] else 0,
            gnn_score=detection_result.get('gnn_score'),
            cnn_score=detection_result.get('cnn_score'),
            redirect_depth=detection_result.get('redirect_depth', 0),
            redirect_chain=detection_result.get('redirect_chain', [])
        )
        
        # Return detailed results
        return {
            "success": True,
            "url_id": url_id,
            "is_phishing": detection_result['is_phishing'],
            "risk_score": round(detection_result['risk_score'], 4),
            "confidence_score": round(detection_result['confidence_score'], 4),
            "gnn_score": round(detection_result.get('gnn_score', 0), 4),
            "cnn_score": round(detection_result.get('cnn_score', 0), 4),
            "redirect_depth": detection_result.get('redirect_depth', 0),
            "risk_factors": detection_result.get('risk_factors', []),
            "explainable_reasons": detection_result.get('explainable_reasons', {}),
            "processing_time_ms": round(processing_time_ms, 2),
            "message": "Scan completed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

