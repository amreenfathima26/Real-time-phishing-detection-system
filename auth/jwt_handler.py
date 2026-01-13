"""
JWT Token Handler - CLEAN REBUILD
"""
from datetime import datetime, timedelta
from typing import Optional, Dict
import jwt
from jwt import PyJWTError

# JWT Secret Key (in production, use environment variable)
SECRET_KEY = "phishing-detection-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 * 24 * 60  # 30 days

def create_access_token(user_id: int, email: str, role: str) -> str:
    """Create JWT access token"""
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> Optional[Dict]:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except PyJWTError:
        return None

def get_current_user(token: str) -> Optional[Dict]:
    """Get current user from token"""
    payload = verify_token(token)
    if payload:
        return {
            "id": int(payload.get("sub")),
            "email": payload.get("email"),
            "role": payload.get("role")
        }
    return None

