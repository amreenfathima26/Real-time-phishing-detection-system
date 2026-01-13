"""
Authentication Manager - CLEAN REBUILD
"""
import bcrypt
from typing import Dict, Optional
from database.database import Database
from auth.jwt_handler import create_access_token

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

class AuthManager:
    """Authentication manager"""
    
    def __init__(self, db: Database):
        self.db = db
    
    def register_user(self, name: str, email: str, password: str, role: str = 'user') -> Dict:
        """Register a new user"""
        try:
            # Check if user exists
            existing = self.db.get_user_by_email(email)
            if existing:
                return {
                    'success': False,
                    'error': 'Email already registered'
                }
            
            # Hash password
            password_hash = hash_password(password)
            
            # Create user
            user_id = self.db.create_user(name, email, password_hash, role)
            
            return {
                'success': True,
                'user_id': user_id,
                'message': 'Registration successful'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def login_user(self, email: str, password: str) -> Dict:
        """Authenticate user login"""
        try:
            # Get user
            user = self.db.get_user_by_email(email)
            if not user:
                return {
                    'success': False,
                    'error': 'Invalid email or password'
                }
            
            # Check if active
            if not user.get('is_active', 1):
                return {
                    'success': False,
                    'error': 'User account is inactive'
                }
            
            # Verify password
            if not verify_password(password, user['password_hash']):
                return {
                    'success': False,
                    'error': 'Invalid email or password'
                }
            
            # Update last login
            self.db.update_last_login(user['id'])
            
            # Generate token
            token = create_access_token(
                user_id=user['id'],
                email=user['email'],
                role=user['role']
            )
            
            return {
                'success': True,
                'user': {
                    'id': user['id'],
                    'name': user['name'],
                    'email': user['email'],
                    'role': user['role']
                },
                'token': token,
                'message': 'Login successful'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

