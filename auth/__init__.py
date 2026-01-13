"""
Authentication module for Real-Time Phishing Detection System
"""

from .auth_manager import AuthManager, verify_password, hash_password
from .jwt_handler import create_access_token, verify_token

__all__ = [
    'AuthManager',
    'verify_password',
    'hash_password',
    'create_access_token',
    'verify_token'
]

