"""
Create default users for testing
"""
from database.database import get_db
from auth.auth_manager import AuthManager

def create_default_users():
    """Create default admin and user accounts"""
    db = get_db()
    auth_manager = AuthManager(db)
    
    users = [
        {"name": "Admin User", "email": "admin@phishing-detection.com", "password": "admin123", "role": "admin"},
        {"name": "Test User", "email": "user@phishing-detection.com", "password": "user123", "role": "user"},
        {"name": "Demo User", "email": "demo@phishing-detection.com", "password": "demo123", "role": "user"}
    ]
    
    for user_data in users:
        result = auth_manager.register_user(**user_data)
        if result.get('success'):
            print(f"Created user: {user_data['email']}")
        else:
            print(f"User {user_data['email']} already exists or error: {result.get('error')}")

if __name__ == "__main__":
    create_default_users()

