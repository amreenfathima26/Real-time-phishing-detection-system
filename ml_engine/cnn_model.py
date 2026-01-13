"""
CNN Model for Webpage/Visual Analysis
Detects brand impersonation and fake login pages
"""

import numpy as np
from typing import Dict, List, Optional
from urllib.parse import urlparse
import re
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available or failed to load. Using fallback CNN model.")

class CNNModel:
    """CNN model for webpage visual/structural analysis"""
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
        
        if TORCH_AVAILABLE:
            self._build_model()
    
    def _build_model(self):
        """Build CNN model architecture"""
        try:
            # Simple CNN for demonstration
            # In production, use pre-trained models like ResNet, EfficientNet
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 16 * 16, 128),
                nn.ReLU(),
                nn.Linear(128, 2),  # Binary classification
                nn.Softmax(dim=1)
            )
            self.model.eval()
            self.is_loaded = True
            print("[CNN MODEL] Initialized successfully")
        except Exception as e:
            print(f"Warning: Could not build CNN model: {e}")
            self.is_loaded = False
    
    def analyze_webpage(self, url: str, html_content: Optional[str] = None,
                       dom_structure: Optional[Dict] = None) -> Dict[str, float]:
        """
        Analyze webpage for phishing indicators
        
        Args:
            url: Webpage URL
            html_content: HTML content (optional)
            dom_structure: DOM structure analysis (optional)
        
        Returns:
            Dict with phishing scores
        """
        # Extract features from URL and available content
        url_features = self._analyze_url_structure(url)
        dom_features = self._analyze_dom_structure(html_content, dom_structure)
        
        # Combine features
        phishing_probability = (
            url_features['suspicious_url'] * 0.4 +
            dom_features['fake_login_form'] * 0.3 +
            dom_features['brand_impersonation'] * 0.3
        )
        
        return {
            'phishing_probability': min(phishing_probability, 1.0),
            'suspicious_url': url_features['suspicious_url'],
            'fake_login_form': dom_features['fake_login_form'],
            'brand_impersonation': dom_features['brand_impersonation'],
            'ssl_issues': url_features.get('ssl_issues', 0.0),
            'lookalike_domain': url_features.get('lookalike_domain', 0.0)
        }
    
    def _analyze_url_structure(self, url: str) -> Dict[str, float]:
        """Analyze URL structure for phishing indicators"""
        features = {
            'suspicious_url': 0.0,
            'ssl_issues': 0.0,
            'lookalike_domain': 0.0
        }
        
        url_lower = url.lower()
        parsed_url = urlparse(url)
        path = parsed_url.path
        
        # 1. GRANULAR KEYWORD DETECTION
        phishing_keywords = ['verify', 'security', 'update', 'account', 'login', 'signin', 'password', 'credential', 'billing', 'secure', 'auth', 'redirect', 'session']
        keyword_score = 0.0
        for kw in phishing_keywords:
            if kw in url_lower:
                keyword_score += 0.25 # Each keyword adds risk
        features['suspicious_url'] += min(keyword_score, 1.0)
        
        # 2. STRUCTURAL RED FLAGS
        # IP address
        if re.search(r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}', url_lower):
            features['suspicious_url'] += 0.7
            
        # URL shorteners
        if re.search(r'bit\.ly|tinyurl|t\.co|goo\.gl|shorturl', url_lower):
            features['suspicious_url'] += 0.4
            
        # Malicious TLDs
        if re.search(r'\.(tk|ml|ga|cf|xyz|top|click|download|monster|account|club|info)', url_lower):
            features['suspicious_url'] += 0.6
            
        # Path length and slashes
        if len(path) > 50:
            features['suspicious_url'] += 0.2
        if path.count('/') > 3:
            features['suspicious_url'] += 0.2
            
        # @ symbol
        if '@' in url_lower:
            features['suspicious_url'] += 0.6
        
        # BRAND IMPERSONATION & HIJACKING CHECK
        brands = ['paypal', 'amazon', 'microsoft', 'apple', 'google', 'facebook', 'twitter', 'netflix', 'chase', 'bank', 'binance', 'coinbase', 'blockchain']
        parsed_url = urlparse(url)
        netloc = parsed_url.netloc.lower()
        domain_parts = netloc.split('.')
        if not domain_parts: return features
        
        domain_main = domain_parts[-2] if len(domain_parts) >= 2 else domain_parts[0]
        tld = domain_parts[-1] if len(domain_parts) >= 2 else ""
        
        # 1. Domain Hijacking / Double Domain Check (e.g., paypal.com.secure.xyz)
        for brand in brands:
            if brand in domain_parts[:-1]: # Brand is in a subdomain or main domain part
                # If it's something like paypal.com.evil.biz
                # The real main domain is 'evil', but 'paypal.com' is in the prefix
                if brand != domain_main:
                    features['lookalike_domain'] = 1.0
                    features['suspicious_url'] = 1.0
                    break

        # 2. Sophisticated Homograph / Confusion Map
        sub_map = {
            'a': ['4', '@', '4', 'ɑ', 'а'], 
            'o': ['0', 'ο', 'о', 'ό'], 
            'i': ['1', 'l', '!', 'і', 'í'], 
            'l': ['1', 'i', 'ǀ'], 
            'e': ['3', 'е', 'é'], 
            's': ['5', '$', 'ƽ'], 
            't': ['7', '†'], 
            'g': ['9', 'ɡ'], 
            'b': ['8', 'Ь'],
            'm': ['rn', 'nn'],
            'w': ['vv', 'v v']
        }

        for brand in brands:
            # Skip fuzzy check if we already flagged as hijacking
            if features['lookalike_domain'] >= 1.0: break
            
            # 2a. Literal check
            if brand in netloc and domain_main != brand:
                features['lookalike_domain'] = 0.95
                features['suspicious_url'] = 1.0
                break
            
            # 2b. Fuzzy/Homograph regex
            pattern = ""
            for char in brand:
                if char in sub_map:
                    options = [char] + sub_map[char]
                    pattern += f"[{''.join(re.escape(o) for o in options)}]"
                else:
                    pattern += re.escape(char)
            
            # Check for the pattern anywhere in the full domain (netloc)
            if re.search(pattern, netloc) and brand not in netloc:
                features['lookalike_domain'] = 1.0
                features['suspicious_url'] = 1.0
                break
        
        # 3. Path Hijacking (e.g., google.com/login/paypal)
        for brand in brands:
            if brand in url_lower and brand != domain_main:
                # If brand is in the path but not the domain
                features['suspicious_url'] = max(features['suspicious_url'], 0.5)
        
        # Check HTTPS
        if not url_lower.startswith('https://'):
            features['ssl_issues'] = 0.8 # Higher penalty for non-HTTPS phishing
        
        features['suspicious_url'] = min(features['suspicious_url'], 1.0)
        return features
        
        # Check HTTPS
        if not url_lower.startswith('https://'):
            features['ssl_issues'] = 0.6
        
        features['suspicious_url'] = min(features['suspicious_url'], 1.0)
        return features
    
    def _analyze_dom_structure(self, html_content: Optional[str],
                              dom_structure: Optional[Dict]) -> Dict[str, float]:
        """Analyze DOM structure for phishing indicators"""
        features = {
            'fake_login_form': 0.0,
            'brand_impersonation': 0.0
        }
        
        if html_content:
            html_lower = html_content.lower()
            
            # Check for login forms
            if re.search(r'<form.*(login|signin|password|username)', html_lower):
                features['fake_login_form'] = 0.5
                
                # Check for suspicious form attributes
                if re.search(r'action=["\'](http://|javascript:)', html_lower):
                    features['fake_login_form'] = 0.9
            
            # Check for brand impersonation
            brands = ['paypal', 'amazon', 'microsoft', 'apple', 'google']
            for brand in brands:
                if brand in html_lower:
                    # Check if it's not the official site
                    if not re.search(rf'{brand}\.com', html_lower):
                        features['brand_impersonation'] = 0.8
                        break
        
        return features
    
    def get_explainable_reasons(self, url: str, html_content: Optional[str] = None) -> Dict[str, any]:
        """Get explainable reasons for prediction"""
        analysis = self.analyze_webpage(url, html_content)
        
        reasons = {}
        
        if analysis['suspicious_url'] > 0.5:
            reasons['suspicious_url'] = f"Suspicious URL structure detected (score: {analysis['suspicious_url']:.2f})"
        
        if analysis['fake_login_form'] > 0.5:
            reasons['fake_login_form'] = f"Potential fake login form detected (score: {analysis['fake_login_form']:.2f})"
        
        if analysis['brand_impersonation'] > 0.5:
            reasons['brand_impersonation'] = f"Possible brand impersonation (score: {analysis['brand_impersonation']:.2f})"
        
        if analysis.get('lookalike_domain', 0) > 0.5:
            reasons['lookalike_domain'] = "Domain appears to be a lookalike of a known brand"
        
        return reasons

