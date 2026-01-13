"""
Main Phishing Detector
Combines NLP, CNN, GNN, and Adversarial Detection models
"""

from typing import Dict, List, Optional, Any
import time

from .nlp_model import NLPModel
from .cnn_model import CNNModel
from .gnn_model import GNNModel
from .adversarial_detector import AdversarialDetector

class PhishingDetector:
    """Main phishing detection engine combining all models"""
    
    def __init__(self):
        """Initialize all detection models"""
        print("[ML ENGINE] Initializing Phishing Detection Engine...")
        
        self.nlp_model = NLPModel()
        self.cnn_model = CNNModel()
        self.gnn_model = GNNModel()
        self.adversarial_detector = AdversarialDetector()
        
        print("[ML ENGINE] Phishing Detection Engine initialized successfully")
    
    def detect_message_phishing(self, content: str, subject: Optional[str] = None,
                               sender: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect phishing in message (email/SMS)
        """
        import re
        
        # NLP Analysis
        nlp_result = self.nlp_model.analyze_text(content, subject)
        nlp_score = nlp_result['phishing_probability']
        
        # Adversarial Detection
        adversarial_result = self.adversarial_detector.detect_ai_generated(content)
        adversarial_score = adversarial_result['ai_probability']
        intent_score = adversarial_result.get('intent_score', 0.0)
        
        # URL EXTRACTION AND ANALYSIS
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls_found = re.findall(url_pattern, content)
        
        max_url_risk = 0.0
        url_risk_factors = []
        for url in urls_found:
            url_detection = self.detect_url_phishing(url)
            if url_detection['risk_score'] > max_url_risk:
                max_url_risk = url_detection['risk_score']
                url_risk_factors = url_detection['risk_factors']
        
        # Combine scores - DYNAMIC WEIGHTS
        if max_url_risk > 0.4:
            confidence_score = (max_url_risk * 0.75) + (nlp_score * 0.2) + (intent_score * 0.05)
        else:
            confidence_score = (nlp_score * 0.6) + (intent_score * 0.3) + (adversarial_score * 0.1)
        
        # ULTRA-ROBUST: MAX RISK OVERRIDE
        # If any single model is extremely confident, the whole thing is phishing
        if nlp_score > 0.8: confidence_score = max(confidence_score, 0.95)
        if intent_score > 0.8: confidence_score = max(confidence_score, 0.95)
        if max_url_risk > 0.8: confidence_score = max(confidence_score, 0.95)

        # Extreme boost for combination of factors
        if nlp_result['urgency_score'] > 0.3 and len(urls_found) > 0:
            confidence_score = min(confidence_score * 1.6, 1.0)
            
        # Determine if phishing - DYNAMIC THRESHOLD
        # High-risk: has URLs or IPs or extreme urgency
        if len(urls_found) > 0 or nlp_result['urgency_score'] > 0.5 or nlp_result['suspicious_keywords'] > 0.5:
            threshold = 0.12 # Very sensitive for high-suspicion
        else:
            threshold = 0.45 # Much higher for plain text/safe-looking messages
            
        is_phishing = confidence_score > threshold
        
        # Risk score (0-1)
        risk_score = confidence_score
        
        # Get explainable reasons
        explainable_reasons = {}
        explainable_reasons.update(self.nlp_model.get_explainable_reasons(content, subject))
        explainable_reasons.update(self.adversarial_detector.get_explainable_reasons(content))
        
        # Risk factors
        risk_factors = []
        risk_factors.extend(nlp_result.get('risk_factors', []))
        if nlp_result['urgency_score'] > 0.3: risk_factors.append("Extreme urgency detected")
        if nlp_result['suspicious_keywords'] > 0.3: risk_factors.append("High-risk phishing keywords")
        if len(urls_found) > 0: risk_factors.append(f"Contains {len(urls_found)} link(s)")
        if max_url_risk > 0.5: risk_factors.append("Malicious link confirmed in message")
        if nlp_score > 0.8: risk_factors.append("NLP model reports extreme phishing probability")
        risk_factors.extend(url_risk_factors[:2])
        
        return {
            'is_phishing': is_phishing,
            'confidence_score': confidence_score,
            'risk_score': risk_score,
            'nlp_score': nlp_score,
            'adversarial_score': adversarial_score,
            'intent_score': intent_score,
            'url_risk_score': max_url_risk,
            'explainable_reasons': explainable_reasons,
            'risk_factors': list(set(risk_factors))
        }
    
    def detect_url_phishing(self, url: str, html_content: Optional[str] = None,
                           redirect_chain: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect phishing in URL/webpage
        
        Args:
            url: URL to analyze
            html_content: HTML content (optional)
            redirect_chain: Redirect chain (optional)
        
        Returns:
            Dict with detection results
        """
        # CNN Analysis (Webpage/Visual)
        cnn_result = self.cnn_model.analyze_webpage(url, html_content)
        cnn_score = cnn_result['phishing_probability']
        
        # GNN Analysis (Domain/Link Intelligence)
        gnn_result = self.gnn_model.analyze_domain(url, redirect_chain)
        gnn_score = gnn_result['phishing_probability']
        
        # Combine scores
        confidence_score = (cnn_score * 0.5) + (gnn_score * 0.5)
        
        # ULTRA-ROBUST: MAX RISK OVERRIDE
        if cnn_score > 0.7: confidence_score = max(confidence_score, 0.9)
        if gnn_score > 0.7: confidence_score = max(confidence_score, 0.9)
        
        # Extreme boost for suspicious TLDs
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.xyz', '.top', '.click', '.download', '.monster', '.account', '.info', '.biz', '.online']
        if any(tld in url.lower() for tld in suspicious_tlds):
            confidence_score = min(confidence_score * 1.6, 1.0)
        
        # Determine if phishing - ULTRA SENSITIVE (0.1)
        is_phishing = confidence_score > 0.1
        
        # Risk score (0-1)
        risk_score = confidence_score
        
        # Get explainable reasons
        explainable_reasons = {}
        explainable_reasons.update(self.cnn_model.get_explainable_reasons(url, html_content))
        explainable_reasons.update(self.gnn_model.get_explainable_reasons(url, redirect_chain))
        
        # Risk factors
        risk_factors = []
        if cnn_result['suspicious_url'] > 0.3: risk_factors.append("Suspicious URL structure")
        if cnn_result['fake_login_form'] > 0.3: risk_factors.append("Visual login form mimicry")
        if cnn_result['brand_impersonation'] > 0.3: risk_factors.append("Brand impersonation detected")
        if gnn_result['suspicious_redirects'] > 0.3: risk_factors.append("Deep/Suspicious redirect chain")
        if gnn_result['cluster_risk'] > 0.3: risk_factors.append("Known malicious infrastructure cluster")
        
        return {
            'is_phishing': is_phishing,
            'confidence_score': confidence_score,
            'risk_score': risk_score,
            'cnn_score': cnn_score,
            'gnn_score': gnn_score,
            'redirect_depth': gnn_result.get('redirect_depth', 0),
            'redirect_chain': redirect_chain or [],
            'explainable_reasons': explainable_reasons,
            'risk_factors': list(set(risk_factors))
        }
    
    def detect_domain_phishing(self, domain: str) -> Dict[str, Any]:
        """
        Detect phishing in domain
        
        Args:
            domain: Domain name to analyze
        
        Returns:
            Dict with detection results
        """
        # Use GNN for domain analysis
        url = f"https://{domain}"
        gnn_result = self.gnn_model.analyze_domain(url)
        
        confidence_score = gnn_result['phishing_probability']
        is_phishing = confidence_score > 0.2
        
        explainable_reasons = self.gnn_model.get_explainable_reasons(url)
        
        risk_factors = []
        if gnn_result['suspicious_domain'] > 0.5:
            risk_factors.append("Suspicious domain structure")
        if gnn_result['cluster_risk'] > 0.5:
            risk_factors.append("Domain in malicious cluster")
        
        return {
            'is_phishing': is_phishing,
            'confidence_score': confidence_score,
            'risk_score': confidence_score,
            'gnn_score': confidence_score,
            'explainable_reasons': explainable_reasons,
            'risk_factors': risk_factors
        }

