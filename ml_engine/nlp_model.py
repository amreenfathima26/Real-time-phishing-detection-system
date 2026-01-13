"""
NLP Model for Phishing Detection
Uses BERT/RoBERTa for semantic analysis of emails/SMS
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import re

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

class NLPModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
            
    def _load_model(self):
        """Lazy load transformers model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.is_loaded = True
        except Exception as e:
            print(f"Warning: NLP transformer load failed: {e}")
            self.is_loaded = False

    def analyze_text(self, text: str, subject: Optional[str] = None) -> Dict[str, float]:
        """Analyze text for phishing indicators"""
        full_text = f"{subject} {text}" if subject else text
        text_clean = full_text.lower().replace('-', ' ').replace('_', ' ')
        
        # 1. Feature Extraction
        urgency_score = self._detect_urgency(text_clean)
        suspicious_keywords = self._detect_suspicious_keywords(text_clean)
        
        # Check for links/URLs in text
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls_found = re.findall(url_pattern, full_text)
        link_density_score = min(len(urls_found) / 2.0, 1.0)
        
        # IP addresses in text are very suspicious
        ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        has_ip = 1.0 if re.search(ip_pattern, full_text) else 0.0
        
        # 2. Semantic Analysis
        if self.is_loaded and TRANSFORMERS_AVAILABLE:
            semantic_score = self._transformer_analysis(text_clean)
        else:
            semantic_score = self._rule_based_semantic_analysis(text_clean)
            
        # 3. Trust Signal Detection (Reduced impact for ultra-robust)
        trust_patterns = [
            r'best regards', r'sincerely', r'thank you', r'thanks', r'cheers',
            r'meeting', r'lunch', r'coffee', r'tomorrow', r'yesterday', r'weekend', r'sync'
        ]
        trust_count = sum(1 for p in trust_patterns if re.search(p, text_clean))
        trust_score = min(trust_count / 5.0, 0.2) # Further reduced

        # 4. Final Probability Calculation
        base_probability = (
            urgency_score * 0.40 +
            suspicious_keywords * 0.40 +
            link_density_score * 0.15 +
            has_ip * 0.05
        )
        
        if semantic_score > 0.4:
            base_probability = max(base_probability, semantic_score)
        
        # Reduce the "safety" effect of trust signals
        phishing_probability = base_probability * (1.0 - (trust_score * 0.1))
        
        # ULTRA-ROBUST: Force high score for critical threats
        if has_ip > 0.5: phishing_probability = max(phishing_probability, 0.9)
        if (urgency_score > 0.4 and suspicious_keywords > 0.4): phishing_probability = max(phishing_probability, 0.9)
        if len(urls_found) > 1 and urgency_score > 0.3: phishing_probability = max(phishing_probability, 0.8)

        phishing_probability = max(0.01, min(0.99, phishing_probability))
        
        return {
            'phishing_probability': float(phishing_probability),
            'urgency_score': float(urgency_score),
            'suspicious_keywords': float(suspicious_keywords),
            'trust_score': float(trust_score),
            'semantic_score': float(semantic_score),
            'num_urls': len(urls_found)
        }

    def _detect_urgency(self, text: str) -> float:
        # Action + Time combinations
        urgent_combinations = [
            ('verify', 'now'), ('update', 'immediately'), ('action', 'required'),
            ('account', 'suspended'), ('payment', 'failed'), ('unauthorized', 'access'),
            ('password', 'expired'), ('security', 'alert'), ('confirm', 'identity')
        ]
        
        base_score = 0.0
        words = text.split()
        for w1, w2 in urgent_combinations:
            if w1 in text and w2 in text:
                base_score += 0.4
        
        # Individual high pressure words
        high_pressure = ['urgent', 'immediate', 'asap', 'hurry', 'limited time', 'warning', 'danger']
        for p in high_pressure:
            if p in text: base_score += 0.2
                
        return min(base_score, 1.0)

    def _detect_suspicious_keywords(self, text: str) -> float:
        # Aggressive keywords
        v_high = ['password', 'credential', 'social security', 'login', 'confirm', 'update your', 'verify your']
        high = ['account', 'details', 'billing', 'payment', 'invoice', 'credit card', 'banking']
        
        score = 0.0
        for p in v_high:
            if p in text: score += 0.5
        for p in high:
            if p in text: score += 0.25
        
        brands = ['paypal', 'amazon', 'microsoft', 'apple', 'google', 'bank', 'irs', 'netflix', 'chase', 'wellsfargo']
        for b in brands:
            if b in text: score += 0.2
            
        return min(score, 1.0)

    def _check_grammar_quality(self, text: str) -> float:
        return 1.0

    def _rule_based_semantic_analysis(self, text: str) -> float:
        patterns = ['account locked', 'payment failed', 'verify identity', 'unauthorized access', 'suspicious activity', 
                    'verify account', 'unusual login', 'limited access', 'action needed']
        if any(p in text for p in patterns):
            return 1.0
        return 0.0

    def _transformer_analysis(self, text: str) -> float:
        """Perform transformer-based classification if model is loaded"""
        if not self.is_loaded or self.model is None:
            return 0.0
            
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                # Assuming index 1 is phishing for a standard classifier
                return float(scores[0][1].item())
        except Exception:
            return 0.0

    def get_explainable_reasons(self, text: str, subject: Optional[str] = None) -> Dict[str, str]:
        analysis = self.analyze_text(text, subject)
        reasons = {}
        if analysis['phishing_probability'] > 0.2:
            if analysis['urgency_score'] > 0.3: reasons['urgency'] = "High pressure/urgency language detected"
            if analysis['suspicious_keywords'] > 0.3: reasons['keywords'] = "Suspicious phishing keywords found"
            if analysis.get('num_urls', 0) > 0: reasons['links'] = f"Contains {analysis['num_urls']} external links"
        return reasons
