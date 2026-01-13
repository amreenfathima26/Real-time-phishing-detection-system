"""
Adversarial AI Phishing Detector
Detects AI-generated phishing content using LLM embeddings
"""

import numpy as np
from typing import Dict, List, Optional
import re
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers/Torch failed to load. Using fallback adversarial detector.")

class AdversarialDetector:
    """Detects AI-generated phishing content"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load LLM for embedding generation"""
        try:
            # Use a general-purpose model for embeddings
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self.is_loaded = True
            print("[ADVERSARIAL DETECTOR] Loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load adversarial detector: {e}")
            self.is_loaded = False
    
    def detect_ai_generated(self, text: str) -> Dict[str, float]:
        """
        Detect if text is AI-generated phishing content
        
        Returns:
            Dict with AI generation scores
        """
        # Rule-based features
        rule_features = self._rule_based_detection(text)
        
        # Embedding-based features
        if self.is_loaded:
            embedding_features = self._embedding_based_detection(text)
        else:
            embedding_features = {'ai_probability': 0.5}
        
        # Stylometry features
        stylometry_features = self._stylometry_analysis(text)
        
        # Combine features
        ai_probability = (
            rule_features['suspicious_patterns'] * 0.3 +
            embedding_features['ai_probability'] * 0.4 +
            stylometry_features['unnatural_patterns'] * 0.3
        )
        
        return {
            'ai_probability': min(ai_probability, 1.0),
            'suspicious_patterns': rule_features['suspicious_patterns'],
            'unnatural_patterns': stylometry_features['unnatural_patterns'],
            'intent_score': rule_features.get('intent_score', 0.0)
        }
    
    def _rule_based_detection(self, text: str) -> Dict[str, float]:
        """Rule-based detection of AI-generated phishing"""
        features = {
            'suspicious_patterns': 0.0,
            'intent_score': 0.0
        }
        
        text_lower = text.lower()
        
        # Check for AI-generated patterns
        # AI-generated text often has:
        # 1. Overly formal language mixed with urgency
        # 2. Repetitive structures
        # 3. Unnatural phrasing
        
        # Check for repetitive sentence structures
        sentences = text.split('.')
        if len(sentences) > 2:
            sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
            if len(set(sentence_lengths)) < len(sentence_lengths) * 0.5:
                features['suspicious_patterns'] += 0.2
        
        # Check for unnatural phrasing patterns
        unnatural_patterns = [
            r'\b(please (kindly|do|be sure to))',
            r'\b(we (would like to|wish to|are pleased to))',
            r'\b(thank you (for|in advance))',
        ]
        
        for pattern in unnatural_patterns:
            if re.search(pattern, text_lower):
                features['suspicious_patterns'] += 0.15
        
        # Check for credential harvesting intent
        credential_patterns = [
            r'(enter|provide|submit|verify) (your|ur) (password|pin|ssn|account)',
            r'(click|visit|go to) (the|this) (link|website|url)',
            r'(urgent|immediate|asap) (action|verification|response) (required|needed)'
        ]
        
        for pattern in credential_patterns:
            if re.search(pattern, text_lower):
                features['intent_score'] += 0.3
        
        features['suspicious_patterns'] = min(features['suspicious_patterns'], 1.0)
        features['intent_score'] = min(features['intent_score'], 1.0)
        
        return features
    
    def _embedding_based_detection(self, text: str) -> Dict[str, float]:
        """Use LLM embeddings to detect AI-generated content"""
        try:
            # Generate embeddings
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # In production, compare with known AI-generated vs human-written embeddings
            # For now, use a simple heuristic based on embedding characteristics
            
            # AI-generated text often has more uniform embeddings
            embedding_std = float(embeddings.std())
            
            # Lower std = more uniform = potentially AI-generated
            ai_probability = max(0, 1 - embedding_std * 2)
            
            return {'ai_probability': ai_probability}
            
        except Exception as e:
            print(f"Error in embedding-based detection: {e}")
            return {'ai_probability': 0.5}
    
    def _stylometry_analysis(self, text: str) -> Dict[str, float]:
        """Stylometry analysis for unnatural patterns"""
        features = {
            'unnatural_patterns': 0.0
        }
        
        # Check for unusual word frequency distributions
        words = text.lower().split()
        if len(words) < 10:
            return features
        
        # Check for repetitive word usage
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # High frequency of certain words = potentially AI-generated
        max_freq = max(word_freq.values()) if word_freq else 0
        if max_freq > len(words) * 0.1:  # More than 10% of words are the same
            features['unnatural_patterns'] += 0.3
        
        # Check for unusual punctuation patterns
        if text.count('!') > text.count('.') * 0.5:
            features['unnatural_patterns'] += 0.2
        
        # Check for all caps words (except short ones)
        all_caps = re.findall(r'\b[A-Z]{4,}\b', text)
        if len(all_caps) > 3:
            features['unnatural_patterns'] += 0.2
        
        features['unnatural_patterns'] = min(features['unnatural_patterns'], 1.0)
        
        return features
    
    def get_explainable_reasons(self, text: str) -> Dict[str, any]:
        """Get explainable reasons for AI detection"""
        detection = self.detect_ai_generated(text)
        
        reasons = {}
        
        if detection['ai_probability'] > 0.6:
            reasons['ai_generated'] = f"Text appears to be AI-generated (probability: {detection['ai_probability']:.2f})"
        
        if detection['suspicious_patterns'] > 0.5:
            reasons['suspicious_patterns'] = f"Detected suspicious AI-generated patterns (score: {detection['suspicious_patterns']:.2f})"
        
        if detection['intent_score'] > 0.5:
            reasons['phishing_intent'] = f"Strong credential harvesting intent detected (score: {detection['intent_score']:.2f})"
        
        return reasons

