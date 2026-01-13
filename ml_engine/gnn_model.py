"""
Graph Neural Network (GNN) Model for Domain/Link Intelligence
Analyzes domain relationships, redirect chains, and infrastructure
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
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
    print("Warning: PyTorch not available or failed to load. Using fallback GNN model.")

class GNNModel:
    """Graph Neural Network for domain and link analysis"""
    
    def __init__(self):
        self.domain_graph = {}  # domain -> {neighbors: set, features: dict}
        self.is_loaded = False
        
        if TORCH_AVAILABLE:
            self._build_model()
    
    def _build_model(self):
        """Build GNN model architecture"""
        try:
            # Simple GNN for demonstration
            # In production, use libraries like PyTorch Geometric
            self.is_loaded = True
            print("[GNN MODEL] Initialized successfully")
        except Exception as e:
            print(f"Warning: Could not build GNN model: {e}")
            self.is_loaded = False
    
    def analyze_domain(self, url: str, redirect_chain: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Analyze domain using graph-based intelligence
        
        Args:
            url: URL to analyze
            redirect_chain: List of URLs in redirect chain
        
        Returns:
            Dict with graph-based scores
        """
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split('/')[0]
        
        # Extract domain features
        domain_features = self._extract_domain_features(domain)
        
        # Analyze redirect chain
        redirect_features = self._analyze_redirect_chain(redirect_chain or [url])
        
        # Graph-based analysis
        graph_features = self._graph_analysis(domain)
        
        # Combine features
        phishing_probability = (
            domain_features['suspicious_domain'] * 0.3 +
            redirect_features['suspicious_redirects'] * 0.3 +
            graph_features['cluster_risk'] * 0.4
        )
        
        return {
            'phishing_probability': min(phishing_probability, 1.0),
            'suspicious_domain': domain_features['suspicious_domain'],
            'suspicious_redirects': redirect_features['suspicious_redirects'],
            'cluster_risk': graph_features['cluster_risk'],
            'domain_age_score': domain_features.get('domain_age_score', 0.5),
            'redirect_depth': redirect_features.get('redirect_depth', 0)
        }
    
    def _extract_domain_features(self, domain: str) -> Dict[str, float]:
        """Extract features from domain name"""
        features = {
            'suspicious_domain': 0.0,
            'domain_age_score': 0.5
        }
        
        domain_lower = domain.lower()
        
        # Check for suspicious patterns - MORE AGGRESSIVE
        suspicious_patterns = [
            (r'[0-9]{3,}', 0.3),  # Many numbers
            (r'[a-z]{1,2}[0-9]{2,}', 0.3),  # Short prefix + numbers
            (r'[a-z]+-[a-z]+-[a-z]+', 0.4),  # Multiple hyphens
            (r'\.(tk|ml|ga|cf|xyz|top|click|download|monster|club|account|verify|security)', 0.8),  # Suspicious TLDs - VERY HIGH SCORE
            (r'(login|signin|update|secure|verify|account|webscr|ebayisapi|paypal)', 0.5), # Keywords in domain
        ]
        
        for pattern, score in suspicious_patterns:
            if re.search(pattern, domain_lower):
                features['suspicious_domain'] += score
        
        # Check domain length (very long is often suspicious for DGAs)
        if len(domain) > 30:
            features['suspicious_domain'] += 0.3
        
        # Check for random character sequences
        if self._is_random_sequence(domain):
            features['suspicious_domain'] += 0.4
        
        features['suspicious_domain'] = min(features['suspicious_domain'], 1.0)
        
        return features
    
    def _is_random_sequence(self, domain: str) -> bool:
        """Check if domain appears to be randomly generated"""
        # Simple heuristic: check for patterns that suggest randomness
        # In production, use more sophisticated analysis
        
        # Check for alternating patterns
        if re.search(r'[a-z][0-9][a-z][0-9]', domain.lower()):
            return True
        
        # Check for long sequences without vowels
        if re.search(r'[bcdfghjklmnpqrstvwxyz]{6,}', domain.lower()):
            return True
        
        return False
    
    def _analyze_redirect_chain(self, redirect_chain: List[str]) -> Dict[str, float]:
        """Analyze redirect chain for phishing indicators"""
        features = {
            'suspicious_redirects': 0.0,
            'redirect_depth': len(redirect_chain)
        }
        
        if len(redirect_chain) == 0:
            return features
        
        # Multiple redirects = suspicious
        if len(redirect_chain) > 3:
            features['suspicious_redirects'] += 0.4
        
        # Check for domain changes
        domains = [urlparse(url).netloc for url in redirect_chain if urlparse(url).netloc]
        unique_domains = len(set(domains))
        
        if unique_domains > 2:
            features['suspicious_redirects'] += 0.3
        
        # Check for suspicious TLDs in chain
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.xyz']
        for url in redirect_chain:
            for tld in suspicious_tlds:
                if tld in url.lower():
                    features['suspicious_redirects'] += 0.2
                    break
        
        features['suspicious_redirects'] = min(features['suspicious_redirects'], 1.0)
        
        return features
    
    def _graph_analysis(self, domain: str) -> Dict[str, float]:
        """Perform graph-based analysis of domain relationships"""
        features = {
            'cluster_risk': 0.0
        }
        
        # In production, this would:
        # 1. Build graph of domain relationships
        # 2. Use GNN to propagate information
        # 3. Detect clusters of malicious domains
        # 4. Identify fast-flux hosting patterns
        
        # For now, use simple heuristics
        if domain not in self.domain_graph:
            self.domain_graph[domain] = {
                'neighbors': set(),
                'features': {}
            }
        
        # Check if domain is in a known malicious cluster
        # (In production, this would use actual graph data)
        if self._is_in_malicious_cluster(domain):
            features['cluster_risk'] = 0.8
        
        return features
    
    def _is_in_malicious_cluster(self, domain: str) -> bool:
        """Check if domain is in a known malicious cluster"""
        # Simple heuristic: check neighbors
        if domain in self.domain_graph:
            neighbors = self.domain_graph[domain]['neighbors']
            # If many neighbors are suspicious, this domain is likely suspicious too
            suspicious_neighbors = sum(1 for n in neighbors if self._is_suspicious_domain(n))
            if suspicious_neighbors > 2:
                return True
        return False
    
    def _is_suspicious_domain(self, domain: str) -> bool:
        """Check if domain is suspicious based on patterns"""
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.xyz', '.top']
        return any(tld in domain.lower() for tld in suspicious_tlds)
    
    def add_domain_relationship(self, source_domain: str, target_domain: str,
                               relationship_type: str = 'redirect'):
        """Add relationship between domains to graph"""
        if source_domain not in self.domain_graph:
            self.domain_graph[source_domain] = {
                'neighbors': set(),
                'features': {}
            }
        
        if target_domain not in self.domain_graph:
            self.domain_graph[target_domain] = {
                'neighbors': set(),
                'features': {}
            }
        
        self.domain_graph[source_domain]['neighbors'].add(target_domain)
        self.domain_graph[target_domain]['neighbors'].add(source_domain)
    
    def get_explainable_reasons(self, url: str, redirect_chain: Optional[List[str]] = None) -> Dict[str, any]:
        """Get explainable reasons for prediction"""
        analysis = self.analyze_domain(url, redirect_chain)
        
        reasons = {}
        
        if analysis['suspicious_domain'] > 0.5:
            reasons['suspicious_domain'] = f"Suspicious domain structure detected (score: {analysis['suspicious_domain']:.2f})"
        
        if analysis['suspicious_redirects'] > 0.5:
            reasons['suspicious_redirects'] = f"Multiple suspicious redirects detected (depth: {analysis['redirect_depth']})"
        
        if analysis['cluster_risk'] > 0.5:
            reasons['cluster_risk'] = f"Domain appears in malicious infrastructure cluster (score: {analysis['cluster_risk']:.2f})"
        
        return reasons

