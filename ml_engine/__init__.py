"""
ML Engine for Real-Time Phishing Detection
Contains: NLP, CNN, GNN, and Adversarial Detection Models
"""

from .phishing_detector import PhishingDetector
from .nlp_model import NLPModel
from .cnn_model import CNNModel
from .gnn_model import GNNModel
from .adversarial_detector import AdversarialDetector

__all__ = [
    'PhishingDetector',
    'NLPModel',
    'CNNModel',
    'GNNModel',
    'AdversarialDetector'
]

