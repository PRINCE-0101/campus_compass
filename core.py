"""Shared training/inference contract. Never fit preprocessing on evaluation data."""
import re
import unicodedata
from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
LABELS = ['academics', 'events', 'transport', 'clubs', 'facilities', 'other']
SEED = 42

def clean_text(text):
    text = unicodedata.normalize('NFKC', str(text)).lower()
    text = re.sub(r'https?://\S+', ' url ', text)
    text = re.sub(r'[^\w\s\']', ' ', text)
    return ' '.join(text.split())

def validate_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Please enter a non-empty query.')
    if len(text) > 1000:
        raise ValueError('Please limit the query to 1,000 characters.')
    if not any(c.isalpha() for c in text):
        raise ValueError('Please include words in your query.')
    return text.strip()

def routed_predictions(pipeline, texts, threshold):
    probabilities = pipeline.predict_proba(texts)
    indices = probabilities.argmax(axis=1)
    labels = pipeline.classes_[indices].copy()
    labels[probabilities.max(axis=1) < threshold] = 'other'
    # No recognizable vocabulary is stronger evidence than a class prior.
    unseen = np.asarray(pipeline.named_steps['tfidf'].transform(texts).getnnz(axis=1)) == 0
    labels[unseen] = 'other'
    return labels

class Predictor:
    def __init__(self, path=None):
        path = Path(path) if path else ROOT / 'models' / 'best.joblib'
        if not path.exists():
            raise FileNotFoundError('Model missing. Run: python train.py')
        # Only load trusted joblib files; pickle-based formats can execute code.
        self.bundle = joblib.load(path)
        self.pipeline = self.bundle['pipeline']

    def predict(self, text):
        text = validate_text(text)
        p = self.pipeline.predict_proba([text])[0]
        top = int(p.argmax())
        known = self.pipeline.named_steps['tfidf'].transform([text]).nnz > 0
        low = float(p[top]) < self.bundle['threshold']
        fallback = low or not known
        ranking = sorted(zip(self.pipeline.classes_, p), key=lambda x: -x[1])
        return {'query': text, 'category': 'other' if fallback else str(self.pipeline.classes_[top]),
                'top_model_category': str(self.pipeline.classes_[top]),
                'confidence': float(p[top]), 'fallback': fallback,
                'reason': 'unseen_vocabulary' if not known else 'low_confidence' if low else 'model_prediction',
                'threshold': self.bundle['threshold'], 'model': self.bundle['model'],
                'scores': {str(label): float(score) for label, score in ranking}}
