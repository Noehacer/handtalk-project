import numpy as np
import config
from logger import get_logger
from holistic_model import extract_static
from sign_classifier import predict_static

log = get_logger(__name__)


def extract_landmarks(image):
    """Backward-compatible: returns shape (21,3) or None."""
    features = extract_static(image)
    if features is None:
        return None
    right = features[63:126].reshape(21, 3)
    left  = features[0:63].reshape(21, 3)
    return right if np.any(right != 0) else (left if np.any(left != 0) else None)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    pts = landmarks.copy()
    pts -= pts[0]
    pts /= max(float(np.linalg.norm(pts[9])), 1e-6)
    return pts.flatten()


def predict_sign(image) -> dict:
    features = extract_static(image)
    if features is None:
        log.debug("Sin manos detectadas.")
        return {"prediction": "No detectado", "confidence": 0.0}
    result = predict_static(features)
    log.info(f"Predicción estática: {result['prediction']} ({result['confidence']:.0%})")
    return result
