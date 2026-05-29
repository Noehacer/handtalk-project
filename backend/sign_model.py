import mediapipe as mp
import cv2
import numpy as np
import config
from logger import get_logger

log = get_logger(__name__)

mp_hands = mp.solutions.hands
_hands_static = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
)

_model       = None
_class_names = None


def _load_model():
    global _model, _class_names
    if not config.STATIC_MODEL_PATH.exists():
        log.warning("Modelo estático no encontrado — usando heurística de respaldo. Ejecuta train_model.py para entrenarlo.")
        return
    try:
        import tensorflow as tf
        _model = tf.keras.models.load_model(str(config.STATIC_MODEL_PATH))
        if config.STATIC_LABELS_PATH.exists():
            _class_names = list(np.load(str(config.STATIC_LABELS_PATH), allow_pickle=True))
        log.info(f"Modelo estático cargado: {len(_class_names)} clases.")
    except Exception as exc:
        log.error(f"No se pudo cargar el modelo estático: {exc}")


def extract_landmarks(image: np.ndarray):
    """Extrae los 21 landmarks de MediaPipe. Retorna array (21,3) o None."""
    rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = _hands_static.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    hand = result.multi_hand_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Centra en la muñeca (landmark 0) y escala por la distancia
    muñeca → MCP dedo medio (landmark 9).
    Retorna array aplanado (63,).
    """
    pts = landmarks.copy()
    pts -= pts[0]
    scale = np.linalg.norm(pts[9])
    pts /= max(float(scale), 1e-6)
    return pts.flatten()


def predict_sign(image: np.ndarray) -> dict:
    """
    Predice la seña en una imagen.
    Retorna {'prediction': str, 'confidence': float}.
    """
    landmarks = extract_landmarks(image)

    if landmarks is None:
        log.debug("Sin mano detectada en la imagen.")
        return {"prediction": "No detectado", "confidence": 0.0}

    normalized = normalize_landmarks(landmarks)

    if _model is not None and _class_names is not None:
        probs      = _model.predict(normalized.reshape(1, -1), verbose=0)[0]
        idx        = int(np.argmax(probs))
        confidence = float(probs[idx])
        label      = _class_names[idx] if confidence >= config.STATIC_CONFIDENCE_THRESHOLD else "No reconocido"
        log.info(f"Predicción estática: {label} ({confidence:.0%})")
        return {"prediction": label, "confidence": round(confidence, 4)}

    # Fallback heurístico hasta que exista un modelo entrenado
    label = "Hola" if landmarks[4][0] < landmarks[8][0] else "Gracias"
    log.debug(f"Heurística de respaldo: {label}")
    return {"prediction": label, "confidence": 0.0}


_load_model()
