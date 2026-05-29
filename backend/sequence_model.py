import mediapipe as mp
import cv2
import numpy as np
from collections import deque
import config
from logger import get_logger
from sign_model import normalize_landmarks
from labels import SEQUENCE_LENGTH, DYNAMIC_LABELS

log = get_logger(__name__)

mp_hands = mp.solutions.hands
_hands_video = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
)

_seq_model      = None
_seq_class_names = None
_buffer         = deque(maxlen=SEQUENCE_LENGTH)


def _load_sequence_model():
    global _seq_model, _seq_class_names
    if not config.SEQ_MODEL_PATH.exists():
        log.warning("Modelo de secuencias no encontrado — detección dinámica desactivada. Ejecuta train_sequence_model.py para entrenarlo.")
        return
    try:
        import tensorflow as tf
        _seq_model = tf.keras.models.load_model(str(config.SEQ_MODEL_PATH))
        if config.SEQ_LABELS_PATH.exists():
            _seq_class_names = list(np.load(str(config.SEQ_LABELS_PATH), allow_pickle=True))
        log.info(f"Modelo de secuencias cargado: {len(_seq_class_names)} clases.")
    except Exception as exc:
        log.error(f"No se pudo cargar el modelo de secuencias: {exc}")


def _extract_landmarks_video(image: np.ndarray):
    rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = _hands_video.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    hand = result.multi_hand_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])


def add_frame(image: np.ndarray) -> bool:
    """
    Agrega un frame al buffer.
    Retorna True cuando el buffer alcanza SEQUENCE_LENGTH (listo para predecir).
    """
    landmarks = _extract_landmarks_video(image)
    if landmarks is None:
        return False
    _buffer.append(normalize_landmarks(landmarks))
    return len(_buffer) == SEQUENCE_LENGTH


def predict_sequence() -> dict:
    """Predice la seña dinámica con los frames acumulados."""
    if _seq_model is None or _seq_class_names is None:
        return {"prediction": "Modelo no disponible", "confidence": 0.0, "frames": len(_buffer)}

    if len(_buffer) < SEQUENCE_LENGTH:
        return {"prediction": "Capturando...", "confidence": 0.0, "frames": len(_buffer)}

    sequence = np.array(list(_buffer)).reshape(1, SEQUENCE_LENGTH, 63)
    probs      = _seq_model.predict(sequence, verbose=0)[0]
    idx        = int(np.argmax(probs))
    confidence = float(probs[idx])
    label      = _seq_class_names[idx] if confidence >= config.DYNAMIC_CONFIDENCE_THRESHOLD else "No reconocido"
    log.info(f"Predicción dinámica: {label} ({confidence:.0%})")
    return {"prediction": label, "confidence": round(confidence, 4), "frames": len(_buffer)}


def clear_buffer():
    _buffer.clear()
    log.debug("Buffer de secuencia reiniciado.")


_load_sequence_model()
