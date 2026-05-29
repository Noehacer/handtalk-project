import numpy as np
import config
from logger import get_logger

log = get_logger(__name__)

_static_model:  object = None
_static_labels: list[str] = []
_temperature:   float = 1.0


def _load():
    global _static_model, _static_labels, _temperature
    if not config.HOLISTIC_STATIC_PATH.exists():
        log.warning("Modelo holístico estático no encontrado. Ejecuta train_holistic_model.py")
        return
    try:
        import tensorflow as tf
        import json
        _static_model  = tf.keras.models.load_model(str(config.HOLISTIC_STATIC_PATH))
        _static_labels = list(np.load(str(config.HOLISTIC_STATIC_LABELS), allow_pickle=True))
        if config.CALIBRATION_PATH.exists():
            cal = json.loads(config.CALIBRATION_PATH.read_text())
            _temperature = float(cal.get("temperature", 1.0))
        log.info(f"Clasificador holístico: {len(_static_labels)} clases, T={_temperature:.3f}")
    except Exception as exc:
        log.error(f"Error cargando clasificador: {exc}")


def predict_static(features: np.ndarray) -> dict:
    """features: array (225,). Returns {'prediction': str, 'confidence': float}."""
    if _static_model is None:
        return {"prediction": "Modelo no disponible", "confidence": 0.0}
    try:
        import tensorflow as tf
        logits     = _static_model(features.reshape(1, -1), training=False).numpy()[0]
        calibrated = tf.nn.softmax(logits / _temperature).numpy()
        idx        = int(np.argmax(calibrated))
        confidence = float(calibrated[idx])
        label      = (_static_labels[idx]
                      if confidence >= config.WS_STATIC_THRESHOLD
                      else "No reconocido")
        return {"prediction": label, "confidence": round(confidence, 4)}
    except Exception as exc:
        log.error(f"Error en predicción estática: {exc}")
        return {"prediction": "Error", "confidence": 0.0}


_load()
