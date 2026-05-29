import numpy as np
import config
from labels import SEQUENCE_LENGTH
from logger import get_logger

log = get_logger(__name__)

_dynamic_model:  object = None
_dynamic_labels: list[str] = []
_temperature:    float = 1.0


def _load():
    global _dynamic_model, _dynamic_labels, _temperature
    if not config.HOLISTIC_DYNAMIC_PATH.exists():
        log.warning("Modelo TCN no encontrado. Ejecuta train_holistic_model.py")
        return
    try:
        import tensorflow as tf
        import json
        _dynamic_model  = tf.keras.models.load_model(str(config.HOLISTIC_DYNAMIC_PATH))
        _dynamic_labels = list(np.load(str(config.HOLISTIC_DYNAMIC_LABELS), allow_pickle=True))
        if config.CALIBRATION_PATH.exists():
            cal = json.loads(config.CALIBRATION_PATH.read_text())
            _temperature = float(cal.get("temperature_dynamic", 1.0))
        log.info(f"Modelo TCN: {len(_dynamic_labels)} clases, T={_temperature:.3f}")
    except Exception as exc:
        log.error(f"Error cargando TCN: {exc}")


def predict_dynamic(sequence: np.ndarray) -> dict:
    """sequence: array (SEQUENCE_LENGTH, 225). Returns {'prediction': str, 'confidence': float}."""
    if _dynamic_model is None:
        return {"prediction": "Modelo no disponible", "confidence": 0.0}
    try:
        import tensorflow as tf
        inp        = sequence.reshape(1, SEQUENCE_LENGTH, 225)
        logits     = _dynamic_model(inp, training=False).numpy()[0]
        calibrated = tf.nn.softmax(logits / _temperature).numpy()
        idx        = int(np.argmax(calibrated))
        confidence = float(calibrated[idx])
        label      = (_dynamic_labels[idx]
                      if confidence >= config.WS_DYNAMIC_THRESHOLD
                      else "No reconocido")
        return {"prediction": label, "confidence": round(confidence, 4)}
    except Exception as exc:
        log.error(f"Error en predicción dinámica: {exc}")
        return {"prediction": "Error", "confidence": 0.0}


_load()
