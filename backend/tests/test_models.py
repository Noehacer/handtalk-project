"""
Tests unitarios para sign_model.py.
Las dependencias ML se mockean en conftest.py.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from sign_model import normalize_landmarks, predict_sign


# ── normalize_landmarks ───────────────────────────────────────────────────────

def _make_landmarks(wrist=(0, 0, 0), mcp9=(1, 0, 0)):
    """Crea 21 landmarks de prueba con posiciones controladas."""
    lm = np.zeros((21, 3))
    lm[0]  = wrist
    lm[9]  = mcp9
    lm[4]  = [0.5, 0.2, 0]  # thumb tip
    lm[8]  = [0.8, 0.5, 0]  # index tip
    return lm


def test_normalize_centers_on_wrist():
    lm     = _make_landmarks(wrist=(0.3, 0.5, 0))
    result = normalize_landmarks(lm)
    pts    = result.reshape(21, 3)
    # Landmark 0 (muñeca) debe quedar en el origen
    np.testing.assert_array_almost_equal(pts[0], [0, 0, 0])


def test_normalize_scale():
    """Landmarks multiplicados por 2 deben dar el mismo resultado normalizado."""
    lm1 = _make_landmarks(wrist=(0, 0, 0), mcp9=(1, 0, 0))
    lm2 = lm1 * 2
    r1 = normalize_landmarks(lm1)
    r2 = normalize_landmarks(lm2)
    np.testing.assert_array_almost_equal(r1, r2)


def test_normalize_output_shape():
    lm     = _make_landmarks()
    result = normalize_landmarks(lm)
    assert result.shape == (63,)


def test_normalize_zero_scale():
    """Si todos los landmarks están en el mismo punto, no debe lanzar excepción."""
    lm     = np.zeros((21, 3))
    result = normalize_landmarks(lm)
    assert result.shape == (63,)
    assert not np.any(np.isnan(result))


# ── predict_sign ──────────────────────────────────────────────────────────────

def test_predict_no_hand(monkeypatch):
    """Cuando MediaPipe no detecta mano, debe retornar 'No detectado'."""
    import sign_model
    monkeypatch.setattr(sign_model, "extract_landmarks", lambda img: None)

    import numpy as np
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = predict_sign(fake_image)

    assert result["prediction"] == "No detectado"
    assert result["confidence"] == 0.0


def test_predict_fallback_heuristic(monkeypatch):
    """Sin modelo entrenado debe usar la heurística thumb vs index."""
    import sign_model
    monkeypatch.setattr(sign_model, "_model", None)
    monkeypatch.setattr(sign_model, "_class_names", None)

    # thumb_tip.x < index_tip.x → "Hola"
    lm = _make_landmarks()
    lm[4][0] = 0.1  # thumb x (menor)
    lm[8][0] = 0.9  # index x (mayor)
    monkeypatch.setattr(sign_model, "extract_landmarks", lambda img: lm)

    import numpy as np
    result = predict_sign(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result["prediction"] == "Hola"


def test_predict_with_model(monkeypatch):
    """Con modelo cargado debe retornar la clase con mayor probabilidad."""
    import numpy as np
    import sign_model
    from unittest.mock import MagicMock

    lm = _make_landmarks()
    monkeypatch.setattr(sign_model, "extract_landmarks", lambda img: lm)

    model_mock = MagicMock()
    model_mock.predict.return_value = np.array([[0.1, 0.8, 0.1]])
    monkeypatch.setattr(sign_model, "_model", model_mock)
    monkeypatch.setattr(sign_model, "_class_names", ["A", "B", "C"])

    result = predict_sign(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result["prediction"] == "B"
    assert abs(result["confidence"] - 0.8) < 1e-4


def test_predict_low_confidence_returns_no_reconocido(monkeypatch):
    """Confianza baja (< umbral) debe retornar 'No reconocido'."""
    import numpy as np
    import sign_model
    from unittest.mock import MagicMock

    lm = _make_landmarks()
    monkeypatch.setattr(sign_model, "extract_landmarks", lambda img: lm)

    model_mock = MagicMock()
    model_mock.predict.return_value = np.array([[0.4, 0.4, 0.2]])
    monkeypatch.setattr(sign_model, "_model", model_mock)
    monkeypatch.setattr(sign_model, "_class_names", ["A", "B", "C"])

    result = predict_sign(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result["prediction"] == "No reconocido"
