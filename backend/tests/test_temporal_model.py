import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from labels import SEQUENCE_LENGTH


def test_predict_dynamic_returns_dict_without_model():
    from temporal_model import predict_dynamic
    seq = np.zeros((SEQUENCE_LENGTH, 225))
    result = predict_dynamic(seq)
    assert "prediction" in result
    assert "confidence" in result
    assert isinstance(result["confidence"], float)


def test_predict_dynamic_confidence_in_range():
    from temporal_model import predict_dynamic
    seq = np.random.rand(SEQUENCE_LENGTH, 225)
    result = predict_dynamic(seq)
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_dynamic_no_model_returns_unavailable():
    import temporal_model
    original = temporal_model._dynamic_model
    temporal_model._dynamic_model = None
    result = temporal_model.predict_dynamic(np.zeros((SEQUENCE_LENGTH, 225)))
    assert result["prediction"] == "Modelo no disponible"
    assert result["confidence"] == 0.0
    temporal_model._dynamic_model = original


def test_predict_dynamic_confidence_rounded():
    from temporal_model import predict_dynamic
    result = predict_dynamic(np.zeros((SEQUENCE_LENGTH, 225)))
    assert result["confidence"] == round(result["confidence"], 4)
