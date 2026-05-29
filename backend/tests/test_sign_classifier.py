import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np


def test_predict_static_returns_dict_without_model():
    from sign_classifier import predict_static
    result = predict_static(np.zeros(225))
    assert "prediction" in result
    assert "confidence" in result
    assert isinstance(result["confidence"], float)


def test_predict_static_confidence_in_range():
    from sign_classifier import predict_static
    result = predict_static(np.random.rand(225))
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_static_no_model_returns_unavailable():
    import sign_classifier
    original = sign_classifier._static_model
    sign_classifier._static_model = None
    result = sign_classifier.predict_static(np.zeros(225))
    assert result["prediction"] == "Modelo no disponible"
    assert result["confidence"] == 0.0
    sign_classifier._static_model = original


def test_predict_static_confidence_is_rounded():
    from sign_classifier import predict_static
    result = predict_static(np.zeros(225))
    # confidence should be rounded to 4 decimal places or 0.0
    assert result["confidence"] == round(result["confidence"], 4)
