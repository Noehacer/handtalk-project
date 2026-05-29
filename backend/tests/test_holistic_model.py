import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np


def test_extract_returns_none_when_no_hands(monkeypatch):
    import holistic_model
    mock_result = type("R", (), {
        "left_hand_landmarks": None,
        "right_hand_landmarks": None,
        "face_landmarks": None,
    })()
    monkeypatch.setattr(holistic_model._extractor_static._holistic, "process", lambda _: mock_result)
    result = holistic_model.extract_static(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result is None


def test_extract_returns_225_features_with_one_hand(monkeypatch):
    import holistic_model
    lm = type("LM", (), {"x": 0.1, "y": 0.2, "z": 0.0})()
    hand = type("Hand", (), {"landmark": [lm] * 21})()
    mock_result = type("R", (), {
        "left_hand_landmarks": None,
        "right_hand_landmarks": hand,
        "face_landmarks": None,
    })()
    monkeypatch.setattr(holistic_model._extractor_static._holistic, "process", lambda _: mock_result)
    features = holistic_model.extract_static(np.zeros((100, 100, 3), dtype=np.uint8))
    assert features is not None
    assert features.shape == (225,)


def test_normalize_scale_zero_does_not_crash():
    from holistic_model import HolisticExtractor
    ext = HolisticExtractor.__new__(HolisticExtractor)
    hand = type("Hand", (), {"landmark": [type("LM", (), {"x": 0.0, "y": 0.0, "z": 0.0})() for _ in range(21)]})()
    result = ext._hand_to_array(hand)
    assert result.shape == (63,)
    assert not np.any(np.isnan(result))


def test_features_total_length():
    from holistic_model import HolisticExtractor
    ext = HolisticExtractor.__new__(HolisticExtractor)
    lm = type("LM", (), {"x": 0.0, "y": 0.0, "z": 0.0})()
    hand = type("Hand", (), {"landmark": [lm] * 21})()
    left  = ext._hand_to_array(hand)
    right = ext._hand_to_array(hand)
    face  = ext._face_keypoints(None)
    features = np.concatenate([left, right, face])
    assert features.shape == (225,)


def test_none_hand_returns_zeros():
    from holistic_model import HolisticExtractor
    ext = HolisticExtractor.__new__(HolisticExtractor)
    result = ext._hand_to_array(None)
    assert result.shape == (63,)
    assert np.all(result == 0)


def test_none_face_returns_zeros():
    from holistic_model import HolisticExtractor
    ext = HolisticExtractor.__new__(HolisticExtractor)
    result = ext._face_keypoints(None)
    assert result.shape == (99,)
    assert np.all(result == 0)
