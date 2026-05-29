"""
Mocks de dependencias pesadas (TF, OpenCV, MediaPipe) para que
los tests corran en CI sin instalar ~3 GB de librerías ML.
"""

import sys
from unittest.mock import MagicMock
import numpy as np

# ── Mocks ──────────────────────────────────────────────────────────────────────

cv2_mock = MagicMock()
cv2_mock.imdecode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
cv2_mock.cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
cv2_mock.IMREAD_COLOR = 1
cv2_mock.COLOR_BGR2RGB = 4
sys.modules['cv2'] = cv2_mock

mp_mock = MagicMock()

# Mock mp.solutions.hands (legacy)
hands_instance = MagicMock()
hands_instance.process.return_value = MagicMock(multi_hand_landmarks=None)
mp_mock.solutions.hands.Hands.return_value = hands_instance

# Mock mp.solutions.holistic (new)
holistic_instance = MagicMock()
holistic_instance.process.return_value = MagicMock(
    left_hand_landmarks=None,
    right_hand_landmarks=None,
    face_landmarks=None,
)
mp_mock.solutions.holistic.Holistic.return_value = holistic_instance
mp_mock.solutions.drawing_utils = MagicMock()
sys.modules['mediapipe'] = mp_mock

tf_mock = MagicMock()
sys.modules['tensorflow'] = tf_mock
