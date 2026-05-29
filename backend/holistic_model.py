import mediapipe as mp
import cv2
import numpy as np
from logger import get_logger

log = get_logger(__name__)

mp_holistic = mp.solutions.holistic

# 33 face landmarks that capture facial expression relevant for LSM
_FACE_KEYPOINTS = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246,
    362, 382, 381, 380, 374, 373, 390, 249, 263,
    466, 388, 387, 386, 385, 384, 398, 4
]  # 33 points → 99 features


class HolisticExtractor:
    def __init__(self, static_image_mode: bool = True):
        self._holistic = mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl, a, b))
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        except Exception:
            return image

    def _hand_to_array(self, hand_landmarks) -> np.ndarray:
        if hand_landmarks is None:
            return np.zeros(63)
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        pts -= pts[0]
        scale = np.linalg.norm(pts[9])
        pts /= max(float(scale), 1e-6)
        return pts.flatten()

    def _face_keypoints(self, face_landmarks) -> np.ndarray:
        if face_landmarks is None:
            return np.zeros(99)
        pts = np.array([
            [face_landmarks.landmark[i].x,
             face_landmarks.landmark[i].y,
             face_landmarks.landmark[i].z]
            for i in _FACE_KEYPOINTS
        ])
        pts -= pts[0]
        return pts.flatten()

    def extract(self, image: np.ndarray) -> np.ndarray | None:
        preprocessed = self.preprocess_image(image)
        rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
        result = self._holistic.process(rgb)

        has_left  = result.left_hand_landmarks is not None
        has_right = result.right_hand_landmarks is not None
        if not has_left and not has_right:
            return None

        left  = self._hand_to_array(result.left_hand_landmarks)
        right = self._hand_to_array(result.right_hand_landmarks)
        face  = self._face_keypoints(result.face_landmarks)
        return np.concatenate([left, right, face])

    def close(self):
        self._holistic.close()


_extractor_static = HolisticExtractor(static_image_mode=True)
_extractor_video  = HolisticExtractor(static_image_mode=False)


def extract_static(image: np.ndarray) -> np.ndarray | None:
    return _extractor_static.extract(image)


def extract_video(image: np.ndarray) -> np.ndarray | None:
    return _extractor_video.extract(image)
