import numpy as np
from logger import get_logger
from holistic_model import extract_video
from temporal_model import predict_dynamic
from session_manager import session_manager
from labels import SEQUENCE_LENGTH

log = get_logger(__name__)

_DEFAULT_SESSION = "default"


def add_frame(image, session_id: str = _DEFAULT_SESSION) -> bool:
    features = extract_video(image)
    if features is None:
        return False
    return session_manager.add_landmark_frame(session_id, features)


def predict_sequence(session_id: str = _DEFAULT_SESSION) -> dict:
    buffer = session_manager.get_landmark_buffer(session_id)
    frames = len(buffer)
    if frames < SEQUENCE_LENGTH:
        return {"prediction": "Capturando...", "confidence": 0.0, "frames": frames}
    sequence = np.array(buffer)
    result   = predict_dynamic(sequence)
    result["frames"] = frames
    log.info(f"Predicción dinámica: {result['prediction']} ({result['confidence']:.0%})")
    return result


def clear_buffer(session_id: str = _DEFAULT_SESSION):
    session_manager.clear_landmark_buffer(session_id)
    log.debug(f"Buffer reiniciado: {session_id}")
