from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
import numpy as np
from labels import SEQUENCE_LENGTH


@dataclass
class SessionData:
    landmark_buffer: deque = field(default_factory=lambda: deque(maxlen=SEQUENCE_LENGTH))
    sign_buffer: list = field(default_factory=list)
    last_sign: str = ""
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, SessionData] = {}
        self._lock = Lock()

    def _get_or_create(self, session_id: str) -> SessionData:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionData()
            return self._sessions[session_id]

    def add_landmark_frame(self, session_id: str, landmarks: np.ndarray) -> bool:
        data = self._get_or_create(session_id)
        data.landmark_buffer.append(landmarks.copy())
        data.updated_at = datetime.now(timezone.utc)
        return len(data.landmark_buffer) == SEQUENCE_LENGTH

    def get_landmark_buffer(self, session_id: str) -> list:
        data = self._get_or_create(session_id)
        return list(data.landmark_buffer)

    def clear_landmark_buffer(self, session_id: str):
        data = self._get_or_create(session_id)
        data.landmark_buffer.clear()

    def add_sign(self, session_id: str, sign: str):
        data = self._get_or_create(session_id)
        upper = sign.upper()
        if upper != data.last_sign.upper():
            data.sign_buffer.append(sign)
            data.last_sign = sign
        data.updated_at = datetime.now(timezone.utc)

    def get_signs(self, session_id: str) -> list[str]:
        data = self._get_or_create(session_id)
        return list(data.sign_buffer)

    def clear(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)

    def cleanup_stale(self, max_age_seconds: int = 300):
        now = datetime.now(timezone.utc)
        with self._lock:
            stale = [
                sid for sid, d in self._sessions.items()
                if (now - d.updated_at).total_seconds() > max_age_seconds
            ]
            for sid in stale:
                del self._sessions[sid]


session_manager = SessionManager()
