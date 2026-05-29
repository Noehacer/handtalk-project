import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from session_manager import SessionManager


def test_two_sessions_independent():
    sm = SessionManager()
    sm.add_sign("session_a", "HOLA")
    sm.add_sign("session_b", "GRACIAS")
    assert sm.get_signs("session_a") == ["HOLA"]
    assert sm.get_signs("session_b") == ["GRACIAS"]


def test_no_duplicate_consecutive_signs():
    sm = SessionManager()
    sm.add_sign("s1", "HOLA")
    sm.add_sign("s1", "HOLA")
    sm.add_sign("s1", "GRACIAS")
    assert sm.get_signs("s1") == ["HOLA", "GRACIAS"]


def test_clear_removes_session():
    sm = SessionManager()
    sm.add_sign("s1", "HOLA")
    sm.clear("s1")
    assert sm.get_signs("s1") == []


def test_landmark_buffer_returns_ready_when_full():
    from labels import SEQUENCE_LENGTH
    sm = SessionManager()
    for i in range(SEQUENCE_LENGTH - 1):
        ready = sm.add_landmark_frame("s1", np.zeros(225))
        assert not ready
    ready = sm.add_landmark_frame("s1", np.zeros(225))
    assert ready


def test_clear_landmark_buffer():
    sm = SessionManager()
    sm.add_landmark_frame("s1", np.zeros(225))
    sm.clear_landmark_buffer("s1")
    buf = sm.get_landmark_buffer("s1")
    assert len(buf) == 0


def test_cleanup_stale_removes_old_sessions():
    from datetime import datetime, timezone, timedelta
    sm = SessionManager()
    sm.add_sign("old_session", "HOLA")
    sm._sessions["old_session"].updated_at = datetime.now(timezone.utc) - timedelta(seconds=400)
    sm.cleanup_stale(max_age_seconds=300)
    assert "old_session" not in sm._sessions
