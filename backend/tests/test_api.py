"""
Tests de integración para la API de HandTalk.
Las dependencias ML (TF, OpenCV, MediaPipe) se mockean en conftest.py.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# ── General ───────────────────────────────────────────────────────────────────

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "HandTalk" in r.json()["message"]


# ── Auth ──────────────────────────────────────────────────────────────────────

def test_register_new_user():
    r = client.post("/auth/register", json={"username": "testuser_ci", "password": "pass1234"})
    assert r.status_code in (201, 409)


def test_login_wrong_password():
    r = client.post("/auth/login", json={"username": "noexiste", "password": "password_incorrecta"})
    assert r.status_code == 401


def test_login_correct():
    client.post("/auth/register", json={"username": "logintest", "password": "abc12345"})
    r = client.post("/auth/login", json={"username": "logintest", "password": "abc12345"})
    assert r.status_code == 200
    assert "access_token" in r.json()


# ── Historial ─────────────────────────────────────────────────────────────────

def test_history_requires_auth():
    r = client.get("/history")
    assert r.status_code == 401


def test_history_with_token():
    client.post("/auth/register", json={"username": "histuser", "password": "hist1234"})
    token = client.post("/auth/login", json={"username": "histuser", "password": "hist1234"}).json()["access_token"]
    r = client.get("/history", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert "history" in r.json()


# ── Diccionario ───────────────────────────────────────────────────────────────

def test_signs_list():
    r = client.get("/signs")
    assert r.status_code == 200
    body = r.json()
    assert "signs" in body
    assert "total" in body


def test_signs_stats():
    r = client.get("/signs/stats")
    assert r.status_code == 200
    assert "total" in r.json()


def test_text_to_sign_known_word():
    r = client.get("/text-to-sign/hola")
    assert r.status_code == 200
    body = r.json()
    assert "found" in body
    assert "word" in body


def test_text_to_sign_unknown_word():
    r = client.get("/text-to-sign/palabrainexistentexyz")
    assert r.status_code == 200
    assert r.json()["found"] is False


def test_text_to_sign_phrase():
    r = client.get("/text-to-sign-phrase/hola%20gracias")
    assert r.status_code == 200
    body = r.json()
    assert "signs" in body
    assert body["total"] == 2


def test_text_to_sign_phrase_empty():
    r = client.get("/text-to-sign-phrase/%20")
    assert r.status_code == 422


# ── Señas dinámicas ───────────────────────────────────────────────────────────

def test_sequence_predict_empty_buffer():
    client.delete("/predict-sign-sequence/reset")
    r = client.get("/predict-sign-sequence/predict")
    assert r.status_code == 200
    assert "prediction" in r.json()


def test_sequence_reset():
    r = client.delete("/predict-sign-sequence/reset")
    assert r.status_code == 200


def test_nlp_clear_requires_auth():
    r = client.delete("/nlp/clear?session_id=test_session")
    assert r.status_code == 401


def test_nlp_clear_endpoint():
    client.post("/auth/register", json={"username": "nlpuser", "password": "nlp12345"})
    token = client.post("/auth/login", json={"username": "nlpuser", "password": "nlp12345"}).json()["access_token"]
    r = client.delete("/nlp/clear?session_id=test_session", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert "message" in r.json()


def test_phrase_response_has_new_fields():
    r = client.get("/text-to-sign-phrase/hola%20gracias")
    assert r.status_code == 200
    body = r.json()
    assert "found_count" in body
    assert "not_found" in body
    assert body["total"] == 2


def test_sign_image_direct_not_found():
    r = client.get("/signs/palabraquenoexiste_xyz/image")
    assert r.status_code == 404


def test_history_is_user_scoped():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from database import save_translation

    # Register and login two users
    client.post("/auth/register", json={"username": "scope_user_a", "password": "pass1234"})
    client.post("/auth/register", json={"username": "scope_user_b", "password": "pass1234"})
    token_a = client.post("/auth/login", json={"username": "scope_user_a", "password": "pass1234"}).json()["access_token"]
    token_b = client.post("/auth/login", json={"username": "scope_user_b", "password": "pass1234"}).json()["access_token"]

    # Insert translations belonging to each user
    save_translation("sign_to_text", "[test]", "HOLA_USER_A", user_id="scope_user_a")
    save_translation("sign_to_text", "[test]", "ADIOS_USER_B", user_id="scope_user_b")

    # User A should only see their own record
    hist_a = client.get("/history", headers={"Authorization": f"Bearer {token_a}"}).json()
    assert hist_a["user"] == "scope_user_a"
    outputs_a = [h["output"] for h in hist_a["history"]]
    assert "HOLA_USER_A" in outputs_a
    assert "ADIOS_USER_B" not in outputs_a

    # User B should only see their own record
    hist_b = client.get("/history", headers={"Authorization": f"Bearer {token_b}"}).json()
    assert hist_b["user"] == "scope_user_b"
    outputs_b = [h["output"] for h in hist_b["history"]]
    assert "ADIOS_USER_B" in outputs_b
    assert "HOLA_USER_A" not in outputs_b
