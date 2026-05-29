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


def test_history_is_user_scoped():
    client.post("/auth/register", json={"username": "user_a_hist", "password": "pass1234"})
    client.post("/auth/register", json={"username": "user_b_hist", "password": "pass1234"})
    token_a = client.post("/auth/login", json={"username": "user_a_hist", "password": "pass1234"}).json()["access_token"]
    token_b = client.post("/auth/login", json={"username": "user_b_hist", "password": "pass1234"}).json()["access_token"]
    r_a = client.get("/history", headers={"Authorization": f"Bearer {token_a}"}).json()
    assert r_a["user"] == "user_a_hist"
    r_b = client.get("/history", headers={"Authorization": f"Bearer {token_b}"}).json()
    assert r_b["user"] == "user_b_hist"
