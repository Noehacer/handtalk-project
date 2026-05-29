"""
Tests unitarios para database.py.
Usa una base de datos temporal por test para garantizar aislamiento.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import database


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Redirige DB_PATH a un archivo temporal y crea las tablas."""
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test.db"))
    database.init_db()


# ── Usuarios ──────────────────────────────────────────────────────────────────

def test_create_user():
    assert database.create_user("alice", "hashed") is True


def test_create_duplicate_user():
    database.create_user("bob", "hashed")
    assert database.create_user("bob", "hashed2") is False


def test_get_user_exists():
    database.create_user("charlie", "abc")
    user = database.get_user("charlie")
    assert user is not None
    assert user["username"] == "charlie"
    assert user["hashed_password"] == "abc"


def test_get_user_not_found():
    assert database.get_user("noexiste") is None


# ── Traducciones ──────────────────────────────────────────────────────────────

def test_save_and_get_history():
    database.save_translation("sign_to_text", "[img]", "Hola", 0.95)
    database.save_translation("text_to_sign", "gracias", "/assets/gracias.png")
    history = database.get_history(10)
    assert len(history) == 2
    # Más reciente primero
    assert history[0]["input"] == "gracias"


def test_history_respects_limit():
    for i in range(10):
        database.save_translation("text_to_sign", f"word{i}", f"img{i}")
    assert len(database.get_history(3)) == 3


def test_history_empty():
    assert database.get_history() == []


# ── Señas ─────────────────────────────────────────────────────────────────────

def test_upsert_and_get_sign():
    database.upsert_sign("hola", "hola.png", "image/png", "saludos")
    sign = database.get_sign("hola")
    assert sign is not None
    assert sign["word"] == "hola"
    assert sign["category"] == "saludos"


def test_upsert_updates_existing():
    database.upsert_sign("hola", "hola.png", "image/png", "saludos")
    database.upsert_sign("hola", "hola_v2.png", "image/png", "saludos")
    assert database.get_sign("hola")["filename"] == "hola_v2.png"


def test_get_sign_not_found():
    assert database.get_sign("xyz_inexistente") is None


def test_get_all_signs():
    database.upsert_sign("si",  "si.png",  category="respuestas")
    database.upsert_sign("no",  "no.png",  category="respuestas")
    database.upsert_sign("hola","hola.png",category="saludos")
    all_signs = database.get_all_signs()
    assert len(all_signs) == 3


def test_get_signs_by_category():
    database.upsert_sign("si",   "si.png",   category="respuestas")
    database.upsert_sign("no",   "no.png",   category="respuestas")
    database.upsert_sign("hola", "hola.png", category="saludos")
    respuestas = database.get_all_signs("respuestas")
    assert len(respuestas) == 2
    assert all(s["category"] == "respuestas" for s in respuestas)


def test_delete_sign():
    database.upsert_sign("adios", "adios.png")
    assert database.delete_sign("adios") is True
    assert database.get_sign("adios") is None


def test_delete_nonexistent_sign():
    assert database.delete_sign("noexiste") is False


def test_count_signs():
    assert database.count_signs() == 0
    database.upsert_sign("a", "a.png")
    database.upsert_sign("b", "b.png")
    assert database.count_signs() == 2
