"""
Tests unitarios para text_to_sign.py.
Cubre: normalización de texto, búsqueda exacta, fuzzy matching, fallback.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import database
from text_to_sign import _normalize, get_sign_image


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test.db"))
    database.init_db()


# ── _normalize ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("input_, expected", [
    ("HOLA",         "hola"),
    ("Hola!",        "hola"),
    ("á é í ó ú",   "a e i o u"),
    ("  espacios  ", "espacios"),
    ("¡Hola!",       "hola"),
    ("por_favor",    "por_favor"),
    ("GRACIAS",      "gracias"),
    ("mañana",       "manana"),
])
def test_normalize(input_, expected):
    assert _normalize(input_) == expected


# ── get_sign_image — búsqueda exacta ─────────────────────────────────────────

def test_exact_match_found(monkeypatch):
    database.upsert_sign("hola", "hola.png", "image/png", "saludos")
    # No existe el archivo físico, base64=None pero found=True
    result = get_sign_image("hola")
    assert result["found"] is True
    assert result["word"] == "hola"
    assert result["category"] == "saludos"


def test_exact_match_not_found():
    result = get_sign_image("xyz_inexistente_abc")
    assert result["found"] is False
    assert result["path"] is None


# ── Búsqueda normalizada ──────────────────────────────────────────────────────

def test_normalized_match():
    database.upsert_sign("hola", "hola.png")
    # "HOLA!" normaliza a "hola"
    result = get_sign_image("HOLA!")
    assert result["found"] is True
    assert result["word"] == "hola"


def test_accents_stripped():
    database.upsert_sign("manana", "manana.png")
    result = get_sign_image("mañana")
    assert result["found"] is True


# ── Fuzzy matching ────────────────────────────────────────────────────────────

def test_fuzzy_match_typo():
    database.upsert_sign("gracias", "gracias.png")
    # "graciass" debería hacer fuzzy match con "gracias"
    result = get_sign_image("graciass")
    assert result["found"] is True
    assert result["word"] == "gracias"
    assert result["suggestion"] == "gracias"


def test_fuzzy_no_match_too_different():
    database.upsert_sign("hola", "hola.png")
    # "xyzabc" no debe coincidir con nada
    result = get_sign_image("xyzabc")
    assert result["found"] is False


# ── Sin sugerencia cuando la coincidencia es exacta ──────────────────────────

def test_no_suggestion_on_exact_match():
    database.upsert_sign("gracias", "gracias.png")
    result = get_sign_image("gracias")
    assert result["suggestion"] is None


# ── Fallback estático ─────────────────────────────────────────────────────────

def test_fallback_when_db_empty():
    # La BD está vacía; el fallback dict debería responder para "hola"
    result = get_sign_image("hola")
    # found puede ser True (si hay imagen) o False (si no hay archivo)
    # pero no debe lanzar excepción
    assert "found" in result
    assert "word" in result


# ── media_type ────────────────────────────────────────────────────────────────

def test_media_type_returned():
    database.upsert_sign("adios", "adios.gif", "image/gif", "saludos")
    result = get_sign_image("adios")
    assert result["media_type"] == "image/gif"


# ── thumbnail_base64 and has_real_image ───────────────────────────────────────

def test_get_sign_image_returns_thumbnail_key():
    from text_to_sign import get_sign_image
    result = get_sign_image("hola")
    assert "thumbnail_base64" in result


def test_get_sign_image_returns_has_real_image_key():
    from text_to_sign import get_sign_image
    result = get_sign_image("hola")
    assert "has_real_image" in result
    assert isinstance(result["has_real_image"], bool)


def test_get_sign_image_not_found_has_all_keys():
    from text_to_sign import get_sign_image
    result = get_sign_image("palabraquenoexiste_xyz_99")
    assert result["found"] is False
    assert "thumbnail_base64" in result
    assert "has_real_image" in result


def test_invalidate_cache_resets():
    from text_to_sign import invalidate_words_cache, _get_words_cache
    invalidate_words_cache()
    words = _get_words_cache()
    assert isinstance(words, list)
