import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest


def test_normalize_removes_invalid_signs():
    from nlp_pipeline import normalize_signs
    signs = ["HOLA", "NO RECONOCIDO", "GRACIAS", "CAPTURANDO...", "GRACIAS"]
    result = normalize_signs(signs)
    assert result == ["HOLA", "GRACIAS"]


def test_normalize_removes_consecutive_duplicates():
    from nlp_pipeline import normalize_signs
    result = normalize_signs(["YO", "YO", "COMER"])
    assert result == ["YO", "COMER"]


def test_normalize_case_insensitive_dedup():
    from nlp_pipeline import normalize_signs
    result = normalize_signs(["Hola", "HOLA", "gracias"])
    assert result == ["Hola", "gracias"]


def test_translate_empty_list():
    import asyncio
    from nlp_pipeline import translate_to_spanish
    result = asyncio.run(translate_to_spanish([]))
    assert result == ""


def test_translate_single_sign_capitalizes():
    import asyncio
    from nlp_pipeline import translate_to_spanish
    result = asyncio.run(translate_to_spanish(["hola"]))
    assert result == "Hola"


def test_translate_fallback_without_api_key(monkeypatch):
    import asyncio
    import config
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    from nlp_pipeline import translate_to_spanish
    result = asyncio.run(translate_to_spanish(["YO", "COMER"]))
    assert "YO" in result and "COMER" in result
