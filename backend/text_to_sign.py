import re
import base64
import unicodedata
from difflib import get_close_matches
import config
from logger import get_logger
from database import get_sign, get_all_signs, count_signs

log = get_logger(__name__)

_FALLBACK = {
    "hola":      "hola.png",
    "gracias":   "gracias.png",
    "si":        "si.png",
    "no":        "no.png",
    "por favor": "por_favor.png",
    "ayuda":     "ayuda.png",
}

_words_cache: list[str] | None = None
_words_cache_count: int = 0
_words_cache_dirty: bool = False


def _get_words_cache() -> list[str]:
    global _words_cache, _words_cache_count, _words_cache_dirty
    current = count_signs()
    if _words_cache is None or current != _words_cache_count or _words_cache_dirty:
        _words_cache = [s["word"] for s in get_all_signs()]
        _words_cache_count = current
        _words_cache_dirty = False
    return _words_cache


def invalidate_words_cache():
    global _words_cache_dirty
    _words_cache_dirty = True


def _normalize(text: str) -> str:
    """Minúsculas, sin acentos, sin puntuación extra, espacios colapsados."""
    text = text.lower().strip()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _lookup(word: str):
    """
    Busca la seña en 3 pasos:
      1. Exacto en BD
      2. Versión normalizada en BD
      3. Similitud (difflib) con cutoff configurable
    Retorna (registro_bd | None, palabra_encontrada | None).
    """
    sign = get_sign(word)
    if sign:
        return sign, word

    norm = _normalize(word)
    sign = get_sign(norm)
    if sign:
        return sign, norm

    all_words = _get_words_cache()
    if all_words:
        matches = get_close_matches(norm, all_words, n=1, cutoff=config.FUZZY_MATCH_CUTOFF)
        if matches:
            log.debug(f"Fuzzy match: '{word}' → '{matches[0]}'")
            return get_sign(matches[0]), matches[0]

    return None, None


def _encode_file(full_path, media_type: str):
    if not full_path.exists():
        return None
    with open(full_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{media_type};base64,{data}"


def get_sign_image(word: str) -> dict:
    """
    Retorna un dict con:
      found, word, path, base64, media_type, category, suggestion.
    """
    original = word.strip()
    sign, matched_word = _lookup(original)

    if sign is None:
        norm     = _normalize(original)
        filename = _FALLBACK.get(norm) or _FALLBACK.get(original.lower())
        if filename:
            path      = f"/assets/{filename}"
            full_path = config.ASSETS_DIR / filename
            b64       = _encode_file(full_path, "image/png")
            found     = b64 is not None
            log.debug(f"Fallback para '{original}': {filename} (found={found})")
            return {"found": found, "word": original, "path": path,
                    "base64": b64, "media_type": "image/png",
                    "category": None, "suggestion": None}

        log.info(f"Seña no encontrada: '{original}'")
        return {"found": False, "word": original, "path": None,
                "base64": None, "media_type": None,
                "category": None, "suggestion": None}

    media_type = sign.get("media_type", "image/png")
    filename   = sign["filename"]
    path       = f"/assets/{filename}" if filename else None
    b64        = _encode_file(config.ASSETS_DIR / filename, media_type) if filename else None
    suggestion = matched_word if _normalize(matched_word) != _normalize(original) else None

    log.debug(f"Seña encontrada: '{original}' → '{matched_word}' ({media_type})")
    return {"found": True, "word": matched_word, "path": path,
            "base64": b64, "media_type": media_type,
            "category": sign.get("category"), "suggestion": suggestion}
