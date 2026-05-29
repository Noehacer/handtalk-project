import re
import base64
import unicodedata
from io import BytesIO
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

_words_cache:        list[str] | None = None
_words_cache_count:  int = 0
_words_cache_dirty:  bool = False
_semantic_model      = None
_sign_embeddings     = None
_sign_words_indexed: list[str] = []


def _get_words_cache() -> list[str]:
    global _words_cache, _words_cache_count, _words_cache_dirty
    current = count_signs()
    if _words_cache is None or current != _words_cache_count or _words_cache_dirty:
        _words_cache        = [s["word"] for s in get_all_signs()]
        _words_cache_count  = current
        _words_cache_dirty  = False
    return _words_cache


def invalidate_words_cache():
    global _words_cache_dirty
    _words_cache_dirty = True


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _spellcheck(word: str) -> str:
    try:
        from spellchecker import SpellChecker
        spell     = SpellChecker(language="es")
        corrected = spell.correction(word)
        return corrected if corrected else word
    except Exception:
        return word


def _semantic_search(query: str) -> str | None:
    global _semantic_model, _sign_embeddings, _sign_words_indexed
    try:
        from sentence_transformers import SentenceTransformer, util
        if _semantic_model is None:
            _semantic_model = SentenceTransformer(config.SEMANTIC_MODEL)
            log.info("Modelo semántico cargado.")
        all_signs = get_all_signs()
        words     = [s["word"] for s in all_signs]
        if words != _sign_words_indexed:
            _sign_embeddings    = _semantic_model.encode(words, convert_to_tensor=True)
            _sign_words_indexed = words
        query_emb  = _semantic_model.encode(query, convert_to_tensor=True)
        scores     = util.cos_sim(query_emb, _sign_embeddings)[0]
        best_idx   = int(scores.argmax())
        best_score = float(scores[best_idx])
        if best_score >= config.SEMANTIC_THRESHOLD:
            log.debug(f"Semántica: '{query}' → '{words[best_idx]}' ({best_score:.2f})")
            return words[best_idx]
    except Exception as exc:
        log.warning(f"Búsqueda semántica no disponible: {exc}")
    return None


def _lookup(word: str):
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
            log.debug(f"Fuzzy: '{word}' → '{matches[0]}'")
            return get_sign(matches[0]), matches[0]
    found = _semantic_search(norm)
    if found:
        return get_sign(found), found
    return None, None


def _make_thumbnail(full_path, media_type: str) -> str | None:
    try:
        from PIL import Image
        img = Image.open(full_path)
        img.thumbnail((120, 120))
        buf = BytesIO()
        fmt = "GIF" if media_type == "image/gif" else "PNG"
        img.save(buf, format=fmt)
        data = base64.b64encode(buf.getvalue()).decode()
        return f"data:{media_type};base64,{data}"
    except Exception:
        return None


def _encode_file(full_path, media_type: str) -> str | None:
    try:
        with open(full_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:{media_type};base64,{data}"
    except Exception:
        return None


def get_sign_image(word: str) -> dict:
    original = word.strip()
    sign, matched_word = _lookup(original)

    if sign is None:
        corrected = _spellcheck(_normalize(original))
        if corrected and corrected != _normalize(original):
            sign, matched_word = _lookup(corrected)

    if sign is None:
        norm     = _normalize(original)
        filename = _FALLBACK.get(norm) or _FALLBACK.get(original.lower())
        if filename:
            full_path = config.ASSETS_DIR / filename
            b64       = _encode_file(full_path, "image/png")
            thumb     = _make_thumbnail(full_path, "image/png") if full_path.exists() else None
            return {"found": b64 is not None, "word": original,
                    "path": f"/assets/{filename}",
                    "base64": b64, "thumbnail_base64": thumb,
                    "media_type": "image/png", "category": None,
                    "suggestion": None, "has_real_image": False}
        return {"found": False, "word": original, "path": None,
                "base64": None, "thumbnail_base64": None, "media_type": None,
                "category": None, "suggestion": None, "has_real_image": False}

    media_type = sign.get("media_type", "image/png")
    filename   = sign["filename"]
    full_path  = config.ASSETS_DIR / filename if filename else None
    b64        = _encode_file(full_path, media_type) if full_path else None
    thumb      = _make_thumbnail(full_path, media_type) if full_path and full_path.exists() else None
    suggestion = matched_word if _normalize(matched_word) != _normalize(original) else None

    return {"found": True, "word": matched_word,
            "path": f"/assets/{filename}" if filename else None,
            "base64": b64, "thumbnail_base64": thumb,
            "media_type": media_type, "category": sign.get("category"),
            "suggestion": suggestion, "has_real_image": bool(sign.get("has_real_image", 0))}
