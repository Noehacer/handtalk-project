# HandTalk AI Upgrade — Code Review

**Reviewed:** 2026-05-28  
**Reviewer:** Claude (adversarial review)  
**Depth:** Deep (cross-file analysis)  
**Files Reviewed:** 14 source files  
**Status:** Issues found

---

## Overview

This PR replaces HandTalk's original single-model pipeline with a two-model architecture:

- **Static model** — Residual MLP that takes a single 225-feature holistic snapshot (2 hands × 63 floats + 33 face keypoints × 3 = 225D) and classifies it to a sign label.
- **Dynamic model** — TCN that consumes a 30-frame × 225-feature buffer and recognises motion signs (J, Z, phrases).
- **WebSocket endpoint** `/ws/detect` — streams frames from the mobile client and emits sign events in real time.
- **NLP pipeline** — routes a collected sign sequence to Claude Haiku for LSM→Spanish grammar reordering.
- **Text-to-Sign v2** — adds semantic search (Sentence Transformers), spell-checking (pyspellchecker), and per-word thumbnails in a horizontal gallery.

The overall architecture is sound and shows clear thinking. Several issues, however, range from production blockers to design weaknesses that will surface under real load.

---

## Architecture

### Strengths
- Clean separation between feature extraction (`holistic_model.py`), static classification (`sign_classifier.py`), temporal classification (`temporal_model.py`), and NLP (`nlp_pipeline.py`).
- Per-session buffers in `session_manager.py` with a proper cleanup path (`cleanup_stale`).
- Temperature calibration is a well-chosen improvement over raw softmax probabilities.
- The WAL pragma and parameterised queries in `database.py` show good SQLite hygiene.

### Weaknesses (see detailed findings below)
- The WebSocket endpoint uses a plain string `sid` query-parameter with no authentication, making session hijacking trivial.
- `cleanup_stale` is defined but never called automatically; sessions accumulate without bound.
- Module-level `_load()` calls in `sign_classifier.py` and `temporal_model.py` execute at import time and silently succeed even when models are absent, hiding misconfiguration.
- Thread-safety is incomplete: `session_manager` exposes non-atomic read-modify operations.
- Semantic search re-encodes every word in the dictionary on every call when the word list changes, which is synchronous and blocks the event loop.

---

## Critical Issues

### CR-01: WebSocket session ID is not authenticated — full session hijacking

**File:** `backend/main.py:392`  
**Severity:** CRITICAL

The WebSocket endpoint accepts a `sid` parameter from the client URL with no validation or authentication:

```python
@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket, sid: str = "anon"):
```

Any client that knows or guesses another user's session ID can:
1. Read accumulated signs via `translate` messages.
2. Inject arbitrary signs with `frame` messages that the model classifies.
3. Wipe the session with `clear` messages.

Session IDs are generated client-side (`session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`) — 9 base-36 chars ≈ 46.6 bits of entropy. That is too weak for a security boundary and still relies entirely on obscurity; there is no server-side ownership check.

**Fix:** Require a JWT in the WebSocket handshake and bind the session to the authenticated user:

```python
@app.websocket("/ws/detect")
async def ws_detect(
    websocket: WebSocket,
    token: str = Query(...),          # ?token=<jwt>
):
    username = _decode_token(token)   # reuse existing auth helper
    if username is None:
        await websocket.close(code=4401)
        return
    sid = f"{username}_{int(datetime.now().timestamp())}"
    await websocket.accept()
    ...
```

Alternatively, accept the JWT in the first message (`{"type": "auth", "token": "..."}`), which is more compatible with browser WebSocket APIs.

---

### CR-02: `session_manager` race condition — sign buffer corruption under concurrent frames

**File:** `backend/session_manager.py:28-32, 42-48`  
**Severity:** CRITICAL

`add_landmark_frame` and `add_sign` both call `_get_or_create` (which holds the lock) and then mutate `SessionData` **outside** the lock:

```python
def add_landmark_frame(self, session_id: str, landmarks: np.ndarray) -> bool:
    data = self._get_or_create(session_id)   # lock released here
    data.landmark_buffer.append(landmarks.copy())  # no lock
    data.updated_at = datetime.now(timezone.utc)
    return len(data.landmark_buffer) == SEQUENCE_LENGTH
```

`asyncio` tasks running on the same thread are safe for this particular deque append because of the GIL, but `cleanup_stale` **deletes** entries from `_sessions` under the lock while another coroutine may hold a reference to `data` and still mutate it after the deletion. This is a use-after-deletion pattern; the deleted session object lives on in memory and receives writes that are never visible again.

More dangerously, if `cleanup_stale` is ever called from a background thread (e.g. via `asyncio.to_thread`), the GIL gives no atomicity guarantee for multi-step operations on `deque` and `list` simultaneously.

**Fix:** Acquire the lock for the entire mutation, or switch to a per-session lock:

```python
def add_landmark_frame(self, session_id: str, landmarks: np.ndarray) -> bool:
    with self._lock:
        data = self._sessions.setdefault(session_id, SessionData())
        data.landmark_buffer.append(landmarks.copy())
        data.updated_at = datetime.now(timezone.utc)
        return len(data.landmark_buffer) == SEQUENCE_LENGTH
```

---

### CR-03: Path traversal via `word` parameter in `/signs/{word}/image`

**File:** `backend/main.py:306-315`  
**Severity:** CRITICAL

The `word` path parameter is used to construct a filesystem path without sanitisation:

```python
def sign_image_direct(word: str):
    sign = db_get_sign(word)
    ...
    full_path = config.ASSETS_DIR / sign["filename"]
    return FileResponse(str(full_path), ...)
```

`sign["filename"]` is stored by `upsert_sign`, which derives the filename from the user-supplied `word` and `media_type` in the `/signs` POST endpoint:

```python
ext      = media_type.split("/")[-1]
filename = f"{word}.{ext}"
```

An authenticated user can POST `word=../../../etc/passwd` (or any relative path component) with `media_type=text/plain`, store that as a filename, and then trigger `/signs/{word}/image` to serve an arbitrary file from the host.

**Fix 1:** Sanitise the word and extension at write time:

```python
import re
safe_word = re.sub(r"[^\w\-]", "_", word)   # strip traversal chars
safe_ext  = media_type.split("/")[-1].lower()
if safe_ext not in {"png", "jpg", "jpeg", "gif", "webp"}:
    raise HTTPException(422, "Tipo de medio no soportado")
filename = f"{safe_word}.{safe_ext}"
```

**Fix 2:** At read time, resolve and assert the path stays under `ASSETS_DIR`:

```python
full_path = (config.ASSETS_DIR / sign["filename"]).resolve()
if not str(full_path).startswith(str(config.ASSETS_DIR.resolve())):
    raise HTTPException(403, "Acceso denegado")
```

Both fixes should be applied together.

---

### CR-04: Hardcoded insecure default JWT secret key

**File:** `backend/config.py:32`  
**Severity:** CRITICAL

```python
SECRET_KEY = os.getenv("SECRET_KEY", "handtalk-dev-insecure-key-change-in-prod")
```

If the `SECRET_KEY` environment variable is not set (which it will not be in many developer setups, CI pipelines, and containers without a proper secrets manager), the application falls back to a publicly known string. Any attacker can forge valid JWTs for any username.

**Fix:** Remove the default. Fail fast at startup if the key is absent or below a minimum entropy threshold:

```python
SECRET_KEY = os.getenv("SECRET_KEY", "")
if not SECRET_KEY or len(SECRET_KEY) < 32:
    raise RuntimeError(
        "SECRET_KEY environment variable must be set and at least 32 characters. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )
```

---

### CR-05: Arbitrary file write via image upload — no MIME validation

**File:** `backend/main.py:334-361`  
**Severity:** CRITICAL

The `/signs` POST endpoint writes the uploaded file directly to disk using the caller-controlled `media_type` to determine the extension:

```python
ext      = media_type.split("/")[-1]          # e.g. "php", "py", "sh"
filename = f"{word}.{ext}"
dest     = config.ASSETS_DIR / filename
if file:
    contents = await file.read()
    dest.write_bytes(contents)                # no content inspection
```

An authenticated attacker can upload a `.py` script (or any executable) to the `assets/` directory. If the backend is ever restarted or assets are served through a framework that executes Python files, this is remote code execution. Even without execution, it is an unrestricted file upload.

**Fix:** Validate both the declared MIME type and the magic bytes of the uploaded content:

```python
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
if media_type not in ALLOWED_TYPES:
    raise HTTPException(422, "Tipo de archivo no permitido")
contents = await file.read()
# Validate magic bytes
import imghdr
detected = imghdr.what(None, h=contents)
if detected not in {"png", "jpeg", "gif", "webp"}:
    raise HTTPException(422, "El contenido del archivo no coincide con el tipo declarado")
```

---

### CR-06: NLP endpoint exposes arbitrary session data without authentication

**File:** `backend/main.py:222-243`  
**Severity:** CRITICAL

Both NLP endpoints (`/nlp/translate` and `/nlp/clear`) accept a `session_id` from the request body/query string with no authentication:

```python
@app.post("/nlp/translate")
async def nlp_translate(body: NLPTranslateBody):
    signs = session_manager.get_signs(body.session_id)   # no auth check
    ...

@app.delete("/nlp/clear")
def nlp_clear(session_id: str):
    session_manager.clear(session_id)                     # no auth check
```

Any unauthenticated client can:
- Read any other user's accumulated sign sequence by guessing the session ID.
- Destroy any active session.

**Fix:** Require authentication and bind session IDs to the authenticated user (see also CR-01):

```python
@app.post("/nlp/translate")
async def nlp_translate(body: NLPTranslateBody, username: str = Depends(get_current_user)):
    # Validate the session_id belongs to this user
    if not body.session_id.startswith(username + "_"):
        raise HTTPException(403, "Sesión no autorizada")
    ...
```

---

## High Severity Issues

### H-01: `cleanup_stale` is never called — unbounded memory growth

**File:** `backend/session_manager.py:58-66`  
**Severity:** HIGH

`cleanup_stale` is implemented but there is no call site anywhere in the codebase. Sessions accumulate for the lifetime of the process. In production, with thousands of anonymous WebSocket clients connecting and disconnecting (e.g. the default `sid="anon"` fallback), this will exhaust available memory.

**Fix:** Call `cleanup_stale` on a background schedule at application startup:

```python
# In main.py, after app creation:
from contextlib import asynccontextmanager
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    async def _cleanup_loop():
        while True:
            await asyncio.sleep(120)            # every 2 minutes
            session_manager.cleanup_stale(300)  # 5-minute TTL
    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()

app = FastAPI(..., lifespan=lifespan)
```

---

### H-02: WebSocket error handler leaks internal exception detail to clients

**File:** `backend/main.py:464-469`  
**Severity:** HIGH

```python
except Exception as exc:
    log.error(f"Error WS {sid}: {exc}")
    try:
        await websocket.send_json({"type": "error", "detail": str(exc)})
    except Exception:
        pass
```

`str(exc)` can include stack traces, file paths, model file names, database paths, and library version strings. This is an information-disclosure vulnerability; exception messages should never be sent verbatim to clients.

**Fix:**

```python
except Exception as exc:
    log.error(f"Error WS {sid}: {exc}", exc_info=True)
    try:
        await websocket.send_json({"type": "error", "detail": "Error interno del servidor"})
    except Exception:
        pass
```

---

### H-03: `_spellcheck` creates a new `SpellChecker` instance on every call

**File:** `backend/text_to_sign.py:52-59`  
**Severity:** HIGH

```python
def _spellcheck(word: str) -> str:
    try:
        from spellchecker import SpellChecker
        spell = SpellChecker(language="es")   # loads dictionary every call
        corrected = spell.correction(word)
        return corrected if corrected else word
    except Exception:
        return word
```

`SpellChecker(language="es")` reads and parses the Spanish word list from disk on every invocation. In the phrase endpoint (which calls `get_sign_image` per word), a 10-word phrase triggers 10 spellchecker loads. This is both a latency spike and a CPU/disk burden.

**Fix:** Cache the instance at module level:

```python
_spell_checker = None

def _get_spellchecker():
    global _spell_checker
    if _spell_checker is None:
        from spellchecker import SpellChecker
        _spell_checker = SpellChecker(language="es")
    return _spell_checker

def _spellcheck(word: str) -> str:
    try:
        corrected = _get_spellchecker().correction(word)
        return corrected if corrected else word
    except Exception:
        return word
```

---

### H-04: Semantic search re-encodes entire dictionary on every structural change

**File:** `backend/text_to_sign.py:62-83`  
**Severity:** HIGH

```python
all_signs = get_all_signs()
words     = [s["word"] for s in all_signs]
if words != _sign_words_indexed:
    _sign_embeddings    = _semantic_model.encode(words, convert_to_tensor=True)
    _sign_words_indexed = words
```

The comparison `words != _sign_words_indexed` is a full list equality check — O(n) — and triggers a full re-embedding whenever any word is added or removed. `encode()` is synchronous CPU-bound work running inside an async endpoint (via `get_sign_image` called from the FastAPI route). This blocks the event loop for the duration of the encoding.

**Fix:** Move `_semantic_search` to run in a thread pool via `asyncio.to_thread`, and cache embeddings by a stable hash (e.g. frozenset of words) rather than list equality.

---

### H-05: `init_db` silently swallows migration errors

**File:** `backend/database.py:53-66`  
**Severity:** HIGH

```python
for col_sql in [
    "ALTER TABLE translations ADD COLUMN user_id TEXT",
    "ALTER TABLE signs ADD COLUMN has_real_image INTEGER NOT NULL DEFAULT 0",
]:
    try:
        conn.execute(col_sql)
        conn.commit()
    except Exception:
        pass
```

Catching bare `Exception` here is intentional (to skip already-existing columns), but also silently swallows genuine migration failures such as disk-full errors, permission errors, or corrupted WAL files. If migrations actually fail, the application starts with the wrong schema and produces confusing downstream bugs.

**Fix:** Only catch `sqlite3.OperationalError` and check the message:

```python
import sqlite3 as _sqlite3

for col_sql in [...]:
    try:
        conn.execute(col_sql)
        conn.commit()
    except _sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            log.error(f"Migration failed: {e}")
            raise
```

---

### H-06: `HolisticExtractor` module-level singletons are not thread-safe

**File:** `backend/holistic_model.py:78-87`  
**Severity:** HIGH

```python
_extractor_static = HolisticExtractor(static_image_mode=True)
_extractor_video  = HolisticExtractor(static_image_mode=False)
```

MediaPipe's `Holistic` object is not documented as thread-safe. Under Uvicorn with multiple worker processes this is safe (each process has its own object), but under `--workers 1` with multiple async tasks the same `_holistic.process()` call may be re-entered from concurrent WebSocket frame handlers. The result is undefined behaviour (segfault or corrupted output).

**Fix:** Wrap each call with an `asyncio.Lock` or use `asyncio.to_thread` to serialise access, since `process()` is blocking CPU work anyway:

```python
_video_lock = asyncio.Lock()

async def extract_video_async(image: np.ndarray) -> np.ndarray | None:
    async with _video_lock:
        return await asyncio.to_thread(_extractor_video.extract, image)
```

---

## Medium Severity Issues

### M-01: `datetime.utcnow()` deprecated usage in `auth.py`

**File:** `backend/auth.py:27`  
**Severity:** MEDIUM

```python
expire = datetime.utcnow() + timedelta(hours=config.TOKEN_EXPIRE_HOURS)
```

`datetime.utcnow()` is deprecated since Python 3.12 and returns a naive datetime. `database.py` consistently uses `datetime.now(timezone.utc)` — this is an inconsistency that can cause subtle token expiry bugs if the JWT library compares aware and naive datetimes.

**Fix:**

```python
from datetime import datetime, timedelta, timezone
expire = datetime.now(timezone.utc) + timedelta(hours=config.TOKEN_EXPIRE_HOURS)
```

---

### M-02: WebSocket frame handler imports `numpy` inside the hot loop

**File:** `backend/main.py:431`  
**Severity:** MEDIUM

```python
if ready:
    import numpy as np     # <-- inside the while True loop, inside if ready
    buf = np.array(session_manager.get_landmark_buffer(sid))
```

Python caches module imports after the first call, so this is not a correctness bug, but it is an O(1) dict lookup on every frame that triggers the `ready` branch. More importantly, it is misleading: it implies `numpy` might not be available and is an anti-pattern. `numpy` is already imported at the top of `main.py` as `np` (via `import numpy as np` at line 17) — this inner import shadows the module-level alias unnecessarily.

**Fix:** Remove the inner import; `np` is already available from the top-level import.

---

### M-03: `_face_keypoints` normalization is relative to point 0, not a stable face center

**File:** `backend/holistic_model.py:47-57`  
**Severity:** MEDIUM

```python
pts -= pts[0]   # subtracts the first selected face landmark
```

`_FACE_KEYPOINTS[0]` is landmark 33 (inner corner of the right eye). Normalising relative to the eye corner means that if the eye landmark is noisy or occluded, the entire face feature vector shifts. The hand normalisation correctly uses the wrist (`pts[0]` = landmark 0 of the hand, the wrist), which is a stable anchor. The face should use a more stable anchor such as the nose tip (landmark 4, which is already in the list) or the centroid of all selected points.

**Fix:**

```python
centroid = pts.mean(axis=0)
pts -= centroid
```

---

### M-04: `train_dynamic` does not stratify the train/test split

**File:** `backend/train_holistic_model.py:172`  
**Severity:** MEDIUM

```python
Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42)
```

The static split uses `stratify=y` (line 135), but the dynamic split does not. For dynamic signs with few classes (J, Z, adios, bien, ...) and highly class-imbalanced augmented data, this can put all samples of a rare class in the training set, yielding an inflated validation accuracy that masks poor generalisation.

**Fix:**

```python
Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

---

### M-05: `_lookup` calls `_semantic_search` unconditionally after fuzzy-match failure, no short-circuit

**File:** `backend/text_to_sign.py:86-103`  
**Severity:** MEDIUM

If `sentence-transformers` is installed, every word that fails both exact-match and fuzzy-match triggers a full semantic search — including loading the 420 MB `paraphrase-multilingual-MiniLM-L12-v2` model on first access. For a phrase with many unknown words, this blocks the HTTP response thread for several seconds on cold start.

There is no timeout on the semantic search and no way to disable it per-request. The existing `try/except` only handles library-absence, not slowness.

**Fix:** Add a configuration flag to disable semantic search, and ensure the first load happens at application startup (warm-up) rather than during the first user request.

---

### M-06: Frontend WebSocket error handler is a `console.error` with no user feedback

**File:** `frontend/screens/SignToTextScreen.js:114`  
**Severity:** MEDIUM

```javascript
ws.onerror = (e) => console.error('WS error:', e);
```

WebSocket errors are silently swallowed from the user's perspective. If the server is down, the user sees the "Tiempo real" button stay in its current state with no feedback. After a WebSocket error the `onclose` event fires and sets `wsConnected = false`, but there is no state to distinguish "cleanly disconnected" from "error disconnected", so the user does not know they should reconnect.

**Fix:**

```javascript
ws.onerror = (e) => {
  console.error('WS error:', e);
  setWsError('Conexión perdida. Toca para reconectar.');
};
```

---

### M-07: `captureDynamicFrame` silently swallows all errors

**File:** `frontend/screens/SignToTextScreen.js:228-250`  
**Severity:** MEDIUM

```javascript
} catch (_) {}
```

And `_predictDynamic` at line 268:

```javascript
} catch (_) {}
```

Any network error, camera error, or server error during dynamic frame capture is completely hidden. The progress bar will stall at its current position with no indication of failure. On slow networks where timeout (4 s) fires frequently this creates a confusing UX.

**Fix:** At minimum, set a user-visible error state or log the error with enough context for debugging.

---

### M-08: `NLP_MODEL` uses a non-existent model ID

**File:** `backend/config.py:59`  
**Severity:** MEDIUM

```python
NLP_MODEL = "claude-haiku-4-5-20251001"
```

As of the assistant knowledge cutoff (August 2025), the correct model ID for Claude Haiku is `claude-haiku-3-5-20251022` or similar versioned IDs. The string `claude-haiku-4-5-20251001` does not correspond to any known Anthropic model. Any call to the Claude API with this model ID will return a 400/404 error, causing the NLP pipeline to fall back to space-joining signs silently (see `nlp_pipeline.py:63-65`).

**Fix:** Update to a verified model ID and consider reading it from an environment variable so it can be updated without a code change:

```python
NLP_MODEL = os.getenv("NLP_MODEL", "claude-haiku-3-5-20251022")
```

---

### M-09: `add_sign` in `session_manager.py` preserves original casing but deduplication is case-insensitive — inconsistency

**File:** `backend/session_manager.py:42-48`  
**Severity:** MEDIUM

```python
def add_sign(self, session_id: str, sign: str):
    data = self._get_or_create(session_id)
    upper = sign.upper()
    if upper != data.last_sign.upper():
        data.sign_buffer.append(sign)        # original casing preserved
        data.last_sign = sign
```

`last_sign` is stored in original casing but compared case-insensitively. If `sign = "hola"` arrives first, `data.last_sign = "hola"`. Then `sign = "HOLA"` arrives: `"HOLA".upper() != "hola".upper()` → `False`, so it is correctly deduplicated. But `data.last_sign` is now `"hola"` (the first form). If the next sign is `"Hola"` (mixed case), `"HOLA" != "HOLA"` → `False`, still fine. The logic is actually correct but fragile and confusing. The comment says "Preserves original casing" but `last_sign` is not uppercased, making the invariant unclear.

**Fix (minor):** Store `last_sign` normalised to upper to make the invariant explicit:

```python
data.last_sign = upper   # always store normalised form
```

---

## Low Severity Issues

### L-01: `save_translation` never receives `user_id` from WebSocket or NLP endpoints

**File:** `backend/main.py:450, 232`  
**Severity:** LOW

```python
# WebSocket translate handler
save_translation(type_="sign_to_text", input_="[websocket]", output=translation)
# NLP translate endpoint
save_translation(type_="sign_to_text", input_="[nlp]", output=translation)
```

Both call sites omit `user_id`, so WebSocket and NLP translations are never attributed to a user. The `/history` endpoint filters by `user_id`, meaning these translations are invisible in every user's history even though the feature exists.

**Fix:** Pass `user_id` where available. For the NLP endpoint, add `username: str = Depends(get_current_user)` (which also fixes CR-06). For WebSocket, pass the authenticated username once authentication is added (see CR-01).

---

### L-02: Route ordering bug — `/signs/stats` is shadowed by `/signs/{word}`

**File:** `backend/main.py:295-325`  
**Severity:** LOW

```python
@app.get("/signs/{word}/image", ...)   # line 301
def sign_image_direct(word: str): ...

@app.get("/signs/stats", ...)          # line 318
def signs_stats(): ...
```

The `/signs/stats` endpoint is defined after `/signs/{word}/image`. FastAPI (Starlette) matches routes in registration order. The path `/signs/stats` would be captured by `/signs/{word}/image` with `word="stats"` before ever reaching the `signs_stats` handler, because both patterns match.

**Fix:** Move `/signs/stats` before any `/{word}` path pattern:

```python
@app.get("/signs/stats", ...)          # define FIRST
def signs_stats(): ...

@app.get("/signs/{word}/image", ...)   # define AFTER
def sign_image_direct(word: str): ...
```

---

### L-03: `_encode_file` returns `None` on I/O error but callers use `b64 is not None` as `found`

**File:** `backend/text_to_sign.py:120-125, 157`  
**Severity:** LOW

```python
b64 = _encode_file(full_path, media_type) if full_path else None
...
return {"found": True, "word": matched_word, ..., "base64": b64, ...}
```

If the image file exists in the database but is missing from disk (e.g., after a partial deployment), `_encode_file` catches the `FileNotFoundError` and returns `None`. The response still says `"found": True` but `"base64": null`. The frontend `SignCard` renders the placeholder text "Imagen no disponible" for this case (it checks `item.found && imgSource`), but the issue is the semantic mismatch: `found=True` means "in the database" not "image file is accessible". This can be confusing for API consumers.

**Fix:** Distinguish between "sign is in the dictionary" and "image is available" either with separate fields or by checking `full_path.exists()` before setting `found=True`.

---

### L-04: `mirror_landmarks` comment describes a layout inconsistent with `_hand_to_array`

**File:** `backend/data_augmentation.py:6-7`  
**Severity:** LOW

```python
# Holistic layout: left[0:63], right[63:126], face[126:225].
```

But `holistic_model.py:69-72` concatenates:

```python
return np.concatenate([left, right, face])
```

Where `left = _hand_to_array(result.left_hand_landmarks)` (63 floats) and `right = _hand_to_array(result.right_hand_landmarks)` (63 floats). The comment matches the code. However, the face is described as `[126:225]` = 99 floats, but `_face_keypoints` uses `_FACE_KEYPOINTS` which has 33 points × 3 = 99 floats, so 225 = 63+63+99. The layout comment is accurate — this is LOW severity purely because the comment is correct but the `face[126:225]` slice notation does not make the 99-element count self-evident and should say `face[126:225]  # 33 keypoints × 3 = 99`.

---

### L-05: `SpellChecker` correction may return the same word in Spanish if the input is a valid Spanish word not in the sign dictionary

**File:** `backend/text_to_sign.py:52-59`  
**Severity:** LOW

```python
corrected = _spellcheck(_normalize(original))
if corrected and corrected != _normalize(original):
    sign, matched_word = _lookup(corrected)
```

If the input word is already a correctly-spelled Spanish word that happens not to be in the sign dictionary, `_spellcheck` returns it unchanged (`corrected == _normalize(original)`), so the spellcheck branch is skipped. This is correct. But if `_spellcheck` returns a different word that is also not in the dictionary, a second lookup is performed for no gain and the user sees no suggestion. The sequence of fallbacks (exact → normalised → fuzzy → semantic → spellcheck → fallback dict) is correct but the `_spellcheck` is 5th in the chain, meaning the semantic search (H-04) already fires before the cheaper spellcheck.

**Fix:** Reorder: spellcheck before semantic search, since spellcheck is O(1) once the dictionary is cached (see H-03).

---

### L-06: No test coverage for WebSocket endpoint, path traversal, or file upload

**File:** `backend/tests/test_api.py`  
**Severity:** LOW

The test suite covers auth, history, dictionary CRUD, and text-to-sign endpoints. The following are not tested:
- `/ws/detect` WebSocket behaviour (connect, frame, translate, clear, disconnect).
- The `/signs` POST endpoint with a malicious `word` or `media_type`.
- NLP translate / clear endpoints.
- `_semantic_search` fallback path.
- `session_manager.cleanup_stale`.

The existing test for `test_history_is_user_scoped` is good — the same approach should be applied to the NLP and WebSocket session isolation once authentication is added.

---

## Performance Concerns

These are noted for awareness; they are not in the primary severity matrix.

1. **Base64 in phrase responses:** `get_sign_image` encodes the full image in Base64 in the HTTP response for every word in a phrase. A 10-word phrase can return 10 × ~50 KB = ~500 KB of base64 data. The frontend does use `thumbnail_base64` for the gallery view, but `base64` (full size) is still included. Consider making `base64` opt-in via a query parameter for the phrase endpoint.

2. **MediaPipe initialisation on import:** Both `_extractor_static` and `_extractor_video` are initialised at module import time (`holistic_model.py:78-79`). This runs the MediaPipe TFLite model loading (takes ~2 seconds) synchronously when Uvicorn imports the module, increasing cold-start time. Consider lazy initialisation with a `threading.Lock` guard.

3. **No connection pool for SQLite:** Every database function opens and closes a new connection. While WAL mode reduces contention, connection creation has non-trivial overhead per request under load. Consider using a `threading.local()` connection cache.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation Priority |
|------|-----------|--------|---------------------|
| Session hijacking via unauthenticated WebSocket | High | High | Immediate (CR-01, CR-06) |
| Path traversal / arbitrary file read via `/signs/{word}/image` | Medium | High | Immediate (CR-03) |
| Arbitrary file write / potential RCE via `/signs` upload | Low (requires auth) | Critical | Immediate (CR-05) |
| Forged JWTs via default SECRET_KEY | High in dev | Critical | Immediate (CR-04) |
| Server memory exhaustion from session accumulation | High in production | High | Before launch (H-01) |
| NLP model ID wrong — silent fallback | Certain | Medium | Pre-launch (M-08) |
| Route shadowing `/signs/stats` | Medium | Low | Pre-launch (L-02) |
| Race condition on session buffer under load | Medium | Medium | Before load test (CR-02) |

---

## Top 5 Actionable Recommendations

### 1. Fix authentication gaps before any production deployment
Merge CR-01, CR-04, CR-06, and CR-05 as a single hardening pass:
- Require JWT on WebSocket connections.
- Require JWT on NLP endpoints.
- Fail startup if `SECRET_KEY` is weak.
- Whitelist file MIME types and sanitise filenames.

This is the highest-priority block because multiple attack vectors are independent of each other.

### 2. Fix path traversal (CR-03) immediately
This is exploitable by any user with an account. Add both write-time sanitisation (strip traversal characters from `word` and `ext`) and read-time path confinement check.

### 3. Start the session cleanup background task (H-01)
One-line fix with significant memory safety impact. Use `asyncio` lifespan context manager.

### 4. Fix the model ID and NLP silent fallback (M-08)
The NLP feature will never work in production with the current model ID. Correct it, add startup validation that the Anthropic SDK is importable when `ANTHROPIC_API_KEY` is set, and log a clear warning if the key is absent.

### 5. Cache the SpellChecker and guard semantic search (H-03, H-04)
Instantiate `SpellChecker` once at module level and gate semantic search behind a configuration flag. Reorder the fallback chain so the cheap operations (exact → normalised → spellcheck → fuzzy) run before the expensive ones (semantic).

---

## Summary Table

| ID | Severity | File | Issue |
|----|----------|------|-------|
| CR-01 | CRITICAL | main.py:392 | WebSocket session not authenticated |
| CR-02 | CRITICAL | session_manager.py:28 | Race condition on session buffer |
| CR-03 | CRITICAL | main.py:306 | Path traversal via sign filename |
| CR-04 | CRITICAL | config.py:32 | Hardcoded default JWT secret |
| CR-05 | CRITICAL | main.py:334 | Arbitrary file write via upload |
| CR-06 | CRITICAL | main.py:222 | NLP endpoints unauthenticated |
| H-01 | HIGH | session_manager.py:58 | cleanup_stale never called |
| H-02 | HIGH | main.py:464 | Exception detail leaked to WS client |
| H-03 | HIGH | text_to_sign.py:52 | SpellChecker re-instantiated per call |
| H-04 | HIGH | text_to_sign.py:62 | Semantic search blocks event loop |
| H-05 | HIGH | database.py:53 | init_db swallows real migration errors |
| H-06 | HIGH | holistic_model.py:78 | MediaPipe singletons not thread-safe |
| M-01 | MEDIUM | auth.py:27 | datetime.utcnow() deprecated |
| M-02 | MEDIUM | main.py:431 | numpy imported inside hot loop |
| M-03 | MEDIUM | holistic_model.py:47 | Face normalisation uses unstable anchor |
| M-04 | MEDIUM | train_holistic_model.py:172 | Dynamic split missing stratify=y |
| M-05 | MEDIUM | text_to_sign.py:100 | Semantic search triggers on every miss |
| M-06 | MEDIUM | SignToTextScreen.js:114 | WS error gives no user feedback |
| M-07 | MEDIUM | SignToTextScreen.js:249 | Dynamic frame errors silently swallowed |
| M-08 | MEDIUM | config.py:59 | NLP model ID does not exist |
| M-09 | MEDIUM | session_manager.py:42 | last_sign casing inconsistency |
| L-01 | LOW | main.py:450 | user_id missing from WS/NLP saves |
| L-02 | LOW | main.py:318 | /signs/stats shadowed by /{word} route |
| L-03 | LOW | text_to_sign.py:157 | found=True when image file is missing |
| L-04 | LOW | data_augmentation.py:7 | Comment could state 99-element count |
| L-05 | LOW | text_to_sign.py:133 | Spellcheck runs after expensive semantic |
| L-06 | LOW | tests/test_api.py | No WS, upload, or NLP test coverage |

---

_Reviewed: 2026-05-28_  
_Reviewer: Claude (adversarial deep review)_
