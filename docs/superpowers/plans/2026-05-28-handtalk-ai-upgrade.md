# HandTalk AI Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade HandTalk de un traductor foto-por-foto a un sistema conversacional en tiempo real con CV holístico (2 manos + cara), modelos ML modernos (Residual MLP + TCN), NLP con gramática LSM via Claude API, y galería visual de señas.

**Architecture:** MediaPipe Holistic extrae 225 features (2 manos + cara); un Residual MLP clasifica señas estáticas y un TCN clasifica secuencias dinámicas; un SessionManager elimina el buffer global; WebSocket `/ws/detect` entrega predicciones en ≤200ms; Claude API traduce secuencias LSM a español natural; `text_to_sign` incluye búsqueda semántica, spell-check y thumbnails para galería.

**Tech Stack:** FastAPI, MediaPipe Holistic, TensorFlow (Residual MLP + TCN), Anthropic SDK (claude-haiku-4-5-20251001), sentence-transformers, pyspellchecker, WebSockets, React Native (expo-camera), SQLite.

**Spec:** `docs/superpowers/specs/2026-05-28-handtalk-ai-improvements-design.md`

---

## Dependency Order

```
Task 1 (bugs)
Task 2 (history/add_sign)
Task 3 (config+reqs)       ← Task 4, 5, 6 depend on this
Task 4 (session_manager)
Task 5 (holistic_model)    ← Task 7, 8 depend on this
Task 6 (data_augmentation)
Task 7 (sign_classifier)   ← Task 10 depends on this
Task 8 (temporal_model)    ← Task 10 depends on this
Task 9 (train script)      ← independent utility
Task 10 (update sign/seq models)  ← Task 14 depends on this
Task 11 (nlp_pipeline)     ← Task 14 depends on this
Task 12 (text_to_sign)     ← Task 15 depends on this
Task 13 (seed + DB migration)
Task 14 (main.py WS + endpoints)  ← Task 16 depends on this
Task 15 (main.py phrase response)
Task 16 (frontend SignToText WS)
Task 17 (frontend TextToSign gallery)
```

---

## Task 1: Fix Critical Bugs

**Files:**
- Modify: `backend/database.py`
- Modify: `backend/sign_model.py`
- Modify: `backend/config.py`
- Modify: `backend/text_to_sign.py`
- Test: `backend/tests/test_database.py`

- [ ] **Step 1: Write failing test for datetime bug**

Add to `backend/tests/test_database.py`:
```python
def test_save_translation_uses_utc_isoformat():
    from database import save_translation, get_history
    save_translation("sign_to_text", "[img]", "HOLA", 0.9)
    history = get_history(1)
    assert history, "No se guardó la traducción"
    ts = history[0]["created_at"]
    # Python 3.12+ utcnow() raises DeprecationWarning; datetime.now(timezone.utc) no
    from datetime import datetime, timezone
    # Must parse without error
    parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert parsed is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && python -m pytest tests/test_database.py::test_save_translation_uses_utc_isoformat -v
```

Expected: PASS (the bug is silent — it won't crash until Python 3.12 strict mode, but running ensures baseline).

- [ ] **Step 3: Fix datetime in database.py**

In `backend/database.py`, replace line 1:
```python
from datetime import datetime
```
with:
```python
from datetime import datetime, timezone
```

Replace every `datetime.utcnow().isoformat()` (occurs 4 times) with `datetime.now(timezone.utc).isoformat()`:
```python
# In save_translation:
(type_, input_, output, confidence, datetime.now(timezone.utc).isoformat()),

# In create_user:
(username, hashed_password, datetime.now(timezone.utc).isoformat()),

# In upsert_sign:
(word, filename, media_type, category, datetime.now(timezone.utc).isoformat()),
```

- [ ] **Step 4: Fix normalize_landmarks scale=0 in sign_model.py**

In `backend/sign_model.py`, replace lines 51-56:
```python
def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    pts = landmarks.copy()
    pts -= pts[0]
    scale = np.linalg.norm(pts[9])
    pts /= max(float(scale), 1e-6)
    return pts.flatten()
```

- [ ] **Step 5: Fix CORS default in config.py**

In `backend/config.py`, replace line 37:
```python
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8081,http://localhost:19006").split(",")
```

- [ ] **Step 6: Add words cache to text_to_sign.py**

In `backend/text_to_sign.py`, add after the imports:
```python
from database import get_sign, get_all_signs, count_signs

_words_cache: list[str] | None = None
_words_cache_size: int = 0


def _get_words_cache() -> list[str]:
    global _words_cache, _words_cache_size
    current = count_signs()
    if _words_cache is None or current != _words_cache_size:
        _words_cache = [s["word"] for s in get_all_signs()]
        _words_cache_size = current
    return _words_cache


def invalidate_words_cache():
    global _words_cache, _words_cache_size
    _words_cache = None
    _words_cache_size = 0
```

Replace the `_lookup` function body line 50 (`all_words = [s["word"] for s in get_all_signs()]`) with:
```python
    all_words = _get_words_cache()
```

- [ ] **Step 7: Run existing tests to confirm no regressions**

```bash
cd backend && python -m pytest tests/ -v
```
Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add backend/database.py backend/sign_model.py backend/config.py backend/text_to_sign.py backend/tests/test_database.py
git commit -m "fix: datetime deprecation, CORS default, scale=0 normalization, words cache"
```

---

## Task 2: Fix History User-Filtering + add_sign Form Params

**Files:**
- Modify: `backend/database.py`
- Modify: `backend/main.py`
- Test: `backend/tests/test_api.py`

- [ ] **Step 1: Write failing test for history isolation**

Add to `backend/tests/test_api.py`:
```python
def test_history_is_user_scoped():
    # Register two users
    client.post("/auth/register", json={"username": "user_a_hist", "password": "pass1234"})
    client.post("/auth/register", json={"username": "user_b_hist", "password": "pass1234"})
    token_a = client.post("/auth/login", json={"username": "user_a_hist", "password": "pass1234"}).json()["access_token"]
    token_b = client.post("/auth/login", json={"username": "user_b_hist", "password": "pass1234"}).json()["access_token"]
    # Both get history — should only see their own (empty on first call)
    hist_a = client.get("/history", headers={"Authorization": f"Bearer {token_a}"}).json()["history"]
    hist_b = client.get("/history", headers={"Authorization": f"Bearer {token_b}"}).json()["history"]
    # Verify response has 'user' field matching token
    r_a = client.get("/history", headers={"Authorization": f"Bearer {token_a}"}).json()
    assert r_a["user"] == "user_a_hist"
    r_b = client.get("/history", headers={"Authorization": f"Bearer {token_b}"}).json()
    assert r_b["user"] == "user_b_hist"
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd backend && python -m pytest tests/test_api.py::test_history_is_user_scoped -v
```
Expected: FAIL (history returns all users' data).

- [ ] **Step 3: Add user_id column to translations table**

In `backend/database.py`, update `init_db()` — add migration after the `executescript`:
```python
def init_db():
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS translations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            type        TEXT    NOT NULL,
            input       TEXT,
            output      TEXT,
            confidence  REAL,
            created_at  TEXT    NOT NULL,
            user_id     TEXT
        );
        CREATE TABLE IF NOT EXISTS users (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            username        TEXT UNIQUE NOT NULL,
            hashed_password TEXT        NOT NULL,
            created_at      TEXT        NOT NULL
        );
        CREATE TABLE IF NOT EXISTS signs (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            word           TEXT UNIQUE NOT NULL,
            filename       TEXT,
            media_type     TEXT NOT NULL DEFAULT 'image/png',
            category       TEXT,
            has_real_image INTEGER NOT NULL DEFAULT 0,
            created_at     TEXT NOT NULL
        );
    """)
    # Migrations for existing DBs
    for col_sql in [
        "ALTER TABLE translations ADD COLUMN user_id TEXT",
        "ALTER TABLE signs ADD COLUMN has_real_image INTEGER NOT NULL DEFAULT 0",
    ]:
        try:
            conn.execute(col_sql)
            conn.commit()
        except Exception:
            pass  # Column already exists
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_translations_user ON translations(user_id)")
        conn.commit()
    except Exception:
        pass
    conn.close()
    log.info(f"Base de datos lista: {DB_PATH}")
```

- [ ] **Step 4: Update save_translation to accept user_id**

In `backend/database.py`, replace `save_translation`:
```python
def save_translation(type_: str, input_: str, output: str, confidence: float = None, user_id: str = None):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO translations (type, input, output, confidence, created_at, user_id) VALUES (?,?,?,?,?,?)",
        (type_, input_, output, confidence, datetime.now(timezone.utc).isoformat(), user_id),
    )
    conn.commit()
    conn.close()
```

- [ ] **Step 5: Update get_history to filter by user_id**

In `backend/database.py`, replace `get_history`:
```python
def get_history(limit: int = 20, user_id: str = None):
    conn = _get_conn()
    if user_id:
        rows = conn.execute(
            "SELECT * FROM translations WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM translations ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
```

- [ ] **Step 6: Update /history endpoint to pass username as user_id**

In `backend/main.py`, replace the `history` endpoint:
```python
@app.get(
    "/history",
    tags=["Historial"],
    summary="Ver historial de traducciones (requiere auth)",
)
def history(limit: int = 20, username: str = Depends(get_current_user)):
    return {"user": username, "history": get_history(limit, user_id=username)}
```

- [ ] **Step 7: Fix add_sign to use Form fields**

In `backend/main.py`, add at the top of imports:
```python
from fastapi import Form
```

Replace the `add_sign` signature:
```python
@app.post(
    "/signs",
    status_code=201,
    tags=["Diccionario"],
    summary="Agregar o actualizar una seña (requiere auth)",
)
async def add_sign(
    word:       str               = Form(...),
    category:   str | None        = Form(default=None),
    media_type: str               = Form(default="image/png"),
    file:       UploadFile | None = File(default=None),
    username:   str               = Depends(get_current_user),
):
```

- [ ] **Step 8: Run failing test to confirm it passes**

```bash
cd backend && python -m pytest tests/test_api.py::test_history_is_user_scoped -v
```
Expected: PASS.

- [ ] **Step 9: Run full test suite**

```bash
cd backend && python -m pytest tests/ -v
```
Expected: all tests PASS.

- [ ] **Step 10: Commit**

```bash
git add backend/database.py backend/main.py backend/tests/test_api.py
git commit -m "fix: history scoped to user, translations.user_id, add_sign uses Form fields"
```

---

## Task 3: Update config.py + requirements.txt

**Files:**
- Modify: `backend/config.py`
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Add new config variables to config.py**

Append to `backend/config.py`:
```python
# ── Model versioning ──────────────────────────────────────────────────────────
MODEL_VERSION          = os.getenv("MODEL_VERSION", "v2")
MODEL_V1_DIR           = MODEL_DIR / "v1"
MODEL_V2_DIR           = MODEL_DIR / "v2"
HOLISTIC_STATIC_PATH   = MODEL_V2_DIR / "holistic_static.keras"
HOLISTIC_STATIC_LABELS = MODEL_V2_DIR / "labels_static.npy"
HOLISTIC_DYNAMIC_PATH  = MODEL_V2_DIR / "holistic_dynamic.keras"
HOLISTIC_DYNAMIC_LABELS = MODEL_V2_DIR / "labels_dynamic.npy"
CALIBRATION_PATH       = MODEL_V2_DIR / "calibration.json"

# ── Anthropic / NLP ───────────────────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
NLP_MODEL          = "claude-haiku-4-5-20251001"
SEMANTIC_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.65"))

# ── WebSocket thresholds ──────────────────────────────────────────────────────
WS_STATIC_THRESHOLD  = float(os.getenv("WS_STATIC_THRESHOLD",  "0.6"))
WS_DYNAMIC_THRESHOLD = float(os.getenv("WS_DYNAMIC_THRESHOLD", "0.7"))
```

- [ ] **Step 2: Update requirements.txt**

Replace entire `backend/requirements.txt`:
```
fastapi
uvicorn[standard]
websockets
opencv-python
mediapipe
tensorflow
numpy
pillow
python-multipart
scikit-learn
slowapi
python-jose[cryptography]
passlib[bcrypt]
bcrypt>=4.0.1,<5.0.0
python-dotenv
anthropic>=0.40.0
sentence-transformers>=3.0.0
pyspellchecker>=0.8.0
```

- [ ] **Step 3: Update .env.example to include new keys**

In `backend/.env.example`, add:
```
# Anthropic API (requerido para traducción LSM→español)
ANTHROPIC_API_KEY=sk-ant-...

# Ajustes opcionales
MODEL_VERSION=v2
SEMANTIC_THRESHOLD=0.65
WS_STATIC_THRESHOLD=0.6
WS_DYNAMIC_THRESHOLD=0.7
```

- [ ] **Step 4: Run existing tests**

```bash
cd backend && python -m pytest tests/ -v
```
Expected: all PASS (config changes are additive).

- [ ] **Step 5: Commit**

```bash
git add backend/config.py backend/requirements.txt backend/.env.example
git commit -m "feat: add model versioning, Anthropic NLP, and WebSocket config variables"
```

---

## Task 4: Create session_manager.py

**Files:**
- Create: `backend/session_manager.py`
- Create: `backend/tests/test_session_manager.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_session_manager.py`:
```python
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
    import time
    from datetime import datetime, timezone, timedelta
    sm = SessionManager()
    sm.add_sign("old_session", "HOLA")
    # Manually age the session
    sm._sessions["old_session"].updated_at = datetime.now(timezone.utc) - timedelta(seconds=400)
    sm.cleanup_stale(max_age_seconds=300)
    assert "old_session" not in sm._sessions
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_session_manager.py -v
```
Expected: ImportError (module not found).

- [ ] **Step 3: Create session_manager.py**

Create `backend/session_manager.py`:
```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd backend && python -m pytest tests/test_session_manager.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/session_manager.py backend/tests/test_session_manager.py
git commit -m "feat: add SessionManager — per-session buffers replacing global deque"
```

---

## Task 5: Create holistic_model.py

**Files:**
- Create: `backend/holistic_model.py`
- Create: `backend/tests/test_holistic_model.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_holistic_model.py`:
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

def test_extract_returns_none_when_no_hands(monkeypatch):
    import holistic_model
    # Mock result with no landmarks
    mock_result = type("R", (), {
        "left_hand_landmarks": None,
        "right_hand_landmarks": None,
        "face_landmarks": None,
    })()
    monkeypatch.setattr(holistic_model._extractor_static._holistic, "process", lambda _: mock_result)
    result = holistic_model.extract_static(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result is None

def test_extract_returns_225_features_with_one_hand(monkeypatch):
    import holistic_model
    # Build a mock hand with 21 landmarks
    lm = type("LM", (), {"x": 0.1, "y": 0.2, "z": 0.0})()
    hand = type("Hand", (), {"landmark": [lm] * 21})()
    mock_result = type("R", (), {
        "left_hand_landmarks": None,
        "right_hand_landmarks": hand,
        "face_landmarks": None,
    })()
    monkeypatch.setattr(holistic_model._extractor_static._holistic, "process", lambda _: mock_result)
    features = holistic_model.extract_static(np.zeros((100, 100, 3), dtype=np.uint8))
    assert features is not None
    assert features.shape == (225,)

def test_normalize_scale_zero_does_not_crash():
    from holistic_model import HolisticExtractor
    ext = HolisticExtractor.__new__(HolisticExtractor)
    # All zeros → scale = 0
    hand = type("Hand", (), {"landmark": [type("LM", (), {"x": 0.0, "y": 0.0, "z": 0.0})() for _ in range(21)]})()
    result = ext._hand_to_array(hand)
    assert result.shape == (63,)
    assert not np.any(np.isnan(result))

def test_features_length_left_right_face():
    from holistic_model import HolisticExtractor
    ext = HolisticExtractor.__new__(HolisticExtractor)
    zeros_hand = type("Hand", (), {"landmark": [type("LM", (), {"x": 0.0, "y": 0.0, "z": 0.0})() for _ in range(21)]})()
    left  = ext._hand_to_array(zeros_hand)
    right = ext._hand_to_array(zeros_hand)
    face  = ext._face_keypoints(None)
    features = np.concatenate([left, right, face])
    assert features.shape == (225,)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd backend && python -m pytest tests/test_holistic_model.py -v
```
Expected: ImportError.

- [ ] **Step 3: Create holistic_model.py**

Create `backend/holistic_model.py`:
```python
import mediapipe as mp
import cv2
import numpy as np
from logger import get_logger

log = get_logger(__name__)

mp_holistic = mp.solutions.holistic

# 33 face landmarks que capturan expresión facial relevante para LSM
_FACE_KEYPOINTS = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246,
    362, 382, 381, 380, 374, 373, 390, 249, 263,
    466, 388, 387, 386, 385, 384, 398, 4
]  # 33 puntos


class HolisticExtractor:
    def __init__(self, static_image_mode: bool = True):
        self._holistic = mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl, a, b))
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        except Exception:
            return image

    def _hand_to_array(self, hand_landmarks) -> np.ndarray:
        if hand_landmarks is None:
            return np.zeros(63)
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        pts -= pts[0]
        scale = np.linalg.norm(pts[9])
        pts /= max(float(scale), 1e-6)
        return pts.flatten()

    def _face_keypoints(self, face_landmarks) -> np.ndarray:
        if face_landmarks is None:
            return np.zeros(99)
        pts = np.array([
            [face_landmarks.landmark[i].x,
             face_landmarks.landmark[i].y,
             face_landmarks.landmark[i].z]
            for i in _FACE_KEYPOINTS
        ])
        pts -= pts[0]
        return pts.flatten()

    def extract(self, image: np.ndarray) -> np.ndarray | None:
        preprocessed = self.preprocess_image(image)
        rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
        result = self._holistic.process(rgb)

        has_left  = result.left_hand_landmarks is not None
        has_right = result.right_hand_landmarks is not None
        if not has_left and not has_right:
            return None

        left  = self._hand_to_array(result.left_hand_landmarks)
        right = self._hand_to_array(result.right_hand_landmarks)
        face  = self._face_keypoints(result.face_landmarks)
        return np.concatenate([left, right, face])

    def close(self):
        self._holistic.close()


_extractor_static = HolisticExtractor(static_image_mode=True)
_extractor_video  = HolisticExtractor(static_image_mode=False)


def extract_static(image: np.ndarray) -> np.ndarray | None:
    return _extractor_static.extract(image)


def extract_video(image: np.ndarray) -> np.ndarray | None:
    return _extractor_video.extract(image)
```

- [ ] **Step 4: Update conftest.py to mock mediapipe holistic**

In `backend/tests/conftest.py`, replace the MediaPipe mock block:
```python
mp_mock = MagicMock()
hands_instance = MagicMock()
hands_instance.process.return_value = MagicMock(multi_hand_landmarks=None)
mp_mock.solutions.hands.Hands.return_value = hands_instance

holistic_instance = MagicMock()
holistic_instance.process.return_value = MagicMock(
    left_hand_landmarks=None,
    right_hand_landmarks=None,
    face_landmarks=None,
)
mp_mock.solutions.holistic.Holistic.return_value = holistic_instance
mp_mock.solutions.drawing_utils = MagicMock()
sys.modules['mediapipe'] = mp_mock
```

- [ ] **Step 5: Run tests**

```bash
cd backend && python -m pytest tests/test_holistic_model.py tests/ -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/holistic_model.py backend/tests/test_holistic_model.py backend/tests/conftest.py
git commit -m "feat: HolisticExtractor — MediaPipe Holistic with 2 hands + face (225 features)"
```

---

## Task 6: Create data_augmentation.py

**Files:**
- Create: `backend/data_augmentation.py`
- Create: `backend/tests/test_data_augmentation.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_data_augmentation.py`:
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np

def test_augment_sample_returns_4_variants():
    from data_augmentation import augment_sample
    v = np.random.rand(225)
    variants = augment_sample(v)
    assert len(variants) == 4
    assert all(vv.shape == (225,) for vv in variants)

def test_mirror_swaps_hands():
    from data_augmentation import mirror_landmarks
    v = np.zeros(225)
    v[0:63] = 1.0   # left hand features = 1
    v[63:126] = 2.0  # right hand features = 2
    mirrored = mirror_landmarks(v)
    # After mirror: left should have right's values and vice versa (negated X)
    assert not np.allclose(mirrored[0:63], v[0:63])

def test_add_noise_changes_values():
    from data_augmentation import add_noise
    v = np.ones(225)
    noisy = add_noise(v, sigma=0.1)
    assert not np.allclose(v, noisy)
    assert noisy.shape == (225,)

def test_scale_landmarks_changes_magnitude():
    from data_augmentation import scale_landmarks
    v = np.ones(225)
    scaled = scale_landmarks(v, factor=2.0)
    assert np.allclose(scaled, 2.0)

def test_augment_sequence_returns_4_variants():
    from data_augmentation import augment_sequence
    from labels import SEQUENCE_LENGTH
    seq = np.random.rand(SEQUENCE_LENGTH, 225)
    variants = augment_sequence(seq)
    assert len(variants) == 4
    assert all(vv.shape == (SEQUENCE_LENGTH, 225) for vv in variants)
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd backend && python -m pytest tests/test_data_augmentation.py -v
```
Expected: ImportError.

- [ ] **Step 3: Create data_augmentation.py**

Create `backend/data_augmentation.py`:
```python
import numpy as np


def mirror_landmarks(v: np.ndarray) -> np.ndarray:
    """Espeja las manos (intercambia izq/der y niega X).
    Asume layout holístico: left[0:63], right[63:126], face[126:225].
    """
    out = v.copy()
    left_part  = v[0:63].copy()
    right_part = v[63:126].copy()
    left_part[0::3]  *= -1
    right_part[0::3] *= -1
    out[0:63]   = right_part
    out[63:126] = left_part
    out[126:225:3] *= -1
    return out


def add_noise(v: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    return v + np.random.normal(0, sigma, v.shape).astype(v.dtype)


def scale_landmarks(v: np.ndarray, factor: float | None = None) -> np.ndarray:
    if factor is None:
        factor = float(np.random.uniform(0.9, 1.1))
    return v * factor


def augment_sample(v: np.ndarray) -> list[np.ndarray]:
    """Retorna 4 variantes de un vector de landmarks (225,)."""
    return [
        v,
        mirror_landmarks(v),
        add_noise(v),
        scale_landmarks(v),
    ]


def augment_sequence(seq: np.ndarray) -> list[np.ndarray]:
    """Retorna 4 variantes de una secuencia (T, 225)."""
    factor = float(np.random.uniform(0.9, 1.1))
    return [
        seq,
        np.array([mirror_landmarks(f) for f in seq]),
        np.array([add_noise(f) for f in seq]),
        np.array([scale_landmarks(f, factor) for f in seq]),
    ]
```

- [ ] **Step 4: Run tests**

```bash
cd backend && python -m pytest tests/test_data_augmentation.py -v
```
Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/data_augmentation.py backend/tests/test_data_augmentation.py
git commit -m "feat: data augmentation — mirror, noise, scale for 4x dataset expansion"
```

---

## Task 7: Create sign_classifier.py (Residual MLP)

**Files:**
- Create: `backend/sign_classifier.py`
- Create: `backend/tests/test_sign_classifier.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_sign_classifier.py`:
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np

def test_predict_static_returns_dict_without_model():
    from sign_classifier import predict_static
    result = predict_static(np.zeros(225))
    assert "prediction" in result
    assert "confidence" in result
    assert isinstance(result["confidence"], float)

def test_predict_static_confidence_in_range():
    from sign_classifier import predict_static
    result = predict_static(np.random.rand(225))
    assert 0.0 <= result["confidence"] <= 1.0

def test_predict_static_no_model_returns_unavailable():
    import sign_classifier
    original = sign_classifier._static_model
    sign_classifier._static_model = None
    result = sign_classifier.predict_static(np.zeros(225))
    assert result["prediction"] == "Modelo no disponible"
    sign_classifier._static_model = original
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd backend && python -m pytest tests/test_sign_classifier.py -v
```
Expected: ImportError.

- [ ] **Step 3: Create sign_classifier.py**

Create `backend/sign_classifier.py`:
```python
import numpy as np
import config
from logger import get_logger

log = get_logger(__name__)

_static_model  = None
_static_labels: list[str] = []
_temperature   = 1.0


def _load():
    global _static_model, _static_labels, _temperature
    if not config.HOLISTIC_STATIC_PATH.exists():
        log.warning("Modelo holístico estático no encontrado. Ejecuta train_holistic_model.py")
        return
    try:
        import tensorflow as tf
        import json
        _static_model  = tf.keras.models.load_model(str(config.HOLISTIC_STATIC_PATH))
        _static_labels = list(np.load(str(config.HOLISTIC_STATIC_LABELS), allow_pickle=True))
        if config.CALIBRATION_PATH.exists():
            cal = json.loads(config.CALIBRATION_PATH.read_text())
            _temperature = float(cal.get("temperature", 1.0))
        log.info(f"Clasificador holístico: {len(_static_labels)} clases, T={_temperature:.3f}")
    except Exception as exc:
        log.error(f"Error cargando clasificador: {exc}")


def predict_static(features: np.ndarray) -> dict:
    """features: array (225,). Retorna {'prediction': str, 'confidence': float}."""
    if _static_model is None:
        return {"prediction": "Modelo no disponible", "confidence": 0.0}
    try:
        import tensorflow as tf
        logits     = _static_model(features.reshape(1, -1), training=False).numpy()[0]
        calibrated = tf.nn.softmax(logits / _temperature).numpy()
        idx        = int(np.argmax(calibrated))
        confidence = float(calibrated[idx])
        label      = (_static_labels[idx]
                      if confidence >= config.WS_STATIC_THRESHOLD
                      else "No reconocido")
        return {"prediction": label, "confidence": round(confidence, 4)}
    except Exception as exc:
        log.error(f"Error en predicción estática: {exc}")
        return {"prediction": "Error", "confidence": 0.0}


_load()
```

- [ ] **Step 4: Run tests**

```bash
cd backend && python -m pytest tests/test_sign_classifier.py -v
```
Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/sign_classifier.py backend/tests/test_sign_classifier.py
git commit -m "feat: sign_classifier — Residual MLP inference with temperature calibration"
```

---

## Task 8: Create temporal_model.py (TCN)

**Files:**
- Create: `backend/temporal_model.py`
- Create: `backend/tests/test_temporal_model.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_temporal_model.py`:
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from labels import SEQUENCE_LENGTH

def test_predict_dynamic_no_model():
    from temporal_model import predict_dynamic
    seq = np.zeros((SEQUENCE_LENGTH, 225))
    result = predict_dynamic(seq)
    assert "prediction" in result
    assert "confidence" in result

def test_predict_dynamic_confidence_range():
    from temporal_model import predict_dynamic
    seq = np.random.rand(SEQUENCE_LENGTH, 225)
    result = predict_dynamic(seq)
    assert 0.0 <= result["confidence"] <= 1.0

def test_predict_dynamic_none_model_returns_unavailable():
    import temporal_model
    original = temporal_model._dynamic_model
    temporal_model._dynamic_model = None
    result = temporal_model.predict_dynamic(np.zeros((SEQUENCE_LENGTH, 225)))
    assert result["prediction"] == "Modelo no disponible"
    temporal_model._dynamic_model = original
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd backend && python -m pytest tests/test_temporal_model.py -v
```
Expected: ImportError.

- [ ] **Step 3: Create temporal_model.py**

Create `backend/temporal_model.py`:
```python
import numpy as np
import config
from labels import SEQUENCE_LENGTH
from logger import get_logger

log = get_logger(__name__)

_dynamic_model  = None
_dynamic_labels: list[str] = []
_temperature    = 1.0


def _load():
    global _dynamic_model, _dynamic_labels, _temperature
    if not config.HOLISTIC_DYNAMIC_PATH.exists():
        log.warning("Modelo TCN no encontrado. Ejecuta train_holistic_model.py")
        return
    try:
        import tensorflow as tf
        import json
        _dynamic_model  = tf.keras.models.load_model(str(config.HOLISTIC_DYNAMIC_PATH))
        _dynamic_labels = list(np.load(str(config.HOLISTIC_DYNAMIC_LABELS), allow_pickle=True))
        if config.CALIBRATION_PATH.exists():
            cal = json.loads(config.CALIBRATION_PATH.read_text())
            _temperature = float(cal.get("temperature_dynamic", 1.0))
        log.info(f"Modelo TCN: {len(_dynamic_labels)} clases, T={_temperature:.3f}")
    except Exception as exc:
        log.error(f"Error cargando TCN: {exc}")


def predict_dynamic(sequence: np.ndarray) -> dict:
    """sequence: array (SEQUENCE_LENGTH, 225). Retorna {'prediction': str, 'confidence': float}."""
    if _dynamic_model is None:
        return {"prediction": "Modelo no disponible", "confidence": 0.0}
    try:
        import tensorflow as tf
        inp        = sequence.reshape(1, SEQUENCE_LENGTH, 225)
        logits     = _dynamic_model(inp, training=False).numpy()[0]
        calibrated = tf.nn.softmax(logits / _temperature).numpy()
        idx        = int(np.argmax(calibrated))
        confidence = float(calibrated[idx])
        label      = (_dynamic_labels[idx]
                      if confidence >= config.WS_DYNAMIC_THRESHOLD
                      else "No reconocido")
        return {"prediction": label, "confidence": round(confidence, 4)}
    except Exception as exc:
        log.error(f"Error en predicción dinámica: {exc}")
        return {"prediction": "Error", "confidence": 0.0}


_load()
```

- [ ] **Step 4: Run tests**

```bash
cd backend && python -m pytest tests/test_temporal_model.py -v
```
Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/temporal_model.py backend/tests/test_temporal_model.py
git commit -m "feat: temporal_model — TCN inference with temperature calibration"
```

---

## Task 9: Create train_holistic_model.py

**Files:**
- Create: `backend/train_holistic_model.py`

*(No unit tests — this is a training script that requires real data and GPU. Verified by running manually.)*

- [ ] **Step 1: Create train_holistic_model.py**

Create `backend/train_holistic_model.py`:
```python
"""
Entrena los modelos holísticos v2 (Residual MLP + TCN) con augmentación ×4.

Uso:
  python train_holistic_model.py              # Entrena ambos
  python train_holistic_model.py --static-only
  python train_holistic_model.py --dynamic-only

Requiere datos recolectados con collect_data.py usando holistic_model.py
(features de 225 dimensiones). Los modelos v1 se respaldan automáticamente.
"""

import numpy as np
import os
import argparse
import json
import shutil
import tensorflow as tf
from sklearn.model_selection import train_test_split

from labels import STATIC_LABELS, DYNAMIC_LABELS, SEQUENCE_LENGTH
from data_augmentation import augment_sample, augment_sequence

BASE_DIR     = os.path.dirname(__file__)
STATIC_DIR   = os.path.join(BASE_DIR, "data", "static")
DYNAMIC_DIR  = os.path.join(BASE_DIR, "data", "dynamic")
MODEL_DIR    = os.path.join(BASE_DIR, "model")
MODEL_V1_DIR = os.path.join(MODEL_DIR, "v1")
MODEL_V2_DIR = os.path.join(MODEL_DIR, "v2")


def backup_v1():
    os.makedirs(MODEL_V1_DIR, exist_ok=True)
    for fname in ["sign_model.h5", "labels.npy", "sequence_model.h5", "seq_labels.npy"]:
        src = os.path.join(MODEL_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(MODEL_V1_DIR, fname))
            print(f"  Respaldado: {fname} → model/v1/")


def _residual_block(x, units_out):
    shortcut = tf.keras.layers.Dense(units_out)(x) if x.shape[-1] != units_out else x
    h = tf.keras.layers.Dense(units_out, activation="relu")(x)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dense(units_out)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    return tf.keras.layers.ReLU()(h + shortcut)


def build_residual_mlp(num_classes: int) -> tf.keras.Model:
    inp = tf.keras.layers.Input(shape=(225,))
    x   = tf.keras.layers.Dense(256, activation="relu")(inp)
    x   = tf.keras.layers.BatchNormalization()(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    x   = _residual_block(x, 256)
    x   = tf.keras.layers.Dropout(0.2)(x)
    x   = _residual_block(x, 128)
    x   = tf.keras.layers.Dropout(0.2)(x)
    x   = _residual_block(x, 64)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _tcn_block(x, filters: int, dilation: int):
    h = tf.keras.layers.Conv1D(filters, 3, dilation_rate=dilation, padding="causal", activation="relu")(x)
    h = tf.keras.layers.LayerNormalization()(h)
    h = tf.keras.layers.Dropout(0.1)(h)
    h = tf.keras.layers.Conv1D(filters, 3, dilation_rate=dilation, padding="causal")(h)
    h = tf.keras.layers.LayerNormalization()(h)
    shortcut = tf.keras.layers.Conv1D(filters, 1)(x) if x.shape[-1] != filters else x
    return tf.keras.layers.ReLU()(h + shortcut)


def build_tcn(num_classes: int) -> tf.keras.Model:
    inp = tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, 225))
    x   = inp
    for d in [1, 2, 4, 8]:
        x = _tcn_block(x, 64, d)
    x   = tf.keras.layers.GlobalAveragePooling1D()(x)
    x   = tf.keras.layers.Dense(64, activation="relu")(x)
    x   = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _calibrate_temperature(model, X_val, y_val):
    logits   = model.predict(X_val, verbose=0)
    best_T, best_nll = 1.0, float("inf")
    for T in np.arange(0.1, 5.1, 0.1):
        cal = tf.nn.softmax(logits / T).numpy()
        nll = -np.mean(np.log(cal[np.arange(len(y_val)), y_val] + 1e-9))
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T


def _callbacks(path):
    return [
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(path, save_best_only=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5),
    ]


def train_static():
    print("\n=== Modelo estático (Residual MLP 225→softmax) ===")
    X, y, found = [], [], []
    for label in STATIC_LABELS:
        folder = os.path.join(STATIC_DIR, label)
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder) if f.endswith(".npy")]
        if not files:
            continue
        idx = len(found)
        found.append(label)
        for f in files:
            s = np.load(os.path.join(folder, f))
            if s.shape == (225,):
                for aug in augment_sample(s):
                    X.append(aug); y.append(idx)

    if not X:
        print("Sin datos. Recolecta con collect_data.py usando holistic_model.py (225D).")
        return None, None, None

    X, y = np.array(X), np.array(y)
    print(f"  {len(X)} muestras aug | {len(found)} clases")
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = build_residual_mlp(len(found))
    save_path = os.path.join(MODEL_V2_DIR, "holistic_static.keras")
    model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=200,
              batch_size=64, callbacks=_callbacks(save_path), verbose=1)
    _, acc = model.evaluate(Xv, yv, verbose=0)
    T = _calibrate_temperature(model, Xv, yv)
    print(f"  Val accuracy: {acc:.2%}  |  T={T:.2f}")
    np.save(os.path.join(MODEL_V2_DIR, "labels_static.npy"), np.array(found))
    return model, found, T


def train_dynamic():
    print("\n=== Modelo dinámico (TCN 30×225→softmax) ===")
    X, y, found = [], [], []
    for label in DYNAMIC_LABELS:
        folder = os.path.join(DYNAMIC_DIR, label)
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder) if f.endswith(".npy")]
        if not files:
            continue
        idx = len(found)
        found.append(label)
        for f in files:
            s = np.load(os.path.join(folder, f))
            if s.shape == (SEQUENCE_LENGTH, 225):
                for aug in augment_sequence(s):
                    X.append(aug); y.append(idx)

    if not X:
        print("Sin datos dinámicos.")
        return None, None, None

    X, y = np.array(X), np.array(y)
    print(f"  {len(X)} secuencias aug | {len(found)} clases")
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_tcn(len(found))
    save_path = os.path.join(MODEL_V2_DIR, "holistic_dynamic.keras")
    model.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=200,
              batch_size=32, callbacks=_callbacks(save_path), verbose=1)
    _, acc = model.evaluate(Xv, yv, verbose=0)
    T = _calibrate_temperature(model, Xv, yv)
    print(f"  Val accuracy: {acc:.2%}  |  T={T:.2f}")
    np.save(os.path.join(MODEL_V2_DIR, "labels_dynamic.npy"), np.array(found))
    return model, found, T


def main():
    parser = argparse.ArgumentParser(description="Entrena modelos holísticos v2")
    parser.add_argument("--static-only",  action="store_true")
    parser.add_argument("--dynamic-only", action="store_true")
    args = parser.parse_args()

    print("Respaldando modelos v1...")
    backup_v1()
    os.makedirs(MODEL_V2_DIR, exist_ok=True)

    calibration = {}

    if not args.dynamic_only:
        _, _, T = train_static()
        if T is not None:
            calibration["temperature"] = T

    if not args.static_only:
        _, _, T = train_dynamic()
        if T is not None:
            calibration["temperature_dynamic"] = T

    if calibration:
        cal_path = os.path.join(MODEL_V2_DIR, "calibration.json")
        with open(cal_path, "w") as f:
            json.dump(calibration, f, indent=2)
        print(f"\nCalibración guardada: {cal_path}")

    print(f"\nListo. Activar con: MODEL_VERSION=v2 uvicorn main:app --reload")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add backend/train_holistic_model.py
git commit -m "feat: train_holistic_model — unified training script with Residual MLP + TCN + augmentation"
```

---

## Task 10: Update sign_model.py and sequence_model.py

**Files:**
- Modify: `backend/sign_model.py`
- Modify: `backend/sequence_model.py`

- [ ] **Step 1: Replace sign_model.py**

Replace full content of `backend/sign_model.py`:
```python
import numpy as np
import config
from logger import get_logger
from holistic_model import extract_static
from sign_classifier import predict_static

log = get_logger(__name__)


def extract_landmarks(image):
    """Backward-compatible: retorna shape (21,3) o None."""
    features = extract_static(image)
    if features is None:
        return None
    right = features[63:126].reshape(21, 3)
    left  = features[0:63].reshape(21, 3)
    return right if np.any(right != 0) else (left if np.any(left != 0) else None)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    pts = landmarks.copy()
    pts -= pts[0]
    pts /= max(float(np.linalg.norm(pts[9])), 1e-6)
    return pts.flatten()


def predict_sign(image) -> dict:
    features = extract_static(image)
    if features is None:
        log.debug("Sin manos detectadas.")
        return {"prediction": "No detectado", "confidence": 0.0}
    result = predict_static(features)
    log.info(f"Predicción estática: {result['prediction']} ({result['confidence']:.0%})")
    return result
```

- [ ] **Step 2: Replace sequence_model.py**

Replace full content of `backend/sequence_model.py`:
```python
import numpy as np
from logger import get_logger
from holistic_model import extract_video
from temporal_model import predict_dynamic
from session_manager import session_manager
from labels import SEQUENCE_LENGTH

log = get_logger(__name__)

_DEFAULT_SESSION = "default"


def add_frame(image, session_id: str = _DEFAULT_SESSION) -> bool:
    features = extract_video(image)
    if features is None:
        return False
    return session_manager.add_landmark_frame(session_id, features)


def predict_sequence(session_id: str = _DEFAULT_SESSION) -> dict:
    buffer = session_manager.get_landmark_buffer(session_id)
    frames = len(buffer)
    if frames < SEQUENCE_LENGTH:
        return {"prediction": "Capturando...", "confidence": 0.0, "frames": frames}
    sequence = np.array(buffer)
    result   = predict_dynamic(sequence)
    result["frames"] = frames
    log.info(f"Predicción dinámica: {result['prediction']} ({result['confidence']:.0%})")
    return result


def clear_buffer(session_id: str = _DEFAULT_SESSION):
    session_manager.clear_landmark_buffer(session_id)
    log.debug(f"Buffer reiniciado: {session_id}")
```

- [ ] **Step 3: Run full test suite**

```bash
cd backend && python -m pytest tests/ -v
```
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add backend/sign_model.py backend/sequence_model.py
git commit -m "refactor: sign_model + sequence_model use HolisticExtractor and SessionManager"
```

---

## Task 11: Create nlp_pipeline.py

**Files:**
- Create: `backend/nlp_pipeline.py`
- Create: `backend/tests/test_nlp_pipeline.py`

- [ ] **Step 1: Write failing tests**

Create `backend/tests/test_nlp_pipeline.py`:
```python
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

def test_translate_empty_list():
    import asyncio
    from nlp_pipeline import translate_to_spanish
    result = asyncio.run(translate_to_spanish([]))
    assert result == ""

def test_translate_single_sign():
    import asyncio
    from nlp_pipeline import translate_to_spanish
    result = asyncio.run(translate_to_spanish(["HOLA"]))
    assert result.lower() == "hola"

def test_translate_fallback_without_api_key(monkeypatch):
    import asyncio
    import config
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    from nlp_pipeline import translate_to_spanish
    result = asyncio.run(translate_to_spanish(["YO", "COMER"]))
    assert "YO" in result or "COMER" in result
```

- [ ] **Step 2: Run to confirm they fail**

```bash
cd backend && python -m pytest tests/test_nlp_pipeline.py -v
```
Expected: ImportError.

- [ ] **Step 3: Create nlp_pipeline.py**

Create `backend/nlp_pipeline.py`:
```python
import config
from logger import get_logger

log = get_logger(__name__)

_INVALID = {"NO RECONOCIDO", "NO DETECTADO", "CAPTURANDO...", "MODELO NO DISPONIBLE", "ERROR"}

_SYSTEM_PROMPT = (
    "Eres un intérprete especializado en Lengua de Señas Mexicana (LSM). "
    "Recibirás una secuencia de señas detectadas en orden LSM (SOV — Sujeto-Objeto-Verbo). "
    "Tradúcelas a una oración natural en español considerando:\n"
    "- LSM usa orden SOV; español usa SVO — reordena adecuadamente\n"
    "- Las negaciones se expresan al final en LSM\n"
    "- Omite artículos y preposiciones que no existen en LSM\n"
    "- Infiere tiempos verbales del contexto si no hay marcadores temporales\n"
    "- Si la secuencia es 1-2 señas, retorna la traducción directa\n"
    "Responde SOLO con la oración traducida. Sin explicaciones."
)


def normalize_signs(signs: list[str]) -> list[str]:
    result, last = [], None
    for s in signs:
        upper = s.upper()
        if upper in _INVALID:
            continue
        if upper != (last or "").upper():
            result.append(s)
            last = s
    return result


async def translate_to_spanish(signs: list[str]) -> str:
    normalized = normalize_signs(signs)
    if not normalized:
        return ""
    if len(normalized) == 1:
        return normalized[0].capitalize()
    if not config.ANTHROPIC_API_KEY:
        return " ".join(normalized)
    try:
        import anthropic
        client   = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        sequence = " → ".join(normalized)
        msg      = await client.messages.create(
            model=config.NLP_MODEL,
            max_tokens=256,
            system=[{
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": f"Señas LSM: {sequence}"}],
        )
        return msg.content[0].text.strip()
    except Exception as exc:
        log.error(f"Error Claude API: {exc}")
        return " ".join(normalized)
```

- [ ] **Step 4: Run tests**

```bash
cd backend && python -m pytest tests/test_nlp_pipeline.py -v
```
Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/nlp_pipeline.py backend/tests/test_nlp_pipeline.py
git commit -m "feat: nlp_pipeline — LSM→Spanish translator with Claude API and prompt caching"
```

---

## Task 12: Update text_to_sign.py (cache + spellcheck + semantic + thumbnail)

**Files:**
- Modify: `backend/text_to_sign.py`
- Test: `backend/tests/test_text_to_sign.py`

- [ ] **Step 1: Write new tests**

Add to `backend/tests/test_text_to_sign.py` (or create if missing):
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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
```

- [ ] **Step 2: Run to confirm new tests fail**

```bash
cd backend && python -m pytest tests/test_text_to_sign.py -v
```
Expected: FAIL (missing thumbnail_base64, has_real_image keys).

- [ ] **Step 3: Replace text_to_sign.py**

Replace full content of `backend/text_to_sign.py`:
```python
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

_words_cache: list[str] | None = None
_words_cache_size: int = 0
_semantic_model    = None
_sign_embeddings   = None
_sign_words_indexed: list[str] = []


def _get_words_cache() -> list[str]:
    global _words_cache, _words_cache_size
    current = count_signs()
    if _words_cache is None or current != _words_cache_size:
        _words_cache     = [s["word"] for s in get_all_signs()]
        _words_cache_size = current
    return _words_cache


def invalidate_words_cache():
    global _words_cache, _words_cache_size, _sign_embeddings, _sign_words_indexed
    _words_cache         = None
    _words_cache_size    = 0
    _sign_embeddings     = None
    _sign_words_indexed  = []


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _spellcheck(word: str) -> str:
    try:
        from spellchecker import SpellChecker
        spell = SpellChecker(language="es")
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
        words = [s["word"] for s in all_signs]
        if words != _sign_words_indexed:
            _sign_embeddings    = _semantic_model.encode(words, convert_to_tensor=True)
            _sign_words_indexed = words
        query_emb = _semantic_model.encode(query, convert_to_tensor=True)
        scores    = util.cos_sim(query_emb, _sign_embeddings)[0]
        best_idx  = int(scores.argmax())
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
```

- [ ] **Step 4: Run tests**

```bash
cd backend && python -m pytest tests/test_text_to_sign.py -v
```
Expected: all PASS.

- [ ] **Step 5: Run full suite**

```bash
cd backend && python -m pytest tests/ -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/text_to_sign.py backend/tests/test_text_to_sign.py
git commit -m "feat: text_to_sign — semantic search, spell-check, words cache, thumbnail generation"
```

---

## Task 13: Improve seed_dictionary.py + DB migration

**Files:**
- Modify: `backend/seed_dictionary.py`

- [ ] **Step 1: Read current seed_dictionary.py**

```bash
cd backend && head -60 seed_dictionary.py
```

- [ ] **Step 2: Replace create_placeholder function**

Find and replace the `create_placeholder` function in `backend/seed_dictionary.py` with:
```python
_CATEGORY_COLORS = {
    "saludos":    (52, 152, 219),
    "cortesias":  (46, 204, 113),
    "familia":    (155, 89, 182),
    "numeros":    (230, 126, 34),
    "colores":    (231, 76, 60),
    "abecedario": (52, 73, 94),
    "preguntas":  (241, 196, 15),
    "tiempo":     (26, 188, 156),
    "verbos":     (211, 84, 0),
    "default":    (127, 140, 141),
}

def create_placeholder(word: str, filename: str, category: str = "default"):
    from PIL import Image, ImageDraw, ImageFont
    size   = 200
    color  = _CATEGORY_COLORS.get(category, _CATEGORY_COLORS["default"])
    img    = Image.new("RGB", (size, size), color)
    draw   = ImageDraw.Draw(img)

    # Fondo con gradiente simulado (banda más clara)
    for i in range(size // 2):
        alpha = int(255 * (i / (size // 2)) * 0.3)
        band_color = tuple(min(255, c + alpha) for c in color)
        draw.rectangle([0, i, size, i + 1], fill=band_color)

    # Icono de mano simplificado (círculos)
    cx, cy = size // 2, size // 2 - 20
    draw.ellipse([cx - 25, cy - 30, cx + 25, cy + 10], fill=(255, 255, 255, 180), outline="white", width=2)
    for dx in [-20, -8, 4, 16]:
        draw.rectangle([cx + dx, cy - 50, cx + dx + 8, cy - 15], fill="white")
    draw.rectangle([cx - 28, cy - 35, cx - 14, cy - 10], fill="white")

    # Texto: palabra
    word_display = word.replace("_", " ").upper()
    font_size = max(14, 28 - max(0, len(word_display) - 6) * 2)
    try:
        font  = ImageFont.truetype("arial.ttf", font_size)
        small = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font  = ImageFont.load_default()
        small = font

    bbox = draw.textbbox((0, 0), word_display, font=font)
    tw   = bbox[2] - bbox[0]
    draw.text(((size - tw) // 2, size - 55), word_display, font=font, fill="white")

    cat_text = f"#{category}"
    bbox2    = draw.textbbox((0, 0), cat_text, font=small)
    tw2      = bbox2[2] - bbox2[0]
    draw.text(((size - tw2) // 2, size - 25), cat_text, font=small, fill=(220, 220, 220))

    dest = ASSETS_DIR / filename
    img.save(str(dest))
```

- [ ] **Step 3: Run full test suite to confirm no regressions**

```bash
cd backend && python -m pytest tests/ -v
```
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add backend/seed_dictionary.py
git commit -m "feat: seed_dictionary — category-colored placeholders with hand icon"
```

---

## Task 14: Add WebSocket + NLP endpoints to main.py

**Files:**
- Modify: `backend/main.py`
- Test: `backend/tests/test_api.py`

- [ ] **Step 1: Add WebSocket and NLP imports to main.py**

Add at the top of `backend/main.py`, after existing imports:
```python
from fastapi import WebSocket, WebSocketDisconnect
import base64 as _b64
import json as _json
import asyncio

import nlp_pipeline
from session_manager import session_manager
from holistic_model import extract_video
from sign_classifier import predict_static
from temporal_model import predict_dynamic
```

- [ ] **Step 2: Add updated PhraseResponse and SignImageResponse models**

Replace the `SignImageResponse` and `PhraseResponse` Pydantic models in `backend/main.py`:
```python
class SignImageResponse(BaseModel):
    found:            bool
    word:             str
    path:             str | None = None
    base64:           str | None = None
    thumbnail_base64: str | None = None
    media_type:       str | None = None
    category:         str | None = None
    suggestion:       str | None = None
    has_real_image:   bool       = False


class PhraseResponse(BaseModel):
    phrase:           str
    translation_hint: str | None = None
    signs:            list[SignImageResponse]
    total:            int
    found_count:      int        = 0
    not_found:        list[str]  = []
```

- [ ] **Step 3: Add /signs/{word}/image endpoint**

Add after the `list_signs` endpoint:
```python
@app.get(
    "/signs/{word}/image",
    tags=["Diccionario"],
    summary="Obtener imagen de una seña directamente",
)
def sign_image_direct(word: str):
    from fastapi.responses import FileResponse
    from database import get_sign as db_get_sign
    sign = db_get_sign(word)
    if not sign or not sign.get("filename"):
        raise HTTPException(status_code=404, detail=f"Seña '{word}' no encontrada.")
    full_path = config.ASSETS_DIR / sign["filename"]
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Imagen no disponible.")
    return FileResponse(str(full_path), media_type=sign.get("media_type", "image/png"))
```

- [ ] **Step 4: Update text_to_sign_phrase endpoint**

Replace the `text_to_sign_phrase` endpoint:
```python
@app.get(
    "/text-to-sign-phrase/{phrase}",
    response_model=PhraseResponse,
    tags=["Traducción"],
    summary="Traducir frase a señas con galería de imágenes",
)
@limiter.limit(config.RATE_PHRASE)
def text_to_sign_phrase(request: Request, phrase: str):
    words = phrase.strip().split()
    if not words:
        raise HTTPException(status_code=422, detail="La frase no puede estar vacía.")
    signs       = [get_sign_image(w) for w in words]
    found_count = sum(1 for s in signs if s["found"])
    not_found   = [s["word"] for s in signs if not s["found"]]
    save_translation(type_="text_to_sign_phrase", input_=phrase, output=f"{found_count}/{len(signs)} señas")
    return {
        "phrase":           phrase,
        "translation_hint": phrase,
        "signs":            signs,
        "total":            len(signs),
        "found_count":      found_count,
        "not_found":        not_found,
    }
```

- [ ] **Step 5: Add NLP REST endpoints**

Add after the sequence endpoints:
```python
# ── NLP ───────────────────────────────────────────────────────────────────────

class NLPTranslateBody(BaseModel):
    session_id: str


@app.post(
    "/nlp/translate",
    tags=["NLP"],
    summary="Traducir buffer de señas LSM → español con Claude",
)
async def nlp_translate(body: NLPTranslateBody):
    signs = session_manager.get_signs(body.session_id)
    if not signs:
        return {"signs": [], "translation": "", "confidence_avg": 0.0}
    translation = await nlp_pipeline.translate_to_spanish(signs)
    save_translation(type_="sign_to_text", input_="[nlp]", output=translation)
    return {"signs": signs, "translation": translation, "confidence_avg": 0.0}


@app.delete(
    "/nlp/clear",
    tags=["NLP"],
    summary="Limpiar buffer de señas de la sesión",
)
def nlp_clear(session_id: str):
    session_manager.clear(session_id)
    return {"message": "Buffer limpiado"}
```

- [ ] **Step 6: Add WebSocket /ws/detect endpoint**

Add at the end of `backend/main.py`, before the final lines:
```python
# ── WebSocket tiempo real ─────────────────────────────────────────────────────

@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket, sid: str = "anon"):
    await websocket.accept()
    log.info(f"WS conectado: sid={sid}")
    last_static = ""

    try:
        while True:
            raw = await websocket.receive_text()
            msg = _json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "frame":
                raw_data = msg.get("data", "")
                _, _, encoded = raw_data.partition(",")
                img_bytes = _b64.b64decode(encoded if encoded else raw_data)
                image     = _decode_image(img_bytes)

                features = extract_video(image)
                if features is None:
                    await websocket.send_json({"type": "status", "message": "Mano no detectada"})
                    continue

                # Predicción estática
                static_result = predict_static(features)
                sign       = static_result["prediction"]
                confidence = static_result["confidence"]

                if sign not in ("No reconocido", "No detectado", "Modelo no disponible", "Error"):
                    if sign != last_static:
                        last_static = sign
                        session_manager.add_sign(sid, sign)
                        await websocket.send_json({
                            "type": "sign", "sign": sign,
                            "confidence": confidence, "mode": "static",
                        })

                # Buffer dinámico
                ready = session_manager.add_landmark_frame(sid, features)
                if ready:
                    import numpy as np
                    buf = np.array(session_manager.get_landmark_buffer(sid))
                    dyn = predict_dynamic(buf)
                    session_manager.clear_landmark_buffer(sid)
                    if dyn["prediction"] not in ("No reconocido", "Modelo no disponible", "Error"):
                        session_manager.add_sign(sid, dyn["prediction"])
                        await websocket.send_json({
                            "type": "sign", "sign": dyn["prediction"],
                            "confidence": dyn["confidence"], "mode": "dynamic",
                        })

            elif msg_type == "translate":
                signs = session_manager.get_signs(sid)
                if not signs:
                    await websocket.send_json({"type": "sentence", "signs": [], "text": "", "confidence_avg": 0.0})
                    continue
                translation = await nlp_pipeline.translate_to_spanish(signs)
                save_translation(type_="sign_to_text", input_="[websocket]", output=translation)
                await websocket.send_json({
                    "type": "sentence", "signs": signs,
                    "text": translation, "confidence_avg": 0.0,
                })

            elif msg_type == "clear":
                session_manager.clear(sid)
                last_static = ""
                await websocket.send_json({"type": "status", "message": "Buffer limpiado"})

    except WebSocketDisconnect:
        session_manager.clear(sid)
        log.info(f"WS desconectado: sid={sid}")
    except Exception as exc:
        log.error(f"Error WS {sid}: {exc}")
        try:
            await websocket.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            pass
```

- [ ] **Step 7: Add API tests for new endpoints**

Add to `backend/tests/test_api.py`:
```python
def test_nlp_clear_endpoint():
    r = client.delete("/nlp/clear?session_id=test_session")
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
```

- [ ] **Step 8: Run all tests**

```bash
cd backend && python -m pytest tests/ -v
```
Expected: all PASS.

- [ ] **Step 9: Commit**

```bash
git add backend/main.py backend/tests/test_api.py
git commit -m "feat: WebSocket /ws/detect real-time, NLP endpoints, phrase gallery response, /signs/{word}/image"
```

---

## Task 15: Update also upsert_sign to set has_real_image

**Files:**
- Modify: `backend/database.py`
- Modify: `backend/main.py` (upsert call in add_sign)

- [ ] **Step 1: Update upsert_sign in database.py**

Replace `upsert_sign` in `backend/database.py`:
```python
def upsert_sign(word: str, filename: str, media_type: str = "image/png",
                category: str = None, has_real_image: bool = False):
    conn = _get_conn()
    conn.execute(
        """INSERT INTO signs (word, filename, media_type, category, has_real_image, created_at)
           VALUES (?,?,?,?,?,?)
           ON CONFLICT(word) DO UPDATE SET
               filename=excluded.filename,
               media_type=excluded.media_type,
               category=excluded.category,
               has_real_image=excluded.has_real_image""",
        (word, filename, media_type, category, int(has_real_image),
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
```

- [ ] **Step 2: Update add_sign endpoint to pass has_real_image=True when file is uploaded**

In `backend/main.py`, update the `upsert_sign` call inside `add_sign`:
```python
    upsert_sign(word=word, filename=filename, media_type=media_type,
                category=category, has_real_image=file is not None)
```

Also call `invalidate_words_cache()` after upsert so semantic search picks up new signs:
```python
    from text_to_sign import invalidate_words_cache
    invalidate_words_cache()
```

And in the `remove_sign` endpoint, also invalidate:
```python
@app.delete("/signs/{word}", ...)
def remove_sign(word: str, username: str = Depends(get_current_user)):
    if not delete_sign(word):
        raise HTTPException(status_code=404, detail=f"Seña '{word}' no encontrada.")
    from text_to_sign import invalidate_words_cache
    invalidate_words_cache()
    return {"message": f"Seña '{word}' eliminada."}
```

- [ ] **Step 3: Run full test suite**

```bash
cd backend && python -m pytest tests/ -v
```
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add backend/database.py backend/main.py
git commit -m "feat: upsert_sign tracks has_real_image, invalidates semantic cache on sign add/remove"
```

---

## Task 16: Frontend — SignToTextScreen.js WebSocket

**Files:**
- Modify: `frontend/screens/SignToTextScreen.js`

- [ ] **Step 1: Read current SignToTextScreen.js**

Read `frontend/screens/SignToTextScreen.js` to understand current structure.

- [ ] **Step 2: Replace HTTP frame logic with WebSocket**

In `frontend/screens/SignToTextScreen.js`, add WebSocket state and replace the frame-capture logic:

Add imports at the top:
```javascript
import React, { useState, useRef, useEffect, useCallback } from 'react';
```

Add WebSocket state after existing state declarations:
```javascript
const wsRef = useRef(null);
const [wsConnected, setWsConnected] = useState(false);
const [accumulatedSigns, setAccumulatedSigns] = useState([]);
const [liveSign, setLiveSign] = useState('');
const sessionId = useRef(`session_${Date.now()}_${Math.random().toString(36).substr(2,9)}`);
```

Add WebSocket connect/disconnect functions:
```javascript
const connectWebSocket = useCallback(() => {
  const wsUrl = API_URL.replace('http://', 'ws://').replace('https://', 'wss://');
  const ws = new WebSocket(`${wsUrl}/ws/detect?sid=${sessionId.current}`);
  
  ws.onopen = () => {
    setWsConnected(true);
    console.log('WS conectado');
  };
  
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'sign') {
      setLiveSign(msg.sign);
      setAccumulatedSigns(prev => {
        const last = prev[prev.length - 1];
        if (last === msg.sign) return prev;
        return [...prev, msg.sign];
      });
    } else if (msg.type === 'sentence') {
      setPrediction(msg.text || msg.signs.join(' '));
      setLiveSign('');
    } else if (msg.type === 'status') {
      console.log('WS status:', msg.message);
    }
  };
  
  ws.onerror = (e) => console.error('WS error:', e);
  ws.onclose = () => { setWsConnected(false); wsRef.current = null; };
  
  wsRef.current = ws;
}, []);

const disconnectWebSocket = useCallback(() => {
  if (wsRef.current) {
    wsRef.current.close();
    wsRef.current = null;
    setWsConnected(false);
  }
}, []);

const requestTranslation = useCallback(() => {
  if (wsRef.current?.readyState === WebSocket.OPEN) {
    wsRef.current.send(JSON.stringify({ type: 'translate' }));
  }
}, []);

const clearSession = useCallback(() => {
  if (wsRef.current?.readyState === WebSocket.OPEN) {
    wsRef.current.send(JSON.stringify({ type: 'clear' }));
  }
  setAccumulatedSigns([]);
  setLiveSign('');
  setPrediction('');
}, []);
```

Replace the frame-capture interval (the HTTP polling loop) with WebSocket frame sending:
```javascript
const sendFrame = useCallback(async (cameraRef) => {
  if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
  try {
    const photo = await cameraRef.current.takePictureAsync({
      quality: 0.3, base64: true, skipProcessing: true,
    });
    wsRef.current.send(JSON.stringify({
      type: 'frame',
      data: `data:image/jpeg;base64,${photo.base64}`,
    }));
  } catch (e) {
    console.warn('Error capturando frame:', e);
  }
}, []);
```

Add `useEffect` to start sending frames when WS is connected and camera is active:
```javascript
useEffect(() => {
  if (!wsConnected || !cameraRef?.current) return;
  const interval = setInterval(() => sendFrame(cameraRef), 150); // ~6fps
  return () => clearInterval(interval);
}, [wsConnected, sendFrame]);
```

Add connect button and translation button to the UI (replace existing detect button area):
```javascript
{/* Botón conectar/desconectar WS */}
<TouchableOpacity onPress={wsConnected ? disconnectWebSocket : connectWebSocket}
  style={[styles.button, wsConnected && styles.buttonActive]}>
  <Text style={styles.buttonText}>
    {wsConnected ? '⏹ Detener' : '▶ Iniciar detección en tiempo real'}
  </Text>
</TouchableOpacity>

{/* Señas acumuladas */}
{accumulatedSigns.length > 0 && (
  <View style={styles.signsContainer}>
    <Text style={styles.signsLabel}>Señas detectadas:</Text>
    <Text style={styles.signsText}>{accumulatedSigns.join(' → ')}</Text>
    <TouchableOpacity onPress={requestTranslation} style={styles.translateButton}>
      <Text style={styles.buttonText}>Traducir a español</Text>
    </TouchableOpacity>
    <TouchableOpacity onPress={clearSession} style={styles.clearButton}>
      <Text style={styles.clearText}>Limpiar</Text>
    </TouchableOpacity>
  </View>
)}

{/* Seña en vivo */}
{liveSign !== '' && (
  <Text style={styles.liveSign}>{liveSign}</Text>
)}
```

- [ ] **Step 3: Add missing styles for new UI elements**

In the `StyleSheet.create({})` block of `SignToTextScreen.js`, add:
```javascript
buttonActive:       { backgroundColor: '#e74c3c' },
signsContainer:     { backgroundColor: 'rgba(0,0,0,0.7)', borderRadius: 8, padding: 12, marginTop: 8 },
signsLabel:         { color: '#aaa', fontSize: 12, marginBottom: 4 },
signsText:          { color: 'white', fontSize: 16, fontWeight: 'bold', flexWrap: 'wrap' },
translateButton:    { backgroundColor: '#27ae60', borderRadius: 6, padding: 8, marginTop: 8, alignItems: 'center' },
clearButton:        { marginTop: 6, alignItems: 'center' },
clearText:          { color: '#aaa', fontSize: 12 },
liveSign:           { fontSize: 32, fontWeight: 'bold', color: '#3498db', textAlign: 'center', marginTop: 8 },
```

- [ ] **Step 4: Commit**

```bash
git add frontend/screens/SignToTextScreen.js
git commit -m "feat: SignToTextScreen — real-time WebSocket sign detection replacing HTTP frame-by-frame"
```

---

## Task 17: Frontend — TextToSignScreen.js Image Gallery

**Files:**
- Modify: `frontend/screens/TextToSignScreen.js`

- [ ] **Step 1: Read current TextToSignScreen.js**

Read `frontend/screens/TextToSignScreen.js` to understand structure.

- [ ] **Step 2: Update API call to use new phrase response fields**

In the function that calls `/text-to-sign-phrase/{phrase}`, update the response handling to use `found_count` and `not_found`:
```javascript
const handleTranslate = async () => {
  if (!inputText.trim()) return;
  setLoading(true);
  try {
    const encoded = encodeURIComponent(inputText.trim());
    const res = await axios.get(`${API_URL}/text-to-sign-phrase/${encoded}`);
    const data = res.data;
    setSigns(data.signs);
    setFoundCount(data.found_count);
    setNotFound(data.not_found || []);
    setTotalCount(data.total);
  } catch (e) {
    console.error(e);
    Alert.alert('Error', 'No se pudo obtener las señas.');
  } finally {
    setLoading(false);
  }
};
```

Add state variables:
```javascript
const [signs, setSigns] = useState([]);
const [foundCount, setFoundCount] = useState(0);
const [notFound, setNotFound] = useState([]);
const [totalCount, setTotalCount] = useState(0);
const [selectedSign, setSelectedSign] = useState(null);
```

- [ ] **Step 3: Replace sign display with horizontal gallery**

Replace the signs display area with:
```javascript
{signs.length > 0 && (
  <View style={styles.galleryContainer}>
    <Text style={styles.galleryHeader}>
      {foundCount}/{totalCount} señas encontradas
    </Text>
    
    <ScrollView horizontal showsHorizontalScrollIndicator={false}
      style={styles.galleryScroll}>
      {signs.map((sign, index) => (
        <TouchableOpacity key={index} style={styles.signThumbnail}
          onPress={() => setSelectedSign(sign)}>
          {sign.found && sign.thumbnail_base64 ? (
            <Image source={{ uri: sign.thumbnail_base64 }}
              style={styles.thumbnailImg} resizeMode="contain" />
          ) : (
            <View style={styles.thumbnailPlaceholder}>
              <Text style={styles.thumbnailNotFound}>?</Text>
            </View>
          )}
          <Text style={styles.thumbnailWord} numberOfLines={1}>
            {sign.word}
          </Text>
          {!sign.found && (
            <Text style={styles.thumbnailUnavailable}>No disponible</Text>
          )}
          {sign.found && !sign.has_real_image && (
            <Text style={styles.thumbnailIllustrative}>Ilustrativa</Text>
          )}
        </TouchableOpacity>
      ))}
    </ScrollView>

    {notFound.length > 0 && (
      <Text style={styles.notFoundText}>
        Sin seña: {notFound.join(', ')}
      </Text>
    )}
  </View>
)}

{/* Modal imagen completa */}
<Modal visible={selectedSign !== null} transparent animationType="fade"
  onRequestClose={() => setSelectedSign(null)}>
  <TouchableOpacity style={styles.modalOverlay} onPress={() => setSelectedSign(null)}>
    {selectedSign?.base64 ? (
      <Image source={{ uri: selectedSign.base64 }} style={styles.modalImage}
        resizeMode="contain" />
    ) : (
      <View style={styles.modalNoImage}>
        <Text style={styles.modalNoImageText}>Seña no disponible: {selectedSign?.word}</Text>
      </View>
    )}
    <Text style={styles.modalWord}>{selectedSign?.word}</Text>
    {selectedSign?.category && (
      <Text style={styles.modalCategory}>#{selectedSign.category}</Text>
    )}
  </TouchableOpacity>
</Modal>
```

Add import at top: `import { Modal } from 'react-native';`

- [ ] **Step 4: Add gallery styles**

Add to the StyleSheet:
```javascript
galleryContainer:    { marginTop: 16 },
galleryHeader:       { fontSize: 13, color: '#888', marginBottom: 8, textAlign: 'center' },
galleryScroll:       { flexDirection: 'row' },
signThumbnail:       { width: 110, marginRight: 10, alignItems: 'center', backgroundColor: '#f5f5f5', borderRadius: 10, padding: 6 },
thumbnailImg:        { width: 90, height: 90, borderRadius: 8 },
thumbnailPlaceholder:{ width: 90, height: 90, borderRadius: 8, backgroundColor: '#ddd', justifyContent: 'center', alignItems: 'center' },
thumbnailNotFound:   { fontSize: 28, color: '#e74c3c' },
thumbnailWord:       { fontSize: 12, fontWeight: 'bold', marginTop: 4, textAlign: 'center' },
thumbnailUnavailable:{ fontSize: 10, color: '#e74c3c', textAlign: 'center' },
thumbnailIllustrative:{ fontSize: 9, color: '#888', textAlign: 'center' },
notFoundText:        { color: '#e74c3c', fontSize: 12, marginTop: 8, textAlign: 'center' },
modalOverlay:        { flex: 1, backgroundColor: 'rgba(0,0,0,0.85)', justifyContent: 'center', alignItems: 'center' },
modalImage:          { width: 280, height: 280, borderRadius: 12 },
modalNoImage:        { width: 280, height: 280, borderRadius: 12, backgroundColor: '#333', justifyContent: 'center', alignItems: 'center' },
modalNoImageText:    { color: 'white', textAlign: 'center', padding: 16 },
modalWord:           { color: 'white', fontSize: 22, fontWeight: 'bold', marginTop: 12 },
modalCategory:       { color: '#aaa', fontSize: 14, marginTop: 4 },
```

- [ ] **Step 5: Commit**

```bash
git add frontend/screens/TextToSignScreen.js
git commit -m "feat: TextToSignScreen — horizontal image gallery, modal, not_found indicators"
```

---

## Final Verification

- [ ] **Start backend and confirm all endpoints respond**

```bash
cd backend && uvicorn main:app --reload --port 8000
```

Check:
- `GET http://localhost:8000/` → `{"message": "HandTalk API v2 funcionando"}`
- `GET http://localhost:8000/docs` → Swagger UI loads with new endpoints visible
- `GET http://localhost:8000/signs` → Returns sign list
- `GET http://localhost:8000/signs/hola/image` → Returns PNG image
- `GET http://localhost:8000/text-to-sign-phrase/hola%20gracias` → Returns `found_count`, `not_found`, `thumbnail_base64`
- WebSocket `ws://localhost:8000/ws/detect?sid=test` → Accepts connection

- [ ] **Run complete test suite**

```bash
cd backend && python -m pytest tests/ -v --tb=short
```
Expected: all tests PASS, no failures.

- [ ] **Final commit**

```bash
git add -A
git commit -m "chore: HandTalk AI upgrade complete — CV Holistic + Residual MLP + TCN + WebSocket + NLP + gallery"
```
