"""
Configuración centralizada del backend.
Todos los parámetros ajustables están aquí; el resto del código los importa.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ── Paths ──────────────────────────────────────────────────────────────────────
ASSETS_DIR   = BASE_DIR / "assets"
DATA_DIR     = BASE_DIR / "data"
MODEL_DIR    = BASE_DIR / "model"
DB_PATH      = BASE_DIR / "db" / "handtalk.db"

STATIC_MODEL_PATH  = MODEL_DIR / "sign_model.h5"
STATIC_LABELS_PATH = MODEL_DIR / "labels.npy"
SEQ_MODEL_PATH     = MODEL_DIR / "sequence_model.h5"
SEQ_LABELS_PATH    = MODEL_DIR / "seq_labels.npy"

# ── MediaPipe ─────────────────────────────────────────────────────────────────
MIN_DETECTION_CONFIDENCE = float(os.getenv("MIN_DETECTION_CONFIDENCE", "0.5"))
MIN_TRACKING_CONFIDENCE  = float(os.getenv("MIN_TRACKING_CONFIDENCE",  "0.5"))

# ── Umbrales del modelo ───────────────────────────────────────────────────────
STATIC_CONFIDENCE_THRESHOLD  = float(os.getenv("STATIC_CONFIDENCE_THRESHOLD",  "0.6"))
DYNAMIC_CONFIDENCE_THRESHOLD = float(os.getenv("DYNAMIC_CONFIDENCE_THRESHOLD", "0.7"))
FUZZY_MATCH_CUTOFF           = float(os.getenv("FUZZY_MATCH_CUTOFF",           "0.75"))

# ── Auth ──────────────────────────────────────────────────────────────────────
import warnings as _w
SECRET_KEY = os.getenv("SECRET_KEY", "handtalk-dev-insecure-key-change-in-prod")
if SECRET_KEY == "handtalk-dev-insecure-key-change-in-prod":
    if os.getenv("ENVIRONMENT", "development") == "production":
        raise RuntimeError(
            "SECRET_KEY no configurada. Establece SECRET_KEY en producción."
        )
    _w.warn("⚠  Usando SECRET_KEY insegura. Configura SECRET_KEY antes de desplegar.", stacklevel=1)
ALGORITHM          = "HS256"
TOKEN_EXPIRE_HOURS = int(os.getenv("TOKEN_EXPIRE_HOURS", "24"))

# ── CORS ──────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8081,http://localhost:19006").split(",")

# ── Rate limits ───────────────────────────────────────────────────────────────
# Estático continuo: 1 foto cada 2s = 30/min → 60/min da margen
RATE_PREDICT      = "60/minute"
# Dinámico: frames a ~5 fps × 60s = 300/min → 600/min da margen
RATE_SEQ_FRAME    = "600/minute"
RATE_TEXT_TO_SIGN = "60/minute"
RATE_PHRASE       = "30/minute"

# ── Model versioning ──────────────────────────────────────────────────────────
MODEL_VERSION           = os.getenv("MODEL_VERSION", "v2")
MODEL_V1_DIR            = MODEL_DIR / "v1"
MODEL_V2_DIR            = MODEL_DIR / "v2"
HOLISTIC_STATIC_PATH    = MODEL_V2_DIR / "holistic_static.keras"
HOLISTIC_STATIC_LABELS  = MODEL_V2_DIR / "labels_static.npy"
HOLISTIC_DYNAMIC_PATH   = MODEL_V2_DIR / "holistic_dynamic.keras"
HOLISTIC_DYNAMIC_LABELS = MODEL_V2_DIR / "labels_dynamic.npy"
CALIBRATION_PATH        = MODEL_V2_DIR / "calibration.json"

# ── Anthropic / NLP ───────────────────────────────────────────────────────────
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
NLP_MODEL          = "claude-haiku-4-5-20251001"
SEMANTIC_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.65"))

# ── WebSocket thresholds ──────────────────────────────────────────────────────
WS_STATIC_THRESHOLD  = float(os.getenv("WS_STATIC_THRESHOLD",  "0.6"))
WS_DYNAMIC_THRESHOLD = float(os.getenv("WS_DYNAMIC_THRESHOLD", "0.7"))
