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
SECRET_KEY         = os.getenv("SECRET_KEY", "handtalk-dev-insecure-key-change-in-prod")
ALGORITHM          = "HS256"
TOKEN_EXPIRE_HOURS = int(os.getenv("TOKEN_EXPIRE_HOURS", "24"))

# ── CORS ──────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ── Rate limits ───────────────────────────────────────────────────────────────
# Estático continuo: 1 foto cada 2s = 30/min → 60/min da margen
RATE_PREDICT      = "60/minute"
# Dinámico: frames a ~5 fps × 60s = 300/min → 600/min da margen
RATE_SEQ_FRAME    = "600/minute"
RATE_TEXT_TO_SIGN = "60/minute"
RATE_PHRASE       = "30/minute"
