"""
Capa de datos con SQLite (sin dependencias externas).
Tablas: translations, users, signs
"""

import sqlite3
from datetime import datetime, timezone
import config
from logger import get_logger

log    = get_logger(__name__)
DB_PATH = config.DB_PATH


def _get_conn():
    from pathlib import Path
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


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
    for col_sql in [
        "ALTER TABLE translations ADD COLUMN user_id TEXT",
        "ALTER TABLE signs ADD COLUMN has_real_image INTEGER NOT NULL DEFAULT 0",
    ]:
        try:
            conn.execute(col_sql)
            conn.commit()
        except Exception:
            pass
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_translations_user ON translations(user_id)")
        conn.commit()
    except Exception:
        pass
    conn.close()
    log.info(f"Base de datos lista: {DB_PATH}")


# ── Traducciones ──────────────────────────────────────────────────────────────

def save_translation(type_: str, input_: str, output: str, confidence: float = None, user_id: str = None):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO translations (type, input, output, confidence, created_at, user_id) VALUES (?,?,?,?,?,?)",
        (type_, input_, output, confidence, datetime.now(timezone.utc).isoformat(), user_id),
    )
    conn.commit()
    conn.close()


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


# ── Usuarios ──────────────────────────────────────────────────────────────────

def create_user(username: str, hashed_password: str) -> bool:
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO users (username, hashed_password, created_at) VALUES (?,?,?)",
            (username, hashed_password, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        log.info(f"Usuario creado: {username}")
        return True
    except sqlite3.IntegrityError:
        log.warning(f"Intento de crear usuario duplicado: {username}")
        return False
    finally:
        conn.close()


def get_user(username: str):
    conn = _get_conn()
    row  = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ── Señas ─────────────────────────────────────────────────────────────────────

def upsert_sign(word: str, filename: str, media_type: str = "image/png", category: str = None):
    conn = _get_conn()
    conn.execute(
        """INSERT INTO signs (word, filename, media_type, category, created_at)
           VALUES (?,?,?,?,?)
           ON CONFLICT(word) DO UPDATE SET
               filename=excluded.filename,
               media_type=excluded.media_type,
               category=excluded.category""",
        (word, filename, media_type, category, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def get_sign(word: str):
    conn = _get_conn()
    row  = conn.execute("SELECT * FROM signs WHERE word=?", (word,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_signs(category: str = None):
    conn = _get_conn()
    if category:
        rows = conn.execute("SELECT * FROM signs WHERE category=? ORDER BY word", (category,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM signs ORDER BY word").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_sign(word: str) -> bool:
    conn = _get_conn()
    cur  = conn.execute("DELETE FROM signs WHERE word=?", (word,))
    conn.commit()
    conn.close()
    deleted = cur.rowcount > 0
    if deleted:
        log.info(f"Seña eliminada: {word}")
    return deleted


def count_signs() -> int:
    conn = _get_conn()
    n    = conn.execute("SELECT COUNT(*) FROM signs").fetchone()[0]
    conn.close()
    return n
