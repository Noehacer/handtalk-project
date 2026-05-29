import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import cv2
import numpy as np
import base64 as _b64
import json as _json

import config
from logger import get_logger
from sign_model import predict_sign
from text_to_sign import get_sign_image
from sequence_model import add_frame, predict_sequence, clear_buffer
from database import (
    init_db, save_translation, get_history,
    upsert_sign, get_all_signs, delete_sign, count_signs,
)
from auth import register, login, get_current_user

import nlp_pipeline
from session_manager import session_manager
from holistic_model import extract_video
from sign_classifier import predict_static
from temporal_model import predict_dynamic as _predict_dynamic

log = get_logger(__name__)

config.ASSETS_DIR.mkdir(exist_ok=True)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="HandTalk API",
    version="2.0.0",
    description="API REST para traducción bidireccional de Lengua de Señas Mexicana.",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory=str(config.ASSETS_DIR)), name="assets")
init_db()

import asyncio as _asyncio

@app.on_event("startup")
async def _start_cleanup_task():
    async def _cleanup_loop():
        while True:
            await _asyncio.sleep(300)  # every 5 minutes
            session_manager.cleanup_stale(max_age_seconds=300)
            log.debug("SessionManager: limpieza de sesiones inactivas completada.")
    _asyncio.create_task(_cleanup_loop())


# ── Schemas / Response models ─────────────────────────────────────────────────

class AuthBody(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6)


class SignBody(BaseModel):
    word:       str
    category:   str | None = None
    media_type: str        = "image/png"


class PredictionResponse(BaseModel):
    prediction: str   = Field(description="Seña detectada o 'No detectado'")
    confidence: float = Field(ge=0.0, le=1.0, description="Confianza del modelo (0–1)")


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


class NLPTranslateBody(BaseModel):
    session_id: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_image(contents: bytes) -> np.ndarray:
    npimg = np.frombuffer(contents, np.uint8)
    img   = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=422, detail="No se pudo decodificar la imagen.")
    return img


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.post(
    "/auth/register",
    status_code=201,
    tags=["Auth"],
    summary="Crear cuenta",
    description="Registra un nuevo usuario con username y contraseña (mínimo 6 caracteres).",
)
def auth_register(body: AuthBody):
    if not register(body.username, body.password):
        raise HTTPException(status_code=409, detail="El usuario ya existe.")
    return {"message": "Usuario registrado correctamente."}


@app.post(
    "/auth/login",
    tags=["Auth"],
    summary="Iniciar sesión",
    description="Devuelve un JWT válido por 24 horas para usar en endpoints protegidos.",
)
def auth_login(body: AuthBody):
    token = login(body.username, body.password)
    if token is None:
        raise HTTPException(status_code=401, detail="Credenciales incorrectas.")
    return {"access_token": token, "token_type": "bearer"}


# ── General ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"], summary="Estado de la API")
def root():
    return {"message": "HandTalk API v2 funcionando", "docs": "/docs"}


# ── Detección estática ────────────────────────────────────────────────────────

@app.post(
    "/predict-sign",
    response_model=PredictionResponse,
    tags=["Detección"],
    summary="Detectar seña en imagen",
    description=(
        "Recibe una imagen (multipart/form-data) y retorna la seña detectada "
        "junto con el nivel de confianza. Límite: 30 peticiones/minuto por IP."
    ),
)
@limiter.limit(config.RATE_PREDICT)
async def predict(request: Request, file: UploadFile = File(...)):
    image  = _decode_image(await file.read())
    result = predict_sign(image)
    save_translation(
        type_="sign_to_text",
        input_="[imagen]",
        output=result["prediction"],
        confidence=result["confidence"],
    )
    log.info(f"POST /predict-sign → {result['prediction']} ({result['confidence']:.0%})")
    return result


# ── Detección dinámica (secuencias) ──────────────────────────────────────────

@app.post(
    "/predict-sign-sequence/frame",
    tags=["Detección"],
    summary="Agregar frame a la secuencia",
    description="Envía frames uno a uno. Cuando `ready=true` puedes llamar a /predict.",
)
@limiter.limit(config.RATE_SEQ_FRAME)
async def add_sequence_frame(request: Request, file: UploadFile = File(...)):
    image = _decode_image(await file.read())
    ready = add_frame(image)
    return {"ready": ready}


@app.get(
    "/predict-sign-sequence/predict",
    tags=["Detección"],
    summary="Predecir seña dinámica",
    description="Usa los frames acumulados para predecir señas en movimiento (J, Z, etc.).",
)
def predict_dynamic():
    result  = predict_sequence()
    finished = result.get("prediction") not in ("Capturando...", "Modelo no disponible", "No reconocido")
    if finished and result.get("confidence", 0) > 0:
        save_translation(
            type_="sign_to_text_dynamic",
            input_="[secuencia]",
            output=result["prediction"],
            confidence=result.get("confidence"),
        )
    return result


@app.delete(
    "/predict-sign-sequence/reset",
    tags=["Detección"],
    summary="Reiniciar buffer de secuencia",
)
def reset_sequence():
    clear_buffer()
    return {"message": "Buffer reiniciado"}


# ── NLP ───────────────────────────────────────────────────────────────────────

@app.post(
    "/nlp/translate",
    tags=["NLP"],
    summary="Traducir buffer de señas LSM → español con Claude",
)
async def nlp_translate(body: NLPTranslateBody, username: str = Depends(get_current_user)):
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
def nlp_clear(session_id: str, username: str = Depends(get_current_user)):
    session_manager.clear(session_id)
    return {"message": "Buffer limpiado"}


# ── Texto → Señas ─────────────────────────────────────────────────────────────

@app.get(
    "/text-to-sign/{word}",
    response_model=SignImageResponse,
    tags=["Traducción"],
    summary="Traducir palabra a seña",
    description=(
        "Busca la imagen de la seña para una palabra. "
        "Usa búsqueda exacta, normalizada (sin acentos) y por similitud. "
        "Incluye `image_base64` para mostrar sin segunda petición."
    ),
)
@limiter.limit(config.RATE_TEXT_TO_SIGN)
def text_to_sign(request: Request, word: str):
    data = get_sign_image(word)
    if data["found"]:
        save_translation(type_="text_to_sign", input_=word, output=data["word"])
    return data


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
    save_translation(type_="text_to_sign_phrase", input_=phrase,
                     output=f"{found_count}/{len(signs)} señas")
    return {
        "phrase":           phrase,
        "translation_hint": phrase,
        "signs":            signs,
        "total":            len(signs),
        "found_count":      found_count,
        "not_found":        not_found,
    }


# ── Diccionario ───────────────────────────────────────────────────────────────

@app.get("/signs", tags=["Diccionario"], summary="Listar todas las señas")
def list_signs(category: str | None = None):
    signs = get_all_signs(category)
    return {"total": len(signs), "signs": signs}


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


@app.get("/signs/stats", tags=["Diccionario"], summary="Estadísticas del diccionario")
def signs_stats():
    all_signs = get_all_signs()
    by_category: dict = {}
    for s in all_signs:
        cat = s.get("category") or "sin_categoria"
        by_category[cat] = by_category.get(cat, 0) + 1
    return {"total": len(all_signs), "by_category": by_category}


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
    # CR-05: validate allowed media types / extensions
    ALLOWED_MEDIA_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
    if media_type not in ALLOWED_MEDIA_TYPES:
        raise HTTPException(status_code=422, detail=f"Tipo de media no permitido: {media_type}. Permitidos: {sorted(ALLOWED_MEDIA_TYPES)}")

    # CR-03: sanitise word to prevent path traversal
    import re
    safe_word = re.sub(r"[^\w\-]", "_", word.strip())
    if not safe_word or safe_word != word.strip():
        raise HTTPException(status_code=422, detail="La palabra contiene caracteres no permitidos.")

    ext      = media_type.split("/")[-1]
    filename = f"{safe_word}.{ext}"
    dest     = config.ASSETS_DIR / filename

    # Confirm resolved path is inside ASSETS_DIR
    try:
        dest.resolve().relative_to(config.ASSETS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=422, detail="Nombre de archivo inválido.")

    if file:
        contents = await file.read()
        dest.write_bytes(contents)
        log.info(f"Imagen subida: {filename} ({len(contents)} bytes) por {username}")
    elif not dest.exists():
        try:
            from seed_dictionary import create_placeholder
            create_placeholder(safe_word, filename, category or "default")
            log.info(f"Placeholder generado: {filename}")
        except Exception as exc:
            log.warning(f"No se pudo generar placeholder: {exc}")

    upsert_sign(word=safe_word, filename=filename, media_type=media_type,
                category=category, has_real_image=file is not None)
    from text_to_sign import invalidate_words_cache
    invalidate_words_cache()
    return {"message": f"Seña '{safe_word}' guardada.", "filename": filename}


@app.delete(
    "/signs/{word}",
    tags=["Diccionario"],
    summary="Eliminar una seña (requiere auth)",
)
def remove_sign(word: str, username: str = Depends(get_current_user)):
    if not delete_sign(word):
        raise HTTPException(status_code=404, detail=f"Seña '{word}' no encontrada.")
    from text_to_sign import invalidate_words_cache
    invalidate_words_cache()
    return {"message": f"Seña '{word}' eliminada."}


# ── Historial ─────────────────────────────────────────────────────────────────

@app.get(
    "/history",
    tags=["Historial"],
    summary="Ver historial de traducciones (requiere auth)",
    description="Retorna las últimas N traducciones. Header requerido: Authorization: Bearer <token>",
)
def history(limit: int = 20, username: str = Depends(get_current_user)):
    return {"user": username, "history": get_history(limit, user_id=username)}


# ── WebSocket tiempo real ─────────────────────────────────────────────────────

@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket, sid: str = "anon", token: str = ""):
    # CR-01: validate JWT before accepting connection
    if token:
        try:
            from auth import verify_token
            _username = verify_token(token)
        except Exception:
            await websocket.close(code=4001)
            log.warning(f"WS rechazado: token inválido para sid={sid}")
            return
    await websocket.accept()
    log.info(f"WS conectado: sid={sid}")
    last_static = ""

    try:
        while True:
            raw      = await websocket.receive_text()
            msg      = _json.loads(raw)
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

                # Static prediction
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

                # Dynamic buffer
                ready = session_manager.add_landmark_frame(sid, features)
                if ready:
                    import numpy as np
                    buf = np.array(session_manager.get_landmark_buffer(sid))
                    dyn = _predict_dynamic(buf)
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
                    await websocket.send_json({
                        "type": "sentence", "signs": [], "text": "", "confidence_avg": 0.0
                    })
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
            await websocket.send_json({"type": "error", "detail": "Error interno del servidor."})
        except Exception:
            pass
