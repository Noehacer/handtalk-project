# HandTalk — Diseño de Mejoras: CV + ML + NLP
**Fecha:** 2026-05-28  
**Versión:** 1.0  
**Estado:** Aprobado

---

## 1. Contexto y Objetivo

HandTalk es un traductor bidireccional de Lengua de Señas Mexicana (LSM) con backend FastAPI y app React Native. La versión actual (2.0) tiene cinco limitaciones principales:

1. **Precisión baja** — Dense NN sobre 63 features (1 mano, sin cara)
2. **Sin gramática LSM** — traduce seña por seña, sin construir oraciones
3. **Bugs críticos** — buffer global compartido, history sin filtro por usuario, datetime deprecado
4. **Sin tiempo real** — la detección seña→texto funciona foto por foto (HTTP request por frame), no como conversación fluida
5. **Sin imágenes en texto→seña** — las respuestas de frase muestran texto plano, sin imágenes visuales de cada seña

El objetivo de esta mejora es implementar un pipeline profesional de CV + ML + NLP que aborde las cinco limitaciones sin romper la arquitectura existente.

---

## 2. Bugs a Corregir

| # | Archivo | Línea | Bug | Fix |
|---|---------|-------|-----|-----|
| 1 | `database.py` | 58 | `datetime.utcnow()` deprecado en Python 3.12+ | `datetime.now(timezone.utc)` |
| 2 | `main.py` | 311 | `/history` retorna traducciones de todos los usuarios | Agregar `user_id` a tabla `translations` y filtrar |
| 3 | `sequence_model.py` | 23 | Buffer global compartido entre todos los usuarios | Mover buffer a `session_manager.py` con clave por sesión |
| 4 | `sign_model.py` | 53 | Si scale=0 tras centrar, normalización omitida → input sucio | Fallback: escala a 1.0 si `scale <= 0` |
| 5 | `config.py` | 37 | `ALLOWED_ORIGINS="*"` por defecto en producción | Cambiar default a `"http://localhost:8081"` |
| 6 | `text_to_sign.py` | 50 | `get_all_signs()` completo en cada búsqueda fuzzy | Cachear lista de palabras en memoria con invalidación |
| 7 | `main.py` | 265 | `add_sign` mezcla query params con multipart form | Mover `word`, `category`, `media_type` a Form fields |

---

## 3. Capa de Visión Artificial (CV)

### 3.1 Reemplazo de MediaPipe Hands → Holistic

**Archivo nuevo:** `backend/holistic_model.py`

Reemplaza el uso de `mp.solutions.hands` en `sign_model.py` y `sequence_model.py` por `mp.solutions.holistic`, que extrae simultáneamente:

| Fuente | Landmarks | Features (×3 coords) |
|--------|-----------|----------------------|
| Mano izquierda | 21 | 63 |
| Mano derecha | 21 | 63 |
| Cara (key points) | 33 | 99 |
| **Total** | **75** | **225** |

Si una mano no está presente en el frame, su segmento de 63 features se rellena con ceros. El modelo aprende a manejar señas de una o dos manos con la misma arquitectura.

**Interfaz pública:**
```python
class HolisticExtractor:
    def extract(self, image: np.ndarray) -> np.ndarray | None
    # Retorna vector (225,) o None si no hay manos detectadas

    def preprocess_image(self, image: np.ndarray) -> np.ndarray
    # Aplica CLAHE para mejorar contraste adaptativo
```

### 3.2 Normalización mejorada

- Centrar cada mano en su propia muñeca (landmark 0 de cada mano)
- Escalar por distancia muñeca→MCP (landmark 9) de la mano dominante detectada
- Si `scale <= 0` (bug #4), usar `scale = 1.0` como fallback en lugar de omitir

### 3.3 Preprocesamiento de imagen

Aplicar CLAHE (`cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`) antes de pasar a MediaPipe. Mejora detección en baja iluminación sin costo computacional significativo.

### 3.4 Data Augmentation

**Archivo nuevo:** `backend/data_augmentation.py`

Transforma cada muestra de landmarks para multiplicar el dataset ×4:

- `mirror_landmarks(v)` — espeja coordenadas X (simula mano izquierda)
- `add_noise(v, sigma=0.01)` — ruido gaussiano para robustez
- `scale_landmarks(v, factor)` — escala aleatoria ±10%
- `augment_sample(v)` → lista de 4 variantes por muestra original

---

## 4. Capa de Machine Learning (ML)

### 4.1 Modelo estático: Residual MLP

**Archivo nuevo:** `backend/sign_classifier.py`  
**Script nuevo:** `backend/train_holistic_model.py`

Reemplaza la Dense NN de `train_model.py`. Arquitectura:

```
Input(225)
  → Dense(256, relu) + BatchNorm + Dropout(0.3)
  → ResidualBlock(256 → 128)
  → ResidualBlock(128 → 64)
  → Dense(num_classes, softmax)
```

`ResidualBlock(in, out)`:
```
x → Dense(out) + BN + ReLU → Dense(out) + BN
shortcut: Dense(out, linear) si in != out
output: relu(main + shortcut)
```

**Mejora estimada:** +20–35% de precisión sobre Dense NN con 63 features.

### 4.2 Modelo dinámico: TCN

**Archivo nuevo:** `backend/temporal_model.py`

Reemplaza el LSTM de `train_sequence_model.py`. Arquitectura:

```
Input(30, 225)
  → TCNBlock(filters=64, dilation=1)
  → TCNBlock(filters=64, dilation=2)
  → TCNBlock(filters=64, dilation=4)
  → TCNBlock(filters=64, dilation=8)
  → GlobalAveragePooling1D
  → Dense(64, relu) + Dropout(0.3)
  → Dense(num_classes, softmax)
```

`TCNBlock`: Conv1D causal + WeightNorm + ReLU + Dropout + residual connection.

**Ventajas sobre LSTM:**
- Paralelizable → entrena ~3× más rápido
- Campo receptivo exponencial con dilaciones
- Sin problemas de gradientes evanescentes

### 4.3 Versioning de modelos

```
backend/model/
  v1/          ← backup automático de modelos actuales
  v2/          ← nuevos modelos
    holistic_static.keras
    holistic_dynamic.keras
    labels_static.npy
    labels_dynamic.npy
    calibration.json
  current.txt  ← contiene "v2" (configurable por env var MODEL_VERSION)
```

`config.py` lee `MODEL_VERSION = os.getenv("MODEL_VERSION", "v2")` para seleccionar la carpeta activa. Rollback inmediato cambiando la variable de entorno.

### 4.4 Calibración de confianza (Temperature Scaling)

Post-entrenamiento, optimizar temperatura `T` en set de validación:

```python
confidence_calibrated = softmax(logits / T)
```

`T` y sus métricas de calibración se guardan en `model/v2/calibration.json`. Hace que `confidence=0.7` signifique realmente "acierto en 70% de casos".

### 4.5 Script de entrenamiento unificado

**Archivo nuevo:** `backend/train_holistic_model.py`

Entrena ambos modelos (estático + dinámico) en un solo script con:
- Carga de dataset + augmentation automática (×4)
- Features holísticas (225D)
- Early stopping (patience=20) + ReduceLROnPlateau
- Backup de modelos v1 antes de guardar v2
- Exporta `calibration.json` con temperatura T y ECE score

---

## 5. Capa NLP

### 5.1 Acumulador de señas por sesión

**Archivo nuevo:** `backend/nlp_pipeline.py`  
**Archivo nuevo:** `backend/session_manager.py`

`session_manager.py` gestiona buffers por sesión, eliminando el buffer global que causaba el bug #3.

**Dos tipos de clave de sesión:**
- **Endpoints existentes** (`/predict-sign-sequence/frame`): usan la IP del cliente como clave (backwards compatible, no requiere cambio en el frontend)
- **Endpoints NLP nuevos** (`/nlp/add-sign`, `/nlp/translate`): usan UUID explícito que el cliente genera y envía en el body

`session_manager.py` gestiona buffers por sesión (UUID o IP):

```python
class SessionManager:
    _sessions: dict[str, SessionData]

    def add_landmark_frame(session_id: str, frame: np.ndarray) -> bool
    def add_detected_sign(session_id: str, sign: str)
    def get_sign_buffer(session_id: str) -> list[str]
    def clear_session(session_id: str)
    def cleanup_stale(max_age_seconds=300)  # GC automático
```

`SessionData` contiene: buffer de landmarks (deque), buffer de señas detectadas, último timestamp.

### 5.2 Traductor LSM → Español con Claude API

`nlp_pipeline.py` implementa:

```python
class LSMTranslator:
    def normalize_sign_sequence(signs: list[str]) -> list[str]
    # Elimina repeticiones, mapea a palabras canónicas LSM

    async def translate_to_spanish(signs: list[str]) -> str
    # Llama Claude API (haiku-4-5-20251001) con prompt caching
```

**System prompt (con cache):**
```
Eres un intérprete especializado en Lengua de Señas Mexicana (LSM).
Recibirás una secuencia de señas detectadas en orden LSM (SOV).
Tradúcelas a una oración natural en español considerando:
- LSM usa orden SOV; español usa SVO — reordena adecuadamente
- Las negaciones se expresan al final en LSM
- Omite artículos/preposiciones que no existen en LSM
- Infiere tiempos verbales del contexto si no hay marcadores temporales
Responde SOLO con la oración traducida. Sin explicaciones. Sin notas.
```

Usa `claude-haiku-4-5-20251001` con `cache_control: ephemeral` en el system prompt para reducir costo ~90%.

### 5.3 Búsqueda semántica texto → seña

**En:** `backend/text_to_sign.py` (4to paso del lookup)

Modelo: `paraphrase-multilingual-MiniLM-L12-v2` de `sentence-transformers`.

- Al iniciar, genera embeddings para todas las palabras del diccionario
- Se actualiza cuando se agregan/eliminan señas vía API
- Búsqueda por similitud coseno (threshold: 0.65) como paso 4 después de exact/normalized/fuzzy
- Ejemplo: "carro" → encuentra "automóvil" si el diccionario tiene esa forma

### 5.4 Corrector ortográfico

**En:** `backend/text_to_sign.py` (preprocesamiento previo al lookup)

Usa `pyspellchecker` con diccionario español. Si la palabra no existe en el diccionario y el corrector sugiere una alternativa con distancia Levenshtein ≤ 2, se usa la sugerencia automáticamente y se incluye `"suggestion"` en la respuesta.

### 5.5 Detección en tiempo real: WebSocket `/ws/detect`

**Este es el modo primario de detección.** Reemplaza el flujo foto-por-foto (HTTP POST por frame) con una conexión WebSocket permanente que se comporta como una conversación fluida.

**Flujo de comunicación:**

```
[Frontend]                          [Backend]
   │── CONNECT /ws/detect?sid=uuid ──▶│
   │                                  │
   │── { type: "frame",               │
   │     data: "<base64 JPEG>" } ────▶│ 1. Decodifica imagen
   │                                  │ 2. HolisticExtractor → 225 features
   │                                  │ 3. Residual MLP (estático)
   │◀─ { type: "sign",                │ 4. Si confianza > umbral:
   │     sign: "HOLA",                │    publica predicción inmediata
   │     confidence: 0.91 } ──────────│
   │                                  │ 5. Acumula en buffer TCN (dinámico)
   │── (más frames) ─────────────────▶│
   │◀─ { type: "sign", sign: "YO" } ──│
   │◀─ { type: "sign", sign: "COMER"}─│
   │                                  │
   │── { type: "translate" } ────────▶│ 6. Claude API: LSM → español
   │◀─ { type: "sentence",            │
   │     signs: ["HOLA","YO","COMER"],│
   │     text: "Hola, yo quiero comer"│
   │     confidence_avg: 0.88 } ──────│
```

**Protocolo de mensajes:**

| Dirección | Tipo | Payload |
|-----------|------|---------|
| Cliente → Servidor | `frame` | `{ data: base64JPEG }` |
| Cliente → Servidor | `translate` | `{}` (solicita traducción del buffer) |
| Cliente → Servidor | `clear` | `{}` (limpia buffer de la sesión) |
| Servidor → Cliente | `sign` | `{ sign, confidence, mode: "static"\|"dynamic" }` |
| Servidor → Cliente | `sentence` | `{ signs[], text, confidence_avg }` |
| Servidor → Cliente | `status` | `{ message }` (ej. "Mano no detectada") |
| Servidor → Cliente | `error` | `{ detail }` |

**Latencia objetivo:** ≤ 200ms desde que llega el frame hasta que se envía la predicción.

**Lógica de predicción en tiempo real:**
- Cada frame se evalúa con el modelo estático (Residual MLP)
- Si `confidence ≥ STATIC_THRESHOLD (0.6)` y la seña es diferente a la última predicha: se publica inmediatamente como `type: "sign"`
- En paralelo, el frame se agrega al buffer TCN (dinámico)
- Cuando el buffer TCN llega a 30 frames y la predicción supera `DYNAMIC_THRESHOLD (0.7)`: también se publica como `type: "sign"` con `mode: "dynamic"`
- La traducción de la oración completa solo se ejecuta cuando el cliente envía `type: "translate"` (control explícito del usuario)

**Los endpoints HTTP de frames existentes se mantienen** (`/predict-sign`, `/predict-sign-sequence/frame`) para compatibilidad con versiones anteriores del frontend, pero el nuevo frontend usará exclusivamente WebSocket.

### 5.6 Nuevos endpoints REST NLP

| Endpoint | Método | Auth | Descripción |
|----------|--------|------|-------------|
| `/nlp/translate` | POST | No | Traduce buffer LSM → español con Claude (alternativa REST al WS) |
| `/nlp/clear` | DELETE | No | Limpia el buffer de la sesión |

**Body `/nlp/translate`:**
```json
{ "session_id": "uuid" }
```

**Response `/nlp/translate`:**
```json
{
  "signs": ["YO", "COMER", "QUERER"],
  "translation": "Yo quiero comer",
  "confidence_avg": 0.84
}
```

---

## 6. Texto → Seña con Galería de Imágenes

### 6.1 Problema actual

El endpoint `/text-to-sign-phrase/{phrase}` retorna un JSON con `base64` por palabra, pero:
- Las imágenes son **placeholders de color** generados por `seed_dictionary.py` (rectángulos con texto), no fotos reales de señas
- El frontend no tiene un componente de galería para mostrarlas ordenadamente
- No hay soporte para señas animadas (GIF) en frases

### 6.2 Enriquecimiento del diccionario visual

**En `seed_dictionary.py`:** Mejorar `create_placeholder()` para generar imágenes más informativas mientras no haya fotos reales:
- Fondo con gradiente de color por categoría
- Icono de mano SVG rasterizado (Pillow) en lugar de solo texto
- Nombre de la seña con fuente grande y legible
- Categoría como subtítulo

**Nueva columna en tabla `signs`:** `has_real_image BOOLEAN DEFAULT 0`
- Cuando se sube una imagen real vía `POST /signs`, se marca `has_real_image = 1`
- El frontend puede mostrar un indicador visual (ej. borde dorado) cuando es imagen real vs placeholder

### 6.3 Endpoint de frase mejorado: `/text-to-sign-phrase/{phrase}`

**Response model ampliado (`PhraseResponse`):**
```json
{
  "phrase": "hola como estas",
  "translation_hint": "Hola ¿cómo estás?",
  "signs": [
    {
      "found": true,
      "word": "hola",
      "index": 0,
      "base64": "data:image/png;base64,...",
      "thumbnail_base64": "data:image/png;base64,...",
      "media_type": "image/png",
      "category": "saludos",
      "has_real_image": false,
      "suggestion": null
    },
    {
      "found": true,
      "word": "como",
      "index": 1,
      "base64": "data:image/png;base64,...",
      "thumbnail_base64": "data:image/png;base64,...",
      "media_type": "image/png",
      "category": "preguntas",
      "has_real_image": false,
      "suggestion": null
    }
  ],
  "total": 2,
  "found_count": 2,
  "not_found": []
}
```

**Nuevo campo `thumbnail_base64`:** versión reducida (120×120 px) de la imagen, generada en el servidor con Pillow al vuelo. Permite al frontend mostrar una tira de miniaturas sin cargar las imágenes completas hasta que el usuario toque una.

**Nuevo campo `not_found`:** lista de palabras sin seña en el diccionario, para que el frontend las muestre claramente en lugar de ignorarlas silenciosamente.

**Nuevo campo `translation_hint`:** la misma frase pasada por el corrector ortográfico y normalizada, para que el usuario vea qué entendió el sistema.

### 6.4 Endpoint nuevo: `GET /signs/{word}/image`

Retorna la imagen de una seña directamente como respuesta HTTP (Content-Type: image/png o image/gif), útil para mostrar en `<Image>` de React Native con una URL directa en lugar de base64 en el JSON.

```
GET /signs/hola/image
→ 200 Content-Type: image/png (binary)
→ 404 si no existe
```

### 6.5 Flujo texto→seña en el frontend (`TextToSignScreen.js`)

El frontend ya existe. Cambios necesarios:

1. Llamar al endpoint de frase mejorado
2. Mostrar **galería horizontal scrollable** con thumbnails
3. Al tocar un thumbnail: mostrar la imagen completa en modal
4. Palabras en `not_found`: mostrar en rojo con texto "Seña no disponible"
5. Si `has_real_image = false`: mostrar chip "Imagen ilustrativa"

---

## 7. Cambios en Base de Datos

### 7.1 Migración de tabla `translations`

Agregar columna `user_id` para filtrar historial por usuario (bug #2):

```sql
ALTER TABLE translations ADD COLUMN user_id TEXT;
CREATE INDEX IF NOT EXISTS idx_translations_user ON translations(user_id);
```

`save_translation()` acepta `user_id` opcional. `get_history()` filtra por `user_id` si se provee.

### 7.2 Migración de tabla `signs`

Agregar columna `has_real_image` para distinguir imágenes reales de placeholders:

```sql
ALTER TABLE signs ADD COLUMN has_real_image INTEGER NOT NULL DEFAULT 0;
```

### 7.3 Nueva tabla `nlp_sessions`

Persistencia ligera de traducciones NLP para historial enriquecido:

```sql
CREATE TABLE IF NOT EXISTS nlp_sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    signs       TEXT NOT NULL,  -- JSON array
    translation TEXT,
    created_at  TEXT NOT NULL
);
```

---

## 8. Cambios en `requirements.txt` y `config.py`

Agregar a `requirements.txt`:
```
anthropic>=0.40.0
sentence-transformers>=3.0.0
pyspellchecker>=0.8.0
uvicorn[standard]
websockets
```

Agregar a `config.py`:
```python
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
NLP_MODEL          = "claude-haiku-4-5-20251001"
SEMANTIC_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_VERSION      = os.getenv("MODEL_VERSION", "v2")
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.65"))
WS_STATIC_THRESHOLD  = float(os.getenv("WS_STATIC_THRESHOLD",  "0.6"))
WS_DYNAMIC_THRESHOLD = float(os.getenv("WS_DYNAMIC_THRESHOLD", "0.7"))
```

---

## 9. Archivos a Crear / Modificar

### Nuevos archivos
| Archivo | Propósito |
|---------|-----------|
| `backend/holistic_model.py` | MediaPipe Holistic extractor — 2 manos + cara (225 features) |
| `backend/data_augmentation.py` | Augmentation de landmarks para entrenamiento (×4) |
| `backend/sign_classifier.py` | Residual MLP para señas estáticas |
| `backend/temporal_model.py` | TCN para señas dinámicas |
| `backend/train_holistic_model.py` | Script unificado de entrenamiento con augmentation |
| `backend/session_manager.py` | Gestión de buffers por sesión (elimina global) |
| `backend/nlp_pipeline.py` | Acumulador de señas + traductor LSM→español con Claude |

### Archivos modificados
| Archivo | Cambios principales |
|---------|---------------------|
| `backend/sign_model.py` | Usar HolisticExtractor, fix normalización scale=0 |
| `backend/sequence_model.py` | Usar SessionManager + HolisticExtractor |
| `backend/text_to_sign.py` | Búsqueda semántica + spell check + caché + thumbnail + not_found |
| `backend/seed_dictionary.py` | Mejorar `create_placeholder()` con diseño por categoría |
| `backend/database.py` | Fix datetime, user_id en translations, has_real_image en signs, tabla nlp_sessions |
| `backend/main.py` | WebSocket `/ws/detect`, fix history, fix add_sign, endpoints NLP + imagen directa |
| `backend/config.py` | Nuevas variables: Claude, sentence-transformers, WS thresholds, model version |
| `backend/requirements.txt` | anthropic, sentence-transformers, pyspellchecker, websockets |
| `frontend/screens/SignToTextScreen.js` | Reemplazar HTTP frame-by-frame → WebSocket en tiempo real |
| `frontend/screens/TextToSignScreen.js` | Galería horizontal de imágenes, modal, chips "no disponible" |

---

## 10. Criterios de Éxito

1. **Tiempo real**: WebSocket `/ws/detect` envía predicciones en ≤200ms por frame, sin esperar botón
2. **Conversación fluida**: el usuario hace señas naturalmente y ve el texto acumularse en pantalla
3. **Galería de señas**: texto→seña muestra imágenes reales (o placeholders mejorados) en galería scrollable con modal al tocar
4. **Dos manos**: detección con ambas manos sin error, expresiones faciales incluidas en features
5. **History filtrado**: `/history` retorna solo las traducciones del usuario autenticado
6. **Sin interferencia**: dos usuarios simultáneos en WebSocket no se corrompen mutuamente
7. **NLP**: `/nlp/translate` retorna oración en español natural aplicando gramática LSM
8. **Búsqueda semántica**: texto→seña encuentra sinónimos (ej. "carro" → "automóvil")
9. **Regresión cero**: todos los tests existentes siguen pasando
10. **Rollback**: cambiar `MODEL_VERSION=v1` restaura modelos anteriores sin reiniciar código
