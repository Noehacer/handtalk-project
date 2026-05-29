# HandTalk

Traductor bidireccional de Lengua de Señas Mexicana (LSM).  
Detecta señas con la cámara del celular y traduce texto a imágenes de señas.

## Stack

| Capa | Tecnología |
|---|---|
| App móvil | React Native + Expo SDK 53 |
| Backend API | FastAPI + uvicorn |
| Visión | MediaPipe Hands + OpenCV |
| ML (señas estáticas) | TensorFlow / Keras — red Dense |
| ML (señas dinámicas) | TensorFlow / Keras — red LSTM |
| Base de datos | SQLite (sqlite3 nativo) |
| Auth | JWT (python-jose) + bcrypt |

---

## Estructura del proyecto

```
handtalk_project/
├── backend/
│   ├── main.py                  # API REST (FastAPI)
│   ├── sign_model.py            # Extracción de landmarks + modelo estático
│   ├── sequence_model.py        # Modelo LSTM para señas dinámicas
│   ├── text_to_sign.py          # Búsqueda de señas con fuzzy matching
│   ├── database.py              # SQLite (traducciones, usuarios, señas)
│   ├── auth.py                  # JWT + bcrypt
│   ├── labels.py                # Clases del modelo
│   ├── collect_data.py          # Herramienta de recolección con webcam
│   ├── train_model.py           # Entrenamiento señas estáticas
│   ├── train_sequence_model.py  # Entrenamiento señas dinámicas
│   ├── seed_dictionary.py       # Poblar BD con 130+ señas + placeholders
│   ├── requirements.txt
│   ├── Dockerfile
│   └── tests/
│       ├── conftest.py          # Mocks de TF/OpenCV/MediaPipe para CI
│       └── test_api.py          # Tests de integración
├── frontend/
│   ├── App.js                   # Navegación + splash + onboarding
│   ├── config.js                # URL del servidor (lee de app.json)
│   ├── theme.js                 # Paleta light/dark
│   ├── i18n.js                  # Traducciones ES/EN
│   ├── storage.js               # Historial local (AsyncStorage)
│   ├── app.json                 # Configuración Expo (cambia apiUrl aquí)
│   ├── eas.json                 # Configuración EAS Build
│   ├── components/
│   │   └── Logo.js
│   └── screens/
│       ├── OnboardingScreen.js
│       ├── HomeScreen.js
│       ├── SignToTextScreen.js
│       ├── TextToSignScreen.js
│       └── HistoryScreen.js
├── docker-compose.yml
├── render.yaml
└── .github/workflows/ci.yml
```

---

## Desarrollo local

### 1. Backend

```bash
cd backend

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Instalar dependencias
pip install -r requirements.txt

# Copiar y configurar variables de entorno
cp .env.example .env

# Poblar diccionario (130 señas + imágenes placeholder)
python seed_dictionary.py

# Iniciar servidor
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI disponible en: `http://localhost:8000/docs`

### 2. Entrenar el modelo de detección

```bash
# Paso 1: recolectar muestras (mínimo 100 por clase)
python collect_data.py

# Paso 2: entrenar
python train_model.py           # señas estáticas → model/sign_model.h5
python train_sequence_model.py  # señas dinámicas → model/sequence_model.h5
```

### 3. Frontend

```bash
cd frontend

# Configurar IP del servidor en app.json → extra → apiUrl
# "apiUrl": "http://TU_IP_LOCAL:8000"

npm install
npx expo start
```

Escanea el QR con Expo Go (iOS/Android).

---

## Docker

```bash
# Construir y levantar el backend
docker compose up --build

# En segundo plano
docker compose up -d

# Ver logs
docker compose logs -f

# Detener
docker compose down
```

---

## Despliegue en la nube (Render)

1. Sube el proyecto a GitHub.
2. En [render.com](https://render.com) → **New Web Service** → conecta el repositorio.
3. Render detecta `render.yaml` automáticamente y configura el servicio.
4. Actualiza `app.json → extra → apiUrl` con la URL de Render.
5. Haz nuevo build de la app con `eas build`.

---

## Publicar la app (APK / TestFlight)

```bash
# Instalar EAS CLI
npm install -g eas-cli

# Login en Expo
eas login

# Configurar proyecto (solo la primera vez)
eas build:configure

# Generar APK para pruebas internas (Android)
eas build --profile preview --platform android

# Build de producción
eas build --profile production --platform android
```

---

## Tests

```bash
cd backend
pip install pytest httpx
pytest tests/ -v
```

Los tests mockean TensorFlow, OpenCV y MediaPipe — corren sin GPU ni dependencias pesadas.

---

## Variables de entorno (backend)

| Variable | Default | Descripción |
|---|---|---|
| `SECRET_KEY` | `handtalk-dev-insecure-key` | Clave para firmar JWT. **Cámbiala en producción.** |
| `ALLOWED_ORIGINS` | `*` | Orígenes permitidos para CORS (ej. `https://tu-app.com`) |

Copia `backend/.env.example` como `backend/.env` y ajusta los valores.
