"""
Herramienta de recolección de datos de entrenamiento.

Uso:
  python collect_data.py

Controles (señas estáticas):
  Letra/número en teclado → guarda muestra para esa clase
  h → hola  |  g → gracias  |  s → si  |  n → no
  p → por_favor  |  y → ayuda
  ESC → salir

Controles (señas dinámicas):
  ESPACIO → iniciar grabación de secuencia
  ESC → salir
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from sign_model import normalize_landmarks
from labels import STATIC_LABELS, DYNAMIC_LABELS, SEQUENCE_LENGTH

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "data", "static")
DYNAMIC_DIR = os.path.join(BASE_DIR, "data", "dynamic")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
_hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Teclas para palabras completas (evitan conflicto con letras del alfabeto)
WORD_KEYS = {
    ord('h'): 'hola',
    ord('g'): 'gracias',
    ord('s'): 'si',
    ord('n'): 'no',
    ord('p'): 'por_favor',
    ord('y'): 'ayuda',
}


def _count_samples(label, mode):
    folder = os.path.join(STATIC_DIR if mode == 'static' else DYNAMIC_DIR, label)
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith('.npy')])


def _save_sample(label, data, mode):
    folder = os.path.join(STATIC_DIR if mode == 'static' else DYNAMIC_DIR, label)
    os.makedirs(folder, exist_ok=True)
    idx = _count_samples(label, mode)
    np.save(os.path.join(folder, f"{idx}.npy"), data)
    return idx + 1


def _extract(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = _hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None, result
    hand = result.multi_hand_landmarks[0]
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
    return landmarks, result


def collect_static():
    cap = cv2.VideoCapture(0)
    print("\n=== SEÑAS ESTÁTICAS ===")
    print("Muestra tu mano y presiona la tecla de la seña.")
    print("Recomendado: mínimo 100 muestras por clase.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        landmarks, result = _extract(frame)

        if result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, "Mano detectada", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Sin mano", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Muestra conteo de las primeras clases
        y = 60
        for label in STATIC_LABELS[:12]:
            count = _count_samples(label, 'static')
            color = (0, 200, 0) if count >= 100 else (0, 165, 255)
            cv2.putText(frame, f"{label}:{count}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y += 18

        cv2.putText(frame, "Tecla=guardar | ESC=salir", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("HandTalk - Datos Estaticos", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if landmarks is not None:
            label = None
            char = chr(key).upper() if 32 <= key <= 126 else None

            if char in STATIC_LABELS:
                label = char
            elif key in WORD_KEYS:
                label = WORD_KEYS[key]

            if label:
                normalized = normalize_landmarks(landmarks)
                count = _save_sample(label, normalized, 'static')
                print(f"  Guardado: {label} → muestra #{count}")

    cap.release()
    cv2.destroyAllWindows()


def collect_dynamic():
    cap = cv2.VideoCapture(0)
    print("\n=== SEÑAS DINÁMICAS ===")
    print("Teclas 0-9 / a-d → seleccionar clase  |  ESC → salir\n")
    for i, label in enumerate(DYNAMIC_LABELS):
        print(f"  [{i if i < 10 else chr(ord('a') + i - 10)}] {label} — {_count_samples(label, 'dynamic')} muestras")

    # Mapeo tecla → índice de clase
    KEY_MAP = {}
    for i in range(len(DYNAMIC_LABELS)):
        if i < 10:
            KEY_MAP[ord(str(i))] = i
        else:
            KEY_MAP[ord(chr(ord('a') + i - 10))] = i

    recording = False
    sequence = []
    current_label = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        landmarks, result = _extract(frame)

        if result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        if recording:
            if landmarks is not None:
                sequence.append(normalize_landmarks(landmarks))

            progress = len(sequence)
            pct = int(progress / SEQUENCE_LENGTH * frame.shape[1])
            cv2.rectangle(frame, (0, frame.shape[0] - 8), (pct, frame.shape[0]), (0, 0, 255), -1)
            cv2.putText(frame, f"Grabando '{current_label}': {progress}/{SEQUENCE_LENGTH}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if progress >= SEQUENCE_LENGTH:
                data = np.array(sequence)
                count = _save_sample(current_label, data, 'dynamic')
                print(f"  Guardado: {current_label} → secuencia #{count}")
                recording = False
                sequence = []
                current_label = None
        else:
            cv2.putText(frame, "Tecla = elegir clase | ESC = salir", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y = 55
            for i, lbl in enumerate(DYNAMIC_LABELS):
                key_char = str(i) if i < 10 else chr(ord('a') + i - 10)
                count = _count_samples(lbl, 'dynamic')
                color = (0, 200, 0) if count >= 50 else (0, 165, 255)
                cv2.putText(frame, f"[{key_char}] {lbl}: {count}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                y += 18

        cv2.imshow("HandTalk - Datos Dinamicos", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if not recording and key in KEY_MAP:
            current_label = DYNAMIC_LABELS[KEY_MAP[key]]
            recording = True
            sequence = []
            print(f"Grabando '{current_label}' — realiza la seña ahora...")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("HandTalk — Recolección de datos")
    print("1. Señas estáticas (letras A-Y, números 0-9, palabras)")
    print("2. Señas dinámicas (J, Z, adiós, etc.)")
    choice = input("Opción: ").strip()

    if choice == "1":
        collect_static()
    elif choice == "2":
        collect_dynamic()
    else:
        print("Opción no válida.")
