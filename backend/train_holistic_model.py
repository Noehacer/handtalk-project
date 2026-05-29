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
    """Temperature scaling: find T that minimises NLL on validation set."""
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
                    X.append(aug)
                    y.append(idx)

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
                    X.append(aug)
                    y.append(idx)

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

    print(f"\nListo. Para activar: MODEL_VERSION=v2 uvicorn main:app --reload")


if __name__ == "__main__":
    main()
