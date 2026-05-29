"""
Entrena el modelo LSTM para señas dinámicas (J, Z, etc.).

Uso:
  python train_sequence_model.py

Requiere haber recolectado datos con collect_data.py (opción 2).
El modelo se guarda en backend/model/sequence_model.h5
"""

import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from labels import DYNAMIC_LABELS, SEQUENCE_LENGTH

BASE_DIR = os.path.dirname(__file__)
DYNAMIC_DIR = os.path.join(BASE_DIR, "data", "dynamic")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_sequence_dataset():
    X, y, labels_found = [], [], []

    for label in DYNAMIC_LABELS:
        folder = os.path.join(DYNAMIC_DIR, label)
        if not os.path.exists(folder):
            continue
        files = [f for f in os.listdir(folder) if f.endswith('.npy')]
        if not files:
            continue

        label_idx = len(labels_found)
        labels_found.append(label)

        for fname in files:
            sample = np.load(os.path.join(folder, fname))
            if sample.shape == (SEQUENCE_LENGTH, 63):
                X.append(sample)
                y.append(label_idx)

    return np.array(X), np.array(y), labels_found


def build_sequence_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, 63)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train():
    print("Cargando dataset de secuencias...")
    X, y, labels = load_sequence_dataset()

    if len(X) == 0:
        print("No hay datos. Ejecuta primero: python collect_data.py (opción 2)")
        return

    print(f"  {len(X)} secuencias  |  {len(labels)} clases: {labels}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_sequence_model(len(labels))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, "sequence_model.h5"),
            save_best_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(patience=7, factor=0.5, verbose=1),
    ]

    print("\nEntrenando...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )

    np.save(os.path.join(MODEL_DIR, "seq_labels.npy"), np.array(labels))

    _, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nModelo guardado en: {MODEL_DIR}/sequence_model.h5")
    print(f"Precisión en validación: {val_acc:.2%}")


if __name__ == "__main__":
    train()
