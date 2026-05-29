"""
python train.py

Carga todas las muestras de data/signs/, entrena un clasificador MLP con
scikit-learn y guarda el modelo en models/sign_classifier.pkl.

Requiere al menos 2 clases con muestras recolectadas.
"""
from __future__ import annotations
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import AppConfig
from core import feature_extractor as fe
from core.hand_tracker import HandLandmark


def load_dataset(data_dir: Path):
    X, y = [], []
    label_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    if len(label_dirs) < 2:
        print(f"ERROR: Se necesitan al menos 2 clases. Solo se encontro: {label_dirs}")
        sys.exit(1)

    for label_dir in label_dirs:
        label   = label_dir.name
        samples = sorted(label_dir.glob("*.npy"))
        if not samples:
            continue
        for path in samples:
            raw  = np.load(path)           # shape (21, 3)
            lms  = [HandLandmark(x, y_, z) for x, y_, z in raw]
            feat = fe.extract(lms)
            X.append(feat)
            y.append(label)
        print(f"  {label:<10} {len(samples):>4} muestras")

    return np.array(X, dtype=np.float32), np.array(y)


def main() -> None:
    cfg      = AppConfig()
    data_dir = cfg.collector.data_dir
    out_path = cfg.classifier.model_path

    print(f"\nCargando datos desde {data_dir} ...")
    X, y = load_dataset(data_dir)
    print(f"\nTotal: {len(X)} muestras  |  {len(set(y))} clases\n")

    le      = LabelEncoder()
    y_enc   = le.fit_transform(y)
    classes = list(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            max_iter=600,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=20,
            verbose=False,
        )),
    ])

    print("Entrenando MLP ...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc    = (y_pred == y_test).mean()
    print(f"\nAccuracy en test: {acc * 100:.1f}%\n")
    print(classification_report(
        y_test, y_pred,
        target_names=classes,
        zero_division=0,
    ))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "classes": classes}, out_path)
    print(f"Modelo guardado en {out_path}")


if __name__ == "__main__":
    main()
