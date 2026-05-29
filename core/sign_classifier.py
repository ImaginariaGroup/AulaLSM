from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np


class SignClassifier:
    """
    Wraps a trained scikit-learn pipeline (StandardScaler + MLPClassifier).

    Usage
    -----
    clf = SignClassifier()
    if clf.load(model_path):
        label, confidence = clf.predict(feature_vector_63d)
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._classes:  List[str] = []
        self._loaded = False

    # ── public API ────────────────────────────────────────────────────────────

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        data = joblib.load(path)
        self._pipeline = data["pipeline"]
        self._classes  = data["classes"]
        self._loaded   = True
        print(f"[SignClassifier] Loaded — {len(self._classes)} classes: {self._classes}")
        return True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def classes(self) -> List[str]:
        return list(self._classes)

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Returns (label, confidence).
        Returns ('?', 0.0) when not loaded or confidence is below threshold.
        """
        if not self._loaded:
            return ("?", 0.0)

        proba = self._pipeline.predict_proba(features.reshape(1, -1))[0]
        idx   = int(np.argmax(proba))
        return (self._classes[idx], float(proba[idx]))
