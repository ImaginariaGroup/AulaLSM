from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


BASE_DIR = Path(__file__).parent

# ── sign vocabulary ───────────────────────────────────────────────────────────
# label → brief description of the handshape (LSM reference)
SIGNS: Dict[str, str] = {
    "A":  "Puno cerrado, pulgar al lado",
    "B":  "Dedos rectos y juntos, pulgar doblado hacia la palma",
    "C":  "Mano curvada formando una C",
    "D":  "Indice recto hacia arriba, demas doblados, pulgar toca el medio",
    "E":  "Dedos doblados hacia la palma, pulgar bajo los dedos",
    "F":  "Indice y pulgar se tocan, otros tres dedos extendidos",
    "G":  "Indice apunta al lado, pulgar paralelo",
    "H":  "Indice y medio extendidos al lado, juntos",
    "I":  "Menique extendido, resto cerrado en puno",
    "J":  "Menique extendido, traza una J en el aire",
    "K":  "Indice arriba, medio diagonal, pulgar entre ellos",
    "L":  "Indice apunta arriba, pulgar al lado (forma L)",
    "M":  "Tres dedos doblados sobre el pulgar",
    "N":  "Dos dedos doblados sobre el pulgar",
    "O":  "Todos los dedos forman una O con el pulgar",
    "P":  "Como K pero apuntando hacia abajo",
    "Q":  "Indice y pulgar apuntan hacia abajo",
    "R":  "Indice y medio cruzados, apuntan arriba",
    "S":  "Puno cerrado, pulgar sobre los dedos",
    "T":  "Puno cerrado, pulgar entre indice y medio",
    "U":  "Indice y medio extendidos juntos hacia arriba",
    "V":  "Indice y medio en V abierta",
    "W":  "Indice, medio y anular extendidos en W",
    "X":  "Indice doblado como un gancho",
    "Y":  "Pulgar y menique extendidos, otros doblados",
    "Z":  "Indice extendido, traza una Z en el aire",
    "ESPACIO": "Mano abierta, palma al frente (pausa entre palabras)",
    "BORRAR":  "Puno cerrado (elimina la ultima letra)",
}

SIGN_LABELS = list(SIGNS.keys())


# ── configuration dataclasses ─────────────────────────────────────────────────

@dataclass
class CameraConfig:
    device_id: int = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 30
    rotation: int = 0


@dataclass
class HandTrackerConfig:
    max_hands: int = 1          # 1 mano es suficiente para el alfabeto
    detection_confidence: float = 0.6
    tracking_confidence: float = 0.5


@dataclass
class ClassifierConfig:
    model_path: Path = field(default_factory=lambda: BASE_DIR / "models" / "sign_classifier.pkl")
    confidence_threshold: float = 0.75  # bajo este valor = "desconocido"


@dataclass
class SentenceConfig:
    hold_frames: int = 20       # frames que debe mantenerse la seña
    cooldown_frames: int = 40   # frames de espera tras aceptar
    agreement_ratio: float = 0.6  # fraccion de frames que deben coincidir


@dataclass
class CollectorConfig:
    samples_per_sign: int = 50
    sample_interval_ms: int = 400   # ms entre una muestra y la siguiente
                                    # 400 ms × 50 = ~20 segundos por seña
                                    # tiempo suficiente para cambiar angulo
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "signs")


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    hand_tracker: HandTrackerConfig = field(default_factory=HandTrackerConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    sentence: SentenceConfig = field(default_factory=SentenceConfig)
    collector: CollectorConfig = field(default_factory=CollectorConfig)
