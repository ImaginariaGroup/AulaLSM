"""
python main.py

Interprete de Lengua de Senas Mexicana en tiempo real.
Requiere haber ejecutado collect.py y train.py primero.
"""
from __future__ import annotations
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMessageBox

from config import AppConfig
from core.event_bus import EventBus
from core.hand_tracker import HandTracker
from core.sign_classifier import SignClassifier
from core.tts_engine import TTSEngine
from ui.main_window import MainWindow

QSS = Path(__file__).parent / "assets" / "styles" / "main.qss"


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("LSM Interprete")
    if QSS.exists():
        app.setStyleSheet(QSS.read_text(encoding="utf-8"))

    config = AppConfig()
    bus    = EventBus()

    classifier = SignClassifier()
    if not classifier.load(config.classifier.model_path):
        QMessageBox.warning(
            None, "Modelo no encontrado",
            "No se encontro el modelo entrenado.\n\n"
            "Pasos:\n"
            "  1. python collect.py   (recolectar muestras)\n"
            "  2. python train.py     (entrenar el modelo)\n"
            "  3. python main.py      (interpretar)",
        )

    tts     = TTSEngine()
    tracker = HandTracker(config.hand_tracker, config.camera, bus)

    window = MainWindow(config, bus, classifier, tts, tracker)
    window.show()

    tts.start()
    tracker.start()

    def _cleanup() -> None:
        tracker.stop()
        tts.stop()

    app.aboutToQuit.connect(_cleanup)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
