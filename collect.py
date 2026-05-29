"""
python collect.py

Abre la interfaz de recoleccion de datos.
Recorre todas las senales del alfabeto LSM y guarda muestras en data/signs/.

Controles
---------
ESPACIO     Iniciar grabacion de la senal actual
FLECHA ->   Saltar a la siguiente senal
"""
from __future__ import annotations
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMessageBox

from config import AppConfig
from core.event_bus import EventBus
from core.hand_tracker import HandTracker
from ui.collector_window import CollectorWindow

QSS = Path(__file__).parent / "assets" / "styles" / "main.qss"


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("LSM Collector")
    if QSS.exists():
        app.setStyleSheet(QSS.read_text(encoding="utf-8"))

    config  = AppConfig()
    bus     = EventBus()
    tracker = HandTracker(config.hand_tracker, config.camera, bus)

    window = CollectorWindow(config, bus, tracker)
    window.all_done.connect(
        lambda: QMessageBox.information(
            window,
            "Completado",
            "Todas las senales tienen suficientes muestras.\n"
            "Ejecuta:  python train.py",
        )
    )
    window.show()
    tracker.start()

    app.aboutToQuit.connect(tracker.stop)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
