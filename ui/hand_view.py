from __future__ import annotations
import numpy as np
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


class HandView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(480, 360)
        self._label.setObjectName("cameraFeed")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

    def update_frame(self, frame: np.ndarray) -> None:
        h, w, ch = frame.shape
        img = QImage(frame.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888)
        self._label.setPixmap(
            QPixmap.fromImage(img).scaled(
                self._label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
