from __future__ import annotations
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QStatusBar, QFrame,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor

from config import AppConfig, SIGNS
from core.event_bus import EventBus
from core.hand_tracker import HandData, HandTracker
from core.sign_classifier import SignClassifier
from core.sentence_builder import SentenceBuilder
from core.tts_engine import TTSEngine
from core import feature_extractor as fe
from ui.bridge import UIBridge
from ui.hand_view import HandView


class ConfidenceBar(QFrame):
    """Horizontal colour bar showing classifier confidence."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("confBar")
        self.setFixedHeight(8)
        self._value = 0.0

    def set_value(self, v: float) -> None:
        self._value = max(0.0, min(1.0, v))
        pct = int(self._value * 100)
        if pct > 75:
            colour = "#00F5D4"
        elif pct > 40:
            colour = "#F5A623"
        else:
            colour = "#E05555"
        self.setStyleSheet(
            f"QFrame#confBar {{ "
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {colour}, stop:{self._value:.2f} {colour},"
            f"stop:{min(self._value+0.001,1):.3f} #1A1A2E, stop:1 #1A1A2E); "
            f"border-radius:4px; }}"
        )


class MainWindow(QMainWindow):
    def __init__(
        self,
        config: AppConfig,
        event_bus: EventBus,
        classifier: SignClassifier,
        tts: TTSEngine,
        hand_tracker: HandTracker,
    ) -> None:
        super().__init__()
        self._cfg        = config
        self._classifier = classifier
        self._tts        = tts
        self._tracker    = hand_tracker
        self._bridge     = UIBridge(event_bus, parent=self)

        self._builder = SentenceBuilder(
            hold_frames=config.sentence.hold_frames,
            cooldown_frames=config.sentence.cooldown_frames,
            agreement_ratio=config.sentence.agreement_ratio,
            on_update=self._on_text_update,
            on_accept=self._on_sign_accepted,
        )

        self.setWindowTitle("LSM — Interprete de Lengua de Senas Mexicana")
        self.resize(1200, 720)
        self._build_ui()
        self._bridge.hand_data_received.connect(self._on_hand_data)

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Left: camera
        self._hand_view = HandView()
        root.addWidget(self._hand_view, stretch=3)

        # Right panel
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setSpacing(10)

        # Current sign — large display
        detect_box = QGroupBox("Senal detectada")
        detect_box.setObjectName("detectBox")
        dl = QVBoxLayout()
        self._lbl_sign = QLabel("—")
        self._lbl_sign.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_sign.setFont(QFont("Segoe UI", 80, QFont.Weight.Bold))
        self._lbl_sign.setObjectName("bigSign")

        self._lbl_conf_text = QLabel("Confianza: —")
        self._lbl_conf_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_conf_text.setObjectName("confText")

        self._conf_bar = ConfidenceBar()

        self._lbl_hint = QLabel("")
        self._lbl_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_hint.setObjectName("hintText")
        self._lbl_hint.setWordWrap(True)

        dl.addWidget(self._lbl_sign)
        dl.addWidget(self._lbl_conf_text)
        dl.addWidget(self._conf_bar)
        dl.addWidget(self._lbl_hint)
        detect_box.setLayout(dl)
        rl.addWidget(detect_box)

        # Text output
        text_box = QGroupBox("Texto acumulado")
        text_box.setObjectName("textBox")
        tl = QVBoxLayout()
        self._lbl_text = QLabel("")
        self._lbl_text.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._lbl_text.setWordWrap(True)
        self._lbl_text.setMinimumHeight(80)
        self._lbl_text.setFont(QFont("Segoe UI", 18))
        self._lbl_text.setObjectName("textOutput")
        tl.addWidget(self._lbl_text)
        text_box.setLayout(tl)
        rl.addWidget(text_box)

        # Action buttons
        btn_row = QHBoxLayout()
        self._btn_speak = QPushButton("Hablar")
        self._btn_speak.setObjectName("primaryBtn")
        self._btn_speak.clicked.connect(self._speak)
        self._btn_clear = QPushButton("Borrar")
        self._btn_clear.setObjectName("secondaryBtn")
        self._btn_clear.clicked.connect(self._clear)
        btn_row.addWidget(self._btn_speak)
        btn_row.addWidget(self._btn_clear)
        rl.addLayout(btn_row)

        # Rotation buttons
        rot_box = QGroupBox("Rotar camara")
        rot_inner = QHBoxLayout()
        for deg in (0, 90, 180, 270):
            btn = QPushButton(f"{deg}deg")
            btn.setObjectName("rotBtn")
            btn.clicked.connect(lambda _, d=deg: self._tracker.set_rotation(d))
            rot_inner.addWidget(btn)
        rot_box.setLayout(rot_inner)
        rl.addWidget(rot_box)

        rl.addStretch()
        root.addWidget(right, stretch=2)

        # Status bar
        bar = QStatusBar()
        self.setStatusBar(bar)
        self._lbl_fps   = QLabel("FPS: —")
        self._lbl_model = QLabel(
            "Modelo cargado" if self._classifier.is_loaded
            else "SIN MODELO — ejecuta train.py primero"
        )
        bar.addWidget(self._lbl_fps)
        bar.addPermanentWidget(self._lbl_model)

    # ── slots ─────────────────────────────────────────────────────────────────

    def _on_hand_data(self, data: HandData) -> None:
        self._hand_view.update_frame(data.annotated_frame)
        self._lbl_fps.setText(f"FPS: {data.fps:.1f}")

        if data.landmarks is None:
            self._lbl_sign.setText("—")
            self._lbl_conf_text.setText("Sin mano en camara")
            self._conf_bar.set_value(0)
            self._lbl_hint.setText("")
            self._builder.push(None, 0.0)
            return

        features = fe.extract(data.landmarks)
        label, conf = self._classifier.predict(features)

        # Display
        display = label if conf >= self._cfg.classifier.confidence_threshold else "?"
        self._lbl_sign.setText(display)
        self._lbl_conf_text.setText(f"Confianza: {conf*100:.1f}%")
        self._conf_bar.set_value(conf)
        self._lbl_hint.setText(SIGNS.get(label, "") if conf > 0.5 else "")

        # Feed sentence builder
        self._builder.push(
            label, conf,
            threshold=self._cfg.classifier.confidence_threshold,
        )

    def _on_text_update(self, text: str) -> None:
        self._lbl_text.setText(text if text else " ")

    def _on_sign_accepted(self, label: str) -> None:
        self.statusBar().showMessage(f"Aceptado: {label}", 1500)

    def _speak(self) -> None:
        text = self._builder.text.strip()
        if text:
            self._tts.speak(text)
            self.statusBar().showMessage(f"Leyendo: {text}", 3000)

    def _clear(self) -> None:
        self._builder.clear()
