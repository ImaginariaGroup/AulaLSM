from __future__ import annotations
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QListWidget,
    QListWidgetItem, QGroupBox, QStatusBar,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from config import AppConfig, SIGNS, SIGN_LABELS
from core.event_bus import EventBus
from core.hand_tracker import HandData, HandTracker
from core import feature_extractor as fe
from ui.bridge import UIBridge
from ui.hand_view import HandView


class CollectorWindow(QMainWindow):
    """
    Data collection tool.

    Workflow per sign
    -----------------
    1. Window shows the sign label + description.
    2. User positions their hand in frame.
    3. User presses SPACE (or the button) to start recording.
    4. Countdown 3-2-1, then records N frames where a hand is detected.
    5. Saves each frame as data/signs/{label}/sample_{n:04d}.npy
    6. Advances to next sign automatically.
    """

    all_done = pyqtSignal()  # emitted when every sign has enough samples

    _COUNTDOWN = 3  # seconds

    def __init__(
        self,
        config: AppConfig,
        event_bus: EventBus,
        hand_tracker: HandTracker,
    ) -> None:
        super().__init__()
        self._cfg     = config
        self._bus     = event_bus
        self._tracker = hand_tracker
        self._bridge  = UIBridge(event_bus, parent=self)

        self._sign_index  = 0
        self._buffer: List[np.ndarray] = []
        self._countdown   = 0
        self._recording   = False
        self._last_sample_ts: float = 0.0   # monotonic time of last saved sample
        self._next_in_ms: float = 0.0       # ms until next sample (for progress bar)

        self.setWindowTitle("LSM — Recolector de datos")
        self.resize(1100, 700)
        self._build_ui()
        self._bridge.hand_data_received.connect(self._on_hand_data)
        self._load_counts()
        self._show_current_sign()

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

        # Right: controls
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setSpacing(10)

        # Current sign display
        sign_box = QGroupBox("Senal actual")
        sign_layout = QVBoxLayout()
        self._lbl_letter = QLabel("A")
        self._lbl_letter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_letter.setFont(QFont("Segoe UI", 72, QFont.Weight.Bold))
        self._lbl_letter.setObjectName("bigLetter")

        self._lbl_desc = QLabel("")
        self._lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_desc.setWordWrap(True)
        self._lbl_desc.setObjectName("signDesc")

        self._lbl_status = QLabel("Presiona ESPACIO para iniciar")
        self._lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_status.setObjectName("statusLabel")

        sign_layout.addWidget(self._lbl_letter)
        sign_layout.addWidget(self._lbl_desc)
        sign_layout.addWidget(self._lbl_status)
        sign_box.setLayout(sign_layout)
        rl.addWidget(sign_box)

        # Progress
        prog_box = QGroupBox("Progreso de esta senal")
        prog_layout = QVBoxLayout()
        prog_layout.setSpacing(6)

        self._progress = QProgressBar()
        self._progress.setMaximum(self._cfg.collector.samples_per_sign)
        self._progress.setTextVisible(True)
        self._progress.setFormat("%v / %m muestras")

        # Interval bar: shows countdown to next sample capture
        lbl_interval = QLabel("Intervalo entre capturas:")
        lbl_interval.setObjectName("signDesc")
        self._interval_bar = QProgressBar()
        self._interval_bar.setMaximum(100)
        self._interval_bar.setValue(0)
        self._interval_bar.setTextVisible(False)
        self._interval_bar.setFixedHeight(8)
        self._interval_bar.setObjectName("intervalBar")

        self._lbl_capture_hint = QLabel("Mueve la mano entre capturas para mejor variedad")
        self._lbl_capture_hint.setObjectName("signDesc")
        self._lbl_capture_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_capture_hint.setWordWrap(True)

        prog_layout.addWidget(self._progress)
        prog_layout.addWidget(lbl_interval)
        prog_layout.addWidget(self._interval_bar)
        prog_layout.addWidget(self._lbl_capture_hint)
        prog_box.setLayout(prog_layout)
        rl.addWidget(prog_box)

        # UI refresh timer for interval bar (runs at ~30 fps)
        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(33)
        self._ui_timer.timeout.connect(self._refresh_interval_bar)
        self._ui_timer.start()

        # Buttons
        btn_row = QHBoxLayout()
        self._btn_record = QPushButton("Grabar (ESPACIO)")
        self._btn_record.setObjectName("primaryBtn")
        self._btn_record.clicked.connect(self._start_recording)

        self._btn_delete = QPushButton("Borrar y repetir")
        self._btn_delete.setObjectName("dangerBtn")
        self._btn_delete.clicked.connect(self._delete_current)

        self._btn_skip = QPushButton("Saltar ->")
        self._btn_skip.setObjectName("secondaryBtn")
        self._btn_skip.clicked.connect(self._next_sign)

        btn_row.addWidget(self._btn_record, stretch=2)
        btn_row.addWidget(self._btn_delete, stretch=2)
        btn_row.addWidget(self._btn_skip, stretch=1)
        rl.addLayout(btn_row)

        # Sign list
        list_box = QGroupBox("Todas las senales")
        list_layout = QVBoxLayout()
        self._sign_list = QListWidget()
        self._sign_list.setObjectName("signList")
        for label in SIGN_LABELS:
            item = QListWidgetItem(f"  {label}")
            self._sign_list.addItem(item)
        list_layout.addWidget(self._sign_list)
        list_box.setLayout(list_layout)
        rl.addWidget(list_box, stretch=1)

        # Rotation buttons
        rot_row = QHBoxLayout()
        rot_row.addWidget(QLabel("Rotar camara:"))
        for deg in (0, 90, 180, 270):
            btn = QPushButton(f"{deg}deg")
            btn.setObjectName("rotBtn")
            btn.clicked.connect(lambda _, d=deg: self._tracker.set_rotation(d))
            rot_row.addWidget(btn)
        rl.addLayout(rot_row)

        root.addWidget(right, stretch=2)

        # Status bar
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)

    # ── sign navigation ───────────────────────────────────────────────────────

    def _show_current_sign(self) -> None:
        if self._sign_index >= len(SIGN_LABELS):
            self.all_done.emit()
            self._lbl_status.setText("Todas las senales completadas!")
            return

        label = SIGN_LABELS[self._sign_index]
        self._lbl_letter.setText(label)
        self._lbl_desc.setText(SIGNS.get(label, ""))
        count = self._sample_count(label)
        self._progress.setValue(count)
        self._lbl_status.setText("Presiona ESPACIO para iniciar grabacion")
        self._btn_record.setEnabled(True)

        # Highlight in list
        self._sign_list.setCurrentRow(self._sign_index)

    def _next_sign(self) -> None:
        self._recording = False
        self._buffer.clear()
        self._sign_index += 1
        self._show_current_sign()

    def _delete_current(self) -> None:
        """Delete all saved samples for the current sign and reset to re-record."""
        self._recording = False
        self._buffer.clear()

        label = SIGN_LABELS[self._sign_index]
        dest  = self._cfg.collector.data_dir / label

        deleted = 0
        if dest.exists():
            for f in dest.glob("*.npy"):
                f.unlink()
                deleted += 1

        self._progress.setValue(0)
        self._update_list_item(label)
        self._btn_record.setEnabled(True)
        self._lbl_status.setText("Presiona ESPACIO para iniciar grabacion")
        self._interval_bar.setValue(0)

        msg = (
            f"Borradas {deleted} muestras de '{label}' — lista para repetir"
            if deleted
            else f"No habia muestras de '{label}' — lista para grabar"
        )
        self._statusbar.showMessage(msg, 4000)

    # ── recording ─────────────────────────────────────────────────────────────

    def _start_recording(self) -> None:
        if self._recording:
            return
        self._buffer.clear()
        self._countdown = self._COUNTDOWN
        self._btn_record.setEnabled(False)
        self._tick_countdown()

    def _tick_countdown(self) -> None:
        if self._countdown > 0:
            self._lbl_status.setText(f"Preparate...  {self._countdown}")
            self._countdown -= 1
            QTimer.singleShot(1000, self._tick_countdown)
        else:
            self._lbl_status.setText("GRABANDO — mueve la mano lentamente")
            self._recording = True
            self._last_sample_ts = time.monotonic()  # primer disparo inmediato

    def _on_hand_data(self, data: HandData) -> None:
        self._hand_view.update_frame(data.annotated_frame)

        if not self._recording:
            return

        if data.landmarks is None:
            self._lbl_status.setText("Sin mano — acerca la mano a la camara")
            return

        interval_s = self._cfg.collector.sample_interval_ms / 1000.0
        now        = time.monotonic()
        elapsed    = now - self._last_sample_ts
        remaining  = max(0.0, interval_s - elapsed)

        # Update ms remaining for the interval bar (handled by _refresh_interval_bar)
        self._next_in_ms = remaining * 1000.0

        # Not enough time elapsed yet — wait
        if elapsed < interval_s:
            return

        # ── capture this sample ───────────────────────────────────────────────
        self._last_sample_ts = now
        label    = SIGN_LABELS[self._sign_index]
        raw      = fe.extract_raw(data.landmarks)
        self._buffer.append(raw)

        existing  = self._sample_count(label)
        collected = existing + len(self._buffer)
        target    = self._cfg.collector.samples_per_sign
        remaining_samples = target - collected

        self._progress.setValue(min(collected, target))
        self._lbl_status.setText(
            f"Capturada #{len(self._buffer)}  —  "
            f"faltan {max(0, remaining_samples)}  —  "
            f"mueve la mano!"
        )

        if len(self._buffer) >= target - existing:
            self._save_samples(label)

    def _refresh_interval_bar(self) -> None:
        """Updates the interval countdown bar at ~30 fps (independent of camera fps)."""
        if not self._recording:
            self._interval_bar.setValue(0)
            return
        interval_ms = self._cfg.collector.sample_interval_ms
        # pct = how close we are to the next capture (100 = ready)
        ready_pct = int(100 * (1.0 - self._next_in_ms / max(interval_ms, 1)))
        self._interval_bar.setValue(max(0, min(100, ready_pct)))

    def _save_samples(self, label: str) -> None:
        self._recording = False
        dest = self._cfg.collector.data_dir / label
        dest.mkdir(parents=True, exist_ok=True)
        existing = self._sample_count(label)
        for i, raw in enumerate(self._buffer):
            np.save(dest / f"sample_{existing + i:04d}.npy", raw)
        saved = len(self._buffer)
        self._buffer.clear()
        self._statusbar.showMessage(
            f"Guardadas {saved} muestras para '{label}'", 3000
        )
        self._update_list_item(label)
        QTimer.singleShot(800, self._next_sign)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _sample_count(self, label: str) -> int:
        d = self._cfg.collector.data_dir / label
        return len(list(d.glob("*.npy"))) if d.exists() else 0

    def _load_counts(self) -> None:
        for i, label in enumerate(SIGN_LABELS):
            self._update_list_item(label, row=i)

    def _update_list_item(self, label: str, row: Optional[int] = None) -> None:
        if row is None:
            row = SIGN_LABELS.index(label)
        count  = self._sample_count(label)
        target = self._cfg.collector.samples_per_sign
        item   = self._sign_list.item(row)
        if item:
            icon   = "OK" if count >= target else f"{count}/{target}"
            item.setText(f"  {label:<8} {icon}")
            if count >= target:
                item.setForeground(QColor("#00F5D4"))

    # ── keyboard shortcut ─────────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._start_recording()
        elif event.key() == Qt.Key.Key_Right:
            self._next_sign()
        else:
            super().keyPressEvent(event)
