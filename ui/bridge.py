from __future__ import annotations
from PyQt6.QtCore import QObject, pyqtSignal
from core.event_bus import EventBus
from core.hand_tracker import HAND_DATA_EVENT


class UIBridge(QObject):
    """Converts EventBus callbacks (worker threads) → Qt signals (main thread)."""
    hand_data_received = pyqtSignal(object)

    def __init__(self, event_bus: EventBus, parent: QObject | None = None) -> None:
        super().__init__(parent)
        event_bus.subscribe(HAND_DATA_EVENT, self.hand_data_received.emit)
