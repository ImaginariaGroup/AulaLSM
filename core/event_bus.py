from __future__ import annotations
from collections import defaultdict
from typing import Any, Callable
import threading


class EventBus:
    _instance: EventBus | None = None
    _creation_lock = threading.Lock()

    def __new__(cls) -> EventBus:
        with cls._creation_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._subscribers: dict[str, list[Callable]] = defaultdict(list)
                inst._rlock = threading.RLock()
                cls._instance = inst
        return cls._instance

    def subscribe(self, event: str, callback: Callable[[Any], None]) -> None:
        with self._rlock:
            self._subscribers[event].append(callback)

    def unsubscribe(self, event: str, callback: Callable[[Any], None]) -> None:
        with self._rlock:
            try:
                self._subscribers[event].remove(callback)
            except ValueError:
                pass

    def publish(self, event: str, data: Any = None) -> None:
        with self._rlock:
            callbacks = list(self._subscribers[event])
        for cb in callbacks:
            try:
                cb(data)
            except Exception:
                pass
