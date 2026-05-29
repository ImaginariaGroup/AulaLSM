from __future__ import annotations
import time
from collections import deque


class FPSCounter:
    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._times.append(time.monotonic())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return 0.0 if elapsed <= 0 else (len(self._times) - 1) / elapsed
