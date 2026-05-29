from __future__ import annotations
from collections import Counter, deque
from typing import Callable, Optional


class SentenceBuilder:
    """
    Converts a stream of per-frame sign predictions into accepted letters
    and builds a text string.

    Algorithm
    ---------
    Maintains a rolling window of the last `hold_frames` predictions.
    A sign is "accepted" when the same label appears in >= `agreement_ratio`
    of the window AND the mandatory cooldown has expired.

    Special labels
    --------------
    ESPACIO  → appends a space
    BORRAR   → removes the last character
    """

    def __init__(
        self,
        hold_frames: int = 20,
        cooldown_frames: int = 40,
        agreement_ratio: float = 0.60,
        on_update: Optional[Callable[[str], None]] = None,
        on_accept: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._hold      = hold_frames
        self._cooldown_max = cooldown_frames
        self._ratio     = agreement_ratio
        self._on_update = on_update
        self._on_accept = on_accept

        self._buffer:   deque[Optional[str]] = deque(maxlen=hold_frames)
        self._text:     str  = ""
        self._cooldown: int  = 0

    # ── public API ────────────────────────────────────────────────────────────

    def push(self, label: Optional[str], confidence: float, threshold: float = 0.75) -> None:
        """Feed one frame's prediction. Call once per frame."""
        if confidence < threshold:
            label = None
        self._buffer.append(label)

        if self._cooldown > 0:
            self._cooldown -= 1
            return

        if len(self._buffer) < self._hold:
            return

        # Majority vote (ignore None)
        valid    = [x for x in self._buffer if x is not None]
        if not valid:
            return
        dominant, count = Counter(valid).most_common(1)[0]
        if count / self._hold >= self._ratio:
            self._accept(dominant)

    def clear(self) -> None:
        self._text = ""
        self._buffer.clear()
        self._cooldown = 0
        if self._on_update:
            self._on_update(self._text)

    @property
    def text(self) -> str:
        return self._text

    @property
    def last_sign(self) -> Optional[str]:
        """Most recent accepted sign (for UI highlight)."""
        valid = [x for x in self._buffer if x is not None]
        if not valid:
            return None
        return Counter(valid).most_common(1)[0][0]

    # ── internal ──────────────────────────────────────────────────────────────

    def _accept(self, label: str) -> None:
        if label == "ESPACIO":
            self._text += " "
        elif label == "BORRAR":
            self._text = self._text[:-1]
        else:
            self._text += label

        self._buffer.clear()
        self._cooldown = self._cooldown_max

        if self._on_accept:
            self._on_accept(label)
        if self._on_update:
            self._on_update(self._text)
