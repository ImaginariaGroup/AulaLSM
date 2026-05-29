from __future__ import annotations
import math
from typing import List

import numpy as np

from core.hand_tracker import HandLandmark

# Landmark 9 = middle finger MCP — used as scale reference
_SCALE_IDX = 9


def extract(landmarks: List[HandLandmark]) -> np.ndarray:
    """
    Converts 21 hand landmarks into a normalized 63-D feature vector.

    Normalization steps
    -------------------
    1. Translation: subtract wrist (landmark 0) so origin = wrist
    2. Scale: divide by wrist-to-middle-MCP distance so palm size = 1

    This makes features invariant to hand position and camera distance,
    which is critical for a classifier to generalise across users.

    Returns
    -------
    np.ndarray of shape (63,) dtype float32, or zeros if input is bad.
    """
    if len(landmarks) != 21:
        return np.zeros(63, dtype=np.float32)

    ox, oy, oz = landmarks[0].x, landmarks[0].y, landmarks[0].z

    pts = np.array(
        [(lm.x - ox, lm.y - oy, lm.z - oz) for lm in landmarks],
        dtype=np.float32,
    )

    # Scale factor: distance from wrist to middle-finger MCP
    scale = math.hypot(float(pts[_SCALE_IDX, 0]), float(pts[_SCALE_IDX, 1]))
    if scale > 1e-6:
        pts /= scale

    return pts.flatten()  # (21, 3) → (63,)


def extract_raw(landmarks: List[HandLandmark]) -> np.ndarray:
    """Returns raw (unnormalized) (21, 3) array for storage during data collection."""
    return np.array(
        [(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32
    )
