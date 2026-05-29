from __future__ import annotations
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarksConnections

from config import HandTrackerConfig, CameraConfig
from core.event_bus import EventBus
from utils.fps_counter import FPSCounter


_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_PATH = Path(__file__).parent.parent / "models" / "hand_landmarker.task"


def ensure_model() -> Path:
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("[HandTracker] Downloading model...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("[HandTracker] Model ready.")
    return _MODEL_PATH


@dataclass
class HandLandmark:
    x: float
    y: float
    z: float


@dataclass
class HandData:
    """Single frame of tracking data published to the EventBus."""
    landmarks: Optional[List[HandLandmark]]  # None when no hand detected
    annotated_frame: np.ndarray              # RGB frame with skeleton overlay
    fps: float = 0.0


HAND_DATA_EVENT = "hand:data"

_CONNECTIONS: List[Tuple[int, int]] = [
    (c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS
]


class HandTracker(threading.Thread):
    def __init__(
        self,
        hand_cfg: HandTrackerConfig,
        cam_cfg: CameraConfig,
        event_bus: EventBus,
    ) -> None:
        super().__init__(daemon=True, name="HandTrackerThread")
        self._hand_cfg  = hand_cfg
        self._cam_cfg   = cam_cfg
        self._bus       = event_bus
        self._running   = False
        self._fps       = FPSCounter()
        self._ts_origin = 0.0

    _ROTATE = {
        90:  cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    def run(self) -> None:
        self._running   = True
        self._ts_origin = time.monotonic()
        model = ensure_model()

        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=self._hand_cfg.max_hands,
            min_hand_detection_confidence=self._hand_cfg.detection_confidence,
            min_hand_presence_confidence=self._hand_cfg.tracking_confidence,
            min_tracking_confidence=self._hand_cfg.tracking_confidence,
        )
        cap = cv2.VideoCapture(self._cam_cfg.device_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._cam_cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cam_cfg.height)
        cap.set(cv2.CAP_PROP_FPS,          self._cam_cfg.target_fps)

        with mp_vision.HandLandmarker.create_from_options(opts) as det:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                rot   = self._ROTATE.get(self._cam_cfg.rotation % 360)
                if rot is not None:
                    frame = cv2.rotate(frame, rot)

                try:
                    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    ts_ms  = int((time.monotonic() - self._ts_origin) * 1_000)
                    result = det.detect_for_video(mp_img, ts_ms)
                except Exception as exc:
                    print(f"[HandTracker] {exc}")
                    continue

                annotated = frame.copy()
                landmarks: Optional[List[HandLandmark]] = None

                if result.hand_landmarks:
                    lm_list   = result.hand_landmarks[0]
                    landmarks = [HandLandmark(lm.x, lm.y, lm.z) for lm in lm_list]
                    self._draw(annotated, landmarks)

                self._fps.tick()
                self._draw_hud(annotated, landmarks is not None)

                self._bus.publish(
                    HAND_DATA_EVENT,
                    HandData(
                        landmarks=landmarks,
                        annotated_frame=cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        fps=self._fps.fps,
                    ),
                )
        cap.release()

    def stop(self) -> None:
        self._running = False

    def set_rotation(self, deg: int) -> None:
        self._cam_cfg.rotation = deg % 360

    def _draw(self, frame: np.ndarray, lm: List[HandLandmark]) -> None:
        h, w, _ = frame.shape
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for s, e in _CONNECTIONS:
            cv2.line(frame, pts[s], pts[e], (80, 220, 180), 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(frame, pt, 5, (220, 220, 220), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 3, (0, 255, 160),   -1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, detected: bool) -> None:
        status = "Mano detectada" if detected else "Sin mano"
        color  = (0, 255, 160) if detected else (60, 60, 180)
        cv2.putText(
            frame, f"FPS {self._fps.fps:.1f}  |  {status}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
        )
