"""Thin wrapper around MediaPipe Face Mesh.

Returns an (N, 3) float array in PIXEL coordinates, or None when no face is
found. Everything downstream works on that array, so it can be unit-tested
without a camera or MediaPipe.

Pass the raw (un-mirrored) camera frame in. Mirror only the image you display.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


class FaceLandmarkExtractor:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
    ) -> None:
        import mediapipe as mp  # imported lazily so tests don't need it

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        lm = result.multi_face_landmarks[0].landmark
        return np.array([(p.x * w, p.y * h, p.z * w) for p in lm], dtype=np.float64)

    def close(self) -> None:
        self._mesh.close()

    def __enter__(self) -> "FaceLandmarkExtractor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class HandLandmarkExtractor:
    """MediaPipe Hands: up to two hands as (21, 3) pixel arrays, or an empty list."""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5) -> None:
        import mediapipe as mp  # imported lazily so tests don't need it

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2, model_complexity=0,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)

    def extract(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return []
        return [np.array([(p.x * w, p.y * h, p.z * w) for p in hand.landmark], dtype=np.float64)
                for hand in result.multi_hand_landmarks]

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "HandLandmarkExtractor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
