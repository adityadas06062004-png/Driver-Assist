"""What is the driver doing: just driving, talking, or on a phone call?

A webcam cannot hear a call, so this looks for the phone-to-ear pose: a hand beside the face, near the
ear, held there for a couple of seconds. Mouth movement tells talking apart from quiet driving. It is a
heuristic, not a trained phone detector. Earbuds are invisible to it (it will only see "talking"), and
a hand at the ear for another reason can look the same, which is why a brief touch is ignored.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Tuple

import numpy as np

from .config import Config

# Face Mesh points: contour beside each ear, top of the forehead, chin.
EAR_RIGHT, EAR_LEFT, FOREHEAD, CHIN = 234, 454, 10, 152
PALM = (0, 5, 9, 13, 17)


class Activity(str, Enum):
    DRIVING = "driving"
    TALKING = "talking"
    PHONE_CALL = "phone_call"


LABELS = {
    Activity.DRIVING: "Driving",
    Activity.TALKING: "Talking",
    Activity.PHONE_CALL: "On a phone call",
}


@dataclass
class Geometry:
    """Where the head is, from a handful of Face Mesh points. Kept for a few seconds so that a hand
    covering the face does not make the ear position disappear with it."""
    right: np.ndarray
    left: np.ndarray
    top: float
    chin: float

    @property
    def width(self) -> float:
        return float(np.linalg.norm(self.right - self.left))

    @property
    def centre_x(self) -> float:
        return float((self.right[0] + self.left[0]) / 2.0)

    def contains(self, point: np.ndarray, pad: float = 0.1) -> bool:
        w = self.width
        x0, x1 = min(self.right[0], self.left[0]) - pad * w, max(self.right[0], self.left[0]) + pad * w
        return bool(x0 < point[0] < x1 and self.top - pad * w < point[1] < self.chin + pad * w)


def geometry_from(face: Optional[np.ndarray]) -> Optional[Geometry]:
    if face is None or len(face) <= max(EAR_RIGHT, EAR_LEFT, FOREHEAD, CHIN):
        return None
    geo = Geometry(face[EAR_RIGHT, :2].copy(), face[EAR_LEFT, :2].copy(), float(face[FOREHEAD, 1]),
                   float(face[CHIN, 1]))
    return geo if geo.width >= 20 else None


@dataclass
class ActivityState:
    activity: Activity = Activity.DRIVING
    hand_at_ear: bool = False        # this frame, before any waiting
    talking: bool = False
    phone_s: float = 0.0             # how long the phone pose has been held (0 when not on a call)
    hand_points: List[Tuple[float, float, bool]] = field(default_factory=list)   # x, y, at_ear
    hand_hides_face: bool = False    # the face is gone and a hand is where it was, so it is not "driver left"

    @property
    def label(self) -> str:
        return LABELS[self.activity]


def hand_center(hand: np.ndarray) -> np.ndarray:
    return hand[list(PALM), :2].mean(axis=0)


def hand_at_ear_geo(geo: Optional[Geometry], hand: np.ndarray, config: Config) -> bool:
    """True when the hand sits beside the head at about ear height."""
    if geo is None:
        return False
    w = geo.width
    c = hand_center(hand)
    near_ear = min(np.linalg.norm(c - geo.right), np.linalg.norm(c - geo.left)) < config.phone_ear_radius * w
    to_the_side = abs(c[0] - geo.centre_x) > config.phone_lateral_min * w
    at_head_height = geo.top - 0.2 * w < c[1] < geo.chin + 0.3 * w
    return bool(near_ear and to_the_side and at_head_height)


def hand_at_ear(face: Optional[np.ndarray], hand: np.ndarray, config: Config) -> bool:
    return hand_at_ear_geo(geometry_from(face), hand, config)


class TalkingDetector:
    """Counts open/close swings of the mouth. Talking swings often; a yawn opens once and stays open."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self._values: Deque[Tuple[float, float]] = deque()

    def reset(self) -> None:
        self._values.clear()

    def update(self, t: float, mar: float, usable: bool = True) -> bool:
        c = self.config
        if not usable or mar != mar:
            self._values.clear()
            return False
        self._values.append((t, mar))
        while self._values and t - self._values[0][0] > c.talk_window_s:
            self._values.popleft()
        if t - self._values[0][0] < 0.6 * c.talk_window_s:
            return False
        return self._reversals() >= c.talk_min_reversals

    def _reversals(self) -> int:
        step = self.config.talk_step
        extreme = self._values[0][1]
        direction = 0
        count = 0
        for _, v in list(self._values)[1:]:
            if direction >= 0 and v > extreme:
                extreme, direction = v, (1 if v - self._values[0][1] > step or direction == 1 else direction)
            elif direction <= 0 and v < extreme:
                extreme, direction = v, (-1 if self._values[0][1] - v > step or direction == -1 else direction)
            elif direction == 1 and extreme - v >= step:
                count += 1
                extreme, direction = v, -1
            elif direction == -1 and v - extreme >= step:
                count += 1
                extreme, direction = v, 1
            elif direction == 0 and abs(v - extreme) >= step:
                direction = 1 if v > extreme else -1
                extreme = v
        return count


class ActivityDetector:
    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self._talk = TalkingDetector(self.config)
        self.reset()

    def reset(self) -> None:
        self._talk.reset()
        self._pose_since: Optional[float] = None
        self._pose_off_since: Optional[float] = None
        self._in_call = False
        self._geo: Optional[Geometry] = None
        self._geo_t = -1e9
        self._face_gone_since: Optional[float] = None

    def update(self, t: float, face: Optional[np.ndarray], hands: Optional[List[np.ndarray]],
               mar: float = float("nan"), usable: bool = True) -> ActivityState:
        c = self.config
        fresh = geometry_from(face)
        if fresh is not None:
            self._geo, self._geo_t = fresh, t
            self._face_gone_since = None
        elif self._face_gone_since is None:
            self._face_gone_since = t
        geo = fresh if fresh is not None else (self._geo if t - self._geo_t <= c.phone_face_memory_s else None)

        pts: List[Tuple[float, float, bool]] = []
        pose = False
        hides_face = False
        for hand in hands or []:
            at_ear = hand_at_ear_geo(geo, hand, c)
            if fresh is None and geo is not None and geo.contains(hand_center(hand)):
                hides_face = True
            centre = hand_center(hand)
            pts.append((float(centre[0]), float(centre[1]), at_ear))
            pose = pose or at_ear

        if pose:
            self._pose_off_since = None
            if self._pose_since is None:
                self._pose_since = t
            if t - self._pose_since >= c.phone_pose_s:
                self._in_call = True
        else:
            if self._pose_since is not None or self._in_call:
                if self._pose_off_since is None:
                    self._pose_off_since = t
                if t - self._pose_off_since >= (c.phone_clear_s if self._in_call else 0.4):
                    self._pose_since, self._pose_off_since, self._in_call = None, None, False

        talking = self._talk.update(t, mar, usable and face is not None)
        phone_s = (t - self._pose_since) if (self._in_call and self._pose_since is not None) else 0.0
        if self._in_call:
            activity = Activity.PHONE_CALL
        elif talking:
            activity = Activity.TALKING
        else:
            activity = Activity.DRIVING
        hides_face = hides_face or (fresh is None and pose)
        if hides_face and self._face_gone_since is not None:
            hides_face = t - self._face_gone_since <= c.hand_hides_face_max_s
        return ActivityState(activity, pose, talking, phone_s, pts, hand_hides_face=bool(hides_face and fresh is None))
