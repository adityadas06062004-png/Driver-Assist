import math

import pytest

from drowsiness import features as F
from helpers import FRAME, make_face


def test_ear_open_and_closed():
    assert F.mean_ear(make_face(ear=0.30)) == pytest.approx(0.30, abs=1e-6)
    assert F.mean_ear(make_face(ear=0.05)) == pytest.approx(0.05, abs=1e-6)


def test_ear_scale_invariant():
    pts = make_face(ear=0.25)
    assert F.mean_ear(pts * 3.0) == pytest.approx(F.mean_ear(pts), abs=1e-6)


def test_mar_closed_and_yawn():
    assert F.mouth_aspect_ratio(make_face(mar=0.05)) == pytest.approx(0.05, abs=1e-6)
    assert F.mouth_aspect_ratio(make_face(mar=0.9)) == pytest.approx(0.9, abs=1e-6)


def test_head_pose_frontal_is_near_zero():
    pitch, yaw, roll = F.head_pose(make_face(), FRAME)
    assert abs(pitch) < 1.0 and abs(yaw) < 1.0 and abs(roll) < 1.0


@pytest.mark.parametrize("pitch,yaw,roll", [
    (20, 0, 0), (-20, 0, 0), (0, 25, 0), (0, -25, 0), (0, 0, 15), (0, 0, -15), (15, -10, 5),
])
def test_head_pose_recovers_known_angles(pitch, yaw, roll):
    got = F.head_pose(make_face(pitch=pitch, yaw=yaw, roll=roll), FRAME)
    assert got == pytest.approx((pitch, yaw, roll), abs=1.5)


def test_pitch_positive_means_head_down_physically():
    """Independent of the decode maths: nodding down shrinks the eye-to-nose gap."""
    def eye_nose_gap(pitch):
        pts = make_face(pitch=pitch)
        eye_y = (pts[33, 1] + pts[263, 1]) / 2
        return pts[1, 1] - eye_y

    assert eye_nose_gap(25) < eye_nose_gap(0) < eye_nose_gap(-25)


def test_yaw_positive_means_turned_toward_image_right_physically():
    """Turning toward image right moves the nose tip right of the eye midpoint."""
    def nose_offset(yaw):
        pts = make_face(yaw=yaw)
        return pts[1, 0] - (pts[33, 0] + pts[263, 0]) / 2

    assert nose_offset(25) > nose_offset(0) > nose_offset(-25)


def test_head_pose_nan_safe():
    assert not math.isnan(F.head_pose(make_face(), FRAME)[0])
