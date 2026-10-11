import math

import numpy as np
import pytest

from drowsiness import Calibration, Config, Level, Monitor
from drowsiness.activity import Activity, ActivityDetector, TalkingDetector, hand_at_ear
from helpers import FRAME, add_head_landmarks, make_face, make_hand

CAL = Calibration(True, "ok", 0.26, 0.195, 0.0, 0.5, 0.0, None, 500)


def face():
    return add_head_landmarks(make_face(ear=0.26))


@pytest.mark.parametrize("cx,cy,expected", [
    (190, 260, True),        # beside the right ear
    (450, 255, True),        # beside the left ear
    (320, 330, False),       # in front of the mouth
    (320, 250, False),       # in front of the nose
    (200, 720, False),       # far below the chin, down at the chest
    (600, 250, False),       # far off to the side
])
def test_hand_position_rules(cx, cy, expected):
    assert hand_at_ear(face(), make_hand(cx, cy), Config()) is expected


def test_no_face_means_no_phone_pose():
    assert hand_at_ear(None, make_hand(190, 260), Config()) is False


def run(det, t, seconds, hands, mar=0.05, fps=30, f=None):
    f = face() if f is None else f
    out = None
    for _ in range(int(seconds * fps)):
        out = det.update(t, f, hands, mar)
        t += 1 / fps
    return t, out


def test_phone_call_needs_the_pose_to_be_held_then_ends_after_it_is_dropped():
    det = ActivityDetector()
    hand = [make_hand(190, 260)]
    t, out = run(det, 0, 1.0, hand)
    assert out.hand_at_ear and out.activity == Activity.DRIVING          # not long enough yet
    t, out = run(det, t, 1.5, hand)
    assert out.activity == Activity.PHONE_CALL and out.phone_s > 2.0
    t, out = run(det, t, 0.5, [])
    assert out.activity == Activity.PHONE_CALL                           # a short dropout is tolerated
    t, out = run(det, t, 1.5, [])
    assert out.activity == Activity.DRIVING and out.phone_s == 0


def test_a_quick_scratch_is_not_a_call():
    det = ActivityDetector()
    t, out = run(det, 0, 1.0, [make_hand(190, 260)])
    t, out = run(det, t, 3.0, [])
    assert out.activity == Activity.DRIVING


def test_a_hand_in_front_of_the_face_is_never_a_call():
    det = ActivityDetector()
    _, out = run(det, 0, 6.0, [make_hand(320, 330)])
    assert out.activity == Activity.DRIVING and not out.hand_at_ear


def mouth_series(det, t, seconds, fn, fps=30):
    out = False
    for _ in range(int(seconds * fps)):
        out = det.update(t, fn(t))
        t += 1 / fps
    return t, out


def test_talking_is_mouth_swinging_but_a_yawn_or_a_closed_mouth_is_not():
    talk = TalkingDetector()
    _, talking = mouth_series(talk, 0, 4, lambda t: 0.12 + 0.10 * math.sin(2 * math.pi * 3 * t))
    assert talking
    _, quiet = mouth_series(TalkingDetector(), 0, 4, lambda t: 0.05)
    assert not quiet
    _, yawn = mouth_series(TalkingDetector(), 0, 4, lambda t: 0.9)
    assert not yawn
    small = mouth_series(TalkingDetector(), 0, 4, lambda t: 0.05 + 0.005 * math.sin(40 * t))[1]
    assert not small                                                      # tiny jitter is noise


def test_talking_detector_forgets_when_the_picture_is_unusable():
    d = TalkingDetector()
    mouth_series(d, 0, 4, lambda t: 0.12 + 0.10 * math.sin(2 * math.pi * 3 * t))
    assert d.update(4.1, 0.1, usable=False) is False


def test_phone_call_beats_talking():
    det = ActivityDetector()
    t = 0.0
    out = None
    for _ in range(150):
        out = det.update(t, face(), [make_hand(190, 260)], 0.12 + 0.10 * math.sin(2 * math.pi * 3 * t))
        t += 1 / 30
    assert out.talking and out.activity == Activity.PHONE_CALL


def feed(mon, t, seconds, hands, **kw):
    pts = add_head_landmarks(make_face(ear=0.26))
    out = []
    for _ in range(int(seconds * 30)):
        out.append(mon.update(pts, t, FRAME, hands=hands))
        t += 1 / 30
    return t, out


def test_monitor_warns_after_five_seconds_on_the_phone():
    mon = Monitor(calibration=CAL)
    t, out = feed(mon, 0, 2, [])
    t, out = feed(mon, t, 4, [make_hand(190, 260)])
    assert out[-1].activity.activity == Activity.PHONE_CALL and out[-1].alert.level == Level.OK
    t, out = feed(mon, t, 3, [make_hand(190, 260)])
    assert out[-1].alert.level >= Level.WARNING and "Phone" in out[-1].message
    assert any(s.alert.sound for s in out)
    t, out = feed(mon, t, 12, [])
    assert out[-1].alert.level == Level.OK


def test_phone_warning_can_be_turned_off_and_detection_can_be_disabled():
    mon = Monitor(Config(phone_alert_s=0), calibration=CAL)
    t, out = feed(mon, 0, 10, [make_hand(190, 260)])
    assert out[-1].activity.activity == Activity.PHONE_CALL and out[-1].alert.level == Level.OK
    mon = Monitor(Config(detect_phone=False), calibration=CAL)
    t, out = feed(mon, 0, 10, [make_hand(190, 260)])
    assert out[-1].activity is None and out[-1].alert.level == Level.OK


def test_driving_normally_stays_quiet():
    mon = Monitor(calibration=CAL)
    t, out = feed(mon, 0, 10, [make_hand(320, 560)])        # hand on the wheel, low in the frame
    assert out[-1].activity.activity == Activity.DRIVING and out[-1].alert.level == Level.OK


def test_recorder_logs_a_phone_call_and_the_summary_counts_it():
    from drowsiness import SessionRecorder, Storage
    db = Storage(":memory:")
    rec = SessionRecorder(db)
    mon = Monitor(calibration=CAL)
    t = 0.0
    pts = add_head_landmarks(make_face(ear=0.26))
    for hands, secs in (([], 1), ([make_hand(190, 260)], 4), ([], 3)):
        for _ in range(int(secs * 30)):
            rec.update(mon.update(pts, t, FRAME, hands=hands))
            t += 1 / 30
    kinds = [e["kind"] for e in db.events(rec.session_id)]
    assert kinds.count("phone_start") == 1 and kinds.count("phone_end") == 1
    assert db.summary(rec.session_id)["phone_calls"] == 1


def test_overlay_draws_the_activity_and_hand_markers():
    from drowsiness.overlay import draw_hud
    mon = Monitor(calibration=CAL)
    t, out = feed(mon, 0, 4, [make_hand(190, 260)])
    img = np.zeros((480, 640, 3), np.uint8)
    draw_hud(img, out[-1], t)
    assert img.any()


def test_qa_scores_the_phone_scenarios():
    from drowsiness.qa import Sample, summarize
    def smp(act, i):
        return Sample(i * 0.1, True, True, 0.25, 120, 300, 0.0, 0, 0, activity=act)
    ok = summarize("Phone call", "phone", [smp("driving", 0), smp("phone_call", 1)])
    assert ok.passed and "recognised" in ok.detail
    assert not summarize("Phone call", "phone", [smp("driving", 0)]).passed
    assert summarize("Touch", "no_phone", [smp("driving", 0), smp("talking", 1)]).passed
    assert not summarize("Touch", "no_phone", [smp("phone_call", 0)]).passed


def test_pipeline_passes_hands_to_the_monitor():
    from drowsiness import Camera
    from drowsiness.pipeline import Pipeline
    from test_pipeline import CAL as PCAL
    from test_pipeline import Cap, Clock, Extractor

    class Hands:
        def __init__(self):
            self.calls = 0

        def extract(self, frame):
            self.calls += 1
            return [make_hand(190, 260)]

    switch, clock = {"ok": True}, Clock()
    cam = Camera(capture_factory=lambda i: Cap(switch), clock=clock)
    cam.open()

    class FaceWithEars(Extractor):
        def extract(self, frame):
            return add_head_landmarks(super().extract(frame))

    hands = Hands()
    pipe = Pipeline(Monitor(calibration=PCAL), cam, FaceWithEars(dict(ear=0.26)), hands=hands,
                    clock=clock, perf_clock=clock)
    steps = [pipe.step() for _ in range(120)]
    assert hands.calls >= 30
    assert steps[-1].state.activity.activity == Activity.PHONE_CALL


def test_phone_pose_is_still_recognised_when_the_hand_hides_the_face():
    det = ActivityDetector()
    hand = [make_hand(190, 260)]
    t, _ = run(det, 0, 1.0, [])                            # face seen, hands down
    out = None
    for _ in range(int(4 * 30)):                           # now the face disappears behind the phone
        out = det.update(t, None, hand, float("nan"))
        t += 1 / 30
    assert out.activity == Activity.PHONE_CALL and out.hand_hides_face


def test_pose_cannot_be_judged_once_the_remembered_face_is_too_old():
    det = ActivityDetector()
    t, _ = run(det, 0, 1.0, [])
    for _ in range(int(12 * 30)):
        out = det.update(t, None, [make_hand(190, 260)], float("nan"))
        t += 1 / 30
    assert out.activity == Activity.DRIVING and not out.hand_hides_face


def test_a_hand_over_the_face_is_not_driver_gone_for_a_while_but_is_after_eight_seconds():
    mon = Monitor(calibration=CAL)
    face_pts = add_head_landmarks(make_face(ear=0.26))
    t = 0.0
    for _ in range(60):
        mon.update(face_pts, t, FRAME, hands=[])
        t += 1 / 30
    over_face = [make_hand(320, 250)]                      # right in front of the nose
    out = []
    for _ in range(int(6 * 30)):
        out.append(mon.update(None, t, FRAME, hands=over_face))
        t += 1 / 30
    assert out[-1].alert.level == Level.OK and out[-1].activity.hand_hides_face
    for _ in range(int(10 * 30)):
        out.append(mon.update(None, t, FRAME, hands=over_face))
        t += 1 / 30
    assert out[-1].alert.level >= Level.WARNING and "Face not visible" in out[-1].message


def test_no_hand_means_a_missing_face_still_alarms_as_before():
    mon = Monitor(calibration=CAL)
    face_pts = add_head_landmarks(make_face(ear=0.26))
    t = 0.0
    for _ in range(60):
        mon.update(face_pts, t, FRAME, hands=[])
        t += 1 / 30
    out = []
    for _ in range(int(7 * 30)):
        out.append(mon.update(None, t, FRAME, hands=[]))
        t += 1 / 30
    assert out[-1].alert.level >= Level.WARNING


def test_dominant_cause_names_the_biggest_ingredient():
    from drowsiness.risk import dominant_cause
    cfg = Config()
    assert dominant_cause({"closure": 1.0}, cfg) == "eyes closed now"
    assert dominant_cause({"nod": 0.8}, cfg, nodding=True) == "head down"
    assert dominant_cause({"perclos": 1.0, "yawn": 0.2}, cfg) == "PERCLOS"
    assert dominant_cause({}, cfg) == "none"


def test_qa_names_the_cause_of_a_false_alarm():
    from drowsiness.qa import Sample, summarize
    s = [Sample(0.0, True, True, 0.25, 120, 300, 0.0, 0, 0),
         Sample(2.7, True, True, 0.25, 120, 300, 0.0, 2, 0, cause="head down")]
    r = summarize("Talking", "quiet", s)
    assert not r.passed and "FALSE ALARM at 2.7s (head down)" in r.detail
