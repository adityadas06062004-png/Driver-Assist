
from drowsiness import FrameStats, HealthMonitor, Issue, Level

GOOD = FrameStats(120, 300, 0.0, 0.3)


def run(h, t, seconds, dt=0.1, **kw):
    out = None
    for _ in range(int(round(seconds / dt))):
        out = h.update(t, **kw)
        t += dt
    return t, out


def test_healthy_driver_has_no_issues():
    _, out = run(HealthMonitor(), 0, 5, face_present=True, stats=GOOD, yaw=2.0)
    assert not out.issues and out.reliable and out.level == Level.OK and out.message == ""


def test_no_face_escalates_warning_then_critical_and_recovers():
    h = HealthMonitor()
    t, out = run(h, 0, 0.3, face_present=False)
    assert Issue.NO_FACE not in out.issues                     # still debouncing
    t, out = run(h, t, 1.0, face_present=False)
    assert Issue.NO_FACE in out.issues and out.level == Level.OK
    t, out = run(h, t, 2.0, face_present=False)                # about 3.3 s missing
    assert out.level == Level.WARNING and "Face not visible" in out.message
    t, out = run(h, t, 3.0, face_present=False)                # about 6.3 s missing
    assert out.level == Level.CRITICAL
    t, out = run(h, t, 1.5, face_present=True, stats=GOOD)
    assert out.level == Level.OK and Issue.NO_FACE not in out.issues


def test_low_light_is_reported_but_does_not_block_measuring():
    t, out = run(HealthMonitor(), 0, 1.5, face_present=True, stats=FrameStats(25, 300, 0.0, 0.3))
    assert Issue.LOW_LIGHT in out.issues and out.reliable


def test_low_light_without_face_is_reported_too():
    h = HealthMonitor()
    t, out = run(h, 0, 1.5, face_present=False, stats=FrameStats(15, float("nan"), float("nan"), float("nan")))
    assert {Issue.LOW_LIGHT, Issue.NO_FACE} <= out.issues


def test_glare_blocks_measuring_then_clears():
    h = HealthMonitor()
    t, out = run(h, 0, 1.5, face_present=True, stats=FrameStats(120, 300, 0.4, 0.3))
    assert Issue.GLARE in out.issues and not out.reliable and "Glare" in out.message
    t, out = run(h, t, 0.5, face_present=True, stats=GOOD)
    assert Issue.GLARE in out.issues                           # not cleared yet (needs 1 s)
    t, out = run(h, t, 1.5, face_present=True, stats=GOOD)
    assert out.reliable and not out.issues


def test_blur_and_distance_block_measuring():
    t, out = run(HealthMonitor(), 0, 1.5, face_present=True, stats=FrameStats(120, 4, 0.0, 0.3))
    assert Issue.BLURRY in out.issues and not out.reliable
    t, out = run(HealthMonitor(), 0, 1.5, face_present=True, stats=FrameStats(120, 300, 0.0, 0.05))
    assert Issue.FACE_TOO_SMALL in out.issues and not out.reliable


def test_looking_away_needs_two_seconds():
    h = HealthMonitor()
    t, out = run(h, 0, 1.5, face_present=True, stats=GOOD, yaw=45)
    assert Issue.LOOKING_AWAY not in out.issues and out.reliable
    t, out = run(h, t, 1.0, face_present=True, stats=GOOD, yaw=45)
    assert Issue.LOOKING_AWAY in out.issues and not out.reliable


def test_camera_lost_goes_critical():
    h = HealthMonitor()
    t, out = run(h, 0, 1.0, face_present=False, camera_ok=False)
    assert Issue.CAMERA_LOST in out.issues and out.level == Level.OK and "Camera lost" in out.message
    assert Issue.NO_FACE not in out.issues                      # not blamed on the driver
    t, out = run(h, t, 1.5, face_present=False, camera_ok=False)
    assert out.level == Level.CRITICAL


def test_flickering_condition_is_debounced():
    h = HealthMonitor()
    t = 0.0
    for i in range(30):
        bad = i % 2 == 0
        out = h.update(t, face_present=True,
                       stats=FrameStats(120, 300, 0.5 if bad else 0.0, 0.3))
        t += 0.2
    assert Issue.GLARE not in out.issues


def test_low_fps_reported():
    t, out = run(HealthMonitor(), 0, 1.5, face_present=True, stats=GOOD, fps=6.0)
    assert Issue.LOW_FPS in out.issues and out.reliable
