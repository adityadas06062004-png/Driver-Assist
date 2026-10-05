import numpy as np

from drowsiness import Camera, FrameGovernor


def test_governor_skips_when_slow_and_recovers():
    g = FrameGovernor(camera_fps=30)
    assert g.stride == 1
    for _ in range(10):
        g.record(0.07)                       # 70 ms per frame at 30 fps camera
    assert g.stride >= 2
    flags = [g.should_process() for _ in range(6)]
    assert flags.count(True) < 6
    for _ in range(60):
        g.record(0.005)
    assert g.stride == 1 and g.effective_fps == 30


def test_governor_caps_stride():
    g = FrameGovernor(camera_fps=30, max_stride=3)
    for _ in range(50):
        g.record(1.0)
    assert g.stride == 3


class FakeCap:
    def __init__(self, reads):
        self.reads = list(reads)
        self.released = False
        self.props = {}

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.props[prop] = value

    def read(self):
        ok = self.reads.pop(0) if self.reads else True
        return (ok, np.zeros((480, 640, 3), np.uint8) if ok else None)

    def release(self):
        self.released = True


class ClosedCap(FakeCap):
    def isOpened(self):
        return False


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_camera_recovers_after_failed_reads():
    clock, made = Clock(), []

    def factory(index):
        made.append(FakeCap([True, False] if len(made) == 0 else []))
        return made[-1]

    cam = Camera(capture_factory=factory, clock=clock, reopen_interval_s=1.0)
    assert cam.open()
    assert cam.read()[0] is True
    assert cam.read() == (False, None) and cam.failures == 1 and made[0].released
    clock.t = 0.3
    assert cam.read() == (False, None) and len(made) == 1       # too soon to reopen
    clock.t = 1.5
    ok, frame = cam.read()
    assert ok and frame is not None and len(made) == 2 and cam.failures == 0


def test_camera_that_cannot_open_keeps_retrying():
    clock, made = Clock(), []

    def factory(index):
        made.append(ClosedCap([]))
        return made[-1]

    cam = Camera(capture_factory=factory, clock=clock, reopen_interval_s=1.0)
    assert not cam.open()
    assert cam.read() == (False, None)
    clock.t = 2.0
    assert cam.read() == (False, None)
    assert len(made) >= 3 and all(m.released for m in made)
