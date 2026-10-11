import numpy as np

from drowsiness import Calibration, Camera, Level, Monitor
from drowsiness.pipeline import Pipeline
from helpers import make_face

CAL = Calibration(True, "ok", 0.26, 0.195, 0.0, 0.5, 0.0, None, 500)


class Clock:
    def __init__(self, dt=1 / 30):
        self.t, self.dt = 0.0, dt

    def __call__(self):
        self.t += self.dt
        return self.t


class Cap:
    """Camera that serves noisy frames while ``ok`` is True."""

    def __init__(self, switch):
        self.switch, self.rng = switch, np.random.default_rng(3)

    def isOpened(self):
        return True

    def set(self, *a):
        pass

    def release(self):
        pass

    def read(self):
        if not self.switch["ok"]:
            return False, None
        return True, np.clip(self.rng.normal(120, 30, (480, 640, 3)), 0, 255).astype(np.uint8)


class Extractor:
    def __init__(self, face):
        self.face, self.calls = face, 0

    def extract(self, frame):
        self.calls += 1
        return make_face(**self.face) if self.face is not None else None


def build(face):
    switch, clock = {"ok": True}, Clock()
    cam = Camera(capture_factory=lambda i: Cap(switch), clock=clock, reopen_interval_s=0.5)
    cam.open()
    ext = Extractor(face)
    mon = Monitor(calibration=CAL)
    pipe = Pipeline(mon, cam, ext, clock=clock, perf_clock=clock)
    return pipe, switch, ext


def test_pipeline_runs_and_alarms_on_closed_eyes():
    pipe, _, ext = build(dict(ear=0.05))
    steps = [pipe.step() for _ in range(120)]
    assert all(s.camera_ok and s.view.shape == (480, 640, 3) for s in steps)
    assert any(s.state.alert and s.state.alert.level == Level.CRITICAL for s in steps)
    assert steps[-1].stats is not None and ext.calls > 0


def test_pipeline_reports_camera_loss_and_recovery():
    pipe, switch, _ = build(dict(ear=0.26))
    for _ in range(30):
        pipe.step()
    switch["ok"] = False
    lost = [pipe.step() for _ in range(120)]
    assert not lost[-1].camera_ok and lost[-1].view.sum() == 0
    assert lost[-1].state.alert.level == Level.CRITICAL
    switch["ok"] = True
    back = [pipe.step() for _ in range(400)]
    assert back[-1].camera_ok and back[-1].state.alert.level == Level.OK


def test_pipeline_skips_frames_when_processing_is_slow():
    switch, clock = {"ok": True}, Clock()
    perf = Clock(dt=0.08)                       # every perf_clock() call advances 80 ms
    cam = Camera(capture_factory=lambda i: Cap(switch), clock=clock)
    cam.open()
    ext = Extractor(dict(ear=0.26))
    pipe = Pipeline(Monitor(calibration=CAL), cam, ext, clock=clock, perf_clock=perf)
    steps = [pipe.step() for _ in range(60)]
    assert ext.calls < 60 and any(not s.new for s in steps)


class DarkCap(Cap):
    def read(self):
        if not self.switch["ok"]:
            return False, None
        return True, np.clip(self.rng.normal(20, 5, (480, 640, 3)), 0, 255).astype(np.uint8)


class SeeingExtractor(Extractor):
    """Remembers how bright the picture was when the detector looked at it."""

    def __init__(self, face):
        super().__init__(face)
        self.seen = []

    def extract(self, frame):
        self.seen.append(float(frame.mean()))
        return super().extract(frame)


def dark_pipe(mode):
    from drowsiness import Config
    switch, clock = {"ok": True}, Clock()
    cam = Camera(capture_factory=lambda i: DarkCap(switch), clock=clock)
    cam.open()
    ext = SeeingExtractor(dict(ear=0.26))
    cfg = Config(night_mode=mode)
    pipe = Pipeline(Monitor(cfg, calibration=CAL), cam, ext, cfg, clock=clock, perf_clock=clock)
    for _ in range(20):
        pipe.step()
    return pipe, ext


def test_auto_night_mode_switches_on_in_the_dark():
    pipe, ext = dark_pipe("auto")
    assert pipe.night_active and min(ext.seen) > 60      # the detector saw a lifted picture


def test_night_mode_off_leaves_the_picture_to_the_basic_brightening():
    pipe, _ = dark_pipe("off")
    assert not pipe.night_active


def test_auto_night_mode_stays_off_in_normal_light():
    pipe, _, = build(dict(ear=0.26))[0], None
    for _ in range(10):
        pipe.step()
    assert not pipe.night_active
