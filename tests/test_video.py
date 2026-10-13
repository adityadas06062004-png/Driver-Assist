import sys

import cv2
import numpy as np
import pytest

from drowsiness import Config, Storage
from drowsiness import video as V
from helpers import make_face

FPS = 10


def write_clip(path, seconds, w=640, h=480):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (w, h))
    # Some texture is needed, otherwise the picture-quality check calls the frame blurry.
    small = np.random.default_rng(0).integers(70, 170, (h // 4, w // 4), dtype=np.uint8)
    base = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    frame = np.dstack([base, base, base])
    for _ in range(int(seconds * FPS)):
        writer.write(frame)
    writer.release()
    return path


class ScriptedFace:
    """Returns a face whose eyes are open, then closed for a while, then open again."""

    def __init__(self, closed_from=11.0, closed_until=15.0, step=1):
        self.n, self.step, self.closed = 0, step, (closed_from, closed_until)

    def extract(self, frame):
        t = self.n * self.step / FPS
        self.n += 1
        shut = self.closed[0] <= t < self.closed[1]
        return make_face(ear=0.05 if shut else 0.26)


def cfg():
    return Config(calibration_s=8.0, detect_phone=False)


def test_a_clip_is_calibrated_and_the_eyes_closed_stretch_raises_an_alarm(tmp_path):
    clip = write_clip(tmp_path / "clip.avi", 20)
    res = V.analyze_video(clip, cfg(), extractor=ScriptedFace())
    assert res.calibrated and res.fps == pytest.approx(FPS) and res.frames_read == 200
    s = res.storage.summary(res.session_id)
    assert s["alerts"]["CRITICAL"] >= 1 and s["max_risk"] > 75
    assert res.peak_frame is not None and res.peak_frame.shape == (480, 640, 3)
    assert 10.5 < res.duration_s < 20
    assert "Average risk" in V.describe(res)


def test_a_clip_shorter_than_the_calibration_time_explains_itself(tmp_path):
    clip = write_clip(tmp_path / "short.avi", 4)
    res = V.analyze_video(clip, cfg(), extractor=ScriptedFace())
    assert not res.calibrated and res.peak_frame is None
    assert "Calibration did not finish" in V.describe(res)


def test_stride_skips_frames_but_keeps_the_timing(tmp_path):
    clip = write_clip(tmp_path / "clip.avi", 20)
    res = V.analyze_video(clip, cfg(), extractor=ScriptedFace(step=2), stride=2)
    assert res.frames_processed == 100 and res.frames_read == 200
    assert res.storage.summary(res.session_id)["alerts"]["CRITICAL"] >= 1


def test_max_seconds_stops_early(tmp_path):
    clip = write_clip(tmp_path / "clip.avi", 20)
    res = V.analyze_video(clip, cfg(), extractor=ScriptedFace(), max_seconds=5)
    assert res.frames_read <= 52 and not res.calibrated


def test_an_annotated_copy_can_be_written(tmp_path):
    clip = write_clip(tmp_path / "clip.avi", 12)
    out = tmp_path / "out" / "annotated.mp4"
    res = V.analyze_video(clip, cfg(), extractor=ScriptedFace(), annotate_to=out)
    assert res.annotated_path == out and out.exists() and out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 50


def test_large_frames_are_scaled_down(tmp_path):
    clip = write_clip(tmp_path / "big.avi", 3, w=1280, h=720)
    res = V.analyze_video(clip, cfg(), extractor=ScriptedFace(), max_width=640)
    assert res.frames_read == 30


def test_missing_file_gives_a_clear_error(tmp_path):
    with pytest.raises(ValueError):
        V.analyze_video(tmp_path / "nope.mp4", cfg(), extractor=ScriptedFace())


def test_sessions_can_be_saved_to_a_real_database(tmp_path):
    clip = write_clip(tmp_path / "clip.avi", 12)
    db = Storage(tmp_path / "x.db")
    res = V.analyze_video(clip, cfg(), storage=db, extractor=ScriptedFace())
    assert db.sessions()[0]["note"] == "video: clip.avi" and res.session_id == db.sessions()[0]["id"]


def test_video_command_runs_end_to_end(tmp_path, monkeypatch, capsys):
    clip = write_clip(tmp_path / "clip.avi", 20)
    monkeypatch.setattr(V, "FaceLandmarkExtractor", lambda: type("F", (ScriptedFace,), {"close": lambda self: None})())
    monkeypatch.setattr(sys, "argv", ["video", str(clip), "--no-phone", "--db", str(tmp_path / "v.db")])
    V.main()
    out = capsys.readouterr().out
    assert "Average risk" in out and "Saved as a session" in out
