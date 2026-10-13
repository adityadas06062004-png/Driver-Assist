"""Run the detector on a recorded video file instead of the live camera.

    driver-assist video clip.mp4
    driver-assist video clip.mp4 --out annotated.mp4 --stride 2

Time comes from the video's own frame rate, so a two minute clip is analysed as two minutes of driving
however fast it is processed. The first seconds are used to calibrate (the driver should look at the road
with eyes open), so a clip needs to be longer than the calibration time to produce any risk score.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .calibration import Calibration
from .config import Config
from .landmarks import FaceLandmarkExtractor, HandLandmarkExtractor
from .monitor import Monitor
from .overlay import draw_hud
from .quality import NightEnhancer, measure, prepare_for_detection
from .recorder import SessionRecorder
from .storage import DEFAULT_DB, Storage


@dataclass
class VideoResult:
    session_id: int
    storage: Storage
    fps: float
    frames_read: int
    frames_processed: int
    duration_s: float
    calibrated: bool
    message: str                       # why calibration failed, if it did
    peak_risk: float
    peak_frame: Optional[np.ndarray]   # BGR frame with the on-screen display, at the highest risk
    annotated_path: Optional[Path]


def analyze_video(path, config: Optional[Config] = None, storage: Optional[Storage] = None,
                  extractor=None, hands=None, stride: int = 1, calibration: Optional[Calibration] = None,
                  annotate_to=None, max_width: int = 960, max_seconds: Optional[float] = None,
                  progress: Optional[Callable[[float], None]] = None) -> VideoResult:
    """Analyse a video file. ``extractor`` and ``hands`` are objects with ``extract(frame)``; the real
    MediaPipe ones are created when they are not given."""
    cfg = config or Config()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open the video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    storage = storage or Storage(":memory:")
    monitor = Monitor(cfg, calibration=calibration)
    recorder = SessionRecorder(storage, camera=-1, note=f"video: {Path(str(path)).name}")
    night = NightEnhancer(cfg.night_clahe_clip, cfg.night_blend)

    own_face = extractor is None
    if own_face:
        extractor = FaceLandmarkExtractor()
    own_hands = hands is None and cfg.detect_phone and own_face
    if own_hands:
        hands = HandLandmarkExtractor()
    use_hands = hands is not None and cfg.detect_phone

    writer = None
    out_path = Path(annotate_to) if annotate_to else None
    peak, peak_frame, last_hands, hand_calls = 0.0, None, [], 0
    index = processed = 0
    state = None
    last_t = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = index / fps
            index += 1
            if max_seconds is not None and t > max_seconds:
                break
            h, w = frame.shape[:2]
            if w > max_width:
                frame = cv2.resize(frame, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]

            if (index - 1) % max(1, stride) == 0:
                seen, _ = prepare_for_detection(frame, cfg, night)
                landmarks = extractor.extract(seen)
                if use_hands:
                    if hand_calls % max(1, cfg.hand_every_n) == 0:
                        last_hands = hands.extract(seen)
                    hand_calls += 1
                state = monitor.update(landmarks, t, (w, h), stats=measure(frame, landmarks), hands=last_hands)
                recorder.update(state)
                processed += 1
                last_t = t
                if state.risk and state.risk.score >= peak and state.phase == "monitoring":
                    peak = state.risk.score
                    snap = frame.copy()
                    draw_hud(snap, state, t, mirror=False)
                    peak_frame = snap

            if out_path is not None and state is not None:
                if writer is None:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                view = frame.copy()
                draw_hud(view, state, t, mirror=False)
                writer.write(view)
            if progress and total and index % 15 == 0:
                progress(min(index / total, 1.0))
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        recorder.close()
        if own_face:
            extractor.close()
        if own_hands:
            hands.close()
    if progress:
        progress(1.0)
    calibrated = monitor.phase == "monitoring"
    return VideoResult(recorder.session_id, storage, fps, index, processed, last_t, calibrated,
                       "" if calibrated else (state.message if state else "no frames could be read"),
                       peak, peak_frame, out_path if writer is not None else None)


def describe(result: VideoResult) -> str:
    s = result.storage.summary(result.session_id)
    lines = [f"Video length analysed: {result.duration_s:.0f}s ({result.frames_processed} frames processed)"]
    if not result.calibrated:
        lines.append(f"Calibration did not finish: {result.message or 'the clip is shorter than the calibration time'}")
        return "\n".join(lines)
    levels = s["seconds_at_level"]
    lines += [f"Average risk {s['avg_risk']:.0f}, peak {s['max_risk']:.0f}",
              "Seconds at each level: " + ", ".join(f"{k} {v}" for k, v in levels.items()),
              f"Warnings {s['alerts']['WARNING']}, critical alerts {s['alerts']['CRITICAL']}, "
              f"yawns {s['yawns']}, head nods {s['nods']}, phone calls {s['phone_calls']}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path")
    parser.add_argument("--out", default=None, help="write a copy of the video with the on-screen display")
    parser.add_argument("--stride", type=int, default=1, help="analyse every Nth frame (faster)")
    parser.add_argument("--calibration-seconds", type=float, default=8.0)
    parser.add_argument("--calibration-file", default=None, help="use a saved calibration instead of the first seconds")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--night", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--no-phone", action="store_true", help="skip hand detection (faster)")
    parser.add_argument("--no-log", action="store_true", help="do not save the session to the database")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()

    cfg = Config(calibration_s=args.calibration_seconds, night_mode=args.night, detect_phone=not args.no_phone)
    saved = None
    if args.calibration_file:
        cal_path = Path(args.calibration_file).expanduser()
        if not cal_path.exists():
            raise SystemExit(f"No calibration file at {cal_path}")
        saved = Calibration.load(cal_path)
    storage_ctx = Storage(":memory:") if args.no_log else Storage(args.db)
    started = time.time()
    with storage_ctx as storage:
        result = analyze_video(args.path, cfg, storage=storage, stride=args.stride, calibration=saved,
                               annotate_to=args.out, max_seconds=args.max_seconds,
                               progress=lambda f: print(f"\r{f * 100:3.0f}%", end="", flush=True))
        print("\r    \r" + describe(result))
        print(f"Took {time.time() - started:.0f}s." + (f" Annotated video: {result.annotated_path}"
                                                       if result.annotated_path else ""))
        if not args.no_log:
            print("Saved as a session. Open it with: driver-assist dashboard")


if __name__ == "__main__":
    main()
