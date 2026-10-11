"""Record labelled examples for training the learned risk model.

    driver-assist collect --label alert  --seconds 60
    driver-assist collect --label drowsy --seconds 45

It calibrates first, then records one row per second (numbers only, no images) to a CSV file.
Each run is one recording session with one label. Record several of each, in different light,
so the model is tested on sessions it has never seen.

Important: "drowsy" here is acted (slow long blinks, drooping eyes, yawns, nodding). Do not do it
while actually driving, and remember that acted data from one person is not real drowsiness data.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import List, Optional

import cv2

from .camera import Camera
from .config import Config
from .landmarks import FaceLandmarkExtractor
from .mlmodel import FEATURES, feature_vector
from .monitor import Monitor
from .overlay import _centered, draw_hud
from .pipeline import Pipeline

DEFAULT_DATA = Path("~/.driver_assist/ml_samples.csv")
COLUMNS = ["session", "label", "t", "rule_risk", *FEATURES]
COUNTDOWN_S = 4.0
INSTRUCTIONS = {
    "alert": "Look at the road. Drive normally: blink, glance at the mirrors, talk.",
    "drowsy": "ACT drowsy: slow long blinks, droopy eyes, yawn, let your head nod.",
}


def make_row(session: str, label: str, state) -> Optional[List]:
    """One CSV row for a monitoring-phase state, or None when the frame is unusable."""
    base = state.calibration.ear_baseline if state.calibration else None
    vec = feature_vector(state.result, base)
    if vec is None or state.risk is None:
        return None
    return [session, label, round(state.result.t, 2), round(state.risk.score, 1),
            *[round(v, 4) for v in vec]]


def append_rows(path: Path, rows: List[List]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(COLUMNS)
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", choices=("alert", "drowsy"), required=True)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--night", choices=("auto", "on", "off"), default="auto")
    args = parser.parse_args()

    cfg = Config(night_mode=args.night)
    camera = Camera(args.camera)
    if not camera.open():
        raise SystemExit("Could not open the camera.")
    monitor = Monitor(cfg)
    session = time.strftime("%Y%m%d-%H%M%S") + "-" + args.label
    rows: List[List] = []
    stage, stage_start, last_row_t = "calibrate", 0.0, -1.0

    with FaceLandmarkExtractor() as extractor:
        pipe = Pipeline(monitor, camera, extractor, cfg)
        while True:
            step = pipe.step()
            now, view = step.state.result.t, step.view
            draw_hud(view, step.state, now, pipe.proc_fps or 0.0)
            h = view.shape[0]
            if stage == "calibrate":
                if step.state.phase == "monitoring":
                    stage, stage_start = "countdown", now
            elif stage == "countdown":
                _centered(view, INSTRUCTIONS[args.label], 75, 0.6)
                left = COUNTDOWN_S - (now - stage_start)
                _centered(view, f"Get ready... {max(left, 0):.0f}", h // 2, 1.6, (0, 215, 255))
                if left <= 0:
                    monitor.engine.reset_windows()
                    monitor.scorer.reset()
                    monitor.alerts.reset()
                    stage, stage_start = "record", now
            else:
                left = args.seconds - (now - stage_start)
                _centered(view, INSTRUCTIONS[args.label], 75, 0.6)
                _centered(view, f"RECORDING {args.label.upper()}  {max(left, 0):.0f}s", h // 2, 1.1, (0, 255, 0))
                if step.new and step.state.phase == "monitoring" and now - last_row_t >= 1.0:
                    row = make_row(session, args.label, step.state)
                    if row:
                        rows.append(row)
                        last_row_t = now
                if left <= 0:
                    break
            cv2.imshow("Driver Assist: collect (q to cancel)", view)
            if cv2.waitKey(1 if step.camera_ok else 50) & 0xFF in (ord("q"), 27):
                rows = []
                break

    camera.release()
    cv2.destroyAllWindows()
    if not rows:
        print("Nothing saved.")
        return
    append_rows(Path(args.data), rows)
    print(f"Saved {len(rows)} rows ({session}) to {Path(args.data).expanduser()}")


if __name__ == "__main__":
    main()
