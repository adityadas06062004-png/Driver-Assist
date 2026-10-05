"""Live monitor: calibrate, then watch the driver and alert.

    python -m drowsiness.live
    python -m drowsiness.live --no-sound --calibration-file ~/.driver_assist/me.json

Keys: q or Esc = quit, r = recalibrate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .alerts import SoundPlayer
from .calibration import Calibration
from .camera import Camera
from .config import Config
from .landmarks import FaceLandmarkExtractor
from .monitor import Monitor
from .overlay import draw_hud
from .pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-sound", action="store_true")
    parser.add_argument("--voice", action="store_true", help="macOS: speak the critical alert")
    parser.add_argument("--calibration-seconds", type=float, default=20.0)
    parser.add_argument("--calibration-file", type=str, default=None,
                        help="load this calibration if it exists, otherwise calibrate and save it")
    args = parser.parse_args()

    cfg = Config(calibration_s=args.calibration_seconds)
    cal_path = Path(args.calibration_file).expanduser() if args.calibration_file else None
    saved = Calibration.load(cal_path) if cal_path is not None and cal_path.exists() else None

    camera = Camera(args.camera)
    if not camera.open():
        raise SystemExit("Could not open the camera. Check macOS camera permission "
                         "for your terminal app, or try --camera 1.")
    monitor = Monitor(cfg, calibration=saved)
    player = SoundPlayer(enabled=not args.no_sound, voice=args.voice)

    with FaceLandmarkExtractor() as extractor:
        pipe = Pipeline(monitor, camera, extractor, cfg)
        while True:
            step = pipe.step()
            state = step.state
            if step.new:
                if state.just_calibrated and cal_path is not None and state.calibration:
                    state.calibration.save(cal_path)
                if state.alert and state.alert.sound:
                    player.play(state.alert.sound)

            draw_hud(step.view, state, state.result.t, pipe.proc_fps or 0.0)
            cv2.imshow("Driver Assist (q quit, r recalibrate)", step.view)
            key = cv2.waitKey(1 if step.camera_ok else 50) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                monitor.recalibrate()

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
