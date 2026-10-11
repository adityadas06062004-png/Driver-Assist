"""Live monitor: calibrate, then watch the driver and alert.

    python -m drowsiness.live
    python -m drowsiness.live --no-sound --calibration-file ~/.driver_assist/me.json

Each run is saved as a session in a local SQLite file (default ~/.driver_assist/driver_assist.db).
View it with:  streamlit run app/dashboard.py     Export it with:  python -m drowsiness.export

Keys: q or Esc = quit, r = recalibrate.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import cv2

from .alerts import SoundPlayer
from .calibration import Calibration
from .camera import Camera
from .config import Config
from .landmarks import FaceLandmarkExtractor, HandLandmarkExtractor
from .mlmodel import DEFAULT_MODEL, LogisticModel
from .monitor import Monitor
from .overlay import draw_hud, screen_light
from .pipeline import Pipeline
from .recorder import SessionRecorder
from .storage import DEFAULT_DB, Storage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-sound", action="store_true")
    parser.add_argument("--voice", action="store_true", help="macOS: speak the critical alert")
    parser.add_argument("--risk", choices=("rules", "ml", "blend"), default="rules",
                        help="how the risk score is made: hand-tuned rules, the learned model, or both averaged")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="trained model file for --risk ml or blend")
    parser.add_argument("--no-phone", action="store_true", help="turn off phone and talking detection")
    parser.add_argument("--phone-alert-seconds", type=float, default=5.0,
                        help="warn after this long with a hand at the ear (0 = never warn)")
    parser.add_argument("--night", choices=("auto", "on", "off"), default="auto",
                        help="software night mode for dim rooms (default: switches on when the scene is dark)")
    parser.add_argument("--screen-light", action="store_true",
                        help="show the picture on a bright white canvas so the screen lights your face")
    parser.add_argument("--no-log", action="store_true", help="do not save events to the database")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite file for the event log")
    parser.add_argument("--calibration-seconds", type=float, default=20.0)
    parser.add_argument("--calibration-file", type=str, default=None,
                        help="load this calibration if it exists, otherwise calibrate and save it")
    args = parser.parse_args()

    cfg = Config(calibration_s=args.calibration_seconds, night_mode=args.night,
                 detect_phone=not args.no_phone, phone_alert_s=args.phone_alert_seconds)
    cal_path = Path(args.calibration_file).expanduser() if args.calibration_file else None
    saved = Calibration.load(cal_path) if cal_path is not None and cal_path.exists() else None

    model = None
    if args.risk != "rules":
        model_path = Path(args.model).expanduser()
        if not model_path.exists():
            raise SystemExit(f"No model at {model_path}. Record data with `driver-assist collect` "
                             "and then run `driver-assist train`.")
        model = LogisticModel.load(model_path)

    camera = Camera(args.camera)
    if not camera.open():
        raise SystemExit("Could not open the camera. Check macOS camera permission "
                         "for your terminal app, or try --camera 1.")
    monitor = Monitor(cfg, calibration=saved, model=model, risk_mode=args.risk)
    player = SoundPlayer(enabled=not args.no_sound, voice=args.voice)
    storage = None if args.no_log else Storage(args.db)
    recorder = SessionRecorder(storage, camera=args.camera) if storage else None

    hands_ctx = HandLandmarkExtractor() if cfg.detect_phone else nullcontext()
    with FaceLandmarkExtractor() as extractor, hands_ctx as hands:
        pipe = Pipeline(monitor, camera, extractor, cfg, hands=hands)
        while True:
            step = pipe.step()
            state = step.state
            if step.new:
                if state.just_calibrated and cal_path is not None and state.calibration:
                    state.calibration.save(cal_path)
                if state.alert and state.alert.sound:
                    player.play(state.alert.sound)
                if recorder:
                    recorder.update(state)

            draw_hud(step.view, state, state.result.t, pipe.proc_fps or 0.0)
            shown = screen_light(step.view) if args.screen_light else step.view
            cv2.imshow("Driver Assist (q quit, r recalibrate)", shown)
            key = cv2.waitKey(1 if step.camera_ok else 50) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                monitor.recalibrate()

    camera.release()
    cv2.destroyAllWindows()
    if recorder:
        recorder.close()
        print(f"Session {recorder.session_id} saved to {storage.path}")
        storage.close()


if __name__ == "__main__":
    main()
