"""Live webcam preview to sanity-check the core engine on your own face.

    python -m drowsiness.preview            # default camera
    python -m drowsiness.preview --camera 1

Press q (or Esc) in the window to quit. On macOS the first run asks for camera
permission for your terminal app: System Settings > Privacy & Security > Camera.
"""

from __future__ import annotations

import argparse
import time

import cv2

from .engine import DrowsinessEngine
from .landmarks import FaceLandmarkExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Check macOS camera permission "
                         "for your terminal app, or try --camera 1.")
    engine = DrowsinessEngine()
    t0 = time.monotonic()
    fps, last = 0.0, t0

    with FaceLandmarkExtractor() as extractor:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.monotonic()
            h, w = frame.shape[:2]
            res = engine.update(extractor.extract(frame), now - t0, (w, h))
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now

            view = cv2.flip(frame, 1)  # mirror only what we display
            if res.face_found:
                lines = [
                    f"EAR {res.ear:.2f}   MAR {res.mar:.2f}",
                    f"Pitch {res.pitch:+.0f}  Yaw {res.yaw:+.0f}  Roll {res.roll:+.0f}",
                    f"PERCLOS {res.perclos * 100:.0f}%   Blinks/min {res.blink_rate:.0f}",
                    f"Yawns {res.yawn_count}   Nods {res.nod_count}",
                    f"Eyes closed {res.eye_closed_s:.1f}s" if res.eyes_closed else "Eyes open",
                ]
            else:
                lines = ["No face detected"]
            lines.append(f"{fps:.0f} FPS")
            for i, text in enumerate(lines):
                cv2.putText(view, text, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Drowsiness preview (q to quit)", view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
