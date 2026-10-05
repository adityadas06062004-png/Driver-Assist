# Driver Assist

Real-time driver drowsiness detection from a webcam: eye closure (EAR) and PERCLOS,
yawning (MAR), blink rate, head nod angle, and a live risk score auto-calibrated per driver.

> **Learning project, not a certified safety device.** Glasses, low light and camera
> angle affect accuracy. Never rely on it to stay safe on the road.

**Status:** Phases 1 to 3 done (detection engine, calibration, risk score, alerts, robustness).
See `docs/PROGRESS.md` for the roadmap and `docs/HOW_IT_WORKS.md` for the algorithms.

## Quick start (macOS)

MediaPipe 0.10.21 needs Python 3.9 to 3.12 (3.11 recommended).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                         # 100 tests, no camera needed
python -m drowsiness.live         # calibrate (20 s), then monitor. q quit, r recalibrate
python -m drowsiness.preview      # raw metrics overlay, no alerts
python -m drowsiness.qa --tag "glasses, daylight"   # guided self-test, writes qa_results.md
```

First run: allow camera access for your terminal app in
System Settings > Privacy & Security > Camera.

Useful flags for `drowsiness.live`: `--no-sound`, `--voice` (macOS speaks the critical alert),
`--calibration-seconds 10`, `--calibration-file ~/.driver_assist/me.json` (reuse a saved calibration).

## Layout

```
src/drowsiness/
  config.py       all thresholds and durations
  landmarks.py    MediaPipe Face Mesh wrapper (pixel landmarks or None)
  features.py     EAR, MAR, head pose (pitch > 0 = head down)
  metrics.py      PERCLOS, blink counter, yawn/nod event detector
  engine.py       landmarks -> per-frame metrics (FrameResult)
  calibration.py  per-driver baselines and thresholds
  risk.py         smoothed 0..100 risk score
  alerts.py       OK/CAUTION/WARNING/CRITICAL with hysteresis, sound player
  health.py       signal health: face lost, glare, blur, low light, camera lost
  quality.py      image quality measures and low-light brightening
  camera.py       camera wrapper that reopens after failures
  perf.py         frame skipping to stay real-time on slow machines
  pipeline.py     camera -> landmarks -> monitor, shared by live and qa
  monitor.py      whole pipeline: calibrate, then monitor
  overlay.py      on-screen display
  live.py         live app     preview.py  metrics-only preview     qa.py  guided self-test
tests/            synthetic-landmark tests (no camera, no MediaPipe)
```

## What happens when the picture is bad

| Situation | Behaviour |
|---|---|
| Face missing 3 s / 6 s | WARNING / CRITICAL alert (a dropped head can leave the frame) |
| Camera unplugged or read fails | Reconnects every second; CRITICAL after 2 s |
| Glare on glasses, blur, face too far, head turned | Measurements paused (no false alarms), hint shown on screen |
| Low light | Frame is auto-brightened; "Low light" hint |
| Slow computer | Processes every 2nd or 3rd frame; timing stays correct |

## Notes

- Feed the engine the raw camera frame. Mirror only the image you display.
- Alerts are free and offline: macOS built-in sounds via `afplay`, terminal bell elsewhere.
