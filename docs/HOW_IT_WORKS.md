# How it works

```
camera -> MediaPipe Face Mesh (478 landmarks) -> features (EAR, MAR, head pose)
       -> rolling metrics (PERCLOS, blinks, yawns, nods) -> calibration baselines
       -> risk score (0..100) -> alert state machine -> sound + on-screen alert
```

## Signals

| Signal | Definition | Drowsy when |
|---|---|---|
| EAR (eye aspect ratio) | `(|p2-p6| + |p3-p5|) / (2 |p1-p4|)` over 6 eye landmarks | below the driver's closed threshold |
| PERCLOS | share of time the eyes were closed in the last 60 s (time-weighted) | rises above about 8 to 30 percent |
| Blink rate | closures of 0.05 to 0.5 s per minute | changes a lot from the driver's baseline |
| MAR (mouth aspect ratio) | mean inner-lip opening / mouth width | above threshold for 1.5 s = yawn |
| Head pitch | `solvePnP` on 6 face points against a generic 3D face model | head down past 15 degrees for 1 s = nod |

## Calibration

Generic thresholds fail because eye shape, glasses, lighting and camera angle differ.
During the first 20 s (first 2 s ignored) the driver looks at the road with eyes open and mouth closed:

- closed-eye threshold = 0.75 x median EAR (clamped 0.10 to 0.25)
- yawn threshold = median MAR + 0.45 (clamped 0.5 to 0.8)
- pitch baseline = median pitch, subtracted from live pitch
- blink baseline = blinks/min when at least 3 blinks were seen

Calibration is rejected, with a reason, if the eyes look closed, the mouth is open,
the head moves too much, or the face is rarely visible. It then restarts automatically.

## Risk score

`raw = max(trend, acute)`

- trend = 0.40 PERCLOS + 0.20 yawns + 0.25 nods + 0.15 blink-rate change (each 0..1, x100)
- acute = eyes closed right now (ramp from 0.4 s to 1.5 s), or head sustained down

The score is smoothed with a fast rise (tau 0.25 s) and slow fall (tau 3 s), so alarms
come quickly but do not flicker off the moment the eyes reopen.

## Alerts

Levels enter at 25 (CAUTION), 50 (WARNING), 75 (CRITICAL). Going up is immediate. Going down needs
the score to fall 8 points below the level's threshold and stay there for 1.5 s (hysteresis).
WARNING beeps every 3 s, CRITICAL every 1.2 s. With eyes fully closed, WARNING appears after
about 1.2 s and CRITICAL after about 1.5 s.

## Robustness

Bad data is worse than no data, so the system checks that it can see the driver before it trusts a number.

- **Image quality** (`quality.py`): brightness of the face box, sharpness (variance of the Laplacian),
  glare (share of saturated pixels around the eyes) and face size relative to the frame.
- **Gating**: with glare, blur, a distant face or a strongly turned head (over 35 degrees), the eye, mouth and head numbers
  are still shown but are not added to PERCLOS, blinks, yawns or nods, and open episodes are dropped.
  The risk score is held instead of changing.
- **Low light** does not block measuring. The frame is gamma-brightened before face detection and a hint is shown.
- **Debouncing** (`health.py`): an issue must last 1 s to appear and be gone for 1 s to clear, so it does not flicker.
- **Face lost**: after 3 s a WARNING, after 6 s a CRITICAL. A driver whose head drops below the camera disappears from view,
  so "face missing" is treated as a possible sign of drowsiness, not only a technical fault.
- **Camera failure** (`camera.py`): a failed read never raises. The device is released and reopened every second; after 2 s
  without frames the alert is CRITICAL, because monitoring is blind.
- **Frame skipping** (`perf.py`): if one frame takes longer than the camera interval, only every 2nd or 3rd frame is processed.
  All timers use timestamps, so PERCLOS, blink and yawn timing stay correct at a lower rate.
- **Health alerts and drowsiness alerts share one state machine**: health raises the score floor to the WARNING or
  CRITICAL entry level, so hysteresis, sound repeat and on-screen display work the same way.

## Self-test (`python -m drowsiness.qa`)

Guided scenarios: baseline, talking, mirror glances, yawning, eyes closed, head drop. "Quiet" scenarios must not raise a
WARNING, "alarm" scenarios must. Run it with and without glasses, in different light and with different people; results
(numbers only) are appended to `qa_results.md` together with your tag.

## Testing

Everything after landmark extraction runs on plain arrays, so tests build synthetic faces with known
EAR, MAR and head pose (no camera, no MediaPipe) and use independent physical checks for sign
conventions (for example, nodding down shrinks the eye-to-nose gap).
