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

## Night mode and on-screen text

`NightEnhancer` in `quality.py` runs before face detection when the scene brightness is below 45 (or when forced with
`--night on`). It applies a gamma lift, CLAHE on the lightness channel, and a 30 percent blend with the previous frame.
The blend is small on purpose, because heavy averaging smears blinks. Image quality checks still use the raw frame, so the
low light hint reports the real brightness.

On-screen text is drawn by `text.py` with the bundled Inter font through Pillow. For each piece of text it measures the
luminance of the pixels underneath and picks light ink over dark areas and dark ink over light areas, using the WCAG
contrast ratio. Coloured labels (alert levels) keep their hue but are lightened or darkened until the contrast is at least
4.5. A soft halo in the opposite tone keeps text readable over busy backgrounds. Glyph masks are cached, which keeps all the
text on a frame to about 6 ms. If Pillow or the font is missing it falls back to OpenCV text.

## Phone and talking detection

MediaPipe Hands finds up to two hands every second frame (the result is reused in between). `activity.py` takes the
palm centre of each hand and checks three things against the face: it is within 0.6 face widths of an ear contour
point (Face Mesh 234 or 454), it is at least 0.4 face widths to the side of the face centre, and it is at head height
(between the forehead and a little below the chin). All three at once is the phone pose. The pose must be held for 2 s
to count as a call and absent for 1 s to end one, so a quick scratch or a one-frame detection miss does nothing.

The head outline (ear points, forehead, chin) is remembered for 8 s. A phone held to the ear often hides the face from
Face Mesh, and without the outline the ear position would vanish exactly when it is needed. If the face is missing and a
hand sits inside the remembered face box, the "face not visible" issue is held back for up to 8 s, because a hand over
the face is not a driver who has left the seat.

Talking is counted from the mouth aspect ratio over a 3 s window: at least four open/close swings of 0.03 or more.
A yawn is one long opening, so it never reaches four swings. A phone call outranks plain talking. After 5 s on a call the
risk score is raised to the WARNING level, using the same floor mechanism as the picture and camera health alerts.

## Learned risk model

`mlmodel.py` holds a logistic regression written in numpy. Its eight inputs are eye openness relative to the driver's own
calibrated baseline, mouth opening, absolute head pitch, PERCLOS, current closure time, blink rate, and yawn and nod counts.
Inputs are standardised, classes are weighted to balance, and a small L2 penalty keeps the weights modest. The output
probability is scaled to 0 to 100 and smoothed with the same fast rise and slow fall as the rule score.

Evaluation is leave-one-session-out: each recording session is held out in turn and predicted by a model trained on the
others. A random split of rows would leak neighbouring frames into the test set and overstate accuracy. The rules are scored
on the same rows (WARNING or higher counts as drowsy) as the baseline to beat. In `blend` mode the final score is the
average of rules and model, and in every mode the closure alarm (eyes closed for 0.4 to 1.5 s and more) is kept as a floor,
so a wrong model prediction cannot silence the most important alert.

## Logging and dashboard

`recorder.py` watches the monitor's state and writes to SQLite (`storage.py`). Three tables: sessions, events and samples.
An event is written when the alert level changes, when a yawn or nod is counted, when a picture or camera issue appears,
and once when calibration finishes. A sample (risk, level, EAR, MAR, PERCLOS, blink rate) is written about once a second,
which is enough to draw the risk timeline without growing the file quickly. Values are rounded and NaN is stored as NULL.

The detection code never imports the database, and the database module only uses the standard library. The Streamlit
dashboard in `app/dashboard.py` reads the same file, so the CLI and the dashboard can be used interchangeably.

## Self-test (`python -m drowsiness.qa`)

Guided scenarios: baseline, talking, mirror glances, yawning, eyes closed, head drop. "Quiet" scenarios must not raise a
WARNING, "alarm" scenarios must. Run it with and without glasses, in different light and with different people; results
(numbers only) are appended to `qa_results.md` together with your tag.

## Testing

Everything after landmark extraction runs on plain arrays, so tests build synthetic faces with known
EAR, MAR and head pose (no camera, no MediaPipe) and use independent physical checks for sign
conventions (for example, nodding down shrinks the eye-to-nose gap).
