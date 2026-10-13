# Driver Assist

Real-time driver drowsiness detection from a webcam: eye closure (EAR) and PERCLOS,
yawning (MAR), blink rate, head nod angle, and a live risk score auto-calibrated per driver.

> **Learning project, not a certified safety device.** Glasses, low light and camera
> angle affect accuracy. Never rely on it to stay safe on the road.

**Status:** Phases 1 to 5 done (detection engine, calibration, risk score, alerts, robustness, dashboard and logging, packaging). Phase 6 stretch goals in progress: software night mode, a learned risk model, phone and talking detection and video file analysis are in.
See `docs/PROGRESS.md` for the roadmap and `docs/HOW_IT_WORKS.md` for the algorithms.

## Quick start (macOS)

MediaPipe 0.10.21 needs Python 3.9 to 3.12 (3.11 recommended).

```bash
./run.sh                          # first run sets everything up, then starts the live monitor
./run.sh dashboard                # the dashboard instead
```

Or set it up by hand:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                         # 110 tests, no camera needed
driver-assist live                # calibrate (20 s), then monitor. q quit, r recalibrate
driver-assist dashboard           # review saved sessions or monitor live in the browser
driver-assist export --help       # events, timeline or sessions to CSV
driver-assist qa --tag "glasses, daylight"   # guided self-test, writes qa_results.md
driver-assist preview             # raw metrics overlay, no alerts
```

The older `python -m drowsiness.live` style commands still work too.

First run: allow camera access for your terminal app in
System Settings > Privacy & Security > Camera.

Useful flags for `drowsiness.live`: `--no-sound`, `--voice` (macOS speaks the critical alert),
`--calibration-seconds 10`, `--calibration-file ~/.driver_assist/me.json` (reuse a saved calibration),
`--no-log` (do not save the session), `--db path.db` (use another database file).

Every run of `drowsiness.live` or the dashboard's Live page is saved as a session in a local SQLite file
(`~/.driver_assist/driver_assist.db`). Nothing leaves your computer, and no video is stored, only numbers
and events. The dashboard's Sessions page shows the risk timeline, time spent at each alert level, the
event log, and CSV downloads.

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
  storage.py      SQLite sessions, events, risk timeline, CSV export
  recorder.py     turns monitor state into stored events
  export.py       python -m drowsiness.export
  cli.py          the driver-assist command (live, dashboard, export, qa, preview)
  dashboard.py    Streamlit dashboard (Sessions and Live pages)
  text.py         anti-aliased on-screen text that adapts to the background (bundled Inter font, OFL licence)
  video.py        analyse a recorded video file
  activity.py     driving, talking or phone call from hand and mouth positions
  mlmodel.py      logistic-regression risk model, metrics, leave-one-session-out test
  collect.py      record labelled examples      train.py  train and report
  overlay.py      on-screen display
  live.py         live app     preview.py  metrics-only preview     qa.py  guided self-test
app/dashboard.py  launcher for `streamlit run app/dashboard.py` from a checkout
Dockerfile, run.sh, .github/workflows/ci.yml
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

## Low light and night mode

When the room gets dark the picture is enhanced before the face is detected: brightness is lifted, local contrast is
boosted so the eyes and mouth stand out, and a little of the previous frame is blended in to calm sensor noise.
It switches on by itself when the scene is dark, and you can control it:

```bash
driver-assist live --night auto        # default: on only when the scene is dark
driver-assist live --night on          # always on, for testing
driver-assist live --screen-light      # white canvas around the video, so your screen lights your face
driver-assist qa --night on --tag "dim room, no glasses"
```

A normal webcam is not an infrared camera, so this helps in a dim room but cannot see in real darkness. Calibrate in the
same light you will drive in, for example `--calibration-file ~/.driver_assist/night.json`.

## Analysing a recorded video

No camera needed. The detector reads the clip at its own frame rate, so a two minute clip counts as two minutes.

```bash
driver-assist video drive.mp4                          # prints a summary and saves it as a session
driver-assist video drive.mp4 --out annotated.mp4      # also writes a copy with the on-screen display
driver-assist video drive.mp4 --stride 2 --no-phone    # faster: every 2nd frame, no hand detection
```

The first seconds of the clip are used to calibrate (8 by default, change it with `--calibration-seconds`), so the driver
should look at the road with eyes open at the start. The dashboard has the same thing on its Video page: upload a clip and
get the risk timeline, the event log and the frame with the highest risk.

## Free hosted demo

The dashboard can be hosted for free on Streamlit Community Cloud, so there is a link to share. A hosted copy cannot use
your camera, so it is meant for the Video page.

1. Push the repository to GitHub (the files `requirements.txt`, `packages.txt` and `.streamlit/config.toml` are already there).
2. Go to share.streamlit.io, sign in with GitHub and choose New app.
3. Pick this repository, the main branch, and `app/dashboard.py` as the main file.
4. Under advanced settings choose Python 3.11, then deploy.

Things to know: free apps go to sleep when nobody uses them and wake up slowly, uploads are limited to 100 MB, and only the
first five minutes of a clip are analysed. Uploaded clips are processed on the hosting server, so say that on the page and do
not use footage of other people without their permission. The free tier has limited memory, so if the app fails to start,
check the logs first.

## Phone and talking detection

The live view shows what the driver is doing: Driving, Talking or On a phone call. A webcam cannot hear a call, so
the phone check looks for the phone-to-ear pose: a hand beside the face at ear height, held for about two seconds.
Mouth movement (the mouth opening and closing several times in three seconds) separates talking from quiet driving,
and a yawn does not count because the mouth opens once and stays open. After five seconds with a hand at the ear the
alert level rises to WARNING and the screen asks for both hands on the wheel.

```bash
driver-assist live --no-phone                    # turn the whole thing off
driver-assist live --phone-alert-seconds 0       # keep the label, never warn
driver-assist qa                                 # includes "Touch your face" (must stay quiet) and "Phone call"
```

When a phone or hand covers the face, the detector keeps using the last seen head outline for up to 8 seconds, so the
pose is still recognised, and the "face not visible" alarm is held back for the same time because the driver is clearly
still there. After 8 seconds without a face the usual alarm applies.

Limits worth knowing: this is a rule based on hand and face position, not a trained phone detector. Earbuds and
hands-free calls are invisible to it, so you will only see "Talking". A hand held at the ear for another reason can look
the same, and a hand outside the camera's view cannot be seen at all. Phone calls are also counted in the dashboard.

## Learned risk model

The default risk score uses hand-tuned weights. You can train a small model on your own recordings and compare it with
the rules on the same data.

```bash
driver-assist collect --label alert --seconds 60     # normal driving behaviour, a few sessions
driver-assist collect --label drowsy --seconds 45    # ACTED drowsiness: slow blinks, droopy eyes, yawns, nods
driver-assist train                                  # fits the model, prints a report, writes ml_report.md
driver-assist live --risk blend                      # or --risk ml
```

Record at least two sessions of each label, ideally in different light. Training tests the model only on sessions it
never saw (leave-one-session-out), and prints the rules' score on the same rows next to it, so you can see whether the
learning actually helps. The model is logistic regression written in numpy, so the whole thing is a small JSON file.
The "eyes closed right now" alarm stays active in every mode as a safety net.

Be clear about the limits: the drowsy examples are acted, from one person, so the accuracy describes that data and not real
drowsy driving. Never record drowsy examples while driving.

## Docker

```bash
docker build -t driver-assist .
docker run --rm -p 8501:8501 -v driver-assist-data:/data driver-assist
```

Open http://localhost:8501. The container serves the dashboard and keeps sessions in the `driver-assist-data` volume.
It cannot use the webcam on a Mac or Windows machine, because Docker Desktop does not pass cameras through, so run
`driver-assist live` natively and the dashboard reads the same data.

## Tests and CI

`pytest -q` runs everything without a camera, because the tests build synthetic faces with known EAR, MAR and head pose.
GitHub Actions runs `ruff` and `pytest` on Python 3.10, 3.11 and 3.12 for every push and pull request.

## Notes

- Feed the engine the raw camera frame. Mirror only the image you display.
- Alerts are free and offline: macOS built-in sounds via `afplay`, terminal bell elsewhere.
