# Driver Assist: progress log

Plan: build phases 1 to 6 one at a time, finishing each before the next.
Constraint: zero budget, free tools and free hosting only. Dev machine: MacBook Air, built-in webcam.

| Phase | Scope | Status |
|---|---|---|
| 1 | Core engine: landmarks, EAR/MAR/head pose, PERCLOS, blink, yawn, nod, tests | Done, verified live on the Mac (about 31 FPS) |
| 2 | Calibration, smoothed risk score, alert state machine, sound + on-screen alert | Done, verified live on the Mac |
| 3 | Robustness: no face, camera loss, low light, FPS, glasses/angle tests | Done in code (100 tests pass); live Mac check and QA runs pending |
| 4 | Streamlit dashboard, SQLite logging, CSV export | Done, verified live on the Mac |
| 5 | Packaging: CLI entry point, Docker, README demo, GitHub Actions CI | Done in code (110 tests pass); Docker build, CI run and demo GIF pending |
| 6 | Stretch: ML classifier, Raspberry Pi, night/IR mode, phone alerts, free hosted demo | In progress: night mode and adaptive text verified on the Mac, learned risk model tried on real recordings (it did not beat the rules), phone and talking detection fixed after the first QA run (169 tests pass); second QA run pending |

## Phase 1 notes
- Pinned `mediapipe==0.10.21` (bundles the face model, no download needed).
- Live check: about 31 FPS; with a generic threshold, EAR hovered at the closed-eye line
  (0.20 vs 0.21) and head pitch read about -17 at rest. That is why Phase 2 calibrates per driver.

## Phase 2 notes
- Calibration measures resting EAR, MAR and pitch and sets per-driver thresholds.
- Risk rise time 0.25 s, fall time 3 s. Eyes fully closed: WARNING at about 1.2 s, CRITICAL at about 1.5 s.
- Residual risk after an event comes from PERCLOS memory (the 60 s window), staying below CAUTION.

## Phase 3 notes
- Face lost: WARNING at 3 s, CRITICAL at 6 s. Camera lost: CRITICAL at 2 s.
- Glare, blur, face too far or head turned past 35 degrees pause the measurements instead of guessing.
- Thresholds for glare (12 percent saturated), sharpness (15) and face size (12 percent of frame width) were set from
  a sample portrait. Tune them from your own `qa_results.md` runs.
- Real-detector smoke test: very dark and blurred frames were paused with no false alarm; a blank frame raised the face-lost alert.

## Phase 4 notes
Sessions, events and a one-per-second risk timeline are saved to `~/.driver_assist/driver_assist.db`. The dashboard has a
Sessions page (summary numbers, risk chart, time per level, event log, CSV downloads, delete) and a Live page that runs
the same pipeline from the browser tab using the Mac camera. Export from the terminal with `python -m drowsiness.export`.
No video or images are ever stored.

## Phase 5 notes
`pip install -e .` now installs a `driver-assist` command with live, dashboard, export, qa and preview. The dashboard moved into
the package so the command can find it after installing; `app/dashboard.py` is a small launcher for source checkouts.
`run.sh` does first-time setup and starts the monitor. The Dockerfile serves the dashboard only, since containers on a Mac
cannot reach the webcam. CI runs ruff and pytest on Python 3.10 to 3.12. The Dockerfile and workflow have not been run
yet, so the first push to GitHub is the real test of CI.

## Phase 6 notes
Step 1, night mode: software only (gamma, CLAHE, light frame blend), automatic below a scene brightness of 45, plus a
`--screen-light` option. Not yet tested on real dark footage, so the dim-room QA run is the real test.
Also done: the HUD text now uses the bundled Inter font with colours that adapt to the background (light on dark,
dark on light), and Ctrl+C on the dashboard command exits quietly.
Step 2, learned risk model: `collect` records labelled one-second rows, `train` fits logistic regression and reports
leave-one-session-out results next to the rules, `live --risk ml|blend` uses it. First real run (3 alert and 3 drowsy sessions plus 4 earlier ones): at fixed thresholds the rules had 75 percent accuracy,
26 percent recall and 3 percent false alarms, the learned model 69 percent, 54 percent and 24 percent. The learned model did
not beat the rules, and its blink and yawn weights had the wrong sign, which points to learning the acting style. The report
now adds a threshold-free comparison (ROC AUC, recall at 5 percent false alarms) and a per-session breakdown.
Step 3, phone and talking detection: hand at the ear for 2 s counts as a call, mouth swings count as talking, a warning
after 5 s on a call, logged to the database and shown in the dashboard, two new QA scenarios (touching your face must
stay quiet, a phone pose must be recognised). Thresholds come from reasoning about face geometry and synthetic tests,
so the QA run on a real camera is the check that matters.
First real QA run of step 3 (8 scenarios): 6 passed. The phone pose was not recognised because the phone hid the face
(found only 55 percent of the time), and touching the face raised a face-not-visible CRITICAL. Fixed by remembering the
head outline for 8 s and excusing a hand-covered face for the same time. The QA table now names the cause of a false
alarm, after one talking scenario raised a WARNING at 2.7 s in one run and none in another.
Left in Phase 6, one at a time: Raspberry Pi, free hosted demo.

## Next
Run `driver-assist qa` again and send the table: the Phone call, Touch your face and Talking rows matter most.
Then the next Phase 6 item.
