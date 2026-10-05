# Driver Assist: progress log

Plan: build phases 1 to 6 one at a time, finishing each before the next.
Constraint: zero budget, free tools and free hosting only. Dev machine: MacBook Air, built-in webcam.

| Phase | Scope | Status |
|---|---|---|
| 1 | Core engine: landmarks, EAR/MAR/head pose, PERCLOS, blink, yawn, nod, tests | Done, verified live on the Mac (about 31 FPS) |
| 2 | Calibration, smoothed risk score, alert state machine, sound + on-screen alert | Done, verified live on the Mac |
| 3 | Robustness: no face, camera loss, low light, FPS, glasses/angle tests | Done in code (100 tests pass); live Mac check and QA runs pending |
| 4 | Streamlit dashboard, SQLite logging, CSV export | Todo |
| 5 | Packaging: CLI entry point, Docker, README demo, GitHub Actions CI | Todo |
| 6 | Stretch: ML classifier, Raspberry Pi, night/IR mode, phone alerts, free hosted demo | Todo |

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

## Next
- Run `python -m drowsiness.live` and check the hints (cover the camera, turn your head, bring a lamp or phone light near your glasses).
- Run `python -m drowsiness.qa --tag "<conditions>"` a few times: with and without glasses, bright and dim light.
- Then Phase 4: Streamlit dashboard, SQLite event log, CSV export.
