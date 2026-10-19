"""All tunable numbers live here so nothing is hard-coded in the logic."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- Default thresholds (Phase 2 calibration replaces these per driver) ---
    ear_closed_threshold: float = 0.21   # eye counted as closed below this EAR
    mar_yawn_threshold: float = 0.60     # mouth counted as open above this MAR
    pitch_nod_threshold_deg: float = 15.0  # head counted as nodding down past this

    # --- Durations (seconds) ---
    perclos_window_s: float = 60.0       # PERCLOS rolling window
    perclos_max_gap_s: float = 0.5       # ignore longer gaps between frames
    blink_window_s: float = 60.0
    min_blink_s: float = 0.05            # shorter closures are noise
    max_blink_s: float = 0.50            # longer closures are not blinks
    blink_rate_min_span_s: float = 10.0  # no blink rate until this much data
    yawn_min_s: float = 1.5              # mouth open this long = yawn
    yawn_window_s: float = 300.0
    nod_min_s: float = 1.0               # head down this long = nod
    nod_window_s: float = 300.0
    face_lost_reset_s: float = 0.5       # face missing this long resets episodes

    # --- Calibration (first N seconds with the driver alert) ---
    calibration_s: float = 20.0
    calibration_warmup_s: float = 2.0      # ignore the first seconds while the driver settles
    calibration_min_samples: int = 60
    calibration_min_face_ratio: float = 0.6
    calibration_min_ear: float = 0.12      # median EAR below this = eyes closed / bad view
    calibration_max_ear: float = 0.45
    calibration_max_mar: float = 0.35      # median MAR above this = mouth open
    calibration_max_pitch_std: float = 10.0  # degrees; more = head moving too much
    ear_closed_ratio: float = 0.75         # closed threshold = ratio * driver's open-eye EAR
    ear_threshold_min: float = 0.10
    ear_threshold_max: float = 0.25
    mar_margin: float = 0.45               # yawn threshold = resting MAR + margin
    mar_threshold_min: float = 0.50
    mar_threshold_max: float = 0.80

    # --- Risk score (0..100) ---
    closure_start_s: float = 0.4           # eyes closed this long starts to count
    closure_full_s: float = 1.5            # ...and this long is maximum risk
    perclos_low: float = 0.08
    perclos_high: float = 0.30
    perclos_min_span_s: float = 15.0       # ignore PERCLOS until the window has this much data
    yawns_full: int = 3                    # yawns per window for full yawn risk
    nods_full: int = 2
    nod_acute_level: float = 0.8           # a sustained nod alone is at least this risky (0..1)
    blink_dev_start: float = 0.5           # relative change from baseline blink rate
    blink_dev_full: float = 1.5
    w_perclos: float = 0.40
    w_yawn: float = 0.20
    w_nod: float = 0.25
    w_blink: float = 0.15
    risk_tau_up_s: float = 0.25            # score rises fast...
    risk_tau_down_s: float = 3.0           # ...and falls slowly

    # --- Alert levels ---
    caution_at: float = 25.0
    warning_at: float = 50.0
    critical_at: float = 75.0
    alert_hysteresis: float = 8.0          # must fall this far below a threshold to step down
    alert_exit_hold_s: float = 1.5
    warning_repeat_s: float = 3.0
    critical_repeat_s: float = 1.2

    # --- Robustness (Phase 3) ---
    no_face_warn_s: float = 3.0            # face missing this long = WARNING (head may have dropped)
    no_face_critical_s: float = 6.0        # ...and this long = CRITICAL
    camera_lost_critical_s: float = 2.0
    low_light_brightness: float = 50.0     # mean gray (0..255) below this = too dark
    low_light_enhance_below: float = 70.0  # auto-brighten the frame below this scene brightness
    night_mode: str = "auto"               # "auto", "on" or "off" (see NightEnhancer)
    night_brightness_below: float = 45.0   # auto night mode switches on below this scene brightness
    night_clahe_clip: float = 3.0          # local contrast boost; higher = stronger, but noisier
    night_blend: float = 0.3               # share of the previous frame mixed in to calm sensor noise
    glare_fraction: float = 0.12           # saturated share of the eye region = glare
    blur_sharpness_min: float = 15.0       # Laplacian variance of the face crop; below = blurry
    face_min_width_ratio: float = 0.12     # face narrower than this share of the frame = too far
    looking_away_yaw_deg: float = 35.0
    looking_away_s: float = 2.0
    issue_persist_s: float = 1.0           # an issue must last this long before it is reported
    issue_clear_s: float = 1.0             # ...and be gone this long before it clears
    min_fps: float = 12.0                  # below this, report LOW_FPS

    # --- Phone and talking detection (Phase 6) ---
    detect_phone: bool = True
    hand_every_n: int = 2                  # run the hand detector on every Nth processed frame
    phone_ear_radius: float = 0.60         # hand centre within this many face widths of an ear = "at the ear"
    phone_lateral_min: float = 0.40        # ...and at least this far to the side of the face centre
    phone_pose_s: float = 2.0              # hand at the ear this long = phone call (a quick scratch is ignored)
    phone_clear_s: float = 1.0             # hand away this long = call over
    phone_alert_s: float = 5.0             # warn after this long on the phone (0 turns the warning off)
    talk_window_s: float = 3.0             # mouth movement is judged over this window
    talk_step: float = 0.03                # a MAR swing smaller than this is noise
    talk_min_reversals: int = 4            # this many open/close swings in the window = talking
    phone_face_memory_s: float = 8.0       # keep using the last seen face outline this long when a hand hides the face
    hand_hides_face_max_s: float = 8.0     # a hand over the face excuses "face not visible" for this long at most
    phone_listen_s: float = 8.0            # no talking: a hand at the ear needs this long to count as a call
    face_flicker_window_s: float = 3.0     # a face that keeps dropping in and out within this window...
    face_flicker_fraction: float = 0.5     # ...and is missing at least this share of it counts as lost

