"""Guided self-test: records how the detector behaves in situations you create.

    python -m drowsiness.qa --tag "glasses, daylight"

It calibrates, then walks you through scenarios (normal driving, talking, mirror
checks, yawning, eyes closed, head drop). Each is scored: quiet scenarios must not
raise a WARNING, alarm scenarios must. Results (numbers only, no images) are appended
to a Markdown file, so you can repeat it with glasses, other lighting, other people.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import cv2

from .alerts import Level
from .camera import Camera
from .config import Config
from .landmarks import FaceLandmarkExtractor
from .monitor import Monitor
from .overlay import _centered, draw_hud
from .pipeline import Pipeline

NAN = float("nan")
COUNTDOWN_S = 4.0

# (name, instruction, seconds, expectation)
SCENARIOS: List[Tuple[str, str, int, str]] = [
    ("Baseline", "Look at the road. Blink normally.", 15, "quiet"),
    ("Talking", "Talk or sing out loud. Look at the road.", 10, "quiet"),
    ("Mirror check", "Glance left and right as if checking mirrors.", 10, "quiet"),
    ("Yawn", "Yawn widely, twice.", 10, "yawn"),
    ("Eyes closed", "Close your eyes and keep them closed.", 6, "alarm"),
    ("Head drop", "Let your head drop forward and hold it down.", 6, "alarm"),
]


@dataclass
class Sample:
    t: float
    face_found: bool
    trusted: bool
    ear: float
    brightness: float
    sharpness: float
    glare: float
    level: int
    yawns: int
    issues: frozenset = field(default_factory=frozenset)


@dataclass
class ScenarioResult:
    name: str
    expect: str
    passed: bool
    detail: str
    frames: int
    detect_rate: float
    trusted_rate: float
    ear_median: float
    ear_min: float
    brightness: float
    sharpness: float
    max_level: str
    issues: str


def _clean(values: Iterable[float]) -> List[float]:
    return [v for v in values if not math.isnan(v)]


def _median(values: Iterable[float]) -> float:
    vals = _clean(values)
    return statistics.median(vals) if vals else NAN


def summarize(name: str, expect: str, samples: List[Sample]) -> ScenarioResult:
    n = len(samples)
    if n == 0:
        return ScenarioResult(name, expect, False, "no frames recorded", 0, 0, 0, NAN, NAN, NAN,
                              NAN, "OK", "")
    t0 = samples[0].t
    max_level = max(s.level for s in samples)
    first_warn = next((s.t - t0 for s in samples if s.level >= Level.WARNING), None)
    yawns = max(s.yawns for s in samples)

    if expect == "quiet":
        passed = max_level < Level.WARNING
        detail = "no alarm" if passed else f"FALSE ALARM at {first_warn:.1f}s"
    elif expect == "alarm":
        passed = first_warn is not None
        detail = (f"alarm after {first_warn:.1f}s (includes your reaction time)"
                  if passed else "NO ALARM")
    else:  # "yawn"
        passed = yawns >= 1
        detail = f"{yawns} yawn(s) counted"

    ears = _clean(s.ear for s in samples)
    issues = sorted({i for s in samples for i in s.issues})
    return ScenarioResult(
        name, expect, passed, detail, n,
        sum(s.face_found for s in samples) / n,
        sum(s.trusted for s in samples) / n,
        _median(s.ear for s in samples), min(ears) if ears else NAN,
        _median(s.brightness for s in samples), _median(s.sharpness for s in samples),
        Level(max_level).name, ", ".join(issues))


def to_markdown(results: List[ScenarioResult], tag: str, when: str) -> str:
    def f(x, spec):
        return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)

    lines = [f"## QA run {when}" + (f" ({tag})" if tag else ""), "",
             "| Scenario | Expect | Result | Detail | Face found | Trusted | EAR med/min | Bright | Sharp | Max level | Issues |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r.name} | {r.expect} | {'PASS' if r.passed else 'FAIL'} | {r.detail} | "
            f"{r.detect_rate * 100:.0f}% | {r.trusted_rate * 100:.0f}% | "
            f"{f(r.ear_median, '.2f')}/{f(r.ear_min, '.2f')} | {f(r.brightness, '.0f')} | "
            f"{f(r.sharpness, '.0f')} | {r.max_level} | {r.issues or '-'} |")
    passed = sum(r.passed for r in results)
    lines += ["", f"**{passed}/{len(results)} scenarios passed.**", ""]
    return "\n".join(lines)


def _sample(step) -> Sample:
    res, stats = step.state.result, step.stats
    return Sample(
        t=res.t, face_found=res.face_found, trusted=res.trusted, ear=res.ear,
        brightness=stats.brightness if stats else NAN,
        sharpness=stats.sharpness if stats else NAN,
        glare=stats.glare if stats else NAN,
        level=int(step.state.alert.level) if step.state.alert else 0,
        yawns=res.yawn_count,
        issues=frozenset(i.value for i in step.state.health.issues) if step.state.health else frozenset())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--tag", default="", help='conditions, e.g. "glasses, daylight"')
    parser.add_argument("--out", default="qa_results.md")
    parser.add_argument("--calibration-seconds", type=float, default=20.0)
    args = parser.parse_args()

    cfg = Config(calibration_s=args.calibration_seconds)
    camera = Camera(args.camera)
    if not camera.open():
        raise SystemExit("Could not open the camera. Check macOS camera permission.")
    monitor = Monitor(cfg)

    results: List[ScenarioResult] = []
    idx, stage, stage_start = 0, "calibrate", 0.0
    samples: List[Sample] = []
    quit_early = False

    with FaceLandmarkExtractor() as extractor:
        pipe = Pipeline(monitor, camera, extractor, cfg)
        while idx < len(SCENARIOS):
            step = pipe.step()
            now = step.state.result.t
            view = step.view

            if stage == "calibrate":
                draw_hud(view, step.state, now, pipe.proc_fps or 0.0)
                if step.state.phase == "monitoring":
                    stage, stage_start = "countdown", now
            else:
                name, instruction, seconds, expect = SCENARIOS[idx]
                draw_hud(view, step.state, now, pipe.proc_fps or 0.0)
                h = view.shape[0]
                _centered(view, f"{idx + 1}/{len(SCENARIOS)}  {name}", 40, 0.9, (255, 255, 255))
                _centered(view, instruction, 75, 0.65, (0, 215, 255))
                if stage == "countdown":
                    left = COUNTDOWN_S - (now - stage_start)
                    _centered(view, f"Get ready... {max(left, 0):.0f}", h // 2, 1.6, (0, 215, 255))
                    if left <= 0:
                        # Fresh windows for every scenario; calibration is kept.
                        monitor.engine.reset_windows()
                        monitor.scorer.reset()
                        monitor.alerts.reset()
                        stage, stage_start, samples = "record", now, []
                else:
                    left = seconds - (now - stage_start)
                    _centered(view, f"GO  {max(left, 0):.0f}s", h // 2, 1.6, (0, 255, 0))
                    if step.new:
                        samples.append(_sample(step))
                    if left <= 0:
                        results.append(summarize(name, expect, samples))
                        idx += 1
                        stage, stage_start = "countdown", now

            cv2.imshow("Driver Assist QA (q to stop)", view)
            if cv2.waitKey(1 if step.camera_ok else 50) & 0xFF in (ord("q"), 27):
                quit_early = True
                break

    camera.release()
    cv2.destroyAllWindows()
    if not results:
        print("No scenarios completed.")
        return
    report = to_markdown(results, args.tag, time.strftime("%Y-%m-%d %H:%M"))
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(report)
    print(f"Appended to {args.out}" + ("  (stopped early)" if quit_early else ""))


if __name__ == "__main__":
    main()
