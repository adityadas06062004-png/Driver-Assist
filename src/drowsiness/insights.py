"""Plain-language readings of a session: a score, a label and a few highlights.

Pure Python (no Streamlit), so the wording rules can be tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

ISSUE_ADVICE: Dict[str, Tuple[str, str]] = {
    "no_face": ("warn", "Your face left the picture. Check that the camera can see you and sits near eye level."),
    "low_light": ("info", "The picture was dark. Add some light, or run with --night on."),
    "glare": ("warn", "Glare on glasses hid the eyes. Tilt the camera or the screen a little."),
    "blurry": ("info", "The picture was blurry. Clean the lens or add light."),
    "face_too_small": ("info", "You were far from the camera. Move closer."),
    "looking_away": ("info", "You looked away from the road for a while."),
    "low_fps": ("info", "The computer struggled to keep up. Close other apps, or analyse every 2nd frame."),
    "camera_lost": ("bad", "The camera stopped sending pictures at some point."),
}
EVENT_NAMES = {
    "alert": "Alert level change", "yawn": "Yawn", "nod": "Head nod", "phone_start": "Phone call started",
    "phone_end": "Phone call ended", "issue": "Picture or camera issue", "calibrated": "Calibration finished",
}


@dataclass
class Insight:
    tone: str     # good, info, warn or bad
    title: str
    text: str


def mmss(seconds: float) -> str:
    s = int(max(seconds, 0))
    return f"{s // 60}:{s % 60:02d}"


def alertness_score(avg_risk: float) -> int:
    """100 means no sign of fatigue, 0 means the risk was at its maximum the whole time."""
    return int(round(max(0.0, min(100.0, 100.0 - avg_risk))))


def alertness_label(score: int) -> Tuple[str, str]:
    if score >= 85:
        return "Alert", "good"
    if score >= 65:
        return "Mostly alert", "good"
    if score >= 45:
        return "Signs of fatigue", "warn"
    return "High fatigue risk", "bad"


def longest_stretch(samples: Sequence[dict], threshold: float = 50.0) -> Optional[Tuple[float, float]]:
    """Start and length in seconds of the longest run of samples at or above the threshold."""
    best, start, prev = None, None, None
    for s in samples:
        risk = s.get("risk")
        if risk is not None and risk >= threshold:
            start = s["t"] if start is None else start
            prev = s["t"]
        elif start is not None:
            length = prev - start
            if best is None or length > best[1]:
                best = (start, length)
            start = None
    if start is not None:
        length = prev - start
        if best is None or length > best[1]:
            best = (start, length)
    return best


def phone_seconds(events: Sequence[dict]) -> float:
    total, started = 0.0, None
    for e in events:
        if e["kind"] == "phone_start":
            started = e["t"]
        elif e["kind"] == "phone_end" and started is not None:
            total += e["t"] - started
            started = None
    return total


def insights(summary: dict, samples: Sequence[dict], events: Sequence[dict]) -> List[Insight]:
    out: List[Insight] = []
    if not samples:
        return [Insight("info", "Not enough data",
                        "This session has no timeline, so there is nothing to summarise yet.")]

    peak = max(samples, key=lambda s: s.get("risk") or 0.0)
    if summary["max_risk"] < 25:
        out.append(Insight("good", "No signs of fatigue",
                           "The risk score never reached the caution level. Eyes, mouth and head all stayed normal."))
    else:
        around = ("yawn", "nod", "phone_start")
        near = [e["kind"] for e in events if abs(e["t"] - peak["t"]) <= 3 and e["kind"] in around]
        why = " A yawn, a head nod or a phone call happened right around then." if near else ""
        out.append(Insight("bad" if summary["max_risk"] >= 75 else "warn", "Highest risk",
                           f"Risk peaked at {summary['max_risk']:.0f} at {mmss(peak['t'])}.{why}"))

    stretch = longest_stretch(samples)
    if stretch and stretch[1] >= 1:
        out.append(Insight("warn", "Longest stretch at warning level or above",
                           f"{stretch[1]:.0f} seconds, starting at {mmss(stretch[0])}."))

    first = next((e for e in events if e["kind"] == "alert" and (e.get("level") or 0) >= 2), None)
    if first is not None:
        out.append(Insight("info", "First alarm", f"The first warning came at {mmss(first['t'])}."))

    if summary["yawns"] >= 3:
        out.append(Insight("warn", "Repeated yawning",
                           f"{summary['yawns']} yawns. Frequent yawning is an early sign of tiredness."))
    elif summary["yawns"] > 0:
        out.append(Insight("info", "Yawning", f"{summary['yawns']} yawn(s) were counted."))
    if summary["nods"] >= 2:
        out.append(Insight("warn", "Head nodding",
                           f"{summary['nods']} head nods. A head that drops forward is a strong sign of drowsiness."))

    if summary.get("phone_calls"):
        secs = phone_seconds(events)
        extra = f" for about {secs:.0f} seconds in total" if secs else ""
        out.append(Insight("warn", "Phone use", f"A hand was held at the ear {summary['phone_calls']} time(s){extra}."))

    counts: Dict[str, int] = {}
    for e in events:
        if e["kind"] == "issue":
            counts[e.get("detail", "")] = counts.get(e.get("detail", ""), 0) + 1
    for code, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        if code in ISSUE_ADVICE:
            tone, advice = ISSUE_ADVICE[code]
            out.append(Insight(tone, f"Picture issue: {code.replace('_', ' ')} ({n}x)", advice))
    return out
