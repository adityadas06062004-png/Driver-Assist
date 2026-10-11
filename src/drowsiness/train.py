"""Train the learned risk model from the recordings made with ``driver-assist collect``.

    driver-assist train
    driver-assist train --data ~/.driver_assist/ml_samples.csv --out ~/.driver_assist/model.json

Reports how well the model does on recording sessions it never saw (leave-one-session-out),
next to the hand-tuned rules on the same rows, so you can see whether learning helps at all.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .collect import DEFAULT_DATA
from .mlmodel import (
    DEFAULT_MODEL,
    FEATURES,
    LogisticModel,
    binary_metrics,
    leave_one_session_out,
    recall_at_false_alarms,
    roc_auc,
)

RULE_THRESHOLD = 50.0  # the rules call "WARNING" at this score


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    X, y, groups, rules = [], [], [], []
    with open(path.expanduser(), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            X.append([float(r[f]) for f in FEATURES])
            y.append(1 if r["label"] == "drowsy" else 0)
            groups.append(r["session"])
            rules.append(float(r["rule_risk"]))
    if not X:
        raise ValueError("the data file has no rows")
    return np.array(X), np.array(y), groups, np.array(rules)


def sessions_per_class(y: np.ndarray, groups: List[str]) -> Dict[int, int]:
    return {c: len({g for g, lab in zip(groups, y) if lab == c}) for c in (0, 1)}


def fmt(m: Dict[str, float]) -> str:
    return (f"accuracy {m['accuracy'] * 100:.0f}%, recall {m['recall'] * 100:.0f}%, "
            f"precision {m['precision'] * 100:.0f}%, false alarms {m['false_alarm_rate'] * 100:.0f}%")


def build_report(X, y, groups, rules, model: LogisticModel, cv_proba=None) -> str:
    counts = sessions_per_class(y, groups)
    lines = [f"## Model training {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"Rows: {len(y)} ({int((y == 0).sum())} alert, {int((y == 1).sum())} drowsy). "
             f"Sessions: {counts[0]} alert, {counts[1]} drowsy.", ""]
    base = binary_metrics(y, (rules >= RULE_THRESHOLD).astype(int))
    lines.append(f"Hand-tuned rules (WARNING or higher counts as drowsy): {fmt(base)}.")
    if cv_proba is not None:
        cv_pred = (cv_proba >= 0.5).astype(int)
        lines.append(f"Learned model on unseen sessions: {fmt(binary_metrics(y, cv_pred))}.")
        lines += ["", "Threshold-free comparison (fairer, because the two scores use different scales):",
                  f"ROC AUC, 0.5 is guessing and 1.0 is perfect: rules {roc_auc(y, rules):.2f}, "
                  f"learned {roc_auc(y, cv_proba):.2f}.",
                  f"Drowsy rows caught when at most 5% of alert rows may trigger: rules "
                  f"{recall_at_false_alarms(y, rules) * 100:.0f}%, "
                  f"learned {recall_at_false_alarms(y, cv_proba) * 100:.0f}%.",
                  "", "Per session, share of rows flagged (rules at 50+, learned at 0.5+):", ""]
        for g in sorted(set(groups)):
            idx = [i for i, grp in enumerate(groups) if grp == g]
            label = "drowsy" if y[idx[0]] else "alert"
            lines.append(f"{g} ({label}, {len(idx)} rows): rules {(rules[idx] >= RULE_THRESHOLD).mean() * 100:.0f}%, "
                         f"learned {cv_pred[idx].mean() * 100:.0f}%")
    else:
        lines.append("Learned model on unseen sessions: not measured, record at least 2 sessions of each label.")
    lines += ["", "Inputs by influence (standardised weight, positive means more drowsy):", ""]
    lines += [f"{i}. {name}: {w:+.2f}" for i, (name, w) in enumerate(model.importance(), 1)]
    lines += ["", "Check the signs: more blinking, yawning or head nodding should push the score toward drowsy. "
              "A negative weight there usually means the model learned your acting style, not drowsiness.", "",
              "The drowsy examples are acted, from a small number of sessions. These numbers describe "
              "this data, not real-world drowsiness.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(DEFAULT_MODEL))
    parser.add_argument("--report", default="ml_report.md")
    args = parser.parse_args()

    path = Path(args.data).expanduser()
    if not path.exists():
        raise SystemExit(f"No data at {path}. Record some first:  driver-assist collect --label alert")
    X, y, groups, rules = load_dataset(path)
    if len(set(y.tolist())) < 2:
        raise SystemExit("Need both labels. Record some 'alert' and some 'drowsy' sessions.")

    counts = sessions_per_class(y, groups)
    cv_proba = None
    if min(counts.values()) >= 2:
        cv_proba = leave_one_session_out(X, y, groups, proba=True)
    model = LogisticModel.fit(X, y)
    model.meta = {"rows": int(len(y)), "sessions": counts, "trained": time.strftime("%Y-%m-%d %H:%M")}
    model.save(args.out)

    report = build_report(X, y, groups, rules, model, cv_proba)
    with open(args.report, "a", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(report)
    print(f"Model saved to {Path(args.out).expanduser()}. Use it with:  driver-assist live --risk blend")


if __name__ == "__main__":
    main()
