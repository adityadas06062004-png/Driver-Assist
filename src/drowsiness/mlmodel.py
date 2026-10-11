"""A small learned alternative to the hand-tuned risk weights.

Plain logistic regression written with numpy, so there is no extra dependency and the whole
model is a handful of numbers in a JSON file. It reads the same per-frame measurements the rules use
(eye openness relative to this driver's baseline, mouth opening, head pitch, PERCLOS, how long the eyes
have been closed, blink rate, yawn and nod counts) and outputs a probability that the driver is drowsy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .engine import FrameResult

FEATURES = ("ear_ratio", "mar", "pitch_abs", "perclos", "eye_closed_s", "blink_rate",
            "yawn_count", "nod_count")
DEFAULT_MODEL = Path("~/.driver_assist/model.json")


def feature_vector(res: FrameResult, ear_baseline: Optional[float]) -> Optional[List[float]]:
    """The model's inputs for one frame, or None when the frame cannot be trusted."""
    if not res.face_found or not res.trusted or math.isnan(res.ear) or math.isnan(res.mar):
        return None
    base = ear_baseline if ear_baseline and ear_baseline > 0 else 0.3
    pitch = 0.0 if math.isnan(res.pitch) else abs(res.pitch)
    return [res.ear / base, res.mar, pitch, res.perclos, res.eye_closed_s, res.blink_rate,
            float(res.yawn_count), float(res.nod_count)]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


@dataclass
class LogisticModel:
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    bias: float
    features: Tuple[str, ...] = FEATURES
    meta: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, l2: float = 0.01, lr: float = 0.3,
            epochs: int = 1500, features: Sequence[str] = FEATURES) -> "LogisticModel":
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or len(X) != len(y) or len(set(y.tolist())) < 2:
            raise ValueError("need rows of features and both classes (0 and 1) to train")
        mean, std = X.mean(axis=0), X.std(axis=0)
        std[std < 1e-9] = 1.0
        Z = (X - mean) / std
        # Balance the classes so a long "alert" recording does not drown out the "drowsy" one.
        pos = max(float(y.sum()), 1.0)
        neg = max(float(len(y) - y.sum()), 1.0)
        sample_w = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))
        w, b = np.zeros(Z.shape[1]), 0.0
        for _ in range(epochs):
            p = _sigmoid(Z @ w + b)
            err = (p - y) * sample_w
            w -= lr * (Z.T @ err / len(y) + l2 * w)
            b -= lr * float(err.mean())
        return cls(mean, std, w, b, tuple(features))

    def predict_proba(self, x: Union[Sequence[float], np.ndarray]) -> Union[float, np.ndarray]:
        arr = np.asarray(x, dtype=np.float64)
        p = _sigmoid(((arr - self.mean) / self.std) @ self.weights + self.bias)
        return float(p) if arr.ndim == 1 else p

    def to_dict(self) -> dict:
        return {"features": list(self.features), "mean": self.mean.tolist(), "std": self.std.tolist(),
                "weights": self.weights.tolist(), "bias": self.bias, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticModel":
        return cls(np.array(d["mean"]), np.array(d["std"]), np.array(d["weights"]), float(d["bias"]),
                   tuple(d["features"]), dict(d.get("meta", {})))

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LogisticModel":
        model = cls.from_dict(json.loads(Path(path).expanduser().read_text()))
        if model.features != FEATURES:
            raise ValueError("this model was trained on different features; train it again")
        return model

    def importance(self) -> List[Tuple[str, float]]:
        """Weights of the standardised inputs, biggest influence first."""
        pairs = list(zip(self.features, self.weights.tolist()))
        return sorted(pairs, key=lambda kv: -abs(kv[1]))


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": (tp + tn) / max(len(yt), 1), "precision": precision, "recall": recall,
            "f1": f1, "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def roc_auc(y_true: Sequence[int], score: Sequence[float]) -> float:
    """Chance that a random drowsy row scores higher than a random alert row (0.5 = guessing, 1.0 = perfect).
    Independent of any threshold, so it compares two scores of different scales fairly."""
    y, sc = np.asarray(y_true), np.asarray(score, dtype=np.float64)
    pos, neg = sc[y == 1], sc[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    # average the ranks of ties
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def recall_at_false_alarms(y_true: Sequence[int], score: Sequence[float], max_fpr: float = 0.05) -> float:
    """Share of drowsy rows caught when the alarm threshold is set so at most ``max_fpr`` of alert rows trigger."""
    y, sc = np.asarray(y_true), np.asarray(score, dtype=np.float64)
    neg, pos = sc[y == 0], sc[y == 1]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    cutoff = np.quantile(neg, 1.0 - max_fpr)
    return float((pos > cutoff).mean())


def leave_one_session_out(X: np.ndarray, y: np.ndarray, groups: Sequence[str],
                          threshold: float = 0.5, proba: bool = False, **fit_kw) -> np.ndarray:
    """Predictions for every row from a model that never saw that row's recording session.
    This is the honest test: random row splits would leak neighbouring frames into the test set."""
    groups = np.asarray(groups)
    pred = np.zeros(len(y), dtype=float if proba else int)
    for g in sorted(set(groups.tolist())):
        test = groups == g
        train = ~test
        if len(set(y[train].tolist())) < 2:
            raise ValueError("every class needs recordings from at least two sessions")
        model = LogisticModel.fit(X[train], y[train], **fit_kw)
        p = model.predict_proba(X[test])
        pred[test] = p if proba else (p >= threshold).astype(int)
    return pred


class Smoother:
    """Fast rise, slow fall, like the rule-based score, so the learned score does not flicker."""

    def __init__(self, tau_up: float, tau_down: float) -> None:
        self.tau_up, self.tau_down = tau_up, tau_down
        self.value = 0.0
        self._t: Optional[float] = None

    def update(self, t: float, raw: float) -> float:
        if self._t is None:
            self.value = raw
        else:
            dt = min(max(t - self._t, 0.0), 1.0)
            tau = self.tau_up if raw > self.value else self.tau_down
            self.value += (raw - self.value) * (1.0 - math.exp(-dt / tau))
        self._t = t
        return self.value

    def reset(self) -> None:
        self.value, self._t = 0.0, None
