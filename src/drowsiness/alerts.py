"""Alert levels with hysteresis, plus a free, offline sound player."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from .config import Config


class Level(IntEnum):
    OK = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3


MESSAGES = {
    Level.OK: "Alert",
    Level.CAUTION: "Signs of fatigue",
    Level.WARNING: "Drowsiness warning: take a break soon",
    Level.CRITICAL: "DROWSY! Pull over and rest",
}


@dataclass
class AlertEvent:
    level: Level
    changed: bool               # level changed on this frame
    sound: Optional[str]        # "warning" / "critical" when a sound should play now
    message: str


class AlertStateMachine:
    """Escalates immediately, steps down only after the score stays clearly lower.

    Hysteresis: to leave a level the score must fall ``alert_hysteresis`` points
    below that level's entry threshold and stay there for ``alert_exit_hold_s``.
    That stops alarms from flickering around a threshold.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        c = self.config
        self._enter = {Level.CAUTION: c.caution_at, Level.WARNING: c.warning_at,
                       Level.CRITICAL: c.critical_at}
        self.level = Level.OK
        self._below_since: Optional[float] = None
        self._last_sound_t: Optional[float] = None

    def reset(self) -> None:
        self.level = Level.OK
        self._below_since = None
        self._last_sound_t = None

    def _target(self, score: float) -> Level:
        target = Level.OK
        for lvl in (Level.CAUTION, Level.WARNING, Level.CRITICAL):
            if score >= self._enter[lvl]:
                target = lvl
        return target

    def update(self, t: float, score: float) -> AlertEvent:
        c = self.config
        target = self._target(score)
        changed = False

        if target > self.level:
            self.level = target
            self._below_since = None
            changed = True
        elif target < self.level:
            if score < self._enter[self.level] - c.alert_hysteresis:
                if self._below_since is None:
                    self._below_since = t
                elif t - self._below_since >= c.alert_exit_hold_s:
                    self.level = target
                    self._below_since = None
                    changed = True
            else:
                self._below_since = None
        else:
            self._below_since = None

        sound = None
        if self.level >= Level.WARNING:
            interval = c.critical_repeat_s if self.level == Level.CRITICAL else c.warning_repeat_s
            if changed or self._last_sound_t is None or t - self._last_sound_t >= interval:
                sound = "critical" if self.level == Level.CRITICAL else "warning"
                self._last_sound_t = t
        else:
            self._last_sound_t = None
        return AlertEvent(self.level, changed, sound, MESSAGES[self.level])


class SoundPlayer:
    """Plays alert sounds without blocking. Free: uses macOS built-in sounds.

    Never overlaps sounds. Falls back to the terminal bell on other systems.
    """

    _MAC_SOUNDS = {"warning": "/System/Library/Sounds/Funk.aiff",
                   "critical": "/System/Library/Sounds/Sosumi.aiff"}

    def __init__(self, enabled: bool = True, voice: bool = False) -> None:
        self.enabled = enabled
        self.voice = voice
        self._proc: Optional[subprocess.Popen] = None

    def play(self, kind: str) -> None:
        if not self.enabled:
            return
        if self._proc is not None and self._proc.poll() is None:
            return  # previous sound still playing
        cmd = self._command(kind)
        if cmd is None:
            self._bell()
            return
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            self._bell()

    def _command(self, kind: str):
        if platform.system() != "Darwin":
            return None
        if self.voice and kind == "critical":
            return ["say", "Warning. You look drowsy. Pull over and rest."]
        return ["afplay", self._MAC_SOUNDS.get(kind, self._MAC_SOUNDS["warning"])]

    @staticmethod
    def _bell() -> None:
        print("\a", end="", flush=True)
