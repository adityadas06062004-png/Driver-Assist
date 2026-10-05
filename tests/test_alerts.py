import subprocess

from drowsiness import AlertStateMachine, Level, SoundPlayer
from drowsiness import alerts as alerts_mod


def test_escalation_is_immediate_and_sounds():
    sm = AlertStateMachine()
    assert sm.update(0.0, 10).level == Level.OK
    e = sm.update(0.1, 30)
    assert e.level == Level.CAUTION and e.changed and e.sound is None
    e = sm.update(0.2, 60)
    assert e.level == Level.WARNING and e.sound == "warning"
    e = sm.update(0.3, 90)
    assert e.level == Level.CRITICAL and e.sound == "critical"


def test_sound_repeats_on_schedule():
    sm = AlertStateMachine()
    assert sm.update(0.0, 90).sound == "critical"
    assert sm.update(0.5, 90).sound is None
    assert sm.update(1.3, 90).sound == "critical"


def test_hysteresis_prevents_flicker():
    sm = AlertStateMachine()
    sm.update(0.0, 90)
    for i in range(1, 100):                 # 9.9 s hovering just under the 75 threshold
        assert sm.update(i * 0.1, 70).level == Level.CRITICAL


def test_steps_down_only_after_hold():
    sm = AlertStateMachine()
    sm.update(0.0, 90)
    assert sm.update(1.0, 60).level == Level.CRITICAL      # below exit line, hold just started
    assert sm.update(2.0, 60).level == Level.CRITICAL      # 1.0 s held, need 1.5
    e = sm.update(2.6, 60)
    assert e.level == Level.WARNING and e.changed


def test_brief_dip_resets_exit_timer():
    sm = AlertStateMachine()
    sm.update(0.0, 90)
    sm.update(1.0, 60)
    sm.update(1.5, 90)                       # back up: timer cancelled
    assert sm.update(2.6, 60).level == Level.CRITICAL


def test_returns_to_ok():
    sm = AlertStateMachine()
    sm.update(0.0, 30)
    sm.update(1.0, 5)
    assert sm.update(2.6, 5).level == Level.OK


class FakeProc:
    def __init__(self, cmd):
        self.cmd, self.running = cmd, True

    def poll(self):
        return None if self.running else 0


def test_sound_player_uses_afplay_and_never_overlaps(monkeypatch):
    started = []
    monkeypatch.setattr(alerts_mod.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: started.append(FakeProc(cmd)) or started[-1])
    p = SoundPlayer()
    p.play("critical")
    p.play("critical")                        # still playing -> ignored
    assert len(started) == 1 and started[0].cmd[0] == "afplay"
    started[0].running = False
    p.play("warning")
    assert len(started) == 2


def test_sound_player_disabled_and_fallback(monkeypatch, capsys):
    monkeypatch.setattr(alerts_mod.platform, "system", lambda: "Darwin")

    def boom(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "Popen", boom)
    SoundPlayer(enabled=False).play("critical")
    assert capsys.readouterr().out == ""
    SoundPlayer().play("critical")
    assert "\a" in capsys.readouterr().out


def test_voice_mode_speaks_only_critical(monkeypatch):
    monkeypatch.setattr(alerts_mod.platform, "system", lambda: "Darwin")
    p = SoundPlayer(voice=True)
    assert p._command("critical")[0] == "say"
    assert p._command("warning")[0] == "afplay"
