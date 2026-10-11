import pytest

from drowsiness import cli


def test_help_lists_every_command(capsys):
    cli.main([])
    out = capsys.readouterr().out
    for name in cli.COMMANDS:
        assert name in out


def test_unknown_command_exits_with_error(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["nope"])
    assert e.value.code == 2
    assert "Unknown command" in capsys.readouterr().err


def test_command_gets_only_its_own_arguments(monkeypatch):
    seen = {}
    monkeypatch.setitem(cli.COMMANDS, "export", lambda: seen.setdefault("argv", list(cli.sys.argv)))
    cli.main(["export", "--what", "samples"])
    assert seen["argv"][1:] == ["--what", "samples"]


def test_export_runs_through_the_cli(tmp_path, capsys):
    from drowsiness import Storage
    db = tmp_path / "x.db"
    with Storage(db) as s:
        sid = s.start_session()
        s.log_event(sid, 1.0, "nod")
    out = tmp_path / "e.csv"
    cli.main(["export", "--db", str(db), "--out", str(out)])
    assert out.read_text().count("\n") >= 2


def test_ctrl_c_on_the_dashboard_exits_quietly(monkeypatch):
    import subprocess
    monkeypatch.setitem(__import__("sys").modules, "streamlit", object())

    def interrupted(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "call", interrupted)
    with pytest.raises(SystemExit) as e:
        cli._dashboard()
    assert e.value.code == 0
