from pathlib import Path

import pytest

from drowsiness import Calibration, Config, Monitor, SessionRecorder, Storage, ui
from helpers import FRAME, add_head_landmarks, make_face, make_hand

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
APP = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")
CAL = Calibration(True, "ok", 0.26, 0.195, 0.0, 0.5, 0.0, None, 500)


def seed(path, sessions=2):
    """A database with a couple of sessions that contain alarms, a yawn, a phone call and a picture issue."""
    with Storage(path) as db:
        for _ in range(sessions):
            rec = SessionRecorder(db)
            mon = Monitor(Config(phone_listen_s=0.0), calibration=CAL)
            t = 0.0
            plan = [(dict(ear=0.26), [], 4), (dict(ear=0.05), [], 3), (dict(ear=0.26), [], 4),
                    (dict(ear=0.26, mar=0.9), [], 3), (dict(ear=0.26), [make_hand(190, 260)], 7),
                    (dict(ear=0.26), [], 3)]
            for face, hands, secs in plan:
                pts = add_head_landmarks(make_face(**face))
                for _ in range(int(secs * 30)):
                    rec.update(mon.update(pts, t, FRAME, hands=hands))
                    t += 1 / 30
            rec.close()


def open_app(db_path, page="Overview"):
    at = streamlit_testing.AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(str(db_path)).run()
    if page != "Overview":
        at.sidebar.radio[0].set_value(page).run()
    return at


@pytest.mark.parametrize("page", ["Overview", "Trends", "Video", "Live", "Guide"])
def test_every_page_renders_with_data(tmp_path, page):
    db = tmp_path / "d.db"
    seed(db)
    at = open_app(db, page)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", ["Overview", "Trends"])
def test_pages_show_a_friendly_empty_state_without_data(tmp_path, page):
    at = open_app(tmp_path / "empty.db", page)
    assert not at.exception
    assert any("Welcome" in m.value for m in at.markdown)


def test_overview_shows_score_cards_tabs_and_highlights(tmp_path):
    db = tmp_path / "d.db"
    seed(db)
    at = open_app(db)
    body = " ".join(m.value for m in at.markdown)
    assert "da-hero" in body and "Average risk" in body and "Phone calls" in body
    assert len(at.tabs) == 5 and [t.label for t in at.tabs][0] == "Timeline"
    assert "Highest risk" in body


def test_choosing_another_session_works(tmp_path):
    db = tmp_path / "d.db"
    seed(db, sessions=3)
    at = open_app(db)
    options = at.selectbox[0].options
    assert len(options) == 3
    at.selectbox[0].set_value(1).run()          # the options are session ids shown with a label
    assert not at.exception
    assert "Session #1" in " ".join(m.value for m in at.markdown)


def test_ui_blocks_escape_text_and_use_the_level_colours():
    assert "&lt;b&gt;" in ui.kpi_card("<b>x</b>", "1")
    assert "&lt;script&gt;" in ui.insight_card("warn", "<script>", "text")
    bar = ui.level_bar({"OK": 30, "CAUTION": 0, "WARNING": 10, "CRITICAL": 10})
    assert ui.LEVEL_COLORS["WARNING"] in bar and "75%" not in bar and "60%" in bar
    assert ui.level_bar({"OK": 0}) == ""
    assert ui.risk_color(10) == ui.LEVEL_COLORS["OK"] and ui.risk_color(80) == ui.LEVEL_COLORS["CRITICAL"]
    panel = ui.live_panel("Warning", "Drowsiness warning", 57.4, "Driving")
    assert "57" in panel and ui.LEVEL_COLORS["WARNING"] in panel
