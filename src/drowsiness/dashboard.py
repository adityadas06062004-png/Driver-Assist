"""Driver Assist dashboard (Streamlit).

    driver-assist dashboard          (or: streamlit run app/dashboard.py)

Pages: Overview (one session in detail), Trends (all sessions), Video (analyse an uploaded clip),
Live (monitor with the webcam), Guide (how to read everything).
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import List

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from drowsiness import ui
from drowsiness.activity import LABELS
from drowsiness.alerts import Level, SoundPlayer
from drowsiness.camera import Camera
from drowsiness.config import Config
from drowsiness.insights import EVENT_NAMES, alertness_label, alertness_score, insights, mmss
from drowsiness.landmarks import FaceLandmarkExtractor, HandLandmarkExtractor
from drowsiness.monitor import Monitor
from drowsiness.overlay import draw_hud
from drowsiness.pipeline import Pipeline
from drowsiness.recorder import SessionRecorder
from drowsiness.storage import DEFAULT_DB, LEVEL_NAMES, Storage
from drowsiness.video import analyze_video, describe

st.set_page_config(page_title="Driver Assist", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{ui.CSS}</style>", unsafe_allow_html=True)

LEVELS = ("OK", "CAUTION", "WARNING", "CRITICAL")
MMSS_EXPR = "floor(datum.value/60) + ':' + (floor(datum.value % 60) < 10 ? '0' : '') + floor(datum.value % 60)"
EVENT_COLORS = {"Alert": "#E5484D", "Yawn": "#E0A800", "Head nod": "#8A6CF0", "Phone call": "#F0832E",
                "Picture issue": "#8B95A7"}


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def fmt_time(epoch: float) -> str:
    return time.strftime("%d %b %Y, %H:%M", time.localtime(epoch))


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def source_of(session: dict) -> str:
    note = session.get("note") or ""
    return note.replace("video: ", "Video: ") if note.startswith("video:") else "Live camera"


# ---------------------------------------------------------------- charts

def timeline_chart(samples: pd.DataFrame, events: pd.DataFrame) -> alt.Chart:
    bands = pd.DataFrame({"lo": [0, 25, 50, 75], "hi": [25, 50, 75, 100], "level": list(LEVELS)})
    scale = alt.Scale(domain=list(LEVELS), range=[ui.LEVEL_COLORS[n] for n in LEVELS])
    y = alt.Y("risk:Q", scale=alt.Scale(domain=[0, 100]), title="Risk score")
    x = alt.X("t:Q", axis=alt.Axis(labelExpr=MMSS_EXPR, title=None, grid=False, tickMinStep=5))
    band = alt.Chart(bands).mark_rect(opacity=0.10).encode(
        y=alt.Y("lo:Q", scale=alt.Scale(domain=[0, 100]), title="Risk score"), y2="hi:Q",
        color=alt.Color("level:N", scale=scale, legend=None))
    area = alt.Chart(samples).mark_area(
        opacity=0.20, interpolate="monotone", color=ui.ACCENT, line={"color": ui.ACCENT, "strokeWidth": 2.4}
    ).encode(x=x, y=y)
    layers = [band, area]
    marks = event_marks(samples, events)
    if not marks.empty:
        layers.append(alt.Chart(marks).mark_point(filled=True, size=95, opacity=0.95, stroke="white", strokeWidth=1)
                      .encode(x="t:Q", y="risk:Q",
                              color=alt.Color("Event:N", scale=alt.Scale(domain=list(EVENT_COLORS),
                                                                         range=list(EVENT_COLORS.values())),
                                              legend=alt.Legend(orient="bottom", title=None)),
                              tooltip=["Time", "Event", "Detail"]))
    return alt.layer(*layers).properties(height=320)


def event_marks(samples: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Events worth drawing on the timeline, placed on the risk line."""
    if events.empty or samples.empty:
        return pd.DataFrame()
    pick = events[events["kind"].isin(["alert", "yawn", "nod", "phone_start", "issue"])].copy()
    pick = pick[(pick["kind"] != "alert") | (pick["level"].fillna(0) >= 2)]
    if pick.empty:
        return pd.DataFrame()
    names = {"alert": "Alert", "yawn": "Yawn", "nod": "Head nod", "phone_start": "Phone call", "issue": "Picture issue"}
    pick["Event"] = pick["kind"].map(names)
    pick["Time"] = pick["t"].map(mmss)
    pick["Detail"] = pick["detail"].fillna("")
    pick["risk"] = np.interp(pick["t"], samples["t"], samples["risk"].fillna(0))
    return pick[["t", "risk", "Event", "Time", "Detail"]]


def signal_chart(samples: pd.DataFrame, column: str, title: str, color: str, rule: float = None,
                 percent: bool = False) -> alt.Chart:
    df = samples[["t", column]].dropna().copy()
    if percent:
        df[column] = df[column] * 100
    base = alt.Chart(df).mark_line(color=color, strokeWidth=2, interpolate="monotone").encode(
        x=alt.X("t:Q", axis=alt.Axis(labelExpr=MMSS_EXPR, title=None, grid=False, tickMinStep=5)),
        y=alt.Y(f"{column}:Q", title=title))
    layers = [base]
    if rule is not None:
        layers.append(alt.Chart(pd.DataFrame({"y": [rule]})).mark_rule(strokeDash=[5, 4], color="#8B95A7").encode(y="y:Q"))
    return alt.layer(*layers).properties(height=170)


# ---------------------------------------------------------------- session report

def render_session(db: Storage, sid: int, allow_delete: bool = False) -> None:
    summary = db.summary(sid)
    session = next(s for s in db.sessions() if s["id"] == sid)
    samples = pd.DataFrame(db.samples(sid))
    events = pd.DataFrame(db.events(sid))
    if samples.empty:
        samples = pd.DataFrame({"t": [], "risk": [], "level": [], "ear": [], "mar": [], "perclos": []})

    score = alertness_score(summary["avg_risk"])
    label, tone = alertness_label(score)
    sub = (f"{fmt_duration(summary['duration_s'])} of driving, average risk {summary['avg_risk']:.0f}, "
           f"peak {summary['max_risk']:.0f}.")
    chips = [fmt_time(session["started_at"]), source_of(session), f"Session #{sid}"]
    html(ui.hero(score, label, ui.TONE_COLORS[tone], sub, chips, ui.level_bar(summary["seconds_at_level"])))

    alerts = summary["alerts"]
    cards = [
        ("Average risk", f"{summary['avg_risk']:.0f}", "out of 100", ui.risk_color(summary["avg_risk"])),
        ("Peak risk", f"{summary['max_risk']:.0f}", "highest moment", ui.risk_color(summary["max_risk"])),
        ("Warnings", alerts["WARNING"], "times the level reached warning", ui.LEVEL_COLORS["WARNING"]),
        ("Critical alerts", alerts["CRITICAL"], "times it reached critical", ui.LEVEL_COLORS["CRITICAL"]),
        ("Yawns", summary["yawns"], "counted", "#E0A800"),
        ("Head nods", summary["nods"], "counted", "#8A6CF0"),
        ("Phone calls", summary["phone_calls"], "hand held at the ear", "#F0832E"),
        ("Picture issues", summary["issues"], "glare, dark, face lost...", "#8B95A7"),
    ]
    for row in (cards[:4], cards[4:]):
        cols = st.columns(4)
        for col, (name, value, hint, color) in zip(cols, row):
            with col:
                html(ui.kpi_card(name, value, hint, color))
        st.write("")

    tabs = st.tabs(["Timeline", "Highlights", "Events", "Signals", "Export"])
    with tabs[0]:
        if samples.empty:
            st.info("No timeline was recorded for this session.")
        else:
            st.altair_chart(timeline_chart(samples, events))
            st.caption("The shaded bands are the alert levels. Dots mark alarms, yawns, nods, phone use and picture issues. "
                       "Drag to pan and scroll to zoom.")
    with tabs[1]:
        items = insights(summary, db.samples(sid), db.events(sid))
        left, right = st.columns(2)
        for i, item in enumerate(items):
            with (left if i % 2 == 0 else right):
                html(ui.insight_card(item.tone, item.title, item.text))
    with tabs[2]:
        events_table(events)
    with tabs[3]:
        if samples.empty or samples["ear"].dropna().empty:
            st.info("No eye or mouth signals were recorded.")
        else:
            st.caption("Eye openness (EAR), the dashed line is this driver's closed-eye threshold.")
            st.altair_chart(signal_chart(samples, "ear", "EAR", "#5B8DEF", session.get("ear_threshold")))
            st.caption("Mouth opening (MAR). Peaks that stay high are yawns, quick swings are talking.")
            st.altair_chart(signal_chart(samples, "mar", "MAR", "#8A6CF0"))
            st.caption("PERCLOS: share of the last minute with the eyes closed, in percent.")
            st.altair_chart(signal_chart(samples, "perclos", "PERCLOS %", "#E5484D", percent=True))
    with tabs[4]:
        c1, c2, c3 = st.columns(3)
        c1.download_button("Download events (CSV)", events.to_csv(index=False) if not events.empty else "",
                           file_name=f"session_{sid}_events.csv", disabled=events.empty, use_container_width=True)
        c2.download_button("Download timeline (CSV)", samples.to_csv(index=False) if len(samples) else "",
                           file_name=f"session_{sid}_timeline.csv", disabled=samples.empty, use_container_width=True)
        if allow_delete:
            with c3.popover("Delete this session", use_container_width=True):
                st.warning("This removes the session and its data for good.")
                if st.button("Yes, delete it", type="primary"):
                    db.delete_session(sid)
                    st.rerun()


def events_table(events: pd.DataFrame) -> None:
    if events.empty:
        st.success("Nothing was logged for this session, which is a good sign.")
        return
    shown = events.copy()
    shown["Time"] = shown["t"].map(mmss)
    shown["Event"] = shown["kind"].map(lambda k: EVENT_NAMES.get(k, k))
    shown["Level"] = shown["level"].map(lambda v: LEVEL_NAMES.get(int(v), "").title() if pd.notna(v) else "")
    shown["Risk"] = shown["risk"]
    shown["Detail"] = shown["detail"].fillna("")
    options = sorted(shown["Event"].unique())
    chosen = st.multiselect("Show these events", options, default=options)
    shown = shown[shown["Event"].isin(chosen)]
    st.dataframe(shown[["Time", "Event", "Level", "Risk", "Detail"]], hide_index=True, height=360,
                 column_config={"Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%d")})


# ---------------------------------------------------------------- pages

def empty_state() -> None:
    html(ui.page_header("Welcome to Driver Assist", "Nothing recorded yet. Pick one of these to get your first session."))
    cols = st.columns(3)
    steps = [
        ("Monitor live", "Open a terminal and run <code>driver-assist live</code>. Calibrate for 20 seconds, then drive."),
        ("Analyse a clip", "Open the <b>Video</b> page in the menu and upload a short driving video."),
        ("Try the live page", "Open the <b>Live</b> page and start monitoring with this computer's camera."),
    ]
    for i, (col, (title, text)) in enumerate(zip(cols, steps), start=1):
        with col:
            html(ui.step_card(i, title, text))


def overview_page(db: Storage) -> None:
    sessions = db.sessions()
    if not sessions:
        empty_state()
        return
    head, pick = st.columns([3, 2])
    with head:
        html(ui.page_header("Overview", "Everything about one drive: how alert you were, what happened and when."))
    labels = {s["id"]: f"#{s['id']}  {fmt_time(s['started_at'])}  ({source_of(s)})" for s in sessions}
    with pick:
        sid = st.selectbox("Session", list(labels), format_func=labels.get)
    render_session(db, sid, allow_delete=True)


def trends_page(db: Storage) -> None:
    html(ui.page_header("Trends", "All your sessions side by side."))
    sessions = db.sessions()
    if not sessions:
        empty_state()
        return
    rows: List[dict] = []
    for s in reversed(sessions):
        summ = db.summary(s["id"])
        risks = [r["risk"] for r in db.samples(s["id"]) if r["risk"] is not None]
        step = max(1, len(risks) // 40)
        rows.append({
            "Session": s["id"], "Started": fmt_time(s["started_at"]), "Source": source_of(s),
            "Length": fmt_duration(summ["duration_s"]), "Seconds": summ["duration_s"],
            "Average risk": round(summ["avg_risk"]), "Peak risk": round(summ["max_risk"]),
            "Warnings": summ["alerts"]["WARNING"], "Critical": summ["alerts"]["CRITICAL"],
            "Yawns": summ["yawns"], "Nods": summ["nods"], "Phone calls": summ["phone_calls"],
            "Risk over time": risks[::step],
        })
    df = pd.DataFrame(rows)
    total = df["Seconds"].sum()
    cols = st.columns(4)
    cards = [("Sessions", len(df), "recorded", ui.ACCENT),
             ("Time monitored", fmt_duration(total), "in total", "#8A6CF0"),
             ("Average risk", f"{df['Average risk'].mean():.0f}", "across sessions", ui.risk_color(df["Average risk"].mean())),
             ("Alarms", int(df["Warnings"].sum() + df["Critical"].sum()), "warnings and critical", "#E5484D")]
    for col, (name, value, hint, color) in zip(cols, cards):
        with col:
            html(ui.kpi_card(name, value, hint, color))
    st.write("")

    if len(df) > 1:
        chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("Session:O", title="Session"), y=alt.Y("Average risk:Q", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Average risk:Q", scale=alt.Scale(domain=[0, 25, 50, 75, 100],
                                                              range=["#2FA66A", "#2FA66A", "#E0A800", "#F0832E", "#E5484D"]),
                            legend=None),
            tooltip=["Session", "Started", "Source", "Average risk", "Peak risk"]).properties(height=240)
        st.altair_chart(chart)
    st.dataframe(df.drop(columns=["Seconds"]).iloc[::-1], hide_index=True, height=380, column_config={
        "Average risk": st.column_config.ProgressColumn("Average risk", min_value=0, max_value=100, format="%d"),
        "Risk over time": st.column_config.LineChartColumn("Risk over time", y_min=0, y_max=100),
    })
    st.caption("Open a session on the Overview page to see its timeline and highlights.")


def video_page(db: Storage) -> None:
    html(ui.page_header("Analyse a video", "Upload a clip of someone driving and get the same report as a live session."))
    cols = st.columns(3)
    for i, (col, (title, text)) in enumerate(zip(cols, [
        ("Pick a clip", "Face clearly visible, looking at the road with eyes open for the first seconds."),
        ("Choose speed", "Analysing every 2nd frame is about twice as fast and almost as accurate."),
        ("Read the report", "You get a score, a timeline, highlights and the moment with the highest risk."),
    ]), start=1):
        with col:
            html(ui.step_card(i, title, text))
    st.write("")
    st.caption("The clip is processed by the machine running this app and deleted afterwards. If this is a hosted copy, "
               "do not upload anything private.")
    up = st.file_uploader("Driving clip", type=["mp4", "mov", "avi", "mkv"])
    c = st.columns(3)
    calib = c[0].slider("Calibration seconds", 5, 30, 10, help="How much of the start is used to learn what 'alert' looks like.")
    stride = c[1].select_slider("Analyse every Nth frame", options=[1, 2, 3], value=2, help="Higher is faster.")
    phone = c[2].checkbox("Check for phone use (slower)", value=False)
    save = st.checkbox("Save the result to my sessions", value=False)
    if up is None:
        st.info("Choose a video above to begin.")
        return
    if not st.button("Analyse the clip", type="primary"):
        return

    with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as fh:
        fh.write(up.getbuffer())
        tmp = fh.name
    bar = st.progress(0.0, text="Analysing")
    try:
        result = analyze_video(tmp, Config(calibration_s=float(calib), detect_phone=phone),
                               storage=db if save else None, stride=int(stride), max_seconds=300,
                               progress=lambda f: bar.progress(f, text="Analysing"))
    except ValueError as exc:
        st.error(str(exc))
        return
    finally:
        os.unlink(tmp)
    bar.empty()

    if not result.calibrated:
        st.warning(describe(result))
        return
    st.toast("Analysis finished")
    if result.peak_frame is not None:
        st.image(cv2.cvtColor(result.peak_frame, cv2.COLOR_BGR2RGB), caption="The moment with the highest risk")
    render_session(result.storage, result.session_id)


def live_page(db: Storage) -> None:
    html(ui.page_header("Live monitoring", "Uses the camera of the computer running this app. On a hosted copy, use Video."))
    with st.container(border=True):
        c = st.columns(4)
        camera_index = c[0].number_input("Camera", 0, 5, 0)
        minutes = c[1].slider("Run for (minutes)", 1, 60, 10)
        sound = c[2].checkbox("Alert sound", value=True)
        phone = c[3].checkbox("Phone detection", value=True)
        start = st.button("Start monitoring", type="primary")
    if not start:
        st.caption("Calibration takes the first 20 seconds: look at the road with your eyes open and your mouth closed. "
                   "Press Stop, or change any setting, to end the run.")
        return

    cfg = Config(detect_phone=phone)
    camera = Camera(int(camera_index))
    if not camera.open():
        st.error("Could not open the camera. Allow camera access for your terminal in System Settings, "
                 "Privacy and Security, Camera, or try another camera number.")
        return

    video_col, side_col = st.columns([3, 1])
    frame_box = video_col.empty()
    with side_col:
        panel = st.empty()
        st.button("Stop")  # clicking reruns the script, which ends the loop below
    monitor = Monitor(cfg)
    player = SoundPlayer(enabled=sound)
    recorder = SessionRecorder(db, camera=int(camera_index))
    deadline = time.monotonic() + minutes * 60
    last_panel = 0.0
    hands_ctx = HandLandmarkExtractor() if phone else None
    try:
        with FaceLandmarkExtractor() as extractor:
            pipe = Pipeline(monitor, camera, extractor, cfg, hands=hands_ctx)
            while time.monotonic() < deadline:
                step = pipe.step()
                state = step.state
                if step.new:
                    if state.alert and state.alert.sound:
                        player.play(state.alert.sound)
                    recorder.update(state)
                draw_hud(step.view, state, state.result.t, pipe.proc_fps or 0.0)
                frame_box.image(cv2.cvtColor(step.view, cv2.COLOR_BGR2RGB), channels="RGB")
                now = time.monotonic()
                if now - last_panel > 0.25:
                    last_panel = now
                    level = state.alert.level if state.alert else Level.OK
                    risk = state.risk.score if state.risk else 0.0
                    activity = LABELS[state.activity.activity] if state.activity else "Calibrating" \
                        if state.phase == "calibrating" else "Monitoring"
                    panel.markdown(ui.live_panel(level.name.title(), state.message or "", risk, activity),
                                   unsafe_allow_html=True)
    finally:
        camera.release()
        recorder.close()
        if hands_ctx is not None:
            hands_ctx.close()
    st.success(f"Session {recorder.session_id} saved. Open the Overview page to review it.")


def guide_page() -> None:
    html(ui.page_header("Guide", "How to read the numbers, and how to get the best results."))
    with st.expander("What the alert levels mean", expanded=True):
        html("".join(ui.insight_card(t, name, text) for t, name, text in [
            ("good", "OK (below 25)", "No sign of fatigue."),
            ("warn", "Caution (25 and above)", "Early signs such as more yawns or slower blinks."),
            ("warn", "Warning (50 and above)", "Clear drowsiness signs. Time for a break soon. A beep repeats every 3 seconds."),
            ("bad", "Critical (75 and above)", "Eyes closed or head down for too long. Pull over. A beep repeats every 1.2 seconds."),
        ]))
    with st.expander("The signals, in plain words"):
        st.markdown(
            "**EAR (eye aspect ratio)** is how open the eyes are, about 0.3 when open and near 0 when closed. "
            "Each driver gets their own closed-eye threshold during calibration.\n\n"
            "**PERCLOS** is the share of the last minute with the eyes closed. It rises when blinks get long.\n\n"
            "**MAR (mouth aspect ratio)** is how open the mouth is. One long opening is a yawn. Quick swings are talking.\n\n"
            "**Head nods** are counted when the head stays tilted down for a second.\n\n"
            "**Phone use** is a hand held beside the face at ear height, together with talking or a long hold.")
    with st.expander("Tips for accurate results"):
        st.markdown(
            "Place the camera near eye level and keep the face well lit, without a bright window behind you.\n\n"
            "Calibrate in the same light and position you will drive in, looking at the road with eyes open.\n\n"
            "Glasses can reflect light. If the picture shows glare, tilt the camera or the screen slightly.\n\n"
            "In a dim room, run with night mode on: `driver-assist live --night on`.")
    with st.expander("Privacy and limits"):
        st.markdown(
            "Everything runs on your computer. Only numbers and events are saved, never video or photos.\n\n"
            "This is a learning project, not a certified safety device. Glasses, low light and camera angle all affect "
            "accuracy, and it must never replace rest or common sense on the road.")


# ---------------------------------------------------------------- shell

def main() -> None:
    with st.sidebar:
        html(ui.brand())
        page = st.radio("Menu", ["Overview", "Trends", "Video", "Live", "Guide"], label_visibility="collapsed")
        st.write("")
        with st.expander("Data source"):
            db_path = st.text_input("Database file", str(DEFAULT_DB))
        st.caption("A learning project, not a certified safety device.")
    db = Storage(Path(db_path).expanduser())
    {"Overview": overview_page, "Trends": trends_page, "Video": video_page, "Live": live_page}.get(
        page, lambda _db: guide_page())(db)


if __name__ == "__main__":
    main()
