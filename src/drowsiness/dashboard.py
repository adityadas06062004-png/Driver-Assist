"""Driver Assist dashboard (Streamlit).

    driver-assist dashboard          (or: streamlit run app/dashboard.py)

Two pages: "Sessions" shows saved runs (summary, risk timeline, event log, CSV download),
"Live" runs the monitor from this page with the webcam and logs it like the CLI does.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from drowsiness.alerts import Level, SoundPlayer
from drowsiness.camera import Camera
from drowsiness.config import Config
from drowsiness.landmarks import FaceLandmarkExtractor
from drowsiness.monitor import Monitor
from drowsiness.overlay import draw_hud
from drowsiness.pipeline import Pipeline
from drowsiness.recorder import SessionRecorder
from drowsiness.storage import DEFAULT_DB, LEVEL_NAMES, Storage

st.set_page_config(page_title="Driver Assist", layout="wide")

LEVEL_COLORS = {"OK": "#2e9e4f", "CAUTION": "#e6b800", "WARNING": "#f08a00", "CRITICAL": "#d62828"}


def fmt_time(epoch: float) -> str:
    return time.strftime("%d %b %Y, %H:%M", time.localtime(epoch))


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def sessions_page(db: Storage) -> None:
    sessions = db.sessions()
    if not sessions:
        st.info("No sessions yet. Open the Live page, or run `python -m drowsiness.live` and come back.")
        return

    labels = {s["id"]: f"#{s['id']}  {fmt_time(s['started_at'])}" for s in sessions}
    sid = st.sidebar.selectbox("Session", list(labels), format_func=labels.get)
    summary = db.summary(sid)

    c = st.columns(5)
    c[0].metric("Duration", fmt_duration(summary["duration_s"]))
    c[1].metric("Average risk", f"{summary['avg_risk']:.0f}")
    c[2].metric("Peak risk", f"{summary['max_risk']:.0f}")
    c[3].metric("Warnings", summary["alerts"]["WARNING"])
    c[4].metric("Critical alerts", summary["alerts"]["CRITICAL"])
    c = st.columns(4)
    c[0].metric("Yawns", summary["yawns"])
    c[1].metric("Head nods", summary["nods"])
    c[2].metric("Phone calls", summary["phone_calls"])
    c[3].metric("Picture or camera issues", summary["issues"])

    samples = pd.DataFrame(db.samples(sid))
    events = pd.DataFrame(db.events(sid))

    st.subheader("Risk over time")
    if samples.empty:
        st.caption("No timeline samples were recorded for this session.")
    else:
        chart = samples.set_index("t")[["risk"]].rename(columns={"risk": "Risk score"})
        st.line_chart(chart, height=260)
        with st.expander("Eye and mouth signals"):
            sig = samples.set_index("t")[["ear", "mar", "perclos"]]
            st.line_chart(sig, height=220)

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Time at each level")
        order = ["OK", "CAUTION", "WARNING", "CRITICAL"]
        per_level = pd.Series({f"{i + 1} {n}": summary["seconds_at_level"].get(n, 0)
                               for i, n in enumerate(order)}, name="seconds")
        st.bar_chart(per_level)
    with right:
        st.subheader("Event log")
        if events.empty:
            st.caption("Nothing was logged, which is a good sign.")
        else:
            shown = events.copy()
            shown["level"] = shown["level"].map(lambda v: LEVEL_NAMES.get(int(v), "") if pd.notna(v) else "")
            shown["time"] = shown["t"].map(lambda v: f"{int(v // 60)}:{int(v % 60):02d}")
            kinds = st.multiselect("Show", sorted(shown["kind"].unique()), default=sorted(shown["kind"].unique()))
            shown = shown[shown["kind"].isin(kinds)]
            st.dataframe(shown[["time", "kind", "level", "risk", "detail"]].round({"risk": 0}),
                         hide_index=True)

    st.subheader("Export")
    d = st.columns(3)
    d[0].download_button("Events (CSV)", events.to_csv(index=False) if not events.empty else "",
                         file_name=f"session_{sid}_events.csv", disabled=events.empty)
    d[1].download_button("Timeline (CSV)", samples.to_csv(index=False) if not samples.empty else "",
                         file_name=f"session_{sid}_timeline.csv", disabled=samples.empty)
    if d[2].button("Delete this session"):
        db.delete_session(sid)
        st.rerun()


def live_page(db: Storage) -> None:
    st.write("Looks at the driver through the webcam. It calibrates for the first 20 seconds, "
             "so look at the road with your eyes open and mouth closed. Press Stop (or any widget) to end.")
    c = st.columns(3)
    camera_index = c[0].number_input("Camera", 0, 5, 0)
    minutes = c[1].slider("Run for (minutes)", 1, 60, 10)
    sound = c[2].checkbox("Alert sound", value=True)
    if not st.button("Start monitoring", type="primary"):
        return

    cfg = Config()
    camera = Camera(int(camera_index))
    if not camera.open():
        st.error("Could not open the camera. Allow camera access for your terminal in "
                 "System Settings > Privacy & Security > Camera, or try another camera number.")
        return

    frame_box, status_box, bar_box = st.empty(), st.empty(), st.empty()
    st.button("Stop")  # clicking reruns the script, which ends the loop below
    monitor = Monitor(cfg)
    player = SoundPlayer(enabled=sound)
    recorder = SessionRecorder(db, camera=int(camera_index))
    deadline = time.monotonic() + minutes * 60
    import cv2

    try:
        with FaceLandmarkExtractor() as extractor:
            pipe = Pipeline(monitor, camera, extractor, cfg)
            while time.monotonic() < deadline:
                step = pipe.step()
                state = step.state
                if step.new:
                    if state.alert and state.alert.sound:
                        player.play(state.alert.sound)
                    recorder.update(state)
                draw_hud(step.view, state, state.result.t, pipe.proc_fps or 0.0)
                frame_box.image(cv2.cvtColor(step.view, cv2.COLOR_BGR2RGB), channels="RGB")
                level = state.alert.level if state.alert else Level.OK
                risk = state.risk.score if state.risk else 0.0
                status_box.markdown(
                    f"**{level.name}** &nbsp; {state.message or ''}", unsafe_allow_html=True)
                bar_box.progress(min(int(risk), 100), text=f"Risk {risk:.0f}")
    finally:
        camera.release()
        recorder.close()
    st.success(f"Session {recorder.session_id} saved. Open the Sessions page to review it.")


def main() -> None:
    st.title("Driver Assist")
    st.caption("A learning project, not a certified safety device. Glasses, low light and camera "
               "angle all affect accuracy.")
    db_path = st.sidebar.text_input("Database file", str(DEFAULT_DB))
    db = Storage(Path(db_path).expanduser())
    page = st.sidebar.radio("Page", ["Sessions", "Live"])
    if page == "Sessions":
        sessions_page(db)
    else:
        live_page(db)


if __name__ == "__main__":
    main()
