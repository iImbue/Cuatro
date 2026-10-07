"""
app.py  –  HackNex PS07 Streamlit Dashboard (P3)

Run:
    streamlit run app.py

What it shows:
- Sidebar: load video, tracks.json, events.json, config.yaml
- Events table with colour-coded behaviour types
- Click any event row → see the evidence thumbnail + annotated frame preview
- Timeline bar (horizontal) showing when each event fired
- Entity summary: which IDs appeared, how long, how many events
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from PIL import Image

from src.render import render_single_frame
from src.utils import (
    event_color,
    fmt_seconds,
    load_json,
)

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HackNex PS07 – Mall Vision",
    page_icon="🏬",
    layout="wide",
)

# ─── Colour helpers for Streamlit (hex strings) ───────────────────────────────

_BEHAVIOUR_HEX: dict[str, str] = {
    "loitering":        "#dc3545",   # red
    "restricted_entry": "#fd7e14",   # orange-red
    "abandoned_bag":    "#ffc107",   # amber
    "storefront_dwell": "#28a745",   # green
    "running":          "#0d6efd",   # blue
    "crowd_buildup":    "#6f42c1",   # purple
}
_DEFAULT_HEX = "#6c757d"


def _hex(behaviour: str) -> str:
    return _BEHAVIOUR_HEX.get(behaviour, _DEFAULT_HEX)


def _badge(behaviour: str) -> str:
    """Return an HTML coloured badge for a behaviour label."""
    color = _hex(behaviour)
    label = behaviour.replace("_", " ").upper()
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:0.78em;font-weight:600">{label}</span>'
    )


# ─── Session state defaults ───────────────────────────────────────────────────

if "selected_event_idx" not in st.session_state:
    st.session_state.selected_event_idx = 0

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏬 Mall Vision System")
    st.caption("HackNex PS07 – P3 Output Module")
    st.markdown("---")

    st.subheader("Load files")

    video_file = st.file_uploader("Video (mp4 / avi)", type=["mp4", "avi", "mov"])
    tracks_file = st.file_uploader("tracks.json", type=["json"])
    events_file = st.file_uploader("events.json", type=["json"])
    config_file = st.file_uploader("config.yaml", type=["yaml", "yml"])

    st.markdown("---")
    st.caption("If no files are uploaded, demo data is loaded automatically.")

    use_mock = st.checkbox("Force mock data", value=False)

# ─── Load data ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data …")
def _load_json_bytes(raw: bytes) -> list:
    import json  # noqa: PLC0415
    return json.loads(raw.decode("utf-8"))


@st.cache_data(show_spinner="Loading config …")
def _load_yaml_bytes(raw: bytes) -> dict:
    return yaml.safe_load(raw.decode("utf-8"))


def _default_path(name: str) -> Path:
    return Path("data") / name


# Load tracks — prefer outputs/tracks.json, fall back to mock
if tracks_file and not use_mock:
    tracks: list = _load_json_bytes(tracks_file.read())
elif Path("outputs/tracks.json").exists() and not use_mock:
    tracks = load_json(Path("outputs/tracks.json"))
elif _default_path("mock_tracks.json").exists():
    tracks = load_json(_default_path("mock_tracks.json"))
else:
    tracks = []

# Load events — prefer outputs/events.json, fall back to mock
if events_file and not use_mock:
    events: list = _load_json_bytes(events_file.read())
elif Path("outputs/events.json").exists() and not use_mock:
    events = load_json(Path("outputs/events.json"))
elif _default_path("mock_events.json").exists():
    events = load_json(_default_path("mock_events.json"))
else:
    events = []

# Load config
if config_file and not use_mock:
    cfg: dict = _load_yaml_bytes(config_file.read())
elif Path("config.yaml").exists():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
else:
    cfg = {}

# ─── Main layout ──────────────────────────────────────────────────────────────

st.title("Mall Vision System — Event Dashboard")
st.caption(
    "P3 Output Module · HackNex PS07 · "
    "Each row is a detected behaviour event. Click a row to inspect."
)

if not events:
    st.warning(
        "No events loaded. Upload events.json in the sidebar, "
        "or make sure data/mock_events.json exists."
    )
    st.stop()

# ─── Section 1: Events table ──────────────────────────────────────────────────

st.markdown("## 📋 Detected Events")

df = pd.DataFrame(events)

# Friendly display columns
display_cols = {
    "entity_id":  "Entity",
    "behaviour":  "Behaviour",
    "start_s":    "Start",
    "end_s":      "End",
    "zone":       "Zone",
    "confidence": "Confidence",
    "why":        "Reason",
}
df_view = df.rename(columns=display_cols)
df_view["Start"] = df_view["Start"].apply(fmt_seconds)
df_view["End"]   = df_view["End"].apply(lambda v: fmt_seconds(v) if pd.notna(v) else "—")
df_view["Confidence"] = df_view["Confidence"].apply(lambda v: f"{v:.0%}" if pd.notna(v) else "—")
df_view["Zone"] = df_view["Zone"].fillna("—")

# Render behaviour badges as HTML
html_rows = []
for _, row in df_view.iterrows():
    badge = _badge(row["Behaviour"])
    html_rows.append(
        f"<tr>"
        f"<td>#{row['Entity']}</td>"
        f"<td>{badge}</td>"
        f"<td>{row['Start']}</td>"
        f"<td>{row['End']}</td>"
        f"<td>{row['Zone']}</td>"
        f"<td>{row['Confidence']}</td>"
        f"<td style='font-size:0.8em'>{row['Reason']}</td>"
        f"</tr>"
    )

table_html = """
<table style="width:100%;border-collapse:collapse;font-size:0.9em">
  <thead>
    <tr style="background:#1e1e1e;color:#fff">
      <th>Entity</th><th>Behaviour</th><th>Start</th><th>End</th>
      <th>Zone</th><th>Conf</th><th>Reason</th>
    </tr>
  </thead>
  <tbody>
""" + "\n".join(html_rows) + "</tbody></table>"

st.markdown(table_html, unsafe_allow_html=True)

# ─── Section 2: Event selector + evidence ─────────────────────────────────────

st.markdown("---")
st.markdown("## 🔍 Inspect an Event")

event_labels = [
    f"#{e.get('entity_id')} — {e.get('behaviour','?').replace('_',' ').upper()}"
    f"  @ {fmt_seconds(e['start_s'])}"
    for e in events
]
sel_idx = st.selectbox(
    "Select event",
    range(len(event_labels)),
    format_func=lambda i: event_labels[i],
    key="selected_event_idx",
)
sel_evt = events[sel_idx]

col_info, col_thumb = st.columns([1, 1])

with col_info:
    st.markdown(
        f"**Entity:** #{sel_evt.get('entity_id')}  \n"
        f"**Behaviour:** {_badge(sel_evt.get('behaviour','?'))}",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**Zone:** {sel_evt.get('zone') or '—'}  \n"
        f"**Time:** {fmt_seconds(sel_evt['start_s'])} → "
        f"{fmt_seconds(sel_evt.get('end_s', sel_evt['start_s']))}  \n"
        f"**Confidence:** {sel_evt.get('confidence', 0):.0%}  \n"
        f"**Why:** {sel_evt.get('why', '—')}"
    )
    if "owner_id" in sel_evt:
        st.markdown(
            f"**Bag owner:** #{sel_evt['owner_id']}  \n"
            f"**Owner last seen:** {fmt_seconds(sel_evt.get('owner_last_seen_s', 0))}"
        )

with col_thumb:
    # Try to load saved thumbnail
    behaviour = sel_evt.get("behaviour", "unknown")
    eid = sel_evt.get("entity_id", 0)
    eframe = sel_evt.get("evidence_frame", 0)
    thumb_name = f"thumb_{behaviour}_entity{eid}_frame{eframe}.jpg"
    thumb_path = Path(cfg.get("thumbnail_dir", "outputs/thumbnails")) / thumb_name

    if thumb_path.exists():
        st.image(str(thumb_path), caption=f"Evidence thumbnail – {thumb_name}", use_column_width=True)
    else:
        # Fall back to generating a preview from the uploaded video
        if video_file:
            video_bytes = video_file.read()
            tmp_path = Path("outputs") / "tmp_preview.mp4"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(video_bytes)

            cap = cv2.VideoCapture(str(tmp_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, eframe)
            ret, raw_frame = cap.read()
            cap.release()

            if ret:
                fps_v = cap.get(cv2.CAP_PROP_FPS) or cfg.get("video_fps", 25)
                annotated = render_single_frame(
                    raw_frame, eframe, fps_v, tracks, events, cfg
                )
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(rgb, caption=f"Frame {eframe} (live preview)", use_column_width=True)
            else:
                st.info(f"Thumbnail not found at {thumb_path}. Run the pipeline first.")
        else:
            st.info(
                f"No thumbnail found at `{thumb_path}`.  \n"
                "Upload a video or run the pipeline to generate thumbnails."
            )

# ─── Section 3: Timeline ──────────────────────────────────────────────────────

st.markdown("---")
st.markdown("## ⏱ Event Timeline")

if events:
    all_ends = [e.get("end_s", e["start_s"] + 5) for e in events]
    max_t = max(all_ends) if all_ends else 60.0

    import matplotlib.pyplot as plt  # noqa: PLC0415

    timeline_data: list[dict] = []
    for e in events:
        timeline_data.append(
            {
                "Entity": f"#{e.get('entity_id')} {e.get('behaviour', '?').replace('_', ' ')}",
                "Duration (s)": max(
                    e.get("end_s", e["start_s"] + 5) - e["start_s"], 1.0
                ),
                "colour": _hex(e.get("behaviour", "")),
            }
        )

    tdf = pd.DataFrame(timeline_data)

    fig, ax = plt.subplots(figsize=(8, max(2, len(tdf) * 0.55)))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    bars = ax.barh(
        tdf["Entity"],
        tdf["Duration (s)"],
        color=tdf["colour"],
        height=0.5,
    )

    # Value labels on each bar
    for bar, val in zip(bars, tdf["Duration (s)"]):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}s",
            va="center",
            color="#cccccc",
            fontsize=9,
        )

    ax.set_xlabel("Duration (seconds)", color="#cccccc")
    ax.tick_params(colors="#cccccc", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.invert_yaxis()  # first event at the top
    plt.tight_layout(pad=0.5)

    st.pyplot(fig, use_container_width=False, transparent=True)
    plt.close(fig)
    st.caption("Bar width = event duration in seconds.")

# ─── Section 4: Entity summary ────────────────────────────────────────────────

st.markdown("---")
st.markdown("## 👥 Entity Summary")

if tracks:
    entity_ids = sorted({r["id"] for r in tracks})
    summary_rows = []
    for eid in entity_ids:
        recs = [r for r in tracks if r["id"] == eid]
        cls_set = {r.get("class", "?") for r in recs}
        t_first = min(r["t"] for r in recs)
        t_last  = max(r["t"] for r in recs)
        n_events = sum(1 for e in events if e.get("entity_id") == eid)
        summary_rows.append(
            {
                "ID":      f"#{eid}",
                "Class":   " / ".join(cls_set),
                "First seen": fmt_seconds(t_first),
                "Last seen":  fmt_seconds(t_last),
                "Track length (s)": round(t_last - t_first, 1),
                "Events":  n_events,
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
else:
    st.info("No track data loaded.")

# ─── Section 5: Video player ──────────────────────────────────────────────────

if video_file:
    st.markdown("---")
    st.markdown("## 🎬 Annotated Video Preview")
    st.caption(
        "This plays the original uploaded video. "
        "Run `python run.py --video …` to generate the fully annotated version."
    )
    # Rewind after earlier read
    video_file.seek(0)
    st.video(video_file)

# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "HackNex 2026 Internal Qualifier · Karunya Institute of Technology and Sciences · "
    "Division of CSE · Mall Vision System · P3 Output Module"
)
