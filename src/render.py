"""
src/render.py  –  P3's core module.

Responsibilities
----------------
1.  Draw bounding boxes and track IDs on every frame.
2.  Overlay zone polygons (loiter = yellow, restricted = red, storefront = green).
3.  Show an event banner at the top of the frame when an event is active.
4.  Save an evidence thumbnail (JPEG) to outputs/thumbnails/ when an event fires.
5.  Write the fully annotated video to outputs/<stem>_annotated.mp4.

Usage (standalone, against mock data)
--------------------------------------
    python -m src.render \
        --video  data/clip1.mp4 \
        --tracks data/mock_tracks.json \
        --events data/mock_events.json \
        --config config.yaml \
        --output outputs/clip1_annotated.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils import (
    event_color,
    fmt_seconds,
    foot_point,
    id_color,
    load_json,
    scale_polygon,
)


# ─── Constants ────────────────────────────────────────────────────────────────

# Reference frame size used when defining zone polygons in config.yaml
_REF_WH = (1280, 720)

# Banner strip height at the top of the frame (pixels)
_BANNER_H = 40

# Zone overlay alpha (0 = invisible, 1 = opaque)
_ZONE_ALPHA = 0.20

# Zone border colours by type (BGR)
_ZONE_COLORS: dict[str, tuple[int, int, int]] = {
    "loiter":     (0, 200, 255),   # yellow
    "restricted": (0, 0, 220),     # red
    "storefront": (0, 200, 80),    # green
}


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _build_frame_index(tracks: list[dict]) -> dict[int, list[dict]]:
    """Group track records by frame number for O(1) lookup."""
    index: dict[int, list[dict]] = {}
    for rec in tracks:
        index.setdefault(rec["frame"], []).append(rec)
    return index


def _active_events(events: list[dict], t: float) -> list[dict]:
    """Return events whose time window covers the current timestamp *t*."""
    return [
        e for e in events
        if e["start_s"] <= t <= e.get("end_s", e["start_s"] + 5)
    ]


def _zone_active(zone: dict, frame_num: int) -> bool:
    """Return True if *frame_num* falls within the zone's active_frames range."""
    af = zone.get("active_frames")
    if af is None:
        return True   # no range set → always active
    return af[0] <= frame_num <= af[1]


def _draw_zones(
    frame: np.ndarray,
    zones_cfg: dict,
    frame_wh: tuple[int, int],
    frame_num: int = 0,
) -> np.ndarray:
    """
    Draw transparent zone polygons and labelled borders onto *frame*.
    Zones with an active_frames range are only drawn when frame_num is inside it.

    Args:
        frame:     BGR image array (will be copied, not mutated)
        zones_cfg: config.yaml['zones'] dict
        frame_wh:  (width, height) of the actual frame
        frame_num: current frame index (used for scene-aware filtering)

    Returns:
        Annotated copy of *frame*.
    """
    overlay = frame.copy()

    for zone_name, zone in zones_cfg.items():
        # ── Scene-aware gate: skip zones not active in this frame ──────────
        if not _zone_active(zone, frame_num):
            continue

        ztype = zone.get("type", "loiter")
        color = _ZONE_COLORS.get(ztype, (128, 128, 128))
        pts_raw = zone.get("polygon", [])
        if not pts_raw:
            continue

        pts = scale_polygon(pts_raw, _REF_WH, frame_wh)
        pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

        # Semi-transparent fill
        cv2.fillPoly(overlay, [pts_arr], color)

        # Solid border
        cv2.polylines(frame, [pts_arr], isClosed=True, color=color, thickness=2)

        # Zone label at the centroid
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        cv2.putText(
            frame, zone_name,
            (cx - 40, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

    # Blend the transparent fill onto the original frame
    cv2.addWeighted(overlay, _ZONE_ALPHA, frame, 1 - _ZONE_ALPHA, 0, frame)
    return frame


def _draw_detections(
    frame: np.ndarray,
    records: list[dict],
) -> np.ndarray:
    """
    Draw bounding boxes, track IDs, class labels, and foot-points
    for all detections in the current frame.
    """
    for rec in records:
        tid = rec["id"]
        cls = rec.get("class", "?")
        conf = rec.get("conf", 0.0)
        x1, y1, x2, y2 = [int(v) for v in rec["bbox"]]
        color = id_color(tid)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label: "#ID class conf"
        label = f"#{tid} {cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Dark background pill behind label
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

        # Foot-point dot
        fp = foot_point(rec["bbox"])
        cv2.circle(frame, fp, 4, color, -1)

    return frame


def _draw_banner(
    frame: np.ndarray,
    active_evts: list[dict],
    t: float,
    frame_w: int,
) -> np.ndarray:
    """
    Draw a coloured banner strip at the top of the frame listing active events.
    One event = one line; banner grows downward if multiple events are active.
    """
    if not active_evts:
        return frame

    LINE_H = 36
    PADDING = 6
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.55
    THICKNESS = 2

    banner_total_h = LINE_H * len(active_evts)

    # Solid dark background for the whole banner
    cv2.rectangle(frame, (0, 0), (frame_w, banner_total_h), (15, 15, 15), -1)

    for i, evt in enumerate(active_evts):
        behaviour = evt.get("behaviour", "unknown")
        eid = evt.get("entity_id", "?")
        zone = evt.get("zone") or ""
        start = fmt_seconds(evt["start_s"])
        color = event_color(behaviour)

        # Left coloured indicator bar
        y_top = i * LINE_H
        y_bot = y_top + LINE_H
        cv2.rectangle(frame, (0, y_top), (5, y_bot), color, -1)

        # Build text — ASCII only (OpenCV font limitation)
        label = behaviour.replace("_", " ").upper()
        text = f" #{eid}  {label}"
        if zone:
            text += f"  [{zone}]"
        text += f"  since {start}  (t={fmt_seconds(t)})"

        y_text = y_top + LINE_H - PADDING - 4
        cv2.putText(
            frame, text,
            (10, y_text),
            FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA,
        )

    return frame


def _save_thumbnail(
    frame: np.ndarray,
    evt: dict,
    thumbnail_dir: str,
) -> str:
    """
    Save a JPEG evidence thumbnail for *evt*.

    Filename format: thumb_<behaviour>_entity<id>_frame<frame>.jpg

    Returns the path of the saved thumbnail.
    """
    Path(thumbnail_dir).mkdir(parents=True, exist_ok=True)

    behaviour = evt.get("behaviour", "unknown")
    eid = evt.get("entity_id", 0)
    eframe = evt.get("evidence_frame", 0)
    fname = f"thumb_{behaviour}_entity{eid}_frame{eframe}.jpg"
    out_path = str(Path(thumbnail_dir) / fname)

    # Stamp the thumbnail with metadata
    thumb = frame.copy()
    lines = [
        f"Entity #{eid}  |  {behaviour.replace('_', ' ').upper()}",
        f"Zone: {evt.get('zone') or 'N/A'}",
        f"t = {fmt_seconds(evt['start_s'])} - {fmt_seconds(evt.get('end_s', evt['start_s']))}",
        evt.get("why", ""),
    ]
    for j, line in enumerate(lines):
        cv2.putText(
            thumb, line,
            (10, 24 + j * 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            event_color(behaviour), 2, cv2.LINE_AA,
        )

    cv2.imwrite(out_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[render] Thumbnail saved → {out_path}")
    return out_path


# ─── Public API ───────────────────────────────────────────────────────────────

def render_video(
    video_path: str,
    tracks_path: str,
    events_path: str,
    config: dict,
    output_path: str,
) -> None:
    """
    Annotate every frame of *video_path* using tracks + events, write to *output_path*.

    Args:
        video_path:  Input video (.mp4 / .avi / …)
        tracks_path: Path to tracks.json  (or mock_tracks.json for dev)
        events_path: Path to events.json  (or mock_events.json for dev)
        config:      Loaded config.yaml dict
        output_path: Destination for annotated video
    """
    # ── Load data ─────────────────────────────────────────────────────────────
    tracks = load_json(tracks_path)
    events = load_json(events_path)
    zones_cfg = config.get("zones", {})
    thumbnail_dir = config.get("thumbnail_dir", "outputs/thumbnails")

    frame_index = _build_frame_index(tracks)
    # Evidence frames we still need to capture (evidence_frame → event)
    evidence_needed: dict[int, list[dict]] = {}
    for evt in events:
        ef = evt.get("evidence_frame")
        if ef is not None:
            evidence_needed.setdefault(ef, []).append(evt)

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[render] Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or config.get("video_fps", 25)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[render] Video : {video_path}  ({frame_w}×{frame_h} @ {fps:.1f} fps, ~{total_frames} frames)")

    # ── Set up writer ─────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    frame_num = 0
    thumbnails_saved: set[int] = set()

    # ── Per-frame loop ────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_num / fps
        frame_wh = (frame_w, frame_h)

        # 1. Zone overlays
        frame = _draw_zones(frame, zones_cfg, frame_wh, frame_num)

        # 2. Detections for this frame
        records = frame_index.get(frame_num, [])
        frame = _draw_detections(frame, records)

        # 3. Event banner
        active = _active_events(events, t)
        frame = _draw_banner(frame, active, t, frame_w)

        # 4. Evidence thumbnails (save once per event, at the evidence frame)
        if frame_num in evidence_needed:
            for evt in evidence_needed[frame_num]:
                eid = evt.get("entity_id", 0)
                if eid not in thumbnails_saved:
                    _save_thumbnail(frame, evt, thumbnail_dir)
                    thumbnails_saved.add(eid)

        # 5. Frame counter (bottom-right)
        ts_label = f"t={fmt_seconds(t)}  f={frame_num}"
        (tw, _th), _ = cv2.getTextSize(ts_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(
            frame, ts_label,
            (frame_w - tw - 8, frame_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
        )

        writer.write(frame)
        frame_num += 1

        if frame_num % 250 == 0:
            pct = 100 * frame_num / max(total_frames, 1)
            print(f"[render]   frame {frame_num}/{total_frames}  ({pct:.0f}%)")

    cap.release()
    writer.release()
    print(f"[render] Done → {output_path}  ({frame_num} frames written)")


def render_single_frame(
    frame: np.ndarray,
    frame_num: int,
    fps: float,
    tracks: list[dict],
    events: list[dict],
    config: dict,
) -> np.ndarray:
    """
    Annotate a single frame in memory (used by the Streamlit dashboard for previews).

    Args:
        frame:     BGR image from cv2
        frame_num: Current frame index
        fps:       Video FPS
        tracks:    Full tracks list (will be filtered to this frame)
        events:    Full events list
        config:    Loaded config dict

    Returns:
        Annotated BGR frame.
    """
    t = frame_num / fps
    frame_wh = (frame.shape[1], frame.shape[0])
    zones_cfg = config.get("zones", {})

    frame = _draw_zones(frame, zones_cfg, frame_wh, frame_num)

    records = [r for r in tracks if r["frame"] == frame_num]
    frame = _draw_detections(frame, records)

    active = _active_events(events, t)
    frame = _draw_banner(frame, active, t, frame_wh[0])

    return frame


# ─── CLI entry point ──────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Render annotated video (P3 standalone)")
    parser.add_argument("--video",  required=True, help="Input video path")
    parser.add_argument("--tracks", default="data/mock_tracks.json")
    parser.add_argument("--events", default="data/mock_events.json")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="outputs/preview_annotated.mp4")
    args = parser.parse_args()

    import yaml  # noqa: PLC0415
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    render_video(
        video_path=args.video,
        tracks_path=args.tracks,
        events_path=args.events,
        config=cfg,
        output_path=args.output,
    )


if __name__ == "__main__":
    _cli()
