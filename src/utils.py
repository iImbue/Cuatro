"""
src/utils.py  –  Shared helpers used by render.py, app.py, and the rest of the pipeline.

Covers:
- JSON load / save
- Foot-point (bottom-center of a bounding box)
- Deterministic per-ID colour map
- Seconds → HH:MM:SS formatter
- Zone polygon scaling (when display size differs from reference size)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ─── JSON I/O ─────────────────────────────────────────────────────────────────

def load_json(path: str | Path) -> Any:
    """Load and return a JSON file. Raises FileNotFoundError if missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[utils] JSON file not found: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Serialise *data* to JSON at *path*, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    print(f"[utils] Saved → {p}")


# ─── Geometry ─────────────────────────────────────────────────────────────────

def foot_point(bbox: list[float]) -> tuple[int, int]:
    """
    Return the bottom-centre pixel of a bounding box.

    Args:
        bbox: [x1, y1, x2, y2]  (pixel coords, any numeric type)

    Returns:
        (cx, y2) as (int, int)
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int(y2))


def scale_polygon(
    polygon: list[list[int]],
    src_wh: tuple[int, int],
    dst_wh: tuple[int, int],
) -> list[tuple[int, int]]:
    """
    Scale a polygon from a reference resolution to a display resolution.

    Args:
        polygon: list of [x, y] points in the source coordinate space
        src_wh:  (width, height) of the reference frame (e.g. 1280×720)
        dst_wh:  (width, height) of the actual frame being rendered

    Returns:
        List of (x, y) int tuples scaled to dst_wh.
    """
    sw, sh = src_wh
    dw, dh = dst_wh
    return [
        (int(x * dw / sw), int(y * dh / sh))
        for x, y in polygon
    ]


# ─── Colour map ───────────────────────────────────────────────────────────────

# Palette of visually distinct BGR colours (OpenCV uses BGR, not RGB).
_PALETTE_BGR: list[tuple[int, int, int]] = [
    (255, 128,   0),   # orange
    (  0, 255, 128),   # spring green
    (128,   0, 255),   # violet
    (  0, 200, 255),   # yellow
    (255,   0, 128),   # rose
    ( 64, 255,  64),   # lime
    (255,  64, 192),   # pink
    (  0, 128, 255),   # gold
    (192,  64,  64),   # steel blue
    ( 64, 192, 255),   # sky
]


def id_color(track_id: int) -> tuple[int, int, int]:
    """
    Return a deterministic BGR colour for a given track ID.

    The same ID always gets the same colour within a session, which makes
    it easy to follow individual entities across frames.
    """
    return _PALETTE_BGR[track_id % len(_PALETTE_BGR)]


# ─── Event colours by behaviour type ──────────────────────────────────────────

_BEHAVIOUR_COLOR_BGR: dict[str, tuple[int, int, int]] = {
    "loitering":       (  0,   0, 220),   # red
    "restricted_entry":(  0,  90, 220),   # red-orange
    "abandoned_bag":   (  0, 165, 255),   # orange
    "storefront_dwell":(  0, 200, 100),   # green (analytics, not an alert)
    "running":         (220, 120,   0),   # blue (stretch)
    "crowd_buildup":   (180,   0, 180),   # purple (stretch)
}

_DEFAULT_EVENT_COLOR = (128, 128, 128)  # grey for unknown types


def event_color(behaviour: str) -> tuple[int, int, int]:
    """Return a BGR colour associated with a behaviour type."""
    return _BEHAVIOUR_COLOR_BGR.get(behaviour, _DEFAULT_EVENT_COLOR)


# ─── Time formatting ──────────────────────────────────────────────────────────

def fmt_seconds(seconds: float) -> str:
    """
    Format a float number of seconds as MM:SS or HH:MM:SS.

    Examples:
        fmt_seconds(4.8)   → '00:04'
        fmt_seconds(75.2)  → '01:15'
        fmt_seconds(3661)  → '1:01:01'
    """
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ─── Config helpers ───────────────────────────────────────────────────────────

def get_zone_threshold(zone_cfg: dict, zone_type: str, global_cfg: dict) -> float:
    """
    Return the effective dwell threshold for a zone.

    Zone-level overrides take precedence over global config values.

    Args:
        zone_cfg:   The dict for a single zone from config.yaml['zones']
        zone_type:  'loiter' | 'storefront'
        global_cfg: The full config dict

    Returns:
        Threshold in seconds as a float.
    """
    key_map = {
        "loiter":     "loiter_seconds",
        "storefront": "storefront_dwell_seconds",
    }
    key = key_map.get(zone_type, "loiter_seconds")
    # Per-zone override wins; fall back to global; fall back to sensible default
    return float(zone_cfg.get(key, global_cfg.get(key, 30)))
