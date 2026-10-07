"""
src/features.py  –  P2 Feature Engine

Responsibilities
----------------
1.  Zone membership  – for every track detection, test whether the entity's
    foot-point lies inside each configured zone polygon.
2.  Dwell time       – accumulate per-entity per-zone presence time (seconds).
3.  Speed            – estimate pixels-per-second movement, normalised by bbox
    height (body-heights / second) so it is camera-distance independent.
4.  Person↔bag assoc – link each bag detection to the nearest visible person
    at the same frame, within a configurable maximum distance.

Input  : tracks.json   (list of {frame, t, id, class, conf, bbox})
Output : features.json (one record per entity, not per frame)

Standalone usage
----------------
    python -m src.features --tracks outputs/tracks.json --output outputs/features.json
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils import foot_point, load_json, save_json, scale_polygon

# Reference frame size used when zone polygons were defined in config.yaml
_REF_WH = (1280, 720)

# COCO bag classes
_BAG_CLASSES = {"backpack", "handbag", "suitcase"}


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _point_in_polygon(point: tuple[int, int], polygon: list[tuple[int, int]]) -> bool:
    """Return True if *point* is inside *polygon* (using OpenCV)."""
    pts = np.array(polygon, dtype=np.float32)
    result = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), measureDist=False)
    return result >= 0


def _euclidean(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _bbox_height(bbox: list[float]) -> float:
    return max(bbox[3] - bbox[1], 1.0)


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


# ─── Zone polygon preparation ─────────────────────────────────────────────────

def _prepare_zones(
    zones_cfg: dict,
    frame_wh: tuple[int, int] | None = None,
) -> dict[str, dict]:
    """
    Return a dict of zone_name → {type, polygon (scaled pts), raw_cfg}.

    If *frame_wh* is None, polygons are kept in reference coordinates.
    """
    prepared: dict[str, dict] = {}
    dst = frame_wh or _REF_WH
    for name, zcfg in zones_cfg.items():
        raw_pts = zcfg.get("polygon", [])
        if not raw_pts:
            continue
        scaled = scale_polygon(raw_pts, _REF_WH, dst)
        prepared[name] = {
            "type":    zcfg.get("type", "loiter"),
            "polygon": scaled,
            "cfg":     zcfg,
        }
    return prepared


# ─── Per-frame grouping ───────────────────────────────────────────────────────

def _group_by_frame(tracks: list[dict]) -> dict[int, list[dict]]:
    idx: dict[int, list[dict]] = defaultdict(list)
    for rec in tracks:
        idx[rec["frame"]].append(rec)
    return idx


# ─── Person↔bag association ───────────────────────────────────────────────────

def _associate_bags(
    frame_records: list[dict],
    max_dist_bodyheights: float = 3.0,
) -> dict[int, int | None]:
    """
    For each bag in *frame_records*, find the nearest person (by centre distance).

    Returns bag_id → person_id (or None if no person within threshold).
    """
    persons = [r for r in frame_records if r["class"] == "person"]
    bags    = [r for r in frame_records if r["class"] in _BAG_CLASSES]

    assoc: dict[int, int | None] = {}
    for bag in bags:
        bag_center = _bbox_center(bag["bbox"])
        best_pid: int | None = None
        best_dist = float("inf")

        for person in persons:
            p_center = _bbox_center(person["bbox"])
            dist_px = _euclidean(bag_center, p_center)
            # Normalise by person bbox height so scale is camera-independent
            body_h = _bbox_height(person["bbox"])
            dist_bh = dist_px / body_h
            if dist_bh < best_dist and dist_bh <= max_dist_bodyheights:
                best_dist = dist_bh
                best_pid = person["id"]

        assoc[bag["id"]] = best_pid

    return assoc


# ─── Main feature engine ──────────────────────────────────────────────────────

def run_features(
    tracks_path: str,
    cfg: dict,
    out_path: str,
) -> list[dict]:
    """
    Compute per-entity features from *tracks_path* and write to *out_path*.

    Args:
        tracks_path: Path to tracks.json
        cfg:         Loaded config.yaml dict
        out_path:    Destination for features.json

    Returns:
        The features list (also written to *out_path*).
    """
    tracks: list[dict] = load_json(tracks_path)
    if not tracks:
        print("[features] WARNING: tracks.json is empty — no features to compute")
        save_json([], out_path)
        return []

    zones_cfg: dict = cfg.get("zones", {})
    fps: float = cfg.get("video_fps", 25.0)
    max_dist_bh: float = float(cfg.get("owner_distance_bodyheights", 2.0))

    # Infer frame size from bbox extents (avoids needing to open the video)
    all_x2 = [r["bbox"][2] for r in tracks]
    all_y2 = [r["bbox"][3] for r in tracks]
    frame_w = int(max(all_x2)) + 1 if all_x2 else _REF_WH[0]
    frame_h = int(max(all_y2)) + 1 if all_y2 else _REF_WH[1]
    frame_wh = (frame_w, frame_h)

    zones = _prepare_zones(zones_cfg, frame_wh)
    frame_index = _group_by_frame(tracks)

    print(f"[features] {len(tracks)} detections, {len(zones)} zones, frame size ~{frame_w}×{frame_h}")

    # ── Per-entity state ───────────────────────────────────────────────────────
    # zone_entry[eid][zone_name] = timestamp when entity first entered this zone
    zone_entry:  dict[int, dict[str, float]] = defaultdict(dict)
    # zone_dwell[eid][zone_name] = total accumulated seconds in zone
    zone_dwell:  dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    # zone_visits[eid][zone_name] = list of (enter_t, exit_t) intervals
    zone_visits: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    # previous position for speed calculation
    prev_pos:  dict[int, tuple[float, float]] = {}
    prev_t:    dict[int, float]               = {}
    speed_samples: dict[int, list[float]]     = defaultdict(list)

    # bag→owner at each frame (carry forward last known)
    bag_owner: dict[int, int | None] = {}
    # last frame the owner was near the bag
    bag_owner_last_frame: dict[int, int] = {}
    # first/last frame of each entity
    entity_first_frame: dict[int, int] = {}
    entity_last_frame:  dict[int, int] = {}
    entity_class:       dict[int, str] = {}

    # Per-entity zone membership per frame (for restricted entry detection)
    entity_zone_frames: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    sorted_frames = sorted(frame_index.keys())

    for frame_num in sorted_frames:
        records = frame_index[frame_num]
        t = frame_num / fps

        # ── Bag-person association this frame ──────────────────────────────────
        frame_assoc = _associate_bags(records, max_dist_bh)
        for bag_id, owner_id in frame_assoc.items():
            if owner_id is not None:
                bag_owner[bag_id] = owner_id
                bag_owner_last_frame[bag_id] = frame_num
            elif bag_id not in bag_owner:
                bag_owner[bag_id] = None

        for rec in records:
            eid   = rec["id"]
            bbox  = rec["bbox"]
            cls   = rec["class"]
            fp    = foot_point(bbox)   # bottom-centre

            entity_class[eid] = cls

            if eid not in entity_first_frame:
                entity_first_frame[eid] = frame_num
            entity_last_frame[eid] = frame_num

            # ── Zone membership ────────────────────────────────────────────────
            for zone_name, zone in zones.items():
                inside = _point_in_polygon(fp, zone["polygon"])

                if inside:
                    entity_zone_frames[eid][zone_name].append(frame_num)
                    if zone_name not in zone_entry[eid]:
                        # Just entered this zone
                        zone_entry[eid][zone_name] = t
                else:
                    if zone_name in zone_entry[eid]:
                        # Just left — close the interval
                        enter_t = zone_entry[eid].pop(zone_name)
                        duration = t - enter_t
                        zone_dwell[eid][zone_name] += duration
                        zone_visits[eid][zone_name].append((enter_t, t))

            # ── Speed (body-heights / second) ──────────────────────────────────
            center = _bbox_center(bbox)
            body_h = _bbox_height(bbox)
            if eid in prev_pos and eid in prev_t:
                dt = t - prev_t[eid]
                if dt > 0:
                    dpx = _euclidean(center, prev_pos[eid])
                    speed_bhs = (dpx / body_h) / dt
                    speed_samples[eid].append(speed_bhs)
            prev_pos[eid] = center
            prev_t[eid]   = t

    # ── Close any still-open zone intervals at end of video ───────────────────
    last_t = (sorted_frames[-1] if sorted_frames else 0) / fps
    for eid, open_zones in zone_entry.items():
        for zone_name, enter_t in open_zones.items():
            duration = last_t - enter_t
            zone_dwell[eid][zone_name] += duration
            zone_visits[eid][zone_name].append((enter_t, last_t))

    # ── Build feature records ──────────────────────────────────────────────────
    all_entity_ids = sorted(
        set(entity_first_frame.keys()),
        key=lambda x: entity_first_frame[x],
    )

    features: list[dict] = []
    for eid in all_entity_ids:
        cls = entity_class.get(eid, "person")
        first_f = entity_first_frame[eid]
        last_f  = entity_last_frame[eid]
        first_t = first_f / fps
        last_t_e = last_f / fps

        speeds = speed_samples.get(eid, [])
        avg_speed = round(sum(speeds) / len(speeds), 3) if speeds else 0.0
        max_speed = round(max(speeds), 3) if speeds else 0.0

        # Dwell summary per zone
        dwell_summary: dict[str, dict] = {}
        for zone_name in zones:
            total_dwell = round(zone_dwell[eid].get(zone_name, 0.0), 2)
            visits = zone_visits[eid].get(zone_name, [])
            # Also check open interval was already closed above
            frames_in_zone = entity_zone_frames[eid].get(zone_name, [])
            dwell_summary[zone_name] = {
                "dwell_s":       total_dwell,
                "visit_count":   len(visits),
                "visits":        [(round(s, 2), round(e, 2)) for s, e in visits],
                "entry_frames":  frames_in_zone[:3],  # first 3 frames inside (for evidence)
            }

        # Bag-specific fields
        owner_id = bag_owner.get(eid) if cls in _BAG_CLASSES else None
        owner_last_frame = bag_owner_last_frame.get(eid) if cls in _BAG_CLASSES else None
        owner_last_t = round(owner_last_frame / fps, 2) if owner_last_frame is not None else None

        features.append({
            "entity_id":        eid,
            "class":            cls,
            "first_frame":      first_f,
            "last_frame":       last_f,
            "first_t":          round(first_t, 2),
            "last_t":           round(last_t_e, 2),
            "avg_speed_bhs":    avg_speed,
            "max_speed_bhs":    max_speed,
            "zone_dwell":       dwell_summary,
            # Bag fields (null for persons)
            "owner_id":         owner_id,
            "owner_last_t":     owner_last_t,
        })

    print(f"[features] Computed features for {len(features)} entities")
    save_json(features, out_path)
    return features


# ─── CLI entry point ──────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="P2 feature engine (standalone)")
    parser.add_argument("--tracks", default="outputs/tracks.json")
    parser.add_argument("--output", default="outputs/features.json")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    import yaml  # noqa: PLC0415
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_features(args.tracks, cfg, args.output)


if __name__ == "__main__":
    _cli()
