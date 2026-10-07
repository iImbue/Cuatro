"""
src/events.py  –  P2 Behaviour / Event Engine

Rules implemented
-----------------
1.  Loitering       – person dwells in a loiter/restricted zone beyond
                      `loiter_seconds` threshold.
2.  Restricted entry – person's foot-point enters a restricted polygon at any
                      point (fires immediately on first entry frame).
3.  Abandoned bag   – bag is stationary AND unattended (owner distance > N
                      body-heights) for `abandoned_bag_seconds`.
4.  Storefront dwell – person dwells in a storefront zone beyond
                      `storefront_dwell_seconds` (analytics, not alert).

Input  : features.json  (output of src/features.py)
Output : events.json    (list of behaviour event records, P3-compatible format)

Standalone usage
----------------
    python -m src.events \
        --features outputs/features.json \
        --output   outputs/events.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils import load_json, save_json


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _fps_from_features(features: list[dict]) -> float:
    """Rough FPS estimate from features (used only for evidence_frame calc)."""
    for feat in features:
        ft = feat.get("first_t", 0.0)
        ff = feat.get("first_frame", 0)
        if ft > 0 and ff > 0:
            return round(ff / ft, 1)
    return 25.0


def _evidence_frame(t: float, fps: float) -> int:
    return int(t * fps)


def _zone_active_at(zone: dict, t: float, fps: float) -> bool:
    """Return True if timestamp *t* (seconds) falls within zone's active_frames range."""
    af = zone.get("active_frames")
    if af is None:
        return True
    start_s = af[0] / fps
    end_s   = af[1] / fps
    return start_s <= t <= end_s


def _filter_visits(visits: list, zone: dict, fps: float) -> list:
    """
    Return only visits whose enter_t falls within the zone's active_frames window.
    Visits outside the scene window are ghost detections from the wrong scene.
    """
    af = zone.get("active_frames")
    if af is None:
        return visits
    start_s = af[0] / fps
    end_s   = af[1] / fps
    return [(enter, exit_) for enter, exit_ in visits if start_s <= enter <= end_s]


def _make_event(
    entity_id: int,
    behaviour: str,
    start_s: float,
    end_s: float,
    zone: str | None,
    confidence: float,
    why: str,
    fps: float,
    extra: dict | None = None,
) -> dict:
    evt: dict = {
        "entity_id":     entity_id,
        "behaviour":     behaviour,
        "zone":          zone,
        "start_s":       round(start_s, 2),
        "end_s":         round(end_s, 2),
        "evidence_frame": _evidence_frame(start_s, fps),
        "confidence":    round(confidence, 2),
        "why":           why,
    }
    if extra:
        evt.update(extra)
    return evt


# ─── Rule: Loitering ─────────────────────────────────────────────────────────

def _rule_loitering(
    feat: dict,
    zones_cfg: dict,
    threshold_s: float,
    fps: float,
) -> list[dict]:
    """
    Fire a loitering event for every visit to a loiter-type zone that exceeds
    the dwell threshold.
    """
    events: list[dict] = []
    eid = feat["entity_id"]

    if feat["class"] != "person":
        return events

    for zone_name, zone_info in zones_cfg.items():
        if zone_info.get("type") != "loiter":
            continue

        # Per-zone threshold overrides global
        thr = float(zone_info.get("loiter_seconds", threshold_s))
        zone_data = feat["zone_dwell"].get(zone_name, {})
        visits = zone_data.get("visits", [])
        visits = _filter_visits(visits, zone_info, fps)  # scene-aware gate

        for enter_t, exit_t in visits:
            duration = exit_t - enter_t
            if duration >= thr:
                confidence = min(0.99, 0.60 + 0.013 * (duration - thr))
                events.append(_make_event(
                    entity_id=eid,
                    behaviour="loitering",
                    start_s=enter_t,
                    end_s=exit_t,
                    zone=zone_name,
                    confidence=confidence,
                    why=(
                        f"Person #{eid} stayed in {zone_name} for "
                        f"{duration:.1f}s (threshold {thr:.0f}s)"
                    ),
                    fps=fps,
                ))

    return events


# ─── Rule: Restricted Entry ───────────────────────────────────────────────────

def _rule_restricted_entry(
    feat: dict,
    zones_cfg: dict,
    fps: float,
) -> list[dict]:
    """
    Fire a restricted_entry event the first time a person enters a restricted zone.
    """
    events: list[dict] = []
    eid = feat["entity_id"]

    if feat["class"] != "person":
        return events

    for zone_name, zone_info in zones_cfg.items():
        if zone_info.get("type") != "restricted":
            continue

        zone_data = feat["zone_dwell"].get(zone_name, {})
        visits = zone_data.get("visits", [])
        visits = _filter_visits(visits, zone_info, fps)  # scene-aware gate

        if not visits:
            continue

        # Fire once per entry (each visit = one entry)
        for enter_t, exit_t in visits:
            entry_frames = zone_data.get("entry_frames", [])
            ev_frame = entry_frames[0] if entry_frames else _evidence_frame(enter_t, fps)
            duration = exit_t - enter_t
            events.append({
                "entity_id":     eid,
                "behaviour":     "restricted_entry",
                "zone":          zone_name,
                "start_s":       round(enter_t, 2),
                "end_s":         round(exit_t, 2),
                "evidence_frame": ev_frame,
                "confidence":    0.95,
                "why": (
                    f"Person #{eid} foot-point entered restricted zone "
                    f"'{zone_name}' at t={enter_t:.1f}s "
                    f"(stayed {duration:.1f}s)"
                ),
            })

    return events


# ─── Rule: Abandoned Bag ─────────────────────────────────────────────────────

def _rule_abandoned_bag(
    feat: dict,
    all_features: list[dict],
    threshold_s: float,
    fps: float,
    zones_cfg: dict,
) -> list[dict]:
    """
    Fire an abandoned_bag event when:
      - entity is a bag class
      - its last known owner has not been near it for >= threshold_s seconds
      - the bag itself is still present (last_t > owner_last_t + threshold_s)
      - the bag's timestamps fall within an abandoned_bag zone's active_frames window
    """
    events: list[dict] = []

    bag_classes = {"backpack", "handbag", "suitcase"}
    if feat["class"] not in bag_classes:
        return events

    eid         = feat["entity_id"]
    owner_id    = feat.get("owner_id")
    owner_last_t = feat.get("owner_last_t")
    bag_last_t  = feat["last_t"]
    bag_first_t = feat["first_t"]

    # Bag must have had a known owner at some point
    if owner_id is None or owner_last_t is None:
        return events

    unattended_s = bag_last_t - owner_last_t

    if unattended_s >= threshold_s:
        abandon_start = owner_last_t

        # Scene-aware gate: check if abandon_start is inside any abandoned_bag zone's window
        in_active_scene = False
        zone_thr = threshold_s
        for zone_name, zone_info in zones_cfg.items():
            if zone_info.get("type") != "abandoned_bag":
                continue
            if _zone_active_at(zone_info, abandon_start, fps):
                in_active_scene = True
                zone_thr = float(zone_info.get("abandoned_bag_seconds", threshold_s))
                break

        if not in_active_scene:
            return events

        if unattended_s >= zone_thr:
            confidence = min(0.99, 0.65 + 0.015 * (unattended_s - zone_thr))
            events.append(_make_event(
                entity_id=eid,
                behaviour="abandoned_bag",
                start_s=abandon_start,
                end_s=bag_last_t,
                zone=None,
                confidence=confidence,
                why=(
                    f"Bag #{eid} ({feat['class']}) last seen with Person #{owner_id} "
                    f"at t={owner_last_t:.1f}s; unattended for {unattended_s:.1f}s "
                    f"(threshold {zone_thr:.0f}s)"
                ),
                fps=fps,
                extra={
                    "owner_id":          owner_id,
                    "owner_last_seen_s": owner_last_t,
                },
            ))

    return events


# ─── Rule: Storefront Dwell ───────────────────────────────────────────────────

def _rule_storefront_dwell(
    feat: dict,
    zones_cfg: dict,
    threshold_s: float,
    fps: float,
) -> list[dict]:
    """
    Fire a storefront_dwell analytics event for visits that exceed the threshold.
    """
    events: list[dict] = []
    eid = feat["entity_id"]

    if feat["class"] != "person":
        return events

    for zone_name, zone_info in zones_cfg.items():
        if zone_info.get("type") != "storefront":
            continue

        thr = float(zone_info.get("storefront_dwell_seconds", threshold_s))
        zone_data = feat["zone_dwell"].get(zone_name, {})
        visits = zone_data.get("visits", [])
        visits = _filter_visits(visits, zone_info, fps)  # scene-aware gate

        for enter_t, exit_t in visits:
            duration = exit_t - enter_t
            if duration >= thr:
                confidence = min(0.99, 0.70 + 0.015 * (duration - thr))
                events.append(_make_event(
                    entity_id=eid,
                    behaviour="storefront_dwell",
                    start_s=enter_t,
                    end_s=exit_t,
                    zone=zone_name,
                    confidence=confidence,
                    why=(
                        f"Person #{eid} browsed {zone_name} for "
                        f"{duration:.1f}s (threshold {thr:.0f}s)"
                    ),
                    fps=fps,
                ))

    return events


# ─── Main event engine ────────────────────────────────────────────────────────

def run_events(
    features_path: str,
    cfg: dict,
    out_path: str,
) -> list[dict]:
    """
    Apply all behaviour rules to *features_path* and write events to *out_path*.

    Args:
        features_path: Path to features.json
        cfg:           Loaded config.yaml dict
        out_path:      Destination for events.json

    Returns:
        The events list (also written to *out_path*).
    """
    features: list[dict] = load_json(features_path)
    if not features:
        print("[events] WARNING: features.json is empty — no events to generate")
        save_json([], out_path)
        return []

    zones_cfg = cfg.get("zones", {})
    fps = cfg.get("video_fps", 25.0)

    # Try to infer fps from features data
    inferred_fps = _fps_from_features(features)
    if inferred_fps > 0:
        fps = inferred_fps

    loiter_thr   = float(cfg.get("loiter_seconds",          30.0))
    abandon_thr  = float(cfg.get("abandoned_bag_seconds",   20.0))
    storefront_thr = float(cfg.get("storefront_dwell_seconds", 10.0))

    print(
        f"[events] {len(features)} entities | "
        f"loiter≥{loiter_thr}s | abandon≥{abandon_thr}s | "
        f"storefront≥{storefront_thr}s"
    )

    all_events: list[dict] = []

    for feat in features:
        all_events.extend(_rule_loitering(feat, zones_cfg, loiter_thr, fps))
        all_events.extend(_rule_restricted_entry(feat, zones_cfg, fps))
        all_events.extend(_rule_abandoned_bag(feat, features, abandon_thr, fps, zones_cfg))
        all_events.extend(_rule_storefront_dwell(feat, zones_cfg, storefront_thr, fps))

    # Sort by start time for clean output
    all_events.sort(key=lambda e: e["start_s"])

    print(f"[events] {len(all_events)} events generated:")
    for evt in all_events:
        print(
            f"  #{evt['entity_id']:>3}  {evt['behaviour']:<20}  "
            f"zone={str(evt.get('zone') or '-'):<20}  "
            f"t={evt['start_s']:.1f}–{evt['end_s']:.1f}s"
        )

    save_json(all_events, out_path)
    return all_events


# ─── CLI entry point ──────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="P2 behaviour/event engine (standalone)")
    parser.add_argument("--features", default="outputs/features.json")
    parser.add_argument("--output",   default="outputs/events.json")
    parser.add_argument("--config",   default="config.yaml")
    args = parser.parse_args()

    import yaml  # noqa: PLC0415
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_events(args.features, cfg, args.output)


if __name__ == "__main__":
    _cli()
