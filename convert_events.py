"""
convert_events.py  –  Converts P2's events.json format to P3's expected format.

Usage:
    python convert_events.py --input outputs/events_p2.json --output outputs/events.json

P2 format:
    {"track_id": 1, "event_type": "Zone Trespassing", "timestamp": 0.0, "description": "..."}

P3 expected format:
    {"entity_id": 1, "behaviour": "restricted_entry", "start_s": 0.0, "end_s": 5.0,
     "zone": null, "evidence_frame": 0, "confidence": 0.9, "why": "..."}
"""

import argparse
import json
from pathlib import Path

# Map P2's event_type strings to P3's behaviour keys
_EVENT_TYPE_MAP = {
    "zone trespassing":  "restricted_entry",
    "loitering":         "loitering",
    "abandoned bag":     "abandoned_bag",
    "storefront dwell":  "storefront_dwell",
    "running":           "running",
    "crowd buildup":     "crowd_buildup",
}

# Default event duration in seconds when end_s is unknown
_DEFAULT_DURATION = 5.0

# Default confidence when not provided
_DEFAULT_CONFIDENCE = 0.85

# Assumed FPS for converting timestamp to frame number
_FPS = 30.0


def convert(input_path: str, output_path: str) -> None:
    with open(input_path, encoding="utf-8") as f:
        p2_events = json.load(f)

    p3_events = []
    for evt in p2_events:
        event_type_raw = evt.get("event_type", "").strip().lower()
        behaviour = _EVENT_TYPE_MAP.get(event_type_raw, event_type_raw.replace(" ", "_"))

        start_s = float(evt.get("timestamp", 0.0))
        end_s = float(evt.get("end_timestamp", start_s + _DEFAULT_DURATION))
        evidence_frame = int(start_s * _FPS)

        p3_events.append({
            "entity_id":      evt.get("track_id", 0),
            "behaviour":      behaviour,
            "zone":           evt.get("zone", None),
            "start_s":        start_s,
            "end_s":          end_s,
            "evidence_frame": evidence_frame,
            "confidence":     float(evt.get("confidence", _DEFAULT_CONFIDENCE)),
            "why":            evt.get("description", ""),
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(p3_events, f, indent=2)

    print(f"Converted {len(p3_events)} events → {output_path}")
    for e in p3_events:
        print(f"  #{e['entity_id']}  {e['behaviour']}  @ {e['start_s']}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert P2 events.json to P3 format")
    parser.add_argument("--input",  default="outputs/events_p2.json", help="P2's events.json")
    parser.add_argument("--output", default="outputs/events.json",    help="P3-compatible output")
    args = parser.parse_args()
    convert(args.input, args.output)
