"""
convert_tracks.py  –  Converts P1's tracks.json format to P3's expected format.

Usage:
    python convert_tracks.py --input outputs/tracks_p1.json --output outputs/tracks.json

P1 format (frame number as key):
    {"0": [{"track_id": 1, "bbox": [150, 150, 200, 250]}], "30": [...]}

P3 expected format (flat list):
    [{"frame": 0, "t": 0.0, "id": 1, "class": "person", "conf": 0.9, "bbox": [150, 150, 200, 250]}]
"""

import argparse
import json
from pathlib import Path

# Default values when P1 doesn't provide them
_DEFAULT_CLASS = "person"
_DEFAULT_CONF  = 0.85
_DEFAULT_FPS   = 25.0


def convert(input_path: str, output_path: str, fps: float = _DEFAULT_FPS) -> None:
    with open(input_path, encoding="utf-8") as f:
        p1_data = json.load(f)

    tracks: list[dict] = []

    # Handle two possible P1 formats:
    # Format A: {"0": [{"track_id": 1, "bbox": [...]}], "30": [...]}
    # Format B: flat list [{"frame": 0, "track_id": 1, "bbox": [...]}]

    if isinstance(p1_data, dict):
        # Format A — frame number is the key
        for frame_str, detections in p1_data.items():
            frame_num = int(frame_str)
            t = round(frame_num / fps, 3)
            for det in detections:
                tracks.append({
                    "frame": frame_num,
                    "t":     t,
                    "id":    det.get("track_id", det.get("id", 0)),
                    "class": det.get("class", det.get("label", _DEFAULT_CLASS)),
                    "conf":  round(float(det.get("conf", det.get("confidence", _DEFAULT_CONF))), 3),
                    "bbox":  det.get("bbox", det.get("box", [0, 0, 0, 0])),
                })

    elif isinstance(p1_data, list):
        # Format B — already a flat list, just normalise field names
        for det in p1_data:
            frame_num = int(det.get("frame", det.get("frame_id", 0)))
            tracks.append({
                "frame": frame_num,
                "t":     round(frame_num / fps, 3),
                "id":    det.get("track_id", det.get("id", 0)),
                "class": det.get("class", det.get("label", _DEFAULT_CLASS)),
                "conf":  round(float(det.get("conf", det.get("confidence", _DEFAULT_CONF))), 3),
                "bbox":  det.get("bbox", det.get("box", [0, 0, 0, 0])),
            })
    else:
        raise ValueError(f"Unrecognised P1 tracks format: {type(p1_data)}")

    # Sort by frame for cleaner output
    tracks.sort(key=lambda r: (r["frame"], r["id"]))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tracks, f, indent=2)

    print(f"Converted {len(tracks)} detections → {output_path}")
    ids = sorted({r["id"] for r in tracks})
    print(f"  Track IDs found : {ids}")
    classes = sorted({r['class'] for r in tracks})
    print(f"  Classes found   : {classes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert P1 tracks.json to P3 format")
    parser.add_argument("--input",  default="outputs/tracks_p1.json", help="P1's tracks.json")
    parser.add_argument("--output", default="outputs/tracks.json",    help="P3-compatible output")
    parser.add_argument("--fps",    default=25.0, type=float,         help="Video FPS (default 25)")
    args = parser.parse_args()
    convert(args.input, args.output, args.fps)
