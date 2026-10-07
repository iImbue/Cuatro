"""
src/tracker.py  –  P1's core module.

Responsibilities
----------------
1.  Load the input video with OpenCV.
2.  Run YOLOv8 + ByteTrack on every frame, filtering to COCO classes:
        person (0), backpack (24), handbag (26), suitcase (28)
3.  Write a flat-list tracks.json where every record is one detection:
        {"frame": 42, "t": 1.68, "id": 3, "class": "person",
         "conf": 0.87, "bbox": [x1, y1, x2, y2]}
4.  Expose run_tracker(video_path, cfg, out_path) so run.py can call it.

Standalone usage
----------------
    python -m src.tracker --video data/clip1.mp4 --output outputs/tracks.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils import save_json

# COCO class IDs the pipeline cares about
_TARGET_CLASSES = {0: "person", 24: "backpack", 26: "handbag", 28: "suitcase"}


def run_tracker(
    video_path: str,
    cfg: dict,
    out_path: str,
) -> list[dict]:
    """
    Detect and track people + bags in *video_path*, write results to *out_path*.

    Args:
        video_path: Path to the input video file.
        cfg:        Loaded config.yaml dict.
        out_path:   Destination path for tracks.json.

    Returns:
        The tracks list (also written to *out_path*).
    """
    # Import here so the module can be imported without ultralytics installed
    # (useful for unit-testing utils / render without a GPU)
    from ultralytics import YOLO  # noqa: PLC0415
    import cv2  # noqa: PLC0415

    # ── Config ────────────────────────────────────────────────────────────────
    model_name   = cfg.get("model",       "yolov8m.pt")
    tracker_cfg  = cfg.get("tracker",     "bytetrack.yaml")
    device       = cfg.get("device",      "cpu")
    conf_person  = cfg.get("conf_person", 0.40)
    conf_bag     = cfg.get("conf_bag",    0.30)
    fallback_fps = cfg.get("video_fps",   25)

    # Use the lower bag confidence as the global threshold so YOLO sees bags;
    # we'll filter person detections to conf_person afterwards.
    global_conf = min(conf_person, conf_bag)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[tracker] Loading model : {model_name}")
    model = YOLO(model_name)

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[tracker] Cannot open video: {video_path}")

    fps         = cap.get(cv2.CAP_PROP_FPS) or fallback_fps
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(
        f"[tracker] Video  : {video_path}  "
        f"({frame_w}×{frame_h} @ {fps:.1f} fps, ~{total} frames)"
    )
    print(f"[tracker] Device : {device}  |  conf_person={conf_person}  conf_bag={conf_bag}")

    # ── Run tracking ──────────────────────────────────────────────────────────
    # model.track streams results frame-by-frame without loading all into memory
    results_gen = model.track(
        source=video_path,
        tracker=tracker_cfg,
        persist=True,           # ByteTrack keeps IDs across frames
        conf=global_conf,
        device=device,
        classes=list(_TARGET_CLASSES.keys()),
        stream=True,            # generator — memory efficient
        verbose=False,
    )

    tracks: list[dict] = []
    frame_num = 0

    for result in results_gen:
        boxes = result.boxes

        if boxes is not None and boxes.id is not None:
            # boxes.id is None when ByteTrack hasn't assigned IDs yet
            ids    = boxes.id.cpu().tolist()
            cls_ids = boxes.cls.cpu().tolist()
            confs  = boxes.conf.cpu().tolist()
            xyxys  = boxes.xyxy.cpu().tolist()

            for tid, cls_id, conf, xyxy in zip(ids, cls_ids, confs, xyxys):
                cls_id = int(cls_id)
                if cls_id not in _TARGET_CLASSES:
                    continue

                class_name = _TARGET_CLASSES[cls_id]

                # Per-class confidence filter
                min_conf = conf_person if class_name == "person" else conf_bag
                if conf < min_conf:
                    continue

                x1, y1, x2, y2 = [round(v, 1) for v in xyxy]
                tracks.append(
                    {
                        "frame": frame_num,
                        "t":     round(frame_num / fps, 3),
                        "id":    int(tid),
                        "class": class_name,
                        "conf":  round(conf, 3),
                        "bbox":  [x1, y1, x2, y2],
                    }
                )

        frame_num += 1

        if frame_num % 250 == 0:
            pct = 100 * frame_num / max(total, 1)
            print(f"[tracker]   frame {frame_num}/{total}  ({pct:.0f}%)")

    print(f"[tracker] Done — {len(tracks)} detections across {frame_num} frames")

    # ── Write output ──────────────────────────────────────────────────────────
    save_json(tracks, out_path)
    return tracks


# ─── CLI entry point ──────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO+ByteTrack tracker (P1 standalone)")
    parser.add_argument("--video",  required=True, help="Input video path")
    parser.add_argument("--output", default="outputs/tracks.json", help="Output tracks.json path")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--device", default=None, help="Override device (0, cpu, cuda:0 …)")
    args = parser.parse_args()

    import yaml  # noqa: PLC0415
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        cfg["device"] = args.device

    run_tracker(
        video_path=args.video,
        cfg=cfg,
        out_path=args.output,
    )


if __name__ == "__main__":
    _cli()
