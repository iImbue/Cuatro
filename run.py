"""
run.py  –  HackNex PS07 entry point
Usage:
    python run.py --video data/clip1.mp4
    python run.py --video data/clip1.mp4 --config config.yaml --no-render
    python run.py --video data/clip1.mp4 --tracks-json outputs/tracks.json
"""

import argparse
import sys
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="HackNex PS07 – Mall Vision System")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip rendering annotated output video (faster for debugging)",
    )
    parser.add_argument(
        "--tracks-json",
        default=None,
        help="Skip detection/tracking and load a pre-built tracks.json instead",
    )
    args = parser.parse_args()

    # ── Validate inputs ────────────────────────────────────────────────────────
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(str(config_path))
    out_dir = Path(cfg.get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.get("thumbnail_dir", "outputs/thumbnails")).mkdir(parents=True, exist_ok=True)

    print(f"[run] Config : {config_path}")
    print(f"[run] Model  : {cfg.get('model')}")
    print(f"[run] Video  : {video_path}")

    # ── Stage A: Detect + Track  (P1 owns src/tracker.py) ─────────────────────
    tracks_path = Path(args.tracks_json) if args.tracks_json else out_dir / "tracks.json"
    if args.tracks_json:
        print(f"[run] Stage A – skipped, using tracks from {tracks_path}")
    else:
        print("[run] Stage A – detect + track")
        from src.tracker import run_tracker  # noqa: PLC0415
        run_tracker(str(video_path), cfg, str(tracks_path))

    # ── Stage B: Feature Engine  (P2 owns src/features.py) ────────────────────
    features_path = out_dir / "features.json"
    print("[run] Stage B – feature engine")
    from src.features import run_features  # noqa: PLC0415
    run_features(str(tracks_path), cfg, str(features_path))

    # ── Stage C: Behaviour / Event Engine  (P2 owns src/events.py) ───────────
    events_path = out_dir / "events.json"
    print("[run] Stage C – behaviour engine")
    from src.events import run_events  # noqa: PLC0415
    run_events(str(features_path), cfg, str(events_path))

    # ── Stage D: Render  (P3 owns src/render.py) ──────────────────────────────
    if not args.no_render:
        print("[run] Stage D – rendering annotated video …")
        from src.render import render_video  # noqa: PLC0415

        stem = video_path.stem
        suffix = cfg.get("annotated_suffix", "_annotated")
        out_video = out_dir / f"{stem}{suffix}.mp4"
        render_video(
            video_path=str(video_path),
            tracks_path=str(tracks_path),
            events_path=str(events_path),
            config=cfg,
            output_path=str(out_video),
        )
        print(f"[run] Annotated video → {out_video}")
    else:
        print("[run] Stage D – render skipped (--no-render flag)")

    print("\n[run] Pipeline complete.")
    print(f"      tracks  → {tracks_path}")
    print(f"      events  → {events_path}")
    print("\n[run] View dashboard:  streamlit run app.py")


if __name__ == "__main__":
    main()
