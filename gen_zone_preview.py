"""
gen_zone_preview.py  –  Renders zone polygons onto reference frames and saves previews.
Run once after rezoning to verify zone placement visually.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT   = Path(__file__).parent
CONFIG = ROOT / "config.yaml"
OUT    = ROOT / "outputs"

# BGR colours per zone type
TYPE_COLORS = {
    "loiter":        (0,   200, 255),   # amber
    "restricted":    (0,   0,   220),   # red
    "storefront":    (50,  220,  50),   # green
    "abandoned_bag": (255, 100,   0),   # blue-ish
}
FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_zones(frame, zones):
    h, w = frame.shape[:2]
    for zname, zdata in zones.items():
        pts    = np.array(zdata["polygon"], dtype="int32")
        ztype  = zdata.get("type", "loiter")
        color  = TYPE_COLORS.get(ztype, (200, 200, 200))

        # Filled translucent overlay
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)

        # Border
        cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)

        # Corner dots
        for pt in pts:
            cv2.circle(frame, tuple(pt), 5, color, -1)
            cv2.circle(frame, tuple(pt), 7, (255,255,255), 1)

        # Label at centroid
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        label = f"{zname} [{ztype}]"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.rectangle(frame, (cx - tw//2 - 4, cy - th - 6),
                              (cx + tw//2 + 4, cy + 4), (20,20,20), -1)
        cv2.putText(frame, label, (cx - tw//2, cy - 2),
                    FONT, 0.5, color, 1, cv2.LINE_AA)
    return frame


def annotate_frame_from_video(video_path, frame_no, zones, out_path):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  [WARN] Could not read frame {frame_no} from {video_path}")
        return
    frame = draw_zones(frame, zones)
    # Header bar
    cv2.rectangle(frame, (0,0), (frame.shape[1], 32), (15,15,15), -1)
    cv2.putText(frame, f"Zone Preview  |  frame {frame_no}  |  {out_path.name}",
                (8, 22), FONT, 0.55, (220,220,220), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), frame)
    print(f"  Saved: {out_path}")


def main():
    if not CONFIG.exists():
        sys.exit(f"[ERROR] config.yaml not found at {CONFIG}")

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    zones = cfg.get("zones", {})
    if not zones:
        sys.exit("[ERROR] No zones found in config.yaml")

    video = ROOT / "data" / "clip1.mp4"
    if not video.exists():
        sys.exit(f"[ERROR] Video not found: {video}")

    OUT.mkdir(exist_ok=True)

    ref_frames = [165, 713, 2378]
    print(f"Rendering zones onto {len(ref_frames)} reference frames...")

    for fn in ref_frames:
        out_path = OUT / f"rezone_frame{fn}_zones.jpg"
        annotate_frame_from_video(video, fn, zones, out_path)

    # Also write a combined 3-panel summary image
    panels = []
    for fn in ref_frames:
        p = OUT / f"rezone_frame{fn}_zones.jpg"
        if p.exists():
            img = cv2.imread(str(p))
            img = cv2.resize(img, (853, 480))
            panels.append(img)

    if len(panels) == 3:
        combined = np.hstack(panels)
        summary_path = OUT / "rezone_summary.jpg"
        cv2.imwrite(str(summary_path), combined)
        print(f"  Summary panel: {summary_path}")

    print("\nDone. Zone polygons in config.yaml:")
    for zname, zdata in zones.items():
        print(f"  {zname} ({zdata['type']}): {zdata['polygon']}")


if __name__ == "__main__":
    main()
