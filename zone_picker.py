"""
zone_picker.py  –  Interactive zone polygon editor for HackNex PS07
====================================================================
Usage:
    python zone_picker.py                       # uses data/clip1.mp4 by default
    python zone_picker.py --video path/to/clip  # custom video

Controls:
    [1] [2] [3]   – switch active zone (Loiter_Zone / Ramp_Restricted / Building_Entry)
    Left-click    – add a point to the active zone
    Right-click   – remove the last point from the active zone
    [c]           – clear all points for the active zone
    [s]           – save zones to config.yaml and export a preview image
    [f] / [b]     – jump forward / backward 300 frames
    [r]           – reset all zones (start over)
    [q] / ESC     – quit without saving
"""

import argparse
import copy
import sys
from pathlib import Path

import cv2
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
CONFIG     = ROOT / "config.yaml"
DEFAULT_VID = ROOT / "data" / "clip1.mp4"
PREVIEW_OUT = ROOT / "outputs" / "rezone_preview.jpg"

# ── Zone colour map (BGR) ──────────────────────────────────────────────────────
ZONE_COLORS = {
    "Loiter_Zone":      (0,   200, 255),   # amber
    "Ramp_Restricted":  (0,   0,   255),   # red
    "Building_Entry":   (0,   255, 120),   # green
}
POINT_RADIUS = 6
LINE_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── State ──────────────────────────────────────────────────────────────────────
zone_names  = ["Loiter_Zone", "Ramp_Restricted", "Building_Entry"]
active_idx  = 0                          # index into zone_names
polygons    = {z: [] for z in zone_names}


def draw_overlay(frame_in, status_msg=""):
    """Return a copy of frame_in with all zone polygons + HUD drawn."""
    frame = frame_in.copy()
    h, w  = frame.shape[:2]

    for i, zname in enumerate(zone_names):
        pts   = polygons[zname]
        color = ZONE_COLORS[zname]
        is_active = (i == active_idx)

        # Draw filled translucent polygon if ≥ 3 points
        if len(pts) >= 3:
            import numpy as np
            overlay = frame.copy()
            cv2.fillPoly(overlay, [np.array(pts, dtype="int32")], color)
            alpha = 0.25 if is_active else 0.15
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            cv2.polylines(frame, [np.array(pts, dtype="int32")],
                          True, color, LINE_THICKNESS + (1 if is_active else 0))

        # Draw points + connecting lines
        for j, pt in enumerate(pts):
            cv2.circle(frame, pt, POINT_RADIUS, color, -1)
            cv2.circle(frame, pt, POINT_RADIUS + 2, (255, 255, 255), 1)
            if j > 0:
                cv2.line(frame, pts[j-1], pt, color, LINE_THICKNESS)
        # Close the polygon preview line
        if len(pts) >= 2:
            cv2.line(frame, pts[-1], pts[0], color, 1)

        # Zone label
        if pts:
            label_pt = pts[0]
            cv2.putText(frame, zname, (label_pt[0]+8, label_pt[1]-8),
                        FONT, 0.5, color, 1, cv2.LINE_AA)

    # ── HUD ────────────────────────────────────────────────────────────────────
    hud_lines = [
        f"Active: [{active_idx+1}] {zone_names[active_idx]}  ({len(polygons[zone_names[active_idx]])} pts)",
        "[1/2/3] switch zone   [LClick] add   [RClick] undo   [c] clear",
        "[f/b] +/-300 frames   [s] save & quit   [r] reset all   [q] quit",
    ]
    if status_msg:
        hud_lines.insert(0, status_msg)

    box_h = 20 * (len(hud_lines) + 1)
    cv2.rectangle(frame, (0, 0), (w, box_h), (20, 20, 20), -1)
    for li, line in enumerate(hud_lines):
        color = (0, 255, 255) if li == 0 and status_msg else (220, 220, 220)
        cv2.putText(frame, line, (8, 18 + li*20), FONT, 0.48, color, 1, cv2.LINE_AA)

    return frame


def save_zones():
    """Merge new polygons into config.yaml and write preview image."""
    with open(CONFIG, "r") as f:
        cfg = yaml.safe_load(f)

    for zname, pts in polygons.items():
        if len(pts) >= 3 and zname in cfg.get("zones", {}):
            cfg["zones"][zname]["polygon"] = pts

    with open(CONFIG, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False, allow_unicode=True)

    return cfg


def mouse_callback(event, x, y, flags, param):
    global active_idx
    zone = zone_names[active_idx]
    if event == cv2.EVENT_LBUTTONDOWN:
        polygons[zone].append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if polygons[zone]:
            polygons[zone].pop()


def main():
    global active_idx

    parser = argparse.ArgumentParser(description="Interactive zone picker")
    parser.add_argument("--video", default=str(DEFAULT_VID),
                        help="Path to video file (default: data/clip1.mp4)")
    parser.add_argument("--frame", type=int, default=300,
                        help="Starting frame number (default: 300)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(f"[ERROR] Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = max(0, min(args.frame, total_frames - 1))

    # Pre-load current zones from config as starting points
    if CONFIG.exists():
        with open(CONFIG) as f:
            cfg = yaml.safe_load(f)
        for zname in zone_names:
            existing = cfg.get("zones", {}).get(zname, {}).get("polygon", [])
            polygons[zname] = [tuple(p) for p in existing]

    win = "Zone Picker  –  HackNex PS07"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.setMouseCallback(win, mouse_callback)

    status = ""
    ret = False

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            # Try to recover — go back a bit
            current_frame = max(0, current_frame - 10)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Cannot read frame, using blank canvas.")
                import numpy as np
                frame = np.zeros((720, 1280, 3), dtype="uint8")

        display = draw_overlay(frame, status)
        cv2.imshow(win, display)
        status = ""   # clear after one frame

        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):          # q / ESC — quit without saving
            print("Quitting without saving.")
            break

        elif key == ord('1'):
            active_idx = 0
        elif key == ord('2'):
            active_idx = 1
        elif key == ord('3'):
            active_idx = 2

        elif key == ord('c'):              # clear active zone
            polygons[zone_names[active_idx]].clear()

        elif key == ord('r'):              # reset all zones
            for z in zone_names:
                polygons[z].clear()
            status = "All zones reset."

        elif key == ord('f'):              # forward 300 frames
            current_frame = min(current_frame + 300, total_frames - 1)

        elif key == ord('b'):              # back 300 frames
            current_frame = max(current_frame - 300, 0)

        elif key == ord('s'):              # save
            # Validate — need ≥3 pts per zone (or keep old if untouched)
            issues = [z for z in zone_names if len(polygons[z]) < 3]
            if issues:
                status = f"WARNING: {', '.join(issues)} have <3 pts — keeping old values."
            cfg = save_zones()
            # Save preview
            preview = draw_overlay(frame, "SAVED to config.yaml")
            PREVIEW_OUT.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(PREVIEW_OUT), preview)
            print(f"\n[OK] Zones saved to {CONFIG}")
            print(f"[OK] Preview saved to {PREVIEW_OUT}")
            print("\nUpdated polygons:")
            for zname, pts in polygons.items():
                print(f"  {zname}: {pts}")
            status = "Saved! Press [q] to exit."

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
