"""
find_bag_scene.py  –  Scans clip1.mp4 with YOLOv8 to find frames where a bag
                      appears without a nearby person (likely abandoned bag scene).
Outputs candidate frame numbers + saves thumbnail images.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

ROOT       = Path(__file__).parent
VIDEO      = ROOT / "data" / "clip1.mp4"
MODEL_PATH = ROOT / "yolov8m.pt"
OUT        = ROOT / "outputs"

# COCO IDs
PERSON_ID  = 0
BAG_IDS    = {24, 26, 28}   # backpack, handbag, suitcase

SCAN_EVERY   = 15            # check every N frames (≈2× per second at 30fps)
CONF         = 0.30
OWNER_DIST   = 180           # px — bag centre closer than this to a person = attended
MAX_SAVES    = 10            # max candidate thumbnails to write


def box_centre(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5


def main():
    if not VIDEO.exists():
        sys.exit(f"[ERROR] {VIDEO} not found")

    model = YOLO(str(MODEL_PATH))
    cap   = cv2.VideoCapture(str(VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total} frames @ {fps:.1f} fps")
    print(f"Scanning every {SCAN_EVERY} frames...\n")

    OUT.mkdir(exist_ok=True)
    candidates = []
    saved      = 0

    frame_no = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF, verbose=False)[0]
        boxes   = results.boxes

        persons, bags = [], []
        for box in boxes:
            cls = int(box.cls[0])
            xyxy = box.xyxy[0].tolist()
            if cls == PERSON_ID:
                persons.append(box_centre(xyxy))
            elif cls in BAG_IDS:
                bags.append((box_centre(xyxy), cls, xyxy))

        # Check each bag — is it far from every person?
        for (bc, bcls, bxyxy) in bags:
            if not persons:
                # No person in frame at all — strong candidate
                alone = True
            else:
                alone = all(dist(bc, pc) > OWNER_DIST for pc in persons)

            if alone:
                ts = frame_no / fps
                candidates.append((frame_no, ts, bcls, bc))
                print(f"  [CANDIDATE] frame {frame_no:6d}  t={ts:6.1f}s  "
                      f"bag_cls={bcls}  centre=({bc[0]:.0f},{bc[1]:.0f})")

                if saved < MAX_SAVES:
                    # Draw annotation
                    ann = frame.copy()
                    x1,y1,x2,y2 = [int(v) for v in bxyxy]
                    cv2.rectangle(ann, (x1,y1), (x2,y2), (0,100,255), 3)
                    cv2.putText(ann, f"BAG ALONE  frame {frame_no}  t={ts:.1f}s",
                                (x1, max(y1-10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,100,255), 2)
                    for pc in persons:
                        cv2.circle(ann, (int(pc[0]),int(pc[1])), 6, (0,255,0), -1)
                    out_p = OUT / f"bag_candidate_frame{frame_no}.jpg"
                    cv2.imwrite(str(out_p), ann)
                    saved += 1

        frame_no += SCAN_EVERY
        if frame_no >= total:
            break

    cap.release()

    print(f"\n── Summary ─────────────────────────────────────────")
    print(f"Scanned {frame_no // SCAN_EVERY} sample frames")
    print(f"Found {len(candidates)} candidate abandoned-bag moments")
    if candidates:
        print("\nTop candidates (by frame):")
        for fn, ts, cls, ctr in candidates[:15]:
            print(f"  frame {fn:6d}  t={ts:6.1f}s  cls={cls}  centre={ctr}")
        best = candidates[len(candidates)//2]  # pick a middle one
        print(f"\nSuggested Abandoned_Bag_Zone frame: {best[0]}")
        print(f"Bag centre at that frame: {best[2]}")


if __name__ == "__main__":
    main()
