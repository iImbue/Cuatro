# HackNex PS07 – Mall Vision System

Autonomous Vision & Behaviour Understanding for mall safety, security, and store analytics.

**HackNex 2026 Internal Qualifier · Karunya Institute of Technology and Sciences · Division of CSE**

---

## What it does

Processes CCTV-style video of a mall corridor/atrium and:

1. **Detects and tracks** people and bags (YOLO + ByteTrack) with stable IDs across occlusion
2. **Understands behaviour** — not just "there's a person", but *what they are doing*
3. **Flags events** with precise entity IDs and timestamps:
   - 🔴 **Loitering** — person stays in a zone beyond a configurable threshold
   - 🔴 **Restricted-area entry** — foot-point crosses a forbidden polygon
   - 🟠 **Abandoned bag** — bag stationary and unattended for T seconds after owner leaves
   - 🟢 **Storefront dwell** — analytics event (not an alert), tracks browsing time
4. **Outputs** an annotated video, evidence thumbnails, events CSV, and a Streamlit dashboard

---

## Tech stack

| Component | Library / Model |
|-----------|----------------|
| Detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (`yolov8m.pt`) |
| Tracking | ByteTrack via `model.track(..., tracker="bytetrack.yaml", persist=True)` |
| Zone geometry | OpenCV `cv2.pointPolygonTest` |
| Rendering | OpenCV |
| Dashboard | Streamlit |
| Config | PyYAML |

COCO classes used: `person` (0), `backpack` (24), `handbag` (26), `suitcase` (28).

---

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. GPU recommended (RTX 4050 or better).

---

## Configure

Edit `config.yaml`:
- Set zone polygons to match your footage resolution (default reference: 1280×720)
- Adjust thresholds (`loiter_seconds`, `abandoned_bag_seconds`, etc.)
- Change `model: yolov8n.pt` for CPU-only machines

---

## Run

```bash
# Full pipeline (detect → track → events → render)
python run.py --video data/clip1.mp4

# Skip detection, use pre-built tracks
python run.py --video data/clip1.mp4 --tracks-json outputs/tracks.json

# Skip rendering (faster for debugging events)
python run.py --video data/clip1.mp4 --no-render

# Dashboard
streamlit run app.py

# P3 standalone render test (uses mock data, no real video needed)
python -m src.render --video data/clip1.mp4 --tracks data/mock_tracks.json --events data/mock_events.json
```

---

## Data pipeline

```
video
  └─► [A] src/tracker.py   → detect people + bags (YOLO), track IDs (ByteTrack)
           └─► tracks.json
                └─► [B] src/features.py  → zones, dwell, speed, person-bag association
                         └─► features.json
                              └─► [C] src/events.py  → behaviour rules + thresholds
                                       └─► events.json
                                            └─► [D] src/render.py  → annotated video
                                                     app.py        → Streamlit dashboard
                                                     eval.py       → precision / recall
```

Each stage reads/writes JSON so team members can develop in parallel.

---

## Reproduce results

```bash
git clone <repo-url>
cd hacknex-ps07
pip install -r requirements.txt
python run.py --video data/clip_loitering.mp4
streamlit run app.py
```

Sample input/output files are in `data/` (mock) and `outputs/` (generated).

---

## Evaluation

```bash
python eval.py --events outputs/events.json --gt data/ground_truth.json
```

Reports precision, recall, and median timestamp error per behaviour class.

---

## MVP scope vs stretch goals

### MVP (must work for demo)
- [x] Person + bag detection and tracking
- [x] Loitering detection
- [x] Restricted-area entry detection
- [x] Abandoned bag detection
- [x] Storefront dwell (analytics)
- [x] Annotated video output
- [x] Evidence thumbnails
- [x] Streamlit dashboard
- [x] Eval script

### Stretch (only if MVP is stable)
- [ ] Running detection (speed normalised by bbox height)
- [ ] Crowd build-up count per zone
- [ ] Counter-flow detection
- [ ] Pose-based fall detection
- [ ] VLM/CLIP caption on flagged clips

---

## Footage

- Self-staged clips filmed on campus / in a mall with consent from all subjects
- Supplemented with public datasets (MOT17, VIRAT) — licences declared below
- No face recognition is performed anywhere in the pipeline

---

## External resources declared

| Resource | Type | Licence |
|----------|------|---------|
| YOLOv8 (`yolov8m.pt`) | Pretrained model | AGPL-3.0 |
| ByteTrack | Tracking algorithm | MIT |
| MOT17 | Supplemental dataset | Non-commercial research |
| VIRAT | Supplemental dataset | Creative Commons |

---

## Known limitations

- Tracking is stable for 3–8 people; ID switches increase in dense crowds
- Small bags (handbag class) may miss detections at low resolution; use `yolov8m` or higher
- Zone polygons are hardcoded for the reference resolution — rescale for other cameras
- Glass floors can cause ghost detections; mitigated by minimum track length filter

---

## Team

| Role | Responsibility |
|------|---------------|
| P1 Vision | Detection + tracking, tracks.json |
| P2 Behaviour | Feature engine, event rules, events.json |
| P3 Output/UI | Rendering, dashboard, evidence thumbnails, abandoned-bag logic |
| P4 Data + Docs | Footage, ground truth, eval.py, README, demo |
