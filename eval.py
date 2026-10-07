"""
eval.py  –  HackNex PS07 Evaluation Script (P4)

Computes per-behaviour precision, recall, F1, and median timestamp error
by comparing the pipeline's events.json against a hand-labelled ground truth.

Ground truth format (data/ground_truth.json)
---------------------------------------------
A list of the same event schema used by events.json:
[
  {
    "entity_id": 1,
    "behaviour": "loitering",
    "start_s": 12.0,
    "end_s": 45.0,
    "zone": "ATM_Area"
  },
  ...
]

Matching rules
--------------
- A predicted event MATCHES a GT event if:
    * behaviour labels are identical
    * |pred.start_s - gt.start_s| <= timestamp_tolerance_s  (default 5 s)
  The closest match wins (greedy, by timestamp delta).
- Duplicate matches are not allowed (one GT event matches at most one prediction).

Usage
-----
    python eval.py --events outputs/events.json --gt data/ground_truth.json
    python eval.py --events outputs/events.json --gt data/ground_truth.json --tol 3
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


# ─── I/O ─────────────────────────────────────────────────────────────────────

def _load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[eval] File not found: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ─── Matching ─────────────────────────────────────────────────────────────────

def _match_events(
    predictions: list[dict],
    ground_truth: list[dict],
    tolerance_s: float,
) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """
    Greedy matching of predicted events to GT events by behaviour + timestamp.

    Returns:
        matched   – list of (pred, gt) pairs
        false_pos – predictions with no GT match  (False Positives)
        false_neg – GT events with no pred match  (False Negatives)
    """
    # Group GT by behaviour for efficient lookup
    gt_by_behaviour: dict[str, list[dict]] = defaultdict(list)
    for gt in ground_truth:
        gt_by_behaviour[gt["behaviour"]].append(gt)

    matched: list[tuple[dict, dict]] = []
    used_gt: set[int] = set()      # indices into ground_truth list

    # Sort predictions by start time so earlier preds get first pick
    preds_sorted = sorted(predictions, key=lambda e: e["start_s"])

    for pred in preds_sorted:
        beh = pred["behaviour"]
        best_gt_idx: int | None = None
        best_delta = float("inf")

        for i, gt in enumerate(ground_truth):
            if gt["behaviour"] != beh:
                continue
            if i in used_gt:
                continue
            delta = abs(pred["start_s"] - gt["start_s"])
            if delta <= tolerance_s and delta < best_delta:
                best_delta = delta
                best_gt_idx = i

        if best_gt_idx is not None:
            matched.append((pred, ground_truth[best_gt_idx]))
            used_gt.add(best_gt_idx)

    matched_pred_ids = {id(p) for p, _ in matched}
    false_pos = [p for p in predictions if id(p) not in matched_pred_ids]
    false_neg = [gt for i, gt in enumerate(ground_truth) if i not in used_gt]

    return matched, false_pos, false_neg


# ─── Per-behaviour metrics ────────────────────────────────────────────────────

def _compute_metrics(
    matched: list[tuple[dict, dict]],
    false_pos: list[dict],
    false_neg: list[dict],
) -> dict[str, dict]:
    """
    Compute precision, recall, F1, and median timestamp error per behaviour.

    Returns:
        dict of behaviour → {tp, fp, fn, precision, recall, f1, median_ts_err_s}
    """
    behaviours: set[str] = set()
    for p, g in matched:
        behaviours.add(p["behaviour"])
    for p in false_pos:
        behaviours.add(p["behaviour"])
    for g in false_neg:
        behaviours.add(g["behaviour"])

    results: dict[str, dict] = {}

    for beh in sorted(behaviours):
        tp_pairs = [(p, g) for p, g in matched if p["behaviour"] == beh]
        fp_list  = [p for p in false_pos  if p["behaviour"] == beh]
        fn_list  = [g for g in false_neg  if g["behaviour"] == beh]

        tp = len(tp_pairs)
        fp = len(fp_list)
        fn = len(fn_list)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        ts_errors = [abs(p["start_s"] - g["start_s"]) for p, g in tp_pairs]
        median_ts_err = statistics.median(ts_errors) if ts_errors else None

        results[beh] = {
            "tp":             tp,
            "fp":             fp,
            "fn":             fn,
            "precision":      round(precision, 3),
            "recall":         round(recall, 3),
            "f1":             round(f1, 3),
            "median_ts_err_s": round(median_ts_err, 2) if median_ts_err is not None else None,
        }

    return results


# ─── Overall (macro-averaged) metrics ─────────────────────────────────────────

def _macro_average(per_class: dict[str, dict]) -> dict:
    if not per_class:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precisions = [v["precision"] for v in per_class.values()]
    recalls    = [v["recall"]    for v in per_class.values()]
    f1s        = [v["f1"]        for v in per_class.values()]
    return {
        "precision": round(statistics.mean(precisions), 3),
        "recall":    round(statistics.mean(recalls),    3),
        "f1":        round(statistics.mean(f1s),        3),
    }


# ─── Pretty printer ───────────────────────────────────────────────────────────

def _print_report(
    per_class: dict[str, dict],
    macro: dict,
    tolerance_s: float,
    n_pred: int,
    n_gt: int,
) -> None:
    SEP = "─" * 78
    print(f"\n{SEP}")
    print("  HackNex PS07 – Evaluation Report")
    print(f"  Predictions: {n_pred}   GT events: {n_gt}   "
          f"Timestamp tolerance: ±{tolerance_s}s")
    print(SEP)
    print(
        f"  {'Behaviour':<22}  {'TP':>4}  {'FP':>4}  {'FN':>4}  "
        f"{'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Med ΔT':>8}"
    )
    print(SEP)
    for beh, m in per_class.items():
        med = f"{m['median_ts_err_s']:.2f}s" if m["median_ts_err_s"] is not None else "  —"
        print(
            f"  {beh:<22}  {m['tp']:>4}  {m['fp']:>4}  {m['fn']:>4}  "
            f"  {m['precision']:>5.1%}  {m['recall']:>5.1%}  {m['f1']:>5.1%}  {med:>8}"
        )
    print(SEP)
    print(
        f"  {'MACRO AVERAGE':<22}  {'':>4}  {'':>4}  {'':>4}  "
        f"  {macro['precision']:>5.1%}  {macro['recall']:>5.1%}  {macro['f1']:>5.1%}"
    )
    print(SEP + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def evaluate(
    events_path: str,
    gt_path: str,
    tolerance_s: float = 5.0,
    output_json: str | None = None,
) -> dict:
    """
    Run evaluation and return a result dict.

    Args:
        events_path:  Path to pipeline's events.json
        gt_path:      Path to ground truth JSON
        tolerance_s:  Max |Δt| in seconds for a match to count
        output_json:  Optional path to write full results JSON

    Returns:
        dict with keys 'per_class' and 'macro'
    """
    predictions  = _load(events_path)
    ground_truth = _load(gt_path)

    matched, fp_list, fn_list = _match_events(predictions, ground_truth, tolerance_s)

    per_class = _compute_metrics(matched, fp_list, fn_list)
    macro     = _macro_average(per_class)

    _print_report(per_class, macro, tolerance_s, len(predictions), len(ground_truth))

    results = {
        "tolerance_s": tolerance_s,
        "n_predictions": len(predictions),
        "n_ground_truth": len(ground_truth),
        "per_class": per_class,
        "macro": macro,
    }

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[eval] Results written → {output_json}")

    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(description="HackNex PS07 Evaluation (P4)")
    parser.add_argument(
        "--events", default="outputs/events.json",
        help="Path to pipeline events.json",
    )
    parser.add_argument(
        "--gt", default="data/ground_truth.json",
        help="Path to ground truth JSON",
    )
    parser.add_argument(
        "--tol", default=5.0, type=float,
        help="Timestamp tolerance in seconds (default 5)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to write results JSON (e.g. outputs/eval_results.json)",
    )
    args = parser.parse_args()

    evaluate(
        events_path=args.events,
        gt_path=args.gt,
        tolerance_s=args.tol,
        output_json=args.output,
    )


if __name__ == "__main__":
    _cli()
