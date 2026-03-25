"""
metrics_reporter.py
===================
Produces a full, reproducible metrics report for every training run.

Metrics computed
----------------
- Accuracy
- Macro F1 / Weighted F1
- Per-class Precision / Recall / F1 / Support
- Confusion Matrix

Reports are saved as JSON files under model_assets/reports/ so you can
compare results across training runs with compare_runs().
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from config import ASSETS_DIR

REPORTS_DIR = ASSETS_DIR / "reports"


# ---------------------------------------------------------------------------
# 1. Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X: np.ndarray,
    y_true_onehot: np.ndarray,
    class_names: list[str],
    split_name: str = "test",
) -> dict:
    """
    Run predictions and compute the full metrics suite.

    Parameters
    ----------
    model         : trained Keras model
    X             : feature array, shape (n_samples, n_features, 1)
    y_true_onehot : one-hot labels,  shape (n_samples, n_classes)
    class_names   : ordered list of emotion labels  e.g. ['angry','happy',...]
    split_name    : label for this evaluation  ('val' or 'test')

    Returns
    -------
    dict with all metrics (serialisable to JSON)
    """
    y_prob = model.predict(X, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_true_onehot, axis=1)

    acc          = float(accuracy_score(y_true, y_pred))
    macro_f1     = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    weighted_f1  = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm           = confusion_matrix(y_true, y_pred).tolist()

    # Per-class breakdown
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        name: {
            "precision": round(report_dict[name]["precision"], 4),
            "recall":    round(report_dict[name]["recall"],    4),
            "f1":        round(report_dict[name]["f1-score"],  4),
            "support":   int(report_dict[name]["support"]),
        }
        for name in class_names
    }

    return {
        "split":        split_name,
        "accuracy":     round(acc, 4),
        "macro_f1":     round(macro_f1, 4),
        "weighted_f1":  round(weighted_f1, 4),
        "per_class":    per_class,
        "confusion_matrix": cm,
        "class_names":  class_names,
    }


# ---------------------------------------------------------------------------
# 2. Print a human-readable summary to the console
# ---------------------------------------------------------------------------

def print_report(report: dict) -> None:
    """Pretty-print a metrics report to stdout."""
    split = report["split"].upper()
    print(f"\n{'─' * 55}")
    print(f"  Metrics Report — {split} set")
    print(f"{'─' * 55}")
    print(f"  Accuracy    : {report['accuracy']:.4f}")
    print(f"  Macro F1    : {report['macro_f1']:.4f}")
    print(f"  Weighted F1 : {report['weighted_f1']:.4f}")
    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print(f"  {'-'*51}")
    for name, vals in report["per_class"].items():
        print(
            f"  {name:<12} {vals['precision']:>10.4f} {vals['recall']:>8.4f} "
            f"{vals['f1']:>8.4f} {vals['support']:>9}"
        )
    print(f"\n  Confusion Matrix  (rows=actual, cols=predicted):")
    names = report["class_names"]
    header = "  " + " " * 10 + "  ".join(f"{n:>7}" for n in names)
    print(header)
    for i, row in enumerate(report["confusion_matrix"]):
        print(f"  {names[i]:<10}" + "  ".join(f"{v:>7}" for v in row))
    print(f"{'─' * 55}\n")


# ---------------------------------------------------------------------------
# 3. Save / load a single report
# ---------------------------------------------------------------------------

def save_report(report: dict, run_tag: str | None = None) -> Path:
    """
    Save a metrics report as a JSON file under model_assets/reports/.

    Parameters
    ----------
    report  : dict returned by evaluate_model()
    run_tag : optional short human label (e.g. 'rms_norm_baseline').
              If omitted, only the timestamp is used.

    Returns
    -------
    Path to the saved file.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    split = report.get("split", "unknown")
    tag   = f"_{run_tag}" if run_tag else ""
    fname = REPORTS_DIR / f"report_{ts}_{split}{tag}.json"

    # Attach timestamp so we can sort later
    report_to_save = {**report, "timestamp": ts, "run_tag": run_tag or ""}
    fname.write_text(json.dumps(report_to_save, indent=2), encoding="utf-8")
    print(f"Report saved -> {fname}")
    return fname


def load_report(path: Path | str) -> dict:
    """Load a previously saved JSON report."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 4. Compare multiple runs
# ---------------------------------------------------------------------------

def compare_runs(
    split_filter: str | None = None,
    tag_filter: str | None = None,
) -> pd.DataFrame:
    """
    Load all saved reports and return a summary DataFrame for comparison.

    Parameters
    ----------
    split_filter : if set, only include reports for this split ('val' or 'test')
    tag_filter   : if set, only include reports whose run_tag contains this string

    Returns
    -------
    pd.DataFrame sorted by timestamp, with one row per run.
    """
    if not REPORTS_DIR.exists():
        print("No reports directory found. Run at least one training run first.")
        return pd.DataFrame()

    records = []
    for path in sorted(REPORTS_DIR.glob("report_*.json")):
        rep = load_report(path)
        if split_filter and rep.get("split") != split_filter:
            continue
        if tag_filter and tag_filter not in rep.get("run_tag", ""):
            continue

        row = {
            "timestamp":   rep.get("timestamp", ""),
            "run_tag":     rep.get("run_tag", ""),
            "split":       rep.get("split", ""),
            "accuracy":    rep.get("accuracy"),
            "macro_f1":    rep.get("macro_f1"),
            "weighted_f1": rep.get("weighted_f1"),
        }
        # Flatten per-class F1
        for name, vals in rep.get("per_class", {}).items():
            row[f"{name}_f1"] = vals.get("f1")

        records.append(row)

    if not records:
        print("No matching reports found.")
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df)} report(s)")
    return df
