"""
Step 7: Tuned Adaptive Random Forest with concept drift detection.

PURPOSE:
  1. Improve accuracy over the initial ARF (92.45%) by:
     - Increasing n_models (more trees = more ensemble diversity)
     - Using ADWIN drift detector (more sensitive than default)
     - Adjusting grace_period for faster tree splits
  2. Monitor concept drift explicitly with ADWIN detector
  3. Log drift events with timestamps for analysis
  4. Compare multiple ARF configurations side-by-side

WHAT IS CONCEPT DRIFT? (for beginners)
  Concept drift means the relationship between features and the target
  changes over time.  For example, coral may have bleached at lower
  temperatures in the 1990s than in the 2010s (due to acclimatization
  or sampling changes).  ARF handles this by detecting when individual
  trees are performing poorly and replacing them with fresh trees
  trained on recent data.

  ADWIN (ADaptive WINdowing) is a drift detection algorithm that
  maintains a variable-length window of recent observations and
  detects when the error rate in the recent window differs
  significantly from the historical rate.

READS:
  data/processed/coral_stream.csv

WRITES:
  results/arf_tuned_metrics.txt
  results/arf_tuned_predictions.csv
  results/drift_events.csv
  results/model_comparison.txt
"""

from pathlib import Path
import pandas as pd
import numpy as np
import time
from river.forest import ARFClassifier
from river.drift import ADWIN
from river import metrics

# ── Paths ────────────────────────────────────────────────────────────
STREAM_PATH = Path("data/processed/coral_stream.csv")
PRED_OUT    = Path("results/arf_tuned_predictions.csv")
METRICS_OUT = Path("results/arf_tuned_metrics.txt")
DRIFT_OUT   = Path("results/drift_events.csv")
COMPARE_OUT = Path("results/model_comparison.txt")

if not STREAM_PATH.is_file():
    raise SystemExit(f"Stream not found: {STREAM_PATH}")

# ── Load stream ──────────────────────────────────────────────────────
df = pd.read_csv(STREAM_PATH)
print(f"Loaded stream: {len(df):,} rows")

ID_COLS = ["Sample_ID", "Site_ID", "date", "Date_Year"]
TARGET  = "bleached"
FEATURE_COLS = [c for c in df.columns if c not in ID_COLS + [TARGET]]

print(f"Features: {len(FEATURE_COLS)}")

# ══════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATIONS TO COMPARE
# ══════════════════════════════════════════════════════════════════════
configs = {
    "ARF_10trees": {
        "model": ARFClassifier(
            n_models=10,
            max_features="sqrt",
            seed=42,
        ),
        "desc": "Original: 10 trees, default drift detector",
    },
    "ARF_25trees": {
        "model": ARFClassifier(
            n_models=25,
            max_features="sqrt",
            grace_period=30,
            seed=42,
        ),
        "desc": "25 trees, faster splits (grace=30)",
    },
    "ARF_25trees_ADWIN": {
        "model": ARFClassifier(
            n_models=25,
            max_features="sqrt",
            grace_period=30,
            drift_detector=ADWIN(delta=0.001),
            warning_detector=ADWIN(delta=0.01),
            seed=42,
        ),
        "desc": "25 trees, ADWIN drift/warning detectors",
    },
    "ARF_50trees_ADWIN": {
        "model": ARFClassifier(
            n_models=50,
            max_features="sqrt",
            grace_period=30,
            drift_detector=ADWIN(delta=0.001),
            warning_detector=ADWIN(delta=0.01),
            seed=42,
        ),
        "desc": "50 trees, ADWIN drift/warning detectors",
    },
}

# ══════════════════════════════════════════════════════════════════════
# RUN ALL CONFIGURATIONS
# ══════════════════════════════════════════════════════════════════════
all_results = {}
best_name = None
best_f1   = 0.0

for config_name, config in configs.items():
    model = config["model"]

    # Metrics
    acc  = metrics.Accuracy()
    f1   = metrics.F1()
    prec = metrics.Precision()
    rec  = metrics.Recall()
    cm   = metrics.ConfusionMatrix()

    # Concept drift monitoring (separate ADWIN on error stream)
    drift_monitor = ADWIN(delta=0.002)
    drift_events = []

    predictions = []
    n = len(df)
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"Training: {config_name} -- {config['desc']}")
    print(f"{'='*60}")
    print(f"{'Obs':>8s}  {'Acc':>7s}  {'Prec':>7s}  {'Rec':>7s}  {'F1':>7s}")
    print("-" * 42)

    for i, row in df.iterrows():
        x = {col: row[col] for col in FEATURE_COLS}
        y = int(row[TARGET])

        # 1. Predict
        y_pred = model.predict_one(x)

        # 2. Evaluate
        if y_pred is not None:
            acc.update(y, y_pred)
            f1.update(y, y_pred)
            prec.update(y, y_pred)
            rec.update(y, y_pred)
            cm.update(y, y_pred)

            # 3. Monitor for concept drift on the error stream
            error = 0 if y_pred == y else 1
            drift_monitor.update(error)
            if drift_monitor.drift_detected:
                drift_events.append({
                    "config": config_name,
                    "observation_index": i,
                    "Date_Year": row["Date_Year"],
                    "date": row["date"],
                    "accuracy_at_drift": acc.get(),
                    "f1_at_drift": f1.get(),
                })

        # 4. Record
        predictions.append({
            "Sample_ID": int(row["Sample_ID"]),
            "Date_Year": row["Date_Year"],
            "y_true": y,
            "y_pred": y_pred if y_pred is not None else -1,
        })

        # 5. Learn
        model.learn_one(x, y)

        if (i + 1) % 3000 == 0 or i == n - 1:
            print(f"{i+1:>8,}  {acc.get():>7.4f}  {prec.get():>7.4f}  "
                  f"{rec.get():>7.4f}  {f1.get():>7.4f}")

    elapsed = time.time() - start_time
    n_evaluated = sum(1 for p in predictions if p["y_pred"] != -1)

    all_results[config_name] = {
        "desc": config["desc"],
        "accuracy": acc.get(),
        "precision": prec.get(),
        "recall": rec.get(),
        "f1": f1.get(),
        "cm": str(cm),
        "n_evaluated": n_evaluated,
        "n_drifts": len(drift_events),
        "drift_events": drift_events,
        "elapsed_sec": elapsed,
        "predictions": predictions,
    }

    if f1.get() > best_f1:
        best_f1 = f1.get()
        best_name = config_name

    print(f"\nDrift events detected: {len(drift_events)}")
    print(f"Time: {elapsed:.1f}s")

# ══════════════════════════════════════════════════════════════════════
# SAVE BEST MODEL PREDICTIONS
# ══════════════════════════════════════════════════════════════════════
best = all_results[best_name]
pred_df = pd.DataFrame(best["predictions"])
pred_df.to_csv(PRED_OUT, index=False)

# ══════════════════════════════════════════════════════════════════════
# SAVE DRIFT EVENTS
# ══════════════════════════════════════════════════════════════════════
all_drifts = []
for r in all_results.values():
    all_drifts.extend(r["drift_events"])
if all_drifts:
    drift_df = pd.DataFrame(all_drifts)
    drift_df.to_csv(DRIFT_OUT, index=False)
    print(f"\nDrift events saved: {DRIFT_OUT.resolve()}")
else:
    print("\nNo drift events detected by any configuration.")
    # Create empty file for consistency
    pd.DataFrame(columns=["config","observation_index","Date_Year","date",
                          "accuracy_at_drift","f1_at_drift"]).to_csv(DRIFT_OUT, index=False)

# ══════════════════════════════════════════════════════════════════════
# COMPARISON REPORT
# ══════════════════════════════════════════════════════════════════════
naive_acc = df[TARGET].mean()

report = []
report.append("MODEL COMPARISON REPORT")
report.append("=" * 70)
report.append(f"Stream: {len(df):,} observations, chronological order")
report.append(f"Naive baseline (always predict 1): {naive_acc:.4f} accuracy")
report.append("")

# Summary table
report.append(f"{'Config':<25s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} "
              f"{'F1':>7s} {'Drifts':>6s} {'Time':>7s}")
report.append("-" * 70)
for name, r in all_results.items():
    marker = " <-- BEST" if name == best_name else ""
    report.append(f"{name:<25s} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
                  f"{r['recall']:>7.4f} {r['f1']:>7.4f} {r['n_drifts']:>6d} "
                  f"{r['elapsed_sec']:>6.1f}s{marker}")

report.append("")
report.append(f"BEST MODEL: {best_name}")
report.append(f"  {all_results[best_name]['desc']}")
report.append("")

# Detailed best model report
report.append("BEST MODEL DETAILED METRICS")
report.append("=" * 70)
report.append(f"Accuracy:   {best['accuracy']:.4f}")
report.append(f"Precision:  {best['precision']:.4f}")
report.append(f"Recall:     {best['recall']:.4f}")
report.append(f"F1 Score:   {best['f1']:.4f}")
report.append(f"Improvement over naive: {best['accuracy'] - naive_acc:+.4f}")
report.append("")
report.append("CONFUSION MATRIX")
report.append(best["cm"])
report.append("")

# Drift analysis
report.append("CONCEPT DRIFT ANALYSIS")
report.append("=" * 70)
if best["drift_events"]:
    report.append(f"Drift events detected: {best['n_drifts']}")
    report.append("")
    for evt in best["drift_events"]:
        report.append(f"  Index {evt['observation_index']:>6d}  "
                      f"Year={int(evt['Date_Year'])}  "
                      f"Date={evt['date']}  "
                      f"Acc={evt['accuracy_at_drift']:.4f}  "
                      f"F1={evt['f1_at_drift']:.4f}")
    report.append("")
    report.append("INTERPRETATION:")
    report.append("  Each drift event indicates that ADWIN detected a statistically")
    report.append("  significant change in the model's error rate. This can happen when:")
    report.append("  - The data distribution changes (new ocean conditions, regions)")
    report.append("  - The model encounters different bleaching patterns than before")
    report.append("  - ARF responds by replacing poorly-performing trees with fresh ones")
else:
    report.append("  No drift events detected. The ADWIN detector did not find")
    report.append("  statistically significant changes in the error stream.")

# Per-year performance for best model
report.append("")
report.append("PERFORMANCE BY YEAR (best model)")
report.append("-" * 40)
eval_df = pred_df[pred_df["y_pred"] != -1].copy()
for year in sorted(eval_df["Date_Year"].unique()):
    yr_data = eval_df[eval_df["Date_Year"] == year]
    if len(yr_data) < 5:
        continue
    yr_acc  = (yr_data["y_true"] == yr_data["y_pred"]).mean()
    yr_rate = yr_data["y_true"].mean()
    report.append(f"  {int(year)}: {len(yr_data):>5,} obs, "
                  f"acc={yr_acc:.3f}, bleach_rate={yr_rate:.3f}")

report_text = "\n".join(report)
COMPARE_OUT.write_text(report_text, encoding="utf-8")

# Also save detailed best-model metrics
METRICS_OUT.write_text(report_text, encoding="utf-8")

print(f"\n{report_text}")
print(f"\nPredictions saved:  {PRED_OUT.resolve()}")
print(f"Comparison saved:   {COMPARE_OUT.resolve()}")
print(f"Drift events saved: {DRIFT_OUT.resolve()}")
