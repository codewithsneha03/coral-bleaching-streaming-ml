"""
Step 6: Train Adaptive Random Forest (ARF) classifier.

PURPOSE:
  Train the main model — River's Adaptive Random Forest — using the
  same prequential evaluation protocol as the baseline, so results
  are directly comparable.

WHAT IS ADAPTIVE RANDOM FOREST? (for beginners)
  ARF is an online version of Random Forest that:
  - Builds many decision trees that each see a random subset of features
  - Each tree votes on the final prediction (ensemble)
  - Monitors each tree for "concept drift" (when the data changes)
  - Replaces drifting trees with fresh ones trained on recent data

  This makes it adaptive: it can adjust to changing conditions over time,
  which is exactly what we expect with coral bleaching data spanning
  decades of changing ocean temperatures.

READS:
  data/processed/coral_stream.csv

WRITES:
  results/arf_predictions.csv
  results/arf_metrics.txt
"""

from pathlib import Path
import pandas as pd
import numpy as np
from river.forest import ARFClassifier
from river import metrics

# ── Paths ────────────────────────────────────────────────────────────
STREAM_PATH = Path("data/processed/coral_stream.csv")
PRED_OUT    = Path("results/arf_predictions.csv")
METRICS_OUT = Path("results/arf_metrics.txt")

if not STREAM_PATH.is_file():
    raise SystemExit(f"Stream not found: {STREAM_PATH}. Run 04_prepare_stream.py first.")

# ── Load stream ──────────────────────────────────────────────────────
df = pd.read_csv(STREAM_PATH)
print(f"Loaded stream: {len(df):,} rows")

ID_COLS = ["Sample_ID", "Site_ID", "date", "Date_Year"]
TARGET  = "bleached"
FEATURE_COLS = [c for c in df.columns if c not in ID_COLS + [TARGET]]

print(f"Features: {len(FEATURE_COLS)}")

# ── Build the ARF model ─────────────────────────────────────────────
# ARF does NOT need feature scaling (tree-based, scale-invariant).
# n_models=10: 10 trees in the ensemble (reasonable for this dataset size).
# max_features='sqrt': each tree sees sqrt(41) ~ 6 features per split.
# seed=42: reproducibility.
model = ARFClassifier(
    n_models=10,
    max_features="sqrt",
    seed=42,
)

# ── Metrics ──────────────────────────────────────────────────────────
acc  = metrics.Accuracy()
f1   = metrics.F1()
prec = metrics.Precision()
rec  = metrics.Recall()
cm   = metrics.ConfusionMatrix()

# ── Prequential evaluation ──────────────────────────────────────────
predictions = []
n = len(df)
report_every = 2000

print(f"\nStarting ARF prequential evaluation on {n:,} observations...")
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

    # 3. Record
    predictions.append({
        "Sample_ID": int(row["Sample_ID"]),
        "Date_Year": row["Date_Year"],
        "y_true": y,
        "y_pred": y_pred if y_pred is not None else -1,
    })

    # 4. Learn
    model.learn_one(x, y)

    if (i + 1) % report_every == 0 or i == n - 1:
        print(f"{i+1:>8,}  {acc.get():>7.4f}  {prec.get():>7.4f}  "
              f"{rec.get():>7.4f}  {f1.get():>7.4f}")

# ── Save predictions ────────────────────────────────────────────────
pred_df = pd.DataFrame(predictions)
pred_df.to_csv(PRED_OUT, index=False)

n_evaluated = pred_df[pred_df["y_pred"] != -1].shape[0]
n_skipped   = pred_df[pred_df["y_pred"] == -1].shape[0]
naive_acc   = df[TARGET].mean()

# ── Final report ─────────────────────────────────────────────────────
report = []
report.append("ADAPTIVE RANDOM FOREST CLASSIFIER")
report.append("=" * 60)
report.append(f"Model:  ARFClassifier(n_models=10, max_features='sqrt', seed=42)")
report.append(f"Stream: {n:,} observations, chronological order")
report.append(f"Evaluated predictions: {n_evaluated:,}")
report.append(f"Skipped (cold start):  {n_skipped:,}")
report.append("")
report.append("METRICS (prequential / test-then-train)")
report.append("-" * 40)
report.append(f"Accuracy:   {acc.get():.4f}")
report.append(f"Precision:  {prec.get():.4f}")
report.append(f"Recall:     {rec.get():.4f}")
report.append(f"F1 Score:   {f1.get():.4f}")
report.append("")
report.append(f"Naive baseline (always predict 1): {naive_acc:.4f} accuracy")
report.append(f"Improvement over naive: {acc.get() - naive_acc:+.4f}")
report.append("")
report.append("CONFUSION MATRIX")
report.append(str(cm))
report.append("")

# Per-year performance
report.append("PERFORMANCE BY YEAR")
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
METRICS_OUT.write_text(report_text, encoding="utf-8")
print(f"\n{report_text}")
print(f"\nPredictions saved: {PRED_OUT.resolve()}")
print(f"Metrics saved:     {METRICS_OUT.resolve()}")
