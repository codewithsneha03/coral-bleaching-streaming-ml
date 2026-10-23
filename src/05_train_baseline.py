"""
Step 5: Train a baseline online classifier (Logistic Regression).

PURPOSE:
  Establish a simple baseline for comparison with Adaptive Random Forest.
  Uses River's online Logistic Regression with prequential (test-then-train)
  evaluation on the chronologically ordered stream.

WHAT IS PREQUENTIAL EVALUATION? (for beginners)
  For each observation in order:
    1. The model predicts the label BEFORE seeing the answer.
    2. We record whether the prediction was correct.
    3. THEN the model learns from the true label.
    4. Move to the next observation.

  This is honest because the model is always evaluated on data
  it has never seen.  No separate train/test split is needed.

READS:
  data/processed/coral_stream.csv

WRITES:
  results/baseline_predictions.csv
  results/baseline_metrics.txt
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from river import linear_model, preprocessing, metrics

# ── Paths ────────────────────────────────────────────────────────────
STREAM_PATH = Path("data/processed/coral_stream.csv")
PRED_OUT    = Path("results/baseline_predictions.csv")
METRICS_OUT = Path("results/baseline_metrics.txt")

if not STREAM_PATH.is_file():
    raise SystemExit(f"Stream not found: {STREAM_PATH}. Run 04_prepare_stream.py first.")

PRED_OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Load stream ──────────────────────────────────────────────────────
df = pd.read_csv(STREAM_PATH)
print(f"Loaded stream: {len(df):,} rows")

# ── Define feature columns (everything except IDs and target) ────────
ID_COLS = ["Sample_ID", "Site_ID", "date", "Date_Year"]
TARGET  = "bleached"
FEATURE_COLS = [c for c in df.columns if c not in ID_COLS + [TARGET]]

print(f"Features: {len(FEATURE_COLS)}")
print(f"Target distribution: {df[TARGET].value_counts().to_dict()}")

# ── Build the model ──────────────────────────────────────────────────
# StandardScaler normalises features online (important for logistic
# regression which is sensitive to feature scales).
# LogisticRegression is a simple linear classifier that learns online.
model = preprocessing.StandardScaler() | linear_model.LogisticRegression()

# ── Metrics ──────────────────────────────────────────────────────────
# Track multiple metrics simultaneously
acc   = metrics.Accuracy()
f1    = metrics.F1()
prec  = metrics.Precision()
rec   = metrics.Recall()
# Confusion matrix
cm    = metrics.ConfusionMatrix()

# ── Prequential evaluation loop ─────────────────────────────────────
predictions = []
n = len(df)
report_every = 2000  # print progress every N observations

print(f"\nStarting prequential evaluation on {n:,} observations...")
print(f"{'Obs':>8s}  {'Acc':>7s}  {'Prec':>7s}  {'Rec':>7s}  {'F1':>7s}")
print("-" * 42)

for i, row in df.iterrows():
    # Extract features as a dict (River's native format)
    x = {col: row[col] for col in FEATURE_COLS}
    y = int(row[TARGET])

    # 1. PREDICT first (before learning)
    y_pred = model.predict_one(x)

    # 2. UPDATE metrics with this prediction
    if y_pred is not None:
        acc.update(y, y_pred)
        f1.update(y, y_pred)
        prec.update(y, y_pred)
        rec.update(y, y_pred)
        cm.update(y, y_pred)

    # 3. Store prediction for later analysis
    predictions.append({
        "Sample_ID": int(row["Sample_ID"]),
        "Date_Year": row["Date_Year"],
        "y_true": y,
        "y_pred": y_pred if y_pred is not None else -1,
    })

    # 4. LEARN from the true label
    model.learn_one(x, y)

    # Progress reporting
    if (i + 1) % report_every == 0 or i == n - 1:
        print(f"{i+1:>8,}  {acc.get():>7.4f}  {prec.get():>7.4f}  "
              f"{rec.get():>7.4f}  {f1.get():>7.4f}")

# ── Save predictions ────────────────────────────────────────────────
pred_df = pd.DataFrame(predictions)
pred_df.to_csv(PRED_OUT, index=False)
print(f"\nPredictions saved: {PRED_OUT.resolve()}")

# ── Compute final metrics ────────────────────────────────────────────
# Count how many predictions were actually made (first prediction may be None)
n_evaluated = pred_df[pred_df["y_pred"] != -1].shape[0]
n_skipped   = pred_df[pred_df["y_pred"] == -1].shape[0]

# Naive baseline comparison: always predict majority class (1 = bleached)
naive_acc = df[TARGET].mean()  # accuracy of always predicting 1

report = []
report.append("BASELINE MODEL: Online Logistic Regression")
report.append("=" * 60)
report.append(f"Model:  StandardScaler | LogisticRegression")
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

# Performance by year
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
print(f"\nMetrics saved: {METRICS_OUT.resolve()}")
