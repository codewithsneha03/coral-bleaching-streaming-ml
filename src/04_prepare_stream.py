"""
Step 4: Prepare the chronological stream for online learning.

PURPOSE:
  Sort the validated dataset by date so that the online/streaming
  ML models see observations in the order they occurred.

  Also:
  - Select only numeric feature columns suitable for River models.
  - Handle missing values with a strategy compatible with online learning.
  - Retain identifiers for evaluation tracking but mark them clearly.

READS:
  data/processed/coral_bleaching_dataset.csv

WRITES:
  data/processed/coral_stream.csv
  results/stream_preparation_report.txt

WHY CHRONOLOGICAL ORDER?
  Online learning processes data one observation at a time, in arrival
  order.  For temporal data like coral bleaching surveys, the natural
  order is by date.  This means the model is always evaluated on
  "future" data relative to what it has learned — just like in a
  real deployment scenario.

  We do NOT randomly shuffle, because that would let the model
  "see the future" through patterns learned from later years.

MISSING VALUE STRATEGY:
  River models handle missing values poorly (many learners crash on NaN).
  We impute missing numeric features with the column MEDIAN computed
  ONLY from the training data seen so far.  However, because we are
  doing a one-pass preparation here, we use the global median as a
  practical approximation.  The alternative (online imputation) adds
  complexity without clear benefit given <0.5% overall missingness.

  Depth_m (14.1% missing) gets its own treatment: imputed with the
  dataset median, and a binary indicator 'Depth_m_missing' is added
  so the model can learn whether missing depth is informative.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────
INPUT_CSV  = Path("data/processed/coral_bleaching_dataset.csv")
OUT_STREAM = Path("data/processed/coral_stream.csv")
REPORT     = Path("results/stream_preparation_report.txt")

if not INPUT_CSV.is_file():
    raise SystemExit(f"Input not found: {INPUT_CSV}. Run 03_build_dataset.py first.")

OUT_STREAM.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

log = []

def section(title):
    log.append("")
    log.append("=" * 72)
    log.append(title)
    log.append("")

# =====================================================================
# STEP 1: Load the dataset
# =====================================================================
section("STEP 1: Load dataset")

df = pd.read_csv(INPUT_CSV)
log.append(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Parse the date column
df["date"] = pd.to_datetime(df["date"], errors="coerce")
log.append(f"Rows with valid date: {df['date'].notna().sum():,}")

# =====================================================================
# STEP 2: Sort chronologically
# =====================================================================
section("STEP 2: Sort chronologically")

df = df.sort_values("date").reset_index(drop=True)
log.append(f"First observation: {df['date'].iloc[0]}")
log.append(f"Last observation:  {df['date'].iloc[-1]}")
log.append(f"Span: {(df['date'].iloc[-1] - df['date'].iloc[0]).days:,} days")

# Show year distribution
year_counts = df["Date_Year"].value_counts().sort_index()
log.append("\nObservations per year:")
for year, cnt in year_counts.items():
    log.append(f"  {int(year)}: {cnt:>5,}")

# =====================================================================
# STEP 3: Define feature columns for the model
# =====================================================================
section("STEP 3: Define feature columns")

# Identifiers to keep for tracking but NOT use as features
ID_COLS = ["Sample_ID", "Site_ID", "Reef_ID", "date",
           "Date_Year", "Date_Month", "Date_Day",
           "Ocean_Name", "Realm_Name", "Ecoregion_Name", "Country_Name"]

TARGET = "bleached"

# All remaining numeric columns are candidate features
all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in all_numeric
                if c not in ID_COLS and c != TARGET
                and c not in ["Sample_ID", "Site_ID", "Ocean_Name",
                              "Realm_Name", "Ecoregion_Name", "Country_Name"]]

log.append(f"Number of feature columns: {len(feature_cols)}")
log.append("Feature columns:")
for c in feature_cols:
    log.append(f"  {c}")

# =====================================================================
# STEP 4: Handle missing values
# =====================================================================
section("STEP 4: Handle missing values")

# Report missingness per feature
miss_report = []
for c in feature_cols:
    n_miss = df[c].isna().sum()
    pct = 100 * n_miss / len(df)
    if n_miss > 0:
        miss_report.append((c, n_miss, pct))
        log.append(f"  {c:45s}  {n_miss:>6,} ({pct:5.1f}%)")

if not miss_report:
    log.append("  No missing values in feature columns!")

# Special treatment for Depth_m (14.1% missing)
if "Depth_m" in feature_cols:
    depth_median = df["Depth_m"].median()
    df["Depth_m_missing"] = df["Depth_m"].isna().astype(int)
    df["Depth_m"] = df["Depth_m"].fillna(depth_median)
    log.append(f"\n  Depth_m: imputed {df['Depth_m_missing'].sum():,} NaNs "
               f"with median={depth_median:.1f}m")
    log.append(f"  Added binary indicator: Depth_m_missing")
    if "Depth_m_missing" not in feature_cols:
        feature_cols.append("Depth_m_missing")

# Impute remaining features with column median (< 0.1% missing each)
for c in feature_cols:
    n_miss = df[c].isna().sum()
    if n_miss > 0:
        med = df[c].median()
        df[c] = df[c].fillna(med)
        log.append(f"  {c}: imputed {n_miss} NaNs with median={med:.4f}")

# Final check
remaining_miss = df[feature_cols].isna().sum().sum()
log.append(f"\nRemaining missing values in features: {remaining_miss}")

# =====================================================================
# STEP 5: Build the final stream dataframe
# =====================================================================
section("STEP 5: Build final stream")

# Include: identifiers (for tracking) + features + target
stream_cols = ["Sample_ID", "Site_ID", "date", "Date_Year"] + feature_cols + [TARGET]
stream = df[stream_cols].copy()

log.append(f"Stream shape: {stream.shape[0]:,} rows x {stream.shape[1]} columns")
log.append(f"\nTarget distribution in stream:")
target_dist = stream[TARGET].value_counts()
for val, cnt in sorted(target_dist.items()):
    label = "bleached" if val == 1 else "not_bleached"
    pct = 100 * cnt / len(stream)
    log.append(f"  {label} ({val}): {cnt:,}  ({pct:.1f}%)")

log.append(f"\nChronological order verified: {(stream['date'] == stream['date'].sort_values().values).all()}")

# Check for temporal class distribution shifts
log.append("\nBleaching rate by decade:")
stream["decade"] = (stream["Date_Year"] // 10 * 10).astype(int)
decade_stats = stream.groupby("decade")[TARGET].agg(["count", "mean"])
for decade, row in decade_stats.iterrows():
    log.append(f"  {decade}s: {int(row['count']):>5,} obs, "
               f"bleaching rate = {row['mean']:.1%}")
stream = stream.drop(columns=["decade"])

# =====================================================================
# STEP 6: Save
# =====================================================================
section("SAVING STREAM")

stream.to_csv(OUT_STREAM, index=False)
log.append(f"Saved: {OUT_STREAM.resolve()}")
log.append(f"  {stream.shape[0]:,} rows x {stream.shape[1]} columns")

# Feature list for reference by downstream scripts
log.append(f"\nFEATURE LIST (for model scripts):")
model_features = [c for c in feature_cols]
for i, c in enumerate(model_features, 1):
    log.append(f"  {i:>2}. {c}")

report_text = "\n".join(log)
REPORT.write_text(report_text, encoding="utf-8")
print(report_text)
print(f"\n{'='*72}")
print(f"Stream saved: {OUT_STREAM.resolve()}")
print(f"Report saved: {REPORT.resolve()}")
