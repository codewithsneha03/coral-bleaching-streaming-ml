"""
Step 3: Build the supervised coral-bleaching dataset.

PURPOSE:
  1. De-duplicate Bleaching_tbl  → one row per Sample_ID
  2. Construct a binary target    → bleached (1) vs not_bleached (0)
  3. Join with Sample_Event_tbl   → dates, depth, site ID
  4. Join with Environmental_tbl  → temperature & heat-stress features
  5. Join with Site_Info_tbl      → lat/lon, exposure, turbidity, etc.
  6. Exclude unlabelable rows     → no outcome info at all
  7. Save cleaned dataset + quality report

READS:
  data/raw/Global_Coral_Bleaching_Database_SQLite_11_24_21.db

WRITES:
  data/processed/coral_bleaching_dataset.csv
  results/data_quality_report.txt

IMPORTANT DESIGN DECISIONS (justified by 02b investigation):
  - Duplicates: 11,038 Sample_IDs appear twice with identical data.
    We keep MIN(Bleaching_ID) per Sample_ID.  For the single mixed-
    severity case (10323635), MAX severity wins.
  - Target: combine Severity_Code AND Percent_Bleached.
    Consistent where both exist (verified in investigation step).
  - Excluded:  Severity_Code = -1 ("% unknown") → ambiguous.
               Both NULL severity & NULL percent  → no outcome.
  - Outcome columns (Severity_Code, Percent_Bleached, etc.) are NOT
    used as predictors — only for target construction.
"""

import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────
DB_PATH = Path("data/raw/Global_Coral_Bleaching_Database_SQLite_11_24_21.db")
OUT_CSV = Path("data/processed/coral_bleaching_dataset.csv")
REPORT  = Path("results/data_quality_report.txt")

if not DB_PATH.is_file():
    raise SystemExit(f"Database not found: {DB_PATH}")

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

con = sqlite3.connect(f"file:{DB_PATH.resolve().as_posix()}?mode=ro", uri=True)
log = []                       # accumulate report lines

def section(title):
    log.append("")
    log.append("=" * 72)
    log.append(title)
    log.append("")

# =====================================================================
# STEP 1: Load & de-duplicate Bleaching_tbl
# =====================================================================
section("STEP 1: Load and de-duplicate Bleaching_tbl")

bleach_raw = pd.read_sql_query("SELECT * FROM Bleaching_tbl", con)
log.append(f"Raw Bleaching_tbl rows: {len(bleach_raw):,}")
log.append(f"Distinct Sample_IDs:    {bleach_raw['Sample_ID'].nunique():,}")

# Strategy: for each Sample_ID, keep the row with the HIGHEST Severity_Code
# (covers the single mixed-severity case) and the LOWEST Bleaching_ID
# as a tie-breaker (deterministic).
# We also want to preserve Percent_Bleached for target construction.
# Since duplicates are near-identical, aggregating MAX severity and
# MAX Percent_Bleached per Sample_ID is safe and simple.

bleach_agg = (
    bleach_raw
    .groupby("Sample_ID", as_index=False)
    .agg({
        "Bleaching_ID":              "min",       # deterministic row ID
        "Severity_Code":             "max",       # worst severity wins
        "Percent_Bleached":          "max",       # highest % wins
        "bleach_intensity":          "max",
        "Bleaching_Prevalence_Score":"max",
        "Bleaching_Level":           "first",
        "Number__Bleached_Colonies": "max",
    })
)

log.append(f"After de-duplication:    {len(bleach_agg):,} rows  "
           f"(removed {len(bleach_raw) - len(bleach_agg):,} duplicates)")

# =====================================================================
# STEP 2: Construct the binary target
# =====================================================================
section("STEP 2: Construct binary target")

# Convert to numeric (they may be stored as text/object)
bleach_agg["Severity_Code"]    = pd.to_numeric(bleach_agg["Severity_Code"],    errors="coerce")
bleach_agg["Percent_Bleached"] = pd.to_numeric(bleach_agg["Percent_Bleached"], errors="coerce")

def assign_target(row):
    """
    Returns:
      1  = bleaching observed
      0  = no bleaching observed
      NaN = cannot determine (exclude from supervised learning)
    """
    sev = row["Severity_Code"]
    pct = row["Percent_Bleached"]

    # Priority 1: Severity_Code (most specific)
    if pd.notna(sev):
        if sev in (1, 2, 3):
            return 1            # confirmed bleaching
        if sev == 0:
            return 0            # confirmed no bleaching
        if sev == -1:
            return np.nan       # "% unknown" → ambiguous, exclude
        return np.nan           # unexpected code

    # Priority 2: Percent_Bleached (when severity is NULL)
    if pd.notna(pct):
        if pct > 0:
            return 1            # bleaching observed
        if pct == 0:
            return 0            # no bleaching observed
        return np.nan

    # Neither field is available
    return np.nan

bleach_agg["bleached"] = bleach_agg.apply(assign_target, axis=1)

# Counts before dropping unlabelable rows
label_counts = bleach_agg["bleached"].value_counts(dropna=False)
log.append("Label distribution before filtering:")
for val, cnt in sorted(label_counts.items(), key=lambda x: str(x[0])):
    label = {1.0: "bleached (1)", 0.0: "not_bleached (0)"}.get(val, "unlabelable (NaN)")
    log.append(f"  {label}: {cnt:,}")

# Keep only rows with a valid label
bleach_labeled = bleach_agg.dropna(subset=["bleached"]).copy()
bleach_labeled["bleached"] = bleach_labeled["bleached"].astype(int)

log.append(f"\nRows with valid label:  {len(bleach_labeled):,}")
log.append(f"  bleached = 1: {(bleach_labeled['bleached'] == 1).sum():,}")
log.append(f"  bleached = 0: {(bleach_labeled['bleached'] == 0).sum():,}")
log.append(f"Rows excluded (no label): {len(bleach_agg) - len(bleach_labeled):,}")

# =====================================================================
# STEP 3: Join with Sample_Event_tbl (dates, depth, site)
# =====================================================================
section("STEP 3: Join with Sample_Event_tbl")

sample_event = pd.read_sql_query("""
    SELECT Sample_ID, Site_ID, Reef_ID,
           Date_Day, Date_Month, Date_Year,
           Depth_m
    FROM Sample_Event_tbl
""", con)

log.append(f"Sample_Event_tbl rows:  {len(sample_event):,}")

# Inner join: keep only labeled bleaching rows that have a sampling event
merged = bleach_labeled.merge(sample_event, on="Sample_ID", how="inner")

log.append(f"After join with Sample_Event: {len(merged):,} rows")
log.append(f"  Lost {len(bleach_labeled) - len(merged):,} labeled rows "
           f"(no matching Sample_Event)")

# =====================================================================
# STEP 4: Join with Environmental_tbl (features)
# =====================================================================
section("STEP 4: Join with Environmental_tbl")

# Environmental_tbl has 17 duplicate Sample_IDs (from investigation).
# De-duplicate by averaging numeric columns per Sample_ID.
env_raw = pd.read_sql_query("SELECT * FROM Environmental_tbl", con)
log.append(f"Environmental_tbl raw rows: {len(env_raw):,}")
log.append(f"  Distinct Sample_IDs:      {env_raw['Sample_ID'].nunique():,}")

# Drop the Environmental_ID (row key) — not needed
env_raw = env_raw.drop(columns=["Environmental_ID"], errors="ignore")

# For duplicated Sample_IDs, take the mean of numeric columns
env_dedup = env_raw.groupby("Sample_ID", as_index=False).mean(numeric_only=True)
log.append(f"  After de-dup (mean):      {len(env_dedup):,} rows")

# Select the environmental feature columns we want
ENV_FEATURES = [
    "Sample_ID",
    "ClimSST",
    "Temperature_Kelvin", "Temperature_Mean",
    "Temperature_Minimum", "Temperature_Maximum",
    "Temperature_Kelvin_Standard_Deviation",
    "Windspeed",
    "SSTA", "SSTA_Standard_Deviation",
    "SSTA_Mean", "SSTA_Minimum", "SSTA_Maximum",
    "SSTA_Frequency", "SSTA_Frequency_Standard_Deviation",
    "SSTA_FrequencyMax", "SSTA_FrequencyMean",
    "SSTA_DHW", "SSTA_DHW_Standard_Deviation",
    "SSTA_DHWMax", "SSTA_DHWMean",
    "TSA", "TSA_Standard_Deviation",
    "TSA_Minimum", "TSA_Maximum", "TSA_Mean",
    "TSA_Frequency", "TSA_Frequency_Standard_Deviation",
    "TSA_FrequencyMax", "TSA_FrequencyMean",
    "TSA_DHW", "TSA_DHW_Standard_Deviation",
    "TSA_DHWMax", "TSA_DHWMean",
]

# Keep only columns that actually exist in the data
available_env = [c for c in ENV_FEATURES if c in env_dedup.columns]
missing_env   = [c for c in ENV_FEATURES if c not in env_dedup.columns]
if missing_env:
    log.append(f"  WARNING: expected env columns not found: {missing_env}")

env_selected = env_dedup[available_env]

merged = merged.merge(env_selected, on="Sample_ID", how="inner")
log.append(f"After join with Environmental: {len(merged):,} rows")

# =====================================================================
# STEP 5: Join with Site_Info_tbl (location & site characteristics)
# =====================================================================
section("STEP 5: Join with Site_Info_tbl")

site_info = pd.read_sql_query("""
    SELECT Site_ID, Latitude_Degrees, Longitude_Degrees,
           Ocean_Name, Realm_Name, Ecoregion_Name, Country_Name,
           Distance_to_Shore, Exposure, Turbidity, Cyclone_Frequency
    FROM Site_Info_tbl
""", con)
log.append(f"Site_Info_tbl rows: {len(site_info):,}")

merged = merged.merge(site_info, on="Site_ID", how="left")
log.append(f"After join with Site_Info: {len(merged):,} rows")
log.append(f"  Rows without site info: "
           f"{merged['Latitude_Degrees'].isna().sum():,}")

con.close()

# =====================================================================
# STEP 6: Construct a proper date and clean up columns
# =====================================================================
section("STEP 6: Construct date column and select final columns")

# Build a proper date from Year/Month/Day
# Some records may have valid year but invalid day/month;
# fall back to Jan 1 or mid-year for those.
def safe_date(row):
    y = row["Date_Year"]
    m = row["Date_Month"]
    d = row["Date_Day"]
    if pd.isna(y):
        return pd.NaT
    y = int(y)
    m = int(m) if pd.notna(m) and 1 <= m <= 12 else 1
    d = int(d) if pd.notna(d) and 1 <= d <= 31 else 1
    try:
        return pd.Timestamp(year=y, month=m, day=d)
    except ValueError:
        # Handle e.g. Feb 30 → fall back to first of month
        return pd.Timestamp(year=y, month=m, day=1)

merged["date"] = merged.apply(safe_date, axis=1)
log.append(f"Valid dates: {merged['date'].notna().sum():,} / {len(merged):,}")
log.append(f"Date range:  {merged['date'].min()} to {merged['date'].max()}")

# ── Define outcome columns that must NOT be used as predictors ───────
OUTCOME_COLS = [
    "Severity_Code", "Percent_Bleached", "bleach_intensity",
    "Bleaching_Prevalence_Score", "Bleaching_Level",
    "Number__Bleached_Colonies", "Bleaching_ID",
]

# ── Define identifier / metadata columns ─────────────────────────────
ID_COLS = [
    "Sample_ID", "Site_ID", "Reef_ID", "date",
    "Date_Year", "Date_Month", "Date_Day",
    "Ocean_Name", "Realm_Name", "Ecoregion_Name", "Country_Name",
]

# ── Target column ────────────────────────────────────────────────────
TARGET = ["bleached"]

# ── Feature columns = everything else ────────────────────────────────
all_cols  = set(merged.columns)
non_feat  = set(OUTCOME_COLS + ID_COLS + TARGET)
FEATURE_COLS = sorted(all_cols - non_feat)

log.append(f"\nFeature columns ({len(FEATURE_COLS)}):")
for c in FEATURE_COLS:
    n_miss = merged[c].isna().sum()
    pct = 100 * n_miss / len(merged)
    log.append(f"  {c:45s}  missing: {n_miss:>6,} ({pct:5.1f}%)")

# ── Final column order ───────────────────────────────────────────────
final_cols = ID_COLS + TARGET + FEATURE_COLS
# Keep only columns that exist
final_cols = [c for c in final_cols if c in merged.columns]
dataset = merged[final_cols].copy()

# =====================================================================
# STEP 7: Summary statistics
# =====================================================================
section("STEP 7: Final dataset summary")

log.append(f"Final dataset shape: {dataset.shape[0]:,} rows × {dataset.shape[1]} columns")
log.append(f"\nTarget distribution:")
target_dist = dataset["bleached"].value_counts()
for val, cnt in sorted(target_dist.items()):
    pct = 100 * cnt / len(dataset)
    label = "bleached" if val == 1 else "not_bleached"
    log.append(f"  {label} ({val}): {cnt:,}  ({pct:.1f}%)")

imbalance_ratio = target_dist.max() / target_dist.min()
log.append(f"\nClass imbalance ratio: {imbalance_ratio:.1f}:1")

log.append(f"\nYear range: {dataset['Date_Year'].min():.0f} - {dataset['Date_Year'].max():.0f}")

n_sites = dataset["Site_ID"].nunique()
log.append(f"Distinct sites: {n_sites:,}")

n_countries = dataset["Country_Name"].nunique()
log.append(f"Distinct countries: {n_countries:,}")

# Overall missingness
total_cells = dataset[FEATURE_COLS].size
missing_cells = dataset[FEATURE_COLS].isna().sum().sum()
log.append(f"\nOverall feature missingness: {missing_cells:,} / {total_cells:,} "
           f"({100*missing_cells/total_cells:.1f}%)")

# =====================================================================
# STEP 8: Data-leakage audit
# =====================================================================
section("STEP 8: Data-leakage audit")

leaked = [c for c in OUTCOME_COLS if c in dataset.columns]
if leaked:
    log.append(f"WARNING: OUTCOME COLUMNS STILL IN DATASET: {leaked}")
    log.append("  These will be removed before modeling.")
else:
    log.append("[OK] No outcome/target-derivation columns in the feature set.")

log.append("\nNote on temporal alignment:")
log.append("  Environmental features (SST, SSTA, DHW, etc.) are recorded at")
log.append("  the SAME sampling event as the bleaching observation.")
log.append("  This means our model predicts bleaching status FROM concurrent")
log.append("  conditions — NOT in advance.  This is an honest 'nowcasting'")
log.append("  task: given today's environmental readings at a reef, is")
log.append("  bleaching present?  We will document this distinction clearly.")

# =====================================================================
# SAVE
# =====================================================================
section("SAVING OUTPUTS")

dataset.to_csv(OUT_CSV, index=False)
log.append(f"Dataset saved: {OUT_CSV.resolve()}")
log.append(f"  {dataset.shape[0]:,} rows × {dataset.shape[1]} columns")

report_text = "\n".join(log)
REPORT.write_text(report_text, encoding="utf-8")
log.append(f"Quality report saved: {REPORT.resolve()}")

print(report_text)
print(f"\n{'='*72}")
print(f"Dataset saved: {OUT_CSV.resolve()}")
print(f"Report saved:  {REPORT.resolve()}")
