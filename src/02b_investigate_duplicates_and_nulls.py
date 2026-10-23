"""
Step 2b: Deep investigation of Bleaching_tbl duplicates and NULL severity rows.

PURPOSE:
  Before building the supervised dataset, we need to understand:
  1. WHY some Sample_IDs appear multiple times in Bleaching_tbl
  2. Whether the repeated rows carry different or identical information
  3. Whether the 27,878 NULL-severity rows have OTHER usable outcome fields
  4. How to safely aggregate or de-duplicate for a one-row-per-event dataset

READS:  data/raw/Global_Coral_Bleaching_Database_SQLite_11_24_21.db
WRITES: results/duplicate_investigation_report.txt
        results/null_severity_rescue_report.txt
"""

import sqlite3
import sys
from pathlib import Path
import pandas as pd

db_path = Path("data/raw/Global_Coral_Bleaching_Database_SQLite_11_24_21.db")
if not db_path.is_file():
    raise SystemExit(f"Database not found: {db_path}")

con = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
out_dir = Path("results")
out_dir.mkdir(exist_ok=True)

report = []

# ── 1. How many Sample_IDs are duplicated, and by how many rows? ─────────
report.append("=" * 72)
report.append("1. DISTRIBUTION OF DUPLICATE COUNTS IN Bleaching_tbl")
report.append("   (How many Sample_IDs appear 2 times, 3 times, etc.)")
report.append("")

dup_dist = pd.read_sql_query("""
    SELECT n AS rows_per_sample_id,
           COUNT(*) AS how_many_sample_ids
    FROM (
        SELECT Sample_ID, COUNT(*) AS n
        FROM Bleaching_tbl
        GROUP BY Sample_ID
    )
    GROUP BY n
    ORDER BY n
""", con)
report.append(dup_dist.to_string(index=False))
report.append("")

# ── 2. For duplicated Sample_IDs: are severity codes the same or different? ──
report.append("=" * 72)
report.append("2. DO DUPLICATE ROWS FOR THE SAME Sample_ID HAVE THE SAME Severity_Code?")
report.append("")

severity_agreement = pd.read_sql_query("""
    SELECT
        CASE
            WHEN cnt_distinct_sev = 1 THEN 'all_same_severity'
            ELSE 'mixed_severity'
        END AS severity_pattern,
        COUNT(*) AS sample_ids_with_this_pattern
    FROM (
        SELECT Sample_ID,
               COUNT(*) AS n,
               COUNT(DISTINCT COALESCE(Severity_Code, -999)) AS cnt_distinct_sev
        FROM Bleaching_tbl
        GROUP BY Sample_ID
        HAVING COUNT(*) > 1
    )
    GROUP BY severity_pattern
""", con)
report.append(severity_agreement.to_string(index=False))
report.append("")

# ── 3. Sample of duplicated rows with DIFFERENT severity codes ───────────
report.append("=" * 72)
report.append("3. SAMPLE OF DUPLICATED Sample_IDs WITH DIFFERENT Severity_Codes")
report.append("   (If these exist, we need an aggregation strategy)")
report.append("")

mixed_sample = pd.read_sql_query("""
    SELECT b.Bleaching_ID, b.Sample_ID, b.Severity_Code,
           b.Percent_Bleached, b.bleach_intensity,
           b.Bleaching_Prevalence_Score
    FROM Bleaching_tbl b
    WHERE b.Sample_ID IN (
        SELECT Sample_ID
        FROM Bleaching_tbl
        GROUP BY Sample_ID
        HAVING COUNT(*) > 1
         AND COUNT(DISTINCT COALESCE(Severity_Code, -999)) > 1
    )
    ORDER BY b.Sample_ID, b.Bleaching_ID
    LIMIT 30
""", con)
if mixed_sample.empty:
    report.append("  (No duplicated Sample_IDs with conflicting severity codes found.)")
else:
    report.append(mixed_sample.to_string(index=False))
report.append("")

# ── 4. Sample of duplicated rows with SAME severity codes ────────────────
report.append("=" * 72)
report.append("4. SAMPLE OF DUPLICATED Sample_IDs WITH SAME Severity_Code")
report.append("   (Are they truly identical rows, or do other fields differ?)")
report.append("")

same_sev_sample = pd.read_sql_query("""
    SELECT b.*
    FROM Bleaching_tbl b
    WHERE b.Sample_ID IN (
        SELECT Sample_ID
        FROM Bleaching_tbl
        GROUP BY Sample_ID
        HAVING COUNT(*) > 1
         AND COUNT(DISTINCT COALESCE(Severity_Code, -999)) = 1
    )
    ORDER BY b.Sample_ID, b.Bleaching_ID
    LIMIT 20
""", con)
report.append(same_sev_sample.to_string(index=False))
report.append("")

# ── 5. Investigate NULL severity rows: what OTHER outcome fields do they have?
report.append("=" * 72)
report.append("5. NULL-SEVERITY ROWS: DO THEY HAVE OTHER USABLE OUTCOME FIELDS?")
report.append("   (Can we rescue labels from Percent_Bleached, bleach_intensity, etc.?)")
report.append("")

null_rescue = pd.read_sql_query("""
    SELECT
        COUNT(*) AS total_null_severity,
        SUM(CASE WHEN Percent_Bleached IS NOT NULL THEN 1 ELSE 0 END) AS has_percent_bleached,
        SUM(CASE WHEN bleach_intensity IS NOT NULL THEN 1 ELSE 0 END) AS has_intensity,
        SUM(CASE WHEN Bleaching_Prevalence_Score IS NOT NULL THEN 1 ELSE 0 END) AS has_prevalence,
        SUM(CASE WHEN Bleaching_Level IS NOT NULL THEN 1 ELSE 0 END) AS has_level,
        SUM(CASE WHEN Number__Bleached_Colonies IS NOT NULL THEN 1 ELSE 0 END) AS has_bleached_colonies,
        SUM(CASE WHEN Percent_Bleached IS NOT NULL
              OR bleach_intensity IS NOT NULL
              OR Bleaching_Prevalence_Score IS NOT NULL
              OR Bleaching_Level IS NOT NULL
              OR Number__Bleached_Colonies IS NOT NULL THEN 1 ELSE 0 END
        ) AS has_any_outcome_field
    FROM Bleaching_tbl
    WHERE Severity_Code IS NULL
""", con)
report.append(null_rescue.to_string(index=False))
report.append("")

# ── 6. Percent_Bleached distribution for NULL severity rows ──────────────
report.append("=" * 72)
report.append("6. DISTRIBUTION OF Percent_Bleached WHERE Severity_Code IS NULL")
report.append("")

pct_dist = pd.read_sql_query("""
    SELECT
        CASE
            WHEN Percent_Bleached IS NULL THEN 'NULL'
            WHEN Percent_Bleached = 0 THEN '0 (no bleaching)'
            WHEN Percent_Bleached > 0 AND Percent_Bleached <= 10 THEN '1-10'
            WHEN Percent_Bleached > 10 AND Percent_Bleached <= 50 THEN '11-50'
            WHEN Percent_Bleached > 50 THEN '>50'
            ELSE 'other'
        END AS pct_range,
        COUNT(*) AS cnt
    FROM Bleaching_tbl
    WHERE Severity_Code IS NULL
    GROUP BY pct_range
    ORDER BY cnt DESC
""", con)
report.append(pct_dist.to_string(index=False))
report.append("")

# ── 7. Percent_Bleached distribution for ALL rows (not just NULL severity) ───
report.append("=" * 72)
report.append("7. DISTRIBUTION OF Percent_Bleached FOR ALL ROWS")
report.append("")

pct_all = pd.read_sql_query("""
    SELECT
        CASE
            WHEN Percent_Bleached IS NULL THEN 'NULL'
            WHEN Percent_Bleached = 0 THEN '0 (no bleaching)'
            WHEN Percent_Bleached > 0 AND Percent_Bleached <= 10 THEN '1-10'
            WHEN Percent_Bleached > 10 AND Percent_Bleached <= 50 THEN '11-50'
            WHEN Percent_Bleached > 50 THEN '>50'
            ELSE 'other'
        END AS pct_range,
        COUNT(*) AS cnt
    FROM Bleaching_tbl
    GROUP BY pct_range
    ORDER BY cnt DESC
""", con)
report.append(pct_all.to_string(index=False))
report.append("")

# ── 8. Cross-tab: Severity_Code vs Percent_Bleached availability ─────────
report.append("=" * 72)
report.append("8. CROSS-TAB: Severity_Code × Percent_Bleached availability")
report.append("   (Do the two fields overlap, complement, or conflict?)")
report.append("")

crosstab = pd.read_sql_query("""
    SELECT
        COALESCE(CAST(Severity_Code AS TEXT), 'NULL') AS severity,
        CASE WHEN Percent_Bleached IS NULL THEN 'pct_NULL' ELSE 'pct_present' END AS pct_status,
        COUNT(*) AS cnt
    FROM Bleaching_tbl
    GROUP BY severity, pct_status
    ORDER BY severity, pct_status
""", con)
report.append(crosstab.to_string(index=False))
report.append("")

# ── 9. Consistency check: does Percent_Bleached agree with Severity_Code? ─
report.append("=" * 72)
report.append("9. CONSISTENCY: Severity_Code vs Percent_Bleached WHERE BOTH EXIST")
report.append("   Severity: 0=No bleaching, 1=1-10%, 2=11-50%, 3=>50%")
report.append("")

consistency = pd.read_sql_query("""
    SELECT Severity_Code,
        MIN(Percent_Bleached) AS min_pct,
        AVG(Percent_Bleached) AS avg_pct,
        MAX(Percent_Bleached) AS max_pct,
        COUNT(*) AS cnt
    FROM Bleaching_tbl
    WHERE Severity_Code IS NOT NULL AND Percent_Bleached IS NOT NULL
    GROUP BY Severity_Code
    ORDER BY Severity_Code
""", con)
report.append(consistency.to_string(index=False))
report.append("")

# ── 10. Environmental coverage for rows WITH vs WITHOUT labels ───────────
report.append("=" * 72)
report.append("10. ENVIRONMENTAL DATA AVAILABILITY: labeled vs unlabeled rows")
report.append("    (Can we build features for the rows that have labels?)")
report.append("")

env_coverage = pd.read_sql_query("""
    SELECT
        CASE
            WHEN b.Severity_Code IS NOT NULL THEN 'has_severity'
            WHEN b.Percent_Bleached IS NOT NULL THEN 'has_pct_only'
            ELSE 'no_label'
        END AS label_status,
        COUNT(*) AS bleaching_rows,
        SUM(CASE WHEN e.Environmental_ID IS NOT NULL THEN 1 ELSE 0 END) AS has_env_data
    FROM Bleaching_tbl b
    LEFT JOIN Environmental_tbl e ON b.Sample_ID = e.Sample_ID
    GROUP BY label_status
    ORDER BY label_status
""", con)
report.append(env_coverage.to_string(index=False))
report.append("")

con.close()

text = "\n".join(report)
(out_dir / "duplicate_investigation_report.txt").write_text(text, encoding="utf-8")
print(text)
print(f"\nSaved: {(out_dir / 'duplicate_investigation_report.txt').resolve()}")
