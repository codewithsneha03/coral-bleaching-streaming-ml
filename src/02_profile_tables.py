"""Step 2: Profile key GCBD tables and label fields. Read-only."""
import sqlite3
import sys
from pathlib import Path
import pandas as pd

if len(sys.argv) != 2:
    raise SystemExit("Usage: python src/02_profile_tables.py data/raw/your_database.db")

db_path = Path(sys.argv[1])
if not db_path.is_file():
    raise SystemExit(f"Database not found: {db_path}")

con = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
out_dir = Path("results")
out_dir.mkdir(exist_ok=True)

queries = {
    "severity_code_counts": """
        SELECT b.Severity_Code, s.Severity_Code AS Severity_Description,
               COUNT(*) AS Records
        FROM Bleaching_tbl b
        LEFT JOIN Severity_Code_LUT s ON b.Severity_Code = s.Severity_ID
        GROUP BY b.Severity_Code, s.Severity_Code ORDER BY b.Severity_Code
    """,
    "prevalence_counts": """
        SELECT b.Bleaching_Prevalence_Score,
               p.Bleaching_Prevalence_Score AS Prevalence_Description,
               COUNT(*) AS Records
        FROM Bleaching_tbl b
        LEFT JOIN Bleaching_Prevalence_Score_LUT p
          ON b.Bleaching_Prevalence_Score = p.Bleaching_Prevalence_ID
        GROUP BY b.Bleaching_Prevalence_Score, p.Bleaching_Prevalence_Score
        ORDER BY b.Bleaching_Prevalence_Score
    """,
    "bleaching_fields_summary": """
        SELECT COUNT(*) AS total_rows,
          COUNT(DISTINCT Sample_ID) AS distinct_sample_ids,
          SUM(CASE WHEN Severity_Code IS NOT NULL THEN 1 ELSE 0 END) AS severity_not_null,
          SUM(CASE WHEN Bleaching_Prevalence_Score IS NOT NULL THEN 1 ELSE 0 END) AS prevalence_not_null,
          SUM(CASE WHEN Percent_Bleached IS NOT NULL THEN 1 ELSE 0 END) AS percent_bleached_not_null,
          SUM(CASE WHEN bleach_intensity IS NOT NULL THEN 1 ELSE 0 END) AS intensity_not_null
        FROM Bleaching_tbl
    """,
    "environment_fields_summary": """
        SELECT COUNT(*) AS total_rows,
          COUNT(DISTINCT Sample_ID) AS distinct_sample_ids,
          SUM(CASE WHEN ClimSST IS NOT NULL THEN 1 ELSE 0 END) AS ClimSST_not_null,
          SUM(CASE WHEN SSTA IS NOT NULL THEN 1 ELSE 0 END) AS SSTA_not_null,
          SUM(CASE WHEN SSTA_DHW IS NOT NULL THEN 1 ELSE 0 END) AS SSTA_DHW_not_null,
          SUM(CASE WHEN Temperature_Mean IS NOT NULL THEN 1 ELSE 0 END) AS Temperature_Mean_not_null
        FROM Environmental_tbl
    """,
    "sample_date_summary": """
        SELECT COUNT(*) AS total_rows,
          COUNT(DISTINCT Sample_ID) AS distinct_sample_ids,
          MIN(Date_Year) AS min_year, MAX(Date_Year) AS max_year,
          SUM(CASE WHEN Date_Year IS NOT NULL AND Date_Month IS NOT NULL
                    AND Date_Day IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_full_date
        FROM Sample_Event_tbl
    """,
    "join_counts": """
        SELECT
          (SELECT COUNT(*) FROM Sample_Event_tbl) AS sample_rows,
          (SELECT COUNT(*) FROM Bleaching_tbl) AS bleaching_rows,
          (SELECT COUNT(*) FROM Environmental_tbl) AS environmental_rows,
          (SELECT COUNT(*) FROM Sample_Event_tbl s JOIN Bleaching_tbl b ON s.Sample_ID=b.Sample_ID) AS sample_bleaching_join_rows,
          (SELECT COUNT(*) FROM Sample_Event_tbl s JOIN Environmental_tbl e ON s.Sample_ID=e.Sample_ID) AS sample_environment_join_rows,
          (SELECT COUNT(*) FROM Sample_Event_tbl s JOIN Bleaching_tbl b ON s.Sample_ID=b.Sample_ID JOIN Environmental_tbl e ON s.Sample_ID=e.Sample_ID) AS all_three_join_rows
    """
}

report = []
for name, query in queries.items():
    df = pd.read_sql_query(query, con)
    report.extend([f"\n{'='*72}\n{name}\n", df.to_string(index=False)])
    df.to_csv(out_dir / f"{name}.csv", index=False)

for table in ("Sample_Event_tbl", "Bleaching_tbl", "Environmental_tbl"):
    query = f'''SELECT COUNT(*) AS rows, COUNT(DISTINCT Sample_ID) AS distinct_sample_ids,
                COUNT(*) - COUNT(DISTINCT Sample_ID) AS duplicate_key_excess
                FROM "{table}"'''
    df = pd.read_sql_query(query, con)
    report.extend([f"\n{'='*72}\nkey_check_{table}\n", df.to_string(index=False)])

sample = pd.read_sql_query('''SELECT Sample_ID, Severity_Code, Percent_Bleached,
    bleach_intensity, Bleaching_Prevalence_Score FROM Bleaching_tbl LIMIT 15''', con)
report.extend([f"\n{'='*72}\nbleaching_target_sample\n", sample.to_string(index=False)])
con.close()
text = "\n".join(report)
(out_dir / "table_profile_report.txt").write_text(text, encoding="utf-8")
print(text)
print(f"\nSaved detailed report: {(out_dir / 'table_profile_report.txt').resolve()}")
print("CSV summaries saved under results/.")
