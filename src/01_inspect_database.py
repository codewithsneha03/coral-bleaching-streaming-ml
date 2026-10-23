"""
Step 1: Inspect the real SQLite database schema.
This script does not modify the database or assume any table/column names.
"""

import sqlite3
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/01_inspect_database.py path/to/GCBD.sqlite")
        raise SystemExit(1)

    database_path = Path(sys.argv[1])

    if not database_path.is_file():
        print(f"ERROR: Database file not found: {database_path}")
        print("Place the GCBD SQLite file in data/raw/ and pass its exact filename.")
        raise SystemExit(1)

    output_path = Path("results/database_schema.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "CORAL BLEACHING PROJECT — DATABASE SCHEMA REPORT",
        f"Database: {database_path.resolve()}",
        "",
    ]

    try:
        # Open in read-only mode so the original database cannot be changed.
        connection = sqlite3.connect(f"file:{database_path.resolve().as_posix()}?mode=ro", uri=True)
        cursor = connection.cursor()

        cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        tables = [row[0] for row in cursor.fetchall()]

        if not tables:
            report_lines.append("No user tables were found.")
        else:
            report_lines.append(f"Number of tables: {len(tables)}")
            report_lines.append("TABLE LIST")
            report_lines.extend(f" - {table}" for table in tables)
            report_lines.append("")

            for table in tables:
                # Table names come from SQLite's own table list; quote safely.
                quoted_table = '"' + table.replace('"', '""') + '"'
                cursor.execute(f"PRAGMA table_info({quoted_table})")
                columns = cursor.fetchall()

                report_lines.append("=" * 72)
                report_lines.append(f"TABLE: {table}")
                report_lines.append("Columns: cid | name | type | notnull | default | primary_key")
                for col in columns:
                    report_lines.append(
                        f"{col[0]} | {col[1]} | {col[2]} | {col[3]} | {col[4]} | {col[5]}"
                    )

                cursor.execute(f"SELECT COUNT(*) FROM {quoted_table}")
                row_count = cursor.fetchone()[0]
                report_lines.append(f"Row count: {row_count}")

                # Show a tiny sample to understand representation, without exporting all records.
                cursor.execute(f"SELECT * FROM {quoted_table} LIMIT 3")
                sample_rows = cursor.fetchall()
                report_lines.append("Sample rows (up to 3):")
                for row in sample_rows:
                    report_lines.append(f"  {row!r}")
                report_lines.append("")

        connection.close()
    except sqlite3.Error as exc:
        print(f"SQLite error: {exc}")
        raise SystemExit(1)

    report = "\n".join(report_lines)
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSchema report saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
