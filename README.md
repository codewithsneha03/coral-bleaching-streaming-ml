# Adaptive Streaming Machine Learning for Coral Bleaching Prediction

Predicting coral bleaching events sequentially over time using historical observations from the **Global Coral Bleaching Database (GCBD, 1980–2020)** and environmental measurements with online streaming machine learning (River).

---

## 🌊 Project Overview

Coral reefs worldwide face severe thermal stress from changing ocean temperatures, leading to mass bleaching events. While traditional batch machine learning models assume static relationships trained on fixed historical datasets, real-world marine environments evolve dynamically over time.

This project implements an **online, streaming machine learning system**:
- Observations arrive chronologically (1980–2016).
- The model evaluates each observation **before learning** (*prequential evaluation* / *test-then-train*).
- Both a baseline model (Online Logistic Regression) and an **Adaptive Random Forest (ARF)** ensemble are compared.
- Evaluated on precision, recall, F1 score, and adaptability across decades.

---

## 📊 Pipeline Architecture & Scripts

The pipeline is organized in structured, reproducible stages under `src/`:

```text
coral_bleaching_project/
├── data/
│   ├── raw/                 # Downloaded GCBD SQLite database (excluded from git)
│   └── processed/           # Cleaned dataset & chronological stream CSVs
├── src/
│   ├── 01_inspect_database.py             # Schema inspection & table row counts
│   ├── 02_profile_tables.py               # Field distributions & key checks
│   ├── 02b_investigate_duplicates_and_nulls.py  # Deep investigation of duplicates & target rescue
│   ├── 03_build_dataset.py                # De-duplication, label construction & multi-table joins
│   ├── 04_prepare_stream.py               # Chronological ordering & online-safe median imputation
│   ├── 05_train_baseline.py               # Online Logistic Regression (prequential evaluation)
│   └── 06_train_arf.py                    # Adaptive Random Forest (ARF) classifier
├── results/
│   ├── database_schema.txt                # SQLite schema report
│   ├── table_profile_report.txt           # Profiling of Bleaching, Environmental, & Sample tables
│   ├── duplicate_investigation_report.txt # Resolution of duplicate Sample_IDs
│   ├── data_quality_report.txt            # Dataset build metrics & leakage audit
│   ├── prediction_problem_definition.txt  # Formal target & feature definition
│   ├── stream_preparation_report.txt      # Chronological stream statistics
│   ├── baseline_metrics.txt               # Logistic Regression prequential performance
│   └── arf_metrics.txt                    # Adaptive Random Forest prequential performance
├── requirements.txt
└── README.md
```

---

## 🔬 Key Methodological Findings & Decisions

1. **Duplicate Resolution:**
   - `Bleaching_tbl` contained 11,041 excess rows compared to unique `Sample_ID`s.
   - Investigation proved 11,038 cases were identical duplicate entries sharing the same severity code.
   - De-duplicated deterministically by taking the maximum severity and minimum row ID.

2. **Target Label Construction:**
   - Explicit `Severity_Code` only covered ~20% of records.
   - By cross-referencing `Percent_Bleached`, an additional 5,472 records were safely rescued into binary classes (`1` = Bleached, `0` = Not Bleached).
   - Ambiguous codes (`Severity_Code = -1`) and records missing both metrics were excluded to prevent label corruption.
   - Resulting dataset: **11,889 high-confidence observations** across **7,099 reef sites** and **73 countries**.

3. **Leakage Prevention & Problem Framing:**
   - Framed honestly as **contemporaneous nowcasting** (predicting bleaching status from co-occurring thermal and environmental features), rather than unverified advance forecasting.
   - All outcome columns (`Severity_Code`, `Percent_Bleached`, etc.) were stripped from the feature matrix.

4. **Stream Preparation:**
   - 41 numeric predictors across thermal stress (SST, SSTA, Degree Heating Weeks, TSA) and site characteristics (depth, distance to shore, exposure, turbidity, cyclone frequency).
   - Sorted strictly chronologically without random shuffling.

---

## 📈 Model Performance (Prequential Evaluation)

Evaluated sequentially across all 11,889 observations using test-then-train protocol:

| Metric | Naive Baseline (Always Bleached) | Baseline (Online Logistic Regression) | Adaptive Random Forest (ARF, 10 Trees) |
|---|:---:|:---:|:---:|
| **Accuracy** | 85.52% | 89.85% | **92.45%** |
| **Precision** | 85.52% | 92.59% | **93.97%** |
| **Recall** | 100.0% | 95.80% | **97.41%** |
| **F1 Score** | 92.19% | 94.17% | **95.66%** |
| **Improvement over Naive** | — | +4.33% | **+6.93%** |

---

## 🚀 Quick Start & Reproduction

### 1. Environment Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Add the Database
Place `Global_Coral_Bleaching_Database_SQLite_11_24_21.db` in `data/raw/`.

### 3. Run Pipeline
```powershell
# Profile & investigate database
python src/01_inspect_database.py data/raw/Global_Coral_Bleaching_Database_SQLite_11_24_21.db
python src/02_profile_tables.py data/raw/Global_Coral_Bleaching_Database_SQLite_11_24_21.db
python src/02b_investigate_duplicates_and_nulls.py

# Construct dataset & streaming order
python src/03_build_dataset.py
python src/04_prepare_stream.py

# Train & evaluate models
python src/05_train_baseline.py
python src/06_train_arf.py
```
