# Adaptive Streaming Machine Learning for Coral Bleaching Prediction

Predicting coral bleaching events sequentially over time using historical observations from the **Global Coral Bleaching Database (GCBD, 1980–2020)** and environmental measurements with online streaming machine learning (River), combined with live satellite telemetry from **NOAA Coral Reef Watch**.

---

## 🌊 Project Overview

Coral reefs worldwide face severe thermal stress from changing ocean temperatures, leading to mass bleaching events. While traditional batch machine learning models assume static relationships trained on fixed historical datasets, real-world marine environments evolve dynamically over time.

This project implements an **online, streaming machine learning system**:
- Observations arrive chronologically (1980–2016).
- The model evaluates each observation **before learning** (*prequential evaluation* / *test-then-train*).
- Both a baseline model (Online Logistic Regression) and an **Adaptive Random Forest (ARF)** ensemble are compared.
- **Concept drift** is actively monitored using the **ADWIN** algorithm, aligning statistical alerts with major real-world global coral bleaching events (1998, 2002, 2005, 2010, 2015).
- **Real-time integration** with NOAA Coral Reef Watch (CRW) 5km daily satellite products via PacIOOS ERDDAP.

---

## 📊 Pipeline Architecture & Scripts

The pipeline is organized in structured, reproducible stages under `src/`:

```text
coral_bleaching_project/
├── data/
│   ├── raw/                 # Downloaded GCBD SQLite database (excluded from git)
│   └── processed/           # Cleaned dataset & chronological stream CSVs
├── docs/
│   └── PROJECT_EXPLANATION.md  # Comprehensive technical & methodological report
├── src/
│   ├── 01_inspect_database.py             # Schema inspection & table row counts
│   ├── 02_profile_tables.py               # Field distributions & key checks
│   ├── 02b_investigate_duplicates_and_nulls.py  # Deep investigation of duplicates & target rescue
│   ├── 03_build_dataset.py                # De-duplication, label construction & multi-table joins
│   ├── 04_prepare_stream.py               # Chronological ordering & online-safe median imputation
│   ├── 05_train_baseline.py               # Online Logistic Regression (prequential evaluation)
│   ├── 06_train_arf.py                    # Initial Adaptive Random Forest (10 trees)
│   ├── 07_train_arf_tuned.py              # Tuned ARF (25 trees) & ADWIN concept drift monitoring
│   ├── 08_noaa_live_connector.py          # Real-time satellite data fetcher (PacIOOS ERDDAP)
│   └── 09_visualize_results.py            # Generates publication-ready figures & diagnostic plots
├── results/
│   ├── figures/                           # Generated evaluation plots (PNG)
│   ├── database_schema.txt                # SQLite schema report
│   ├── table_profile_report.txt           # Profiling of Bleaching, Environmental, & Sample tables
│   ├── duplicate_investigation_report.txt # Resolution of duplicate Sample_IDs
│   ├── data_quality_report.txt            # Dataset build metrics & leakage audit
│   ├── prediction_problem_definition.txt  # Formal target & feature definition
│   ├── stream_preparation_report.txt      # Chronological stream statistics
│   ├── baseline_metrics.txt               # Logistic Regression prequential performance
│   ├── arf_metrics.txt                    # Initial ARF prequential performance
│   ├── model_comparison.txt               # Multi-model comparison report
│   └── drift_events.csv                   # Historical concept drift events logged by ADWIN
├── requirements.txt
└── README.md
```

---

## 📈 Model Performance Benchmark (Prequential Evaluation)

Evaluated sequentially across all 11,889 observations under the strict test-then-train protocol:

| Model / Configuration | Accuracy | Precision (Bleached) | Recall (Bleached) | F1 Score | Improvement over Naive |
|---|:---:|:---:|:---:|:---:|:---:|
| **Naive Baseline (Always Bleached)** | 85.52% | 85.52% | 100.00% | 92.19% | — |
| **Baseline: Online Logistic Regression** | 89.85% | 92.59% | 95.80% | 94.17% | +4.33% |
| **ARF Initial (10 Trees)** | 92.45% | 93.97% | 97.41% | 95.66% | +6.93% |
| **ARF Tuned (25 Trees, Grace=30)** | **93.18%** | **94.37%** | **97.87%** | **96.08%** | **+7.66%** |

### 🔍 Concept Drift Detection via ADWIN
Our error-stream ADWIN detector captured 9 statistically significant concept drift events, independently matching historical global marine heatwaves:
- **1999:** Post-1998 1st Global Bleaching Event
- **2002:** Great Barrier Reef Mass Bleaching Event
- **2005:** Caribbean Mass Bleaching Event
- **2006, 2008, 2010:** 2nd Global Bleaching Event (Indian Ocean & Coral Triangle)
- **2012, 2015:** 3rd Global Bleaching Event (2014–2017 Pan-tropical Bleaching)

---

## 🛰️ Real-Time Telemetry: NOAA Coral Reef Watch
[`src/08_noaa_live_connector.py`](file:///c:/Users/Sneha/Downloads/coral_bleaching_project_starter/coral_bleaching_project/src/08_noaa_live_connector.py) pulls daily 5km satellite parameters directly via the PacIOOS ERDDAP server:
- `CRW_SST` (Sea Surface Temperature)
- `CRW_SSTANOMALY` (SST Departure)
- `CRW_HOTSPOT` (Immediate thermal stress above threshold)
- `CRW_DHW` (Degree Heating Weeks)
- `CRW_BAA` (Bleaching Alert Area Level)

Mapped directly into our trained ARF feature space to estimate real-time bleaching risk for key global reef provinces (Great Barrier Reef, Florida Keys, Maldives, Fiji, Red Sea, Hawaii, Palau, Belize).

---

## 🚀 Quick Start & Reproduction

### 1. Environment Setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Add Database
Place `Global_Coral_Bleaching_Database_SQLite_11_24_21.db` in `data/raw/`.

### 3. Run Pipeline
```powershell
# 1. Profile & investigate database
python src/01_inspect_database.py data/raw/Global_Coral_Bleaching_Database_SQLite_11_24_21.db
python src/02_profile_tables.py data/raw/Global_Coral_Bleaching_Database_SQLite_11_24_21.db
python src/02b_investigate_duplicates_and_nulls.py

# 2. Construct dataset & streaming order
python src/03_build_dataset.py
python src/04_prepare_stream.py

# 3. Train models
python src/05_train_baseline.py
python src/06_train_arf.py
python src/07_train_arf_tuned.py

# 4. Live NOAA telemetry & visual analytics
python src/08_noaa_live_connector.py
python src/09_visualize_results.py
```

For full architectural and mathematical details, see [docs/PROJECT_EXPLANATION.md](file:///c:/Users/Sneha/Downloads/coral_bleaching_project_starter/coral_bleaching_project/docs/PROJECT_EXPLANATION.md).
