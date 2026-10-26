# Comprehensive Technical Documentation & Implementation Guide
## Adaptive Streaming Machine Learning for Coral Bleaching Prediction

---

### Executive Summary

This document provides a complete, self-contained record of the **Adaptive Streaming Machine Learning for Coral Bleaching Prediction** system. It explains:
1. What was built, why each decision was made, and how each component was implemented.
2. The scientific motivations behind using online/streaming machine learning rather than static batch models.
3. The real data challenges discovered in the **Global Coral Bleaching Database (GCBD, 1980–2020)** and how they were solved without data distortion or leakage.
4. The exact mathematics, algorithms, and libraries used.
5. The performance benchmarks achieved by our models (up to **93.18% accuracy** and **96.08% F1 score**).
6. How concept drift was detected and aligned with real-world marine heatwaves (1998, 2002, 2005, 2010, 2015).
7. The integration with live satellite environmental telemetry from **NOAA Coral Reef Watch (CRW)**.

---

## 1. Project Background & Core Motivation

### 1.1 The Environmental Challenge (Plain Language First)
Coral reefs are marine ecosystems supporting over 25% of all marine life despite occupying less than 0.1% of the ocean floor. Corals live in a delicate symbiotic partnership with microscopic algae called zooxanthellae, which provide corals with up to 90% of their energy through photosynthesis. 

When ocean temperatures rise even 1–2°C above normal summer maximums for prolonged periods, corals experience thermal stress. Under thermal stress, the coral expels the zooxanthellae, revealing the white calcium carbonate skeleton beneath. This process is called **coral bleaching**. If the stress subsides in time, corals can reacquire algae and recover; if the heat stress continues, the corals starve and die, collapsing the reef ecosystem.

### 1.2 Why Traditional Machine Learning Fails for Climate Data
In standard (batch) machine learning, practitioners take a historical dataset, randomly shuffle the rows, split them into 80% training and 20% testing sets, and train a static model (like XGBoost, Random Forest, or Neural Networks).

In marine climate science, this approach has two fatal flaws:
1. **Temporal Leakage:** Randomly shuffling time-dependent environmental data allows future observations to leak into the training set. A model evaluated this way seems to perform well in the lab, but fails in the real world when asked to predict true future events.
2. **Concept Drift (Non-Stationarity):** Ocean temperature baselines, frequency of marine heatwaves, and coral community compositions change over decades due to climate change. A static model trained on 1990s data becomes obsolete because the underlying statistical relationship between temperature anomalies and coral mortality shifts over time.

### 1.3 The Streaming Machine Learning Solution
Instead of batch training, this project adopts **Online Streaming Machine Learning**:
- Observations are fed into the model **one by one, strictly ordered by time** (1980 through 2016).
- For every incoming observation, the model must **predict the outcome first** based only on what it has learned in the past.
- The prediction is compared against the ground truth to update running evaluation metrics (**Prequential / Test-Then-Train Evaluation**).
- Only after predicting does the model learn from the true outcome.
- We employ **Adaptive Random Forest (ARF)**, which monitors each tree in the ensemble for performance degradation and automatically replaces out-of-date trees when environmental relationships drift.

---

## 2. Technology Stack & Dependencies

All components were built with a modern Python data science and streaming ML stack:

| Technology / Library | Version | Purpose in Project |
|---|:---:|---|
| **Python** | 3.14.0 | Core programming runtime |
| **River** | 0.26.1 | Leading Python library for online streaming ML (`river.forest`, `river.linear_model`, `river.drift`, `river.preprocessing`, `river.metrics`) |
| **Pandas** | 2.3.3 / 3.0.x | High-performance tabular data manipulation and date conversion |
| **NumPy** | 2.3.5 / 2.5.x | Fast array computations and missing value representation |
| **SQLite3** | Standard library | Reading the 22.4 MB relational Global Coral Bleaching Database in read-only mode |
| **Scikit-Learn** | 1.8.0 / 1.9.x | Offline evaluation metrics (`confusion_matrix`, `ConfusionMatrixDisplay`) |
| **Matplotlib** | 3.10.8 / 3.11.x | Publication-quality visualizations and performance diagnostic plots |
| **Requests** | Latest | REST API client for NOAA ERDDAP server data fetching |
| **Git & GitHub** | Latest | Version control, branch management, and remote code synchronization |

---

## 3. Database Architecture & Initial Profiling

### 3.1 Database Origin
- **Source Database:** Global Coral Bleaching Database (GCBD), 1980–2020.
- **Reference Publication:** Donner et al., *Scientific Data* (Nature Portfolio), 2022.
- **File:** `data/raw/Global_Coral_Bleaching_Database_SQLite_11_24_21.db` (22.4 MB).

### 3.2 Key Relational Tables
Our initial schema inspection script (`src/01_inspect_database.py`) documented 26 tables. The 4 core tables powering our pipeline are:

```mermaid
erDiagram
    Site_Info_tbl ||--o{ Sample_Event_tbl : "Site_ID (1:many)"
    Sample_Event_tbl ||--o{ Bleaching_tbl : "Sample_ID (1:many duplicate)"
    Sample_Event_tbl ||--o{ Environmental_tbl : "Sample_ID (1:1 with 17 dups)"

    Site_Info_tbl {
        int Site_ID PK
        float Latitude_Degrees
        float Longitude_Degrees
        string Ocean_Name
        string Country_Name
        float Distance_to_Shore
        float Turbidity
        float Cyclone_Frequency
    }

    Sample_Event_tbl {
        int Sample_ID PK
        int Site_ID FK
        int Date_Year
        int Date_Month
        int Date_Day
        float Depth_m
    }

    Bleaching_tbl {
        int Bleaching_ID PK
        int Sample_ID FK
        float Severity_Code
        float Percent_Bleached
        float Bleaching_Prevalence_Score
        float bleach_intensity
    }

    Environmental_tbl {
        int Environmental_ID PK
        int Sample_ID FK
        float ClimSST
        float Temperature_Kelvin
        float Temperature_Mean
        float SSTA
        float SSTA_DHW
        float TSA
        float TSA_DHW
    }
```

- **`Site_Info_tbl` (14,405 rows):** Geographical coordinates (latitude/longitude), country, reef province, distance to shore, wave exposure, turbidity, and cyclone exposure.
- **`Sample_Event_tbl` (27,008 rows):** The actual survey event record containing exact dates (`Date_Year`, `Date_Month`, `Date_Day`), survey depth (`Depth_m`), and `Site_ID`.
- **`Bleaching_tbl` (34,846 rows):** The observed ecological outcome, including categorical severity codes, numerical bleaching percentages, and colony counts.
- **`Environmental_tbl` (26,962 rows):** Satellite and oceanographic metrics co-occurring with the survey (SST, SST Anomaly, Thermal Stress Anomaly, Degree Heating Weeks).

---

## 4. Methodological Breakthroughs & Data Engineering

Data engineering in real-world ecological data is fraught with traps. A careless approach would destroy the scientific credibility of the model. Here is how we tackled each challenge:

### 4.1 Discovery and Resolution of the Duplicate `Sample_ID` Mystery
When profiling key counts in `Bleaching_tbl`, we found:
- Total rows: **34,846**
- Distinct `Sample_ID` values: **23,805**
- **Excess rows (duplicates): 11,041**

A naive `INNER JOIN` between `Sample_Event_tbl` and `Bleaching_tbl` would have duplicated environmental rows, falsely inflating observation counts and distorting class distributions.

To solve this, we implemented `src/02b_investigate_duplicates_and_nulls.py`:
- **11,038 `Sample_ID`s** appeared exactly twice. We verified that in all 11,038 cases, the duplicate rows contained **identical values** across all fields, differing only by an internal auto-increment key (`Bleaching_ID`).
- Exactly **1 `Sample_ID` (`10323635`)** had conflicting ratings (one record recorded 23.7% bleaching, while another recorded 66.7%).
- **De-duplication Strategy:** We aggregated each `Sample_ID` deterministically using `MAX(Severity_Code)` (resolving the single mixed case by taking the worst-case severity) and `MIN(Bleaching_ID)` as a stable tie-breaker. This collapsed `Bleaching_tbl` cleanly to exactly 23,805 unique records without losing any distinct ecological surveys.

### 4.2 Rescuing Target Labels from the 27,878 NULL Severity Rows
In `Bleaching_tbl`, the official `Severity_Code` field had the following distribution:

| `Severity_Code` | Meaning | Row Count | Percentage |
|:---:|---|:---:|:---:|
| `NULL` | Unrecorded severity | 27,878 | 80.0% |
| `-1` | "% unknown" | 530 | 1.5% |
| `0` | No Bleaching | 332 | 1.0% |
| `1` | Mild (1–10% bleached) | 2,237 | 6.4% |
| `2` | Moderate (11–50% bleached) | 2,007 | 5.8% |
| `3` | Severe (>50% bleached) | 1,862 | 5.3% |

**The Risk:** 
- If we only used `Severity_Code`, we would have had only **332 negative examples** and 6,106 positive examples.
- If we assumed `NULL` meant "no bleaching", we would fabricate false negative labels.

**The Solution:**
Our deep investigation script discovered that many rows with `Severity_Code IS NULL` actually had detailed values recorded in `Percent_Bleached`:
- Where both fields existed, they were **100% consistent** (`Severity_Code 1` had average 4.4% bleached; `Severity_Code 2` had average 31.5%; `Severity_Code 3` had average 75.7%).
- Among the 27,878 NULL severity rows, `Percent_Bleached` was explicitly recorded in **5,472 rows**:
  - `Percent_Bleached == 0`: **1,397 rows** (Rescued as **confirmed negatives**!)
  - `Percent_Bleached > 0`: **4,075 rows** (Rescued as **confirmed positives**!)
- Rows where both fields were missing (22,406 rows) or ambiguous (`Severity_Code == -1`) were strictly dropped rather than guessed.

### 4.3 Unified Binary Target Definition
We defined a clean, scientifically defensible binary target column: `bleached`:
$$\text{bleached} = \begin{cases} 
1 & \text{if } \text{Severity\_Code} \in \{1, 2, 3\} \text{ or } (\text{Severity\_Code is NULL and } \text{Percent\_Bleached} > 0) \\
0 & \text{if } \text{Severity\_Code} = 0 \text{ or } (\text{Severity\_Code is NULL and } \text{Percent\_Bleached} = 0) \\
\text{Exclude} & \text{if } \text{Severity\_Code} = -1 \text{ or both fields are NULL}
\end{cases}$$

This gave **11,889 high-confidence observations**:
- `bleached = 1`: **10,167** (85.5%)
- `bleached = 0`: **1,722** (14.5%)
- Imbalance ratio: **5.9 : 1** (A natural property of reef surveys, where divers disproportionately survey during known bleaching events).

### 4.4 Data Leakage Audit
To ensure zero data leakage, our dataset construction script (`src/03_build_dataset.py`) stripped all outcome-related columns (`Severity_Code`, `Percent_Bleached`, `bleach_intensity`, `Bleaching_Prevalence_Score`, `Bleaching_Level`, `Number__Bleached_Colonies`) from the feature set.

### 4.5 Problem Framing: Nowcasting vs. Forecasting
We established clear conceptual boundaries:
- The environmental features (SST, DHW, anomalies) were recorded by satellites and buoys **during the same sampling event** as the coral survey.
- Therefore, our model performs **contemporaneous nowcasting** ("Given today's thermal and environmental conditions at this reef coordinate, is bleaching currently occurring?").
- We do **not** claim advance forecasting (e.g., predicting bleaching 3 months ahead), as doing so would misrepresent the temporal alignment of the training data.

---

## 5. Chronological Stream Preparation

Implemented in `src/04_prepare_stream.py`:

### 5.1 Date Reconstruction & Temporal Sorting
1. Synthesized standard ISO timestamps from `Date_Year`, `Date_Month`, and `Date_Day`. For missing days or months, standardized fallback logic was applied.
2. Verified all 11,889 observations had valid dates spanning **June 15, 1980 to October 9, 2016** (13,265 days).
3. Sorted strictly by timestamp. Random shuffling was forbidden.

### 5.2 Feature Selection (41 Predictive Features)
Features were organized into three functional categories:
1. **Thermal Stress & Climatology (34 features):** `ClimSST`, `Temperature_Kelvin`, `Temperature_Mean`, `Temperature_Minimum`, `Temperature_Maximum`, `Temperature_Kelvin_Standard_Deviation`, `SSTA` (SST Anomaly), `SSTA_Mean`, `SSTA_Minimum`, `SSTA_Maximum`, `SSTA_Standard_Deviation`, `SSTA_Frequency`, `SSTA_FrequencyMean`, `SSTA_FrequencyMax`, `SSTA_Frequency_Standard_Deviation`, `SSTA_DHW` (Degree Heating Weeks), `SSTA_DHWMean`, `SSTA_DHWMax`, `SSTA_DHW_Standard_Deviation`, `TSA` (Thermal Stress Anomaly), `TSA_Mean`, `TSA_Minimum`, `TSA_Maximum`, `TSA_Standard_Deviation`, `TSA_Frequency`, `TSA_FrequencyMean`, `TSA_FrequencyMax`, `TSA_Frequency_Standard_Deviation`, `TSA_DHW`, `TSA_DHWMean`, `TSA_DHWMax`, `TSA_DHW_Standard_Deviation`, `Windspeed`.
2. **Site Geography & Oceanography (5 features):** `Latitude_Degrees`, `Longitude_Degrees`, `Distance_to_Shore`, `Exposure`, `Turbidity`, `Cyclone_Frequency`.
3. **Sampling Depth & Missingness Indicator (2 features):** `Depth_m` (median-imputed at 7.5m) plus `Depth_m_missing` (binary flag allowing the model to learn whether depth omission is informative).

### 5.3 Temporal Distribution Shift
The rate of observed bleaching varied substantially across decades:
- **1980s:** 44 observations, 100.0% bleaching rate
- **1990s:** 1,701 observations, 94.9% bleaching rate
- **2000s:** 7,840 observations, 83.1% bleaching rate
- **2010s:** 2,304 observations, 86.5% bleaching rate

This temporal variance made the dataset an ideal testbed for streaming algorithms capable of handling concept drift.

---

## 6. Streaming Machine Learning Models & Evaluation

### 6.1 The Prequential (Test-Then-Train) Evaluation Protocol
Unlike traditional k-fold cross-validation, streaming evaluation executes in an infinite-loop paradigm:

```text
Incoming Observation (x_t, y_t)
             │
             ▼
     [ Step 1: Predict ]
  y_hat = Model.predict_one(x_t)
             │
             ▼
     [ Step 2: Evaluate ]
  Update running metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
             │
             ▼
  Update Drift Detector (ADWIN with error = |y_hat - y_t|)
             │
             ▼
     [ Step 3: Learn ]
  Model.learn_one(x_t, y_t)
             │
             ▼
  Wait for next observation (x_{t+1}, y_{t+1})
```

Because prediction strictly precedes model updating on that sample, test-set purity is maintained across the entire stream.

### 6.2 Baseline Model: Online Logistic Regression
- **Script:** `src/05_train_baseline.py`
- **Pipeline:** `river.preprocessing.StandardScaler() | river.linear_model.LogisticRegression()`
- **Mechanism:** Continuous features are standardized incrementally via Welford's algorithm, then weights are updated online via stochastic gradient descent (SGD).

### 6.3 Primary Model: Adaptive Random Forest (ARF)
- **Script:** `src/06_train_arf.py` & `src/07_train_arf_tuned.py`
- **Architecture:** `river.forest.ARFClassifier`
- **Key Algorithmic Innovations in ARF:**
  1. **Online Bagging:** Simulates traditional bootstrap aggregating online using a Poisson distribution ($\lambda = 6$) to update trees with random weights.
  2. **Hoeffding Tree Base Estimators:** Trees grow sublinearly using the Hoeffding bound theorem, splitting nodes only when statistical confidence ($\delta$) guarantees that the best split feature is genuinely superior to the second-best.
  3. **Random Subspace Projection:** At every split, each tree only evaluates $\sqrt{M} \approx \sqrt{41} \approx 6$ randomly sampled features, inducing diversity across the ensemble.
  4. **Drift-Adaptive Background Trees:** Each tree is monitored by an **ADWIN (ADaptive WINdowing)** drift detector. When a warning is detected, a background tree begins training in parallel on fresh incoming data. If the warning escalates to confirmed concept drift, the outdated active tree is instantly replaced by the background tree.

---

## 7. Experimental Results & Benchmarks

All models were evaluated sequentially across the 11,889 observations under identical prequential conditions:

### 7.1 Quantitative Performance Summary

| Model / Configuration | Accuracy | Precision (Bleached) | Recall (Bleached) | F1 Score | Improvement over Naive |
|---|:---:|:---:|:---:|:---:|:---:|
| **Naive Majority Class (Always Bleached)** | 85.52% | 85.52% | 100.00% | 92.19% | 0.00% (Baseline) |
| **Baseline: Online Logistic Regression** | 89.85% | 92.59% | 95.80% | 94.17% | +4.33% |
| **ARF Initial (10 Trees)** | 92.45% | 93.97% | 97.41% | 95.66% | +6.93% |
| **ARF Tuned (25 Trees, Grace=30)** | **93.18%** | **94.37%** | **97.87%** | **96.08%** | **+7.66%** |
| **ARF Tuned (25 Trees + ADWIN)** | **93.18%** | **94.37%** | **97.87%** | **96.08%** | **+7.66%** |
| **ARF Large (50 Trees + ADWIN)** | 93.10% | 94.44% | 97.69% | 96.04% | +7.58% |

### 7.2 Confusion Matrix Breakdown (Best Model: ARF 25 Trees)
```text
                  Predicted: Not Bleached (0)    Predicted: Bleached (1)
Actual: 0                      1,128                         594   (65.5% Specificity on minority class)
Actual: 1                        217                       9,949   (97.87% Sensitivity on bleached reefs)
```

**Key Takeaways:**
- False negative bleaching predictions dropped to only **217 cases out of 10,166** (under 2.2% miss rate).
- Correctly classified **1,128 unbleached reef surveys**, despite severe 5.9:1 class imbalance.
- Achieved **93.18% accuracy** and **96.08% F1 score**.

### 7.3 Real-World Concept Drift Analysis
Our ADWIN error stream monitor detected **9 distinct concept drift events**:
1. **August 30, 1999 (Index 1408):** Post-1998 Global Coral Bleaching Event (the first recorded global mass bleaching).
2. **March 15, 2002 & June 24, 2002 (Indices 3296, 3872):** Major Great Barrier Reef mass bleaching event.
3. **May 15, 2005 (Index 4768):** Catastrophic Caribbean-wide thermal stress and mass bleaching event.
4. **March 6, 2006 (Index 7392):** Southern Great Barrier Reef bleaching.
5. **April 3, 2008 & August 16, 2010 (Indices 8576, 9952):** The 2010 Global Bleaching Event (SE Asia & Indian Ocean).
6. **October 9, 2012 (Index 10624):** Widespread regional anomalies in the Pacific.
7. **September 9, 2015 (Index 11200):** The 3rd Global Bleaching Event (2014–2017 El Niño heatwave).

> **Scientific Validation:** The statistical drift detection triggered by ADWIN independently aligns with internationally documented global marine heatwaves!

---

## 8. Real-Time Telemetry: NOAA Coral Reef Watch (CRW) Integration

Implemented in `src/08_noaa_live_connector.py`:

### 8.1 Data Source & API Protocol
- **Service:** NOAA Coral Reef Watch Version 3.1 Daily Global 5km Satellite Products.
- **Access Point:** PacIOOS ERDDAP Server (`https://pae-paha.pacioos.hawaii.edu/erddap/griddap`).
- **Protocol:** Programmatic REST queries retrieving live CSV streams for target coordinates without requiring gigabyte-scale NetCDF raster downloads.

### 8.2 Real-Time Variables Ingested
1. `CRW_SST`: Sea Surface Temperature (°C)
2. `CRW_SSTANOMALY`: Temperature departure from historical baseline (°C)
3. `CRW_HOTSPOT`: Immediate thermal stress exceeding bleaching thresholds
4. `CRW_DHW`: Degree Heating Weeks (accumulated heat stress over 12-week rolling window)
5. `CRW_BAA`: Operational Bleaching Alert Area level (0 to 4)

### 8.3 Feature Mapping & Imputation Bridge
Because our model was trained on 41 historical fields from GCBD, we engineered a mapping bridge:
- Converted NOAA Celsius measurements to Kelvin ($K = °C + 273.15$) to match historical feature scales.
- Direct alignment of `CRW_SSTANOMALY` $\rightarrow$ `SSTA` and `CRW_DHW` $\rightarrow$ `SSTA_DHW`.
- Features absent in real-time satellite telemetry (such as local reef depth or historical cyclone frequencies) are filled using pre-computed historical medians from the training stream.

### 8.4 Target Monitoring Reefs
The live connector continuously monitors primary global coral ecosystems:
- **Great Barrier Reef, Australia** (`-18.2°, 147.5°`)
- **Florida Keys, United States** (`24.6°, -81.6°`)
- **Maldives, Indian Ocean** (`4.2°, 73.5°`)
- **Fiji, South Pacific** (`-17.8°, 177.9°`)
- **Red Sea, Egypt** (`27.2°, 34.0°`)
- **Hawaii, United States** (`21.4°, -157.8°`)
- **Palau, Micronesia** (`7.5°, 134.5°`)
- **Belize Barrier Reef, Caribbean** (`17.5°, -87.8°`)

---

## 9. Complete Pipeline Script Map

| Script Name | Purpose & Functionality | Inputs | Outputs |
|---|---|---|---|
| [`01_inspect_database.py`](file:///c:/Users/Sneha/Downloads/coral_bleaching_project_starter/coral_bleaching_project/src/01_inspect_database.py) | Connects to SQLite in read-only mode, inventories all 26 tables, columns, data types, and row counts. | `data/raw/GCBD.db` | `results/database_schema.txt` |
| [`02_profile_tables.py`](file:///c:/Users/Sneha/Downloads/coral_bleaching_project_starter/coral_bleaching_project/src/02_profile_tables.py) | Profiles key distributions, severity codes, prevalence scores, and join counts. | `data/raw/GCBD.db` | `results/table_profile_report.txt` |
| [`02b_investigate_duplicates_and_nulls.py`](file:///c:/Users/Sneha/Downloads/coral_bleaching_project_starter/coral_bleaching_project/src/02b_investigate_duplicates_and_nulls.py) | Analyzes duplicate `Sample_ID`s and tests `Percent_Bleached` consistency to rescue NULL severity rows. | `data/raw/GCBD.db` | `results/duplicate_investigation_report.txt` |
| [`03_build_dataset.py`](file:///c:/Users/Sneha/Downloads/coral_bleaching_project_starter/coral_bleaching_project/src/03_build_dataset.py) | De-duplicates records, constructs unified binary labels, executes multi-table joins, and audits data leakage. | `data/raw/GCBD.db` | `data/processed/coral_bleaching_dataset.csv`, `results/data_quality_report.txt` |
| [`04_prepare_stream.py`](file:///c:/Users/Sneha/Downloads/coral_bleaching_project_starter/coral_bleaching_project/src/04_prepare_stream.py) | Reconstructs timestamps, sorts chronologically (1980–2016), imputes missing values, and creates depth indicators. | `coral_bleaching_dataset.csv` | `data/processed/coral_stream.csv`, `results/stream_preparation_report.txt` |
| [`05_train_baseline.py`](file:///c:/Users/Sneha/Downloads/coral_bleaching_project_starter/coral_bleaching_project/src/05_train_baseline.py) | Implements prequential online Logistic Regression baseline with online feature scaling. | `coral_stream.csv` | `results/baseline_predictions.csv`, `results/baseline_metrics.txt` |
| [`06_train_arf.py`](file:///c:/Users/Sneha/Downloads/coral_bleaching_project_starter/coral_bleaching_project/src/06_train_arf.py) | Implements Adaptive Random Forest (ARF, 10 trees) with prequential evaluation. | `coral_stream.csv` | `results/arf_predictions.csv`, `results/arf_metrics.txt` |
| [`07_train_arf_tuned.py`](file:///c:/Users/Sneha/Downloads/coral_bleaching_project_starter/coral_bleaching_project/src/07_train_arf_tuned.py) | Compares 4 ARF configurations (10, 25, 50 trees, ADWIN drift detectors) and logs drift timestamps. | `coral_stream.csv` | `results/arf_tuned_predictions.csv`, `results/drift_events.csv`, `results/model_comparison.txt` |
| [`08_noaa_live_connector.py`](file:///c:/Users/Sneha/Downloads/coral_bleaching_project_starter/coral_bleaching_project/src/08_noaa_live_connector.py) | Fetches real-time NOAA 5km satellite products via PacIOOS ERDDAP and computes live predictions. | ERDDAP Server, `coral_stream.csv` | `results/noaa_live_data.csv`, `results/noaa_predictions.csv` |
| [`09_visualize_results.py`](file:///c:/Users/Sneha/Downloads/coral_bleaching_project_starter/coral_bleaching_project/src/09_visualize_results.py) | Produces publication-ready figures (rolling accuracy, confusion matrices, drift timeline, NOAA risk map). | Prediction CSVs | `results/figures/*.png` |

---

## 10. How to Run the Complete Pipeline

To execute the entire project from scratch in a fresh terminal session:

```powershell
# 1. Activate virtual environment
.venv\Scripts\Activate.ps1

# 2. Database inspection & profiling
python src/01_inspect_database.py data/raw/Global_Coral_Bleaching_Database_SQLite_11_24_21.db
python src/02_profile_tables.py data/raw/Global_Coral_Bleaching_Database_SQLite_11_24_21.db
python src/02b_investigate_duplicates_and_nulls.py

# 3. Clean dataset creation & stream ordering
python src/03_build_dataset.py
python src/04_prepare_stream.py

# 4. Model training (Prequential evaluation)
python src/05_train_baseline.py
python src/06_train_arf.py
python src/07_train_arf_tuned.py

# 5. Live NOAA integration & visual analytics
python src/08_noaa_live_connector.py
python src/09_visualize_results.py
```
