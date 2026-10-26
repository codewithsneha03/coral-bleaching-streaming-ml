"""
Step 8: NOAA Coral Reef Watch live data connector.

PURPOSE:
  Fetch real-time or recent environmental data from NOAA Coral Reef Watch
  (CRW) via the PacIOOS ERDDAP server, which serves the CRW v3.1 5km
  daily global products.

  Available products:
    - CRW_SST:       Sea Surface Temperature (Celsius)
    - CRW_SSTANOMALY: SST Anomaly (difference from climatological mean)
    - CRW_HOTSPOT:   HotSpot (positive anomaly above bleaching threshold)
    - CRW_BAA:       Bleaching Alert Area (0-4 alert level)
    - CRW_DHW:       Degree Heating Weeks (accumulated thermal stress)

  The script queries a specific lat/lon point (or a small bounding box)
  for the most recent data, maps NOAA fields to our model features, and
  generates a prediction using the trained ARF model.

IMPORTANT DISTINCTIONS:
  - NOAA provides ENVIRONMENTAL data, not observed bleaching labels.
  - Our model was trained on GCBD historical observations with concurrent
    environmental measurements.
  - When we predict on NOAA data, we are extrapolating: the model says
    "given these conditions, bleaching would likely have been observed
    at similar sites in the training data."
  - This is NOT a confirmed bleaching detection.

READS:
  data/processed/coral_stream.csv  (to re-train model before prediction)

WRITES:
  results/noaa_live_data.csv
  results/noaa_predictions.csv

REQUIRES:
  pip install requests  (should already be available)
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import json
import time as _time

# ── Paths ────────────────────────────────────────────────────────────
STREAM_PATH = Path("data/processed/coral_stream.csv")
NOAA_OUT    = Path("results/noaa_live_data.csv")
PRED_OUT    = Path("results/noaa_predictions.csv")
REPORT_OUT  = Path("results/noaa_connector_report.txt")

NOAA_OUT.parent.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# NOAA ERDDAP CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
# PacIOOS ERDDAP hosts NOAA CRW v3.1 5km daily products
ERDDAP_BASE = "https://pae-paha.pacioos.hawaii.edu/erddap/griddap"

# Dataset IDs for different CRW products
DATASETS = {
    "dhw":       "dhw_5km",           # Degree Heating Weeks
    "sst":       "CRW_sst_v3_1",      # SST
    "sst_anom":  "CRW_sstanomaly_v3_1", # SST Anomaly
    "hotspot":   "CRW_hotspot_v3_1",   # HotSpot
    "baa":       "CRW_baa_v3_1_op",    # Bleaching Alert Area
}

# Well-known reef locations for demonstration
REEF_LOCATIONS = {
    "Great_Barrier_Reef_AU": {"lat": -18.2, "lon": 147.5, "country": "Australia"},
    "Florida_Keys_US":       {"lat": 24.6,  "lon": -81.6, "country": "United States"},
    "Maldives":              {"lat": 4.2,   "lon": 73.5,  "country": "Maldives"},
    "Fiji":                  {"lat": -17.8, "lon": 177.9, "country": "Fiji"},
    "Red_Sea_Egypt":         {"lat": 27.2,  "lon": 34.0,  "country": "Egypt"},
    "Hawaii_US":             {"lat": 21.4,  "lon": -157.8,"country": "United States"},
    "Palau":                 {"lat": 7.5,   "lon": 134.5, "country": "Palau"},
    "Caribbean_Belize":      {"lat": 17.5,  "lon": -87.8, "country": "Belize"},
}


def fetch_erddap_point(dataset_id, variable, lat, lon, days_back=7):
    """
    Fetch the latest available data for a single point from ERDDAP.

    Returns a list of dicts with 'time', 'latitude', 'longitude', variable.
    """
    # ERDDAP requires time range; look back N days from today
    end_date = datetime.utcnow().strftime("%Y-%m-%dT00:00:00Z")
    start_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%dT00:00:00Z")

    # Construct ERDDAP query URL for CSV output
    url = (f"{ERDDAP_BASE}/{dataset_id}.csv"
           f"?{variable}"
           f"[({start_date}):1:({end_date})]"
           f"[({lat}):1:({lat})]"
           f"[({lon}):1:({lon})]")

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            # ERDDAP CSV has 2 header lines: column names, then units
            lines = resp.text.strip().split("\n")
            if len(lines) >= 3:
                headers = lines[0].split(",")
                data = []
                for line in lines[2:]:  # skip units row
                    vals = line.split(",")
                    row = dict(zip(headers, vals))
                    data.append(row)
                return data
            return []
        else:
            print(f"  ERDDAP returned HTTP {resp.status_code} for {dataset_id}/{variable}")
            return []
    except requests.RequestException as e:
        print(f"  Network error fetching {dataset_id}: {e}")
        return []


def fetch_noaa_for_location(name, lat, lon, days_back=7):
    """
    Fetch all available CRW products for a single location.
    Returns a list of observation dicts (one per day).
    """
    print(f"  Fetching NOAA data for {name} ({lat}, {lon})...")

    # Try to fetch SST from a known working dataset
    # The dataset IDs may vary; try multiple patterns
    sst_datasets = [
        ("CRW_sst_v3_1", "analysed_sst"),
        ("dhw_5km", "CRW_SST"),
    ]
    anom_datasets = [
        ("CRW_sstanomaly_v3_1", "analysed_sst_anomaly"),
        ("dhw_5km", "CRW_SSTANOMALY"),
    ]
    dhw_datasets = [
        ("dhw_5km", "CRW_DHW"),
    ]
    hotspot_datasets = [
        ("CRW_hotspot_v3_1", "hotspot"),
        ("dhw_5km", "CRW_HOTSPOT"),
    ]
    baa_datasets = [
        ("dhw_5km", "CRW_BAA"),
    ]

    # Try to get all variables from the dhw_5km combined dataset first
    combined_url = (f"{ERDDAP_BASE}/dhw_5km.csv"
                    f"?CRW_SST,CRW_SSTANOMALY,CRW_HOTSPOT,CRW_DHW,CRW_BAA"
                    f"[({(datetime.utcnow() - timedelta(days=days_back)).strftime('%Y-%m-%dT00:00:00Z')})"
                    f":1:"
                    f"({datetime.utcnow().strftime('%Y-%m-%dT00:00:00Z')})]"
                    f"[({lat}):1:({lat})]"
                    f"[({lon}):1:({lon})]")

    try:
        resp = requests.get(combined_url, timeout=30)
        if resp.status_code == 200:
            lines = resp.text.strip().split("\n")
            if len(lines) >= 3:
                headers = [h.strip() for h in lines[0].split(",")]
                observations = []
                for line in lines[2:]:
                    vals = [v.strip() for v in line.split(",")]
                    row = {}
                    for h, v in zip(headers, vals):
                        try:
                            row[h] = float(v) if v not in ("NaN", "", "nan") else np.nan
                        except ValueError:
                            row[h] = v
                    row["location"] = name
                    row["query_lat"] = lat
                    row["query_lon"] = lon
                    observations.append(row)
                print(f"    Retrieved {len(observations)} daily observations")
                return observations
            print(f"    No data rows returned")
            return []
        else:
            print(f"    ERDDAP HTTP {resp.status_code}")
            return []
    except requests.RequestException as e:
        print(f"    Network error: {e}")
        return []


def map_noaa_to_model_features(noaa_obs, location_info):
    """
    Map NOAA CRW variables to our trained model's feature space.

    IMPORTANT: Our model was trained with 41 features from GCBD.
    NOAA only provides a subset of these. Missing features are filled
    with training-set medians (computed from the stream file).

    This mapping is approximate -- the NOAA products and the GCBD
    environmental fields may have slightly different definitions,
    spatial resolution, and temporal aggregation.
    """
    # Features our model expects (from 04_prepare_stream.py)
    feature_medians = {}  # Will be populated from the training stream

    mapped = {}

    # Direct mappings (NOAA -> closest GCBD equivalent)
    if "CRW_SST" in noaa_obs and not np.isnan(noaa_obs.get("CRW_SST", np.nan)):
        # NOAA SST is in Celsius; GCBD Temperature_Kelvin is in Kelvin
        sst_c = noaa_obs["CRW_SST"]
        mapped["Temperature_Kelvin"] = sst_c + 273.15
        mapped["Temperature_Mean"] = sst_c + 273.15

    if "CRW_SSTANOMALY" in noaa_obs and not np.isnan(noaa_obs.get("CRW_SSTANOMALY", np.nan)):
        mapped["SSTA"] = noaa_obs["CRW_SSTANOMALY"]
        mapped["TSA"] = noaa_obs["CRW_SSTANOMALY"]

    if "CRW_DHW" in noaa_obs and not np.isnan(noaa_obs.get("CRW_DHW", np.nan)):
        mapped["SSTA_DHW"] = noaa_obs["CRW_DHW"]
        mapped["TSA_DHW"] = noaa_obs["CRW_DHW"]

    if "CRW_HOTSPOT" in noaa_obs and not np.isnan(noaa_obs.get("CRW_HOTSPOT", np.nan)):
        mapped["SSTA_Maximum"] = noaa_obs["CRW_HOTSPOT"]

    # Location-based features
    mapped["Latitude_Degrees"] = location_info.get("lat", 0)
    mapped["Longitude_Degrees"] = location_info.get("lon", 0)

    return mapped


def train_model_from_stream(stream_path):
    """
    Re-train the best ARF model on the full historical stream.
    Returns (model, feature_cols, feature_medians).
    """
    from river.forest import ARFClassifier
    from river.drift import ADWIN

    df = pd.read_csv(stream_path)
    id_cols = ["Sample_ID", "Site_ID", "date", "Date_Year"]
    target = "bleached"
    feature_cols = [c for c in df.columns if c not in id_cols + [target]]

    # Compute medians for filling missing NOAA features
    feature_medians = df[feature_cols].median().to_dict()

    # Train the best configuration from step 7
    model = ARFClassifier(
        n_models=25,
        max_features="sqrt",
        grace_period=30,
        drift_detector=ADWIN(delta=0.001),
        warning_detector=ADWIN(delta=0.01),
        seed=42,
    )

    print(f"Training ARF on {len(df):,} historical observations...")
    for _, row in df.iterrows():
        x = {col: row[col] for col in feature_cols}
        y = int(row[target])
        model.learn_one(x, y)

    print("Training complete.")
    return model, feature_cols, feature_medians


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    report = []
    report.append("NOAA CORAL REEF WATCH LIVE DATA CONNECTOR")
    report.append("=" * 60)
    report.append(f"Run time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    report.append(f"Data source: NOAA CRW v3.1 via PacIOOS ERDDAP")
    report.append(f"ERDDAP base: {ERDDAP_BASE}")
    report.append("")

    # 1. Fetch NOAA data for all reef locations
    print("=" * 60)
    print("STEP 1: Fetching live NOAA data for reef locations")
    print("=" * 60)

    all_noaa_data = []
    for loc_name, loc_info in REEF_LOCATIONS.items():
        observations = fetch_noaa_for_location(
            loc_name, loc_info["lat"], loc_info["lon"], days_back=7
        )
        for obs in observations:
            obs["location_name"] = loc_name
            obs["country"] = loc_info["country"]
        all_noaa_data.extend(observations)
        _time.sleep(0.5)  # Be polite to the ERDDAP server

    if not all_noaa_data:
        report.append("WARNING: No NOAA data could be retrieved.")
        report.append("This may be due to network issues or ERDDAP maintenance.")
        report.append("The connector is designed to work when the server is available.")
        REPORT_OUT.write_text("\n".join(report), encoding="utf-8")
        print("\n".join(report))
        # Create empty output files
        pd.DataFrame().to_csv(NOAA_OUT, index=False)
        pd.DataFrame().to_csv(PRED_OUT, index=False)
        print(f"\nReport saved: {REPORT_OUT.resolve()}")
        sys.exit(0)

    noaa_df = pd.DataFrame(all_noaa_data)
    noaa_df.to_csv(NOAA_OUT, index=False)
    print(f"\nNOAA data saved: {NOAA_OUT.resolve()}")
    print(f"Total observations: {len(noaa_df)}")

    report.append(f"Locations queried: {len(REEF_LOCATIONS)}")
    report.append(f"Total NOAA observations retrieved: {len(noaa_df)}")
    report.append("")

    # 2. Train model on historical data
    print("\n" + "=" * 60)
    print("STEP 2: Training ARF model on historical stream")
    print("=" * 60)

    model, feature_cols, feature_medians = train_model_from_stream(STREAM_PATH)

    # 3. Generate predictions for each NOAA observation
    print("\n" + "=" * 60)
    print("STEP 3: Generating predictions from NOAA data")
    print("=" * 60)

    predictions = []
    for _, obs in noaa_df.iterrows():
        loc_name = obs.get("location_name", "unknown")
        loc_info = REEF_LOCATIONS.get(loc_name, {})

        # Map NOAA fields to model features
        mapped = map_noaa_to_model_features(obs.to_dict(), loc_info)

        # Fill all missing features with training medians
        x = {}
        for feat in feature_cols:
            if feat in mapped and not (isinstance(mapped[feat], float) and np.isnan(mapped[feat])):
                x[feat] = mapped[feat]
            else:
                x[feat] = feature_medians.get(feat, 0.0)

        # Predict
        y_pred = model.predict_one(x)
        try:
            proba = model.predict_proba_one(x)
            prob_bleached = proba.get(1, 0.0)
        except Exception:
            prob_bleached = None

        predictions.append({
            "location": loc_name,
            "country": obs.get("country", ""),
            "latitude": loc_info.get("lat", ""),
            "longitude": loc_info.get("lon", ""),
            "time": obs.get("time", ""),
            "CRW_SST": obs.get("CRW_SST", np.nan),
            "CRW_SSTANOMALY": obs.get("CRW_SSTANOMALY", np.nan),
            "CRW_DHW": obs.get("CRW_DHW", np.nan),
            "CRW_HOTSPOT": obs.get("CRW_HOTSPOT", np.nan),
            "CRW_BAA": obs.get("CRW_BAA", np.nan),
            "predicted_bleaching": y_pred,
            "bleaching_probability": prob_bleached,
            "n_features_from_noaa": len(mapped),
            "n_features_imputed": len(feature_cols) - len(mapped),
        })

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(PRED_OUT, index=False)
    print(f"\nPredictions saved: {PRED_OUT.resolve()}")

    # 4. Summary report
    report.append("PREDICTION SUMMARY")
    report.append("-" * 40)
    for loc_name in REEF_LOCATIONS:
        loc_preds = pred_df[pred_df["location"] == loc_name]
        if loc_preds.empty:
            report.append(f"  {loc_name}: No data")
            continue
        latest = loc_preds.iloc[-1]
        risk = "HIGH RISK" if latest["predicted_bleaching"] == 1 else "LOW RISK"
        prob_str = f" (prob={latest['bleaching_probability']:.2f})" if latest["bleaching_probability"] is not None else ""
        report.append(f"  {loc_name}: {risk}{prob_str}")
        if not np.isnan(latest.get("CRW_SST", np.nan)):
            report.append(f"    SST: {latest['CRW_SST']:.1f}C, "
                          f"Anomaly: {latest.get('CRW_SSTANOMALY', 'N/A')}, "
                          f"DHW: {latest.get('CRW_DHW', 'N/A')}")

    report.append("")
    report.append("IMPORTANT CAVEATS")
    report.append("-" * 40)
    report.append("1. Predictions use NOAA satellite data mapped to a model trained")
    report.append("   on field-survey observations. The feature mapping is approximate.")
    report.append("2. Many model features (41 total) have no NOAA equivalent and are")
    report.append("   filled with historical medians. Predictions are LESS RELIABLE")
    report.append("   than evaluations on the training stream.")
    report.append("3. A prediction of 'bleaching' means conditions RESEMBLE those")
    report.append("   that historically co-occurred with observed bleaching -- it is")
    report.append("   NOT a confirmed bleaching detection.")
    report.append("4. Always verify predictions with field surveys or additional sources.")

    report_text = "\n".join(report)
    REPORT_OUT.write_text(report_text, encoding="utf-8")
    print(f"\n{report_text}")
    print(f"\nReport saved: {REPORT_OUT.resolve()}")
