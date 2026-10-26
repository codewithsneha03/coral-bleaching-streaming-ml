"""
Step 9: Generate all evaluation visualizations.

PURPOSE:
  Create publication-quality plots comparing model performance,
  showing concept drift, class distribution, and NOAA predictions.

READS:
  results/baseline_predictions.csv
  results/arf_tuned_predictions.csv    (or arf_predictions.csv)
  results/drift_events.csv
  results/noaa_predictions.csv
  data/processed/coral_stream.csv

WRITES:
  results/figures/class_distribution.png
  results/figures/prediction_performance_over_time.png
  results/figures/confusion_matrices.png
  results/figures/model_comparison.png
  results/figures/concept_drift_timeline.png
  results/figures/noaa_predictions_map.png
  results/figures/rolling_accuracy.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ── Paths ────────────────────────────────────────────────────────────
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

STREAM     = Path("data/processed/coral_stream.csv")
BASELINE   = Path("results/baseline_predictions.csv")
ARF_TUNED  = Path("results/arf_tuned_predictions.csv")
ARF_ORIG   = Path("results/arf_predictions.csv")
DRIFT_CSV  = Path("results/drift_events.csv")
NOAA_PRED  = Path("results/noaa_predictions.csv")

# Use tuned ARF if available, otherwise original
ARF_PATH = ARF_TUNED if ARF_TUNED.is_file() else ARF_ORIG

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# ══════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════
stream_df = pd.read_csv(STREAM) if STREAM.is_file() else None
base_df   = pd.read_csv(BASELINE) if BASELINE.is_file() else None
arf_df    = pd.read_csv(ARF_PATH) if ARF_PATH.is_file() else None
drift_df  = pd.read_csv(DRIFT_CSV) if DRIFT_CSV.is_file() else None
noaa_df   = pd.read_csv(NOAA_PRED) if NOAA_PRED.is_file() else None

# Filter valid predictions
if base_df is not None:
    base_df = base_df[base_df["y_pred"] != -1]
if arf_df is not None:
    arf_df = arf_df[arf_df["y_pred"] != -1]

# ══════════════════════════════════════════════════════════════════════
# PLOT 1: Class Distribution
# ══════════════════════════════════════════════════════════════════════
if stream_df is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall distribution
    counts = stream_df["bleached"].value_counts().sort_index()
    colors = ["#2ecc71", "#e74c3c"]
    labels = ["Not Bleached (0)", "Bleached (1)"]
    axes[0].bar(labels, counts.values, color=colors, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(counts.values):
        pct = 100 * v / counts.sum()
        axes[0].text(i, v + 100, f"{v:,}\n({pct:.1f}%)", ha="center", fontweight="bold")
    axes[0].set_title("Overall Class Distribution")
    axes[0].set_ylabel("Number of Observations")

    # Distribution by year
    year_groups = stream_df.groupby("Date_Year")["bleached"].agg(["sum", "count"])
    year_groups["not_bleached"] = year_groups["count"] - year_groups["sum"]
    year_groups = year_groups[year_groups["count"] >= 5]  # skip tiny years

    axes[1].bar(year_groups.index, year_groups["sum"], label="Bleached",
                color="#e74c3c", alpha=0.8)
    axes[1].bar(year_groups.index, year_groups["not_bleached"],
                bottom=year_groups["sum"], label="Not Bleached",
                color="#2ecc71", alpha=0.8)
    axes[1].set_title("Class Distribution by Year")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Number of Observations")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_distribution.png", bbox_inches="tight")
    plt.close()
    print("Saved: class_distribution.png")

# ══════════════════════════════════════════════════════════════════════
# PLOT 2: Rolling Accuracy Over Time
# ══════════════════════════════════════════════════════════════════════
if base_df is not None and arf_df is not None:
    fig, ax = plt.subplots(figsize=(14, 6))

    window = 500  # rolling window size

    for df, name, color in [(base_df, "Baseline (LogReg)", "#3498db"),
                            (arf_df, "ARF (Tuned)", "#e74c3c")]:
        correct = (df["y_true"] == df["y_pred"]).astype(int)
        rolling_acc = correct.rolling(window=window, min_periods=50).mean()
        ax.plot(range(len(rolling_acc)), rolling_acc, label=name,
                color=color, alpha=0.8, linewidth=1.2)

    # Mark drift events if available
    if drift_df is not None and not drift_df.empty:
        for _, evt in drift_df.iterrows():
            idx = int(evt["observation_index"])
            if idx < len(arf_df):
                ax.axvline(x=idx, color="orange", linestyle="--",
                          alpha=0.6, linewidth=0.8)
        # Add one labeled line for the legend
        ax.axvline(x=-1, color="orange", linestyle="--", alpha=0.6,
                  label="Drift Detected")

    ax.axhline(y=stream_df["bleached"].mean(), color="gray", linestyle=":",
              alpha=0.5, label=f"Naive baseline ({stream_df['bleached'].mean():.3f})")

    ax.set_xlabel("Observation Number (Chronological)")
    ax.set_ylabel(f"Rolling Accuracy (window={window})")
    ax.set_title("Model Performance Over Time (Prequential Evaluation)")
    ax.legend(loc="lower right")
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)

    # Add year annotations on top
    if stream_df is not None:
        year_changes = stream_df.groupby("Date_Year").apply(
            lambda g: g.index[0]).to_dict()
        for year, idx in year_changes.items():
            if int(year) % 5 == 0:
                ax.annotate(str(int(year)), xy=(idx, 1.02),
                          fontsize=8, color="gray", ha="center")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "rolling_accuracy.png", bbox_inches="tight")
    plt.close()
    print("Saved: rolling_accuracy.png")

# ══════════════════════════════════════════════════════════════════════
# PLOT 3: Confusion Matrices Side by Side
# ══════════════════════════════════════════════════════════════════════
if base_df is not None and arf_df is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, df, title in [(axes[0], base_df, "Baseline (Online LogReg)"),
                          (axes[1], arf_df, "ARF (Tuned)")]:
        cm = confusion_matrix(df["y_true"], df["y_pred"], labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Not Bleached", "Bleached"])
        disp.plot(ax=ax, cmap="Blues", values_format=",d")
        ax.set_title(title)

    plt.suptitle("Confusion Matrices (Prequential Evaluation)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrices.png", bbox_inches="tight")
    plt.close()
    print("Saved: confusion_matrices.png")

# ══════════════════════════════════════════════════════════════════════
# PLOT 4: Model Comparison Bar Chart
# ══════════════════════════════════════════════════════════════════════
if base_df is not None and arf_df is not None:
    fig, ax = plt.subplots(figsize=(10, 6))

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    models = {}
    for df, name in [(base_df, "Baseline\n(Online LogReg)"),
                     (arf_df, "ARF\n(Tuned)")]:
        models[name] = {
            "Accuracy": accuracy_score(df["y_true"], df["y_pred"]),
            "Precision": precision_score(df["y_true"], df["y_pred"]),
            "Recall": recall_score(df["y_true"], df["y_pred"]),
            "F1 Score": f1_score(df["y_true"], df["y_pred"]),
        }

    # Add naive baseline
    naive_acc = stream_df["bleached"].mean()
    models["Naive\n(Always 1)"] = {
        "Accuracy": naive_acc,
        "Precision": naive_acc,
        "Recall": 1.0,
        "F1 Score": 2 * naive_acc / (1 + naive_acc),
    }

    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    x = np.arange(len(metric_names))
    width = 0.25
    colors = ["#95a5a6", "#3498db", "#e74c3c"]

    for i, (model_name, metrics_dict) in enumerate(models.items()):
        vals = [metrics_dict[m] for m in metric_names]
        bars = ax.bar(x + i * width, vals, width, label=model_name,
                     color=colors[i], edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                   fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (Prequential Evaluation)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.set_ylim(0.7, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_comparison.png", bbox_inches="tight")
    plt.close()
    print("Saved: model_comparison.png")

# ══════════════════════════════════════════════════════════════════════
# PLOT 5: Per-Year Accuracy Comparison
# ══════════════════════════════════════════════════════════════════════
if base_df is not None and arf_df is not None:
    fig, ax = plt.subplots(figsize=(14, 6))

    for df, name, color, marker in [(base_df, "Baseline", "#3498db", "o"),
                                     (arf_df, "ARF (Tuned)", "#e74c3c", "s")]:
        year_acc = df.groupby("Date_Year").apply(
            lambda g: (g["y_true"] == g["y_pred"]).mean()
            if len(g) >= 5 else np.nan
        ).dropna()
        ax.plot(year_acc.index, year_acc.values, marker=marker, label=name,
               color=color, linewidth=1.5, markersize=5)

    # Add bleaching rate for context
    if stream_df is not None:
        yr_rate = stream_df.groupby("Date_Year")["bleached"].mean()
        yr_rate = yr_rate[yr_rate.index.isin(
            stream_df.groupby("Date_Year").size()[
                stream_df.groupby("Date_Year").size() >= 5
            ].index
        )]
        ax.plot(yr_rate.index, yr_rate.values, marker="^", label="Bleaching Rate",
               color="#2ecc71", linewidth=1, markersize=4, linestyle="--", alpha=0.7)

    ax.set_xlabel("Year")
    ax.set_ylabel("Accuracy / Rate")
    ax.set_title("Model Accuracy and Bleaching Rate by Year")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.05)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "prediction_performance_over_time.png", bbox_inches="tight")
    plt.close()
    print("Saved: prediction_performance_over_time.png")

# ══════════════════════════════════════════════════════════════════════
# PLOT 6: Concept Drift Timeline
# ══════════════════════════════════════════════════════════════════════
if drift_df is not None and not drift_df.empty:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top: accuracy over time with drift markers
    if arf_df is not None:
        correct = (arf_df["y_true"] == arf_df["y_pred"]).astype(int)
        rolling = correct.rolling(window=300, min_periods=30).mean()
        axes[0].plot(range(len(rolling)), rolling, color="#e74c3c",
                    alpha=0.8, linewidth=1)
        axes[0].set_ylabel("Rolling Accuracy (w=300)")
        axes[0].set_title("Concept Drift Detection Timeline")

    for _, evt in drift_df.iterrows():
        idx = int(evt["observation_index"])
        axes[0].axvline(x=idx, color="orange", linewidth=1.5, alpha=0.7)
        axes[1].axvline(x=idx, color="orange", linewidth=1.5, alpha=0.7)

    # Bottom: F1 at drift events
    axes[1].scatter(drift_df["observation_index"], drift_df["f1_at_drift"],
                   c="orange", s=60, zorder=5, edgecolors="black", linewidth=0.5)
    axes[1].set_xlabel("Observation Index")
    axes[1].set_ylabel("F1 at Drift Event")
    axes[1].set_title("F1 Score at Each Drift Detection Point")
    axes[1].grid(True, alpha=0.3)
    axes[0].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "concept_drift_timeline.png", bbox_inches="tight")
    plt.close()
    print("Saved: concept_drift_timeline.png")
else:
    print("No drift events -- skipping concept_drift_timeline.png")

# ══════════════════════════════════════════════════════════════════════
# PLOT 7: NOAA Predictions Map (scatter on lat/lon)
# ══════════════════════════════════════════════════════════════════════
if noaa_df is not None and not noaa_df.empty and "latitude" in noaa_df.columns:
    fig, ax = plt.subplots(figsize=(14, 7))

    # Get latest prediction per location
    latest = noaa_df.groupby("location").last().reset_index()

    if "predicted_bleaching" in latest.columns and len(latest) > 0:
        bleached = latest[latest["predicted_bleaching"] == 1]
        not_bleached = latest[latest["predicted_bleaching"] == 0]

        if len(not_bleached) > 0:
            ax.scatter(not_bleached["longitude"], not_bleached["latitude"],
                      c="#2ecc71", s=150, marker="o", label="Low Risk",
                      edgecolors="black", linewidth=0.5, zorder=5)
        if len(bleached) > 0:
            ax.scatter(bleached["longitude"], bleached["latitude"],
                      c="#e74c3c", s=150, marker="X", label="High Risk (Bleaching)",
                      edgecolors="black", linewidth=0.5, zorder=5)

        for _, row in latest.iterrows():
            ax.annotate(row["location"].replace("_", " "),
                       (row["longitude"], row["latitude"]),
                       textcoords="offset points", xytext=(10, 5),
                       fontsize=8, alpha=0.8)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("NOAA Live Predictions: Bleaching Risk at Reef Locations")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(FIG_DIR / "noaa_predictions_map.png", bbox_inches="tight")
        plt.close()
        print("Saved: noaa_predictions_map.png")
    else:
        print("NOAA predictions empty -- skipping map plot")
else:
    print("No NOAA predictions file -- skipping map plot")

print(f"\nAll figures saved to: {FIG_DIR.resolve()}")
