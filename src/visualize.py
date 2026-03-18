import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PLOTS_DIR = os.path.join(PROCESSED_DIR, "plots")

PALETTE = {
    "co2": "#E63946", "ch4": "#F4A261", "n2o": "#2A9D8F",
    "temp": "#264653", "solar": "#E9C46A",
    "lr": "#457B9D", "rf": "#2A9D8F", "gbt": "#E63946",
}

plt.rcParams.update({
    "figure.dpi": 150, "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})


def save(name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {path}")


def plot_timeseries(df):
    dates = pd.to_datetime(df["date"])
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    fig.suptitle("Greenhouse Gases vs Global Temperature Anomaly", fontsize=14, y=1.01)

    for ax, (col, ylabel, color, style) in zip(axes, [
        ("temp_anomaly_c", "Temperature Anomaly (C)", PALETTE["temp"], "fill"),
        ("co2_ppm",        "CO2 (ppm)",               PALETTE["co2"],  "line"),
        ("ch4_ppb",        "CH4 (ppb)",               PALETTE["ch4"],  "line"),
        ("n2o_ppb",        "N2O (ppb)",               PALETTE["n2o"],  "line"),
    ]):
        if col not in df.columns:
            ax.set_ylabel(ylabel)
            continue
        vals = df[col]
        if style == "fill":
            ax.fill_between(dates, vals, 0, where=(vals >= 0), color=PALETTE["co2"], alpha=0.6, label="Warm")
            ax.fill_between(dates, vals, 0, where=(vals < 0), color=PALETTE["solar"], alpha=0.6, label="Cool")
            ax.axhline(0, color="black", linewidth=0.7)
        else:
            ax.plot(dates, vals, color=color, linewidth=1.0)
            ax.plot(dates, vals.rolling(12, center=True).mean(), color=color,
                    linewidth=2.0, linestyle="--", label="12-mo avg")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc="upper left")

    axes[-1].set_xlabel("Year")
    plt.tight_layout()
    save("timeseries_ghg_vs_temp.png")


def plot_feature_importance(imp_df):
    labels = {
        "co2_ppm": "CO2 (ppm)", "co2_deseasonalized": "CO2 Deseasonalized",
        "ch4_ppb": "CH4 (ppb)", "n2o_ppb": "N2O (ppb)",
        "co2_growth_rate": "CO2 Growth Rate", "ch4_growth_rate": "CH4 Growth Rate",
        "co2_12mo_avg": "CO2 12-mo Avg", "ch4_12mo_avg": "CH4 12-mo Avg",
        "tsi_anomaly": "Solar Anomaly", "total_ghg": "Total GHG",
        "total_anthropogenic_co2": "Anthropogenic CO2", "energy_per_capita": "Energy per Capita",
        "volcanic_flag": "Volcanic Flag", "time_since_1960": "Time Since 1960",
    }

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Feature Importance Rankings", fontsize=14)

    top = imp_df.head(12).copy()
    top["label"] = top["feature"].map(labels).fillna(top["feature"])
    colors = [PALETTE["co2"] if any(x in f for x in ("co2", "ghg", "anthrop"))
              else PALETTE["solar"] if any(x in f for x in ("tsi", "volcanic"))
              else PALETTE["ch4"] if "ch4" in f
              else PALETTE["n2o"] if "n2o" in f
              else "#999999" for f in top["feature"]]
    ax.barh(top["label"][::-1], top["consensus_score"][::-1], color=colors[::-1])
    ax.set_xlabel("Consensus Importance Score (normalized)")
    ax.set_title("Consensus Ranking (All 3 Models)")
    ax.legend(handles=[
        Line2D([0], [0], color=PALETTE["co2"],   lw=8, label="CO2 / GHG"),
        Line2D([0], [0], color=PALETTE["ch4"],   lw=8, label="CH4"),
        Line2D([0], [0], color=PALETTE["n2o"],   lw=8, label="N2O"),
        Line2D([0], [0], color=PALETTE["solar"], lw=8, label="Solar / Volcanic"),
        Line2D([0], [0], color="#999999",         lw=8, label="Other"),
    ], fontsize=8)

    top8 = imp_df.head(8).copy()
    top8["label"] = top8["feature"].map(labels).fillna(top8["feature"])
    x = np.arange(len(top8))
    for i, (col, lbl, clr) in enumerate(zip(
        ["linear_reg", "random_forest", "grad_boost"],
        ["Linear Regression", "Random Forest", "Gradient Boost"],
        [PALETTE["lr"], PALETTE["rf"], PALETTE["gbt"]]
    )):
        if col in top8.columns:
            ax2.bar(x + i * 0.28, top8[col], width=0.28, label=lbl, color=clr, alpha=0.85)
    ax2.set_xticks(x + 0.28)
    ax2.set_xticklabels(top8["label"], rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("Normalized Importance")
    ax2.set_title("Per-Model Comparison (Top 8)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    save("feature_importance_bar.png")


def scatter(df, xcol, ycol, xlabel, ylabel, title, color, filename):
    if xcol not in df.columns:
        return
    sub = df[[xcol, ycol]].dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sub[xcol], sub[ycol], color=color, alpha=0.35, s=10, label="Monthly")
    z = np.polyfit(sub[xcol], sub[ycol], 1)
    xs = np.linspace(sub[xcol].min(), sub[xcol].max(), 200)
    ax.plot(xs, np.poly1d(z)(xs), color="black", linewidth=1.5, label=f"Trend (slope={z[0]:.4f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    save(filename)


def plot_model_comparison(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Model Performance Comparison", fontsize=13)
    colors = [PALETTE["lr"], PALETTE["rf"], PALETTE["gbt"]]

    for ax, (metric, ylabel, higher_better) in zip(axes, [
        ("r2",   "R2 Score (higher = better)", True),
        ("mae",  "MAE (C, lower = better)",    False),
        ("rmse", "RMSE (C, lower = better)",   False),
    ]):
        vals = results_df[metric].values
        bars = ax.bar(results_df["model"], vals, color=colors, alpha=0.85)
        ax.set_ylabel(ylabel)
        ax.set_title(metric.upper())
        ax.set_xticklabels(results_df["model"], rotation=20, ha="right", fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        best = np.argmax(vals) if higher_better else np.argmin(vals)
        bars[best].set_edgecolor("gold")
        bars[best].set_linewidth(2.5)

    plt.tight_layout()
    save("model_comparison.png")


def plot_residuals(df):
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    features = [f for f in ["co2_ppm", "ch4_ppb", "n2o_ppb", "tsi_wm2",
                             "co2_growth_rate", "co2_12mo_avg", "volcanic_flag",
                             "time_since_1960"] if f in df.columns]
    sub = df[features + ["temp_anomaly_c"]].dropna()
    split = int(len(sub) * 0.70)
    X_tr, X_te = sub[features].iloc[:split], sub[features].iloc[split:]
    y_tr, y_te = sub["temp_anomaly_c"].iloc[:split], sub["temp_anomaly_c"].iloc[split:]

    sc = StandardScaler()
    X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Residual Analysis (Test Set)", fontsize=13)

    for ax, (name, model, Xte, clr) in zip(axes, [
        ("Linear Regression", LinearRegression().fit(X_tr_s, y_tr), X_te_s, PALETTE["lr"]),
        ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr, y_tr), X_te, PALETTE["rf"]),
        ("Gradient Boost", GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr), X_te, PALETTE["gbt"]),
    ]):
        preds = model.predict(Xte)
        resid = y_te.values - preds
        ax.scatter(preds, resid, color=clr, alpha=0.4, s=10)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("Predicted Anomaly (C)")
        ax.set_ylabel("Residual (C)")
        ax.set_title(name)
        ax.text(0.05, 0.92, f"RMSE={np.sqrt(np.mean(resid**2)):.4f}",
                transform=ax.transAxes, fontsize=8)

    plt.tight_layout()
    save("residuals.png")


def run():
    print("=" * 50)
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "merged_dataset.csv"))
    imp_df = pd.read_csv(os.path.join(PROCESSED_DIR, "feature_importance.csv"))
    results_df = pd.read_csv(os.path.join(PROCESSED_DIR, "model_results.csv"))

    plot_timeseries(df)
    plot_feature_importance(imp_df)
    scatter(df, "co2_ppm", "temp_anomaly_c", "CO2 Concentration (ppm)",
            "Temperature Anomaly (C)", "CO2 vs Global Temperature Anomaly",
            PALETTE["co2"], "scatter_co2_vs_temp.png")
    scatter(df, "tsi_anomaly", "temp_anomaly_c", "Solar Irradiance Anomaly",
            "Temperature Anomaly (C)", "Solar Irradiance Anomaly vs Global Temperature",
            PALETTE["solar"], "scatter_tsi_vs_temp.png")
    scatter(df, "ch4_ppb", "temp_anomaly_c", "CH4 Concentration (ppb)",
            "Temperature Anomaly (C)", "CH4 vs Global Temperature Anomaly",
            PALETTE["ch4"], "scatter_ch4_vs_temp.png")
    plot_model_comparison(results_df)
    plot_residuals(df)

    print(f"all plots -> {PLOTS_DIR}")


if __name__ == "__main__":
    run()
