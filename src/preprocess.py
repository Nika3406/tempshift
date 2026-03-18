import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def _raw(f):
    return os.path.join(RAW_DIR, f)


# Eruption windows where volcanic aerosols produced measurable cooling signals.
# Source: Robock (2000), Smithsonian GVP.
VOLCANIC_WINDOWS = [
    ("1963-03", "1964-06"),  # Agung
    ("1968-05", "1969-03"),  # Fernandina
    ("1974-09", "1975-06"),  # Fuego
    ("1982-04", "1983-12"),  # El Chichon
    ("1991-06", "1993-06"),  # Pinatubo
]


def load_sources():
    co2 = pd.read_csv(_raw("noaa_co2_monthly.csv"))[["date", "co2_ppm", "co2_deseasonalized"]]
    ch4 = pd.read_csv(_raw("noaa_ch4_monthly.csv"))[["date", "ch4_ppb"]]
    n2o = pd.read_csv(_raw("noaa_n2o_monthly.csv"))[["date", "n2o_ppb"]]
    temp = pd.read_csv(_raw("nasa_temp_monthly.csv"))[["date", "temp_anomaly_c"]]

    solar_path = _raw("ncei_solar_monthly.csv")
    solar = pd.read_csv(solar_path)[["date", "tsi_wm2"]] if os.path.exists(solar_path) \
        else pd.DataFrame(columns=["date", "tsi_wm2"])

    # OWID is annual - broadcast each year's value to all 12 months before merging
    owid_raw = pd.read_csv(_raw("owid_co2_annual.csv"))
    owid_rows = []
    for _, row in owid_raw.iterrows():
        for month in range(1, 13):
            r = row.to_dict()
            r["date"] = f"{int(row['year'])}-{str(month).zfill(2)}"
            owid_rows.append(r)
    owid = pd.DataFrame(owid_rows)
    owid_keep = [c for c in ["date", "year", "co2", "methane", "nitrous_oxide",
                              "land_use_change_co2", "cement_co2", "coal_co2",
                              "oil_co2", "gas_co2", "total_ghg", "energy_per_capita",
                              "population", "gdp"] if c in owid.columns]
    owid = owid[owid_keep]

    for name, df in [("co2", co2), ("ch4", ch4), ("n2o", n2o),
                     ("temp", temp), ("solar", solar), ("owid", owid)]:
        print(f"  {name}: {len(df)} rows")

    merged = temp.copy()
    for df in [co2, ch4, n2o, solar, owid]:
        merged = merged.merge(df, on="date", how="left")
    merged = merged.sort_values("date").reset_index(drop=True)
    print(f"  merged: {merged.shape}")
    return merged


def clean(df):
    for col in ["co2_ppm", "co2_deseasonalized", "ch4_ppb", "n2o_ppb", "tsi_wm2"]:
        if col in df.columns:
            df[col] = df[col].replace([-999.99, -9.99, -99.99, -999, -9999], np.nan)

    # ch4/n2o have no limit since they start mid-record and need to fill back to 1984
    for col, limit in [("co2_ppm", 6), ("co2_deseasonalized", 6),
                       ("tsi_wm2", 12), ("temp_anomaly_c", 6),
                       ("ch4_ppb", None), ("n2o_ppb", None)]:
        if col in df.columns:
            before = df[col].isna().sum()
            df[col] = df[col].interpolate(method="linear", limit=limit, limit_direction="both")
            filled = before - df[col].isna().sum()
            if filled > 0:
                print(f"  interpolated {filled} NaNs in {col}")

    owid_cols = ["co2", "methane", "nitrous_oxide", "land_use_change_co2", "cement_co2",
                 "coal_co2", "oil_co2", "gas_co2", "total_ghg", "energy_per_capita",
                 "population", "gdp"]
    for col in owid_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    before = len(df)
    df = df.dropna(subset=["temp_anomaly_c", "co2_ppm"])
    print(f"  dropped {before - len(df)} rows missing temp or CO2")
    return df


def engineer(df):
    dt = pd.to_datetime(df["date"])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["time_since_1960"] = (df["year"] - 1960) * 12 + (df["month"] - 1)

    df["co2_growth_rate"] = df["co2_ppm"].diff().fillna(0)
    df["co2_12mo_avg"] = df["co2_ppm"].rolling(12, min_periods=6).mean()

    if "ch4_ppb" in df.columns:
        df["ch4_growth_rate"] = df["ch4_ppb"].diff().fillna(0)
        df["ch4_12mo_avg"] = df["ch4_ppb"].rolling(12, min_periods=6).mean()

    if "tsi_wm2" in df.columns:
        baseline = df.loc[(df["year"] >= 1961) & (df["year"] <= 1990), "tsi_wm2"].mean()
        df["tsi_anomaly"] = df["tsi_wm2"] - baseline
        print(f"  TSI baseline (1961-1990): {baseline:.4f}")

    df["volcanic_flag"] = 0
    for start, end in VOLCANIC_WINDOWS:
        df.loc[(df["date"] >= start) & (df["date"] <= end), "volcanic_flag"] = 1
    print(f"  flagged {df['volcanic_flag'].sum()} volcanic months")

    sector_cols = [c for c in ["coal_co2", "oil_co2", "gas_co2", "cement_co2", "land_use_change_co2"]
                   if c in df.columns]
    if sector_cols:
        df["total_anthropogenic_co2"] = df[sector_cols].sum(axis=1)

    df = df.drop(columns=["year_x", "year_y"], errors="ignore")
    return df


def run():
    print("=" * 50)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = load_sources()
    df = clean(df)
    df = engineer(df)

    # Trim to overlap window - CH4 global starts 1983, clean overlap from 1984
    before = len(df)
    df = df[df["date"] >= "1984-01"].reset_index(drop=True)
    print(f"  trimmed {before} -> {len(df)} rows (1984-01 onward)")

    out = os.path.join(PROCESSED_DIR, "merged_dataset.csv")
    df.to_csv(out, index=False)
    print(f"\n{df.shape[0]} rows x {df.shape[1]} cols, {df['date'].iloc[0]} -> {df['date'].iloc[-1]}")
    print(f"saved -> {out}")

    print("\nexample rows:")
    print(df.head(3).to_string(index=False))

    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    print(f"\nnull counts: {nulls.to_dict() if len(nulls) else 'none'}")
    return df


if __name__ == "__main__":
    run()
