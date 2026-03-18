import os
import re
import time
import requests
import pandas as pd
from io import StringIO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


def _get(url, timeout=60):
    print(f"    -> {url}")
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def _save(df, filename):
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, filename)
    df.to_csv(path, index=False)
    return path


def _date_col(df):
    df["date"] = (
        df["year"].astype(float).astype(int).astype(str)
        + "-"
        + df["month"].astype(float).astype(int).astype(str).str.zfill(2)
    )
    return df


def _decimal_to_year_month(decimal_year):
    # CH4/N2O files from NOAA use decimal year format instead of year+month columns
    year = int(decimal_year)
    month = max(1, min(12, int(round((decimal_year - year) * 12)) + 1))
    return year, month


# NOAA GML data files. CO2 has explicit year/month cols; CH4/N2O use decimal year only.
_GML_SOURCES = {
    "co2": {
        "url": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt",
        "columns": ["year", "month", "decimal_date", "co2_ppm",
                    "co2_deseasonalized", "num_days", "std_dev", "uncertainty"],
        "keep": ["year", "month", "decimal_date", "co2_ppm", "co2_deseasonalized"],
        "output": "noaa_co2_monthly.csv",
    },
    "ch4": {
        "url": "https://gml.noaa.gov/webdata/ccgg/trends/ch4/ch4_mm_gl.txt",
        "columns": ["decimal_date", "average", "trend", "average_unc", "trend_unc"],
        "keep": ["decimal_date", "average"],
        "output": "noaa_ch4_monthly.csv",
    },
    "n2o": {
        "url": "https://gml.noaa.gov/webdata/ccgg/trends/n2o/n2o_mm_gl.txt",
        "columns": ["decimal_date", "average", "trend", "average_unc", "trend_unc"],
        "keep": ["decimal_date", "average"],
        "output": "noaa_n2o_monthly.csv",
    },
}


def collect_noaa_gml():
    print("\n[1/4] NOAA GML")
    results = {}
    for key, cfg in _GML_SOURCES.items():
        raw = _get(cfg["url"])
        lines = [l for l in raw.splitlines() if l.strip() and not l.strip().startswith("#")]
        df = pd.read_csv(StringIO("\n".join(lines)), sep=r"\s+", header=None, names=cfg["columns"])
        df.replace([-999.99, -9.99, -99.99, -999, -9999], pd.NA, inplace=True)
        df = df[cfg["keep"]].copy()

        if "year" in df.columns and "month" in df.columns:
            df = _date_col(df)
        else:
            ym = df["decimal_date"].apply(_decimal_to_year_month)
            df["year"] = ym.apply(lambda x: x[0])
            df["month"] = ym.apply(lambda x: x[1])
            gas_col = "ch4_ppb" if key == "ch4" else "n2o_ppb"
            df = df.rename(columns={"average": gas_col})[["year", "month", gas_col]].copy()
            df = _date_col(df)

        path = _save(df, cfg["output"])
        print(f"  {key}: {len(df)} rows ({df['date'].iloc[0]} -> {df['date'].iloc[-1]})")
        results[key] = df
    return results


_GISS_MONTHLY_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
_GISS_ANNUAL_URL = (
    "https://data.giss.nasa.gov/gistemp/graphs_v4/graph_data/"
    "Global_Mean_Estimates_based_on_Land_and_Ocean_Data/graph.csv"
)
_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def collect_nasa_giss():
    print("\n[2/4] NASA GISS")
    results = {}

    raw = _get(_GISS_MONTHLY_URL)
    lines = raw.splitlines()
    hi = next(i for i, l in enumerate(lines) if l.strip().startswith("Year"))
    df = pd.read_csv(StringIO("\n".join(lines[hi:])), na_values=["***", "****", ""])
    df = df[["Year"] + [m for m in _MONTH_NAMES if m in df.columns]].copy()
    df = df.melt(id_vars="Year", var_name="month_name", value_name="temp_anomaly_c")
    df = df.dropna(subset=["temp_anomaly_c"])
    df["month"] = df["month_name"].map({m: i + 1 for i, m in enumerate(_MONTH_NAMES)})
    df["year"] = df["Year"].astype(int)
    df = _date_col(df)
    df = df[["date", "year", "month", "temp_anomaly_c"]].sort_values("date").reset_index(drop=True)
    _save(df, "nasa_temp_monthly.csv")
    print(f"  monthly: {len(df)} rows ({df['date'].iloc[0]} -> {df['date'].iloc[-1]})")
    results["monthly"] = df

    raw = _get(_GISS_ANNUAL_URL)
    lines = raw.splitlines()
    hi = next(i for i, l in enumerate(lines) if "Year" in l)
    df2 = pd.read_csv(StringIO("\n".join(lines[hi:])), na_values=["***", "****", ""])
    df2.columns = [c.strip() for c in df2.columns]
    col_map = {}
    for c in df2.columns:
        lc = c.lower()
        if lc == "year":
            col_map[c] = "year"
        elif "smooth" in lc or lc == "no_smoothing":
            col_map[c] = "temp_anomaly_annual_c"
        elif "lowess" in lc:
            col_map[c] = "temp_anomaly_smoothed_c"
    df2 = df2.rename(columns=col_map).dropna(subset=["temp_anomaly_annual_c"])
    df2["year"] = df2["year"].astype(int)
    keep = [c for c in ["year", "temp_anomaly_annual_c", "temp_anomaly_smoothed_c"] if c in df2.columns]
    df2 = df2[keep].reset_index(drop=True)
    _save(df2, "nasa_temp_annual.csv")
    print(f"  annual: {len(df2)} rows ({int(df2['year'].min())} -> {int(df2['year'].max())})")
    results["annual"] = df2

    return results


# Real TSI sources. NCEI requires netCDF now and blocks CSV access via robots.txt,
# so we try LASP and PMOD first, then fall back to SILSO sunspot numbers as a proxy.
_LASP_URL = "https://lasp.colorado.edu/data/sorce/tsi_data/historical/sorce_tsi_L3_c24h_latest.txt"
_PMOD_URL = "https://www.pmodwrc.ch/en/research-development/solar-physics/tsi-composite/composite_d41_62_2302.dat"
_SILSO_URL = "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.txt"


def collect_solar():
    print("\n[3/4] Solar Irradiance")

    for label, url in [("LASP", _LASP_URL), ("PMOD", _PMOD_URL)]:
        try:
            raw = _get(url)
            lines = [l for l in raw.splitlines()
                     if l.strip() and not l.strip().startswith(("#", ";"))]
            rows = []
            for line in lines:
                parts = line.split()
                try:
                    float(parts[0])
                    rows.append(parts)
                except (ValueError, IndexError):
                    continue
            if not rows:
                continue

            if len(rows[0]) >= 3:
                df = pd.DataFrame(rows, columns=[f"c{i}" for i in range(len(rows[0]))])
                df = df.iloc[:, :3].copy()
                df.columns = ["year", "month", "tsi_wm2"]
                df["year"] = df["year"].astype(float).astype(int)
                df["month"] = df["month"].astype(float).astype(int)
            else:
                df = pd.DataFrame(rows, columns=["decimal_year", "tsi_wm2"])
                df["decimal_year"] = df["decimal_year"].astype(float)
                df["year"] = df["decimal_year"].astype(int)
                df["month"] = ((df["decimal_year"] % 1) * 12 + 1).round().astype(int).clip(1, 12)

            df["tsi_wm2"] = pd.to_numeric(df["tsi_wm2"], errors="coerce")
            df = df[df["tsi_wm2"] > 1300].dropna(subset=["tsi_wm2"])
            if len(df) < 100:
                continue
            df = _date_col(df)
            df = df[["date", "year", "month", "tsi_wm2"]].sort_values("date").reset_index(drop=True)
            _save(df, "ncei_solar_monthly.csv")
            print(f"  [{label}] {len(df)} rows, TSI {df['tsi_wm2'].min():.1f}-{df['tsi_wm2'].max():.1f} W/m2")
            return {"solar": df}
        except Exception as e:
            print(f"  {label} failed: {e}")

    # SILSO fallback - tsi_wm2 column will contain sunspot numbers, not W/m2
    print("  Using SILSO sunspot proxy")
    raw = _get(_SILSO_URL)
    rows = []
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            try:
                rows.append({"year": int(parts[0]), "month": int(parts[1]), "tsi_wm2": float(parts[3])})
            except ValueError:
                continue
    df = pd.DataFrame(rows)
    df = df[df["tsi_wm2"] >= 0]
    df = _date_col(df)
    df = df[["date", "year", "month", "tsi_wm2"]].sort_values("date").reset_index(drop=True)
    _save(df, "ncei_solar_monthly.csv")
    print(f"  [SILSO] {len(df)} rows (sunspot proxy, not W/m2)")
    return {"solar": df}


_OWID_URL = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
_OWID_COLS = [
    "country", "year", "co2", "co2_growth_prct", "co2_per_capita",
    "methane", "nitrous_oxide", "land_use_change_co2", "cement_co2",
    "coal_co2", "oil_co2", "gas_co2", "total_ghg", "energy_per_capita",
    "population", "gdp",
]


def collect_owid():
    print("\n[4/4] Our World in Data")
    df = pd.read_csv(StringIO(_get(_OWID_URL)))
    df = df[df["country"] == "World"].copy()
    available = [c for c in _OWID_COLS if c in df.columns]
    df = df[available].sort_values("year").reset_index(drop=True)
    feat_cols = [c for c in available if c not in ("country", "year")]
    df = df.dropna(how="all", subset=feat_cols)
    _save(df, "owid_co2_annual.csv")
    print(f"  {len(df)} rows ({int(df['year'].min())} -> {int(df['year'].max())})")
    return {"owid": df}


def run_all():
    print("=" * 50)
    t0 = time.time()
    all_data = {}
    for key, fn in [("noaa_gml", collect_noaa_gml), ("nasa_giss", collect_nasa_giss),
                    ("solar", collect_solar), ("owid", collect_owid)]:
        try:
            all_data[key] = fn()
        except Exception as e:
            print(f"  ERROR in {key}: {e}")
            all_data[key] = {}
    print(f"\nDone in {time.time() - t0:.1f}s. Files -> {RAW_DIR}")
    return all_data


if __name__ == "__main__":
    run_all()
