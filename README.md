# Climate Root Cause Analysis
CMPSC 445 - Applied Machine Learning, Penn State Abington

Regression pipeline that pulls climate data from four sources, merges it,
and uses feature importance to figure out what's actually driving global
temperature change.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Run these in order — each step depends on the previous one's output.

```bash
python3 src/collect.py       # downloads raw CSVs to data/raw/
python3 src/preprocess.py    # merges and cleans everything
python3 src/model.py         # trains models, saves feature rankings
python3 src/visualize.py     # generates plots
```

---

## Data Sources

| Source | What we use |
|--------|-------------|
| [NOAA GML](https://gml.noaa.gov/ccgg/trends/) | CO2, CH4, N2O monthly concentrations |
| [NASA GISS](https://data.giss.nasa.gov/gistemp/) | Global surface temperature anomaly |
| [NOAA NCEI](https://www.ncei.noaa.gov/data/total-solar-irradiance/access/) | Solar irradiance (falls back to SILSO sunspot proxy if unavailable) |
| [Our World in Data](https://ourworldindata.org/co2-emissions) | Sector emissions, total GHG, energy use |

The final merged dataset covers January 1984 through present at monthly resolution.

## Notes

- The NCEI solar TSI files are blocked by robots.txt and their newer format
  is netCDF, so `collect.py` tries LASP and PMOD first and falls back to
  SILSO sunspot numbers as a solar activity proxy. The column is still called
  `tsi_wm2` but contains sunspot numbers if the fallback triggers.

- Models are trained to predict month-over-month temperature *change* rather
  than absolute anomaly. The test period (2015-2026) runs about 0.5C warmer
  than the training period (1984-2015) on average, which causes any model
  trained on the earlier window to systematically underpredict. Predicting
  the change removes that problem. Absolute temperature is reconstructed
  from the predictions for reporting.

- Train/test split is chronological (70/30), not random. Shuffling would
  let the model see future data during training.

## Output

```
data/
  raw/                    downloaded source files
  processed/
    merged_dataset.csv
    model_results.csv
    feature_importance.csv
    plots/                7 figures
```
