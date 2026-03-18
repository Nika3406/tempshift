import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

FEATURES = [
    "co2_ppm", "co2_deseasonalized", "ch4_ppb", "n2o_ppb",
    "co2_growth_rate", "ch4_growth_rate", "co2_12mo_avg", "ch4_12mo_avg",
    "tsi_anomaly", "total_ghg", "total_anthropogenic_co2",
    "energy_per_capita", "volcanic_flag", "time_since_1960",
]


def metrics(y_true, y_pred, label=None):
    m = {
        "r2":   round(r2_score(y_true, y_pred), 4),
        "mae":  round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
    }
    if label:
        print(f"    {label}: R2={m['r2']}  MAE={m['mae']}  RMSE={m['rmse']}")
    return m


def load_data():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "merged_dataset.csv"))

    # Predicting absolute temperature fails on a chronological split because
    # 2015-2026 runs ~0.5C warmer than 1984-2015 on average - any model trained
    # on the earlier period will systematically underpredict the test set.
    # Predicting the month-over-month change removes that distribution shift.
    # Absolute temperature is reconstructed from predicted changes for reporting.
    df["temp_change"] = df["temp_anomaly_c"].diff()
    df["temp_lag1"] = df["temp_anomaly_c"].shift(1)
    df["temp_lag12"] = df["temp_anomaly_c"].shift(12)
    df["co2_change_12mo"] = df["co2_ppm"].diff(12)

    features = [f for f in FEATURES + ["temp_lag1", "temp_lag12", "co2_change_12mo"]
                if f in df.columns]
    sub = df[features + ["temp_change", "temp_anomaly_c", "date"]].dropna().reset_index(drop=True)

    split = int(len(sub) * 0.70)
    train, test = sub.iloc[:split], sub.iloc[split:]

    print(f"  {sub.shape[0]} rows x {len(features)} features")
    print(f"  train: {len(train)} ({train['date'].iloc[0]} -> {train['date'].iloc[-1]})")
    print(f"  test:  {len(test)} ({test['date'].iloc[0]} -> {test['date'].iloc[-1]})")

    load_data.y_test_abs = test["temp_anomaly_c"].values
    load_data.test_lag1 = test["temp_lag1"].values

    return train[features], train["temp_change"], test[features], test["temp_change"], features


def train_linear(X_tr, y_tr, X_te, y_te, features):
    sc = StandardScaler()
    model = LinearRegression().fit(sc.fit_transform(X_tr), y_tr)
    preds = model.predict(sc.transform(X_te))
    m = metrics(y_te, preds)
    imp = pd.Series(np.abs(model.coef_), index=features).sort_values(ascending=False)
    print(f"  linear  diff-R2={m['r2']}", end="")
    if hasattr(load_data, "test_lag1"):
        abs_r2 = metrics(load_data.y_test_abs, load_data.test_lag1 + preds)["r2"]
        print(f"  abs-R2={abs_r2}", end="")
    print(f"  MAE={m['mae']}  top: {imp.index[:3].tolist()}")
    return m, imp, model, sc


def train_rf(X_tr, y_tr, X_te, y_te, features):
    model = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                  max_features=0.7, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    m = metrics(y_te, preds)
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print(f"  rf      diff-R2={m['r2']}", end="")
    if hasattr(load_data, "test_lag1"):
        abs_r2 = metrics(load_data.y_test_abs, load_data.test_lag1 + preds)["r2"]
        print(f"  abs-R2={abs_r2}", end="")
    print(f"  MAE={m['mae']}  top: {imp.index[:3].tolist()}")
    return m, imp, model


def train_gbt(X_tr, y_tr, X_te, y_te, features):
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                             subsample=0.8, colsample_bytree=0.7,
                             reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)
        label = "xgb"
    except ImportError:
        model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                                          subsample=0.8, min_samples_leaf=5, random_state=42)
        label = "gbt"
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    m = metrics(y_te, preds)
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print(f"  {label}     diff-R2={m['r2']}", end="")
    if hasattr(load_data, "test_lag1"):
        abs_r2 = metrics(load_data.y_test_abs, load_data.test_lag1 + preds)["r2"]
        print(f"  abs-R2={abs_r2}", end="")
    print(f"  MAE={m['mae']}  top: {imp.index[:3].tolist()}")
    return m, imp, model


def consensus_importance(imp_lr, imp_rf, imp_gbt):
    def norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s

    df = pd.DataFrame({
        "linear_reg":    norm(imp_lr),
        "random_forest": norm(imp_rf),
        "grad_boost":    norm(imp_gbt),
    })
    df["consensus_score"] = df.mean(axis=1)
    return df.sort_values("consensus_score", ascending=False).reset_index().rename(columns={"index": "feature"})


def run():
    print("=" * 50)
    X_tr, y_tr, X_te, y_te, features = load_data()

    print()
    m_lr, imp_lr, model_lr, sc = train_linear(X_tr, y_tr, X_te, y_te, features)
    m_rf, imp_rf, model_rf     = train_rf(X_tr, y_tr, X_te, y_te, features)
    m_gbt, imp_gbt, model_gbt  = train_gbt(X_tr, y_tr, X_te, y_te, features)

    results = pd.DataFrame([
        {"model": "Linear Regression", **m_lr},
        {"model": "Random Forest", **m_rf},
        {"model": "Gradient Boost", **m_gbt},
    ])
    results.to_csv(os.path.join(PROCESSED_DIR, "model_results.csv"), index=False)

    imp_df = consensus_importance(imp_lr, imp_rf, imp_gbt)
    imp_df.to_csv(os.path.join(PROCESSED_DIR, "feature_importance.csv"), index=False)

    print("\nfeature ranking:")
    print(imp_df[["feature", "consensus_score"]].to_string(index=False))
    print("=" * 50)

    return {"results": results, "importance": imp_df,
            "models": {"linear": (model_lr, sc), "rf": model_rf, "gbt": model_gbt},
            "features": features}


if __name__ == "__main__":
    run()
