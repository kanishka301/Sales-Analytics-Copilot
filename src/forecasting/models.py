"""
Models.

Baselines / statistical (one model per series, history only):
    naive, snaive (same week last year), mavg4, theta, ets
Machine learning (one GLOBAL model across all 70 series):
    lgbm  -> median forecast + 10%/90% quantile models = an 80% prediction interval

The LightGBM target is log1p(units).
"""
import warnings

import numpy as np
import pandas as pd

from src.config import HORIZON, MODELS_DIR
from src.forecasting.features import FEATURES, extend_with_future, make_features

# ---------------------------------------------------------------- baselines
def naive(hist: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(hist[-1], h)


def seasonal_naive(hist: np.ndarray, h: int, m: int = 52) -> np.ndarray:
    if len(hist) < m:
        return naive(hist, h)
    start = len(hist) - m
    return hist[start:start + h].astype(float)


def moving_average(hist: np.ndarray, h: int, w: int = 4) -> np.ndarray:
    return np.repeat(hist[-w:].mean(), h)


def theta(hist: np.ndarray, h: int) -> np.ndarray:
    try:
        from statsmodels.tsa.forecasting.theta import ThetaModel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ThetaModel(pd.Series(hist.astype(float)), period=52, deseasonalize=True).fit()
            return np.clip(np.asarray(res.forecast(h)), 0, None)
    except Exception:
        return seasonal_naive(hist, h)


def ets(hist: np.ndarray, h: int) -> np.ndarray:
    """Holt-Winters (damped additive trend, additive yearly seasonality). Slow: ~1-2s per series."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(
                hist.astype(float), trend="add", damped_trend=True,
                seasonal="add", seasonal_periods=52, initialization_method="estimated",
            ).fit()
            return np.clip(np.asarray(fit.forecast(h)), 0, None)
    except Exception:
        return seasonal_naive(hist, h)


BASELINES = {
    "naive": naive,
    "snaive": seasonal_naive,
    "mavg4": moving_average,
    "theta": theta,
    "ets": ets,
}

# ---------------------------------------------------------------- LightGBM
QUANTILES = {"lo": 0.10, "hi": 0.90}
PARAMS = dict(
    n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1,
)


def fit_lgbm_bundle(train: pd.DataFrame) -> dict:
    import lightgbm as lgb
    X = train[FEATURES]
    y = np.log1p(train["units"].to_numpy())
    models = {"mid": lgb.LGBMRegressor(objective="regression", **PARAMS).fit(X, y)}
    for name, alpha in QUANTILES.items():
        models[name] = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **PARAMS).fit(X, y)
    return models


def predict_bundle(models: dict, X: pd.DataFrame) -> dict:
    out = {k: np.clip(np.expm1(m.predict(X)), 0, None) for k, m in models.items()}
    # quantile models are trained separately and can cross; fix that
    out["lo"] = np.minimum(out["lo"], out["mid"])
    out["hi"] = np.maximum(out["hi"], out["mid"])
    return out


# ------------------------------------------------- final model + real forecast
FINAL_PATH = MODELS_DIR / "lgbm_final.joblib"


def _fingerprint(panel: pd.DataFrame) -> str:
    """Identifies the data a model was trained on (last week, size, total units).
    A saved model is reused only if this matches, so re-running data_prep with a different
    sample size never silently reuses a stale model."""
    return f"{panel['week_start'].max().date()}|{len(panel)}|{panel['units'].sum():.0f}"


def train_final(panel: pd.DataFrame) -> dict:
    feats = make_features(panel)
    train = feats.dropna(subset=[f"lag_{HORIZON}"])
    models = fit_lgbm_bundle(train)
    return {"models": models, "trained_through": str(panel["week_start"].max().date()),
            "fingerprint": _fingerprint(panel)}


def load_or_train_final(panel: pd.DataFrame) -> dict:
    import joblib
    if FINAL_PATH.exists():
        bundle = joblib.load(FINAL_PATH)
        if bundle.get("fingerprint") == _fingerprint(panel):
            return bundle
    bundle = train_final(panel)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, FINAL_PATH)
    return bundle


def forecast_future(bundle: dict, panel: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Forecast the `horizon` weeks after the end of the data, for every series."""
    last = panel["week_start"].max()
    feats = make_features(extend_with_future(panel, horizon))
    fut = feats[feats["week_start"] > last].copy()
    pred = predict_bundle(bundle["models"], fut[FEATURES])
    fut["forecast"], fut["lower"], fut["upper"] = pred["mid"], pred["lo"], pred["hi"]
    return fut[["series_id", "store_id", "state_id", "dept_id", "cat_id", "week_start",
                "forecast", "lower", "upper"]]
