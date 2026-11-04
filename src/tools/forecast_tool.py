"""Forecast tools used by the agent. The model is trained once and cached."""
import pandas as pd

from src.config import HORIZON, REPORTS_DIR

ALLOWED = {"state_id", "store_id", "cat_id", "dept_id"}
_cache = {}


def _load():
    if "panel" not in _cache:
        from src.db import get_cursor
        from src.forecasting.features import load_weekly_panel
        from src.forecasting.models import forecast_future, load_or_train_final
        panel = load_weekly_panel(get_cursor())
        bundle = load_or_train_final(panel)
        _cache["panel"] = panel
        _cache["future"] = forecast_future(bundle, panel, HORIZON)
    return _cache["panel"], _cache["future"]


def forecast_demand(filters=None, horizon=HORIZON) -> dict:
    """Forecast units for the next `horizon` weeks (max 4) for a state, store, category or department."""
    horizon = int(horizon)
    if not 1 <= horizon <= HORIZON:
        return {"error": f"horizon must be between 1 and {HORIZON} weeks."}
    filters = filters or {}
    bad = set(filters) - ALLOWED
    if bad:
        return {"error": f"Unsupported filter(s) {sorted(bad)}. Forecasts support {sorted(ALLOWED)} only."}

    panel, future = _load()
    mask_f = pd.Series(True, index=future.index)
    mask_p = pd.Series(True, index=panel.index)
    for k, v in filters.items():
        vals = [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
        mask_f &= future[k].astype(str).isin(vals)
        mask_p &= panel[k].astype(str).isin(vals)
    f, p = future[mask_f], panel[mask_p]
    if f.empty:
        return {"error": f"No series match {filters}."}

    weeks = sorted(f["week_start"].unique())[:horizon]
    agg = (f[f["week_start"].isin(weeks)].groupby("week_start")[["forecast", "lower", "upper"]].sum().round(0))
    recent = p.groupby("week_start")["units"].sum().tail(8).round(0)
    return {
        "scope": filters or {"all": True},
        "series_count": int(f["series_id"].nunique()),
        "forecast": [{"week_start": str(i.date()), "forecast_units": int(r["forecast"]),
                      "lower_80": int(r["lower"]), "upper_80": int(r["upper"])} for i, r in agg.iterrows()],
        "recent_actual_weekly_units": {str(i.date()): int(v) for i, v in recent.items()},
        "model": "LightGBM global model (log1p target) with 10%/90% quantile models",
        "assumptions": ["Price stays at its last 4-week average.", "No SNAP days or events assumed in the forecast weeks.",
                        "Lower/upper bounds are sums of per-series 80% bounds for the selected scope, "
                        "so they are conservative (wider than a true 80% interval for the total)."],
        "data_ends": str(panel["week_start"].max().date()),
    }


def backtest_summary() -> dict:
    """How accurate is the forecasting model? Reads the saved backtest results."""
    path = REPORTS_DIR / "forecast_metrics.csv"
    if not path.exists():
        return {"error": "No backtest yet. Run `python -m src.forecasting.evaluate` first."}
    mt = pd.read_csv(path)
    keep = mt[mt["level"].isin(["all_series", "total"])]
    keep = keep.astype(object).where(keep.notna(), None)      # NaN -> None (valid JSON)
    out = {"metric": "WAPE (lower is better); skill_vs_snaive > 0 means better than seasonal-naive",
           "results": keep.to_dict("records")}
    cov = mt["coverage_80"].dropna()
    if len(cov):
        out["lgbm_80pct_interval_coverage"] = float(cov.iloc[0])
    return out
