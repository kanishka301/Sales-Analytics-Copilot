"""
Feature engineering for weekly forecasting at store x department level (70 series).

We forecast HORIZON (=4) weeks ahead with ONE model. Every lag / rolling feature is
shifted by at least HORIZON weeks, so at forecast time all of them are already known.
That avoids leakage and avoids recursive forecasting.

Known-in-advance features (price level, SNAP days, event days, calendar) are allowed
because a retailer plans them ahead. avg_price is the plain mean of item prices (NOT
revenue/units), so it carries no information about realised sales.
"""
import numpy as np
import pandas as pd

from src.config import HORIZON

LAGS = sorted({l for l in (HORIZON, HORIZON + 1, HORIZON + 2, HORIZON + 3, 13, 26, 52) if l >= HORIZON})
ROLLS = (4, 8, 13)
CATEGORICAL = ["store_id", "dept_id"]
NUMERIC = (
    [f"lag_{l}" for l in LAGS]
    + [f"rmean_{w}" for w in ROLLS]
    + ["trend_ratio", "avg_price", "price_ratio", "snap_days", "event_days", "woy", "month"]
)
FEATURES = NUMERIC + CATEGORICAL

PANEL_SQL = """
SELECT
    store_id || '_' || dept_id AS series_id,
    store_id, state_id, dept_id, cat_id, wm_yr_wk,
    MIN(date) AS week_start,
    CAST(SUM(units) AS DOUBLE)   AS units,
    CAST(SUM(revenue) AS DOUBLE) AS revenue,
    AVG(sell_price)              AS avg_price,
    COUNT(DISTINCT CASE WHEN snap = 1 THEN date END) AS snap_days,
    COUNT(DISTINCT CASE WHEN event_name_1 IS NOT NULL OR event_name_2 IS NOT NULL THEN date END) AS event_days,
    COUNT(DISTINCT date) AS n_days
FROM sales_daily
GROUP BY store_id, state_id, dept_id, cat_id, wm_yr_wk
HAVING COUNT(DISTINCT date) = 7          -- drop the partial last week
ORDER BY series_id, week_start
"""


def load_weekly_panel(cursor) -> pd.DataFrame:
    """One row per (store, department, week)."""
    df = cursor.execute(PANEL_SQL).fetchdf()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values(["series_id", "week_start"]).reset_index(drop=True)
    df["units"] = df["units"].astype(float)
    df["avg_price"] = df.groupby("series_id")["avg_price"].transform(lambda s: s.ffill().bfill())
    return df


def make_features(panel: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Add lag/rolling/calendar features. Works on history AND on future placeholder rows."""
    df = panel.sort_values(["series_id", "week_start"]).reset_index(drop=True).copy()
    sid = df["series_id"]

    g_units = df.groupby("series_id")["units"]
    for l in LAGS:
        df[f"lag_{l}"] = g_units.shift(l)

    shifted = g_units.shift(horizon)
    for w in ROLLS:
        df[f"rmean_{w}"] = shifted.groupby(sid).transform(
            lambda s, w=w: s.rolling(w, min_periods=max(2, w // 2)).mean()
        )
    df["trend_ratio"] = (df["rmean_4"] / df["rmean_13"]).replace([np.inf, -np.inf], np.nan)

    price_shift = df.groupby("series_id")["avg_price"].shift(horizon)
    price_base = price_shift.groupby(sid).transform(lambda s: s.rolling(13, min_periods=4).mean())
    df["price_ratio"] = (df["avg_price"] / price_base).replace([np.inf, -np.inf], np.nan)

    df["woy"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month

    for c in CATEGORICAL:
        df[c] = pd.Categorical(df[c], categories=sorted(df[c].astype(str).unique()))
    return df


def extend_with_future(panel: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """
    Append `horizon` empty weeks per series so make_features can build their features.
    Assumptions (stated in the README): price stays at its last 4-week average,
    no SNAP days and no events are assumed for the future weeks.
    """
    rows = []
    for sid, g in panel.groupby("series_id"):
        last = g.iloc[-1]
        price = g["avg_price"].tail(4).mean()
        for k in range(1, horizon + 1):
            rows.append({
                "series_id": sid, "store_id": last["store_id"], "state_id": last["state_id"],
                "dept_id": last["dept_id"], "cat_id": last["cat_id"], "wm_yr_wk": np.nan,
                "week_start": last["week_start"] + pd.Timedelta(weeks=k),
                "units": np.nan, "revenue": np.nan, "avg_price": price,
                "snap_days": 0, "event_days": 0, "n_days": 7,
            })
    return pd.concat([panel, pd.DataFrame(rows)], ignore_index=True)
