"""
Root-cause statistics. These functions take plain DataFrames (no database), so they
are easy to unit-test. The database wrappers live in analysis_tools.py.
"""
import numpy as np
import pandas as pd
from scipy import stats


def _r(x, n=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


# ------------------------------------------------------------------ price / volume
def price_volume_effect(a: pd.DataFrame, b: pd.DataFrame, key: str = "id") -> dict:
    """
    Split the REVENUE change between period A (baseline) and B (comparison) into:
      price effect   = (P_b - P_a) * Q_b          for series sold in both periods
      volume effect  = (Q_b - Q_a) * P_a          for series sold in both periods
      new/lost items = revenue of series sold in only one period
    where P = revenue / units. The three parts add up exactly to the total change.
    a, b need columns: key, units, revenue.
    """
    ga = a.groupby(key)[["units", "revenue"]].sum().rename(columns={"units": "q_a", "revenue": "r_a"})
    gb = b.groupby(key)[["units", "revenue"]].sum().rename(columns={"units": "q_b", "revenue": "r_b"})
    m = ga.join(gb, how="outer").fillna(0.0)
    m["p_a"] = m["r_a"] / m["q_a"].replace(0, np.nan)
    m["p_b"] = m["r_b"] / m["q_b"].replace(0, np.nan)
    both = (m["q_a"] > 0) & (m["q_b"] > 0)

    price = float(((m["p_b"] - m["p_a"]) * m["q_b"])[both].sum())
    volume = float(((m["q_b"] - m["q_a"]) * m["p_a"])[both].sum())
    new_lost = float(m.loc[~both, "r_b"].sum() - m.loc[~both, "r_a"].sum())
    total = float(m["r_b"].sum() - m["r_a"].sum())
    ra, qa, qb = float(m["r_a"].sum()), float(m["q_a"].sum()), float(m["q_b"].sum())

    def share(x):
        return _r(100 * x / total) if total else None

    return {
        "revenue_a": _r(ra), "revenue_b": _r(m["r_b"].sum()),
        "revenue_change": _r(total), "revenue_change_pct": _r(100 * total / ra) if ra else None,
        "units_a": _r(qa, 0), "units_b": _r(qb, 0), "units_change_pct": _r(100 * (qb - qa) / qa) if qa else None,
        "price_effect": _r(price), "volume_effect": _r(volume), "new_or_lost_items_effect": _r(new_lost),
        "price_effect_share_pct": share(price), "volume_effect_share_pct": share(volume),
        "new_or_lost_share_pct": share(new_lost),
        "reconciliation_error": _r(total - (price + volume + new_lost), 4),
    }


def segment_contributions(a: pd.DataFrame, b: pd.DataFrame, by: str, value: str = "revenue", top: int = 8) -> list:
    """Which segments (state, store, category...) explain the change in `value`?"""
    ga = a.groupby(by)[value].sum().rename("a")
    gb = b.groupby(by)[value].sum().rename("b")
    df = pd.concat([ga, gb], axis=1).fillna(0.0)
    df["change"] = df["b"] - df["a"]
    total = df["change"].sum()
    df["share_of_total_change_pct"] = 100 * df["change"] / total if total != 0 else np.nan
    df["change_pct"] = 100 * df["change"] / df["a"].replace(0, np.nan)
    df = df.reindex(df["change"].abs().sort_values(ascending=False).index).head(top)
    return df.reset_index().round(2).replace({np.nan: None}).to_dict("records")


# ------------------------------------------------------------------ events / SNAP
def flag_effect(daily: pd.DataFrame, flag: str) -> dict:
    """
    Do days with `flag` True (event day, SNAP day) sell more than other days?
    Units are first divided by the weekday average so weekends do not bias the result.
    daily needs columns: date, units, <flag>.
    """
    d = daily.copy()
    d["wd"] = pd.to_datetime(d["date"]).dt.dayofweek
    d["adj"] = d["units"] / d.groupby("wd")["units"].transform("mean")
    on, off = d.loc[d[flag].astype(bool), "adj"], d.loc[~d[flag].astype(bool), "adj"]
    if len(on) < 3 or len(off) < 3:
        return {"note": f"Too few '{flag}' days ({len(on)}) vs others ({len(off)}) to test."}
    t, p = stats.ttest_ind(on, off, equal_var=False)
    return {
        "flag_days": int(len(on)), "other_days": int(len(off)),
        "lift_pct": _r(100 * (on.mean() / off.mean() - 1)),
        "p_value": _r(p, 4), "significant_at_5pct": bool(p < 0.05),
        "method": "Welch t-test on weekday-adjusted daily units",
    }


# ------------------------------------------------------------------ elasticity
def price_elasticity(df: pd.DataFrame) -> dict:
    """
    Log-log regression with series and week fixed effects:
        log(units) = series_effect + week_effect + beta * log(price)
    beta is the price elasticity (about -1.5 means +10% price -> ~-15% units).
    This is an ASSOCIATION from observational data, not a causal estimate.
    df needs columns: id, week, units, avg_price.
    """
    d = df[(df["units"] > 0) & (df["avg_price"] > 0)].copy()
    d = d[d.groupby("id")["id"].transform("size") >= 8]
    if len(d) < 50 or d["id"].nunique() < 3:
        return {"error": "Not enough data for an elasticity estimate (need 3+ series with 8+ weeks)."}

    d["y"], d["x"] = np.log(d["units"]), np.log(d["avg_price"])
    for _ in range(6):  # alternating demeaning = two-way fixed effects
        for col in ("y", "x"):
            d[col] = d[col] - d.groupby("id")[col].transform("mean")
            d[col] = d[col] - d.groupby("week")[col].transform("mean")

    x, y = d["x"].to_numpy(), d["y"].to_numpy()
    sxx = float(x @ x)
    if sxx < 1e-9:
        return {"error": "No price variation in this selection, elasticity cannot be estimated."}
    beta = float(x @ y) / sxx
    resid = y - beta * x
    dof = max(len(d) - d["id"].nunique() - d["week"].nunique() - 1, 1)
    se = float(np.sqrt((resid @ resid) / dof / sxx))
    tcrit = stats.t.ppf(0.975, dof)
    return {
        "elasticity": _r(beta, 3), "std_error": _r(se, 3),
        "ci95": [_r(beta - tcrit * se, 3), _r(beta + tcrit * se, 3)],
        "n_obs": int(len(d)), "n_series": int(d["id"].nunique()),
        "interpretation": f"A 10% higher price is associated with about {beta * 10:.1f}% change in weekly units.",
        "caveat": "Observational association; promotions and seasonality can bias it. Not causal.",
    }


# ------------------------------------------------------------------ anomalies
def detect_anomalies(series: pd.Series, window: int = 13, z_thresh: float = 2.5) -> list:
    """
    Flag points far from the mean of the PREVIOUS `window` points (z-score).
    Limitation: ignores yearly seasonality, so a normal holiday peak can be flagged.
    """
    s = series.astype(float)
    prev_mean = s.shift(1).rolling(window, min_periods=max(4, window // 2)).mean()
    prev_std = s.shift(1).rolling(window, min_periods=max(4, window // 2)).std()
    z = (s - prev_mean) / prev_std.replace(0, np.nan)
    out = pd.DataFrame({"value": s, "expected": prev_mean, "z": z})
    out = out[out["z"].abs() >= z_thresh].round(2)
    out.index = [str(i)[:10] for i in out.index]
    return out.reset_index(names="week").replace({np.nan: None}).to_dict("records")
