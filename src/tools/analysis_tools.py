"""Tools that pull data from DuckDB and hand it to the pure functions in stats_tool.py."""
import datetime as dt

import pandas as pd

from src.tools import stats_tool as st

ALLOWED_FILTERS = {"state_id", "store_id", "cat_id", "dept_id", "item_id"}
SEG_COLS = "state_id, store_id, cat_id, dept_id, item_id"


def _cursor():
    from src.db import get_cursor
    return get_cursor()


def _date(x, name: str) -> dt.date:
    try:
        return dt.date.fromisoformat(str(x)[:10])
    except Exception:
        raise ValueError(f"{name} must be a date like 2015-03-31, got {x!r}")


def _where(filters, start=None, end=None):
    clauses, params = [], []
    if start is not None:
        clauses.append("date >= ?")
        params.append(start)
    if end is not None:
        clauses.append("date <= ?")
        params.append(end)
    for k, v in (filters or {}).items():
        if k not in ALLOWED_FILTERS:
            raise ValueError(f"Unsupported filter '{k}'. Allowed: {sorted(ALLOWED_FILTERS)}")
        vals = [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
        clauses.append(f"{k} IN ({', '.join('?' for _ in vals)})")
        params.extend(vals)
    return ("WHERE " + " AND ".join(clauses)) if clauses else "", params


def _period_df(cur, start, end, filters) -> pd.DataFrame:
    where, params = _where(filters, start, end)
    sql = (f"SELECT id, {SEG_COLS}, CAST(SUM(units) AS DOUBLE) AS units, CAST(SUM(revenue) AS DOUBLE) AS revenue "
           f"FROM sales_daily {where} GROUP BY id, {SEG_COLS}")
    return cur.execute(sql, params).fetchdf()


def _daily_df(cur, start, end, filters) -> pd.DataFrame:
    where, params = _where(filters, start, end)
    sql = f"""
        SELECT date, CAST(SUM(units) AS DOUBLE) AS units,
               MAX(CASE WHEN event_name_1 IS NOT NULL OR event_name_2 IS NOT NULL THEN 1 ELSE 0 END) = 1 AS is_event,
               MAX(snap) = 1 AS is_snap
        FROM sales_daily {where} GROUP BY date ORDER BY date"""
    return cur.execute(sql, params).fetchdf()


def _single_state(filters) -> bool:
    f = filters or {}
    n = lambda k: len(f[k]) if isinstance(f.get(k), (list, tuple)) else (1 if k in f else 0)
    return n("state_id") == 1 or n("store_id") == 1


# ------------------------------------------------------------------ decompose_change
def decompose_change(period_a_start, period_a_end, period_b_start, period_b_end,
                     filters=None, by="cat_id") -> dict:
    """Why did sales change between period A (baseline) and period B? Periods must be equal length."""
    if by not in ALLOWED_FILTERS:
        raise ValueError(f"'by' must be one of {sorted(ALLOWED_FILTERS)}")
    a0, a1 = _date(period_a_start, "period_a_start"), _date(period_a_end, "period_a_end")
    b0, b1 = _date(period_b_start, "period_b_start"), _date(period_b_end, "period_b_end")
    if (a1 - a0).days != (b1 - b0).days:
        return {"error": f"Periods must have equal length (A={(a1 - a0).days + 1} days, B={(b1 - b0).days + 1} days)."}

    cur = _cursor()
    a, b = _period_df(cur, a0, a1, filters), _period_df(cur, b0, b1, filters)
    if a.empty or b.empty:
        return {"error": "No data in one of the periods for these filters."}

    out = {
        "period_a": f"{a0} to {a1}", "period_b": f"{b0} to {b1}", "filters": filters or {},
        "price_volume_decomposition": st.price_volume_effect(a, b),
        "top_segments_by_revenue_change": st.segment_contributions(a, b, by, "revenue"),
        "top_segments_by_units_change": st.segment_contributions(a, b, by, "units"),
        "segment_dimension": by,
    }
    daily = pd.concat([_daily_df(cur, a0, a1, filters), _daily_df(cur, b0, b1, filters)])
    out["event_effect"] = st.flag_effect(daily, "is_event")
    out["snap_effect"] = (st.flag_effect(daily, "is_snap") if _single_state(filters)
                          else {"note": "SNAP effect only computed for a single state or store."})
    out["notes"] = ["Data has no inventory, promotion or weather fields, so stockouts and promotions "
                    "cannot be separated from price effects.",
                    "'new_or_lost_items' means item-store series with zero units in one of the two periods; "
                    "it includes sparse low-volume items, not only real launches or delistings."]
    return out


# ------------------------------------------------------------------ detect_anomalies
def detect_anomalies(filters=None, start=None, end=None, metric="units") -> dict:
    if metric not in ("units", "revenue"):
        raise ValueError("metric must be 'units' or 'revenue'")
    s0 = _date(start, "start") if start else None
    s1 = _date(end, "end") if end else None
    where, params = _where(filters, s0, s1)
    sql = (f"SELECT MIN(date) AS week_start, CAST(SUM({metric}) AS DOUBLE) AS value FROM sales_daily {where} "
           f"GROUP BY wm_yr_wk HAVING COUNT(DISTINCT date) = 7 ORDER BY week_start")
    df = _cursor().execute(sql, params).fetchdf()
    if len(df) < 20:
        return {"error": "Need at least 20 full weeks of data."}
    series = pd.Series(df["value"].to_numpy(), index=pd.to_datetime(df["week_start"]))
    return {
        "metric": metric, "weeks_checked": int(len(series)), "filters": filters or {},
        "anomalies": st.detect_anomalies(series),
        "limitation": "z-score vs previous 13 weeks; yearly seasonal peaks (e.g. holidays) may be flagged.",
    }


# ------------------------------------------------------------------ price_elasticity
def price_elasticity(filters=None, start=None, end=None) -> dict:
    s0 = _date(start, "start") if start else None
    s1 = _date(end, "end") if end else None
    where, params = _where(filters, s0, s1)
    sql = f"""
        SELECT id, wm_yr_wk AS week, CAST(SUM(units) AS DOUBLE) AS units, AVG(sell_price) AS avg_price
        FROM sales_daily
        WHERE id IN (SELECT id FROM (SELECT DISTINCT id FROM sales_daily {where}) ORDER BY hash(id) LIMIT 2000)
          {('AND ' + where[6:]) if where else ''}
        GROUP BY id, wm_yr_wk"""
    df = _cursor().execute(sql, params + params).fetchdf()
    res = st.price_elasticity(df)
    res["scope"] = filters or {"all": True}
    res["sampled_series_cap"] = 2000
    return res
