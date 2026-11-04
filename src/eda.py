"""
STEP 2: Exploratory analysis. Saves 5 figures to reports/figures/ and prints key findings.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR
from src.db import get_cursor


def save(name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=150)
    plt.close()
    print(f"  saved {name}")


def main():
    cur = get_cursor()

    # 1. Weekly total units with a 52-week trend
    w = cur.execute("""SELECT MIN(date) AS week_start, CAST(SUM(units) AS DOUBLE) AS units
                       FROM sales_daily GROUP BY wm_yr_wk HAVING COUNT(DISTINCT date) = 7
                       ORDER BY week_start""").fetchdf()
    plt.figure(figsize=(10, 4))
    plt.plot(w["week_start"], w["units"], lw=0.8, label="weekly units")
    plt.plot(w["week_start"], w["units"].rolling(52, min_periods=26).mean(), lw=2, label="52-week average")
    plt.title("Total weekly units")
    plt.legend()
    save("eda_1_weekly_total.png")
    peak = w.loc[w["units"].idxmax()]
    print(f"  peak week: {peak['week_start'].date()} ({peak['units']:,.0f} units)")

    # 2. Weekly units by category
    c = cur.execute("""SELECT MIN(date) AS week_start, cat_id, CAST(SUM(units) AS DOUBLE) AS units
                       FROM sales_daily GROUP BY wm_yr_wk, cat_id HAVING COUNT(DISTINCT date) = 7
                       ORDER BY week_start""").fetchdf()
    plt.figure(figsize=(10, 4))
    for cat, g in c.groupby("cat_id"):
        plt.plot(g["week_start"], g["units"], lw=1, label=cat)
    plt.title("Weekly units by category")
    plt.legend()
    save("eda_2_by_category.png")

    # 3. Revenue by store
    s = cur.execute("""SELECT store_id, CAST(SUM(revenue) AS DOUBLE) AS revenue
                       FROM sales_daily GROUP BY store_id ORDER BY revenue DESC""").fetchdf()
    plt.figure(figsize=(8, 4))
    plt.bar(s["store_id"], s["revenue"] / 1e6)
    plt.ylabel("Revenue (millions, sampled items)")
    plt.title("Revenue by store")
    save("eda_3_by_store.png")

    # 4. Price change vs unit change (week over week, per item-store series)
    # Weekly units of one item in one store are mostly 0, so a median of % changes is always 0.
    # Instead, per price-change bucket we compare total units this week with total units last week,
    # using only series that sold something last week.
    p = cur.execute("""
        SELECT id, wm_yr_wk, CAST(SUM(units) AS DOUBLE) AS units, AVG(sell_price) AS price
        FROM sales_daily
        WHERE id IN (SELECT id FROM (SELECT DISTINCT id FROM sales_daily) ORDER BY hash(id) LIMIT 1500)
        GROUP BY id, wm_yr_wk ORDER BY id, wm_yr_wk""").fetchdf()
    p["prev_units"] = p.groupby("id")["units"].shift(1)
    p["dp"] = p.groupby("id")["price"].pct_change() * 100
    p = p.dropna(subset=["dp", "prev_units"])
    p = p[p["prev_units"] > 0]
    bins = [-100, -10, -2, 2, 10, 1000]
    labels = ["price down >10%", "down 2-10%", "flat (+-2%)", "up 2-10%", "up >10%"]
    p["bucket"] = pd.cut(p["dp"], bins=bins, labels=labels)
    g = p.groupby("bucket", observed=True)
    change = 100 * (g["units"].sum() / g["prev_units"].sum() - 1)
    plt.figure(figsize=(8, 4))
    plt.bar(change.index.astype(str), change.values)
    plt.axhline(0, color="black", lw=0.8)
    plt.ylabel("change in total units vs previous week (%)")
    plt.title("Do price changes move demand?")
    plt.xticks(rotation=15)
    save("eda_4_price_vs_units.png")
    print("  unit change vs previous week, by price move (%):\n" + change.round(1).to_string())
    print(f"  (based on {len(p):,} series-weeks; observational, promotions and seasonality not controlled)")

    # 5. Event and SNAP lift (weekday-adjusted)
    d = cur.execute("""SELECT date, event_type_1, CAST(SUM(units) AS DOUBLE) AS units, MAX(snap) AS snap
                       FROM sales_daily GROUP BY date, event_type_1 ORDER BY date""").fetchdf()
    d["wd"] = pd.to_datetime(d["date"]).dt.dayofweek
    d["adj"] = d["units"] / d.groupby("wd")["units"].transform("mean")
    base = d.loc[d["event_type_1"].isna(), "adj"].mean()
    lift = {t: 100 * (g["adj"].mean() / base - 1) for t, g in d.dropna(subset=["event_type_1"]).groupby("event_type_1")}
    lift["SNAP day"] = 100 * (d.loc[d["snap"] == 1, "adj"].mean() / d.loc[d["snap"] == 0, "adj"].mean() - 1)
    plt.figure(figsize=(8, 4))
    plt.bar(list(lift), list(lift.values()))
    plt.axhline(0, color="black", lw=0.8)
    plt.ylabel("lift vs normal days (%)")
    plt.title("Event and SNAP effect on daily units (weekday-adjusted)")
    save("eda_5_event_snap_lift.png")
    print("  lift %:", {k: round(float(v), 1) + 0.0 for k, v in lift.items()})

    print("\nDone. Open reports/figures/ and write 3-4 sentences about what you see in your README.")


if __name__ == "__main__":
    main()
