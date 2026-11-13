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
    plt.savefig(FIGURES_DIR / name)
    plt.close()
    print(f"  saved {name}")


def main():
    cur = get_cursor()

    # 1. Weekly total units with a 52-week trend
    w = cur.execute("""SELECT MIN(date) AS week_start, CAST(SUM(units) AS DOUBLE) AS units
                       FROM sales_daily GROUP BY wm_yr_wk HAVING COUNT(DISTINCT date) = 7
                       ORDER BY week_start""").fetchdf() #converts the result into a floating-point number.
# This ensures the units column is represented numerically as a double-precision value, which is convenient for subsequent calculations.
    plt.figure(figsize=(10, 4))
    plt.plot(w["week_start"], w["units"], lw=0.8, label="weekly units")
    plt.plot(w["week_start"], w["units"].rolling(52, min_periods=26).mean(), lw=2, label="52-week average")
    plt.title("Total weekly units")
    plt.legend()
    save("eda_1_weekly_total.png")

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
    # 4. Price change vs units sold

    p = cur.execute("""
        SELECT id,wm_yr_wk,SUM(units) AS units,AVG(sell_price) AS price
        FROM sales_daily
        GROUP BY id, wm_yr_wk
        ORDER BY id, wm_yr_wk
    """).fetchdf()
    # Calculate previous week's units and price
    p["prev_units"] = p.groupby("id")["units"].shift(1)
    p["prev_price"] = p.groupby("id")["price"].shift(1)
    # Calculate percentage changes
    p["unit_change"] = p.groupby("id")["units"].pct_change() * 100
    p["price_change"] = p.groupby("id")["price"].pct_change() * 100
    # Remove missing and infinite values
    p = p.replace([np.inf, -np.inf], np.nan).dropna(subset=["unit_change", "price_change"])

    # Plot
    plt.figure(figsize=(8, 4))
    plt.scatter(p["price_change"], p["unit_change"])
    plt.xlabel("Price change (%)")
    plt.ylabel("Units sold change (%)")
    plt.title("Price change vs demand change")
    save("eda_4_price_vs_units.png")

    # 5. Event and SNAP impact

    d = cur.execute("""
        SELECT date,event_type_1,SUM(units) AS units,MAX(snap) AS snap
        FROM sales_daily
        GROUP BY date, event_type_1
        ORDER BY date
        """).fetchdf()

    # Average sales on normal days
    normal = d.loc[d["event_type_1"].isna(), "units"].mean()

    # Average sales on event days
    event = d.loc[d["event_type_1"].notna(), "units"].mean()

    # Average sales on SNAP and non-SNAP days
    snap = d.loc[d["snap"] == 1, "units"].mean()
    non_snap = d.loc[d["snap"] == 0, "units"].mean()

    # Calculate percentage lift
    labels = ["Event days", "SNAP days"]
    lift = [
        (event / normal - 1) * 100,
        (snap / non_snap - 1) * 100
    ]

    # Plot
    plt.figure(figsize=(7, 4))
    plt.bar(labels, lift)
    plt.axhline(0, color="black", lw=0.8)
    plt.ylabel("Sales lift (%)")
    plt.title("Event and SNAP impact on sales")
    save("eda_5_event_snap_lift.png")

if __name__ == "__main__":
    main()
