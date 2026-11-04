"""
STEP 1: Data preparation.

1. Reads the 3 raw M5 CSVs straight into DuckDB.
2. Reshapes sales from WIDE (one column per day: d_1 ... d_1941)
   to LONG (one row per item-store-day).
3. Joins calendar info (date, events, SNAP) and weekly prices.
4. Drops rows from BEFORE an item was first sold in a store
   (M5 has no price for those weeks, so a "0 sales" there is not a real zero).
5. Builds two tables: sales_daily and sales_weekly.
6. Prints sanity checks so you can confirm the data looks right.

from the project root:
    python -m src.data_prep                 # sample of 600 items
    python -m src.data_prep --n-items 0     # all 3,049 items (needs more RAM/time)
"""
import argparse
import time

import duckdb

from src.config import (
    CALENDAR_CSV, PRICES_CSV, SALES_CSV,
    PROCESSED_DIR, DB_PATH, DAILY_PARQUET,
)


def sql_path(p):
    """DuckDB wants forward slashes, even on Windows."""
    return str(p).replace("\\", "/")


def check_inputs():
    missing = [p for p in (CALENDAR_CSV, PRICES_CSV, SALES_CSV) if not p.exists()]
    if missing:
        names = "\n  ".join(str(p) for p in missing)
        raise SystemExit(
            "Missing raw files:\n  " + names +
            "\n\nDownload the M5 data from Kaggle (competition: m5-forecasting-accuracy),"
            "\nunzip it, and put calendar.csv, sell_prices.csv and "
            "sales_train_evaluation.csv in data/raw/"
        )


def build(n_items):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()  # always rebuild from scratch so results are reproducible
    con = duckdb.connect(str(DB_PATH))

    # ---- 1. Load raw CSVs -------------------------------------------------
    print("Loading calendar ...")
    con.execute(f"""
        CREATE TABLE calendar AS
        SELECT * FROM read_csv_auto('{sql_path(CALENDAR_CSV)}')
    """)

    print("Loading prices ...")
    con.execute(f"""
        CREATE TABLE prices AS
        SELECT * FROM read_csv_auto('{sql_path(PRICES_CSV)}')
    """)

    print("Loading sales (wide format) ...")
    con.execute(f"""
        CREATE TABLE sales_wide_full AS
        SELECT * FROM read_csv_auto('{sql_path(SALES_CSV)}')
    """)

    # ---- 2. Optional sampling by ITEM -------------------------------------
    # We sample items (products), not rows, so every sampled item keeps all
    # 10 stores. That matters later for state/store root-cause analysis.
    if n_items and n_items > 0:
        print(f"Sampling {n_items} items (deterministic) ...")
        con.execute(f"""
            CREATE TABLE sales_wide AS
            SELECT * FROM sales_wide_full
            WHERE item_id IN (
                SELECT item_id FROM (SELECT DISTINCT item_id FROM sales_wide_full)
                ORDER BY hash(item_id)
                LIMIT {int(n_items)}
            )
        """)
    else:
        con.execute("CREATE TABLE sales_wide AS SELECT * FROM sales_wide_full")
    con.execute("DROP TABLE sales_wide_full")

    # ---- 3. Wide -> long --------------------------------------------------
    print("Reshaping wide -> long (UNPIVOT) ...")
    con.execute("""
        CREATE TABLE sales_long AS
        UNPIVOT sales_wide
        ON COLUMNS(* EXCLUDE (id, item_id, dept_id, cat_id, store_id, state_id))
        INTO NAME d VALUE units
    """)
    con.execute("DROP TABLE sales_wide")

    # ---- 4. Join calendar + prices ----------------------------------------
    # INNER JOIN with prices removes pre-launch rows (no price = not on sale yet).
    print("Joining calendar and prices ...")
    con.execute("""
        CREATE TABLE sales_daily AS
        SELECT
            s.id, s.item_id, s.dept_id, s.cat_id, s.store_id, s.state_id,
            c.date, c.wm_yr_wk, c.wday, c.month, c.year,
            c.event_name_1, c.event_type_1, c.event_name_2, c.event_type_2,
            CASE s.state_id
                WHEN 'CA' THEN c.snap_CA
                WHEN 'TX' THEN c.snap_TX
                WHEN 'WI' THEN c.snap_WI
            END AS snap,
            p.sell_price,
            s.units,
            s.units * p.sell_price AS revenue
        FROM sales_long s
        JOIN calendar c ON s.d = c.d
        JOIN prices   p ON p.store_id = s.store_id
                       AND p.item_id  = s.item_id
                       AND p.wm_yr_wk = c.wm_yr_wk
    """)
    con.execute("DROP TABLE sales_long")

    # ---- 5. Weekly table (what we will forecast later) --------------------
    print("Building weekly table ...")
    con.execute("""
        CREATE TABLE sales_weekly AS
        SELECT
            id, item_id, dept_id, cat_id, store_id, state_id,
            wm_yr_wk,
            MIN(date)       AS week_start,
            SUM(units)      AS units,
            SUM(revenue)    AS revenue,
            AVG(sell_price) AS avg_price,
            SUM(snap)       AS snap_days,
            COUNT(*)        AS days_in_week
        FROM sales_daily
        GROUP BY id, item_id, dept_id, cat_id, store_id, state_id, wm_yr_wk
    """)

    # ---- 6. Save a parquet copy of the daily table ------------------------
    print("Writing parquet ...")
    con.execute(f"COPY sales_daily TO '{sql_path(DAILY_PARQUET)}' (FORMAT PARQUET)")

    return con


def sanity_checks(con):
    print("\n" + "=" * 60)
    print("SANITY CHECKS (read these, they tell you if the data is right)")
    print("=" * 60)

    n_daily, n_series, d_min, d_max = con.execute("""
        SELECT COUNT(*), COUNT(DISTINCT id), MIN(date), MAX(date) FROM sales_daily
    """).fetchone()
    print(f"sales_daily rows : {n_daily:,}")
    print(f"time series      : {n_series:,}  (item x store)")
    print(f"date range       : {d_min} -> {d_max}")

    n_weekly = con.execute("SELECT COUNT(*) FROM sales_weekly").fetchone()[0]
    print(f"sales_weekly rows: {n_weekly:,}")

    nulls = con.execute("""
        SELECT SUM(CASE WHEN sell_price IS NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN units IS NULL THEN 1 ELSE 0 END)
        FROM sales_daily
    """).fetchone()
    print(f"null prices / null units (both should be 0): {nulls[0]} / {nulls[1]}")

    zero_share = con.execute(
        "SELECT AVG(CASE WHEN units = 0 THEN 1.0 ELSE 0.0 END) FROM sales_daily"
    ).fetchone()[0]
    print(f"share of item-store-days with zero sales: {zero_share:.1%} "
          "(retail data is sparse, this is normal)")

    print("\nUnits by category:")
    for cat, units in con.execute(
        "SELECT cat_id, SUM(units) FROM sales_daily GROUP BY cat_id ORDER BY 2 DESC"
    ).fetchall():
        print(f"  {cat:<12}{int(units):>14,}")

    print("\nUnits by state:")
    for st, units in con.execute(
        "SELECT state_id, SUM(units) FROM sales_daily GROUP BY state_id ORDER BY 2 DESC"
    ).fetchall():
        print(f"  {st:<12}{int(units):>14,}")

    print("\nSample rows:")
    print(con.execute("SELECT * FROM sales_daily LIMIT 3").fetchdf().T.to_string())


def main():
    parser = argparse.ArgumentParser(description="Prepare M5 data into DuckDB.")
    parser.add_argument(
        "--n-items", type=int, default=600,
        help="Number of items to sample (default 600, about 6,000 series). "
             "Use 0 for all 3,049 items.",
    )
    args = parser.parse_args()

    check_inputs()
    t0 = time.time()
    con = build(args.n_items)
    sanity_checks(con)
    con.close()
    print(f"\nDone in {time.time() - t0:.0f}s. Database: {DB_PATH}")


if __name__ == "__main__":
    main()
