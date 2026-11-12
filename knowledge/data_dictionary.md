TABLE sales_daily: one row per item, store and day (Walmart M5 data, 2011-01-29 to 2016-05-22). Columns: id (item_store key, one time series), item_id, dept_id, cat_id, store_id, state_id, date (DATE), wm_yr_wk (Walmart week id, weeks run Saturday to Friday), wday, month (1-12), year, event_name_1, event_type_1, event_name_2, event_type_2, snap (1 if SNAP food-assistance purchases allowed that day in that state), sell_price (price per unit that week), units (units sold that day), revenue (units * sell_price).

TABLE sales_weekly: one row per item, store and week. Columns: id, item_id, dept_id, cat_id, store_id, state_id, wm_yr_wk, week_start (DATE), units, revenue, avg_price, snap_days, days_in_week. Use sales_daily when you need exact dates; use sales_weekly for week-level trends.

HIERARCHY: 3 states (CA, TX, WI), 10 stores (CA_1, CA_2, CA_3, CA_4, TX_1, TX_2, TX_3, WI_1, WI_2, WI_3), 3 categories (FOODS, HOBBIES, HOUSEHOLD), 7 departments (FOODS_1, FOODS_2, FOODS_3, HOBBIES_1, HOBBIES_2, HOUSEHOLD_1, HOUSEHOLD_2). item_id looks like FOODS_3_090. Stores belong to the state in their name prefix.

METRICS: sales means units sold unless the question says revenue or dollars. Revenue = SUM(revenue). Average selling price = SUM(revenue) / SUM(units). A sales drop is the percentage change in SUM(units) between two equal-length periods. Always compare equal-length periods, for example the same month in two years.

DATA LIMITS: the data has no inventory, no promotions flag and no weather. A zero in units may mean a stockout or simply no demand. Rows before an item was first priced in a store are removed. The dataset is a SAMPLE of items, so totals are smaller than the real Walmart totals. Only the last week of the data is partial and is excluded from weekly forecasts.

EVENTS: event_name_1 and event_type_1 hold holidays and events (types: Sporting, Cultural, National, Religious). Use event_name_1 IS NOT NULL to find event days. SNAP days differ by state, use the snap column together with state_id.

DATES: use DATE literals like DATE '2015-03-31' or the year and month columns. For week-level work use wm_yr_wk or week_start in sales_weekly. Use year = 2015 AND month = 12 for December 2015.

TOOLS GUIDANCE: use run_sql for facts and totals, decompose_change to explain why a metric changed between two equal-length periods, forecast_demand for the next 4 weeks, detect_anomalies to find unusual weeks, price_elasticity for price sensitivity, backtest_summary for forecast accuracy.
