"""
STEP 3: Rolling-origin backtest.

For each of the last N blocks of 4 weeks ("origins"):
    - train ONLY on data before the block,
    - forecast the 4 weeks of the block,
    - compare with what really happened.
Random train/test splits are never used (they leak the future in time series).

Metric: WAPE = sum|actual - forecast| / sum|actual|   (lower is better)
Skill vs seasonal-naive = 1 - WAPE_model / WAPE_snaive  (positive = better than baseline)
"""
import argparse
import time

import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, HORIZON, REPORTS_DIR
from src.forecasting.features import FEATURES, load_weekly_panel, make_features
from src.forecasting.models import BASELINES, fit_lgbm_bundle, predict_bundle

DEFAULT_MODELS = "naive,snaive,mavg4,theta,lgbm"


def wape(actual, pred) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    denom = np.abs(actual).sum()
    return float(np.abs(actual - pred).sum() / denom) if denom else float("nan")


def run_backtest(panel: pd.DataFrame, n_origins: int, models: list) -> pd.DataFrame:
    feats = make_features(panel)
    weeks = np.sort(panel["week_start"].unique())
    test_weeks = weeks[-n_origins * HORIZON:]
    origins = [pd.Timestamp(x) for x in test_weeks[::HORIZON]]
    frames = []

    for o in origins:
        t0 = time.time()
        end = o + pd.Timedelta(weeks=HORIZON)
        hist = panel[panel["week_start"] < o]
        test = feats[(feats["week_start"] >= o) & (feats["week_start"] < end)]
        test = test.sort_values(["series_id", "week_start"])
        hist_by = {sid: g["units"].to_numpy() for sid, g in hist.groupby("series_id")}
        base = test[["series_id", "cat_id", "state_id", "week_start", "units"]].rename(
            columns={"units": "actual"}).assign(origin=o)

        for name in models:
            if name == "lgbm":
                continue
            fn = BASELINES[name]
            preds = [fn(hist_by[sid], len(g)) for sid, g in test.groupby("series_id", sort=False)]
            frames.append(base.assign(model=name, pred=np.concatenate(preds), lo=np.nan, hi=np.nan))

        if "lgbm" in models:
            train = feats[feats["week_start"] < o].dropna(subset=[f"lag_{HORIZON}"])
            bundle = fit_lgbm_bundle(train)
            p = predict_bundle(bundle, test[FEATURES])
            frames.append(base.assign(model="lgbm", pred=p["mid"], lo=p["lo"], hi=p["hi"]))
        print(f"  origin {o.date()} done in {time.time() - t0:.0f}s")

    return pd.concat(frames, ignore_index=True)


def metrics_table(res: pd.DataFrame) -> pd.DataFrame:
    cats = sorted(res["cat_id"].unique())
    rows = []
    for m, d in res.groupby("model"):
        rows.append(dict(model=m, level="all_series", wape=wape(d["actual"], d["pred"])))
        for c in cats:
            s = d[d["cat_id"] == c]
            rows.append(dict(model=m, level=f"cat:{c}", wape=wape(s["actual"], s["pred"])))
        tot = d.groupby("week_start")[["actual", "pred"]].sum()
        rows.append(dict(model=m, level="total", wape=wape(tot["actual"], tot["pred"])))
    mt = pd.DataFrame(rows)

    if "snaive" in set(mt["model"]):
        base = mt[mt["model"] == "snaive"].set_index("level")["wape"]
        mt["skill_vs_snaive"] = mt.apply(lambda r: 1 - r["wape"] / base[r["level"]], axis=1)
    else:
        mt["skill_vs_snaive"] = np.nan

    mt["coverage_80"] = np.nan
    if "lgbm" in set(res["model"]):
        d = res[res["model"] == "lgbm"]
        cov = float(((d["actual"] >= d["lo"]) & (d["actual"] <= d["hi"])).mean())
        mt.loc[(mt["model"] == "lgbm") & (mt["level"] == "all_series"), "coverage_80"] = cov
    return mt.round(4)


def make_figures(res: pd.DataFrame, mt: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    piv = mt[mt["level"].isin(["all_series", "total"])].pivot(index="model", columns="level", values="wape")
    ax = piv.sort_values("all_series").plot(kind="bar", figsize=(8, 4))
    ax.set_ylabel("WAPE (lower is better)")
    ax.set_title("Backtest accuracy by model")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "backtest_wape.png", dpi=150)
    plt.close()

    if "lgbm" in set(res["model"]):
        d = res[res["model"] == "lgbm"].groupby("week_start")[["actual", "pred", "lo", "hi"]].sum()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(d.index, d["actual"], label="actual", color="black")
        ax.plot(d.index, d["pred"], label="LightGBM forecast", color="tab:blue")
        ax.fill_between(d.index, d["lo"], d["hi"], alpha=0.2, label="80% interval (sum of series bounds, conservative)")
        ax.set_title("Total weekly units: forecast vs actual (backtest weeks)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "backtest_forecast_vs_actual.png", dpi=150)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-origins", type=int, default=5, help="number of 4-week test blocks")
    ap.add_argument("--models", default=DEFAULT_MODELS)
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    from src.db import get_cursor
    panel = load_weekly_panel(get_cursor())
    print(f"Panel: {panel['series_id'].nunique()} series, "
          f"{panel['week_start'].nunique()} weeks "
          f"({panel['week_start'].min().date()} -> {panel['week_start'].max().date()})")
    print(f"Backtesting {args.n_origins} origins x {HORIZON} weeks, models: {models}")

    res = run_backtest(panel, args.n_origins, models)
    mt = metrics_table(res)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(REPORTS_DIR / "forecast_backtest.csv", index=False)
    mt.to_csv(REPORTS_DIR / "forecast_metrics.csv", index=False)
    make_figures(res, mt)

    print("\nWAPE by model (lower is better):")
    show = mt[mt["level"].isin(["all_series", "total"])].pivot(
        index="model", columns="level", values="wape").sort_values("all_series")
    print(show.to_string())
    if "snaive" in set(mt["model"]):
        print("\nSkill vs seasonal-naive (positive = beats the baseline):")
        sk = mt[mt["level"].isin(["all_series", "total"])].pivot(
            index="model", columns="level", values="skill_vs_snaive").sort_values("all_series", ascending=False)
        print(sk.to_string())
    cov = mt["coverage_80"].dropna()
    if len(cov):
        print(f"\nLightGBM 80% interval coverage: {cov.iloc[0]:.1%} (ideal is about 80%)")
    print(f"\nSaved: reports/forecast_metrics.csv, forecast_backtest.csv, figures/")


if __name__ == "__main__":
    main()
