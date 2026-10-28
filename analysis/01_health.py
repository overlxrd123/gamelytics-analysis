"""P1 activity: complete UTC dates, true adjacent 90-day periods, additive growth decomposition."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import matplotlib.pyplot as plt
from src import load, plots, metrics
plots.setup()


def activity_metrics(reg, auth, end):
    start = end-pd.Timedelta(days=365)
    dates = pd.date_range(start, end-pd.Timedelta(days=1))
    # Seed rolling MAU with preceding history; zero-login dates remain on calendar.
    history = auth.loc[auth.auth_date >= start-pd.Timedelta(days=29)]
    sets = history.groupby("auth_date").uid.agg(set)
    rows = []
    for day in dates:
        ids = set().union(*(sets.get(day-pd.Timedelta(days=k), set()) for k in range(30)))
        dau = len(sets.get(day, set()))
        rows.append(dict(date=day, dau=dau, mau30=len(ids), stickiness=dau/len(ids) if ids else float("nan")))
    return pd.DataFrame(rows)


def main():
    reg, auth, _ = load.load_all()
    reg, auth, end = metrics.snapshot(reg, auth)
    activity = activity_metrics(reg, auth, end)
    metrics.save_report(activity, "health_daily.csv")
    pairs = auth[["uid", "auth_date"]].drop_duplicates().merge(reg[["uid", "reg_date"]], on="uid", validate="many_to_one")
    # Mutually exclusive tenure buckets, identity: new + older = all DAU.
    pairs["new_0_29d"] = (pairs.auth_date-pairs.reg_date).dt.days < 30
    rows = []
    for name, lo, hi in [("previous90", end-pd.Timedelta(days=180), end-pd.Timedelta(days=90)),
                         ("latest90", end-pd.Timedelta(days=90), end)]:
        part = pairs.loc[pairs.auth_date.ge(lo) & pairs.auth_date.lt(hi)]
        rows.append(dict(period=name, start=str(lo.date()), end_exclusive=str(hi.date()),
                         dau= len(part)/90, new_dau=part.new_0_29d.sum()/90,
                         older_dau=(~part.new_0_29d).sum()/90))
    growth = pd.DataFrame(rows)
    metrics.save_report(growth, "growth_decomposition.csv")
    print(growth.to_string(index=False))
    print("Descriptive additive decomposition; not acquisition attribution or causal impact.")
    fig, ax = plt.subplots(figsize=(10,4))
    activity.set_index("date")[["dau", "mau30"]].plot(ax=ax)
    ax.set_title("P1：完整日期日活与滚动30日活跃")
    plots.savefig(fig,"01_activity_trend.png")
    fig, ax = plt.subplots(figsize=(10,4))
    activity.set_index("date").stickiness.plot(ax=ax)
    ax.set_title("P1：DAU / 滚动30日MAU（已补窗口历史）")
    plots.savefig(fig,"01_stickiness.png")
    fig, ax = plt.subplots(figsize=(9,4))
    reg.groupby("reg_date").size().cumsum().plot(ax=ax)
    ax.set_title("P1：累计注册用户（完整日期）")
    plots.savefig(fig,"01_registration_growth.png")


if __name__ == "__main__":
    main()
