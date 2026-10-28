"""Shared calendar-day definitions. Snapshot is exclusive; last partial date is excluded."""
import numpy as np
import pandas as pd
from . import config


def snapshot(reg, auth):
    end = auth.auth_date.max()  # start of last observed (possibly partial) UTC date
    return reg.loc[reg.reg_date < end].copy(), auth.loc[auth.auth_date < end].copy(), end


def user_days(reg, auth):
    x = auth[["uid", "auth_date"]].merge(reg[["uid", "reg_date"]], on="uid", validate="many_to_one")
    x["day"] = (x.auth_date - x.reg_date).dt.days
    return x.loc[x.day >= 0, ["uid", "auth_date", "day"]].drop_duplicates()


def retention(reg, pairs, end, days=(1, 7, 30), by_month=False):
    # Filter by horizon first instead of repeatedly scanning all events per cohort.
    rows = []
    users = reg[["uid", "reg_date"]].copy()
    users["cohort"] = users.reg_date.dt.strftime("%Y-%m") if by_month else "all"
    cohorts = sorted(users.cohort.unique())
    for n in days:
        eligible = users.loc[users.reg_date + pd.Timedelta(days=n) < end]
        counts = eligible.groupby("cohort").size()
        hits = pairs.loc[pairs.day.eq(n), ["uid"]].drop_duplicates().merge(
            eligible[["uid", "cohort"]], on="uid", validate="one_to_one")
        kept = hits.groupby("cohort").size()
        for cohort in cohorts:
            den = int(counts.get(cohort, 0));num = int(kept.get(cohort, 0))
            rows.append(dict(cohort=cohort, day=n, eligible=den, retained=num,
                             rate=num/den if den else np.nan))
    return pd.DataFrame(rows)


def candidate_wide(reg, auth, ab, end):
    """Keep unknown behavior null. Numeric ID equality is an unverified linkage."""
    recent = auth.loc[auth.auth_date >= end-pd.Timedelta(days=30)]
    last = auth.groupby("uid").auth_date.max().rename("last_date")
    frequency = recent.groupby("uid").auth_date.nunique().rename("recent_active_days")
    full = auth.groupby("uid").size().rename("login_cnt")
    x = ab.merge(reg[["uid", "reg_date"]], left_on="user_id", right_on="uid", how="left", validate="one_to_one")
    x["matched"] = x.reg_date.notna()
    x = x.join(last, on="user_id").join(frequency, on="user_id").join(full, on="user_id")
    # Only linked registered users have observable zero events in this window.
    x.loc[x.matched, "recent_active_days"] = x.loc[x.matched, "recent_active_days"].fillna(0)
    x.loc[x.matched, "login_cnt"] = x.loc[x.matched, "login_cnt"].fillna(0)
    x.loc[~x.matched, "last_date"] = pd.NaT
    x.loc[~x.matched, ["recent_active_days", "login_cnt"]] = np.nan
    x["recency"] = (end-x.last_date).dt.days
    x["tenure_days"] = (end-x.reg_date).dt.days
    x["reg_month"] = x.reg_date.dt.strftime("%Y-%m")
    return x


def churn_dataset(reg, auth, end):
    obs, label = config.OBS_WINDOW_DAYS, config.CHURN_GAP_DAYS
    pairs = user_days(reg, auth)
    eligible = reg.loc[reg.reg_date + pd.Timedelta(days=obs+label) <= end, ["uid", "reg_date"]]
    x = pairs.loc[pairs.day < obs].sort_values(["uid", "day"])
    feat = x.groupby("uid").agg(login_cnt=("day", "size"), last_login_day=("day", "max"))
    gaps = x.groupby("uid").day.diff()
    feat["max_gap"] = gaps.groupby(x.uid).max().fillna(0)
    result = eligible.merge(feat, on="uid", how="left", validate="one_to_one")
    result[["login_cnt", "max_gap"]] = result[["login_cnt", "max_gap"]].fillna(0)
    result["days_since_last_login"] = obs-result.last_login_day.fillna(-1)
    returned = pairs.loc[pairs.day.ge(obs) & pairs.day.lt(obs+label), "uid"]
    result["churn"] = (~result.uid.isin(returned)).astype(int)
    result["prediction_date"] = result.reg_date+pd.Timedelta(days=obs)
    result["label_end"] = result.reg_date+pd.Timedelta(days=obs+label)
    return result


def save_report(df, name):
    config.REPORT_DIR.mkdir(exist_ok=True, parents=True)
    df.to_csv(config.REPORT_DIR/name, index=False, encoding="utf-8-sig")
