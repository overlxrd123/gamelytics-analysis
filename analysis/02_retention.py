"""模块 2：留存分析 — 两种口径 + Cohort 热力图

回答的问题：
1. 用户注册后能留下来多久？流失最严重的是第几天？
2. 不同时期注册的用户（cohort），留存是变好还是变差？

两种留存口径（都算，都讲得清）：
- 严格口径 D_N：注册后第 N 天「当天」仍有登录（AppsFlyer/Adjust 同款）
- 累计口径  C_N：注册后 N 天内「至少回来过一次」（单调递增，更直观）

右删失处理：数据截至 2020-09-23，最近注册的用户还没满 N 天，
  不能放进 D_N/C_N 的分母，否则会人为拉低留存。

产出：figures/02_retention_curve.png + 02_cohort_heatmap.png
"""
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config, plots, load

plots.setup()

RETENTION_DAYS = list(range(1, 31))          # 留存曲线：第 1~30 天
KEY_DAYS = [1, 3, 7, 14, 30]                 # 热力图/报告关注的关键节点


def build_user_day(auth, reg):
    """登录日志 join 注册日期，算每条登录距注册第几天，并去重成 (uid, day)"""
    merged = auth[["uid", "auth_date"]].merge(
        reg[["uid", "reg_date"]], on="uid", how="inner"
    )
    merged["day"] = (merged["auth_date"] - merged["reg_date"]).dt.days
    merged = merged[merged["day"] >= 0]
    pairs = merged[["uid", "day"]].drop_duplicates()
    return merged, pairs


def strict_retention(reg, pairs, cutoff):
    """严格口径：注册后第 N 天「当天」登录"""
    curve = {}
    for N in RETENTION_DAYS:
        den = (reg["reg_date"] <= cutoff - pd.Timedelta(days=N)).sum()
        num = pairs[pairs["day"] == N]["uid"].nunique()
        curve[N] = num / den if den else np.nan
    return pd.Series(curve, name="strict")


def cumulative_retention(reg, merged, cutoff):
    """累计口径：注册后 N 天内至少回访一次（单调递增）"""
    first_ret = merged[merged["day"] >= 1].groupby("uid")["day"].min().rename("first_day")
    fr = reg[["uid", "reg_date"]].merge(first_ret, on="uid", how="left")  # 未回访=NaN
    curve = {}
    for N in RETENTION_DAYS:
        eligible = fr["reg_date"] <= cutoff - pd.Timedelta(days=N)
        den = eligible.sum()
        num = (eligible & (fr["first_day"] <= N)).sum()
        curve[N] = num / den if den else np.nan
    return pd.Series(curve, name="cumulative")


def cohort_retention(reg, pairs, cutoff, cohorts):
    """分月度 cohort 的严格留存矩阵（行列都可能因右删失出现 NaN）"""
    rows = []
    for c in cohorts:
        c_users = reg[reg["reg_date"].dt.to_period("M") == c]
        total = c_users["uid"].nunique()
        row = {"cohort": str(c), "总用户": total}
        for N in KEY_DAYS:
            den = c_users[c_users["reg_date"] <= cutoff - pd.Timedelta(days=N)]["uid"].nunique()
            if den == 0:
                row[f"D{N}"] = np.nan
            else:
                num = pairs[(pairs["day"] == N) & (pairs["uid"].isin(c_users["uid"]))]["uid"].nunique()
                row[f"D{N}"] = num / den
        rows.append(row)
    return pd.DataFrame(rows).set_index("cohort")


def main():
    reg, auth, _ = load.load_all()
    cutoff = auth["auth_date"].max()

    print("=" * 60)
    print("模块 2：留存分析（严格 + 累计两种口径）")
    print("=" * 60)

    merged, pairs = build_user_day(auth, reg)

    # —— 两种留存曲线 ——
    strict = strict_retention(reg, pairs, cutoff)
    cumul = cumulative_retention(reg, merged, cutoff)

    print(f"\n[留存对比]（已处理右删失）")
    print(f"  {'节点':<6} {'严格(当天登录)':<16} {'累计(N天内回来)':<16}")
    for N in KEY_DAYS:
        print(f"  D{N:<5} {strict[N]:>14.2%}   {cumul[N]:>14.2%}")

    # —— Cohort 热力图 ——
    all_cohorts = sorted(reg["reg_date"].dt.to_period("M").unique())
    cohorts = all_cohorts[-12:]
    cmat = cohort_retention(reg, pairs, cutoff, cohorts)
    print(f"\n[Cohort 留存矩阵] 近 {len(cohorts)} 个注册月（D1~D30，NaN=右删失）")
    print(cmat.round(3).to_string())

    cmat_d7 = cmat["D7"].dropna()
    print(f"\n[趋势] D7 留存：最早 {cmat_d7.iloc[0]:.2%} → 最新 {cmat_d7.iloc[-1]:.2%}（基本持平=留存质量没提升）")

    # ===== 图1：两种留存曲线并排 =====
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    strict.plot(ax=axes[0], marker="o", ms=3, lw=1.6, color=plots.palette(0))
    axes[0].set_title("严格口径：第 N 天当天登录")
    axes[0].set_xlabel("注册后第 N 天")
    axes[0].set_ylabel("留存率")
    axes[0].grid(alpha=0.3)

    cumul.plot(ax=axes[1], marker="o", ms=3, lw=1.6, color=plots.palette(1))
    axes[1].set_title("累计口径：N 天内至少回来一次")
    axes[1].set_xlabel("注册后第 N 天")
    axes[1].set_ylabel("累计留存率")
    axes[1].grid(alpha=0.3)
    for N in KEY_DAYS:
        axes[1].annotate(f"D{N}: {cumul[N]:.1%}", (N, cumul[N]),
                         textcoords="offset points", xytext=(6, 6), fontsize=8)
    plots.savefig(fig, "02_retention_curve.png")

    # ===== 图2：Cohort 热力图 =====
    data = cmat[[f"D{N}" for N in KEY_DAYS]].astype(float).values
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0, vmax=np.nanmax(data))
    ax.set_xticks(range(len(KEY_DAYS)))
    ax.set_xticklabels([f"D{N}" for N in KEY_DAYS])
    ax.set_yticks(range(len(cmat)))
    ax.set_yticklabels(cmat.index)
    ax.set_title("Cohort 留存热力图（按月注册分组）")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.0%}" if not np.isnan(v) else "·",
                    ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="留存率")
    plots.savefig(fig, "02_cohort_heatmap.png")

    print(f"\n[图表] 已保存到 figures/")


if __name__ == "__main__":
    main()
