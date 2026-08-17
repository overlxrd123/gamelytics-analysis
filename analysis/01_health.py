"""模块 1：运营健康度 — 注册增长 + DAU/WAU/MAU + 粘性

回答的问题：
1. 这个游戏整体盘子多大、还在不在增长？
2. 近期（近 12 个月）日活/周活/月活是什么水平、趋势如何？
3. 用户粘性（DAU/MAU）怎么样，是健康还是衰减？

产出：figures/01_*.png 三张图 + 控制台关键指标
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

RECENT_DAYS = 365  # 近 12 个月作为"近期"观察窗口


def registration_growth(reg: pd.DataFrame) -> pd.DataFrame:
    """每日注册 + 累计注册（全时段，展示指数增长）"""
    daily = reg.groupby("reg_date").size().rename("daily_reg")
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()), fill_value=0)
    cum = daily.cumsum().rename("cum_reg")
    return pd.DataFrame({"daily_reg": daily, "cum_reg": cum})


def activity_metrics(auth: pd.DataFrame, cutoff: pd.Timestamp):
    """近期窗口内的 DAU(日) / WAU(周) / MAU(月) / 粘性(DAU/滚动30日MAU)"""
    win = auth[auth.auth_date >= cutoff]
    # 每日活跃用户集合
    daily_sets = win.groupby("auth_date").uid.agg(set)
    dates = sorted(daily_sets.index)

    dau = pd.Series({d: len(s) for d, s in daily_sets.items()}, name="DAU")

    # 滚动 30 日 MAU（滑动窗口求并集）
    mau = {}
    buf = []
    for d in dates:
        buf.append(daily_sets[d])
        if len(buf) > 30:
            buf.pop(0)
        mau[d] = len(set().union(*buf)) if buf else 0
    mau = pd.Series(mau, name="MAU30")

    # 周活 / 月活（日历口径）
    wau = win.set_index("auth_date").resample("W").uid.nunique().rename("WAU")
    mau_m = win.set_index("auth_date").resample("ME").uid.nunique().rename("MAU_month")

    stickiness = (dau / mau).rename("stickiness")
    return dau, wau, mau_m, mau, stickiness


def main():
    reg, auth, _ = load.load_all()
    cutoff = auth.auth_date.max() - pd.Timedelta(days=RECENT_DAYS)

    print("=" * 60)
    print("模块 1：运营健康度")
    print("=" * 60)

    # —— 注册增长 ——
    growth = registration_growth(reg)
    total_reg = int(growth.cum_reg.iloc[-1])
    print(f"\n[注册] 累计注册用户 {total_reg:,}，时间跨度 {growth.index.min().date()} ~ {growth.index.max().date()}")

    # —— 近期活跃 ——
    dau, wau, mau_m, mau, stick = activity_metrics(auth, cutoff)
    print(f"\n[活跃] 近 {RECENT_DAYS} 天（{cutoff.date()} ~ {auth.auth_date.max().date()}）")
    print(f"  日均 DAU : {dau.mean():,.0f}")
    print(f"  峰值 DAU : {dau.max():,}")
    print(f"  平均周活 : {wau.mean():,.0f}")
    print(f"  平均月活 : {mau_m.mean():,.0f}")
    print(f"  平均粘性 DAU/MAU : {stick.mean():.1%}")

    # 增长率：近 90 天 vs 前 90 天
    dau_early = dau.iloc[:90].mean()
    dau_late = dau.iloc[-90:].mean()
    print(f"\n[趋势] DAU 前90天均值 {dau_early:,.0f} → 后90天均值 {dau_late:,.0f}，增长 {dau_late/dau_early-1:+.1%}")

    # ===== 图1：注册增长（累计，对数轴展示指数增长） =====
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    growth["cum_reg"].plot(ax=ax[0], color=plots.palette(0), lw=1.6)
    ax[0].set_yscale("log")
    ax[0].set_title("累计注册（对数轴）— 指数增长")
    ax[0].set_ylabel("累计注册用户")
    growth["daily_reg"].rolling(30).mean().plot(ax=ax[1], color=plots.palette(1), lw=1.6)
    ax[1].set_title("每日注册（30 日滑动平均）")
    ax[1].set_ylabel("新增用户/日")
    plots.savefig(fig, "01_registration_growth.png")

    # ===== 图2：DAU/WAU/MAU 趋势 =====
    fig, ax = plt.subplots(figsize=(12, 4.2))
    dau.plot(ax=ax, lw=0.8, color=plots.palette(0), label="DAU(日活)", alpha=0.7)
    wau.plot(ax=ax, lw=1.6, color=plots.palette(1), label="WAU(周活)")
    mau_m.plot(ax=ax, lw=1.6, color=plots.palette(2), label="MAU(月活)")
    ax.set_title(f"近 {RECENT_DAYS} 天活跃趋势（DAU/WAU/MAU）")
    ax.legend()
    plots.savefig(fig, "01_activity_trend.png")

    # ===== 图3：粘性 =====
    fig, ax = plt.subplots(figsize=(12, 3.6))
    stick.plot(ax=ax, lw=1.4, color=plots.palette(3))
    ax.axhline(stick.mean(), ls="--", color="gray", lw=1, label=f"均值 {stick.mean():.1%}")
    ax.set_title("用户粘性 DAU / 滚动30日MAU")
    ax.set_ylim(0, 1)
    ax.legend()
    plots.savefig(fig, "01_stickiness.png")

    print(f"\n[图表] 已保存到 figures/ 目录")


if __name__ == "__main__":
    main()
