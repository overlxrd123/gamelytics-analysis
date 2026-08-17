"""模块 6：结论建议 — 把前五个模块串成一份「诊断结论 + 行动建议」

回答的问题：
1. 这个游戏到底出了什么问题？（用 5 个关键数字一图看懂）
2. 问题之间是什么因果链？
3. 按优先级，业务上应该先做什么、再做什么？

产出：figures/06_dashboard.png（一页诊断仪表盘）
说明：这里只做「轻量重算」几个头号指标用来画图，完整结论见 README.md。
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


def headline(reg, auth, ab):
    """轻量重算 5 个头号指标（都是快速 groupby，不跑模型）"""
    cutoff = auth["auth_date"].max()

    # 1. 严格留存 D1/D7/D30
    merged = auth[["uid", "auth_date"]].merge(reg[["uid", "reg_date"]], on="uid", how="inner")
    merged["day"] = (merged["auth_date"] - merged["reg_date"]).dt.days
    pairs = merged[merged["day"] >= 0][["uid", "day"]].drop_duplicates()
    ret = {}
    for N in [1, 7, 30]:
        den = (reg["reg_date"] <= cutoff - pd.Timedelta(days=N)).sum()
        ret[N] = pairs[pairs["day"] == N]["uid"].nunique() / den

    # 2. 付费率 + 鲸鱼集中度
    total_rev = ab["revenue"].sum()
    pay_rate = (ab["revenue"] > 0).mean()
    whales = ab[ab["revenue"] > 10000]
    whale_cnt = len(whales) / len(ab)
    whale_rev = whales["revenue"].sum() / total_rev
    arpu = total_rev / len(ab)
    arppu = total_rev / (ab["revenue"] > 0).sum()

    # 3. 用户构成：活跃（近30天登录） vs 流失
    last = auth.groupby("uid")["auth_date"].max()
    recency = (cutoff - last).dt.days
    active = (recency <= 30).sum()
    lapsed = (recency > 30).sum()

    # 4. 流失驱动：近14天「距上次登录天数」在流失/留存两组间的对比
    obs = merged[(merged["day"] >= 0) & (merged["day"] < 14)]
    last_obs = obs.groupby("uid")["day"].max().rename("last_day")
    returned = set(merged[(merged["day"] >= 14) & (merged["day"] < 28)]["uid"].unique())
    feat = reg[["uid"]].merge(last_obs, on="uid", how="left")
    feat["last_day"] = feat["last_day"].fillna(-1)
    feat["days_since"] = np.where(feat["last_day"] >= 0, 13 - feat["last_day"], 14)
    feat["churn"] = (~feat["uid"].isin(returned)).astype(int)
    g = feat.groupby("churn")["days_since"].mean()

    return {
        "D1": ret[1], "D7": ret[7], "D30": ret[30],
        "pay_rate": pay_rate, "whale_cnt": whale_cnt, "whale_rev": whale_rev,
        "arpu": arpu, "arppu": arppu,
        "active": active, "lapsed": lapsed, "total": len(reg),
        "retain_days_since": g[0], "churn_days_since": g[1],
    }


def main():
    reg, auth, ab = load.load_all()
    h = headline(reg, auth, ab)

    print("=" * 60)
    print("模块 6：结论建议（一页诊断）")
    print("=" * 60)

    print(f"\n[核心数字]")
    print(f"  严格留存     : D1={h['D1']:.1%}  D7={h['D7']:.1%}  D30={h['D30']:.1%}  ← 第1天断崖")
    print(f"  付费率       : {h['pay_rate']:.2%}  ARPU={h['arpu']:.2f}  ARPPU={h['arppu']:.2f}")
    print(f"  鲸鱼集中度   : 前 {h['whale_cnt']:.2%} 的用户贡献 {h['whale_rev']:.1%} 的收入")
    print(f"  用户构成     : 活跃(30天内) {h['active']/h['total']:.1%}  |  流失 {h['lapsed']/h['total']:.1%}")
    print(f"  流失驱动     : 留存组距上次登录 {h['retain_days_since']:.1f} 天 vs 流失组 {h['churn_days_since']:.1f} 天")

    print(f"\n[因果链] 拉新驱动增长 → 留存质量不涨 → 付费极低且集中 → 付费用户整体沉睡 → 90%盘面已流失")
    print(f"  → 近期性(多久没登录)是流失的第一信号，7 天未登录就该干预。")

    print(f"\n[行动建议（按优先级）]")
    print(f"  P0 止血 : ①召回沉睡付费用户(鲸鱼0.03%贡献43.5%收入，付费用户中位已沉睡2.7年) ②上线7天未登录实时预警")
    print(f"  P1 留存 : ③修D1留存(首日体验/次日回访奖励) ④破冰付费(3.6%高频白嫖是转化金矿)")
    print(f"  P2 提质 : ⑤提升留存质量(cohort D7五年持平=产品力没提升) ⑥A/B扩样本再验证ARPU")

    # ===== 一页诊断仪表盘（2x2） =====
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("游戏运营诊断结论（Gamelytics 数据分析）", fontsize=15, fontweight="bold")

    # 左上：留存断崖
    ax = axes[0][0]
    days = ["D1", "D7", "D30"]
    vals = [h["D1"], h["D7"], h["D30"]]
    bars = ax.bar(days, [v * 100 for v in vals], color=plots.palette(0), alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{v:.1%}", ha="center", fontsize=11)
    ax.set_title("严格留存：第1天断崖（76%注册即流失）")
    ax.set_ylabel("留存率")
    ax.set_ylim(0, 8)

    # 右上：鲸鱼集中度
    ax = axes[0][1]
    labels = ["人数占比", "收入占比"]
    whale = [h["whale_cnt"] * 100, h["whale_rev"] * 100]
    ax.bar(labels, whale, color=[plots.palette(1), plots.palette(3)], alpha=0.85)
    for b, v in zip(ax.patches, whale):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v:.1f}%", ha="center", fontsize=11)
    ax.set_title("鲸鱼经济：极少数人撑起收入")
    ax.set_ylabel("占比 %")

    # 左下：用户构成
    ax = axes[1][0]
    sizes = [h["active"], h["lapsed"]]
    ax.pie(sizes, labels=["活跃(30天内)", "流失"], autopct="%.1f%%",
           colors=[plots.palette(2), "#d9d9d9"], startangle=90)
    ax.set_title("用户构成：近30天 90% 已流失（高频核心仅4.5%）")

    # 右下：流失驱动
    ax = axes[1][1]
    ax.bar(["留存用户", "流失用户"], [h["retain_days_since"], h["churn_days_since"]],
           color=[plots.palette(2), plots.palette(3)], alpha=0.85)
    ax.set_title("流失第一信号：距上次登录天数")
    ax.set_ylabel("平均距上次登录天数")
    ax.annotate("近期性是第一预测因子", xy=(0.5, 0.55), xycoords="axes fraction",
                ha="center", fontsize=10, style="italic", color="gray")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plots.savefig(fig, "06_dashboard.png")
    print(f"\n[图表] 已保存到 figures/06_dashboard.png")


if __name__ == "__main__":
    main()
