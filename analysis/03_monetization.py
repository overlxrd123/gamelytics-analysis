"""模块 3：付费变现 — 付费率/ARPU/ARPPU + 鲸鱼长尾 + A/B 测试

回答的问题：
1. 多少人付费？付费的玩家平均花多少？（付费率 / ARPU / ARPPU）
2. 收入是不是集中在极少数"鲸鱼"身上？（长尾 / 帕累托 / 大中小R分层）
3. 两种推广方案(a/b)谁更能拉动付费？（A/B 测试 + 显著性检验）
4. 付费玩家是不是比白嫖玩家更活跃？（付费 vs 活跃，连接模块2）

关键口径：
- 付费率 = revenue>0 的用户占比
- ARPU  = 总收入 / 全部用户（每用户平均收入）
- ARPPU = 总收入 / 付费用户（每付费用户平均收入）—— 鲸鱼经济下 ARPPU >> ARPU
"""
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src import config, plots, load

plots.setup()

# 大中小R 分界（贴合本数据：最小付费=200，中位数≈2052，鲸鱼≈3.7万）
SEGMENTS = [
    ("非付费", 0, 0),
    ("小额付费(≤1000)", 0, 1000),
    ("中额付费(1000~10000)", 1000, 10000),
    ("大额付费/鲸鱼(>10000)", 10000, np.inf),
]


def overview(ab):
    """付费率 / ARPU / ARPPU / 中位数"""
    total_users = len(ab)
    payers = ab[ab["revenue"] > 0]
    total_rev = ab["revenue"].sum()
    return {
        "总用户": total_users,
        "付费用户": len(payers),
        "付费率": len(payers) / total_users,
        "总收入": total_rev,
        "ARPU": total_rev / total_users,
        "ARPPU": total_rev / len(payers) if len(payers) else np.nan,
        "revenue中位数": ab["revenue"].median(),
    }


def segments(ab):
    """大中小R 分层：人数占比 vs 收入占比"""
    rows = []
    total_users = len(ab)
    total_rev = ab["revenue"].sum()
    for name, lo, hi in SEGMENTS:
        if name == "非付费":
            mask = ab["revenue"] == 0
        else:
            mask = (ab["revenue"] > lo) & (ab["revenue"] <= hi)
        sub = ab[mask]
        rows.append({
            "分层": name,
            "人数": len(sub),
            "人数占比": len(sub) / total_users,
            "收入": sub["revenue"].sum(),
            "收入占比": sub["revenue"].sum() / total_rev,
            "人均": sub["revenue"].mean(),
        })
    return pd.DataFrame(rows)


def concentration(ab):
    """帕累托：前 X% 的用户贡献了多少收入"""
    rev_sorted = ab["revenue"].sort_values(ascending=False)
    total = rev_sorted.sum()
    cum = rev_sorted.cumsum() / total
    n = len(rev_sorted)
    out = {}
    for pct in [1, 2, 5, 10]:
        k = max(1, int(n * pct / 100))
        out[f"前{pct}%用户"] = cum.iloc[k - 1]
    # 付费用户内部的集中度：前 1% 的「付费玩家」贡献多少
    payers = rev_sorted[rev_sorted > 0]
    if len(payers):
        pcum = payers.cumsum() / payers.sum()
        out["付费玩家里前1%贡献"] = pcum.iloc[max(0, int(len(payers) * 0.01) - 1)]
    return out


def ab_test(ab):
    """A/B 测试：付费率 z 检验 + ARPU bootstrap + Mann-Whitney"""
    g = ab.groupby("testgroup")["revenue"]
    a = ab[ab["testgroup"] == "a"]["revenue"].values
    b = ab[ab["testgroup"] == "b"]["revenue"].values

    # 付费率 z 检验
    na, nb = len(a), len(b)
    pa, pb = (a > 0).mean(), (b > 0).mean()
    p_pool = (pa * na + pb * nb) / (na + nb)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / na + 1 / nb))
    z = (pa - pb) / se
    z_pval = 2 * (1 - stats.norm.cdf(abs(z)))

    # ARPU bootstrap 置信区间
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(3000):
        sa = rng.choice(a, na, replace=True)
        sb = rng.choice(b, nb, replace=True)
        diffs.append(sa.mean() - sb.mean())
    ci = np.percentile(diffs, [2.5, 97.5])

    # Mann-Whitney U（非参，适合长尾）
    u, mw_pval = stats.mannwhitneyu(a, b, alternative="two-sided")

    return {
        "A付费率": pa, "B付费率": pb, "付费率z": z, "付费率p值": z_pval,
        "A_ARPU": a.mean(), "B_ARPU": b.mean(),
        "ARPU差(95%CI)": ci, "MW_U_p值": mw_pval,
    }


def payer_activity(ab, auth):
    """付费玩家 vs 非付费玩家的登录活跃度"""
    logins = auth.groupby("uid").size().rename("login_cnt")
    df = ab.merge(logins, left_on="user_id", right_index=True, how="left").fillna({"login_cnt": 0})
    df["payer"] = df["revenue"] > 0
    g = df.groupby("payer")["login_cnt"].agg(["mean", "median", "count"])
    g.index = ["非付费", "付费"]
    return g


def main():
    reg, auth, ab = load.load_all()

    print("=" * 60)
    print("模块 3：付费变现")
    print("=" * 60)

    # —— 概览 ——
    ov = overview(ab)
    print(f"\n[付费概览]")
    print(f"  总用户       : {ov['总用户']:,}")
    print(f"  付费用户     : {ov['付费用户']:,}")
    print(f"  付费率       : {ov['付费率']:.2%}")
    print(f"  总收入       : {ov['总收入']:,.0f}")
    print(f"  ARPU(人均)   : {ov['ARPU']:.2f}")
    print(f"  ARPPU(付费人均): {ov['ARPPU']:.2f}  ← 是 ARPU 的 {ov['ARPPU']/ov['ARPU']:.0f} 倍")
    print(f"  revenue中位数 : {ov['revenue中位数']:.0f}（中位数=0，说明绝大多数人没付费）")

    # —— 分层 ——
    seg = segments(ab)
    print(f"\n[大中小R 分层]")
    print(seg.round(4).to_string(index=False))
    whales = seg[seg["分层"].str.contains("鲸鱼")]
    print(f"  → 结论：{whales['人数占比'].iloc[0]:.2%} 的鲸鱼，贡献了 {whales['收入占比'].iloc[0]:.2%} 的收入")

    # —— 集中度 ——
    conc = concentration(ab)
    print(f"\n[收入集中度(帕累托)]")
    for k, v in conc.items():
        print(f"  {k:<16} 贡献 {v:.2%} 的收入")

    # —— A/B 测试 ——
    res = ab_test(ab)
    print(f"\n[A/B 测试 a vs b]")
    print(f"  付费率: A={res['A付费率']:.4%}  B={res['B付费率']:.4%}  (z={res['付费率z']:.1f}, p={res['付费率p值']:.4f})")
    print(f"  ARPU : A={res['A_ARPU']:.4f}  B={res['B_ARPU']:.4f}  差={res['A_ARPU']-res['B_ARPU']:+.4f}")
    print(f"  ARPU差 95%CI: [{res['ARPU差(95%CI)'][0]:.4f}, {res['ARPU差(95%CI)'][1]:.4f}]  (Mann-Whitney p={res['MW_U_p值']:.4f})")

    # —— 付费 vs 活跃 ——
    act = payer_activity(ab, auth)
    print(f"\n[付费 vs 活跃度]")
    print(act.round(1).to_string())

    # ===== 图1：长尾 + 分层 =====
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # 左：付费金额分布（付费玩家，log x）
    payers = ab[ab["revenue"] > 0]["revenue"]
    axes[0].hist(np.log10(payers), bins=40, color=plots.palette(0), alpha=0.8)
    axes[0].set_title("付费玩家金额分布（对数轴）")
    axes[0].set_xlabel("log10(付费金额)")
    axes[0].set_ylabel("人数")
    # 右：分层收入占比
    labels = seg["分层"]
    rev_share = seg["收入占比"].values
    cnt_share = seg["人数占比"].values
    x = np.arange(len(labels))
    axes[1].bar(x - 0.2, cnt_share, 0.38, label="人数占比", color=plots.palette(1))
    axes[1].bar(x + 0.2, rev_share, 0.38, label="收入占比", color=plots.palette(3))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("各层「人数占比 vs 收入占比」")
    axes[1].legend()
    plots.savefig(fig, "03_revenue_longtail.png")

    # ===== 图2：A/B 测试 =====
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["A组", "B组"], [res["A付费率"], res["B付费率"]], color=[plots.palette(0), plots.palette(1)])
    axes[0].set_title("付费率对比")
    axes[0].set_ylabel("付费率")
    axes[1].bar(["A组", "B组"], [res["A_ARPU"], res["B_ARPU"]], color=[plots.palette(0), plots.palette(1)])
    axes[1].set_title("ARPU 对比")
    axes[1].set_ylabel("人均收入")
    plots.savefig(fig, "03_ab_test.png")

    print(f"\n[图表] 已保存到 figures/")


if __name__ == "__main__":
    main()
