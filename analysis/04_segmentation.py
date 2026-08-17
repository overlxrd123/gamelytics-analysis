"""模块 4：用户分群 — RFM 业务分群 + K-Means 行为聚类

回答的问题：
1. 用户按「最近活跃 R / 登录频次 F / 付费金额 M」能分成几类？每类什么画像？
2. 用聚类算法从行为特征自动发现用户群，和业务分群能否互相印证？

两种方法（都做，互为补充）：
- RFM 分群：业务规则，R/F/M 各按中位数分高/低 → 8 类，含义直观
- K-Means：数据驱动，标准化行为特征后聚类 → 自动发现用户群，再画像

产出：figures/04_segmentation.png
"""
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src import config, plots, load

plots.setup()

RFM_LABELS = {
    (1, 1, 1): "重要价值客户",
    (1, 1, 0): "活跃潜力客户",
    (1, 0, 1): "新付费客户",
    (1, 0, 0): "新用户/低活",
    (0, 1, 1): "待挽留付费客户",
    (0, 1, 0): "流失中的活跃用户",
    (0, 0, 1): "沉睡付费客户",
    (0, 0, 0): "完全流失",
}


def build_features(reg, auth, ab, cutoff):
    """构造用户级行为特征表（R/F/M + 活跃指标）"""
    login_stats = auth.groupby("uid").agg(
        login_cnt=("auth_ts", "count"),
        active_days=("auth_date", "nunique"),
        first_auth=("auth_dt", "min"),
        last_auth=("auth_dt", "max"),
    )
    df = ab.merge(login_stats, left_on="user_id", right_index=True, how="left")
    df = df.merge(reg[["uid", "reg_dt"]], left_on="user_id", right_on="uid", how="left")

    df["login_cnt"] = df["login_cnt"].fillna(0).astype(int)
    df["active_days"] = df["active_days"].fillna(0).astype(int)

    # 距最后登录天数（从未登录的用户用"注册到现在"作为 recency，代表一直没活跃）
    df["recency"] = (cutoff - df["last_auth"]).dt.days
    df["recency"] = df["recency"].fillna((cutoff - df["reg_dt"]).dt.days)
    # 既无登录记录、也无注册记录的用户（约10%），视为"从未活跃"，recency 取最大值
    df["recency"] = df["recency"].fillna(df["recency"].max())
    df["lifespan"] = (df["last_auth"] - df["first_auth"]).dt.days.fillna(0)
    df["log_monetary"] = np.log1p(df["revenue"])  # 长尾取对数，供聚类用
    return df


def rfm_segments(df):
    """RFM 业务分群：R/F 按中位数分高(1)/低(0)，M 按是否付费分"""
    r_high = (df["recency"] <= df["recency"].median()).astype(int)
    f_high = (df["login_cnt"] >= df["login_cnt"].median()).astype(int)
    m_high = (df["revenue"] > 0).astype(int)

    seg_key = list(zip(r_high, f_high, m_high))
    df = df.copy()
    df["rfm"] = [RFM_LABELS[k] for k in seg_key]

    prof = df.groupby("rfm").agg(
        人数=("user_id", "count"),
        付费率=("revenue", lambda s: (s > 0).mean()),
        人均收入=("revenue", "mean"),
        平均登录次数=("login_cnt", "mean"),
    )
    prof["人数占比"] = prof["人数"] / len(df)
    return df, prof


def kmeans_clusters(df, k=5):
    """K-Means 行为聚类 + 画像"""
    feats = ["recency", "login_cnt", "active_days", "lifespan", "log_monetary"]
    X = df[feats].values
    X = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    df = df.copy()
    df["cluster"] = labels

    prof = df.groupby("cluster").agg(
        人数=("user_id", "count"),
        付费率=("revenue", lambda s: (s > 0).mean()),
        人均收入=("revenue", "mean"),
        平均登录次数=("login_cnt", "mean"),
        平均活跃天数=("active_days", "mean"),
        平均距今未登录=("recency", "mean"),
    )
    return df, prof, X


def main():
    reg, auth, ab = load.load_all()
    cutoff = auth["auth_date"].max()

    print("=" * 60)
    print("模块 4：用户分群（RFM + K-Means）")
    print("=" * 60)

    df = build_features(reg, auth, ab, cutoff)

    # —— RFM 分群 ——
    df, rfm_prof = rfm_segments(df)
    print(f"\n[RFM 业务分群]（R=最近活跃，F=登录频次，M=是否付费）")
    print(rfm_prof.round(3).to_string())

    # —— K-Means 聚类 ——
    # 先看肘部，选 k
    feats = ["recency", "login_cnt", "active_days", "lifespan", "log_monetary"]
    X = StandardScaler().fit_transform(df[feats].values)
    inertias = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        inertias.append(km.inertia_)
    print(f"\n[K-Means 肘部] k=2..8 的 inertia: {[f'{v:,.0f}' for v in inertias]}")

    K = 5
    df, km_prof, _ = kmeans_clusters(df, k=K)
    print(f"\n[K-Means 聚类（k={K}）各簇画像]")
    print(km_prof.round(3).to_string())

    # ===== 图：K-Means 聚类散点（活跃天数 vs 登录次数，颜色=簇，大小=收入） =====
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [plots.palette(i) for i in range(K)]
    for c in range(K):
        sub = df[df["cluster"] == c]
        ax.scatter(sub["active_days"], sub["login_cnt"],
                   s=10 + np.log1p(sub["revenue"]) * 2,
                   c=[colors[c]], alpha=0.4, label=f"簇{c} ({len(sub):,})")
    ax.set_xlabel("活跃天数")
    ax.set_ylabel("登录次数")
    ax.set_title(f"K-Means 用户分群（k={K}，点大小=付费金额）")
    ax.legend()
    plots.savefig(fig, "04_segmentation.png")

    print(f"\n[图表] 已保存到 figures/")


if __name__ == "__main__":
    main()
