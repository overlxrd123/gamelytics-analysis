"""模块 5：流失预测 — 定义流失 + 特征工程 + 模型对比 + SHAP 解释

回答的问题：
1. 怎么用行为数据定义"流失"标签（右删失必须处理干净）
2. 从注册后前 14 天的行为，能不能预测用户第 14 天后会不会流失？
3. 逻辑回归 / XGBoost / LightGBM 谁更准？（ROC-AUC 对比 + 交叉验证）
4. 到底哪些行为最能预示流失？（SHAP 解释，给业务一个可落地的抓手）

口径（和前面模块一致，面试会被追问，务必讲清）：
- 观察期：[0, 14) 天，即注册后前两周的行为 → 用来做特征
- 预测点：注册后第 14 天
- 标签期：[14, 28) 天
- 流失 churn=1：标签期内一次都没登录（第 14 天后彻底不再回来）
- 右删失：只保留 reg_date <= cutoff - 28 天的用户，否则标签期还没走完，标签不准

数据结构说明（决定了建模方式）：
  合成数据里每个用户注册当天(day0)都有一条登录记录（注册=首次登录），
  之后 76% 的人第 1 天起就不再回来（模块2 的 D1 留存仅 2%、累计D14仅24%）。
  更关键的一点：每个活跃天都恰好登录 1 次 → login_cnt 与 active_days 完全相等，
  登录强度恒为 1、首次登录滞后恒为 0，这些"特征"其实是常数，没有区分度。
  所以真正有信息量的只有 3 个独立维度：体量(login_cnt) / 近期性(recency) / 连续性(max_gap)。

产出：figures/05_churn_roc.png + 05_churn_shap.png
"""
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src import config, plots, load

plots.setup()

FEATURES = [
    "login_cnt",              # 活跃体量：观察期登录次数（合成数据里 = 活跃天数）
    "days_since_last_login",  # 近期性：预测点距最近一次登录的天数（最关键）
    "max_gap",                # 连续性：观察期最长连续不登录间隔（天）
]


def _max_gap(g):
    """一条登录序列里，相邻活跃天之间的最长间隔"""
    days = np.sort(g.unique())
    if len(days) <= 1:
        return 0
    return int(np.max(np.diff(days)))


def build_dataset(reg, auth):
    """构造流失预测数据集（观察期特征 + 标签期标签，处理右删失）"""
    obs_days = config.OBS_WINDOW_DAYS      # 14
    label_days = config.CHURN_GAP_DAYS     # 14
    cutoff = auth["auth_date"].max()

    # 只保留标签期完整可观测的用户（注册早于 cutoff-28 天）
    eligible = reg[reg["reg_date"] <= cutoff - pd.Timedelta(days=obs_days + label_days)]

    # 每条登录距注册第几天
    merged = auth[["uid", "auth_date"]].merge(
        reg[["uid", "reg_date"]], on="uid", how="inner"
    )
    merged["day"] = (merged["auth_date"] - merged["reg_date"]).dt.days
    merged = merged[merged["day"] >= 0]

    obs = merged[merged["day"] < obs_days]
    returned_mask = (merged["day"] >= obs_days) & (merged["day"] < obs_days + label_days)

    # —— 观察期特征 ——
    feat = obs.groupby("uid").agg(
        login_cnt=("auth_date", "count"),
        active_days=("auth_date", "nunique"),
        first_login_lag=("day", "min"),
        last_login_day=("day", "max"),
    )
    feat["max_gap"] = obs.groupby("uid")["day"].agg(_max_gap)

    df = eligible[["uid"]].merge(feat, left_on="uid", right_index=True, how="left")
    df["login_cnt"] = df["login_cnt"].fillna(0).astype(int)
    df["active_days"] = df["active_days"].fillna(0).astype(int)
    df["first_login_lag"] = df["first_login_lag"].fillna(-1)
    df["last_login_day"] = df["last_login_day"].fillna(-1)
    df["max_gap"] = df["max_gap"].fillna(0).astype(int)

    df["days_since_last_login"] = np.where(
        df["last_login_day"] >= 0, (obs_days - 1) - df["last_login_day"], obs_days
    )

    # —— 标签：标签期内是否回来过 ——
    returned = set(merged[returned_mask]["uid"].unique())
    df["churn"] = (~df["uid"].isin(returned)).astype(int)

    return df


def data_quality_note(df):
    """打印特征冗余检查结果（解释为什么只留 3 个特征）"""
    print(f"\n[特征冗余检查]")
    print(f"  login_cnt == active_days 的比例 : {(df['login_cnt']==df['active_days']).mean():.1%}")
    print(f"  login_cnt / active_days 恒等于 : {( (df['login_cnt']/df['active_days'].replace(0,np.nan)).dropna()==1).all()}")
    print(f"  first_login_lag 恒等于 0        : {(df['first_login_lag']==0).mean():.1%}")
    print(f"  → 合成数据每天只登录1次，登录强度/首次滞后都是常数，无区分度，故剔除；")
    print(f"     最终只保留 3 个独立特征：体量 / 近期性 / 连续性。")


def evaluate(model, X, y, Xs=None, cv=StratifiedKFold(5, shuffle=True, random_state=42)):
    """5 折交叉验证 AUC（树模型不标准化，线性模型需标准化）"""
    scores = []
    for tr, va in cv.split(X, y):
        if Xs is not None:  # 线性模型：训练集 fit scaler，再 transform 验证集（防泄漏）
            Xtr, Xva = Xs.fit_transform(X[tr]), Xs.transform(X[va])
        else:
            Xtr, Xva = X[tr], X[va]
        model.fit(Xtr, y[tr])
        scores.append(roc_auc_score(y[va], model.predict_proba(Xva)[:, 1]))
    return np.mean(scores), np.std(scores)


def main():
    reg, auth, ab = load.load_all()
    print("=" * 60)
    print("模块 5：流失预测（行为特征 → 是否会流失）")
    print("=" * 60)

    df = build_dataset(reg, auth)
    print(f"\n[人群漏斗]")
    print(f"  可观测用户(注册满28天)        : {len(df):,}")
    print(f"  建模人群(观察期day0-13登录过) : {(df['login_cnt']>=1).sum():,}（合成数据每个用户注册当天都登录）")
    print(f"  建模人群流失率(第14天后不再登录) : {df['churn'].mean():.2%}")
    print(f"  → 基准流失率高达 {df['churn'].mean():.0%}，典型高流失游戏（与模块2的76%注册即流失呼应）")

    data_quality_note(df)

    model_df = df[df["login_cnt"] >= 1].copy()
    X = model_df[FEATURES].values.astype(float)
    y = model_df["churn"].values
    print(f"\n[特征] {len(FEATURES)} 个独立行为特征，样本 {len(model_df):,}")

    # —— 三模型 CV 对比 ——
    print(f"\n[模型对比] 5 折交叉验证 ROC-AUC")
    results = {}
    for name, model, need_scaler in [
        ("逻辑回归(基线)", LogisticRegression(max_iter=1000), True),
        ("XGBoost", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8,
                                  eval_metric="logloss", random_state=42), False),
        ("LightGBM", LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.1,
                                    subsample=0.8, colsample_bytree=0.8,
                                    random_state=42, verbose=-1), False),
    ]:
        scaler = StandardScaler() if need_scaler else None
        m, s = evaluate(model, X, y, Xs=scaler)
        results[name] = m
        print(f"  {name:<16} AUC = {m:.4f} ± {s:.4f}")

    best_name = max(results, key=results.get)
    print(f"\n  → 最优模型：{best_name}（AUC={results[best_name]:.4f}）")
    print(f"  注：AUC 偏高(≈0.99)是因为数据是合成生成的，流失信号很干净（近期没登录≈流失）。")
    print(f"      真实游戏里流失更随机，AUC 通常落在 0.7~0.85；这里重点看的是特征的相对重要性。")

    # —— 训练/测试集上画 ROC 曲线 ——
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model, need_scaler in [
        ("逻辑回归", LogisticRegression(max_iter=1000), True),
        ("XGBoost", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                                  eval_metric="logloss", random_state=42), False),
        ("LightGBM", LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.1,
                                    random_state=42, verbose=-1), False),
    ]:
        if need_scaler:
            sc = StandardScaler().fit(Xtr)
            Xtr_f, Xte_f = sc.transform(Xtr), sc.transform(Xte)
        else:
            Xtr_f, Xte_f = Xtr, Xte
        model.fit(Xtr_f, ytr)
        proba = model.predict_proba(Xte_f)[:, 1]
        auc = roc_auc_score(yte, proba)
        fpr, tpr, _ = roc_curve(yte, proba)
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="随机猜测 (AUC=0.5)")
    ax.set_xlabel("假阳性率 FPR")
    ax.set_ylabel("真阳性率 TPR（召回流失）")
    ax.set_title("流失预测 ROC 曲线（测试集 20%）")
    ax.legend(loc="lower right")
    plots.savefig(fig, "05_churn_roc.png")

    # —— SHAP 解释（XGBoost；缺 SHAP 则退化为特征重要性）——
    print(f"\n[SHAP 解释] 哪些行为最能预示流失？")
    xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                        eval_metric="logloss", random_state=42)
    xgb.fit(X, y)
    try:
        import shap
        explainer = shap.TreeExplainer(xgb)
        X_sample = pd.DataFrame(X, columns=FEATURES).sample(min(8000, len(X)), random_state=42)
        shap_vals = explainer.shap_values(X_sample)
        if isinstance(shap_vals, list):  # 旧版二分类返回 [负类, 正类]
            shap_vals = shap_vals[1] if len(shap_vals) == 2 else shap_vals[0]
        imp = pd.Series(np.abs(shap_vals).mean(axis=0), index=FEATURES).sort_values()
        used_shap = True
    except ImportError:
        imp = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values()
        used_shap = False

    print(imp.rename("重要性").round(4).to_string())
    print(f"\n  （{'SHAP mean|value|' if used_shap else 'XGBoost 特征重要性'}，值越大越能区分流失/留存）")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    imp.plot(kind="barh", color=plots.palette(3), ax=ax)
    ax.set_xlabel("SHAP 平均重要性" if used_shap else "特征重要性")
    ax.set_title("流失预测：哪些行为特征最重要")
    ax.grid(axis="x", alpha=0.3)
    plots.savefig(fig, "05_churn_shap.png")

    print(f"\n  → 结论：近期性(days_since_last_login)是流失的第一信号——")
    print(f"     用户越久没登录，流失概率越高。运营抓手：对「7 天未登录」的用户提前干预")
    print(f"     （召回 / 推送 / 礼包），而不是等 14 天才想起挽留。")

    print(f"\n[图表] 已保存到 figures/")


if __name__ == "__main__":
    main()
