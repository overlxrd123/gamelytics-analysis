"""Read-only source-data audit. Run from any directory; no model training or plot writes.

Outputs aggregate evidence only. Equality of numeric IDs is a candidate linkage,
not proof that the AB and activity files share a user namespace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def table(df):
    """Small Markdown tables without an optional tabulate dependency."""
    def cell(value):
        if pd.isna(value):
            return "unknown"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    rows = [[cell(v) for v in row] for row in df.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(df.columns) + " |",
                      "| " + " | ".join(["---"] * len(df.columns)) + " |"] +
                     ["| " + " | ".join(row) + " |" for row in rows])


def payment_summary(df, population):
    payers = df.revenue.gt(0)
    revenue = float(df.revenue.sum())
    return dict(population=population, users=len(df), payers=int(payers.sum()),
                revenue=revenue, payer_rate=float(payers.mean()) if len(df) else np.nan,
                arpu=revenue / len(df) if len(df) else np.nan,
                arppu=revenue / payers.sum() if payers.sum() else np.nan,
                whale_users=int(df.revenue.gt(10000).sum()),
                whale_revenue_share=float(df.loc[df.revenue.gt(10000), "revenue"].sum() / revenue)
                if revenue else np.nan)


def compare_groups(df, population):
    a = df.loc[df.testgroup.eq("a"), "revenue"]
    b = df.loc[df.testgroup.eq("b"), "revenue"]
    na, nb = len(a), len(b)
    if not na or not nb:
        return dict(population=population, a_n=na, b_n=nb)
    pa, pb = a.gt(0).mean(), b.gt(0).mean()
    pooled = (a.gt(0).sum() + b.gt(0).sum()) / (na + nb)
    se_null = np.sqrt(pooled * (1 - pooled) * (1 / na + 1 / nb))
    se_diff = np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    diff = float(pa - pb)
    pvalue = float(2 * stats.norm.sf(abs(diff / se_null))) if se_null else np.nan
    return dict(population=population, a_n=na, b_n=nb, a_payer_rate=float(pa),
                b_payer_rate=float(pb), diff_a_minus_b=diff,
                relative_lift_a_vs_b=diff / pb if pb else np.nan,
                diff_ci95_low=diff - stats.norm.ppf(.975) * se_diff,
                diff_ci95_high=diff + stats.norm.ppf(.975) * se_diff,
                two_sided_z_p=pvalue, a_arpu=float(a.mean()), b_arpu=float(b.mean()))


def audit(data_dir, out_dir, report_path, expected_a_share):
    sources = {}
    frames = {}
    for name, filename in [("reg", "reg_data.csv"), ("auth", "auth_data.csv"), ("ab", "ab_test.csv")]:
        path = data_dir / filename
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        sources[name] = dict(file=filename, bytes=path.stat().st_size, sha256=digest.hexdigest())
        frames[name] = pd.read_csv(path, sep=";")
    reg, auth, ab = (frames[k] for k in ["reg", "auth", "ab"])
    required = [(reg, ["uid", "reg_ts"]), (auth, ["uid", "auth_ts"]),
                (ab, ["user_id", "revenue", "testgroup"])]
    for df, columns in required:
        if not set(columns).issubset(df.columns):
            raise ValueError(f"Missing columns: {columns}")
        if df[columns].isna().any().any():
            raise ValueError(f"Nulls in required columns: {columns}")
    if reg.uid.duplicated().any() or ab.user_id.duplicated().any():
        raise ValueError("Duplicate user keys: resolve before auditing joins")
    if not set(ab.testgroup.unique()).issubset({"a", "b"}):
        raise ValueError("Unexpected AB group; expected labels a/b")
    reg = reg.assign(reg_dt=pd.to_datetime(reg.reg_ts, unit="s"))
    auth = auth.assign(auth_dt=pd.to_datetime(auth.auth_ts, unit="s"))
    cutoff = auth.auth_dt.max().normalize()
    last = auth.groupby("uid").auth_dt.max()
    linked = ab.merge(reg, left_on="user_id", right_on="uid", how="left", validate="one_to_one")
    linked["matched"] = linked.reg_dt.notna()
    linked["auth_matched"] = linked.user_id.isin(last.index)
    matched = linked.loc[linked.matched].copy()
    unknown = linked.loc[~linked.matched].copy()

    schema = pd.DataFrame([
        dict(source=k, rows=len(df), unique_users=df[key].nunique(),
             duplicate_rows=int(df.duplicated().sum()), null_cells=int(df.isna().sum().sum()))
        for k, df, key in [("reg", reg, "uid"), ("auth", auth, "uid"), ("ab", ab, "user_id")]])
    match_rows = []
    for group, part in [("all", linked)] + list(linked.groupby("testgroup")):
        match_rows.append(dict(group=group, users=len(part), matched=int(part.matched.sum()),
                               unmatched=int((~part.matched).sum()), match_rate=float(part.matched.mean()),
                               auth_match_rate=float(part.auth_matched.mean())))
    matching = pd.DataFrame(match_rows)
    payments = pd.DataFrame([payment_summary(ab, "AB_all"), payment_summary(matched, "P2_candidate_matched"),
                             payment_summary(unknown, "P3_unknown")])
    experiments = pd.DataFrame([compare_groups(ab, "AB_all_primary"),
                                compare_groups(matched, "P2_sensitivity"),
                                compare_groups(unknown, "P3_descriptive")])
    reg["reg_month"] = reg.reg_dt.dt.strftime("%Y-%m")
    coverage = reg.groupby("reg_month").agg(reg_users=("uid", "size"))
    monthly_matched = matched.groupby(matched.reg_dt.dt.strftime("%Y-%m")).size()
    coverage["ab_matched_users"] = monthly_matched.reindex(coverage.index, fill_value=0)
    coverage["coverage_rate"] = coverage.ab_matched_users / coverage.reg_users
    coverage = coverage.reset_index()
    dates = pd.DataFrame([dict(population=name, users=len(part), min_reg=str(part.reg_dt.min()),
                               median_reg=str(part.reg_dt.median()), max_reg=str(part.reg_dt.max()))
                          for name, part in [("P1_all_reg", reg), ("P2_candidate_matched", matched)]])
    ordered = reg.sort_values(["reg_ts", "uid"])
    oldest_prefix = set(matched.uid) == set(ordered.head(len(matched)).uid)
    rho = float(stats.spearmanr(reg.uid, reg.reg_ts).statistic)
    ab_contiguous = (ab.user_id.min() == 1 and ab.user_id.max() == len(ab)
                     and ab.user_id.nunique() == len(ab))
    auth_orphans = int((~auth.uid.isin(reg.uid)).sum())
    before_reg = auth[["uid", "auth_dt"]].merge(reg[["uid", "reg_dt"]], on="uid", validate="many_to_one")
    before_reg_count = int(before_reg.auth_dt.lt(before_reg.reg_dt).sum())
    del before_reg
    a_n = int(ab.testgroup.eq("a").sum())
    expected = np.array([expected_a_share, 1 - expected_a_share]) * len(ab)
    srm = stats.chisquare([a_n, len(ab) - a_n], f_exp=expected)
    cross = pd.crosstab(linked.testgroup, linked.matched).reindex(index=["a", "b"], columns=[False, True], fill_value=0)
    missing_balance_p = float(stats.chi2_contingency(cross, correction=False).pvalue)
    recency = (cutoff - last.dt.normalize()).dt.days
    payer_recency = (cutoff - matched.loc[matched.revenue.gt(0), "user_id"].map(last).dt.normalize()).dt.days
    whale_recency = (cutoff - matched.loc[matched.revenue.gt(10000), "user_id"].map(last).dt.normalize()).dt.days
    # Snapshot definition intentionally matches current project: <=30 days is active.
    inactive_rate = float(recency.reindex(reg.uid).gt(30).mean())
    facts = dict(cutoff_date=str(cutoff.date()), sources=sources, uid_reg_ts_spearman=rho,
                 ab_ids_contiguous_1_to_n=bool(ab_contiguous),
                 matched_equals_oldest_registered_prefix=bool(oldest_prefix),
                 matched_reg_population_share=len(matched) / len(reg),
                 unknown_revenue_share=float(unknown.revenue.sum() / ab.revenue.sum()),
                 auth_orphan_rows=auth_orphans, auth_before_registration_rows=before_reg_count,
                 srm_expected_a_share=expected_a_share, srm_chi2=float(srm.statistic),
                 srm_p=float(srm.pvalue), unmatched_balance_chi2_p=missing_balance_p,
                 snapshot_inactive_gt30_rate=inactive_rate,
                 matched_payer_median_recency_days=float(payer_recency.median()),
                 matched_whale_median_recency_days=float(whale_recency.median()))
    # Accounting checks detect dropped or duplicated records in the audit itself.
    assert len(matched) + len(unknown) == len(ab)
    assert np.isclose(matched.revenue.sum() + unknown.revenue.sum(), ab.revenue.sum())
    assert coverage.ab_matched_users.sum() == len(matched)
    assert matching.iloc[0].matched + matching.iloc[0].unmatched == len(ab)

    claims = pd.DataFrame([
        ["H1", "增长全靠拉新", "P1", "contradicted", "相邻90日活跃构成分解：注册未满30日用户贡献+479.04（23.40%），较老用户贡献+1568.29（76.60%）；原结论不成立", "增长由新老用户共同贡献；这是活跃构成的加法分解，不是获客渠道归因"],
        ["H2", "五年留存不变，产品迭代无效", "P1", "partially supported", "02_retention.py只取最近12个注册月；无版本/迭代信息", "报告实际观察区间；不能判断产品迭代效果"],
        ["H3", "全量付费率0.92%，鲸鱼贡献43.5%", "AB_all", "partially supported", "付费数据只覆盖AB样本，不能外推100万注册用户", "明确AB全表或P2分母；全量付费率无法计算"],
        ["H4", "付费用户整体沉睡2.7年/94.6%用户已死", "P2 + P3", "partially supported", "候选ID关联待证；早期队列偏差；未匹配被填0和最大recency", "仅报告P2截至截断日的未活跃时间；P3行为未知；旧聚类比例不可继续引用"],
        ["H5", "AB差异显著但无实际意义，应不上线", "AB_all主分析/P2敏感性", "partially supported", "付费率名义显著性可复算；随机化、实验周期和成本无法核实", "统计差异与业务收益分开；证据不足以建议全量上线"],
        ["H6", "拉新→留存→付费→沉睡因果链", "混用P1与AB样本", "cannot verify", "人群不一致；描述统计和预测不能识别因果", "拆成独立证据与待验证假设，不画已证实因果链"],
    ], columns=["id", "original_headline", "population", "status", "evidence_and_limits", "revised_claim"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    outputs = {"schema": schema, "matching": matching, "payments": payments,
               "ab_comparison": experiments, "cohort_coverage": coverage,
               "registration_dates": dates, "headline_ledger": claims}
    for name, df in outputs.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    try:
        evidence_path = out_dir.relative_to(ROOT).as_posix()
    except ValueError:
        evidence_path = str(out_dir)
    report = f"""# Gamelytics 数据审计

本报告由 `analysis/00_data_audit.py` 从三份本地 CSV 生成；不修改原数据、不重跑或覆盖原图表与模型。
数值身份以输出 `summary.json` 的 SHA-256 为准。数据日截断点：**{cutoff.date()}**，不是运行当天。
数据时间戳按 Unix UTC 解释，日期统计使用无时区 UTC 日；最后一天可能不完整。

## 1. 审计结论与关联边界

- AB ID 是否恰为 1..N 连续段：**{ab_contiguous}**。
- 注册 ID 与注册时间 Spearman：**{rho:.9f}**。
- 按 ID 相等候选关联后，匹配集合是否恰等于最早的同等数量注册用户：**{oldest_prefix}**。
- 候选匹配覆盖全量注册用户 **{len(matched)/len(reg):.4%}**。未匹配用户的注册时间未知，不能称整个 AB 表都是早期注册队列。
- P3 收入占 AB 总收入 **{facts['unknown_revenue_share']:.4%}**。无法关联不等于零活跃或流失，不应填零后参与行为分群。
- 仓库文档未提供可核实的原始下载来源、生成脚本、共同用户命名空间或实验分配说明。
  README 的“合成数据”是项目现有说明，本次未独立证实生成机制。ID 数值重合与顺序结构不能证明跨表身份一致。

## 2. 表粒度、完整性与 population

{table(schema)}

- reg：候选主键 uid，一用户一注册记录；auth：一用户多条登录事件；ab：user_id 一用户一收入与分组记录。
- auth 中未匹配 reg 的记录：{auth_orphans:,}；登录早于注册时间的记录：{before_reg_count:,}。
- P1：全量注册/登录，用于健康度、留存和行为预测。
- P2：AB 与 reg 按数值 ID 候选匹配用户；行为关联与分群以共同 ID 体系成立为前提，且不代表全量队列。
- P3：AB 未匹配用户，只描述其分组、收入和规模；行为信息 unknown。
- AB 全表：保留为 A/B 主比较人群；P2 作为敏感性分析，不因无法关联而删除实验记录。

## 3. 匹配率与队列覆盖

{table(matching)}

{table(dates)}

完整逐月覆盖见 `{evidence_path}/cohort_coverage.csv`，每月分母为该月全量注册用户。
不为 P3 编造注册月份；不将较早队列的长期累计收入与较新队列直接比较。

## 4. 付费统计：分母必须明确

{table(payments)}

金额单位与累计收入观察周期未知；鲸鱼阈值 revenue > 10000 只是现有项目规则，不是经核实的行业标准。
AB_all 与 P2 的数值不能混用。P3 有收入但无行为关联证据。
全量注册用户的付费率、ARPU、ARPPU 和收入集中度无法由此确定。

## 5. A/B：全表主分析与子集敏感性

{table(experiments)}

- 差异方向统一为 A-B；比例按 0..1 输出，乘100为百分点。相对提升以 B 为分母。
- 付费率使用双侧合并比例 z 检验，差异区间使用未合并标准误的95%正态近似区间。
- SRM：**假设**预期 A 占比为 {expected_a_share:.2%}，卡方={srm.statistic:.6f}，p={srm.pvalue:.6f}。
  真实分流比例未知；此处仅是假设性检查。未检出失衡不能证明随机化成功。
- 分组与是否匹配的独立性检查（Pearson卡方、无连续性校正）：p={missing_balance_p:.6f}。
  该检查不能排除其他协变量差异或证明分组无偏；P2 是筛选后子集，不能替代全表。
- 既往 ARPU Bootstrap 区间本次未重新计算，不能把旧值当作本次审计结果。
- 名义显著性不证明因果；随机分配机制、实验时间、收入窗口、主指标、MDE、护栏和成本均无法核实。
- 决策：**现有证据不足以建议全量上线**；也不能据此断言两方案等效或永远不值得采用。
  不根据 p 值不断加样或延长实验；先定义实验与业务阈值。

## 6. 活跃与未活跃：截至数据截断日

- 按现有项目口径，P1 最后登录距截断日 >30 天占比：**{inactive_rate:.4%}**。
  这是快照未活跃状态，不是永久流失；<=30天对应含端点的31个日历日期，需与严格30日窗口区别。
- P2 付费用户最后登录距截断日中位数：**{payer_recency.median():.1f}天**；鲸鱼中位数：**{whale_recency.median():.1f}天**。
  只在候选ID关联成立时具有用户级含义，不能写成“距今天数”或外推当前全量用户。
- 未匹配用户不进入这些行为统计；早期队列的 tenure 较长，需按队列/固定观察窗比较后才讨论差异。

## 7. 六条 headline 结论台账

supported：证据和表述均支持；partially supported：部分数字可复核但解释、范围或前提需修订；
contradicted：现有证据与旧表述方向相反；cannot verify：现有资料不能验证。没有强行给六条旧表述分配 supported，不代表基础统计不可用。

{table(claims)}

## 8. 首次审计时发现的问题（历史台账）

以下描述针对首次审计时的旧实现；当前修复与验证状态见 `docs/validation.md`，不是当前代码缺陷清单。

1. `03_monetization.py` 的 payer_activity 将未匹配登录填0；应保留匹配状态与unknown。
2. `04_segmentation.py` 将未匹配用户填0/最大recency；还同时使用 login_cnt 与 active_days，需检验冗余并避免重复加权。
3. `02_retention.py` 只取最后12个月，与 README “五年”不一致；应在输出中写明时间范围。
4. `06_conclusion.py` 的行为特征对比未使用模块5的满28日资格筛选，存在标签观察不完整问题；混合人群的旧图不能直接用于决策。
5. `05_churn.py` 标签仅表示 day14..27 未登录，不是永久不返回。尚需时间外验证和简单近期性规则比较；预测关联不能证明召回有效。
6. `01_health.py` 当前比较365日窗口的首90日与尾90日，不是连续两个90日；滚动MAU起始窗口缺少此前历史，需单独核验。

## 9. 运行、复核与后续边界

```bash
python analysis/00_data_audit.py
# 可选：指定数据/输出位置及假设分流比例
python analysis/00_data_audit.py --expected-a-share 0.5
```

依赖仅使用现有 pandas、numpy、scipy。输出全部为聚合数据，不导出用户明细。
`--data-dir`、`--out-dir`、`--report` 可覆盖默认路径。重复运行相同输入应得到相同证据。
审计校验：主键/必填字段、组标签、匹配人数守恒、收入守恒、逐月匹配人数守恒。
该脚本保留首次审计的六条旧结论台账与历史快照口径（含最后观测日、recency>30）。
当前分析统一排除最后可能不完整日，使用严格完整30日窗口，因此快照数值可能不同。
修复状态、SQL对账与运行记录见 `docs/validation.md`；业务决策见 `docs/decision_memo.md`。
"""
    report_path.write_text(report, encoding="utf-8")
    print(f"Audit complete: {report_path}")
    print(matching.to_string(index=False))
    print(experiments.to_string(index=False))
    print(f"Accounting checks passed. Evidence: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "data_audit")
    parser.add_argument("--report", type=Path, default=ROOT / "docs" / "data_audit.md")
    parser.add_argument("--expected-a-share", type=float, default=.5,
                        help="Assumed allocation, NOT verified experiment metadata")
    args = parser.parse_args()
    if not 0 < args.expected_a_share < 1:
        parser.error("--expected-a-share must be between 0 and 1")
    audit(args.data_dir, args.out_dir, args.report, args.expected_a_share)


if __name__ == "__main__":
    main()
