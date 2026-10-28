"""Build reviewer-facing documentation from current computed evidence, never hand-enter scores."""
import sys,json,platform,importlib.metadata
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src import config
R=config.BASE_DIR;P=config.REPORT_DIR;D=R/"docs"


def table(df):
    def text(x):
        if isinstance(x,float):return f"{x:.2e}" if 0 < abs(x) < 1e-5 else f"{x:.6f}"
        return str(x).replace("|","/")
    return "\n".join(["| "+" | ".join(df.columns)+" |","| "+" | ".join(["---"]*len(df.columns))+" |"]+
              ["| "+" | ".join(text(v) for v in row)+" |" for row in df.itertuples(index=False,name=None)])


def main():
    ab=pd.read_csv(P/"ab_results.csv");a=ab.iloc[0];s=ab.iloc[1]
    pop=pd.read_csv(P/"payment_populations.csv")
    churn=pd.read_csv(P/"churn_validation.csv");temporal=churn.loc[churn.split.eq("temporal_primary")]
    base=temporal.loc[temporal.model.eq("recency_only")].iloc[0]
    xgb=temporal.loc[temporal.model.eq("xgboost")].iloc[0]
    growth=pd.read_csv(P/"growth_decomposition.csv");g0,g1=growth.iloc[0],growth.iloc[1]
    delta=g1.dau-g0.dau;newdelta=g1.new_dau-g0.new_dau;olddelta=g1.older_dau-g0.older_dau
    split=json.loads((P/"churn_split.json").read_text(encoding="utf8"))
    plan=json.loads((P/"ab_planning.json").read_text(encoding="utf8"))
    sql=json.loads((P/"sql_reconciliation/checks.json").read_text(encoding="utf8"))
    pipeline=json.loads((P/"run_logs/pipeline.json").read_text(encoding="utf8"))
    tests=(P/"run_logs/tests.log").read_text(encoding="utf8")
    if any(row["exit_code"] for row in pipeline) or len(pipeline)!=8:raise RuntimeError("Incomplete pipeline")
    if any(row["status"]!="PASS" for row in sql["checks"]):raise RuntimeError("SQL mismatch")
    D.mkdir(exist_ok=True)
    (D/"decision_memo.md").write_text(f"""# 一页决策案例：A/B 方案是否值得采用

**建议：现有证据不足以建议全量上线；先核实实验背景，再按预先约定的收益门槛验证。**
这里的“上线”是决策讨论，不代表原方案、对照关系或真实实验过程已获证实。

## 问题、人群与证据

主分析为 AB 全表 {int(a.a_n+a.b_n):,} 人，不是100万注册用户。P2按数值ID候选匹配到早期注册队列，只作敏感性分析；P3行为未知，仍保留在全表比较中。

| 指标 | AB全表主比较 A-B | P2匹配子集敏感性 A-B |
| --- | --- | --- |
| 付费率差（百分点） | {a.diff_a_minus_b*100:.4f} | {s.diff_a_minus_b*100:.4f} |
| 付费率差95%区间（百分点） | [{a.diff_ci95_low*100:.4f}, {a.diff_ci95_high*100:.4f}] | [{s.diff_ci95_low*100:.4f}, {s.diff_ci95_high*100:.4f}] |
| 双侧名义p值 | {a.two_sided_z_p:.4f} | {s.two_sided_z_p:.4f} |
| ARPU差 | {a.arpu_diff_a_minus_b:.4f} | {s.arpu_diff_a_minus_b:.4f} |
| ARPU差95% Bootstrap区间 | [{a.arpu_ci95_low:.4f}, {a.arpu_ci95_high:.4f}] | [{s.arpu_ci95_low:.4f}, {s.arpu_ci95_high:.4f}] |

A付费率相对B高约 {a.relative_lift_a_vs_b:.2%}，但平均收入差的区间跨0：不能确认平均收入提升，也不证明两方案等效。MW检验针对分布/秩，不能替代均值收入结论。多个指标的事后观察仅作探索，不声称已预注册或控制全部多重比较错误。

## 备选动作与成本假设

- 全量采用：目前缺少随机化、成本与护栏证据，不推荐。
- 保持现状并核查：当前建议；核实分组机制、收入窗口、方案内容、身份映射与预期分流比例。
- 背景可确认后开展新实验：在固定观察期与预定样本量下验证，不能只为追求显著而不断加样。

假设A相对B每位分配用户增量成本为 c，则净收益差=ARPU差-c；金额单位未知，只能在同一假设单位下讨论。`reports/ab_cost_scenarios.csv` 给出 c=0/0.5/1/2 的敏感性分析，均非实际成本或收益。历史收入观察期未知，不能年化，也不能当未来收益预测。

## 验证与决策规则（设计方案，非已完成实验）

身份和随机化可确认后：以净收入/分配用户为拟议主指标，付费率为次指标；投诉、退款、留存等作为待采集护栏。预先约定最低可接受净收益、护栏非劣界值、观察周期、异常与提前停止规则。达到收益门槛且护栏通过才考虑上线；区间不确定则按预先计划判定，不持续窥视p值。

样本量示例：若基线付费率 {plan['control_payer_rate']:.4%}、绝对MDE 0.1个百分点、双侧α=0.05、power=80%、1:1分流，正态近似每组约 {plan['n_per_group']:,} 人。**这是次指标的规划示例，不是历史设计，也不是收入主指标所需样本量。** 实验周期、主指标方差和实际护栏数据仍未知。

证据：`reports/ab_results.csv`（3000次经验分布Bootstrap，种子42）、`reports/ab_planning.json`；SQL与Pandas对账见 `docs/validation.md`。
""",encoding="utf8")
    (D/"metric_contract.md").write_text("""# 人群、口径与数据边界

## 数据身份与时间

来源下载地址、生成脚本、实验协议未提供；README原称合成数据，但无法独立验证生成机制。原始文件身份见 `reports/data_audit/summary.json` 的SHA-256。数值ID相等仅为候选关联，不能证明共同用户命名空间。
Unix秒按UTC解释；使用UTC日历日期，不宣称对应业务当地日。分析快照为最后观测日期的00:00，exclusive，舍弃该可能不完整日期的所有事件和注册。

## 人群

| 人群 | 定义 | 用途与边界 |
| --- | --- | --- |
| P1 | 快照前全量注册与对应登录 | 健康度、留存、行为预测；每项另施观察期资格 |
| AB_all | AB文件全表 | 收入描述、A/B主比较；不代表P1 |
| P2 | AB按数字ID匹配快照前注册 | 行为关联、分群；以身份映射成立为条件，只覆盖早期队列 |
| P3 | 未匹配注册的AB用户 | 仅报告已知的收入、分组、规模，行为保持NULL |

## 指标

- 活跃：该完整日期至少一条登录，用户日去重；缺日志日期补0。
- MAU30：当天及此前29个完整日期的用户并集；输出开始前补足历史，不用截断窗口模拟完整MAU。
- 最近30日：`[snapshot-30日,snapshot)`；未活跃不等于永久流失。
- 严格D_N：注册日期+N小于snapshot的用户为分母，在该日期登录的去重用户为分子，二者使用同一资格集合。
- N日内回访：相对日1..N内至少登录一次，不含day0。各N资格集合不同，跨N曲线不保证严格单调，不能据30日未回访断言永不返回。
- 注册月队列：按注册日期月份分组；最新月份仅使用满相应观察期者，并保留eligible列。
- 付费率：在指定AB人群内revenue>0的用户比例；ARPU与ARPPU分别按全部样本、付费样本分母。金额单位、收入起止时间未知。
- 鲸鱼：revenue>10000，仅为本项目阈值，不是行业通用分类。
- 模型：前14个日历日期(day0..13)为观察期，第14日00:00预测，day14..27无登录为正类；仅保留注册+28日<=snapshot者。近期性=14减观察期最后登录相对日。
- 时间外验证：按注册日期80%分位划分，所有训练标签在首次测试预测时已完成；随机留出仅为参考，不是主结果，不复用旧五折成绩。
- Top10%：按预测分数排序，预算为floor(测试人数×10%)；边界并列采用均匀随机打破并列下的期望命中，不利用标签排序。

## 分群口径

未知用户排除行为分群；已匹配且观察窗口内确无事件才填0。R/F使用截断日recency和固定30日活跃天数，M为AB累计付费标记（时间窗口未知）。不以相对中位数生成“高价值/已死”等价值断言。
K-Means使用recency截顶90日与最近30日活跃天数，标准化后k=4描述性划分，不声称最优k；不重复加入相同的登录次数/活跃天数。输出注册月、tenure分层和完整首30日活跃对比；队列分层不能消除未观测混杂，也未证明业务干预效果。

## 审计历史口径

`data_audit.md`保留首次审计全表与最后观测日、recency>30的旧快照，用于追溯。现行快照排除最后不完整日并采用严格30个完整日期，因此新旧人数/未活跃率略有差别，以对应文件的分母为准。
""",encoding="utf8")
    (D/"interview_notes.md").write_text(f"""# 项目讲解与简历表述

## 三分钟讲法

我用注册、登录和AB收入数据做游戏运营分析。最初最有价值的发现是数据关系问题：AB只有约90.06%的数字ID匹配到注册表，匹配集合又恰好是最早36.4555%的注册用户，因此我不能把付费结果外推到全量，更不能把未匹配者当零活跃。

我先拆出P1行为、AB全表、P2候选匹配和P3未知人群，统一完整UTC日期与留存观察窗口，再用五份SQLite查询与Pandas对账，包括404770行候选用户宽表。活跃增长按相邻90日分解后，新老用户均贡献增长，撤回了原“全靠拉新”的说法。

决策案例聚焦A/B：全表付费率差约{a.diff_a_minus_b*100:.4f}个百分点，p={a.two_sided_z_p:.4f}，但ARPU差的区间跨0，随机化和成本也未知。我把统计证据与上线判断分开，给出带明确假设的成本分析和后续实验设计，没有编造上线收益。

流失模型用前14日预测后14日未回访，做时间切分并排除标签尚不可得的训练样本。单一近期性AUC {base.roc_auc:.4f}，XGBoost {xgb.roc_auc:.4f}；Top10%预算下两者命中相同，因此复杂模型没有展示明显业务优势，也不能证明对这些用户发礼包最有效。

## 可直接采用的简历 bullet（按版面选2—3条）

- 审计100万注册、960万登录及40.5万AB样本，识别9.94%跨表未匹配与早期队列覆盖偏差，分离未知行为与可观测零值，统一留存与完整日期口径。
- 编写5份SQLite审计/指标查询，与Pandas逐项对账，核验404770行候选用户宽表；构建分人群看板和A/B决策memo，明确统计显著性、收入不确定性与外推边界。
- 采用带标签完成约束的时间外验证，近期性基线与XGBoost AUC分别为{base.roc_auc:.4f}/{xgb.roc_auc:.4f}，比较固定触达预算下的命中与召回，避免将预测相关性当作召回收益。

以上是离线项目产出，不写“推动上线”“收入提升”或真实生产效果。

## 五条SQL现场练习

见sql/01..05：解释匹配率的分母、左连接后NULL与0的区别；解释A/B匹配子集为何只作敏感性；写清付费率/ARPU分母；手写D1/D7/D30资格与去重；构建宽表保留unknown。额外口述如何用LAG比较相邻注册月，以及为何相邻月未满观察期时不能直接比较。

## 面试追问准备

- 数字ID碰巧相同能证明同人吗？不能，需来源/映射协议，当前行为关联是条件性分析。
- p<0.05为什么还不建议上线？实验机制未知，收益门槛与成本未知，收入证据不足。
- SRM通过说明随机化成功吗？不能，且这里预期1:1也是假设。
- 辛普森悖论：整体与分层方向可能因权重不同而反转；本项目未声称观察到了该现象。
- 高AUC为什么不等于好召回策略？预测风险不同于可被干预改变的风险，需随机干预验证。
""",encoding="utf8")
    versions={}
    for name in ["pandas","numpy","scipy","matplotlib","scikit-learn","xgboost","lightgbm","shap","pytest"]:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]="not installed"
    (P/"environment.json").write_text(json.dumps(dict(python=platform.python_version(),packages=versions),indent=2),encoding="utf8")
    validation=f"""# 验证与改造结果

## 执行证据

{table(pd.DataFrame(pipeline))}

边界测试输出：

```text
{tests.strip()}
```

SQL执行引擎SQLite {sql['sqlite_version']}；原始CSV重新加载建库，非手工输入结果。浮点容差rtol=atol=1e-10。

{table(pd.DataFrame(sql['checks']))}

全量候选宽表逐行比较8个字段；不发布用户明细，只保存聚合核对结果。SQL文件SHA-256与执行证据见 `reports/sql_reconciliation/checks.json`。原始文件指纹见审计summary，环境版本见 `reports/environment.json`。

## 关键纠正

- 排除最后可能不完整日，快照为2020-09-23 00:00 exclusive；当前P1为完整日期注册用户，非原始100万全表。
- 原增长比较把365日窗口首尾90日当相邻90日；现在为{g0.start}至{g0.end_exclusive}、{g1.start}至{g1.end_exclusive}，都按左闭右开。
- 日均DAU变化 {delta:.2f}；注册未满30日用户贡献 {newdelta:.2f}（{newdelta/delta:.2%}），较老用户贡献 {olddelta:.2f}（{olddelta/delta:.2%}）。这是活跃构成的加法分解，不是获客渠道归因。
- 留存分母/分子使用相同资格；报告写实际最近12个月范围，不推断产品迭代效果。
- P3行为未知；分群剔除unknown、不重复加入登录次数与活跃天数，附队列/tenure与首30日对比。
- A/B全表保留为主比较，P2敏感性，付费率差区间与ARPU区间分别报告；未核实实验机制，不把名义显著性当因果。
- 看板去掉“已死”“永久流失”、混合人群因果链、无证据ROI与LTV承诺。

## 模型验证

{table(temporal[["model","train_users","test_users","roc_auc","average_precision","precision_at_capacity","recall_at_capacity"]])}

首次测试预测日 {split['first_test_prediction']}；训练标签最晚完成日 {split['max_train_label_end']}；排除边界用户 {split['embargo_excluded']:,}。标签在预测时可获得，不跨时点使用未知未来训练标签。
正类为day14..27未登录，不是永久流失。正类率较高，AP需结合该基线解释。复杂模型相对近期性规则的AUC差很小，本次不做显著性优越声明；Top10%预算下未显示优势。3/5/7/10日阈值是离线探索，不是已验证的生产干预阈值。

## 首次审计问题修复状态

| 问题 | 状态 |
| --- | --- |
| 未匹配填零/最大recency | 已修复，unknown保留、行为分群排除 |
| 早期队列外推全量 | 已修正文案并分人群输出 |
| 五年与12月冲突、旧因果链、ROI承诺 | 已移除 |
| 观察窗口/不完整日、滚动MAU历史、相邻90日 | 已统一并重跑 |
| 模型标签可得性、近期性基线与触达预算 | 已验证并输出结果 |
| SQL证据与逐行核对 | 5份查询全部通过 |
| 数据来源、共同身份、实验协议 | 仍未知；不能靠代码修复 |
| 真正生产干预收益、上线效果 | 未发生，不作声称 |

## 改造范围与可恢复性

保留原数据、Git历史和用户笔记本，不创建提交或覆盖用户其他项目。改造前工作区（含未提交脚本）已另行备份交付。根目录原笔记本保留作历史入口，可能含旧输出，当前交付入口为 `python run_all.py`，不要混用旧Notebook输出。
BI按岗位需要再选做；本次交付现有Matplotlib分人群单页看板，不新增无依据的业务漏斗或实验结果。
"""
    (D/"validation.md").write_text(validation,encoding="utf8")
    (D/"runbook.md").write_text("""# 运行与复核

## 环境与输入

使用Python3.10+，建议独立虚拟环境；已验证版本见reports/environment.json。原始文件放在data/：reg_data.csv、auth_data.csv、ab_test.csv，分号分隔。当前无经核实的公开下载地址，不编造来源；文件未随交付包公开。

```bash
python -m pip install -r requirements-dev.txt
python run_all.py
```

从项目根目录执行以上命令；run_all.py内部使用绝对脚本路径，也可以通过绝对路径从其他目录调用。依次执行审计、6个分析、5条SQL对账、边界测试和文档生成；任一步失败即停止，详见reports/run_logs/。SQL用Python内置SQLite，不需要MySQL服务、Linux或云服务器。
首次建SQLite及索引需要时间和额外磁盘空间（数据本体、索引及临时文件），具体以设备为准；data/audit.sqlite是可重建中间库，不应提交Git。修改原始CSV后应重跑全链路，避免混用旧报告。

## 单步排查

```bash
python analysis/00_data_audit.py
python analysis/03_monetization.py
python analysis/05_churn.py
python analysis/07_sql_reconcile.py
python -m pytest tests -q
```

03报告里的Bootstrap=3000，seed=42；对经验金额频次进行多项式重采样，与逐用户有放回抽样是同一经验分布Bootstrap，抽样序列不同于旧实现，所以区间会略变。05主结果为时间外测试，旧五折0.986不能与新结果混标。
所有脚本仅读取原始CSV，写入figures/、reports/和docs/；07覆盖可重建的data/audit.sqlite。输出聚合证据，逐行宽表只在内存核对、不发布用户明细。
SHAP在兼容性失败时可回退为时间外样本置换重要性，reports/churn_importance.csv记录实际方法，图表不能一律声称SHAP。

## 看哪些文件

- README.md：项目入口、现行结论与限制。
- docs/data_audit.md：首次审计事实与旧结论台账，快照口径与当前分析区别见metric_contract.md。
- docs/metric_contract.md：人群、窗口、标签与分群定义。
- docs/decision_memo.md：一页A/B决策案例。
- docs/validation.md：运行证据、SQL与模型结果、修复状态。
- docs/interview_notes.md：三分钟讲法与简历bullet。
- sql/：5条可执行查询。
- figures/06_dashboard.png：当前按人群标注的单页看板。

## 故障

先看对应run_logs文件，不跳过失败步骤继续发布。依赖错误使用同一个Python环境安装requirements-dev.txt；SQL差异优先核对输入文件指纹与窗口定义。不要用填0或放宽误差掩盖匹配/对账错误。缺少来源协议时保持条件性解释，不“修复”数据来满足预期业务结论。
""",encoding="utf8")
    (R/"README.md").write_text(f"""# Gamelytics：可复核的游戏数据分析

围绕一个问题：**AB表中的付费差异，是否足以支持采用某个方案？**
项目价值是数据审计、口径统一、SQL复核与有边界的决策，不声称真实上线或收入提升。

## 先看结论

- 数据为100万注册、9,601,013条登录、404,770条AB记录。来源/生成脚本和实验协议未获核实，原称合成数据，结果不作真实业务外推。
- AB数字ID有9.94%无法匹配注册；匹配集合恰为最早36.4555%的注册用户。数字相等不等于身份映射已证实。
- AB全表付费率差A-B为{a.diff_a_minus_b*100:.4f}个百分点（名义p={a.two_sided_z_p:.4f}）；ARPU差95%区间为[{a.arpu_ci95_low:.2f}, {a.arpu_ci95_high:.2f}]。**目前证据不足以建议全量上线**，不是已证明无差异。
- 相邻90日日均DAU增长由新老用户共同贡献，撤回“全靠拉新”的旧结论。完整日期与各指标分母见口径文档。
- 时间外测试近期性基线AUC={base.roc_auc:.4f}、XGBoost={xgb.roc_auc:.4f}；Top10%预算下无命中优势，不能据高AUC宣称召回收益。

## 交付与证据

| 内容 | 入口 |
| --- | --- |
| 数据审计、六条旧headline台账 | [审计报告](docs/data_audit.md) |
| 人群/时间窗口/指标合同 | [指标口径](docs/metric_contract.md) |
| AB问题、备选动作、成本假设与验证 | [一页决策案例](docs/decision_memo.md) |
| 5条SQLite查询、与Pandas逐项/逐行对账 | [SQL](sql/) / [执行证据](reports/sql_reconciliation/checks.json) |
| 模型、基线、固定预算、修复状态 | [验证报告](docs/validation.md) |
| 可运行说明与故障排查 | [Runbook](docs/runbook.md) |
| 三分钟讲解、简历表述 | [面试材料](docs/interview_notes.md) |

## 快速运行

```bash
python -m pip install -r requirements-dev.txt
python run_all.py
```

需本地data/的三份原始CSV，来源未核实且文件不随仓库发布。SQLite为Python内置，无需额外数据库服务；脚本记录日志，失败即停止。数据库在data/audit.sqlite，可重建且不入库。输入指纹见reports/data_audit/summary.json；验证环境见reports/environment.json。

## 分析范围

P1使用完整日期注册与登录，做健康度、留存、预测。AB全表做收入和A/B主比较。P2为候选匹配早期队列，条件性分析行为分群；P3行为未知，不填0、不参与行为分群。P2仅为实验敏感性分析，不能取代全表。
分析统一排除2020-09-23可能不完整当天，快照为当天00:00（exclusive）；首次审计保留旧快照用于追溯。留存为各自满观察期人群，模型仅定义day14..27未登录，不是永久流失。

## 当前看板

![分人群证据看板](figures/06_dashboard.png)

![AB差异与区间](figures/03_ab_test.png)

![时间外预测与简单基线](figures/05_churn_roc.png)

## 已知限制

共同用户ID体系、随机分组机制、实验周期、收入单位/窗口、成本与护栏未知。分组均衡不证明随机化；事后p值只作名义证据。模型预测关联不能识别因果或可挽回人群。没有完整行为事件链，不做多步转化漏斗；没有实际干预，不写“推动”“提升收益”。
原笔记本保留为历史入口，可能含旧输出；当前结果以run_all.py生成文件为准。BI按目标岗位需求再选做，本次保留静态单页看板。
""",encoding="utf8")
    print("Built README, metric contract, decision memo, validation, runbook and interview notes")


if __name__=="__main__":main()
