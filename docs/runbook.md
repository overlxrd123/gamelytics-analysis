# 运行与复核

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
