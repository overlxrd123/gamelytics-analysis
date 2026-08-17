"""全局配置：路径 + 指标口径定义

所有分析模块共用这里的口径，保证"活跃/流失/付费"全项目一致。
"""
from pathlib import Path

# —— 路径 ——
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"
REPORT_DIR = BASE_DIR / "reports"

REG_CSV = DATA_DIR / "reg_data.csv"
AUTH_CSV = DATA_DIR / "auth_data.csv"
AB_CSV = DATA_DIR / "ab_test.csv"

# —— 指标口径（面试会被问，写清楚便于讲） ——
# 活跃用户：某天产生了至少 1 条登录事件(auth)的用户
# 留存 D_N：注册后第 N 天仍登录的用户占当日注册用户的比例
D_RETENTION = [1, 3, 7, 14, 30]  # 关注的留存节点

# 流失定义：观察期内最后一次登录距"截断日"超过 N 天（模块5用）
CHURN_GAP_DAYS = 14
# 预测需要足够的历史观察期，取最后一批用户留出"标签期"
OBS_WINDOW_DAYS = 14   # 用前 14 天行为预测第 15+ 天是否流失

# —— 付费分层（游戏行业通用：大R/中R/小R/非付费） ——
# Minnow 小R / Dolphin 中R / Whale 大R 的金额分界（revenue 单位=货币单位）
PAYER_SEGMENTS = {
    "非付费": 0,
    "小R(Minnow)": 100,
    "中R(Dolphin)": 1000,
    "大R(Whale)": 10**9,  # 上限
}

# —— matplotlib 中文字体（Windows 用微软雅黑，避免图表中文乱码） ——
PLOT_FONT = "Microsoft YaHei"
