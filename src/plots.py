"""matplotlib 统一配置：中文字体 + 非交互后端 + 存图

所有分析模块 import 这里的 setup()，图表才能正确显示中文并保存到 figures/。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # 非交互后端，直接存 PNG（CLI 环境无窗口）
import matplotlib.pyplot as plt

from . import config

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#CCB974"]


def setup() -> None:
    plt.rcParams["font.sans-serif"] = [config.PLOT_FONT, "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示
    plt.rcParams["figure.dpi"] = 110


def savefig(fig, name: str) -> str:
    """保存图到 figures/，返回路径"""
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def palette(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]
