"""数据加载 + 时间戳解析

三份原始数据都是分号分隔(;)的 CSV，时间戳是 Unix 秒。
统一转成 pandas 的 datetime 列，供各分析模块直接使用。
"""
from __future__ import annotations

import pandas as pd

from . import config

# Unix秒按UTC解释，使用无时区UTC日期；不推断业务当地时区。
def _ts_to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, unit="s")


def load_reg() -> pd.DataFrame:
    """注册表：100 万用户，uid + 注册时间"""
    df = pd.read_csv(config.REG_CSV, sep=";")
    df["reg_dt"] = _ts_to_dt(df["reg_ts"])
    df["reg_date"] = df["reg_dt"].dt.normalize()
    return df


def load_auth() -> pd.DataFrame:
    """登录表：960 万条登录记录，uid + 登录时间"""
    df = pd.read_csv(config.AUTH_CSV, sep=";")
    df["auth_dt"] = _ts_to_dt(df["auth_ts"])
    df["auth_date"] = df["auth_dt"].dt.normalize()
    return df


def load_ab() -> pd.DataFrame:
    """AB 测试表：40.5 万用户，user_id + 收入 + 分组(a/b)"""
    df = pd.read_csv(config.AB_CSV, sep=";")
    return df


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """一次性加载三份数据"""
    return load_reg(), load_auth(), load_ab()
