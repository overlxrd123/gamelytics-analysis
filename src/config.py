"""Paths and model windows. Calendar snapshot is exclusive, defined in src/metrics.py."""
from pathlib import Path
BASE_DIR=Path(__file__).resolve().parents[1]
DATA_DIR=BASE_DIR/"data"
FIG_DIR=BASE_DIR/"figures"
REPORT_DIR=BASE_DIR/"reports"
REG_CSV=DATA_DIR/"reg_data.csv"
AUTH_CSV=DATA_DIR/"auth_data.csv"
AB_CSV=DATA_DIR/"ab_test.csv"
D_RETENTION=[1,3,7,14,30]
OBS_WINDOW_DAYS=14
CHURN_GAP_DAYS=14
PLOT_FONT="Microsoft YaHei"
