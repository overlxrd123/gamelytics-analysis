"""P1 retention on fully observed UTC dates; no product-version or causal claims."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import load, plots, metrics
plots.setup()


def main():
    reg, auth, _ = load.load_all()
    reg, auth, end = metrics.snapshot(reg, auth)
    pairs = metrics.user_days(reg, auth)
    strict = metrics.retention(reg, pairs, end, range(1,31))
    cohort = metrics.retention(reg, pairs, end, (1,7,30), True)
    first = pairs.loc[pairs.day.ge(1)].groupby("uid").day.min()
    rows = []
    for n in range(1,31):
        ids = reg.loc[reg.reg_date+pd.Timedelta(days=n)<end,"uid"]
        count = int(first.reindex(ids).le(n).sum())
        rows.append(dict(day=n, eligible=len(ids), returned=count, rate=count/len(ids)))
    cumulative = pd.DataFrame(rows)
    metrics.save_report(strict,"retention_daily.csv")
    metrics.save_report(cohort,"retention_cohort.csv")
    metrics.save_report(cumulative,"return_within_n_days.csv")
    print(strict.loc[strict.day.isin([1,7,30])].to_string(index=False))
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(strict.day,strict.rate*100,marker=".")
    axes[0].set_title("P1：第N天当天留存")
    axes[1].plot(cumulative.day,cumulative.rate*100,marker=".")
    axes[1].set_title("P1：N天内至少回访一次（不含注册日）")
    for ax in axes: ax.set(xlabel="注册后天数",ylabel="比例 %")
    plots.savefig(fig,"02_retention_curve.png")
    matrix = cohort.pivot(index="cohort",columns="day",values="rate").tail(12)
    fig, ax = plt.subplots(figsize=(7,6))
    im=ax.imshow(matrix.values*100,aspect="auto",cmap="YlGnBu")
    ax.set_xticks(range(len(matrix.columns)),[f"D{x}" for x in matrix.columns])
    ax.set_yticks(range(len(matrix)),matrix.index)
    ax.set_title(f"P1：{matrix.index[0]} 至 {matrix.index[-1]} 注册队列")
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            v=matrix.iloc[i,j]
            ax.text(j,i,f"{v*100:.1f}%" if np.isfinite(v) else "未满观察期",ha="center",va="center",fontsize=8)
    fig.colorbar(im,ax=ax,label="留存率 %")
    plots.savefig(fig,"02_cohort_heatmap.png")
    print("No version metadata: stable retention is not evidence of ineffective iteration.")


if __name__ == "__main__":
    main()
