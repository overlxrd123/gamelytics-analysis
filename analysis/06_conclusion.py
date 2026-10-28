"""Population-labelled dashboard. No mixed-population causal chain."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import matplotlib.pyplot as plt
from src import load,plots,metrics,config
plots.setup()


def main():
    reg,auth,ab=load.load_all();reg,auth,end=metrics.snapshot(reg,auth)
    retention=metrics.retention(reg,metrics.user_days(reg,auth),end)
    last=auth.groupby("uid").auth_date.max().reindex(reg.uid)
    # Strict 30 complete calendar dates: [end-30d,end).
    active=last.ge(end-pd.Timedelta(days=30)).sum()
    payers=ab.revenue.gt(0).mean();whales=ab.loc[ab.revenue.gt(10000),"revenue"].sum()/ab.revenue.sum()
    metrics.save_report(pd.DataFrame([dict(snapshot_exclusive=str(end.date()),population="P1",users=len(reg),
                         active_last30=int(active),inactive_last30=int(len(reg)-active),inactive_rate=1-active/len(reg))]),"snapshot_activity.csv")
    wide=metrics.candidate_wide(reg,auth,ab,end)
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    fig.suptitle(f"Gamelytics：分人群证据概览 / 截断日 {end.date()}（不含当天）",fontsize=14)
    axes[0,0].bar([f"D{x}" for x in retention.day],retention.rate*100)
    for i,v in enumerate(retention.rate):axes[0,0].text(i,v*100+.1,f"{v:.2%}",ha="center")
    axes[0,0].set(title="P1：全量注册队列留存（各自满观察期）",ylabel="留存率 %")
    axes[0,1].pie([active,len(reg)-active],labels=["30日活跃","30日未活跃"],autopct="%.1f%%")
    axes[0,1].set_title("P1：最近完整30日快照 / 非永久流失")
    axes[1,0].bar(["付费率","鲸鱼收入占比"],[payers*100,whales*100])
    for i,v in enumerate([payers,whales]):axes[1,0].text(i,v*100+.5,f"{v:.2%}",ha="center")
    axes[1,0].set(title="AB全表：付费与集中度（非全量用户）",ylabel="比例 %")
    axes[1,1].bar(["P2候选匹配","P3行为未知"],[wide.matched.sum(),(~wide.matched).sum()])
    axes[1,1].set(title="AB与注册表：按数字ID候选关联",ylabel="用户数")
    fig.text(.5,.025,"身份映射与实验随机化未经核实；人口范围分别标注；不据此拼接因果链或承诺收益。",ha="center",fontsize=10)
    fig.tight_layout(rect=[0,.06,1,.95]);plots.savefig(fig,"06_dashboard.png")
    print("Dashboard updated. Decision and validation plan: docs/decision_memo.md")


if __name__=="__main__":main()
