"""Conditional P2 snapshot segmentation; exclude unknowns; fixed 30-day frequency."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src import load,plots,metrics
plots.setup()


def build_features(reg,auth,ab,end):
    return metrics.candidate_wide(reg,auth,ab,end)


def main():
    reg,auth,ab=load.load_all();reg,auth,end=metrics.snapshot(reg,auth)
    wide=build_features(reg,auth,ab,end)
    unknown=wide.loc[~wide.matched]
    metrics.save_report(pd.DataFrame([dict(population="P3_unknown",users=len(unknown),revenue=unknown.revenue.sum(),
                                          payers=int(unknown.revenue.gt(0).sum()))]),"segmentation_unknown.csv")
    df=wide.loc[wide.matched & wide.last_date.notna()].copy()
    # RFM-style snapshot flags: explicit thresholds, no misleading relative value labels.
    df["rfm"]=np.select([df.recency.le(7),df.recency.le(30)],["R_1_7d","R_8_30d"],default="R_gt30d")
    df["rfm"] += np.where(df.recent_active_days.ge(4),"_F_ge4","_F_lt4")
    df["rfm"] += np.where(df.revenue.gt(0),"_M_payer","_M_nonpayer")
    def profile(part,keys):
        return part.groupby(keys,observed=True).agg(users=("user_id","size"),
            mean_recent_active_days=("recent_active_days","mean"),median_recency=("recency","median"),
            payer_rate=("revenue",lambda s:s.gt(0).mean()),mean_cumulative_revenue=("revenue","mean")).reset_index()
    metrics.save_report(profile(df,["rfm"]),"rfm_profiles.csv")
    # Both features use current behavior, not duplicated lifetime login counts.
    df["recency_capped90"]=df.recency.clip(upper=90)
    features=["recency_capped90","recent_active_days"]
    x=StandardScaler().fit_transform(df[features])
    model=KMeans(n_clusters=4,n_init=10,random_state=42).fit(x)
    df["cluster"]=model.labels_
    metrics.save_report(profile(df,["cluster"]),"cluster_profiles.csv")
    df["tenure_band"]=pd.cut(df.tenure_days,bins=[0,365,730,1460,np.inf],right=False)
    metrics.save_report(profile(df,["reg_month"]),"segmentation_cohort_profiles.csv")
    metrics.save_report(profile(df,["tenure_band"]),"segmentation_tenure_profiles.csv")
    # Reference first-30-day behavior for matched cohorts with complete exposure.
    eligible=df.loc[df.reg_date+pd.Timedelta(days=30)<=end,["user_id","reg_month"]].copy()
    pairs=metrics.user_days(reg,auth)
    counts=pairs.loc[pairs.day.lt(30)].groupby("uid").size()
    eligible["active_first30"]=eligible.user_id.map(counts).fillna(0)
    cohort=eligible.groupby("reg_month").agg(users=("user_id","size"),mean_first30=("active_first30","mean")).reset_index()
    metrics.save_report(cohort,"segmentation_first30_by_cohort.csv")
    print(profile(df,["cluster"]).to_string(index=False))
    print("k=4 is a descriptive choice, not a proven optimum; R/F thresholds are illustrative. No causal/value labels.")
    fig,ax=plt.subplots(figsize=(10,5))
    sample=df.sample(min(12000,len(df)),random_state=42)
    ax.scatter(sample.recency_capped90,sample.recent_active_days,c=sample.cluster,s=8,alpha=.3,cmap="tab10")
    ax.set(title="P2候选匹配用户：行为分群（k=4）",xlabel="距截断日天数（上限90）",ylabel="最近完整30日活跃天数")
    plots.savefig(fig,"04_segmentation.png")


if __name__=="__main__":main()
