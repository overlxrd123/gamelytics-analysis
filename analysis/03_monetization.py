"""AB full sample primary; linked sample sensitivity; unknown behavior stays unknown."""
import sys,json,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from src import load, plots, metrics, config
spec=importlib.util.spec_from_file_location("audit",Path(__file__).with_name("00_data_audit.py"))
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
plots.setup()


def bootstrap_mean_diff(a,b,repeats=3000,seed=42):
    # Multinomial counts over distinct observed amounts == exact empirical iid bootstrap.
    rng=np.random.default_rng(seed)
    av,ac=np.unique(a,return_counts=True);bv,bc=np.unique(b,return_counts=True)
    diffs=[]
    for start in range(0,repeats,100):
        size=min(100,repeats-start)
        am=rng.multinomial(len(a),ac/ac.sum(),size=size)@av/len(a)
        bm=rng.multinomial(len(b),bc/bc.sum(),size=size)@bv/len(b)
        diffs.extend(am-bm)
    return np.percentile(diffs,[2.5,97.5])


def payer_activity(reg,auth,ab,end):
    wide=metrics.candidate_wide(reg,auth,ab,end)
    part=wide.loc[wide.matched].copy()
    part["payer"]=part.revenue.gt(0)
    # Fixed window reduces exposure differences; ID association remains conditional.
    return part.groupby("payer").agg(users=("user_id","size"),
              mean_recent_active_days=("recent_active_days","mean"),
              median_recency=("recency","median")).reset_index()


def main():
    reg,auth,ab=load.load_all();reg,auth,end=metrics.snapshot(reg,auth)
    wide=metrics.candidate_wide(reg,auth,ab,end)
    results=[]
    for population,part in [("AB_all_primary",ab),("P2_sensitivity",wide.loc[wide.matched])]:
        row=audit.compare_groups(part,population)
        a=part.loc[part.testgroup.eq("a"),"revenue"].to_numpy()
        b=part.loc[part.testgroup.eq("b"),"revenue"].to_numpy()
        lo,hi=bootstrap_mean_diff(a,b)
        row.update(arpu_diff_a_minus_b=float(a.mean()-b.mean()),arpu_ci95_low=lo,arpu_ci95_high=hi,
                   bootstrap_repeats=3000,bootstrap_seed=42,
                   mw_distribution_p=float(stats.mannwhitneyu(a,b,alternative="two-sided").pvalue))
        results.append(row)
    result=pd.DataFrame(results);metrics.save_report(result,"ab_results.csv")
    summaries=pd.DataFrame([audit.payment_summary(part,name) for name,part in
             [("AB_all",ab),("P2_candidate_matched",wide.loc[wide.matched]),("P3_unknown",wide.loc[~wide.matched])]])
    metrics.save_report(summaries,"payment_populations.csv")
    metrics.save_report(payer_activity(reg,auth,ab,end),"payer_activity_matched.csv")
    # A-B net value at hypothetical cost per assigned user; no actual cost is observed.
    primary=result.iloc[0]
    scenarios=pd.DataFrame([dict(assumed_incremental_cost_a_vs_b=c,
                  net_mean=primary.arpu_diff_a_minus_b-c,
                  net_ci95_low=primary.arpu_ci95_low-c,net_ci95_high=primary.arpu_ci95_high-c)
                  for c in [0,0.5,1,2]])
    metrics.save_report(scenarios,"ab_cost_scenarios.csv")
    # Planning illustration only: two independent proportions, equal allocation.
    p0=float(primary.b_payer_rate);mde=.001;p1=p0+mde;pbar=(p0+p1)/2
    n=int(np.ceil(((stats.norm.ppf(.975)*np.sqrt(2*pbar*(1-pbar))+
         stats.norm.ppf(.8)*np.sqrt(p0*(1-p0)+p1*(1-p1)))/mde)**2))
    config.REPORT_DIR.joinpath("ab_planning.json").write_text(json.dumps(dict(
        status="hypothetical_not_historical_design",control_payer_rate=p0,absolute_mde=mde,
        alpha=.05,power=.8,n_per_group=n,allocation="1:1",duration="unknown",cost_unit="unknown"),indent=2),encoding="utf8")
    print(result.to_string(index=False))
    print("MW tests distributions/ranks, not a substitute for mean revenue inference. No causal assignment evidence.")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    payers=ab.loc[ab.revenue.gt(0),"revenue"]
    axes[0].hist(np.log10(payers),bins=35)
    axes[0].set(title="AB全表：付费金额分布",xlabel="log10 金额（单位未知）",ylabel="人数")
    p=summaries.iloc[0]
    axes[1].bar(["人数占比","收入占比"],[p.whale_users/p.users*100,p.whale_revenue_share*100])
    axes[1].set(title="AB全表：金额>10000的用户",ylabel="比例 %")
    plots.savefig(fig,"03_revenue_longtail.png")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(len(result))
    for ax,key,lo,hi,scale,title in [(axes[0],"diff_a_minus_b","diff_ci95_low","diff_ci95_high",100,"付费率差 A-B（百分点）"),
                                    (axes[1],"arpu_diff_a_minus_b","arpu_ci95_low","arpu_ci95_high",1,"ARPU差 A-B（单位未知）")]:
        means=result[key].to_numpy()*scale
        ax.errorbar(x,means,yerr=[means-result[lo].to_numpy()*scale,result[hi].to_numpy()*scale-means],fmt="o",capsize=5)
        ax.axhline(0,color="gray",ls="--");ax.set_xticks(x,["AB全表","P2敏感性"]);ax.set_title(title+" / 95%区间")
    fig.tight_layout();plots.savefig(fig,"03_ab_test.png")


if __name__=="__main__": main()
