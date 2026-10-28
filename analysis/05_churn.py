"""P1 day14..27 non-return prediction: temporal embargo, recency baselines, capacity metrics."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_score,recall_score
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from src import load,plots,metrics,config
plots.setup()
FEATURES=["login_cnt","days_since_last_login","max_gap"]


def build_dataset(reg,auth):
    reg,auth,end=metrics.snapshot(reg,auth)
    return metrics.churn_dataset(reg,auth,end)


def temporal_split(df):
    boundary=df.reg_date.quantile(.8).normalize()
    test=df.loc[df.reg_date.ge(boundary)]
    first_prediction=test.prediction_date.min()
    # Every training label complete before the first held-out scoring date.
    train=df.loc[df.label_end.le(first_prediction) & df.reg_date.lt(boundary)]
    return train,test


def capacity_metrics(y,score,fraction=.1):
    # Expected performance under uniform random tie-breaking at the budget boundary.
    y=np.asarray(y);score=np.asarray(score);k=max(1,int(np.floor(len(y)*fraction)))
    threshold=np.partition(score,len(score)-k)[len(score)-k]
    above=score>threshold;equal=score==threshold
    positives=float(y[above].sum())+float(y[equal].sum())*(k-above.sum())/equal.sum()
    return dict(capacity_fraction=fraction,selected=k,precision_at_capacity=positives/k,
                recall_at_capacity=positives/y.sum() if y.sum() else 0)


def main():
    reg,auth,_=load.load_all();df=build_dataset(reg,auth)
    train,test=temporal_split(df)
    random_train,random_test=train_test_split(df,test_size=.2,stratify=df.churn,random_state=42)
    rows=[];rules=[];fitted={};curves=[]
    for split,tr,te in [("temporal_primary",train,test),("random_reference",random_train,random_test)]:
        models={"logistic":make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42)),
                "xgboost":XGBClassifier(n_estimators=150,max_depth=4,learning_rate=.1,n_jobs=2,random_state=42,eval_metric="logloss"),
                "lightgbm":LGBMClassifier(n_estimators=150,num_leaves=15,n_jobs=2,random_state=42,verbosity=-1)}
        scores={"recency_only":te.days_since_last_login.to_numpy()}
        for name,model in models.items():
            model.fit(tr[FEATURES],tr.churn)
            scores[name]=model.predict_proba(te[FEATURES])[:,1]
            if split=="temporal_primary":fitted[name]=model
        for name,score in scores.items():
            row=dict(split=split,model=name,train_users=len(tr),test_users=len(te),test_positive_rate=float(te.churn.mean()),
                     roc_auc=float(roc_auc_score(te.churn,score)),average_precision=float(average_precision_score(te.churn,score)))
            row.update(capacity_metrics(te.churn,score));rows.append(row)
            if split=="temporal_primary":curves.append((name,te.churn.to_numpy(),score))
        if split=="temporal_primary":
            for days in [3,5,7,10]:
                pred=te.days_since_last_login.ge(days)
                rules.append(dict(days_threshold=days,selected=int(pred.sum()),selected_fraction=float(pred.mean()),
                                  precision=float(precision_score(te.churn,pred,zero_division=0)),
                                  recall=float(recall_score(te.churn,pred,zero_division=0))))
    result=pd.DataFrame(rows);metrics.save_report(result,"churn_validation.csv")
    metrics.save_report(pd.DataFrame(rules),"churn_rules.csv")
    split_info=dict(boundary_reg_date=str(test.reg_date.min().date()),
                    first_test_prediction=str(test.prediction_date.min().date()),
                    max_train_label_end=str(train.label_end.max().date()),
                    train_users=len(train),test_users=len(test),embargo_excluded=len(df)-len(train)-len(test),
                    snapshot_exclusive=str(auth.auth_date.max().date()),label="no login on relative days 14..27",
                    tie_policy="expected precision/recall under uniform random ties",capacity_fraction=.1)
    config.REPORT_DIR.joinpath("churn_split.json").write_text(json.dumps(split_info,indent=2),encoding="utf8")
    fig,ax=plt.subplots(figsize=(8,6))
    for name,y,score in curves:
        fpr,tpr,_=roc_curve(y,score);ax.plot(fpr,tpr,label=f"{name} AUC={roc_auc_score(y,score):.4f}")
    ax.plot([0,1],[0,1],ls="--",c="gray");ax.set(title="P1：时间外验证",xlabel="FPR",ylabel="TPR");ax.legend()
    plots.savefig(fig,"05_churn_roc.png")
    # Explain a training-only fitted model on unseen temporal test users.
    sample=test.sample(min(3000,len(test)),random_state=42)
    try:
        import shap
        values=shap.TreeExplainer(fitted["xgboost"]).shap_values(sample[FEATURES])
        if isinstance(values,list):values=values[-1]
        importance=np.abs(values).mean(axis=0);method="SHAP mean absolute value"
    except (ImportError,ValueError,TypeError,AttributeError) as exc:
        p=permutation_importance(fitted["xgboost"],sample[FEATURES],sample.churn,scoring="roc_auc",n_repeats=5,random_state=42,n_jobs=1)
        importance=p.importances_mean;method="held-out permutation ROC-AUC decrease"
        print("SHAP unavailable:",type(exc).__name__)
    imp=pd.DataFrame(dict(feature=FEATURES,importance=importance,method=method))
    metrics.save_report(imp,"churn_importance.csv")
    fig,ax=plt.subplots(figsize=(9,4))
    ax.barh(imp.feature,imp.importance);ax.set_title("时间外样本：模型特征解释（非因果）");ax.set_xlabel(method)
    plots.savefig(fig,"05_churn_shap.png")
    print(result.to_string(index=False));print(pd.DataFrame(rules).to_string(index=False))
    print("Threshold comparison is exploratory, not a proven intervention policy; high AUC is not proof of external validity.")


if __name__=="__main__":main()
