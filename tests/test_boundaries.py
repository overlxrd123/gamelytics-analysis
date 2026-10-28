"""Adversarial checks for population, calendar and leakage boundaries."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from src import metrics
ROOT=Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location("tested_"+name,ROOT/"analysis"/name)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_unknown_is_not_zero_and_known_window_zero_is_observable():
    reg=pd.DataFrame({"uid":[1],"reg_date":pd.to_datetime(["2020-01-01"])})
    auth=pd.DataFrame({"uid":[1,99],"auth_date":pd.to_datetime(["2020-01-01","2020-02-01"])})
    ab=pd.DataFrame({"user_id":[1,99],"testgroup":["a","b"],"revenue":[1,100]})
    x=metrics.candidate_wide(reg,auth,ab,pd.Timestamp("2020-03-01")).set_index("user_id")
    assert x.loc[1,"recent_active_days"]==0
    assert x.loc[1,"recency"]==60
    assert x.loc[99,["login_cnt","recent_active_days","recency","tenure_days"]].isna().all()
    assert x.revenue.sum()==101


def test_retention_excludes_partial_day_and_deduplicates_logins():
    reg=pd.DataFrame({"uid":[1,2],"reg_date":pd.to_datetime(["2020-01-01","2020-01-02"])})
    auth=pd.DataFrame({"uid":[1,1,2],"auth_date":pd.to_datetime(["2020-01-02","2020-01-02","2020-01-03"])})
    r,a,end=metrics.snapshot(reg,auth)
    x=metrics.retention(r,metrics.user_days(r,a),end,days=[1]).iloc[0]
    assert x.eligible==1 and x.retained==1 and x.rate==1


def test_label_window_edges_and_future_login_not_a_feature():
    reg=pd.DataFrame({"uid":[1,2,3],"reg_date":pd.to_datetime(["2020-01-01"]*2+["2020-01-04"])})
    auth=pd.DataFrame({"uid":[1,1,1,2,2,3],"auth_date":pd.to_datetime([
        "2020-01-01","2020-01-14","2020-01-15","2020-01-01","2020-01-29","2020-01-04"])})
    x=metrics.churn_dataset(reg,auth,pd.Timestamp("2020-01-29")).set_index("uid")
    assert set(x.index)=={1,2}
    assert x.loc[1,"login_cnt"]==2 and x.loc[1,"days_since_last_login"]==1
    assert x.loc[1,"churn"]==0 and x.loc[2,"churn"]==1
    assert x.loc[2,"login_cnt"]==1  # day28 login must not affect observation or label


def test_temporal_embargo_prevents_unavailable_training_labels():
    m=module("05_churn.py")
    dates=pd.date_range("2020-01-01",periods=30,freq="3D")
    df=pd.DataFrame({"reg_date":dates,"prediction_date":dates+pd.Timedelta(days=14),"label_end":dates+pd.Timedelta(days=28)})
    train,test=m.temporal_split(df)
    assert len(train)>0 and len(test)>0
    assert train.label_end.max()<=test.prediction_date.min()
    assert train.reg_date.max()<test.reg_date.min()
    assert len(train)+len(test)<len(df)


def test_capacity_ties_do_not_depend_on_user_order():
    m=module("05_churn.py")
    y=np.array([1,0,1,0,1,0,1,0,1,0])
    a=m.capacity_metrics(y,np.ones(10),.2)
    b=m.capacity_metrics(y[::-1],np.ones(10),.2)
    assert a==b and a["precision_at_capacity"]==.5 and a["recall_at_capacity"]==.2


def test_empirical_bootstrap_known_constant_amounts():
    m=module("03_monetization.py")
    lo,hi=m.bootstrap_mean_diff(np.array([5.,5.]),np.array([2.,2.,2.]),repeats=120)
    assert lo==hi==3
