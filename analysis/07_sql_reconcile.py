"""Execute five SQLite queries against original CSVs, compare every result with independent Pandas calculations."""
import sys,sqlite3,json,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from pandas.testing import assert_frame_equal
from src import config,load,metrics


def check(name,sql,pandas_result,keys,rows):
    a=sql.sort_values(keys).reset_index(drop=True)
    b=pandas_result[a.columns].sort_values(keys).reset_index(drop=True)
    assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-10)
    rows.append(dict(query=name,status="PASS",rows_compared=len(a),columns_compared=len(a.columns),rtol=1e-10,atol=1e-10))


def main():
    start=time.perf_counter();reg,auth,ab=load.load_all()
    db=config.DATA_DIR/"audit.sqlite"  # reproducible scratch database, ignored with data/
    # Rebuild each run, never reuse a stale database. Source CSVs are not modified.
    with sqlite3.connect(db) as conn:
        for name,df,columns in [("reg",reg,["uid","reg_ts"]),("auth",auth,["uid","auth_ts"]),("ab",ab,["user_id","revenue","testgroup"])]:
            df[columns].to_sql(name,conn,if_exists="replace",index=False,chunksize=50000)
        conn.executescript("CREATE UNIQUE INDEX reg_uid ON reg(uid); CREATE UNIQUE INDEX ab_uid ON ab(user_id); CREATE INDEX auth_uid_ts ON auth(uid,auth_ts);")
        rows=[];out=config.REPORT_DIR/"sql_reconciliation";out.mkdir(exist_ok=True)
        def query(name):
            path=config.BASE_DIR/"sql"/name
            return pd.read_sql_query(path.read_text(encoding="utf8"),conn)
        candidate=reg.assign(cohort=reg.reg_date.dt.strftime("%Y-%m"),matched=reg.uid.isin(ab.user_id).astype(int))
        cohort=candidate.groupby("cohort").agg(reg_users=("uid","size"),matched=("matched","sum")).reset_index()
        cohort["coverage_rate"]=cohort.matched/cohort.reg_users
        name="01_matching_cohorts.sql";actual=query(name);check(name,actual,cohort,["cohort"],rows);actual.to_csv(out/(name+".csv"),index=False)
        b=ab.assign(matched=ab.user_id.isin(reg.uid).astype(int),payer=ab.revenue.gt(0).astype(int))
        group=b.groupby("testgroup").agg(users=("user_id","size"),matched=("matched","sum"),payers=("payer","sum"),revenue=("revenue","sum")).reset_index()
        group["unmatched"]=group.users-group.matched
        name="02_ab_balance.sql";actual=query(name);check(name,actual,group,["testgroup"],rows);actual.to_csv(out/(name+".csv"),index=False)
        pop=[]
        for name,part in [("AB_all",b),("P2_candidate_matched",b.loc[b.matched.eq(1)]),("P3_unknown",b.loc[b.matched.eq(0)])]:
            total=part.revenue.sum();pay=part.payer.sum()
            pop.append(dict(population=name,users=len(part),payers=pay,revenue=total,payer_rate=pay/len(part),arpu=total/len(part),
                       arppu=total/pay,whale_users=int(part.revenue.gt(10000).sum()),whale_revenue_share=part.loc[part.revenue.gt(10000),"revenue"].sum()/total))
        name="03_payment_populations.sql";actual=query(name);check(name,actual,pd.DataFrame(pop),["population"],rows);actual.to_csv(out/(name+".csv"),index=False)
        r,a,end=metrics.snapshot(reg,auth)
        retention=metrics.retention(r,metrics.user_days(r,a),end,by_month=True)
        retention=retention.loc[retention.eligible.gt(0)]
        name="04_cohort_retention.sql";actual=query(name);check(name,actual,retention,["cohort","day"],rows);actual.to_csv(out/(name+".csv"),index=False)
        wide=metrics.candidate_wide(r,a,ab,end)
        name="05_candidate_user_wide.sql";actual=query(name);check(name,actual,wide,["user_id"],rows)
        # Do not publish per-user rows; publish aggregate verification evidence only.
        summary=actual.groupby("matched").agg(users=("user_id","size"),known_recent=("recent_active_days","count"),revenue=("revenue","sum")).reset_index()
        summary.to_csv(out/(name+".summary.csv"),index=False)
        manifest={"sqlite_version":sqlite3.sqlite_version,"database":"data/audit.sqlite (ignored; recreated each run)",
                  "elapsed_seconds":round(time.perf_counter()-start,3),"checks":rows,
                  "sql_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((config.BASE_DIR/"sql").glob("*.sql"))}}
        (out/"checks.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__=="__main__": main()
