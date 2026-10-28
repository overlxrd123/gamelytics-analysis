"""Run the reproducible project pipeline from any working directory."""
import os,sys,subprocess,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    env=os.environ.copy();env["PYTHONIOENCODING"]="utf-8"
    env.setdefault("OMP_NUM_THREADS","2");env.setdefault("OPENBLAS_NUM_THREADS","2")
    logs=ROOT/"reports/run_logs";logs.mkdir(parents=True,exist_ok=True)
    records=[]
    steps=["00_data_audit.py","01_health.py","02_retention.py","03_monetization.py","04_segmentation.py","05_churn.py","06_conclusion.py","07_sql_reconcile.py"]
    for name in steps:
        start=time.perf_counter()
        with (logs/(name+".log")).open("w",encoding="utf8") as log:
            result=subprocess.run([sys.executable,str(ROOT/"analysis"/name)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(step=name,exit_code=result.returncode,seconds=round(time.perf_counter()-start,3)))
        (logs/"pipeline.json").write_text(json.dumps(records,indent=2),encoding="utf8")
        print(name,"PASS" if result.returncode==0 else "FAIL",flush=True)
        if result.returncode:
            print("See",logs/(name+".log"));return result.returncode
    tests=subprocess.run([sys.executable,"-m","pytest","tests","-q"],cwd=ROOT,env=env,capture_output=True,text=True,encoding="utf-8",errors="replace")
    (logs/"tests.log").write_text((tests.stdout or "")+(tests.stderr or ""),encoding="utf8")
    if tests.returncode:print((tests.stdout or "")+(tests.stderr or ""));return tests.returncode
    return subprocess.run([sys.executable,str(ROOT/"analysis/08_build_docs.py")],cwd=ROOT,env=env).returncode

if __name__=="__main__":sys.exit(main())
