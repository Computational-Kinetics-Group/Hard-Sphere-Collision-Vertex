#!/usr/bin/env python3
"""Integrity/reproducibility checks for the code/data companion."""
from pathlib import Path
import csv, json, math, shutil, subprocess, sys, tempfile

ROOT=Path(__file__).resolve().parent

def run(*args):
    subprocess.run([sys.executable,*map(str,args)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)

def numeric_csv_close(a,b,tol=5e-15):
    ra=list(csv.reader(open(a))); rb=list(csv.reader(open(b)))
    assert ra[0]==rb[0] and len(ra)==len(rb), f'CSV shape/header mismatch: {a} vs {b}'
    for i,(xa,xb) in enumerate(zip(ra[1:],rb[1:]),start=2):
        assert len(xa)==len(xb)
        for ca,cb in zip(xa,xb):
            try:
                fa=float(ca); fb=float(cb)
                if not math.isclose(fa,fb,rel_tol=tol,abs_tol=tol):
                    raise AssertionError(f'numeric mismatch line {i}: {fa} vs {fb}')
            except ValueError:
                assert ca==cb, f'text mismatch line {i}: {ca!r} vs {cb!r}'

def json_close(a,b,tol=5e-15,path=""):
    if type(a)!=type(b):
        raise AssertionError(f"JSON type mismatch at {path}: {type(a)} vs {type(b)}")
    if isinstance(a,dict):
        if a.keys()!=b.keys(): raise AssertionError(f"JSON keys mismatch at {path}")
        for k in a: json_close(a[k],b[k],tol,path+"/"+str(k))
    elif isinstance(a,list):
        if len(a)!=len(b): raise AssertionError(f"JSON length mismatch at {path}")
        for i,(x,y) in enumerate(zip(a,b)): json_close(x,y,tol,path+f"[{i}]")
    elif isinstance(a,float):
        if not math.isclose(a,b,rel_tol=tol,abs_tol=tol):
            raise AssertionError(f"JSON numeric mismatch at {path}: {a} vs {b}")
    else:
        if a!=b: raise AssertionError(f"JSON mismatch at {path}: {a!r} vs {b!r}")

def main():
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        # Exact analytic tables.
        pq=td/'pq.csv'; scalar=td/'scalar.csv'
        run(ROOT/'coefficients/generate_hard_sphere_PQ.py','--max-stress-degree','14','--output',pq)
        assert pq.read_bytes()==(ROOT/'coefficients/hard_sphere_PQ_coefficients_through_degree15.csv').read_bytes()
        run(ROOT/'coefficients/generate_scalar_vertex_table.py','--max-m','6','--output',scalar)
        assert scalar.read_bytes()==(ROOT/'coefficients/hard_sphere_scalar_vertex_coefficients.csv').read_bytes()

        # Wall-driven aggregation from the eight raw replica files.
        wall=td/'wall'; wall.mkdir()
        src=ROOT/'validation/wall_driven/data'
        for p in src.glob('campaign_*_r*.json'): shutil.copy2(p,wall/p.name)
        run(ROOT/'validation/wall_driven/scripts/aggregate_wall_campaign.py','--input-dir',wall,'--output-dir',wall)
        numeric_csv_close(wall/'natural_dsmc_campaign_summary.csv',src/'natural_dsmc_campaign_summary.csv')
        numeric_csv_close(wall/'natural_dsmc_campaign_metrics.csv',src/'natural_dsmc_campaign_metrics.csv')
        json_close(json.load(open(wall/'natural_dsmc_campaign_aggregate.json')),json.load(open(src/'natural_dsmc_campaign_aggregate.json')))

        # Baseline shock publication tables from retained cross-replica JSON.
        shock=td/'shock'; shock.mkdir(); ssrc=ROOT/'validation/shock/data'
        run(ROOT/'validation/shock/scripts/export_baseline_tables.py','--crossrep',ssrc/'shock_M5_crossrep.json','--output-dir',shock)
        numeric_csv_close(shock/'shock_M5_production_summary.csv',ssrc/'shock_M5_production_summary.csv')
        numeric_csv_close(shock/'shock_M5_convergence.csv',ssrc/'shock_M5_convergence.csv')

    # Manifest paths.
    manifest=ROOT/'publication_data/figure_data_manifest.csv'
    missing=[]
    for r in csv.DictReader(open(manifest)):
        if not (ROOT/r['data_file']).exists(): missing.append(r['data_file'])
    assert not missing, 'Missing manifest files: '+', '.join(missing)

    rows=sum(1 for _ in open(ROOT/'coefficients/hard_sphere_PQ_coefficients_through_degree15.csv'))-1
    assert rows==28626, rows
    print('OK: exact coefficient tables regenerate; wall/shock baseline tables reproduce; manifest is complete.')
    print(f'OK: coefficient table contains {rows} independent entries.')

if __name__=='__main__': main()
