#!/usr/bin/env python3
"""Combine the two replicas of each final wall-driven DSMC case.

Reads campaign_<case>_r1.json and campaign_<case>_r2.json from --input-dir.
Writes the aggregate JSON, flat comparison CSV, and normalized-RMS metrics used
in the Supplemental Material.
"""
import argparse, csv, json, math
from pathlib import Path
import numpy as np

CASES = ['couette','fourier','combined','strong']
ORDERS = [str(n) for n in range(2,9)]
QROWS = [('S_xx','S',0,0),('S_yy','S',1,1),('S_xy','S',0,1),('H_x','H',0,None),('H_y','H',1,None)]


def combine_value(a,b):
    return 0.5*(np.asarray(a,float)+np.asarray(b,float))


def combine_unc(a,b,ea,eb):
    a=np.asarray(a,float); b=np.asarray(b,float); ea=np.asarray(ea,float); eb=np.asarray(eb,float)
    return np.sqrt((ea*ea+eb*eb)/4.0 + ((a-b)/2.0)**2)


def tolist(x):
    return np.asarray(x).tolist()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir',default='.')
    ap.add_argument('--output-dir',default='.')
    args=ap.parse_args()
    inp=Path(args.input_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    agg={'cases':{},'sampling_positions':[0.175,0.325,0.525,0.675,0.825],
         'notes':{'replicates':2,'near_wall_positions':[0.175,0.825],'hermite_orders':[2,3,4,5,6,7,8]}}
    raw={}
    for case in CASES:
        r1=json.load(open(inp/f'campaign_{case}_r1.json'))
        r2=json.load(open(inp/f'campaign_{case}_r2.json'))
        raw[case]=(r1,r2)
        samples=[]
        for s1,s2 in zip(r1['samples'],r2['samples']):
            s={'y':s1['y']}
            for key in ['S_event','S_weak','H_event','H_weak']:
                s[key]=tolist(combine_value(s1[key],s2[key]))
                s[key+'_unc']=tolist(combine_unc(s1[key],s2[key],s1[key+'_se'],s2[key+'_se']))
            s['reconstruction']={}
            for N in ORDERS:
                s['reconstruction'][N]={}
                for key in ['S','H']:
                    s['reconstruction'][N][key]=tolist(combine_value(s1['reconstruction'][N][key],s2['reconstruction'][N][key]))
            samples.append(s)
        agg['cases'][case]=samples
    with open(out/'natural_dsmc_campaign_aggregate.json','w') as f:
        json.dump(agg,f,indent=2)

    header=['case','y_over_L','quantity','event','event_uncertainty','weak','weak_uncertainty','hermite_N8','hermite_replica_spread']
    with open(out/'natural_dsmc_campaign_summary.csv','w',newline='') as f:
        w=csv.writer(f, lineterminator='\n'); w.writerow(header)
        for case in CASES:
            r1,r2=raw[case]
            for k,s in enumerate(agg['cases'][case]):
                for q,sector,i,j in QROWS:
                    if sector=='S':
                        ev=s['S_event'][i][j]; eu=s['S_event_unc'][i][j]; wk=s['S_weak'][i][j]; wu=s['S_weak_unc'][i][j]; hv=s['reconstruction']['8']['S'][i][j]
                        a=r1['samples'][k]['reconstruction']['8']['S'][i][j]; b=r2['samples'][k]['reconstruction']['8']['S'][i][j]
                    else:
                        ev=s['H_event'][i]; eu=s['H_event_unc'][i]; wk=s['H_weak'][i]; wu=s['H_weak_unc'][i]; hv=s['reconstruction']['8']['H'][i]
                        a=r1['samples'][k]['reconstruction']['8']['H'][i]; b=r2['samples'][k]['reconstruction']['8']['H'][i]
                    w.writerow([case,s['y'],q,ev,eu,wk,wu,hv,abs(a-b)/2.0])

    with open(out/'natural_dsmc_campaign_metrics.csv','w',newline='') as f:
        w=csv.writer(f, lineterminator='\n'); w.writerow(['quantity','subset','N8_normalized_RMS_vs_weak'])
        for sec in ['S','H']:
            for subset in ['all','near','bulk']:
                num=den=0.0
                for case in CASES:
                    for s in agg['cases'][case]:
                        near=abs(s['y']-.175)<1e-12 or abs(s['y']-.825)<1e-12
                        if subset=='near' and not near: continue
                        if subset=='bulk' and near: continue
                        wk=np.asarray(s[sec+'_weak']); pr=np.asarray(s['reconstruction']['8'][sec])
                        num += float(np.sum((pr-wk)**2)); den += float(np.sum(wk**2))
                w.writerow([sec,subset,math.sqrt(num/den)])


if __name__=='__main__': main()
