#!/usr/bin/env python3
"""Export the baseline Mach-5 production and convergence tables from shock_M5_crossrep.json."""
import argparse, csv, json, math
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--crossrep',default='shock_M5_crossrep.json')
    ap.add_argument('--output-dir',default='.')
    args=ap.parse_args()
    x=json.load(open(args.crossrep)); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    with open(out/'shock_M5_production_summary.csv','w',newline='') as f:
        fields=['x_over_L','stress_event','stress_event_se','stress_weak','stress_weak_se','stress_Xi_N14','stress_Xi_N14_se',
                'heat_event','heat_event_se','heat_weak','heat_weak_se','heat_Xi_N14','heat_Xi_N14_se']
        w=csv.writer(f, lineterminator='\n'); w.writerow(fields)
        for s in x['samples']:
            w.writerow([s['x'],s['event_S'][0][0],s['event_S_se'][0][0],s['weak_S'][0][0],s['weak_S_se'][0][0],
                        s['conv']['14']['S'][0][0],s['N14_S_se'][0][0],s['event_H'][0],s['event_H_se'][0],s['weak_H'][0],s['weak_H_se'][0],
                        s['conv']['14']['H'][0],s['N14_H_se'][0]])
    with open(out/'shock_M5_convergence.csv','w',newline='') as f:
        w=csv.writer(f, lineterminator='\n'); w.writerow(['N','stress_normalized_RMS','heat_normalized_RMS'])
        for N in range(2,15):
            sw=np.array([s['weak_S'][0][0] for s in x['samples']]); sp=np.array([s['conv'][str(N)]['S'][0][0] for s in x['samples']])
            hw=np.array([s['weak_H'][0] for s in x['samples']]); hp=np.array([s['conv'][str(N)]['H'][0] for s in x['samples']])
            se=math.sqrt(float(np.sum((sp-sw)**2)/np.sum(sw**2))); he=math.sqrt(float(np.sum((hp-hw)**2)/np.sum(hw**2)))
            w.writerow([N,se,he])


if __name__=='__main__': main()
