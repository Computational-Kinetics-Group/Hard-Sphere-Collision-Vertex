import numpy as np, math, pickle, json, time, argparse, os
from scipy.special import poch
ap=argparse.ArgumentParser(); ap.add_argument('--workdir',default='.'); args=ap.parse_args(); WORKDIR=os.path.abspath(args.workdir)
MAXH=14; DIAM=1.; THETA=5.5; SAMPLE_CELLS=[18,21,23,24,25,26,28,31]; BURN=5000; SAMPLE=5000; DT=3e-4; BLOCK=250
MIS=[(a,b,c) for a in range(MAXH+1) for b in range(MAXH+1-a) for c in range(MAXH+1-a-b)]
DEG=np.array([sum(x) for x in MIS]); SQ=np.array([math.sqrt(math.factorial(a)*math.factorial(b)*math.factorial(c)) for a,b,c in MIS])
def mf(nu):return math.factorial(nu[0])*math.factorial(nu[1])*math.factorial(nu[2])
def af(a):return math.factorial(a[0])*math.factorial(a[1])*math.factorial(a[2])
def Fcoeff(N):
 m={};C=-(32/5)*math.sqrt(math.pi)
 for k in range(N):
  ck=C*float(poch(-.5,k)/poch(3.5,k))*(-1.)**k/math.factorial(k)/4**k
  for p in range(k+1):
   for q in range(k+1-p):
    r=k-p-q;base=(2*p,2*q,2*r);amp=ck*math.factorial(k)/(math.factorial(p)*math.factorial(q)*math.factorial(r))
    for i in range(3):
     for j in range(3):
      a=list(base);a[i]+=1;a[j]+=1;a=tuple(a);m.setdefault(a,np.zeros((3,3)));m[a][i,j]+=amp/4
    for h in range(3):
     a=list(base);a[h]+=2;a=tuple(a);m.setdefault(a,np.zeros((3,3)))
     for i in range(3):m[a][i,i]-=amp/12
 return m
print('precompute',flush=True); t=time.time();F=Fcoeff(MAXH);M=len(MIS);PT=np.zeros((M,M,3,3));QT=np.zeros((M,M,3))
for ii,nu in enumerate(MIS):
 for jj,lam in enumerate(MIS):
  alpha=tuple(nu[k]+lam[k] for k in range(3));fm=F.get(alpha)
  if fm is not None:PT[ii,jj]=(-1.)**sum(lam)*af(alpha)/(SQ[ii]*SQ[jj])*fm
  sp=SQ[ii]*SQ[jj];qv=np.zeros(3)
  for j in range(3):
   if nu[j]>0:
    aa=list(alpha);aa[j]-=1;aa=tuple(aa);fm=F.get(aa)
    if fm is not None:
     ne=list(nu);ne[j]-=1;ne=tuple(ne);qv+=.5*af(aa)*(-1.)**sum(lam)*sp/(mf(ne)*mf(lam))*fm[:,j]
   if lam[j]>0:
    aa=list(alpha);aa[j]-=1;aa=tuple(aa);fm=F.get(aa)
    if fm is not None:
     le=list(lam);le[j]-=1;le=tuple(le);qv+=.5*af(aa)*(-1.)**(sum(lam)-1)*sp/(mf(nu)*mf(le))*fm[:,j]
  QT[ii,jj]=qv
print('tensor time',time.time()-t,flush=True)
def rec(N,W,heat=False):
 ind=np.where(DEG<=N)[0];Ws=W[np.ix_(ind,ind)]
 return np.einsum('ab,abij->ij',Ws,PT[np.ix_(ind,ind)],optimize=True) if not heat else np.einsum('ab,abi->i',Ws,QT[np.ix_(ind,ind)],optimize=True)

def process(rep):
 st=pickle.load(open(os.path.join(WORKDIR,f'shock_r{rep}.pkl'),'rb')); nb=SAMPLE//BLOCK;bd=np.minimum(BLOCK,SAMPLE-np.arange(nb)*BLOCK)*DT
 Se=st['Se']/(SAMPLE*DT);He=st['He']/(SAMPLE*DT);See=np.zeros_like(Se);Hee=np.zeros_like(He)
 for k in range(len(SAMPLE_CELLS)):
  sr=st['Sb'][k]/bd[:,None,None];hr=st['Hb'][k]/bd[:,None];See[k]=sr.std(0,ddof=1)/math.sqrt(nb);Hee[k]=hr.std(0,ddof=1)/math.sqrt(nb)
 prof={}
 for key,arr in [('n',st['profn']),('T',st['proft'])]:prof[key]=np.divide(arr,st['profc'],out=np.zeros_like(arr),where=st['profc']>0)
 prof['Ux']=np.divide(st['profu'][:,0],st['profc'],out=np.zeros(50),where=st['profc']>0)
 ss=[]
 for k,c in enumerate(SAMPLE_CELLS):
  ns=np.array(st['ns'][k]);Ts=np.array(st['Ts'][k]);AL=np.array(st['al'][k]);AF=np.array(st['af'][k]);ws=np.array(st['ws'][k]);wh=np.array(st['wh'][k])
  WLs=np.einsum('s,si,sj->ij',ns*ns*Ts**1.5,AL,AL)/len(AL);WLh=np.einsum('s,si,sj->ij',ns*ns*Ts**2,AL,AL)/len(AL)
  WFs=np.einsum('s,si,sj->ij',ns*ns*THETA**1.5,AF,AF)/len(AF);WFh=np.einsum('s,si,sj->ij',ns*ns*THETA**2,AF,AF)/len(AF)
  rL={};rF={}
  for N in range(2,MAXH+1):rL[str(N)]={'S':rec(N,WLs).tolist(),'H':rec(N,WLh,True).tolist()};rF[str(N)]={'S':rec(N,WFs).tolist(),'H':rec(N,WFh,True).tolist()}
  ss.append({'x':(c+.5)/50,'n':float(ns.mean()),'T':float(Ts.mean()),'S_event':Se[k].tolist(),'H_event':He[k].tolist(),'S_event_se':See[k].tolist(),'H_event_se':Hee[k].tolist(),'S_weak':ws.mean(0).tolist(),'H_weak':wh.mean(0).tolist(),'S_weak_se':(ws.std(0,ddof=1)/math.sqrt(len(ws))).tolist(),'H_weak_se':(wh.std(0,ddof=1)/math.sqrt(len(wh))).tolist(),'local':rL,'fixed':rF})
 return st,prof,ss
out={'theta_fixed':THETA,'max_herm':MAXH,'reps':{}}
for r in [1,2]:
 st,p,s=process(r);out['reps'][str(r)]={'profile':{k:v.tolist() for k,v in p.items()},'samples':s,'hist':st['hist'].tolist(),'hist_local':np.mean(st['hist_local'],axis=0).tolist(),'diagnostics':{'maxg':st['maxg'],'hits':st['hits'],'meanNP':float(np.mean(st['np_hist']))}}
# metadata
G=5/3;Ma=5;T1=1;U1=Ma*math.sqrt(G);R=((G+1)*Ma**2)/((G-1)*Ma**2+2);PR=1+2*G/(G+1)*(Ma**2-1);T2=PR/R;U2=U1/R;N1=1/(math.sqrt(2)*math.pi*.05)
out['states']={'upstream':{'n':N1,'U':U1,'T':1.},'downstream':{'n':N1*R,'U':U2,'T':T2},'density_ratio':R}
json.dump(out,open(os.path.join(WORKDIR,'shock_M5_aggregate.json'),'w'),indent=2)
print('DONE')
