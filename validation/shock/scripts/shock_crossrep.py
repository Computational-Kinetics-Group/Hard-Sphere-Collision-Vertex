import numpy as np, math, pickle, json, argparse, os
from scipy.special import poch
ap=argparse.ArgumentParser(); ap.add_argument('--workdir',default='.'); args=ap.parse_args(); WORKDIR=os.path.abspath(args.workdir)
MAXH=14;TH=5.5;CELLS=[18,21,23,24,25,26,28,31]
MIS=[(a,b,c) for a in range(MAXH+1) for b in range(MAXH+1-a) for c in range(MAXH+1-a-b)];DEG=np.array([sum(x) for x in MIS]);SQ=np.array([math.sqrt(math.factorial(a)*math.factorial(b)*math.factorial(c)) for a,b,c in MIS])
def mf(n):return math.factorial(n[0])*math.factorial(n[1])*math.factorial(n[2])
def af(a):return math.factorial(a[0])*math.factorial(a[1])*math.factorial(a[2])
def fc():
 m={};C=-(32/5)*math.sqrt(math.pi)
 for k in range(MAXH):
  ck=C*float(poch(-.5,k)/poch(3.5,k))*(-1.)**k/math.factorial(k)/4**k
  for p in range(k+1):
   for q in range(k+1-p):
    r=k-p-q;b=(2*p,2*q,2*r);amp=ck*math.factorial(k)/(math.factorial(p)*math.factorial(q)*math.factorial(r))
    for i in range(3):
     for j in range(3):
      a=list(b);a[i]+=1;a[j]+=1;a=tuple(a);m.setdefault(a,np.zeros((3,3)));m[a][i,j]+=amp/4
    for h in range(3):
     a=list(b);a[h]+=2;a=tuple(a);m.setdefault(a,np.zeros((3,3)))
     for i in range(3):m[a][i,i]-=amp/12
 return m
F=fc();M=len(MIS);PT=np.zeros((M,M,3,3));QT=np.zeros((M,M,3))
for ii,nu in enumerate(MIS):
 for jj,la in enumerate(MIS):
  alpha=tuple(nu[k]+la[k] for k in range(3));fm=F.get(alpha)
  if fm is not None:PT[ii,jj]=(-1.)**sum(la)*af(alpha)/(SQ[ii]*SQ[jj])*fm
  sp=SQ[ii]*SQ[jj];qv=np.zeros(3)
  for j in range(3):
   if nu[j]>0:
    aa=list(alpha);aa[j]-=1;aa=tuple(aa);fm=F.get(aa)
    if fm is not None:
     ne=list(nu);ne[j]-=1;ne=tuple(ne);qv+=.5*af(aa)*(-1.)**sum(la)*sp/(mf(ne)*mf(la))*fm[:,j]
   if la[j]>0:
    aa=list(alpha);aa[j]-=1;aa=tuple(aa);fm=F.get(aa)
    if fm is not None:
     le=list(la);le[j]-=1;le=tuple(le);qv+=.5*af(aa)*(-1.)**(sum(la)-1)*sp/(mf(nu)*mf(le))*fm[:,j]
  QT[ii,jj]=qv
def rec(N,W,heat=False):
 ind=np.where(DEG<=N)[0];W=W[np.ix_(ind,ind)]
 return np.einsum('ab,abij->ij',W,PT[np.ix_(ind,ind)],optimize=True) if not heat else np.einsum('ab,abi->i',W,QT[np.ix_(ind,ind)],optimize=True)
s1=pickle.load(open(os.path.join(WORKDIR,'shock_r1.pkl'),'rb'));s2=pickle.load(open(os.path.join(WORKDIR,'shock_r2.pkl'),'rb'))
out={'theta':TH,'samples':[]}
for k,c in enumerate(CELLS):
 A1=np.array(s1['af'][k]);A2=np.array(s2['af'][k]);n1=np.array(s1['ns'][k]);n2=np.array(s2['ns'][k]);m=min(len(A1),len(A2));A1=A1[:m];A2=A2[:m];n1=n1[:m];n2=n2[:m]
 def Wrange(sl):
  X=A1[sl];Y=A2[sl];ww=n1[sl]*n2[sl];W=np.einsum('s,si,sj->ij',ww,X,Y)/len(X);return .5*(W+W.T)
 W=Wrange(slice(None));conv={}
 for N in range(2,MAXH+1):conv[str(N)]={'S':rec(N,TH**1.5*W).tolist(),'H':rec(N,TH**2*W,True).tolist()}
 # block uncertainty from 5 blocks of matched snapshots
 nb=5;bs=m//nb;SB=[];HB=[]
 for b in range(nb):
  sl=slice(b*bs,(b+1)*bs if b<nb-1 else m);wb=Wrange(sl);SB.append(rec(MAXH,TH**1.5*wb));HB.append(rec(MAXH,TH**2*wb,True))
 SB=np.array(SB);HB=np.array(HB)
 # combined event/weak across replicas
 def comb(field,sefield):
  a=np.array([s1[field][k] if field in s1 else 0])
 # derive event arrays
 SAMPLE=5000;DT=3e-4;BLOCK=250;bd=np.minimum(BLOCK,SAMPLE-np.arange(SAMPLE//BLOCK)*BLOCK)*DT
 vals={}
 for tag,st in [('r1',s1),('r2',s2)]:
  Se=st['Se'][k]/(SAMPLE*DT);He=st['He'][k]/(SAMPLE*DT);Sr=st['Sb'][k]/bd[:,None,None];Hr=st['Hb'][k]/bd[:,None]
  vals[tag]={'Se':Se,'He':He,'See':Sr.std(0,ddof=1)/math.sqrt(len(Sr)),'Hee':Hr.std(0,ddof=1)/math.sqrt(len(Hr)),
             'Sw':np.mean(np.array(st['ws'][k]),0),'Hw':np.mean(np.array(st['wh'][k]),0),'Swe':np.std(np.array(st['ws'][k]),0,ddof=1)/math.sqrt(len(st['ws'][k])),'Hwe':np.std(np.array(st['wh'][k]),0,ddof=1)/math.sqrt(len(st['wh'][k]))}
 def cmb(key,ekey):
  a,b=vals['r1'][key],vals['r2'][key];ea,eb=vals['r1'][ekey],vals['r2'][ekey];return .5*(a+b),np.sqrt((ea*ea+eb*eb)/4+((a-b)/2)**2)
 Se,See=cmb('Se','See');He,Hee=cmb('He','Hee');Sw,Swe=cmb('Sw','Swe');Hw,Hwe=cmb('Hw','Hwe')
 out['samples'].append({'x':(c+.5)/50,'event_S':Se.tolist(),'event_S_se':See.tolist(),'event_H':He.tolist(),'event_H_se':Hee.tolist(),'weak_S':Sw.tolist(),'weak_S_se':Swe.tolist(),'weak_H':Hw.tolist(),'weak_H_se':Hwe.tolist(),'conv':conv,'N14_S_se':(SB.std(0,ddof=1)/math.sqrt(nb)).tolist(),'N14_H_se':(HB.std(0,ddof=1)/math.sqrt(nb)).tolist()})
json.dump(out,open(os.path.join(WORKDIR,'shock_M5_crossrep.json'),'w'),indent=2)
print('done')
