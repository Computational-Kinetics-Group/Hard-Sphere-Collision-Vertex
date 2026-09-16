import numpy as np, math, pickle, os, sys, time
from scipy.special import ndtr

state_path=sys.argv[1]; nsteps=int(sys.argv[2])
# constants
GAMMA=5/3; MACH=5.0; T1=1.0; U1=MACH*math.sqrt(GAMMA*T1)
R=((GAMMA+1)*MACH**2)/((GAMMA-1)*MACH**2+2); PR=1+2*GAMMA/(GAMMA+1)*(MACH**2-1)
T2=T1*PR/R; U2=U1/R
L=1.0; DIAM=1.0; SIG=math.pi; LAM1=.05; N1=1/(math.sqrt(2)*SIG*LAM1); N2=N1*R
NC=50; DX=L/NC; TARGET=18000; DT=3e-4; GMAJ=28.; BURN=5000; SAMPLE=5000; TOTAL=BURN+SAMPLE
STRIDE=100; BLOCK=250; MAXH=14; WEAK_PAIRS=1500; THETA=5.5; SAMPLE_CELLS=[18,21,23,24,25,26,28,31]

# multiindices and coeff helper only
MIS=[(a,b,c) for a in range(MAXH+1) for b in range(MAXH+1-a) for c in range(MAXH+1-a-b)]
SQ=np.array([math.sqrt(math.factorial(a)*math.factorial(b)*math.factorial(c)) for a,b,c in MIS])
def hcoeff(x):
    m=len(x); HH=[]
    for j in range(3):
        h=np.empty((MAXH+1,m)); h[0]=1.; h[1]=x[:,j]
        for n in range(1,MAXH): h[n+1]=x[:,j]*h[n]-n*h[n-1]
        HH.append(h)
    out=np.empty(len(MIS))
    for q,(a,b,c) in enumerate(MIS): out[q]=np.mean(HH[0][a]*HH[1][b]*HH[2][c])/SQ[q]
    return out

def flux(n,U,T,side):
    s=math.sqrt(T); a=U/s; ph=math.exp(-a*a/2)/math.sqrt(2*math.pi)
    return n*(U*ndtr(a)+s*ph) if side==1 else n*(s*ph-U*ndtr(-a))
def sampler(U,T,side,rng):
    s=math.sqrt(T); lo,hi=(0.,max(U+9*s,9*s)) if side==1 else (min(U-9*s,-9*s),0.)
    g=np.linspace(lo,hi,16000); p=np.abs(g)*np.exp(-.5*((g-U)/s)**2); c=np.empty_like(g); c[0]=0;c[1:]=np.cumsum(.5*(p[1:]+p[:-1])*np.diff(g));c/=c[-1]
    return lambda k: np.interp(rng.random(k),c,g)

if not os.path.exists(state_path):
    rep=int(os.path.basename(state_path).split('r')[-1].split('.')[0]); rng=np.random.default_rng(2026091400+rep)
    xc=(np.arange(NC)+.5)*DX; s=.5*(1+np.tanh((xc-.5)/.07)); ni=N1+(N2-N1)*s; ui=N1*U1/ni; Ti=T1+(T2-T1)*s
    totalphys=float(np.sum(ni)*DX); W=totalphys/TARGET
    xs=[];vs=[]
    for c in range(NC):
        m=max(1,int(round(ni[c]*DX/W))); xx=(c+rng.random(m))*DX
        vv=np.column_stack([ui[c]+np.sqrt(Ti[c])*rng.normal(size=m),np.sqrt(Ti[c])*rng.normal(size=m),np.sqrt(Ti[c])*rng.normal(size=m)])
        xs.append(xx);vs.append(vv)
    st=dict(rep=rep,step=0,rng=rng.bit_generator.state,x=np.concatenate(xs),v=np.vstack(vs),W=W,
      Se=np.zeros((len(SAMPLE_CELLS),3,3)),He=np.zeros((len(SAMPLE_CELLS),3)),Nev=np.zeros(len(SAMPLE_CELLS),int),
      Sb=np.zeros((len(SAMPLE_CELLS),SAMPLE//BLOCK,3,3)),Hb=np.zeros((len(SAMPLE_CELLS),SAMPLE//BLOCK,3)),
      profn=np.zeros(NC),profu=np.zeros((NC,3)),proft=np.zeros(NC),profc=np.zeros(NC),
      ns=[[] for _ in SAMPLE_CELLS],Ts=[[] for _ in SAMPLE_CELLS],al=[[] for _ in SAMPLE_CELLS],af=[[] for _ in SAMPLE_CELLS],
      ws=[[] for _ in SAMPLE_CELLS],wh=[[] for _ in SAMPLE_CELLS],hist=np.zeros(152),hist_local=[],maxg=0.,hits=0,np_hist=[])
else:
    st=pickle.load(open(state_path,'rb')); rng=np.random.default_rng(); rng.bit_generator.state=st['rng']
W=st['W']; x=st['x'];v=st['v']; left= sampler(U1,T1,1,rng); right=sampler(U2,T2,-1,rng); JL=flux(N1,U1,T1,1); JR=flux(N2,U2,T2,-1)
cellmap={c:k for k,c in enumerate(SAMPLE_CELLS)}; hedges=np.linspace(-6,13,153)
def inject(side):
    if side==1:
        k=rng.poisson(JL*DT/W)
        if not k:return np.empty(0),np.empty((0,3))
        vx=left(k); rem=rng.random(k)*DT; xx=vx*rem; TT=T1
    else:
        k=rng.poisson(JR*DT/W)
        if not k:return np.empty(0),np.empty((0,3))
        vx=right(k); rem=rng.random(k)*DT; xx=L+vx*rem; TT=T2
    vv=np.column_stack([vx,np.sqrt(TT)*rng.normal(size=k),np.sqrt(TT)*rng.normal(size=k)]); return xx,vv
start=time.time(); end=min(TOTAL,st['step']+nsteps)
for step in range(st['step'],end):
    x=x+v[:,0]*DT; keep=(x>=0)&(x<L); x=x[keep];v=v[keep]
    xx,vv=inject(1)
    if len(xx):x=np.concatenate([x,xx]);v=np.vstack([v,vv])
    xx,vv=inject(-1)
    if len(xx):x=np.concatenate([x,xx]);v=np.vstack([v,vv])
    cells=np.minimum((x/DX).astype(int),NC-1)
    if step>=BURN and (step-BURN)%STRIDE==0:
        st['np_hist'].append(len(x))
        for c in range(NC):
            idx=np.where(cells==c)[0]
            if len(idx)<12:continue
            U=v[idx].mean(0);C=v[idx]-U;T=np.mean(np.sum(C*C,1))/3;n=W*len(idx)/DX
            st['profn'][c]+=n;st['profu'][c]+=U;st['proft'][c]+=T;st['profc'][c]+=1
        for kk,c in enumerate(SAMPLE_CELLS):
            idx=np.where(cells==c)[0]
            if len(idx)<30:continue
            U=v[idx].mean(0);C=v[idx]-U;T=np.mean(np.sum(C*C,1))/3;n=W*len(idx)/DX
            st['ns'][kk].append(n);st['Ts'][kk].append(T);st['al'][kk].append(hcoeff(C/np.sqrt(T)));st['af'][kk].append(hcoeff(C/np.sqrt(THETA)))
            mp=WEAK_PAIRS; ia=rng.choice(idx,mp);ib=rng.choice(idx,mp);eq=ia==ib
            while np.any(eq):ib[eq]=rng.choice(idx,np.count_nonzero(eq));eq=ia==ib
            a=v[ia];b=v[ib];g=a-b;gn=np.linalg.norm(g,axis=1);gg=g[:,:,None]*g[:,None,:];tr=np.sum(g*g,1);gstf=gg-np.eye(3)[None,:,:]*tr[:,None,None]/3
            st['ws'][kk].append(-(math.pi/4)*n*n*np.mean(gn[:,None,None]*gstf,0));Vr=.5*(a+b)-U
            st['wh'][kk].append(-(math.pi/4)*n*n*np.mean(gn[:,None]*np.einsum('nj,nij->ni',Vr,gstf),0))
            if c==25:
                st['hist']+=np.histogram(v[idx,0],hedges)[0];st['hist_local'].append((n,U[0],T))
    order=np.argsort(cells);cnt=np.bincount(cells,minlength=NC);starts=np.r_[0,np.cumsum(cnt)]
    for c in range(NC):
        idx=order[starts[c]:starts[c+1]];m=len(idx)
        if m<2:continue
        Uc=v[idx].mean(0); exp=.5*m*(m-1)*W*SIG*GMAJ*DT/DX; nc=min(int(exp+rng.random()),m//2)
        if nc<1:continue
        ch=rng.choice(idx,2*nc,replace=False);ia=ch[:nc];ib=ch[nc:];g=v[ia]-v[ib];gn=np.linalg.norm(g,1 if False else 2,axis=1)
        if len(gn):st['maxg']=max(st['maxg'],float(np.max(gn)));st['hits']+=int(np.count_nonzero(gn>GMAJ))
        acc=rng.random(nc)<np.minimum(gn/GMAJ,1)
        if not np.any(acc):continue
        ia=ia[acc];ib=ib[acc];g=g[acc];gn=gn[acc];V=.5*(v[ia]+v[ib]);sig=rng.normal(size=(len(ia),3));sig/=np.linalg.norm(sig,axis=1)[:,None];gp=gn[:,None]*sig;vap=V+.5*gp;vbp=V-.5*gp
        if step>=BURN and c in cellmap:
            kk=cellmap[c];dP=.5*(gp[:,:,None]*gp[:,None,:]-g[:,:,None]*g[:,None,:]);tr=np.trace(dP,axis1=1,axis2=2);dP-=np.eye(3)[None,:,:]*tr[:,None,None]/3;dQ=np.einsum('nj,nij->ni',V-Uc,dP);scale=W/DX
            st['Se'][kk]+=scale*dP.sum(0);st['He'][kk]+=scale*dQ.sum(0);st['Nev'][kk]+=len(ia);bi=(step-BURN)//BLOCK;st['Sb'][kk,bi]+=scale*dP.sum(0);st['Hb'][kk,bi]+=scale*dQ.sum(0)
        v[ia]=vap;v[ib]=vbp
st['step']=end;st['x']=x;st['v']=v;st['rng']=rng.bit_generator.state
pickle.dump(st,open(state_path,'wb'),protocol=pickle.HIGHEST_PROTOCOL)
print('advanced',end,'/',TOTAL,'NP',len(x),'elapsed',round(time.time()-start,2),'maxg',round(st['maxg'],2),'hits',st['hits'])
