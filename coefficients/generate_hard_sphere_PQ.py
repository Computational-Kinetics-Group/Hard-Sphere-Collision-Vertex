#!/usr/bin/env python3
"""Generate exact normalized hard-sphere Hermite production tensors P and Q.

Normalization:
  S_ij(z,u)=V(r^2) r_<i r_j>, r=z-u,
  H_i(z,u)=1/2 (z_j+u_j) S_ij(z,u),
  V(s)=sum_m c_m s^m,
  c_0=-8/5,
  c_{m+1}=-c_m (2m-1)/[4(m+1)(2m+7)].

The output coefficients are divided by Gamma_HS=kappa*sqrt(pi).
Hermite basis: He_nu/sqrt(nu!).  Only one representative of the incoming
exchange symmetry (nu,lambda)<->(lambda,nu) is written.
"""
import argparse, csv, itertools, math
from fractions import Fraction

COMP={(0,0):'xx',(0,1):'xy',(0,2):'xz',(1,1):'yy',(1,2):'yz',(2,2):'zz'}

def scalar_coeffs(maxm):
    c=[Fraction(-8,5)]
    for m in range(maxm):
        c.append(-c[-1]*Fraction(2*m-1,4*(m+1)*(2*m+7)))
    return c

def comp3(m):
    for a in range(m+1):
        for b in range(m+1-a):
            yield a,b,m-a-b

def multinomial3(m,a,b,c):
    return math.factorial(m)//(math.factorial(a)*math.factorial(b)*math.factorial(c))

def rpoly(m,i,j,c):
    base={(2*a,2*b,2*d):Fraction(multinomial3(m,a,b,d),1) for a,b,d in comp3(m)}
    R={}
    if i!=j:
        e=[0,0,0]; e[i]+=1; e[j]+=1; R[tuple(e)]=Fraction(1)
    else:
        for k in range(3):
            e=[0,0,0]; e[k]=2; R[tuple(e)]=Fraction(2,3) if k==i else Fraction(-1,3)
    out={}
    for a,ca in base.items():
        for b,cb in R.items():
            q=tuple(a[k]+b[k] for k in range(3))
            out[q]=out.get(q,Fraction(0))+c*ca*cb
    return out

def splits(alpha):
    for nu in itertools.product(*[range(a+1) for a in alpha]):
        lam=tuple(alpha[k]-nu[k] for k in range(3))
        yield tuple(nu),lam

def multifac(a):
    p=1
    for n in a: p*=math.factorial(n)
    return p

def square_decompose(n):
    rem=n; outside=1; inside=1; p=2
    while p*p<=rem:
        e=0
        while rem%p==0:
            rem//=p; e+=1
        outside*=p**(e//2)
        if e%2: inside*=p
        p+=1
    if rem>1: inside*=rem
    return outside,inside

def algebraic(q,root_n):
    outside,inside=square_decompose(root_n); q*=outside
    sign='-' if q<0 else ''; q=abs(q); num=q.numerator; den=q.denominator
    if inside==1: core=str(num)
    elif num==1: core=f'sqrt({inside})'
    else: core=f'{num}*sqrt({inside})'
    return sign+core+(f'/{den}' if den!=1 else '')

def generate(max_stress_degree):
    maxm=(max_stress_degree-2)//2; cs=scalar_coeffs(maxm)
    P={}
    for m,c in enumerate(cs):
        D=2*m+2
        if D>max_stress_degree: break
        for i in range(3):
            for j in range(3):
                for alpha,cr in rpoly(m,i,j,c).items():
                    for nu,lam in splits(alpha):
                        b=1
                        for a,n in zip(alpha,nu): b*=math.comb(a,n)
                        mon=cr*b*((-1)**sum(lam))
                        P[(i,j,nu,lam)]=P.get((i,j,nu,lam),Fraction(0))+mon
    Q={}
    for (i,j,nu,lam),mon in P.items():
        if not mon: continue
        a=list(nu); a[j]+=1; a=tuple(a)
        Q[(i,a,lam)]=Q.get((i,a,lam),Fraction(0))+mon*Fraction(1,2)
        b=list(lam); b[j]+=1; b=tuple(b)
        Q[(i,nu,b)]=Q.get((i,nu,b),Fraction(0))+mon*Fraction(1,2)
    return P,Q

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--max-stress-degree',type=int,default=14)
    ap.add_argument('--output',default='hard_sphere_PQ.csv')
    args=ap.parse_args()
    P,Q=generate(args.max_stress_degree)
    rows=[]
    for (i,j),name in COMP.items():
        for (ii,jj,nu,lam),mon in P.items():
            if (ii,jj)!=(i,j) or not mon or nu>lam: continue
            rows.append(['P',name,sum(nu)+sum(lam),*nu,*lam,algebraic(mon,multifac(nu)*multifac(lam))])
    for i,name in enumerate('xyz'):
        for (ii,nu,lam),mon in Q.items():
            if ii!=i or not mon or nu>lam: continue
            rows.append(['Q',name,sum(nu)+sum(lam),*nu,*lam,algebraic(mon,multifac(nu)*multifac(lam))])
    rows.sort(key=lambda x:(x[0],x[1],x[2],*x[3:9]))
    header=['sector','component','total_degree','nu_x','nu_y','nu_z','lambda_x','lambda_y','lambda_z','value_over_kappa_sqrtpi']
    with open(args.output,'w',newline='') as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f'wrote {len(rows)} independent entries to {args.output}')

if __name__=='__main__': main()
