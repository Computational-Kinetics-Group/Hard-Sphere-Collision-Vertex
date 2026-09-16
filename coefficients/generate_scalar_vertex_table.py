#!/usr/bin/env python3
"""Generate the scalar radial-series coefficients used by the collision vertex.

V(s) = sum_m c_m s^m,
 c_0 = -8/5,
 c_{m+1} = -c_m (2m-1) / [4(m+1)(2m+7)].

The associated exact total incoming Hermite degrees are 2m+2 for stress and
2m+3 for heat flux.
"""
import argparse, csv
from fractions import Fraction


def scalar_coeffs(max_m):
    c = [Fraction(-8, 5)]
    for m in range(max_m):
        c.append(-c[-1] * Fraction(2*m - 1, 4*(m+1)*(2*m+7)))
    return c


def fmt(q):
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-m', type=int, default=6)
    ap.add_argument('--output', default='hard_sphere_scalar_vertex_coefficients.csv')
    args = ap.parse_args()
    with open(args.output, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['m','stress_total_degree','heat_total_degree','c_m_in_V_s'])
        for m, c in enumerate(scalar_coeffs(args.max_m)):
            w.writerow([m, 2*m+2, 2*m+3, fmt(c)])
    print(f'wrote {args.max_m+1} rows to {args.output}')


if __name__ == '__main__':
    main()
