#!/usr/bin/env python3
"""Extended collision-level DSMC validation for the hard-sphere stress/heat vertex.

The code samples two unit-temperature displaced Maxwellians, performs a Bird/Nanbu-style
hard-sphere collision selection with a batchwise relative-speed majorant, scatters accepted
pairs isotropically, and estimates the collisional stress and heat-flux increments.

Normalization matches the manuscript convention kappa=1:
    S_ij = (pi/2) E[ |g| Delta Pi_ij ],
    H_i  = (pi/2) E[ |g| Delta Q_i ].
"""

import numpy as np
import pandas as pd
from scipy.special import hyp1f1


def xi_exact(chi, kappa=1.0):
    return -(32.0/5.0)*kappa*np.sqrt(np.pi)*hyp1f1(-0.5, 3.5, -chi)


def simulate(chi, U, nhat, batches=40, candidates_per_batch=50_000,
             seed=0, kappa=1.0):
    rng = np.random.default_rng(seed)
    U = np.asarray(U, float)
    nhat = np.asarray(nhat, float)
    nhat = nhat/np.linalg.norm(nhat)
    d = np.sqrt(chi)*nhat
    z = U + d
    u = U - d

    D = np.outer(d, d) - np.eye(3)*(chi/3.0)
    DD = np.sum(D*D)
    xiex = xi_exact(chi, kappa)
    Sex = xiex*D
    Hex = Sex @ U

    batch_xi, batch_S, batch_H, acc = [], [], [], []

    for _ in range(batches):
        m = candidates_per_batch
        v = z + rng.normal(size=(m, 3))
        vs = u + rng.normal(size=(m, 3))
        g = v - vs
        gn = np.linalg.norm(g, axis=1)

        # No-time-counter style thinning. Conditional on the sampled candidate batch,
        # gmax * sum_{accepted}(...) / m is an unbiased estimator of
        # (1/m) sum |g|(...).
        gmax = gn.max()
        mask = rng.random(m) < gn/gmax
        ga = g[mask]
        gna = gn[mask]
        V = (v[mask] + vs[mask])/2.0

        # Elastic hard-sphere scattering: post-collision relative direction is isotropic.
        sigma = rng.normal(size=(len(ga), 3))
        sigma /= np.linalg.norm(sigma, axis=1)[:, None]
        gp = gna[:, None]*sigma

        # Pair changes of traceless second moment and heat-flux moment.
        dP = 0.5*(gp[:, :, None]*gp[:, None, :] - ga[:, :, None]*ga[:, None, :])
        tr = np.trace(dP, axis1=1, axis2=2)
        dP -= np.eye(3)[None, :, :]*(tr[:, None, None]/3.0)
        dQ = np.einsum('nj,nij->ni', V, dP)

        S = (kappa*np.pi/2.0)*gmax*dP.sum(axis=0)/m
        H = (kappa*np.pi/2.0)*gmax*dQ.sum(axis=0)/m

        batch_S.append(S)
        batch_H.append(H)
        batch_xi.append(np.sum(S*D)/DD)
        acc.append(mask.mean())

    batch_S = np.asarray(batch_S)
    batch_H = np.asarray(batch_H)
    batch_xi = np.asarray(batch_xi)

    S = batch_S.mean(axis=0)
    H = batch_H.mean(axis=0)
    xi = batch_xi.mean()
    xi_se = batch_xi.std(ddof=1)/np.sqrt(batches)

    if np.linalg.norm(Hex) > 0:
        ratio = np.einsum('bi,i->b', batch_H, Hex)/np.dot(Hex, Hex)
        ratio_mean = ratio.mean()
        ratio_se = ratio.std(ddof=1)/np.sqrt(batches)
        Hpred = np.einsum('bij,j->bi', batch_S, U)
        residual = np.einsum('bi,i->b', batch_H-Hpred, Hex)/np.dot(Hex, Hex)
        residual_mean = residual.mean()
        residual_se = residual.std(ddof=1)/np.sqrt(batches)
    else:
        ratio_mean = ratio_se = residual_mean = residual_se = np.nan

    return {
        'chi': chi,
        'xi_exact': xiex,
        'xi_dsmc': xi,
        'xi_se': xi_se,
        'heat_ratio': ratio_mean,
        'heat_ratio_se': ratio_se,
        'contraction_residual': residual_mean,
        'contraction_residual_se': residual_se,
        'stress_tensor_rel_error': np.linalg.norm(S-Sex)/np.linalg.norm(Sex),
        'heat_vector_rel_error': (np.linalg.norm(H-Hex)/np.linalg.norm(Hex)
                                  if np.linalg.norm(Hex) > 0 else np.nan),
        'acceptance': float(np.mean(acc)),
        'candidate_pairs': batches*candidates_per_batch,
        'accepted_collisions_est': float(np.mean(acc)*batches*candidates_per_batch),
    }


def main():
    nhat = np.array([1.0, 2.0, -1.0])/np.sqrt(6.0)
    Uboost = np.array([2.0, -1.0, 1.5])
    chis = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    rows = []
    for k, chi in enumerate(chis):
        # More samples at the smallest drift because the anisotropic signal is weakest there.
        b_boost = 100 if chi == 0.25 else 40
        boosted = simulate(chi, Uboost, nhat, batches=b_boost,
                           candidates_per_batch=50_000,
                           seed=(20261234 if chi == 0.25 else 20261109+100*k))
        unboosted = simulate(chi, np.zeros(3), nhat, batches=40,
                             candidates_per_batch=50_000,
                             seed=20261009+100*k)
        rows.append({
            'chi': chi,
            'xi_exact': boosted['xi_exact'],
            'xi_dsmc_unboosted': unboosted['xi_dsmc'],
            'se_unboosted': unboosted['xi_se'],
            'xi_dsmc_boosted': boosted['xi_dsmc'],
            'se_boosted': boosted['xi_se'],
            'heat_ratio_exact': boosted['heat_ratio'],
            'se_heat_ratio': boosted['heat_ratio_se'],
            'contraction_residual': boosted['contraction_residual'],
            'se_contraction_residual': boosted['contraction_residual_se'],
            'stress_tensor_rel_error_boosted': boosted['stress_tensor_rel_error'],
            'heat_vector_rel_error_boosted': boosted['heat_vector_rel_error'],
            'acceptance_boosted': boosted['acceptance'],
            'candidate_pairs_boosted': boosted['candidate_pairs'],
            'accepted_collisions_est_boosted': boosted['accepted_collisions_est'],
        })

    df = pd.DataFrame(rows)
    df.to_csv('extended_dsmc_validation_results.csv', index=False)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
