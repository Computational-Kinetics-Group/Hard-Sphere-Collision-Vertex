# Data provenance

This file distinguishes regenerated data products from frozen publication data.

## Analytic coefficient data

- `coefficients/hard_sphere_PQ_coefficients_through_degree15.csv` — generated exactly by `generate_hard_sphere_PQ.py` with `--max-stress-degree 14`.
- `coefficients/hard_sphere_scalar_vertex_coefficients.csv` — generated exactly by `generate_scalar_vertex_table.py` with `--max-m 6`.

## Wall-driven DSMC

- `validation/wall_driven/data/campaign_*_r*.json` — eight retained raw final-campaign outputs: four wall-driven cases, two replicas each.
- `natural_dsmc_campaign_aggregate.json`, `natural_dsmc_campaign_summary.csv`, and `natural_dsmc_campaign_metrics.csv` — regenerated from those eight JSON files by `aggregate_wall_campaign.py`.

The replica combination used for event/weak-form uncertainties is

\[
\sigma_{12}=\sqrt{(\sigma_1^2+\sigma_2^2)/4+[(x_1-x_2)/2]^2}.
\]

The Hermite replica spread in the flat summary is \(|x_1-x_2|/2\). The normalized RMS metrics use the full stress tensor or full heat vector, not only the components displayed in a particular plot.

## Baseline Mach-5 shock

- `shock_M5_aggregate.json` — retained baseline post-processed replica data.
- `shock_M5_crossrep.json` — retained independent-replica Hermite products and combined event/weak-form measurements.
- `shock_M5_production_summary.csv` and `shock_M5_convergence.csv` — regenerated exactly from `shock_M5_crossrep.json` by `export_baseline_tables.py`.

The original shock scripts were retained. In the repository copies, only the file-path handling in `shock_postprocess.py` and `shock_crossrep.py` was made portable by adding `--workdir`.

## Extended and derived Mach-5 shock studies

The following are frozen final publication data products:

- `sampling_scaling_convergence.csv`
- `sampling_block_errors.csv`
- `shock_total_degree_convergence.csv`
- `shock_center_heat_degree_sectors.csv`
- `shock_center_distribution_revised.csv`
- `shock_frame_production_convergence.csv`
- `shock_frame_distribution_convergence.csv`
- `shock_frame_study_summary.csv`

They support the final Supplemental comparisons (sampling floor, exact total-degree grading, shock-center degree sectors, and Gaussian-reference study). Their large intermediate simulation/checkpoint files are not bundled.

## Controlled coherent-leg DSMC

- `validation/engineered/dsmc_extended_validation.py` — retained collision-level validation script.
- `extended_dsmc_validation_results.csv` — its retained output table.
