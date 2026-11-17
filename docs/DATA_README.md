# Data and code: Equal-Budget Benchmark of Surrogate-Assisted Search Strategies for Conceptual Design of a Tailless Blended-Wing-Body UAV with an Euler-Calibrated Stability Constraint

Author: Mudit Atrey. This version supersedes the earlier versions of this record, whose manuscript ("Physics-Regularized Geometric Manifolds...") made claims that the controlled experiments reported here do not support. Nothing is withheld in this version.

## Contents
- `data/original_campaign/database.csv`: all 10,087 evaluations of the original campaign (9,800-design Latin hypercube pool and 287 active-learning evaluations), with full 14-variable design vectors. Ranges are on the 160 Wh/kg database basis; multiply by 1.2706 for the 203.3 Wh/kg basis of the paper.
- `data/results_revision/`: first-revision experiments (active-learning replications, controls, clamp and bound studies, grid convergence, battery and mass sensitivity, five-design Euler CFD, NeuralFoil check).
- `data/results_v3/`: benchmark and audit runs of the current manuscript, one CSV per run with every evaluation in order and the full design vector. Main folders: `turbo_fix`, `turbo_fix_extra` (TuRBO), `al_repair`, `al_repair_extra` (AL-R), `adapt_fix` (adaptive local search), `al_hard`, `al_nofeas`, `al_repair_nofeas`, `turbo_feas_fix` (classifier-weight ablation), `bound_prices_fix`, `pool_size_v2`, `indep_pool`, `long_budget`, `cfd`, `cfd_oos`, `cfd_calibration`, `robustness`, `v4_summary`. Folders without `_fix` or `_v2` are first implementations or superseded layouts, kept for transparency (see the manuscript, Supplementary Table "first implementations", and `pool_size_v2/README.txt`).
- `code/flying_wing_optimizer/`: the evaluator (geometry, vortex-lattice aerodynamics, mass and mission model) used for every evaluation.
- `code/experiments/`: the strategy runners (TuRBO, active learning, adaptive and static local search, pool generation) and analysis scripts; `launchers/` holds the shell scripts that ran each batch.
- `manuscript/`: LaTeX sources of the paper and the supplement, every table, the figures (PDF and editable SVG), and the scripts that generate all numbers, tables and figures from the run files.
- `cfd_solution_files.zip` (separate file): SU2 restart files of every Euler case.

## Reproducing the numbers
Energy basis: the released evaluator (`code/flying_wing_optimizer/config.py`) uses a usable energy density of 208.25 Wh/kg, a value set before the battery-pack derivation of the paper; the battery comments in `config.py` are superseded by Supplementary Section S10. Ranges returned by the evaluator are converted to the 203.3 Wh/kg basis of the paper by the factor 203.3/208.25, and ranges stored in the database and run histories (160 Wh/kg basis) by 203.3/160. With that conversion, re-evaluating the stored design vector of the best design of the original campaign reproduces the stored range within five parts per million.

Python 3 with numpy, pandas, scipy, scikit-learn, matplotlib and aerosandbox. The generator scripts expect the repository layout of the author's machine; set the paths at the top of each script. Order: make_numbers.py, make_paper_figures.py, make_fig_planform.py, make_numbers_v4.py, make_numbers_pool.py, make_figures_v5.py, make_numbers_long.py, make_figures_bench.py; then compile supplement.tex and main_revised.tex twice each.

## Licence
Manuscript, data and figures: CC BY 4.0. Code: MIT.
