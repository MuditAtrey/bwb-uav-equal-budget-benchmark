## Equal-budget controls (287 evaluations after the 9,800-sample pool; range on 160 Wh/kg basis)
| arm | seeds | best (mean) | sd | valid rate | median valid |
|---|---|---|---|---|---|
| lhs | 3 | 215.8 | 9.6 | 0.28 | 95.4 |
| uniform | 3 | 232.6 | 9.5 | 0.28 | 95.7 |
| local5 | 3 | 286.3 | 3.8 | 0.81 | 245.9 |
| local10 | 10 | 287.4 | 3.7 | 0.64 | 243.1 |
Reference: Phase 1 best 277.8 km; paper AL 298.7 km
AL replication al_seed0.csv: n=287 best=310.4 valid=0.68; first 60 evals valid=0.15
AL replication al_seed1.csv: n=287 best=301.1 valid=0.67; first 60 evals valid=0.27
AL replication al_seed2.csv: n=117 best=292.0 valid=0.15; first 60 evals valid=0.02
AL replication al_seed3.csv: n=287 best=304.2 valid=0.59; first 60 evals valid=0.17
AL replication al_seed4.csv: n=287 best=294.4 valid=0.70; first 60 evals valid=0.02
AL replication al_seed5.csv: n=287 best=312.8 valid=0.66; first 60 evals valid=0.25
AL replication al_seed6.csv: n=287 best=311.2 valid=0.69; first 60 evals valid=0.20
AL replication al_seed7.csv: n=287 best=307.8 valid=0.70; first 60 evals valid=0.18
AL replication al_seed8.csv: n=287 best=290.3 valid=0.49; first 60 evals valid=0.12
AL replication al_seed9.csv: n=287 best=309.8 valid=0.64; first 60 evals valid=0.07

## Manifold vs no clamp (widened c_blend 0.15-0.65)
clamp best per seed [np.float64(305.2), np.float64(277.6), np.float64(317.0), np.float64(308.7), np.float64(295.5), np.float64(275.3), np.float64(315.2), np.float64(206.1), np.float64(279.7), np.float64(211.8)] mean 279.2 valid 0.55 inverted share of samples 0.07
noclamp best per seed [np.float64(253.3), np.float64(247.2), np.float64(304.2), np.float64(328.6), np.float64(311.7), np.float64(297.9), np.float64(272.4), np.float64(236.8), np.float64(287.7), np.float64(215.1)] mean 275.5 valid 0.47 inverted share of samples 0.07

## Paper bounds vs relaxed champion-active bounds
paper best per seed [np.float64(276.1), np.float64(276.2), np.float64(286.4), np.float64(295.0), np.float64(287.8), np.float64(278.0), np.float64(289.2), np.float64(279.1), np.float64(294.8), np.float64(281.0)] mean 284.3 clamp active share 0.0
wide best per seed [np.float64(357.7), np.float64(370.5), np.float64(352.0), np.float64(350.2), np.float64(350.5), np.float64(373.0), np.float64(368.2), np.float64(288.2), np.float64(337.9), np.float64(256.6)] mean 340.5 clamp active share 0.031

## Cross-solver (AeroBuildup vs pipeline), n = 72  Spearman pooled = 0.977  top-only = 0.912  median AB/pipeline = 1.46
