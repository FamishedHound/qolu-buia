# QOLU return-distribution diagnostics across scenarios and seeds

Generated 2026-10-02T22:26:00+00:00.  Tag `final`.

Method `qolu_buia`, window `E495_N165`, configuration `50x50_f1.0_cap1`, reward reading `team`, 3 stakeholders, 1000 training iterations, 200 evaluation iterations, scenarios [1, 2, 3, 4], seeds [0, 1, 2].

12 of 12 cells finished.  Every number below is scored against `tail_return`; the other two realised-return readings are in the per-cell JSON beside this file, and are never mixed with this one.

## 1. Per cell

| Scenario | Seed | CRPS skill vs clim. | vs init. | PIT KS | cov 50 % | cov 90 % | MAE | RMSE | boundary mass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | -8.0158 | 0.4380 | 0.8217 | 0.0121 | 0.5370 | 88.4292 | 89.1430 | 0.0004 |
| 1 | 1 | -5.7162 | 0.4774 | 0.7688 | 0.0139 | 0.6306 | 80.5507 | 81.6396 | 0.0003 |
| 1 | 2 | -7.1406 | 0.4188 | 0.8116 | 0.0187 | 0.5273 | 91.4349 | 92.3438 | 0.0004 |
| 2 | 0 | -8.5740 | 0.4046 | 0.8356 | 0.0120 | 0.4522 | 90.4261 | 91.1173 | 0.0003 |
| 2 | 1 | -5.3049 | 0.4871 | 0.7491 | 0.0144 | 0.7774 | 77.7601 | 78.8695 | 0.0003 |
| 2 | 2 | -7.5952 | 0.4052 | 0.8093 | 0.0157 | 0.4895 | 90.4517 | 91.2713 | 0.0004 |
| 3 | 0 | -6.5936 | 0.4085 | 0.8105 | 0.0127 | 0.6005 | 90.4697 | 91.4435 | 0.0004 |
| 3 | 1 | -4.5820 | 0.4837 | 0.7345 | 0.0151 | 0.6849 | 78.5295 | 79.9057 | 0.0003 |
| 3 | 2 | -5.8242 | 0.3862 | 0.8015 | 0.0192 | 0.3954 | 93.9866 | 95.2429 | 0.0004 |
| 4 | 0 | -10.0429 | 0.4038 | 0.8652 | 0.0115 | 0.1601 | 96.3487 | 96.9834 | 0.0004 |
| 4 | 1 | -6.8902 | 0.4732 | 0.7942 | 0.0141 | 0.4316 | 85.6432 | 86.4778 | 0.0005 |
| 4 | 2 | -8.6541 | 0.4117 | 0.8565 | 0.0142 | 0.2031 | 94.7131 | 95.4916 | 0.0004 |

## 2. Aggregated over every cell

| Statistic | mean ± SE | SD | n |
| --- | --- | --- | --- |
| appraisals scored | 30000.0000 ± 0.0000 | 0.0000 | 12 |
| independent observations (effective n) | 30000.0000 ± 0.0000 | 0.0000 | 12 |
| CRPS skill vs climatology | -7.0778 ± 0.4584 | 1.5878 | 12 |
| CRPS skill vs initialisation | 0.4332 ± 0.0107 | 0.0369 | 12 |
| CRPS | 53.8607 ± 1.3831 | 4.7913 | 12 |
| CRPS of climatology | 6.8249 ± 0.2806 | 0.9721 | 12 |
| PIT KS statistic | 0.8049 ± 0.0114 | 0.0395 | 12 |
| PIT KS p-value | 0.0000 ± 0.0000 | 0.0000 | 12 |
| coverage of the central 50 % interval | 0.0145 ± 0.0007 | 0.0025 | 12 |
| mean width of the central 50 % interval | 93.2997 ± 1.2755 | 4.4183 | 12 |
| coverage of the central 90 % interval | 0.4908 ± 0.0523 | 0.1812 | 12 |
| mean width of the central 90 % interval | 205.8258 ± 1.8023 | 6.2434 | 12 |
| MAE of E[Z] | 88.2286 ± 1.8165 | 6.2925 | 12 |
| RMSE of E[Z] | 89.1608 ± 1.7779 | 6.1587 | 12 |
| bias of E[Z] (predicted - realised) | -87.8150 ± 1.8350 | 6.3565 | 12 |
| SD of the realised return | 15.1322 ± 0.3796 | 1.3150 | 12 |
| boundary-atom mass | 0.0004 ± 0.0000 | 0.0000 | 12 |
| realised returns outside the support | 0.0000 ± 0.0000 | 0.0000 | 12 |
| SD of the realised return, in atoms | 1.8915 ± 0.0475 | 0.1644 | 12 |
| atoms the realised returns land on | 23.5833 ± 0.1486 | 0.5149 | 12 |

The spread is across scenarios *and* seeds, so it is the spread a chapter claim has to survive, not a within-run standard error.

## 2b. The same statistics against each realised-return reading

| Realised return | CRPS skill vs climatology | CRPS skill vs initialisation | PIT KS statistic | coverage of the central 50 % interval | coverage of the central 90 % interval | MAE of E[Z] | RMSE of E[Z] | SD of the realised return | SD of the realised return, in atoms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n_step_reward_sum | -14.9924 ± 0.8149 | -0.2271 ± 0.0278 | 0.7880 ± 0.0054 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 64.4671 ± 1.3054 | 64.5094 ± 1.3137 | 0.5220 ± 0.0368 | 0.0653 ± 0.0046 |
| n_step_target | -7.8356 ± 0.9442 | 0.6470 ± 0.0069 | 0.4887 ± 0.0040 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.6507 ± 0.0511 | 1.9048 ± 0.0605 | 2.1847 ± 0.3335 | 0.2731 ± 0.0417 |
| tail_return | -7.0778 ± 0.4584 | 0.4332 ± 0.0107 | 0.8049 ± 0.0114 | 0.0145 ± 0.0007 | 0.4908 ± 0.0523 | 88.2286 ± 1.8165 | 89.1608 ± 1.7779 | 15.1322 ± 0.3796 | 1.8915 ± 0.0475 |

`n_step_target` is the only one of the three the network is literally regressed onto, so read its row as the fit to the training target and `tail_return` as the fit to the long-run return.  The three live on different scales and are never mixed.

## 3. By scenario

| Scenario | CRPS skill vs climatology | PIT KS statistic | coverage of the central 90 % interval | MAE of E[Z] | RMSE of E[Z] |
| --- | --- | --- | --- | --- | --- |
| 1 | -6.9575 ± 0.6701 | 0.8007 ± 0.0162 | 0.5650 ± 0.0329 | 86.8049 ± 3.2453 | 87.7088 ± 3.1722 |
| 2 | -7.1580 ± 0.9687 | 0.7980 ± 0.0256 | 0.5730 ± 0.1027 | 86.2126 ± 4.2263 | 87.0860 ± 4.1085 |
| 3 | -5.6666 ± 0.5860 | 0.7822 ± 0.0240 | 0.5603 ± 0.0859 | 87.6619 ± 4.6777 | 88.8640 ± 4.6115 |
| 4 | -8.5290 ± 0.9122 | 0.8386 ± 0.0223 | 0.2649 ± 0.0842 | 92.2350 ± 3.3295 | 92.9842 ± 3.2816 |

## 4. By stakeholder objective

| Objective | CRPS skill vs climatology | CRPS skill vs initialisation | PIT KS statistic | coverage of the central 50 % interval | coverage of the central 90 % interval | MAE of E[Z] | RMSE of E[Z] | boundary-atom mass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| land_use | -7.1401 ± 0.4314 | 0.4289 ± 0.0085 | 0.8062 ± 0.0116 | 0.0142 ± 0.0007 | 0.4789 ± 0.0580 | 89.2765 ± 1.2082 | 90.1731 ± 1.1858 | 0.0004 ± 0.0000 |
| urban | -7.0298 ± 0.4771 | 0.4328 ± 0.0127 | 0.8084 ± 0.0114 | 0.0146 ± 0.0009 | 0.5400 ± 0.0598 | 87.5829 ± 2.4120 | 88.4870 ± 2.3833 | 0.0004 ± 0.0000 |
| woodland | -7.0635 ± 0.4724 | 0.4377 ± 0.0121 | 0.8048 ± 0.0116 | 0.0146 ± 0.0008 | 0.4535 ± 0.0574 | 87.8265 ± 1.9737 | 88.7278 ± 1.9433 | 0.0003 ± 0.0000 |

## 5. Verdict

Across 12 cell(s) scored against `tail_return`, CRPS skill against climatology averages -7.0778 (SD 1.5878, SE 0.4584); no cell beat(s) climatology, so no cell's distribution carries more information than the run's own marginal histogram of returns.  CRPS skill against the network at initialisation averages +0.4332 (SD 0.0369).  0 of 12 cell(s) have a randomised PIT consistent with uniformity, 12 of 12 hold the distribution inside the atom support, and 12 of 12 have an atom spacing fine enough to express the realised variation.

The KS p-value is reported as a descriptive threshold, not as a significance test: an evaluation rollout's appraisals are not independent (50 share a state within an iteration, and consecutive iterations share a landscape), so the effective sample size is far below the nominal one and a p-value computed as if the sample were i.i.d. is anti-conservative at these sizes.  Read the KS statistic, which is an effect size, rather than the p-value beside it.

| Question | Cells |
| --- | --- |
| Is `Z` state-conditional (positive **CRPS skill** vs climatology)? | 0 of 12 |
| Is `Z` calibrated (randomised PIT consistent with uniform)? | 0 of 12 |
| Is the atom support adequate (low **boundary**-atom mass)? | 12 of 12 |
| Is the atom spacing fine enough to express the signal? | 12 of 12 |

The policy row is not repeated here: the decision path reads `Z` only through `E[Z]` by construction, which is a property of the code and not of any run.  See any cell's own `distribution_diagnostics.md`, section 1.

