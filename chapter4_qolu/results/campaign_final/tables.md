# Campaign `final` - result tables

* **generated**: 2026-10-03T00:18:29+00:00
* **raster**: lcm2015_gb_1km_dominant_aggregate_class.img
* **iterations**: 1000
* **methods**: ['no_regulator', 'ltos', 'pmaddpg', 'mitm_learned', 'qolu_buia']
* **scenarios**: [1, 2, 3, 4]
* **seeds**: [0, 1, 2, 3, 4]
* **runs aggregated**: 100

Three readings of the same five metrics are reported and never silently mixed:

* **(a) RAW** - RAW metric values as chapter4_qolu/env/evaluation.py defines them; distances are normalised by the grid diagonal.
* **(b) absolute-normalised** - absolute-normalised: the same metrics with RewardConfig.max_distance set to a physical distance, so proximity saturates at a fixed number of kilometres instead of at the grid diagonal.
* **(c) cross-method min-max** - cross-method min-max within one configuration x scenario x metric cell (worst method 0, best method 1), rescaled from the RAW view; the transparent analogue of the published tables' 'normalized between 0 and 1'.

Cells are `mean ± SE` over the campaign's seeds. **Bold** marks the best method in a column. `*` marks a baseline whose paired t-test against QOLU+BUIA (paired over seeds) has p < 0.05 on the view of the table it appears in; the min-max tables use the RAW-view test, which a per-cell affine rescaling leaves unchanged.  The RAW and absolute tables add the corrected conversion metric C_align (lower is better) when the records carry it.

Configurations are compared with the published ordering; every configuration is listed. See `pattern_check.md` for the ranking comparison against the published Tables 4-7.

## Main tables - view (c): cross-method min-max normalisation

### Configuration `50x50_f1.0_cap1`, Scenario 1 (land use 0.33, woodland 0.33, urban 0.33)

| Method | Land Use | Woodland | Urban |
| --- | --- | --- | --- |
| No regulator | **1.000 ± 0.008** | 0.181 ± 0.017* | 0.030 ± 0.052* |
| LToS | 0.082 ± 0.383 | 0.025 ± 0.195* | 0.405 ± 0.416 |
| P-MADDPG | 0.113 ± 0.059 | 0.000 ± 0.021* | **1.000 ± 0.148** |
| MitM (learned) | 0.000 ± 0.037 | 0.422 ± 0.000* | 0.000 ± 0.065* |
| QOLU+BUIA | 0.566 ± 0.279 | **1.000 ± 0.169** | 1.000 ± 0.197 |

*Table C1 - Scenario 1, configuration 50x50_f1.0_cap1 (50x50 grid, conversion fraction 1.0, cap 1): view (c) cross-method min-max: within each column the worst method scores 0 and the best 1, rescaled from the RAW view. Cells are mean ± SE over 5 seed(s); * marks p < 0.05 in a paired t-test against QOLU+BUIA on the RAW view, of which this view is a per-cell affine rescaling that leaves the test unchanged; bold marks the best method in the column.*

### Configuration `50x50_f1.0_cap1`, Scenario 2 (land use 0.50, woodland 0.25, urban 0.25)

| Method | Land Use | Woodland | Urban |
| --- | --- | --- | --- |
| No regulator | 0.658 ± 0.006 | 0.206 ± 0.021* | 0.396 ± 0.035 |
| LToS | 0.006 ± 0.288 | 0.008 ± 0.235* | 0.635 ± 0.268 |
| P-MADDPG | 0.000 ± 0.056* | 0.000 ± 0.021* | **1.000 ± 0.037** |
| MitM (learned) | **1.000 ± 0.008** | 0.489 ± 0.002 | 0.000 ± 0.015 |
| QOLU+BUIA | 0.800 ± 0.251 | **1.000 ± 0.195** | 0.819 ± 0.396 |

*Table C2 - Scenario 2, configuration 50x50_f1.0_cap1 (50x50 grid, conversion fraction 1.0, cap 1): view (c) cross-method min-max: within each column the worst method scores 0 and the best 1, rescaled from the RAW view. Cells are mean ± SE over 5 seed(s); * marks p < 0.05 in a paired t-test against QOLU+BUIA on the RAW view, of which this view is a per-cell affine rescaling that leaves the test unchanged; bold marks the best method in the column.*

### Configuration `50x50_f1.0_cap1`, Scenario 3 (land use 0.25, woodland 0.50, urban 0.25)

| Method | Land Use | Woodland | Urban |
| --- | --- | --- | --- |
| No regulator | **1.000 ± 0.009** | 0.180 ± 0.018* | 0.000 ± 0.054 |
| LToS | 0.080 ± 0.410 | 0.072 ± 0.218* | 0.411 ± 0.261 |
| P-MADDPG | 0.036 ± 0.063 | 0.000 ± 0.014* | **1.000 ± 0.145** |
| MitM (learned) | 0.000 ± 0.028 | 0.430 ± 0.000* | 0.115 ± 0.086 |
| QOLU+BUIA | 0.715 ± 0.309 | **1.000 ± 0.169** | 0.781 ± 0.361 |

*Table C3 - Scenario 3, configuration 50x50_f1.0_cap1 (50x50 grid, conversion fraction 1.0, cap 1): view (c) cross-method min-max: within each column the worst method scores 0 and the best 1, rescaled from the RAW view. Cells are mean ± SE over 5 seed(s); * marks p < 0.05 in a paired t-test against QOLU+BUIA on the RAW view, of which this view is a per-cell affine rescaling that leaves the test unchanged; bold marks the best method in the column.*

### Configuration `50x50_f1.0_cap1`, Scenario 4 (land use 0.25, woodland 0.25, urban 0.50)

| Method | Land Use | Woodland | Urban |
| --- | --- | --- | --- |
| No regulator | 0.814 ± 0.007 | 0.238 ± 0.022* | 0.000 ± 0.051 |
| LToS | 0.122 ± 0.297 | 0.064 ± 0.257* | 0.349 ± 0.385 |
| P-MADDPG | 0.069 ± 0.076* | 0.000 ± 0.016* | **1.000 ± 0.136*** |
| MitM (learned) | 0.000 ± 0.025* | 0.549 ± 0.000* | 0.069 ± 0.055 |
| QOLU+BUIA | **1.000 ± 0.131** | **1.000 ± 0.132** | 0.270 ± 0.227 |

*Table C4 - Scenario 4, configuration 50x50_f1.0_cap1 (50x50 grid, conversion fraction 1.0, cap 1): view (c) cross-method min-max: within each column the worst method scores 0 and the best 1, rescaled from the RAW view. Cells are mean ± SE over 5 seed(s); * marks p < 0.05 in a paired t-test against QOLU+BUIA on the RAW view, of which this view is a per-cell affine rescaling that leaves the test unchanged; bold marks the best method in the column.*

## Appendix A - view (a): RAW metric values

### Configuration `50x50_f1.0_cap1`, Scenario 1 (land use 0.33, woodland 0.33, urban 0.33)

| Method | Land Use | Woodland | Urban | C_align |
| --- | --- | --- | --- | --- |
| No regulator | **0.549 ± 0.000** | 0.187 ± 0.005* | 0.977 ± 0.000* | 0.399 ± 0.000* |
| LToS | 0.529 ± 0.008 | 0.139 ± 0.059* | 0.977 ± 0.000 | 0.399 ± 0.000* |
| P-MADDPG | 0.530 ± 0.001 | 0.132 ± 0.006* | **0.978 ± 0.000** | 0.400 ± 0.000* |
| MitM (learned) | 0.527 ± 0.001 | 0.259 ± 0.000* | 0.977 ± 0.000* | 0.398 ± 0.000* |
| QOLU+BUIA | 0.539 ± 0.006 | **0.433 ± 0.051** | 0.978 ± 0.000 | **0.315 ± 0.013** |

*Table A1 - Scenario 1, configuration 50x50_f1.0_cap1 (50x50 grid, conversion fraction 1.0, cap 1): view (a) RAW metric values exactly as chapter4_qolu/env/evaluation.py defines them, with distances normalised by the grid diagonal. Cells are mean ± SE over 5 seed(s); * marks p < 0.05 in a paired t-test against QOLU+BUIA on this RAW view; bold marks the best method in the column; C_align is lower-is-better, so its bold marks the lowest.*

### Configuration `50x50_f1.0_cap1`, Scenario 2 (land use 0.50, woodland 0.25, urban 0.25)

| Method | Land Use | Woodland | Urban | C_align |
| --- | --- | --- | --- | --- |
| No regulator | 0.549 ± 0.000 | 0.187 ± 0.005* | 0.977 ± 0.000 | 0.399 ± 0.000* |
| LToS | 0.530 ± 0.008 | 0.138 ± 0.058* | 0.977 ± 0.000 | 0.399 ± 0.000* |
| P-MADDPG | 0.529 ± 0.002* | 0.136 ± 0.005* | **0.978 ± 0.000** | 0.400 ± 0.000* |
| MitM (learned) | **0.559 ± 0.000** | 0.257 ± 0.000 | 0.977 ± 0.000 | 0.399 ± 0.000* |
| QOLU+BUIA | 0.553 ± 0.007 | **0.384 ± 0.048** | 0.977 ± 0.000 | **0.301 ± 0.016** |

*Table A2 - Scenario 2, configuration 50x50_f1.0_cap1 (50x50 grid, conversion fraction 1.0, cap 1): view (a) RAW metric values exactly as chapter4_qolu/env/evaluation.py defines them, with distances normalised by the grid diagonal. Cells are mean ± SE over 5 seed(s); * marks p < 0.05 in a paired t-test against QOLU+BUIA on this RAW view; bold marks the best method in the column; C_align is lower-is-better, so its bold marks the lowest.*

### Configuration `50x50_f1.0_cap1`, Scenario 3 (land use 0.25, woodland 0.50, urban 0.25)

| Method | Land Use | Woodland | Urban | C_align |
| --- | --- | --- | --- | --- |
| No regulator | **0.549 ± 0.000** | 0.187 ± 0.005* | 0.977 ± 0.000 | 0.399 ± 0.000* |
| LToS | 0.530 ± 0.008 | 0.155 ± 0.063* | 0.977 ± 0.000 | 0.399 ± 0.000* |
| P-MADDPG | 0.529 ± 0.001 | 0.134 ± 0.004* | **0.978 ± 0.000** | 0.400 ± 0.000* |
| MitM (learned) | 0.529 ± 0.001 | 0.259 ± 0.000* | 0.977 ± 0.000 | 0.398 ± 0.000* |
| QOLU+BUIA | 0.543 ± 0.006 | **0.425 ± 0.049** | 0.978 ± 0.000 | **0.308 ± 0.014** |

*Table A3 - Scenario 3, configuration 50x50_f1.0_cap1 (50x50 grid, conversion fraction 1.0, cap 1): view (a) RAW metric values exactly as chapter4_qolu/env/evaluation.py defines them, with distances normalised by the grid diagonal. Cells are mean ± SE over 5 seed(s); * marks p < 0.05 in a paired t-test against QOLU+BUIA on this RAW view; bold marks the best method in the column; C_align is lower-is-better, so its bold marks the lowest.*

### Configuration `50x50_f1.0_cap1`, Scenario 4 (land use 0.25, woodland 0.25, urban 0.50)

| Method | Land Use | Woodland | Urban | C_align |
| --- | --- | --- | --- | --- |
| No regulator | 0.549 ± 0.000 | 0.187 ± 0.005* | 0.977 ± 0.000 | 0.399 ± 0.000* |
| LToS | 0.530 ± 0.008 | 0.146 ± 0.060* | 0.977 ± 0.000 | 0.399 ± 0.000* |
| P-MADDPG | 0.529 ± 0.002* | 0.131 ± 0.004* | **0.978 ± 0.000*** | 0.400 ± 0.000* |
| MitM (learned) | 0.527 ± 0.001* | 0.259 ± 0.000* | 0.977 ± 0.000 | 0.399 ± 0.000* |
| QOLU+BUIA | **0.554 ± 0.003** | **0.365 ± 0.031** | 0.977 ± 0.000 | **0.314 ± 0.011** |

*Table A4 - Scenario 4, configuration 50x50_f1.0_cap1 (50x50 grid, conversion fraction 1.0, cap 1): view (a) RAW metric values exactly as chapter4_qolu/env/evaluation.py defines them, with distances normalised by the grid diagonal. Cells are mean ± SE over 5 seed(s); * marks p < 0.05 in a paired t-test against QOLU+BUIA on this RAW view; bold marks the best method in the column; C_align is lower-is-better, so its bold marks the lowest.*

## Appendix B - view (b): absolute-normalised metric values

### Configuration `50x50_f1.0_cap1`, Scenario 1 (land use 0.33, woodland 0.33, urban 0.33)

| Method | Land Use | Woodland | Urban | C_align |
| --- | --- | --- | --- | --- |
| No regulator | 0.979 ± 0.001 | 0.187 ± 0.005* | 0.838 ± 0.000* | 0.399 ± 0.000* |
| LToS | 0.969 ± 0.001* | 0.139 ± 0.059* | 0.839 ± 0.002 | 0.399 ± 0.000* |
| P-MADDPG | 0.970 ± 0.000* | 0.132 ± 0.006* | **0.842 ± 0.001** | 0.400 ± 0.000* |
| MitM (learned) | 0.970 ± 0.000* | 0.259 ± 0.000* | 0.838 ± 0.000* | 0.398 ± 0.000* |
| QOLU+BUIA | **0.980 ± 0.002** | **0.433 ± 0.051** | 0.842 ± 0.001 | **0.315 ± 0.013** |

*Table B1 - Scenario 1, configuration 50x50_f1.0_cap1 (50x50 grid, conversion fraction 1.0, cap 1): view (b) absolute-normalised: the same metrics with RewardConfig.max_distance set to a physical distance, so proximity saturates at a fixed number of kilometres rather than at the grid diagonal (max_distance = 10.0 km). Cells are mean ± SE over 5 seed(s); * marks p < 0.05 in a paired t-test against QOLU+BUIA on this absolute-normalised view; bold marks the best method in the column; C_align is lower-is-better, so its bold marks the lowest.*

### Configuration `50x50_f1.0_cap1`, Scenario 2 (land use 0.50, woodland 0.25, urban 0.25)

| Method | Land Use | Woodland | Urban | C_align |
| --- | --- | --- | --- | --- |
| No regulator | 0.979 ± 0.001 | 0.187 ± 0.005* | 0.838 ± 0.000 | 0.399 ± 0.000* |
| LToS | 0.969 ± 0.001* | 0.138 ± 0.058* | 0.839 ± 0.002 | 0.399 ± 0.000* |
| P-MADDPG | 0.970 ± 0.000* | 0.136 ± 0.005* | **0.842 ± 0.000** | 0.400 ± 0.000* |
| MitM (learned) | 0.980 ± 0.000 | 0.257 ± 0.000 | 0.835 ± 0.000 | 0.399 ± 0.000* |
| QOLU+BUIA | **0.981 ± 0.003** | **0.384 ± 0.048** | 0.841 ± 0.003 | **0.301 ± 0.016** |

*Table B2 - Scenario 2, configuration 50x50_f1.0_cap1 (50x50 grid, conversion fraction 1.0, cap 1): view (b) absolute-normalised: the same metrics with RewardConfig.max_distance set to a physical distance, so proximity saturates at a fixed number of kilometres rather than at the grid diagonal (max_distance = 10.0 km). Cells are mean ± SE over 5 seed(s); * marks p < 0.05 in a paired t-test against QOLU+BUIA on this absolute-normalised view; bold marks the best method in the column; C_align is lower-is-better, so its bold marks the lowest.*

### Configuration `50x50_f1.0_cap1`, Scenario 3 (land use 0.25, woodland 0.50, urban 0.25)

| Method | Land Use | Woodland | Urban | C_align |
| --- | --- | --- | --- | --- |
| No regulator | **0.979 ± 0.001** | 0.187 ± 0.005* | 0.838 ± 0.000 | 0.399 ± 0.000* |
| LToS | 0.970 ± 0.002 | 0.155 ± 0.063* | 0.840 ± 0.001 | 0.399 ± 0.000* |
| P-MADDPG | 0.970 ± 0.000* | 0.134 ± 0.004* | **0.842 ± 0.001** | 0.400 ± 0.000* |
| MitM (learned) | 0.971 ± 0.000* | 0.259 ± 0.000* | 0.838 ± 0.000 | 0.398 ± 0.000* |
| QOLU+BUIA | 0.979 ± 0.003 | **0.425 ± 0.049** | 0.841 ± 0.001 | **0.308 ± 0.014** |

*Table B3 - Scenario 3, configuration 50x50_f1.0_cap1 (50x50 grid, conversion fraction 1.0, cap 1): view (b) absolute-normalised: the same metrics with RewardConfig.max_distance set to a physical distance, so proximity saturates at a fixed number of kilometres rather than at the grid diagonal (max_distance = 10.0 km). Cells are mean ± SE over 5 seed(s); * marks p < 0.05 in a paired t-test against QOLU+BUIA on this absolute-normalised view; bold marks the best method in the column; C_align is lower-is-better, so its bold marks the lowest.*

### Configuration `50x50_f1.0_cap1`, Scenario 4 (land use 0.25, woodland 0.25, urban 0.50)

| Method | Land Use | Woodland | Urban | C_align |
| --- | --- | --- | --- | --- |
| No regulator | 0.979 ± 0.001 | 0.187 ± 0.005* | 0.838 ± 0.000 | 0.399 ± 0.000* |
| LToS | 0.970 ± 0.002* | 0.146 ± 0.060* | 0.839 ± 0.002 | 0.399 ± 0.000* |
| P-MADDPG | 0.970 ± 0.001* | 0.131 ± 0.004* | **0.842 ± 0.001*** | 0.400 ± 0.000* |
| MitM (learned) | 0.970 ± 0.000* | 0.259 ± 0.000* | 0.838 ± 0.000 | 0.399 ± 0.000* |
| QOLU+BUIA | **0.981 ± 0.001** | **0.365 ± 0.031** | 0.839 ± 0.001 | **0.314 ± 0.011** |

*Table B4 - Scenario 4, configuration 50x50_f1.0_cap1 (50x50 grid, conversion fraction 1.0, cap 1): view (b) absolute-normalised: the same metrics with RewardConfig.max_distance set to a physical distance, so proximity saturates at a fixed number of kilometres rather than at the grid diagonal (max_distance = 10.0 km). Cells are mean ± SE over 5 seed(s); * marks p < 0.05 in a paired t-test against QOLU+BUIA on this absolute-normalised view; bold marks the best method in the column; C_align is lower-is-better, so its bold marks the lowest.*
