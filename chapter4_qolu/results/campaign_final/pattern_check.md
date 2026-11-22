# Campaign `final` - ranking-pattern check

The published tables rank the methods **QOLU+BUIA > P-MADDPG > LToS > MitM** in every scenario, with QOLU+BUIA *strictly* first on the metric its scenario emphasises (Scenario 1: all three; Scenario 2: Land Use; Scenario 3: Woodland; Scenario 4: Urban) and tied with P-MADDPG elsewhere.  A cell below agrees when the rebuild reproduces that ordering: QOLU+BUIA first on an emphasised metric, in the top two otherwise, with the baselines present ordered P-MADDPG > LToS > MitM.

Rankings are taken from the RAW view; min-max rescaling is affine per cell and cannot change an ordering.

The comparison is of the *ordering* only.  The published cells are quoted to two decimals on a 0.05 lattice and every published standard error is 0.01, 0.02 or 0.03, so the values themselves are not compared.  The study region is `lcm2015_gb_1km_dominant_aggregate_class.img` cropped to window E495_N165, which the window scan selected; the paper states no extent of its own.  The three baselines are re-implementations; see the README for what each one does.

## Configuration `50x50_f1.0_cap1` (50x50 grid, conversion fraction 1.0, cap 1)

| Cell | Metric | Emphasised | Published order | Rebuild order | Agrees? |
| --- | --- | --- | --- | --- | --- |
| Scenario 1 | Land Use | yes | QOLU+BUIA > P-MADDPG > LToS > MitM | QOLU+BUIA > P-MADDPG > LToS > MitM (learned) | yes |
| Scenario 1 | Woodland | yes | QOLU+BUIA > P-MADDPG > LToS > MitM | QOLU+BUIA > MitM (learned) > LToS > P-MADDPG | no - the baselines are not in the published order |
| Scenario 1 | Urban | yes | QOLU+BUIA > P-MADDPG > LToS > MitM | P-MADDPG > QOLU+BUIA > LToS > MitM (learned) | no - QOLU+BUIA ranked 2 (the published tables put it first) |
| Scenario 2 | Land Use | yes | QOLU+BUIA > P-MADDPG > LToS > MitM | MitM (learned) > QOLU+BUIA > LToS > P-MADDPG | no - QOLU+BUIA ranked 2 (the published tables put it first) |
| Scenario 2 | Woodland | no | QOLU+BUIA > P-MADDPG > LToS > MitM | QOLU+BUIA > MitM (learned) > LToS > P-MADDPG | no - the baselines are not in the published order |
| Scenario 2 | Urban | no | QOLU+BUIA > P-MADDPG > LToS > MitM | P-MADDPG > QOLU+BUIA > LToS > MitM (learned) | yes |
| Scenario 3 | Land Use | no | QOLU+BUIA > P-MADDPG > LToS > MitM | QOLU+BUIA > LToS > P-MADDPG > MitM (learned) | no - the baselines are not in the published order |
| Scenario 3 | Woodland | yes | QOLU+BUIA > P-MADDPG > LToS > MitM | QOLU+BUIA > MitM (learned) > LToS > P-MADDPG | no - the baselines are not in the published order |
| Scenario 3 | Urban | no | QOLU+BUIA > P-MADDPG > LToS > MitM | P-MADDPG > QOLU+BUIA > LToS > MitM (learned) | yes |
| Scenario 4 | Land Use | no | QOLU+BUIA > P-MADDPG > LToS > MitM | QOLU+BUIA > LToS > P-MADDPG > MitM (learned) | no - the baselines are not in the published order |
| Scenario 4 | Woodland | no | QOLU+BUIA > P-MADDPG > LToS > MitM | QOLU+BUIA > MitM (learned) > LToS > P-MADDPG | no - the baselines are not in the published order |
| Scenario 4 | Urban | yes | QOLU+BUIA > P-MADDPG > LToS > MitM | P-MADDPG > LToS > QOLU+BUIA > MitM (learned) | no - QOLU+BUIA ranked 3 (the published tables put it first) |

3 of 12 cells agree with the published order.

## Overall

3 of 12 cells across every configuration reproduce the published ranking pattern.
