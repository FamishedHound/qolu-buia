# QOLU-BUIA: multi-objective reinforcement learning for urban land-use planning

> Code for Chapter 4 of the PhD thesis *Multi-Level Agent-Based Land Use Decision-Making Using Multi-Agent Reinforcement Learning* (Lukasz Pelcner, Lancaster University, 2026). The method was introduced in our [IJCAI 2025 paper](https://www.ijcai.org/proceedings/2025/27).

A bottom-up investor agent (**BUIA**) proposes parcels to convert to urban land on a 50 x 50 km window of the UKCEH Land Cover Map 2015, at 1 km resolution. Three top-down stakeholder agents (**QOLU**) appraise each proposal, and a consensus of their appraisals decides whether the conversion goes ahead. QOLU is a categorical distributional DQN (C51) with double Q-learning, a dueling head, n-step returns and noisy layers. The comparison methods are P-MADDPG, LToS, MitM and a condition without a regulator, in which every proposal is converted.

This is a re-implementation of the method of the paper. The results in Chapter 4 of the thesis were produced with it, and the records of those runs are included.

The three baselines are also re-implemented for this environment. P-MADDPG follows the published algorithm of the Chapter 3 paper. LToS is written in PyTorch on a lattice of districts, with a DGN low level and a DDPG high level that shares reward between neighbouring districts. MitM is a reconstruction of the approach of Bone et al. (2011) from the published summary of that work, with one learner per objective over a vector of criterion weights. All of them run in the same environment, with the same seeds and metrics as QOLU+BUIA.

If you use this code, please cite the paper (see [Citation](#citation)).

## What is where

| | |
|---|---|
| `chapter4_qolu/env`, `data` | the land-use environment, its metrics, and the raster loader |
| `chapter4_qolu/agent` | QOLU and BUIA |
| `chapter4_qolu/baselines` | P-MADDPG, LToS, MitM |
| `chapter4_qolu/run_scenarios.py` | the four-scenario comparison |
| `chapter4_qolu/make_figures.py`, `make_thesis_tables.py` | figures and tables from finished runs |
| `chapter4_qolu/run_distribution_diagnostics.py`, `diagnostics` | accuracy of the learned return distributions (Table 4.8) |
| `chapter4_qolu/run_window_scan.py`, `run_sensitivity.py` | scans over candidate study windows and over the settings the paper leaves open, scored against the ordering of the published tables |
| `chapter4_qolu/run_hparam_sensitivity.py` | sensitivity of the comparison to QOLU's hyper-parameters |
| `chapter4_qolu/results/campaign_final` | records, tables and figures of Chapter 4 |
| `chapter4_qolu/results/final_diag` | the distribution diagnostics |
| `chapter4_qolu/tests` | tests |

## Setup

Python 3.10 or newer (tested with 3.13).

```bash
git clone https://github.com/FamishedHound/qolu-buia.git
cd qolu-buia
pip install -r requirements.txt
```

Run everything from the repository root. On Windows, clone into a folder with a short path or enable long paths (`git config --global core.longpaths true`): the longest file path in the repository is 160 characters.

## Data

No land-cover data is distributed with the code, and the run records are published without their land-cover grids, because both derive from the UKCEH Land Cover Map, which has its own licence. For the same reason the land-cover maps (Figure 4.1 and the map of each run) are not in `results/`; step 2 below draws them from the raster.

### Obtaining the raster used in Chapter 4

The results were computed on *Land Cover Map 2015 (1km dominant aggregate class, GB)*, Rowland et al. (2017), NERC Environmental Information Data Centre, <https://doi.org/10.5285/711c8dc1-0f4e-42ad-a703-8b5d19c92247>. Download it from that page under the licence stated there, unzip it, and copy the files of its `data/` folder into `chapter4_qolu/data_raw/lcm2015_gb_1km/`, so that the raster is

```
chapter4_qolu/data_raw/lcm2015_gb_1km/lcm2015_gb_1km_dominant_aggregate_class.img
```

The `.img` file is read with rasterio. `data_raw/` is ignored by git.

## Reproducing the results of Chapter 4

**From the records in the repository** (no raster, nothing is run):

```bash
# Tables 4.4 to 4.8
python -m chapter4_qolu.make_thesis_tables \
    --campaign chapter4_qolu/results/campaign_final \
    --diagnostics chapter4_qolu/results/final_diag/distribution_diagnostics_final

# Figure 4.3 and the other figures computed from the metrics
python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --no-land-cover
```

**Running the experiments again** (needs the raster). `RASTER` below is the `.img` file above. The records of the thesis runs are in `results/campaign_final`, and a run whose record is on disk is skipped, so the commands below write to a new tag, `rerun`.

```bash
RASTER=chapter4_qolu/data_raw/lcm2015_gb_1km/lcm2015_gb_1km_dominant_aggregate_class.img

# 1. the comparison: 5 methods x 4 scenarios x 5 seeds, 1000 iterations per run (Tables 4.4 to 4.7)
python -m chapter4_qolu.run_scenarios --raster $RASTER \
    --tag rerun --window E495_N165 --configs 50:1.0:1 --qolu-reward-mode team \
    --methods no_regulator ltos pmaddpg mitm_learned qolu_buia --scenarios 1 2 3 4 \
    --seeds 0 1 2 3 4 --iterations 1000 --decimals 3 --parallel 6 --resume

# 2. the figures, including the land-cover maps (Figures 4.1 and 4.3)
python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_rerun \
    --raster $RASTER --region E495_N165
python -m chapter4_qolu.figures.region_overview --campaign chapter4_qolu/results/campaign_rerun \
    --raster $RASTER --scenario 1 --seed 0

# 3. the return-distribution diagnostics (Table 4.8)
python -m chapter4_qolu.run_distribution_diagnostics --raster $RASTER \
    --tag rerun --window E495_N165 --config 50:1.0:1 --method qolu_buia \
    --qolu-reward-mode team --scenarios 1 2 3 4 --seeds 0 1 2 \
    --iterations 1000 --eval-iterations 200 --parallel 4 --resume \
    --out chapter4_qolu/results/rerun_diag

# 4. the thesis tables
python -m chapter4_qolu.make_thesis_tables \
    --campaign chapter4_qolu/results/campaign_rerun \
    --diagnostics chapter4_qolu/results/rerun_diag/distribution_diagnostics_rerun
```

`--window E495_N165` is the 50 x 50 km window whose south-west corner is at British National Grid E 495 km, N 165 km. `mitm_learned` is the MitM row of the thesis tables. A one-minute check that everything runs: `python -m chapter4_qolu.run_scenarios --raster $RASTER --smoke`.

**Not produced by a command here.** Figure 4.2 of the thesis (the land-preference pies) is taken from the published paper; the `land_preference_*` figures in `results/campaign_final/figures` are this package's own computation from the 1 km aggregate classes and are not the same figures. Figure 4.1 and the map of each run need the raster and a re-run (steps 1 and 2); `figures_index.md` still lists those map files. Figure 4.3 is `scenario_comparison_final`, drawn with slightly different labels from the printed figure. Drawing figures needs `matplotlib`, which is not in `requirements.txt`.

Tests: `python -m pytest chapter4_qolu/tests`

## Citation

```bibtex
@inproceedings{pelcner2025multiobjective,
  title     = {Multi-Objective Quantile-Based Reinforcement Learning for Modern Urban Planning},
  author    = {Pelcner, Lukasz and Marcolino, Leandro Soriano and do Carmo Alves, Matheus Aparecido and Harrison, Paula A. and Atkinson, Peter M.},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)},
  pages     = {232--239},
  year      = {2025},
  doi       = {10.24963/ijcai.2025/27}
}

@phdthesis{pelcner2026thesis,
  title  = {Multi-Level Agent-Based Land Use Decision-Making Using Multi-Agent Reinforcement Learning},
  author = {Pelcner, Lukasz},
  school = {Lancaster University},
  year   = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
