# Chapter 4 figures

Every figure below was generated from files already on disk - per-run JSON records and a campaign's `summary.json` - so none of them needs a run to be repeated.  Each writes its own numbers as a `.csv` of the same name, plus a `.pdf` and a `.png`.

The `.csv` files of the land-cover figures - the before / after maps (`land_cover_scenario<k>_seed<n>`), the study-region maps (`region_before_after_final`, `region_converted_final`, `region_overview_final`) and the land-preference pies (`land_preference`) - are not distributed with this repository: they hold the land-cover classes of the study region, derived from the UKCEH Land Cover Map 2015, which may not be redistributed, and the run records are published without the land-cover grids these figures are drawn from.  The `.pdf` and `.png` files of these figures are images of LCM2015 data.  Data owned by UK Centre for Ecology & Hydrology © Database Right/Copyright UKCEH.

## `scenario_comparison_final`

Scenario comparison in the published layout: one panel per scenario, methods in the published order, three bars per method (Land Use, Woodland Loss, Urban), view (b) absolute-normalised: the same metrics with RewardConfig.max_distance set to a physical distance, so proximity saturates at a fixed number of kilometres rather than at the grid diagonal.

* **files**: `scenario_comparison_final.csv`, `scenario_comparison_final.pdf`, `scenario_comparison_final.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --view absolute`
* **note**: the published figure carries no error bars; these are ± one standard error over seeds
* **note**: 'Woodland Loss' is the published legend's wording; the metric is the share of the initial woodland and arable area still standing, so higher is better, as in the published tables
* **note**: drawn on view (absolute); the published values are absolute 0-1 scores whose normalisation constants are unknown

## `land_cover_scenario1_seed0`

Land cover before and after development, Scenario 1, seed 0, one column per method.  Every method in the cell starts from the same landscape.

* **files**: `land_cover_scenario1_seed0.pdf`, `land_cover_scenario1_seed0.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 1 --seed 0`
* **note**: drawn in the 10 LCM2015 aggregate classes; the paper's own legend uses 8 classes derived from the 21 LCM target classes, a product not available here

## `land_cover_scenario1_seed1`

Land cover before and after development, Scenario 1, seed 1, one column per method.  Every method in the cell starts from the same landscape.

* **files**: `land_cover_scenario1_seed1.pdf`, `land_cover_scenario1_seed1.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 1 --seed 1`
* **note**: drawn in the 10 LCM2015 aggregate classes; the paper's own legend uses 8 classes derived from the 21 LCM target classes, a product not available here

## `land_cover_scenario1_seed2`

Land cover before and after development, Scenario 1, seed 2, one column per method.  Every method in the cell starts from the same landscape.

* **files**: `land_cover_scenario1_seed2.pdf`, `land_cover_scenario1_seed2.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 1 --seed 2`
* **note**: drawn in the 10 LCM2015 aggregate classes; the paper's own legend uses 8 classes derived from the 21 LCM target classes, a product not available here

## `land_cover_scenario1_seed3`

Land cover before and after development, Scenario 1, seed 3, one column per method.  Every method in the cell starts from the same landscape.

* **files**: `land_cover_scenario1_seed3.pdf`, `land_cover_scenario1_seed3.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 1 --seed 3`
* **note**: drawn in the 10 LCM2015 aggregate classes; the paper's own legend uses 8 classes derived from the 21 LCM target classes, a product not available here

## `land_cover_scenario1_seed4`

Land cover before and after development, Scenario 1, seed 4, one column per method.  Every method in the cell starts from the same landscape.

* **files**: `land_cover_scenario1_seed4.pdf`, `land_cover_scenario1_seed4.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 1 --seed 4`
* **note**: drawn in the 10 LCM2015 aggregate classes; the paper's own legend uses 8 classes derived from the 21 LCM target classes, a product not available here

## `curves_scenario1`

Per-iteration series for Scenario 1, one panel per series; the line is the mean over seeds and the band the min-to-max envelope.

* **files**: `curves_scenario1.csv`, `curves_scenario1.pdf`, `curves_scenario1.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 1`
* **note**: the band is a min-max envelope over seeds, not a confidence interval; the per-seed values are in the CSV

## `table_bars_scenario1_minmax`

Scenario 1, view (c) cross-method min-max: within each column the worst method scores 0 and the best 1, rescaled from the RAW view.  Bars are the mean over seeds and the error bars one standard error.

* **files**: `table_bars_scenario1_minmax.csv`, `table_bars_scenario1_minmax.pdf`, `table_bars_scenario1_minmax.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 1 --view minmax`
* **note**: the published Tables 4-7 are on none of the three views - see reporting/published.py - so these bars are not comparable with the paper's Figure 5 cell by cell

## `table_bars_scenario1_raw`

Scenario 1, view (a) RAW metric values exactly as chapter4_qolu/env/evaluation.py defines them, with distances normalised by the grid diagonal.  Bars are the mean over seeds and the error bars one standard error.

* **files**: `table_bars_scenario1_raw.csv`, `table_bars_scenario1_raw.pdf`, `table_bars_scenario1_raw.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 1 --view raw`
* **note**: the published Tables 4-7 are on none of the three views - see reporting/published.py - so these bars are not comparable with the paper's Figure 5 cell by cell

## `table_bars_scenario1_absolute`

Scenario 1, view (b) absolute-normalised: the same metrics with RewardConfig.max_distance set to a physical distance, so proximity saturates at a fixed number of kilometres rather than at the grid diagonal.  Bars are the mean over seeds and the error bars one standard error.

* **files**: `table_bars_scenario1_absolute.csv`, `table_bars_scenario1_absolute.pdf`, `table_bars_scenario1_absolute.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 1 --view absolute`
* **note**: the published Tables 4-7 are on none of the three views - see reporting/published.py - so these bars are not comparable with the paper's Figure 5 cell by cell

## `land_cover_scenario2_seed0`

Land cover before and after development, Scenario 2, seed 0, one column per method.  Every method in the cell starts from the same landscape.

* **files**: `land_cover_scenario2_seed0.pdf`, `land_cover_scenario2_seed0.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 2 --seed 0`
* **note**: drawn in the 10 LCM2015 aggregate classes; the paper's own legend uses 8 classes derived from the 21 LCM target classes, a product not available here

## `land_cover_scenario2_seed1`

Land cover before and after development, Scenario 2, seed 1, one column per method.  Every method in the cell starts from the same landscape.

* **files**: `land_cover_scenario2_seed1.pdf`, `land_cover_scenario2_seed1.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 2 --seed 1`
* **note**: drawn in the 10 LCM2015 aggregate classes; the paper's own legend uses 8 classes derived from the 21 LCM target classes, a product not available here

## `land_cover_scenario2_seed2`

Land cover before and after development, Scenario 2, seed 2, one column per method.  Every method in the cell starts from the same landscape.

* **files**: `land_cover_scenario2_seed2.pdf`, `land_cover_scenario2_seed2.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 2 --seed 2`
* **note**: drawn in the 10 LCM2015 aggregate classes; the paper's own legend uses 8 classes derived from the 21 LCM target classes, a product not available here

## `land_cover_scenario2_seed3`

Land cover before and after development, Scenario 2, seed 3, one column per method.  Every method in the cell starts from the same landscape.

* **files**: `land_cover_scenario2_seed3.pdf`, `land_cover_scenario2_seed3.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 2 --seed 3`
* **note**: drawn in the 10 LCM2015 aggregate classes; the paper's own legend uses 8 classes derived from the 21 LCM target classes, a product not available here

## `land_cover_scenario2_seed4`

Land cover before and after development, Scenario 2, seed 4, one column per method.  Every method in the cell starts from the same landscape.

* **files**: `land_cover_scenario2_seed4.pdf`, `land_cover_scenario2_seed4.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 2 --seed 4`
* **note**: drawn in the 10 LCM2015 aggregate classes; the paper's own legend uses 8 classes derived from the 21 LCM target classes, a product not available here

## `curves_scenario2`

Per-iteration series for Scenario 2, one panel per series; the line is the mean over seeds and the band the min-to-max envelope.

* **files**: `curves_scenario2.csv`, `curves_scenario2.pdf`, `curves_scenario2.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 2`
* **note**: the band is a min-max envelope over seeds, not a confidence interval; the per-seed values are in the CSV

## `table_bars_scenario2_minmax`

Scenario 2, view (c) cross-method min-max: within each column the worst method scores 0 and the best 1, rescaled from the RAW view.  Bars are the mean over seeds and the error bars one standard error.

* **files**: `table_bars_scenario2_minmax.csv`, `table_bars_scenario2_minmax.pdf`, `table_bars_scenario2_minmax.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 2 --view minmax`
* **note**: the published Tables 4-7 are on none of the three views - see reporting/published.py - so these bars are not comparable with the paper's Figure 5 cell by cell

## `table_bars_scenario2_raw`

Scenario 2, view (a) RAW metric values exactly as chapter4_qolu/env/evaluation.py defines them, with distances normalised by the grid diagonal.  Bars are the mean over seeds and the error bars one standard error.

* **files**: `table_bars_scenario2_raw.csv`, `table_bars_scenario2_raw.pdf`, `table_bars_scenario2_raw.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 2 --view raw`
* **note**: the published Tables 4-7 are on none of the three views - see reporting/published.py - so these bars are not comparable with the paper's Figure 5 cell by cell

## `table_bars_scenario2_absolute`

Scenario 2, view (b) absolute-normalised: the same metrics with RewardConfig.max_distance set to a physical distance, so proximity saturates at a fixed number of kilometres rather than at the grid diagonal.  Bars are the mean over seeds and the error bars one standard error.

* **files**: `table_bars_scenario2_absolute.csv`, `table_bars_scenario2_absolute.pdf`, `table_bars_scenario2_absolute.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 2 --view absolute`
* **note**: the published Tables 4-7 are on none of the three views - see reporting/published.py - so these bars are not comparable with the paper's Figure 5 cell by cell

## `land_cover_scenario3_seed0`

Land cover before and after development, Scenario 3, seed 0, one column per method.  Every method in the cell starts from the same landscape.

* **files**: `land_cover_scenario3_seed0.pdf`, `land_cover_scenario3_seed0.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 3 --seed 0`
* **note**: drawn in the 10 LCM2015 aggregate classes; the paper's own legend uses 8 classes derived from the 21 LCM target classes, a product not available here

## `land_cover_scenario3_seed1`

Land cover before and after development, Scenario 3, seed 1, one column per method.  Every method in the cell starts from the same landscape.

* **files**: `land_cover_scenario3_seed1.pdf`, `land_cover_scenario3_seed1.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 3 --seed 1`
* **note**: drawn in the 10 LCM2015 aggregate classes; the paper's own legend uses 8 classes derived from the 21 LCM target classes, a product not available here

## `land_cover_scenario3_seed2`

Land cover before and after development, Scenario 3, seed 2, one column per method.  Every method in the cell starts from the same landscape.

* **files**: `land_cover_scenario3_seed2.pdf`, `land_cover_scenario3_seed2.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 3 --seed 2`
* **note**: drawn in the 10 LCM2015 aggregate classes; the paper's own legend uses 8 classes derived from the 21 LCM target classes, a product not available here

## `land_cover_scenario3_seed3`

Land cover before and after development, Scenario 3, seed 3, one column per method.  Every method in the cell starts from the same landscape.

* **files**: `land_cover_scenario3_seed3.pdf`, `land_cover_scenario3_seed3.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 3 --seed 3`
* **note**: drawn in the 10 LCM2015 aggregate classes; the paper's own legend uses 8 classes derived from the 21 LCM target classes, a product not available here

## `land_cover_scenario3_seed4`

Land cover before and after development, Scenario 3, seed 4, one column per method.  Every method in the cell starts from the same landscape.

* **files**: `land_cover_scenario3_seed4.pdf`, `land_cover_scenario3_seed4.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 3 --seed 4`
* **note**: drawn in the 10 LCM2015 aggregate classes; the paper's own legend uses 8 classes derived from the 21 LCM target classes, a product not available here

## `curves_scenario3`

Per-iteration series for Scenario 3, one panel per series; the line is the mean over seeds and the band the min-to-max envelope.

* **files**: `curves_scenario3.csv`, `curves_scenario3.pdf`, `curves_scenario3.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 3`
* **note**: the band is a min-max envelope over seeds, not a confidence interval; the per-seed values are in the CSV

## `table_bars_scenario3_minmax`

Scenario 3, view (c) cross-method min-max: within each column the worst method scores 0 and the best 1, rescaled from the RAW view.  Bars are the mean over seeds and the error bars one standard error.

* **files**: `table_bars_scenario3_minmax.csv`, `table_bars_scenario3_minmax.pdf`, `table_bars_scenario3_minmax.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 3 --view minmax`
* **note**: the published Tables 4-7 are on none of the three views - see reporting/published.py - so these bars are not comparable with the paper's Figure 5 cell by cell

## `table_bars_scenario3_raw`

Scenario 3, view (a) RAW metric values exactly as chapter4_qolu/env/evaluation.py defines them, with distances normalised by the grid diagonal.  Bars are the mean over seeds and the error bars one standard error.

* **files**: `table_bars_scenario3_raw.csv`, `table_bars_scenario3_raw.pdf`, `table_bars_scenario3_raw.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 3 --view raw`
* **note**: the published Tables 4-7 are on none of the three views - see reporting/published.py - so these bars are not comparable with the paper's Figure 5 cell by cell

## `table_bars_scenario3_absolute`

Scenario 3, view (b) absolute-normalised: the same metrics with RewardConfig.max_distance set to a physical distance, so proximity saturates at a fixed number of kilometres rather than at the grid diagonal.  Bars are the mean over seeds and the error bars one standard error.

* **files**: `table_bars_scenario3_absolute.csv`, `table_bars_scenario3_absolute.pdf`, `table_bars_scenario3_absolute.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 3 --view absolute`
* **note**: the published Tables 4-7 are on none of the three views - see reporting/published.py - so these bars are not comparable with the paper's Figure 5 cell by cell

## `land_cover_scenario4_seed0`

Land cover before and after development, Scenario 4, seed 0, one column per method.  Every method in the cell starts from the same landscape.

* **files**: `land_cover_scenario4_seed0.pdf`, `land_cover_scenario4_seed0.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 4 --seed 0`
* **note**: drawn in the 10 LCM2015 aggregate classes; the paper's own legend uses 8 classes derived from the 21 LCM target classes, a product not available here

## `land_cover_scenario4_seed1`

Land cover before and after development, Scenario 4, seed 1, one column per method.  Every method in the cell starts from the same landscape.

* **files**: `land_cover_scenario4_seed1.pdf`, `land_cover_scenario4_seed1.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 4 --seed 1`
* **note**: drawn in the 10 LCM2015 aggregate classes; the paper's own legend uses 8 classes derived from the 21 LCM target classes, a product not available here

## `land_cover_scenario4_seed2`

Land cover before and after development, Scenario 4, seed 2, one column per method.  Every method in the cell starts from the same landscape.

* **files**: `land_cover_scenario4_seed2.pdf`, `land_cover_scenario4_seed2.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 4 --seed 2`
* **note**: drawn in the 10 LCM2015 aggregate classes; the paper's own legend uses 8 classes derived from the 21 LCM target classes, a product not available here

## `land_cover_scenario4_seed3`

Land cover before and after development, Scenario 4, seed 3, one column per method.  Every method in the cell starts from the same landscape.

* **files**: `land_cover_scenario4_seed3.pdf`, `land_cover_scenario4_seed3.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 4 --seed 3`
* **note**: drawn in the 10 LCM2015 aggregate classes; the paper's own legend uses 8 classes derived from the 21 LCM target classes, a product not available here

## `land_cover_scenario4_seed4`

Land cover before and after development, Scenario 4, seed 4, one column per method.  Every method in the cell starts from the same landscape.

* **files**: `land_cover_scenario4_seed4.pdf`, `land_cover_scenario4_seed4.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 4 --seed 4`
* **note**: drawn in the 10 LCM2015 aggregate classes; the paper's own legend uses 8 classes derived from the 21 LCM target classes, a product not available here

## `curves_scenario4`

Per-iteration series for Scenario 4, one panel per series; the line is the mean over seeds and the band the min-to-max envelope.

* **files**: `curves_scenario4.csv`, `curves_scenario4.pdf`, `curves_scenario4.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 4`
* **note**: the band is a min-max envelope over seeds, not a confidence interval; the per-seed values are in the CSV

## `table_bars_scenario4_minmax`

Scenario 4, view (c) cross-method min-max: within each column the worst method scores 0 and the best 1, rescaled from the RAW view.  Bars are the mean over seeds and the error bars one standard error.

* **files**: `table_bars_scenario4_minmax.csv`, `table_bars_scenario4_minmax.pdf`, `table_bars_scenario4_minmax.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 4 --view minmax`
* **note**: the published Tables 4-7 are on none of the three views - see reporting/published.py - so these bars are not comparable with the paper's Figure 5 cell by cell

## `table_bars_scenario4_raw`

Scenario 4, view (a) RAW metric values exactly as chapter4_qolu/env/evaluation.py defines them, with distances normalised by the grid diagonal.  Bars are the mean over seeds and the error bars one standard error.

* **files**: `table_bars_scenario4_raw.csv`, `table_bars_scenario4_raw.pdf`, `table_bars_scenario4_raw.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 4 --view raw`
* **note**: the published Tables 4-7 are on none of the three views - see reporting/published.py - so these bars are not comparable with the paper's Figure 5 cell by cell

## `table_bars_scenario4_absolute`

Scenario 4, view (b) absolute-normalised: the same metrics with RewardConfig.max_distance set to a physical distance, so proximity saturates at a fixed number of kilometres rather than at the grid diagonal.  Bars are the mean over seeds and the error bars one standard error.

* **files**: `table_bars_scenario4_absolute.csv`, `table_bars_scenario4_absolute.pdf`, `table_bars_scenario4_absolute.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --scenario 4 --view absolute`
* **note**: the published Tables 4-7 are on none of the three views - see reporting/published.py - so these bars are not comparable with the paper's Figure 5 cell by cell

## `region_before_after_final`

Study region `final`: land cover before (left) and after (right) the run, in the published eight-class palette, sea left white.

* **files**: `region_before_after_final.pdf`, `region_before_after_final.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --region --scenario 1 --seed 0`
* **note**: orientation `paper` - transposed so west is at the top and north on the left, matching the published Figure 1
* **note**: Suburban is empty: the LCM2015 *aggregate* product merges suburban and urban into one 'built-up areas and gardens' class, and only the 21 *target* classes - a product not available here - split them.

## `region_overview_final`

Land cover of BNG E 410-650 km, N 60-270 km with the study window outlined, and the window before and after one run, north up, in the published eight-class palette. This is Figure 4.1 of the thesis.

* **files**: `region_overview_final.pdf`, `region_overview_final.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.figures.region_overview --campaign chapter4_qolu/results/campaign_final --raster chapter4_qolu/data_raw/lcm2015_gb_1km/lcm2015_gb_1km_dominant_aggregate_class.img --scenario 1 --seed 0`
* **note**: the window is cut from the same raster as the region, and the command checks that the cut equals the run's stored initial grid before it draws.

## `region_converted_final`

Study region `final`: the 809 parcels this run converted to urban, over a greyed-out landscape.

* **files**: `region_converted_final.pdf`, `region_converted_final.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --region --scenario 1 --seed 0`
* **note**: this is what the three published metrics cannot show: they are functions of the final landscape, so two policies that develop the same number of parcels in different places can score alike

## `land_preference_arable_land`

Land preference of **Arable Land**: how the eight-cell neighbourhood of a arable land parcel is composed, initial landscape.  The dominant slice is labelled with its percentage.

* **files**: `land_preference_arable_land.pdf`, `land_preference_arable_land.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --region --scenario 1 --seed 0`
* **note**: shares are over land cells only; sea, out-of-coverage and saltwater cells are neither centres nor neighbours

## `land_preference_grassland`

Land preference of **Grassland**: how the eight-cell neighbourhood of a grassland parcel is composed, initial landscape.  The dominant slice is labelled with its percentage.

* **files**: `land_preference_grassland.pdf`, `land_preference_grassland.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --region --scenario 1 --seed 0`
* **note**: shares are over land cells only; sea, out-of-coverage and saltwater cells are neither centres nor neighbours

## `land_preference_heather_grassland`

Land preference of **Heather Grassland**: how the eight-cell neighbourhood of a heather grassland parcel is composed, initial landscape.  The dominant slice is labelled with its percentage.

* **files**: `land_preference_heather_grassland.pdf`, `land_preference_heather_grassland.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --region --scenario 1 --seed 0`
* **note**: shares are over land cells only; sea, out-of-coverage and saltwater cells are neither centres nor neighbours

## `land_preference_littoral_sediment`

Land preference of **Littoral Sediment**: how the eight-cell neighbourhood of a littoral sediment parcel is composed, initial landscape.  The dominant slice is labelled with its percentage.

* **files**: `land_preference_littoral_sediment.pdf`, `land_preference_littoral_sediment.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --region --scenario 1 --seed 0`
* **note**: shares are over land cells only; sea, out-of-coverage and saltwater cells are neither centres nor neighbours

## `land_preference_freshwater`

Land preference of **Freshwater**: how the eight-cell neighbourhood of a freshwater parcel is composed, initial landscape.  The dominant slice is labelled with its percentage.

* **files**: `land_preference_freshwater.pdf`, `land_preference_freshwater.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --region --scenario 1 --seed 0`
* **note**: shares are over land cells only; sea, out-of-coverage and saltwater cells are neither centres nor neighbours

## `land_preference_urban`

Land preference of **Urban**: how the eight-cell neighbourhood of a urban parcel is composed, initial landscape.  The dominant slice is labelled with its percentage.

* **files**: `land_preference_urban.pdf`, `land_preference_urban.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --region --scenario 1 --seed 0`
* **note**: shares are over land cells only; sea, out-of-coverage and saltwater cells are neither centres nor neighbours

## `land_preference_woodland`

Land preference of **Woodland**: how the eight-cell neighbourhood of a woodland parcel is composed, initial landscape.  The dominant slice is labelled with its percentage.

* **files**: `land_preference_woodland.pdf`, `land_preference_woodland.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --region --scenario 1 --seed 0`
* **note**: shares are over land cells only; sea, out-of-coverage and saltwater cells are neither centres nor neighbours

## `land_preference_suburban`

Land preference of **Suburban**: how the eight-cell neighbourhood of a suburban parcel is composed, initial landscape.  The dominant slice is labelled with its percentage.

* **files**: `land_preference_suburban.pdf`, `land_preference_suburban.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --region --scenario 1 --seed 0`
* **note**: Suburban is empty: the LCM2015 *aggregate* product merges suburban and urban into one 'built-up areas and gardens' class, and only the 21 *target* classes - a product not available here - split them.
* **note**: shares are over land cells only; sea, out-of-coverage and saltwater cells are neither centres nor neighbours

## `colours_legend_pie`

The eight land classes of the published legend, in the published colours.

* **files**: `colours_legend_pie.pdf`, `colours_legend_pie.png`
* **source**: `chapter4_qolu/results/campaign_final`
* **command**: `python -m chapter4_qolu.make_figures --campaign chapter4_qolu/results/campaign_final --region --scenario 1 --seed 0`
* **note**: Suburban is empty: the LCM2015 *aggregate* product merges suburban and urban into one 'built-up areas and gardens' class, and only the 21 *target* classes - a product not available here - split them.
