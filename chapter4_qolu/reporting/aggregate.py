"""Aggregation of a campaign's per-run records into ``summary.json``.

Three *views* of the same five metrics are carried side by side and are never
silently mixed (each table caption says which one it shows):

``raw``
    the metric values exactly as :mod:`chapter4_qolu.env.evaluation` defines
    them, with distances normalised by the grid diagonal.  On the Northern
    Ireland land cover that default squeezes ``land_use`` into roughly
    0.03-0.08 for *every* method and scale, because the nearest bog /
    mountain / heath parcel is almost always a few parcels away while the
    diagonal of a 50 x 50 grid is 70 parcels long.  The
    column is therefore nearly uninformative on its own, which is exactly why
    the other two views exist.

``absolute``
    the same metrics recomputed with ``RewardConfig.max_distance`` set to a
    physical distance (``--max-distance-km``, 10 km by default), so a developed
    parcel 10 km from the protected habitat scores 1.0 whatever the extent.
    This is a *different reading of the metric*, not a rescaling of ``raw``.

``minmax``
    a cross-method min-max rescaling *within* one configuration x scenario x
    metric cell: the worst method scores 0 and the best 1.  This is the
    transparent analogue of the published tables' "normalized between 0 and 1";
    it is ordinal information only and says nothing about absolute quality.

Paired tests are computed per view - ``tests_by_view["raw"]`` on RAW and
``tests_by_view["absolute"]`` on the absolute view - so a table can mark the
test of the view it shows.  min-max is an affine image of RAW within a cell,
which leaves a paired t statistic unchanged, so it uses the RAW test.  The
``tests`` block carries the RAW tests of the three published metrics.

``c_align`` - the total-variation conversion metric - is lower-is-better, and
every ordering here (the ranking, the min-max rescaling) respects that.  A
record that lacks it is scored on it from its stored grids
(:mod:`chapter4_qolu.reporting.derived`) and ``derived_metrics`` counts how
many were.

Design note: ``minmax`` rescales the ``raw`` view, because ``raw`` is the
definitional one.  Min-max is purely ordinal on whatever it is applied to, but
``absolute`` clips distances at ``max_distance`` before averaging, so the two
views can order methods differently; applying it to ``raw`` keeps the main
tables tied to the metric definitions the rest of the package uses.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from chapter4_qolu.reporting.derived import (
    derived_metric_counts,
    with_derived_metrics,
)
from chapter4_qolu.reporting.stats import paired_t_test, summarise

__all__ = [
    "ALL_METRICS",
    "LOWER_IS_BETTER",
    "PUBLISHED_METRICS",
    "REFERENCE_METHOD",
    "TESTED_METRICS",
    "TESTED_VIEWS",
    "VIEW_DESCRIPTIONS",
    "aggregate_records",
    "load_run_records",
    "min_max_normalise",
    "rank_methods",
    "saturation_iteration",
]

#: The three columns of the paper's Tables 4-7.
PUBLISHED_METRICS: tuple[str, ...] = ("land_use", "woodland", "urban")
#: Every metric :func:`chapter4_qolu.env.evaluation.evaluate_landscape` returns.
ALL_METRICS: tuple[str, ...] = (
    "land_use",
    "woodland",
    "urban",
    "conversion_alignment",
    "c_align",
    "well_being",
)
#: Metrics for which a *lower* value is better (the total-variation ``C_align``).
LOWER_IS_BETTER: frozenset[str] = frozenset({"c_align"})
#: Metrics the per-view paired tests cover; the conversion count is tested too.
TESTED_METRICS: tuple[str, ...] = (*PUBLISHED_METRICS, "c_align")
#: Views a paired test is computed on (min-max reuses the RAW test).
TESTED_VIEWS: tuple[str, ...] = ("raw", "absolute")
#: The method every baseline is paired against ("US" in the published tables).
REFERENCE_METHOD = "qolu_buia"

#: Design note: a flat cell (every method identical) maps to 0.5 rather than to
#: 0 or 1, so that a tie is not silently presented as a win or a loss.
_FLAT_CELL_VALUE = 0.5

VIEW_DESCRIPTIONS: Mapping[str, str] = {
    "raw": (
        "RAW metric values as chapter4_qolu/env/evaluation.py defines them; "
        "distances are normalised by the grid diagonal."
    ),
    "absolute": (
        "absolute-normalised: the same metrics with RewardConfig.max_distance "
        "set to a physical distance, so proximity saturates at a fixed number "
        "of kilometres instead of at the grid diagonal."
    ),
    "minmax": (
        "cross-method min-max within one configuration x scenario x metric "
        "cell (worst method 0, best method 1), rescaled from the RAW view; the "
        "transparent analogue of the published tables' 'normalized between 0 "
        "and 1'."
    ),
}

#: Urban cover share above which a landscape counts as saturated.
SATURATION_THRESHOLD = 0.99


def saturation_iteration(
    urban_fraction: Sequence[float], threshold: float = SATURATION_THRESHOLD
) -> int | None:
    """First iteration whose urban cover reaches ``threshold``, else ``None``.

    Design note: the comparison is ``>=`` here, whereas
    :func:`chapter4_qolu.sweep_saturation.first_saturated_iteration` uses a
    strict ``>``.  The campaign reports "the landscape is full" and
    ``urban_saturation`` itself defaults to exactly 0.99, so a run sitting on
    the threshold has to count as saturated.
    """
    for iteration, value in enumerate(urban_fraction):
        if float(value) >= threshold:
            return int(iteration)
    return None


def rank_methods(
    values: Mapping[str, float], lower_is_better: bool = False
) -> tuple[str, ...]:
    """Method names ordered best first (highest value, or lowest if asked).

    Ties are broken by method name so that a summary is byte-for-byte
    reproducible.
    """
    sign = 1.0 if lower_is_better else -1.0
    return tuple(
        name
        for name, _ in sorted(
            values.items(), key=lambda item: (sign * item[1], item[0])
        )
    )


def min_max_normalise(
    values: Mapping[str, float], lower_is_better: bool = False
) -> dict[str, float]:
    """Rescale a cell's per-method values so worst = 0 and best = 1."""
    if not values:
        return {}
    lowest = min(values.values())
    highest = max(values.values())
    span = highest - lowest
    if span <= 0.0:
        return {name: _FLAT_CELL_VALUE for name in values}
    if lower_is_better:
        return {name: (highest - value) / span for name, value in values.items()}
    return {name: (value - lowest) / span for name, value in values.items()}


def load_run_records(campaign_dir: Path) -> list[dict[str, Any]]:
    """Read every per-run JSON record under a campaign directory."""
    directory = Path(campaign_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"campaign directory not found: {directory}")
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*/scenario*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(record, dict) and "final_metrics" in record:
            records.append(record)
    return records


def _metric_values(record: Mapping[str, Any], view: str) -> Mapping[str, float]:
    key = "final_metrics" if view == "raw" else "final_metrics_absolute"
    metrics = record.get(key) or record.get("final_metrics") or {}
    return {name: float(metrics[name]) for name in ALL_METRICS if name in metrics}


def _series(runs: Sequence[Mapping[str, Any]], view: str, metric: str) -> list[float]:
    values = []
    for run in runs:
        metrics = _metric_values(run, view)
        if metric in metrics:
            values.append(metrics[metric])
    return values


def _method_block(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(runs, key=lambda run: int(run["seed"]))
    block: dict[str, Any] = {
        "n_seeds": len(ordered),
        "seeds": [int(run["seed"]) for run in ordered],
        "runtime_seconds": summarise(
            [float(run.get("runtime_seconds", 0.0)) for run in ordered]
        ).as_record(),
        "total_conversions": summarise(
            [float(run.get("total_conversions", 0)) for run in ordered]
        ).as_record(),
        "saturation_iterations": [run.get("saturation_iteration") for run in ordered],
    }
    for view in ("raw", "absolute"):
        block[view] = {
            metric: summarise(values).as_record()
            for metric in ALL_METRICS
            if (values := _series(ordered, view, metric))
        }
    return block


def _add_minmax_view(methods: dict[str, dict[str, Any]]) -> None:
    """Fill each method's ``minmax`` block from the cell's ``raw`` means."""
    for method in methods.values():
        method["minmax"] = {}
    for metric in ALL_METRICS:
        means = {
            name: float(block["raw"][metric]["mean"])
            for name, block in methods.items()
            if metric in block["raw"]
        }
        if not means:
            continue
        lower = metric in LOWER_IS_BETTER
        scaled = min_max_normalise(means, lower_is_better=lower)
        lowest = min(means.values())
        highest = max(means.values())
        span = highest - lowest
        for name, value in scaled.items():
            raw_block = methods[name]["raw"][metric]
            raw_se = float(raw_block["se"])
            raw_sd = float(raw_block["sd"])
            methods[name]["minmax"][metric] = {
                "mean": value,
                # The same affine map applied to the spread; undefined when the
                # cell is flat, in which case there is nothing to show either.
                "sd": (raw_sd / span) if span > 0.0 else 0.0,
                "se": (raw_se / span) if span > 0.0 else 0.0,
                "n": int(raw_block["n"]),
                # ... and to each seed, so the min-max view can be re-tested or
                # re-plotted from summary.json alone.
                "values": [
                    (
                        (
                            (highest - float(seed_value))
                            if lower
                            else (float(seed_value) - lowest)
                        )
                        / span
                        if span > 0.0
                        else value
                    )
                    for seed_value in raw_block.get("values", [])
                ],
            }


def _with_conversions(run: Mapping[str, Any], view: str) -> dict[str, float]:
    values = dict(_metric_values(run, view))
    if run.get("total_conversions") is not None:
        values["total_conversions"] = float(run["total_conversions"])
    return values


def _paired_tests(
    runs_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    reference: str,
    view: str = "raw",
    metrics: Sequence[str] = PUBLISHED_METRICS,
    skip_missing: bool = False,
) -> dict[str, dict[str, dict[str, object]]]:
    """Paired tests of every other method against ``reference``, seed by seed.

    ``view`` decides which metric values are paired; ``"total_conversions"``
    may be named in ``metrics`` and is the same in every view.  With
    ``skip_missing`` a metric no seed pair carries is left out rather than
    reported as an impossible test - the per-view blocks use that, the RAW
    ``tests`` block keeps its every-published-metric shape.
    """
    if reference not in runs_by_method:
        return {}
    reference_by_seed = {
        int(run["seed"]): _with_conversions(run, view)
        for run in runs_by_method[reference]
    }
    tests: dict[str, dict[str, dict[str, object]]] = {}
    for method, runs in runs_by_method.items():
        if method == reference:
            continue
        by_seed = {int(run["seed"]): _with_conversions(run, view) for run in runs}
        shared = sorted(set(by_seed) & set(reference_by_seed))
        tests[method] = {}
        for metric in metrics:
            paired = [
                seed
                for seed in shared
                if metric in reference_by_seed[seed] and metric in by_seed[seed]
            ]
            if skip_missing and not paired:
                continue
            treatment = [reference_by_seed[seed][metric] for seed in paired]
            control = [by_seed[seed][metric] for seed in paired]
            tests[method][metric] = paired_t_test(treatment, control).as_record()
    return tests


def _scenario_block(
    runs: Sequence[Mapping[str, Any]], reference: str
) -> dict[str, Any]:
    runs_by_method: dict[str, list[Mapping[str, Any]]] = {}
    for run in runs:
        runs_by_method.setdefault(str(run["method"]), []).append(run)

    methods = {
        method: _method_block(method_runs)
        for method, method_runs in sorted(runs_by_method.items())
    }
    _add_minmax_view(methods)

    ranking: dict[str, list[str]] = {}
    for metric in ALL_METRICS:
        means = {
            name: float(block["raw"][metric]["mean"])
            for name, block in methods.items()
            if metric in block["raw"]
        }
        if means:
            ranking[metric] = list(
                rank_methods(means, lower_is_better=metric in LOWER_IS_BETTER)
            )

    first = runs[0]
    return {
        "scenario": int(first["scenario"]),
        "weights": dict(first.get("weights", {})),
        "iterations": int(first.get("iterations", 0)),
        "methods": methods,
        "ranking": ranking,
        "tests": _paired_tests(runs_by_method, reference),
        "tests_by_view": {
            view: _paired_tests(
                runs_by_method,
                reference,
                view=view,
                metrics=(*TESTED_METRICS, "total_conversions"),
                skip_missing=True,
            )
            for view in TESTED_VIEWS
        },
    }


def _configuration_record(record: Mapping[str, Any]) -> dict[str, Any]:
    configuration = dict(record.get("configuration") or {})
    configuration.setdefault("label", "unknown")
    return configuration


def _absolute_distance(records: Sequence[Mapping[str, Any]]) -> float | None:
    distances = {
        float(record["absolute_max_distance_km"])
        for record in records
        if record.get("absolute_max_distance_km") is not None
    }
    return distances.pop() if len(distances) == 1 else None


def aggregate_records(
    records: Iterable[Mapping[str, Any]],
    reference: str = REFERENCE_METHOD,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate per-run records into the campaign summary payload.

    Records are read, never modified: a record that lacks a derivable
    metric is scored on it from its stored grids in a copy.
    """
    originals = list(records)
    if not originals:
        raise ValueError("cannot aggregate an empty campaign")
    runs = [with_derived_metrics(record) for record in originals]

    by_configuration: dict[str, list[Mapping[str, Any]]] = {}
    configurations: dict[str, dict[str, Any]] = {}
    for run in runs:
        configuration = _configuration_record(run)
        label = str(configuration["label"])
        configurations.setdefault(label, configuration)
        by_configuration.setdefault(label, []).append(run)

    summary_configurations = []
    for label in sorted(by_configuration):
        by_scenario: dict[int, list[Mapping[str, Any]]] = {}
        for run in by_configuration[label]:
            by_scenario.setdefault(int(run["scenario"]), []).append(run)
        summary_configurations.append(
            {
                **configurations[label],
                "scenarios": [
                    _scenario_block(by_scenario[scenario], reference)
                    for scenario in sorted(by_scenario)
                ],
            }
        )

    return {
        "campaign": dict(metadata or {}),
        "reference_method": reference,
        "published_metrics": list(PUBLISHED_METRICS),
        "metrics": list(ALL_METRICS),
        "views": dict(VIEW_DESCRIPTIONS),
        "absolute_max_distance_km": _absolute_distance(runs),
        "tested_views": list(TESTED_VIEWS),
        "lower_is_better": sorted(LOWER_IS_BETTER),
        "derived_metrics": derived_metric_counts(originals),
        "n_runs": len(runs),
        "configurations": summary_configurations,
    }
