"""Reporting helpers for the four-scenario campaign runner.

Split out of :mod:`chapter4_qolu.run_scenarios` so that each concern stays a
small, separately testable file:

* :mod:`~chapter4_qolu.reporting.paths` - result-file layout and resume logic;
* :mod:`~chapter4_qolu.reporting.stats` - the paired t-test, Cohen's d and the
  mean / SD / SE summaries, implemented on numpy alone;
* :mod:`~chapter4_qolu.reporting.aggregate` - per-run records to
  ``summary.json``, including the three normalisation views;
* :mod:`~chapter4_qolu.reporting.tables` - ``tables.md`` and ``tables.tex`` in
  the layout of the paper's Tables 4-7;
* :mod:`~chapter4_qolu.reporting.pattern` - ``pattern_check.md``, the
  comparison against the published *ranking*, not the published values.

Nothing here runs an environment or an agent, so a campaign's tables can be
regenerated from the JSON records alone.
"""

from __future__ import annotations

from chapter4_qolu.reporting.aggregate import (
    ALL_METRICS,
    PUBLISHED_METRICS,
    REFERENCE_METHOD,
    aggregate_records,
    load_run_records,
    min_max_normalise,
    rank_methods,
    saturation_iteration,
)
from chapter4_qolu.reporting.paths import (
    ScaleConfig,
    campaign_directory,
    is_complete,
    parse_scale_config,
    run_json_path,
    should_run,
)
from chapter4_qolu.reporting.pattern import check_cell, render_pattern_check
from chapter4_qolu.reporting.stats import (
    PairedTTest,
    SampleSummary,
    cohens_d_paired,
    paired_t_test,
    summarise,
)
from chapter4_qolu.reporting.tables import render_latex, render_markdown

__all__ = [
    "ALL_METRICS",
    "PUBLISHED_METRICS",
    "REFERENCE_METHOD",
    "PairedTTest",
    "SampleSummary",
    "ScaleConfig",
    "aggregate_records",
    "campaign_directory",
    "check_cell",
    "cohens_d_paired",
    "is_complete",
    "load_run_records",
    "min_max_normalise",
    "paired_t_test",
    "parse_scale_config",
    "rank_methods",
    "render_latex",
    "render_markdown",
    "render_pattern_check",
    "run_json_path",
    "saturation_iteration",
    "should_run",
    "summarise",
]
