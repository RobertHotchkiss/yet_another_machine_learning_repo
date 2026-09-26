import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl


SCRIPT = Path(__file__).parents[1] / "scripts" / "walk_forward_overnight_range.py"
SPEC = importlib.util.spec_from_file_location("walk_forward_overnight_range", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
walk_forward = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = walk_forward
SPEC.loader.exec_module(walk_forward)


def test_month_helpers_create_exclusive_calendar_sequence():
    assert walk_forward.add_months(date(2010, 1, 1), -60) == date(2005, 1, 1)
    assert walk_forward.months_between(date(2024, 11, 1), date(2025, 2, 1)) == [
        date(2024, 11, 1), date(2024, 12, 1), date(2025, 1, 1),
    ]


def test_r_gate_maximizes_total_r_subject_to_coverage():
    validation = pl.DataFrame({
        "r_return": [-1.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    })
    gate, metrics = walk_forward.select_r_gate(
        validation, np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), 0.10,
    )

    assert 0.4 < gate <= 0.5
    assert metrics["validation_total_r"] == 12.0
    assert metrics["validation_selected_trades"] == 6.0


def test_cli_defaults_match_monthly_five_year_plan():
    args = walk_forward.build_parser().parse_args([
        "--start-month", "2010-01-01", "--end-month", "2025-01-01", "--history-months", "60",
    ])
    assert args.start_month == date(2010, 1, 1)
    assert args.end_month == date(2025, 1, 1)
    assert args.history_months == 60
    assert args.validation_months == 12
    assert args.coverage_threshold == 0.0
