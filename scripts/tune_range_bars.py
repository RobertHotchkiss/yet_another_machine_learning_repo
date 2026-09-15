"""Resumable Optuna tuning on validation folds only.

The final chronological test partition is intentionally not fitted or scored
here.  Copy the selected parameters to the notebook and run its final test
once, after tuning is complete.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Make ``python scripts/tune_range_bars.py`` work from an uninstalled checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import optuna

from range_bars_ml import ExperimentConfig, FeatureConfig, ModelConfig, load_range_bars, run_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune range-bar LightGBM settings using validation folds only.")
    parser.add_argument("--parquet-path", type=Path, required=True)
    parser.add_argument("--slice-offset", type=int, default=-50_000)
    parser.add_argument("--slice-length", type=int, default=30_000)
    parser.add_argument("--windows", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--time-column", default="close_time")
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--minimum-coverage", type=float, default=0.0)
    parser.add_argument("--abstention-penalty", type=float, default=-0.001)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=155)
    parser.add_argument("--study-name", default="range_bar_lgbm_validation")
    parser.add_argument("--storage-path", type=Path, default=PROJECT_ROOT / "data" / "range_bar_tuning.sqlite3")
    return parser


def sampled_model_config(trial: optuna.Trial, args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        calibration_fraction=args.calibration_fraction,
        random_state=args.seed,
        n_estimators=trial.suggest_int("n_estimators", 200, 1_500),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
        num_leaves=trial.suggest_int("num_leaves", 7, 127),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
        subsample=trial.suggest_float("subsample", 0.60, 1.00),
        # LightGBM only applies row subsampling when this is non-zero.
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.60, 1.00),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    )


def make_objective(frame: Any, args: argparse.Namespace):
    feature = FeatureConfig(windows=tuple(args.windows), time_column=args.time_column)

    def objective(trial: optuna.Trial) -> float:
        model = sampled_model_config(trial, args)
        config = ExperimentConfig(
            feature=feature, model=model, test_fraction=args.test_fraction,
            n_folds=args.folds, minimum_coverage=args.minimum_coverage,
            abstention_penalty=args.abstention_penalty,
        )
        trial.set_user_attr("model_config", asdict(model))
        try:
            result = run_validation(frame, config)
        except ValueError as error:
            # Invalid coverage/no-signal configurations remain visible and rank last.
            trial.set_user_attr("status", "invalid")
            trial.set_user_attr("error", str(error))
            return float("-inf")
        trial.set_user_attr("status", "complete")
        trial.set_user_attr("thresholds", {"long": result.long_threshold, "short": result.short_threshold})
        trial.set_user_attr("metrics", result.validation_metrics)
        return result.validation_metrics["score"]

    return objective


def trial_summary(study: optuna.Study) -> list[dict[str, Any]]:
    return [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
            "attributes": trial.user_attrs,
        }
        for trial in study.trials
    ]


def json_safe(value: Any) -> Any:
    """Convert non-finite diagnostic values (for empty directions) to null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.trials < 1:
        raise ValueError("--trials must be positive")
    args.storage_path.parent.mkdir(parents=True, exist_ok=True)
    bars = load_range_bars(args.parquet_path).slice(args.slice_offset, args.slice_length)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{args.storage_path.resolve().as_posix()}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(make_objective(bars, args), n_trials=args.trials)
    summary_path = args.storage_path.with_suffix(".json")
    summary_path.write_text(json.dumps(json_safe({"study_name": study.study_name, "trials": trial_summary(study)}), indent=2, allow_nan=False), encoding="utf-8")
    if study.best_trial.value == float("-inf"):
        print("No valid trial met the requested scoring constraints.")
        return 2
    best = study.best_trial
    print(f"Best validation score: {best.value:.6f} (trial {best.number})")
    print(f"Parameters: {json.dumps(best.params, sort_keys=True)}")
    print(f"Trial summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
