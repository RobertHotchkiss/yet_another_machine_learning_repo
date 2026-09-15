import importlib.util
from pathlib import Path

from range_bars_ml.data import prepare_dataset
from range_bars_ml.model import ModelConfig, fit_selective_model
from range_bars_ml.pipeline import ExperimentConfig, run_validation

from .test_features import bars


def test_model_config_complexity_settings_reach_lightgbm():
    frame, features = prepare_dataset(bars(600))
    model = fit_selective_model(
        frame, features,
        ModelConfig(n_estimators=2, min_child_samples=33, subsample=0.7,
                    subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.2, n_jobs=1),
    )
    params = model.estimator.get_params()
    assert params["min_child_samples"] == 33
    assert params["subsample"] == 0.7
    assert params["subsample_freq"] == 1
    assert params["colsample_bytree"] == 0.8
    assert params["reg_lambda"] == 1.2
    assert params["n_jobs"] == 1


def test_validation_only_does_not_predict_the_final_test_rows(monkeypatch):
    import range_bars_ml.pipeline as pipeline

    seen_rows = []
    original = pipeline.fit_selective_model

    def capture_fit(frame, *args, **kwargs):
        seen_rows.append(frame.height)
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(pipeline, "fit_selective_model", capture_fit)
    result = run_validation(bars(2_000), ExperimentConfig(n_folds=2, model=ModelConfig(n_estimators=2, n_jobs=1)))
    assert len(seen_rows) == 2
    assert result.validation_metrics["signals"] > 0
    assert result.split_plan.test.size > 0


def test_tuning_cli_defaults_and_seeded_sampling():
    script = Path(__file__).parents[1] / "scripts" / "tune_range_bars.py"
    spec = importlib.util.spec_from_file_location("tune_range_bars", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args(["--parquet-path", "bars.parquet"])
    assert args.slice_offset == -50_000
    assert args.slice_length == 30_000
    assert args.windows == list(range(1, 11))
    assert args.minimum_coverage == 0.0
    assert args.abstention_penalty == -0.001
