from pathlib import Path

from range_bars_ml.data import prepare_dataset
from range_bars_ml.model import ModelConfig, SelectiveSignalModel
from range_bars_ml.pipeline import ExperimentConfig, run_experiment

from .test_features import bars


def test_pipeline_scores_and_persists_selective_model():
    result = run_experiment(
        bars(2_000),
        ExperimentConfig(n_folds=2, model=ModelConfig(n_estimators=5)),
    )
    featured, _ = prepare_dataset(bars(100))
    prediction = result.model.predict(featured.tail(2))
    assert prediction.columns == ["probability_up", "signal"]
    assert set(prediction["signal"]) <= {"long", "short", "abstain"}

    artifact = Path(".range-bars-test-model.joblib")
    try:
        result.model.save(artifact)
        restored = SelectiveSignalModel.load(artifact)
        assert restored.predict(featured.tail(1))["signal"].item() in {"long", "short", "abstain"}
    finally:
        artifact.unlink(missing_ok=True)


def test_edge_margin_is_based_on_development_data_and_saved():
    result = run_experiment(
        bars(2_000),
        ExperimentConfig(n_folds=2, edge_margin=0.02, model=ModelConfig(n_estimators=2, n_jobs=1)),
    )
    metadata = result.model.metadata
    assert metadata["edge_margin"] == 0.02
    assert result.model.long_threshold >= metadata["long_gate"]
    assert result.model.short_threshold <= metadata["short_gate"]
