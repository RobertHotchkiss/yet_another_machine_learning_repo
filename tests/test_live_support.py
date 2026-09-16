from types import SimpleNamespace
import json

from range_bars_ml.data import prepare_dataset
from range_bars_ml.features import FeatureConfig
from range_bars_ml.live_support import LiveScoreboard, LiveStateStore, latest_feature_frame

from .test_features import bars


def test_live_feature_warmup_returns_only_a_complete_model_row():
    source = bars(40)
    _, columns = prepare_dataset(source)
    model = SimpleNamespace(metadata={"feature_config": {"windows": tuple(range(1, 11)), "time_column": "close_time"}}, feature_columns=columns)
    assert latest_feature_frame([], model) is None
    frame = latest_feature_frame(source.to_dicts(), model)
    assert frame is not None
    assert frame.columns == columns
    assert frame.height == 1


def test_live_scoreboard_waits_for_next_bar_to_settle_signal():
    scoreboard = LiveScoreboard(-0.001)
    first = {"open": 100.0, "close": 101.0, "close_time": bars(1)["close_time"].item()}
    second = {"open": 100.0, "close": 99.0, "close_time": bars(2)["close_time"].item(1)}
    assert scoreboard.settle_then_record(first, 0.8, "long") is None
    settled = scoreboard.settle_then_record(second, 0.2, "short")
    assert settled is not None
    assert settled["target"] == 0
    assert settled["score"] == -1.0
    # The append-only event writer makes empty directional precision JSON-safe.
    assert json.dumps(LiveStateStore._json_row({"comparison": scoreboard.comparison()}), allow_nan=False)


def test_live_state_event_payload_makes_nonfinite_metrics_json_safe():
    payload = LiveStateStore._json_row({"precision": float("nan")})
    assert payload == {"precision": None}
    assert json.dumps(payload, allow_nan=False)
