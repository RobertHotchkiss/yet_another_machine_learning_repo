# Selective range-bar direction model

The library predicts the direction of the next completed range bar from data
available at the current bar close. It only emits `long` or `short` when a
calibrated probability passes its validation-selected confidence threshold;
otherwise it emits `abstain`.

Evaluation scores a correct directional prediction as `+1`, an incorrect one
as `-1`, and an abstention as `-0.001`. Thresholds are selected to maximize
this net validation score, so the model may deliberately abstain throughout
when it has no useful edge.

```python
from range_bars_ml import ExperimentConfig, load_range_bars, run_experiment

bars = load_range_bars(r"C:\data\range_bars.parquet")
result = run_experiment(bars, ExperimentConfig())

print(result.validation_metrics)
print(result.test_metrics)  # untouched newest 20% of labelled bars
result.model.save("artifacts/range_bar_model.joblib")
```

For inference, first build a feature-ready frame using the same feature
configuration stored in `result.model.metadata["feature_config"]`, then call
`result.model.predict(feature_frame)`. Preserve sufficient completed-bar
history (at least the largest configured lookback) when generating live
features.

## Feature families

`features.py` keeps each causal family in a separate builder. Raw market
columns are used only while deriving features and are excluded from the
returned feature dataframe. In addition to normalised bar shape, returns, and
volume imbalance, every configured window includes:

- bought, sold, and imbalance dollar shares relative to that same window's
  total dollar volume; and
- one feature set for every bar in the window: its time share, bought/sold/
  total dollar-volume shares, and binary up/down direction. `lag_0` is the
  current completed bar; larger lags move backwards in time.

Every share denominator is calculated over the exact same trailing window.

## Live Binance-data / Deribit-execution trading

`scripts/trade_binance_to_deribit.py` receives public live Binance USD-M
BTCUSDT trade ticks, constructs the 0.1% `renko_ish` bars, and computes all
model features from Binance data only. It executes its market orders only on
Deribit mainnet `BTC-PERPETUAL`; Binance API credentials are not used.

The native range-bar engine must be built once before use:

```powershell
cd crates\live_range_bars_rs
maturin develop --release
cd ..\..
$env:DERIBIT_API_KEY = "..."
$env:DERIBIT_API_SECRET = "..."
.venv\Scripts\python scripts\trade_binance_to_deribit.py --live-execution
```

The runner refuses to start without `--live-execution`, credentials, a flat
Deribit BTC perpetual position, and no working Deribit BTC perpetual orders.
It enters with $10 market orders, uses $20 market orders to reverse a confirmed
$10 position, and uses a reduce-only $10 market order to flatten on abstention.
State and append-only signal/execution logs are written beneath `data\live`.
Stop it with Ctrl+C.
