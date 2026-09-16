import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def runner_module():
    path = Path(__file__).parents[1] / "scripts" / "trade_binance_to_deribit.py"
    spec = importlib.util.spec_from_file_location("trade_binance_to_deribit_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_requires_explicit_live_execution(runner_module, monkeypatch):
    monkeypatch.delenv("DERIBIT_API_KEY", raising=False)
    monkeypatch.delenv("DERIBIT_API_SECRET", raising=False)

    with pytest.raises(RuntimeError, match="--live-execution"):
        runner_module.require_live_execution_credentials(runner_module.build_parser().parse_args([]))


def test_cli_requires_deribit_mainnet_credentials_after_arming(runner_module, monkeypatch):
    monkeypatch.delenv("DERIBIT_API_KEY", raising=False)
    monkeypatch.delenv("DERIBIT_API_SECRET", raising=False)
    args = runner_module.build_parser().parse_args(["--live-execution"])

    with pytest.raises(RuntimeError, match="DERIBIT_API_KEY"):
        runner_module.require_live_execution_credentials(args)

    monkeypatch.setenv("DERIBIT_API_KEY", "test-key")
    monkeypatch.setenv("DERIBIT_API_SECRET", "test-secret")
    runner_module.require_live_execution_credentials(args)


def test_runner_uses_distinct_binance_data_and_deribit_execution_instruments(runner_module):
    assert str(runner_module.BINANCE_INSTRUMENT) == "BTCUSDT-PERP.BINANCE"
    assert str(runner_module.DERIBIT_INSTRUMENT) == "BTC-PERPETUAL.DERIBIT"
