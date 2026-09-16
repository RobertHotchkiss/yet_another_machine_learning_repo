//! Stateful, live-safe implementation of the project's 0.1% renko-ish bars.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
struct BarState {
    first_trade_id: i64,
    last_trade_id: i64,
    timestamp_us: i64,
    close_time_us: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    total_sold_volume: f64,
    total_bought_volume: f64,
    total_sold_dollar_volume: f64,
    total_bought_dollar_volume: f64,
}

impl BarState {
    fn new(trade: &Trade) -> Self {
        let (sold_volume, bought_volume) = if trade.is_buyer_maker {
            (trade.quantity, 0.0)
        } else {
            (0.0, trade.quantity)
        };
        let (sold_dollar, bought_dollar) = if trade.is_buyer_maker {
            (trade.quote_quantity, 0.0)
        } else {
            (0.0, trade.quote_quantity)
        };
        Self {
            first_trade_id: trade.trade_id,
            last_trade_id: trade.trade_id,
            timestamp_us: trade.timestamp_us,
            close_time_us: trade.timestamp_us,
            open: trade.price,
            high: trade.price,
            low: trade.price,
            close: trade.price,
            total_sold_volume: sold_volume,
            total_bought_volume: bought_volume,
            total_sold_dollar_volume: sold_dollar,
            total_bought_dollar_volume: bought_dollar,
        }
    }

    fn update(&mut self, trade: &Trade) {
        self.high = self.high.max(trade.price);
        self.low = self.low.min(trade.price);
        self.close = trade.price;
        self.close_time_us = trade.timestamp_us;
        self.last_trade_id = trade.trade_id;
        if trade.is_buyer_maker {
            self.total_sold_volume += trade.quantity;
            self.total_sold_dollar_volume += trade.quote_quantity;
        } else {
            self.total_bought_volume += trade.quantity;
            self.total_bought_dollar_volume += trade.quote_quantity;
        }
    }
}

#[derive(Debug)]
struct Trade {
    trade_id: i64,
    price: f64,
    quantity: f64,
    quote_quantity: f64,
    is_buyer_maker: bool,
    timestamp_us: i64,
}

#[pyclass(get_all, frozen)]
#[derive(Clone)]
struct CompletedBar {
    first_trade_id: i64,
    last_trade_id: i64,
    timestamp_us: i64,
    close_time_us: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    total_sold_volume: f64,
    total_bought_volume: f64,
    total_sold_dollar_volume: f64,
    total_bought_dollar_volume: f64,
}

impl From<BarState> for CompletedBar {
    fn from(bar: BarState) -> Self {
        Self {
            first_trade_id: bar.first_trade_id,
            last_trade_id: bar.last_trade_id,
            timestamp_us: bar.timestamp_us,
            close_time_us: bar.close_time_us,
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            total_sold_volume: bar.total_sold_volume,
            total_bought_volume: bar.total_bought_volume,
            total_sold_dollar_volume: bar.total_sold_dollar_volume,
            total_bought_dollar_volume: bar.total_bought_dollar_volume,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct Snapshot {
    range_percent: f64,
    last_trade_id: Option<i64>,
    current: Option<BarState>,
}

#[pyclass]
struct RenkoIshAggregator {
    range_percent: f64,
    last_trade_id: Option<i64>,
    current: Option<BarState>,
}

#[pymethods]
impl RenkoIshAggregator {
    #[new]
    fn new(range_percent: f64) -> PyResult<Self> {
        if !range_percent.is_finite() || range_percent <= 0.0 {
            return Err(PyValueError::new_err(
                "range_percent must be finite and positive",
            ));
        }
        Ok(Self {
            range_percent,
            last_trade_id: None,
            current: None,
        })
    }

    /// Process one ordered Binance trade. Duplicate/older IDs are ignored.
    fn process_trade(
        &mut self,
        trade_id: i64,
        price: f64,
        quantity: f64,
        quote_quantity: f64,
        is_buyer_maker: bool,
        timestamp_us: i64,
    ) -> PyResult<Option<CompletedBar>> {
        if !price.is_finite()
            || price <= 0.0
            || !quantity.is_finite()
            || quantity < 0.0
            || !quote_quantity.is_finite()
            || quote_quantity < 0.0
        {
            return Err(PyValueError::new_err(
                "trade price must be positive and quantities must be finite and non-negative",
            ));
        }
        if self.last_trade_id.is_some_and(|last| trade_id <= last) {
            return Ok(None);
        }
        let trade = Trade {
            trade_id,
            price,
            quantity,
            quote_quantity,
            is_buyer_maker,
            timestamp_us,
        };
        self.last_trade_id = Some(trade_id);
        let Some(bar) = self.current.as_mut() else {
            self.current = Some(BarState::new(&trade));
            return Ok(None);
        };
        // The test is intentionally before update; the trigger trade is then included.
        let up_move = bar.high.max(trade.price) - bar.open;
        let down_move = bar.open - bar.low.min(trade.price);
        let closes =
            up_move > bar.open * self.range_percent || down_move > bar.open * self.range_percent;
        bar.update(&trade);
        if closes {
            Ok(self.current.take().map(CompletedBar::from))
        } else {
            Ok(None)
        }
    }

    #[getter]
    fn last_trade_id(&self) -> Option<i64> {
        self.last_trade_id
    }

    fn snapshot(&self) -> PyResult<String> {
        serde_json::to_string(&Snapshot {
            range_percent: self.range_percent,
            last_trade_id: self.last_trade_id,
            current: self.current.clone(),
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[staticmethod]
    fn from_snapshot(snapshot: &str) -> PyResult<Self> {
        let state: Snapshot = serde_json::from_str(snapshot)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Self::new(state.range_percent).map(|mut aggregator| {
            aggregator.last_trade_id = state.last_trade_id;
            aggregator.current = state.current;
            aggregator
        })
    }
}

#[pymodule]
fn live_range_bars_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CompletedBar>()?;
    module.add_class::<RenkoIshAggregator>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trade(id: i64, price: f64, maker: bool) -> (i64, f64, f64, f64, bool, i64) {
        (id, price, 2.0, price * 2.0, maker, id * 1_000)
    }

    #[test]
    fn closes_strictly_after_the_point_one_percent_boundary_and_includes_trigger() {
        let mut aggregator = RenkoIshAggregator::new(0.001).unwrap();
        assert!(
            aggregator
                .process_trade_tuple(trade(1, 100.0, false))
                .unwrap()
                .is_none()
        );
        assert!(
            aggregator
                .process_trade_tuple(trade(2, 100.1, false))
                .unwrap()
                .is_none()
        );
        let bar = aggregator
            .process_trade_tuple(trade(3, 100.1001, true))
            .unwrap()
            .unwrap();
        assert_eq!(bar.first_trade_id, 1);
        assert_eq!(bar.last_trade_id, 3);
        assert_eq!(bar.total_bought_volume, 4.0);
        assert_eq!(bar.total_sold_volume, 2.0);
    }

    #[test]
    fn snapshot_recovers_active_bar_and_deduplicates() {
        let mut aggregator = RenkoIshAggregator::new(0.001).unwrap();
        aggregator
            .process_trade_tuple(trade(1, 100.0, false))
            .unwrap();
        let snapshot = aggregator.snapshot().unwrap();
        let mut restored = RenkoIshAggregator::from_snapshot(&snapshot).unwrap();
        assert!(
            restored
                .process_trade_tuple(trade(1, 110.0, false))
                .unwrap()
                .is_none()
        );
        assert!(
            restored
                .process_trade_tuple(trade(2, 100.2, true))
                .unwrap()
                .is_some()
        );
    }

    impl RenkoIshAggregator {
        fn process_trade_tuple(
            &mut self,
            trade: (i64, f64, f64, f64, bool, i64),
        ) -> PyResult<Option<CompletedBar>> {
            self.process_trade(trade.0, trade.1, trade.2, trade.3, trade.4, trade.5)
        }
    }
}
