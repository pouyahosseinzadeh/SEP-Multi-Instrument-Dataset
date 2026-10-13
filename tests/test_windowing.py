import numpy as np
import pandas as pd

from sep_dataset.dataset import extract_window


def test_extract_window_orders_oldest_to_newest():
    index = pd.date_range("2024-01-01T00:00:00Z", periods=12, freq="5min")
    series = pd.Series(np.arange(12), index=index)
    values, missing = extract_window(
        series=series,
        anchor_time=pd.Timestamp("2024-01-01T01:00:00Z"),
        lookback_hours=1,
        cadence="5min",
        steps=12,
        fill_value=-9999,
    )
    assert missing == 0
    assert values.tolist() == list(range(12))


def test_extract_window_fills_empty_series():
    values, missing = extract_window(
        series=pd.Series(dtype=float),
        anchor_time=pd.Timestamp("2024-01-01T01:00:00Z"),
        lookback_hours=1,
        cadence="5min",
        steps=12,
        fill_value=-9999,
    )
    assert missing == 1.0
    assert (values == -9999).all()
