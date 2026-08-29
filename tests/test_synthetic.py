import numpy as np

from world_state.synthetic import (
    aggregate_metrics,
    build_baseline_forecasts,
    build_synthetic_dataset,
)


def test_synthetic_dataset_has_expected_coordinates_and_physical_ranges():
    ds = build_synthetic_dataset(periods=20)

    assert set(ds.dims) == {"time", "latitude", "longitude"}
    assert ds.sizes["time"] == 20
    assert float(ds.humidity.min()) >= 0
    assert float(ds.humidity.max()) <= 100
    assert float(ds.precipitation.min()) >= 0
    assert np.all(np.diff(ds.time.values) == np.timedelta64(6, "h"))


def test_forecasts_never_use_future_data_at_issue_time():
    ds = build_synthetic_dataset(periods=20)
    forecasts = build_baseline_forecasts(ds, horizons=(6, 24))

    assert (forecasts.valid_at > forecasts.issued_at).all()
    assert set(forecasts.forecast_horizon_hours) == {6, 24}
    assert set(forecasts.model) == {"persistence", "climatology"}


def test_persistence_is_exactly_the_issued_value():
    ds = build_synthetic_dataset(periods=12)
    forecasts = build_baseline_forecasts(ds, horizons=(6,))
    row = forecasts.loc[
        (forecasts.model == "persistence")
        & (forecasts.target == "temperature")
        & (forecasts.latitude == ds.latitude.values[0])
        & (forecasts.longitude == ds.longitude.values[0])
    ].iloc[0]

    expected = float(
        ds.temperature.sel(
            time=row.issued_at, latitude=row.latitude, longitude=row.longitude
        ).values
    )
    assert row.prediction == expected


def test_metric_table_is_complete():
    ds = build_synthetic_dataset(periods=20)
    forecasts = build_baseline_forecasts(ds, horizons=(6, 12))
    metrics = aggregate_metrics(forecasts)

    assert len(metrics) == 2 * 6 * 2
    assert (metrics.rmse >= 0).all()
    assert (metrics.samples > 0).all()
