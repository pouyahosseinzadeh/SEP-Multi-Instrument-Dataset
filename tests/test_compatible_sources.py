import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests

from sep_dataset.compatible_sources import (
    FEATURES, INTEGRAL_COLUMNS, LEGACY_END, CompatibleSourceStore,
    SourceDataError, parse_legacy_goes, parse_noaa_csv,
)


def _noaa_text(columns, rows):
    return "Metadata Description:\norbit:\n1998-01-01,75\ndata:\ntime_tag," + ",".join(columns) + "\n" + "\n".join(rows)


def test_legacy_predictors_use_uncorrected_differential_and_quality_flags():
    columns = []
    for feature in ("P4", "P5", "P6"):
        columns += [f"{feature}W_UNCOR_FLUX", f"{feature}W_COR_FLUX", f"{feature}W_QUAL_FLAG", f"{feature}W_NUM_PTS"]
    text = _noaa_text(columns, [
        "2014-01-01 00:00:00,0.0267,9,0,5,0.0189,9,0,5,0.00486,9,0,5",
        "2014-01-01 00:05:00,0.0267,9,1,5,0.0189,9,0,0,-99999,9,0,5",
    ])
    frame = parse_legacy_goes(text, "protons")
    assert frame[["P4", "P5", "P6"]].iloc[0].tolist() == [0.0267, 0.0189, 0.00486]
    assert frame[["P4", "P5", "P6"]].iloc[1].isna().all()
    assert "UNCOR" in frame.source_product.iloc[0]


def test_corrected_integrals_are_distinct_from_predictor_channels():
    columns = []
    for feature in INTEGRAL_COLUMNS:
        columns += [f"ZP{feature}W", f"ZP{feature}W_QUAL_FLAG"]
    frame = parse_legacy_goes(_noaa_text(columns, ["2014-01-01 00:00:00,15,0,10,0,5,0,2,0,1,0"]), "integral")
    assert frame[list(INTEGRAL_COLUMNS)].iloc[0].tolist() == [15, 10, 5, 2, 1]
    assert frame.channel_status.iloc[0] == "corrected_integral_measurement"


def test_noaa_missing_or_duplicate_data_header_fails():
    with pytest.raises(SourceDataError, match="unique"):
        parse_noaa_csv("Metadata only\n")
    with pytest.raises(SourceDataError, match="unique"):
        parse_noaa_csv("time_tag,F\n2014-01-01,1\ntime_tag,F\n2014-01-02,2\n")


def test_strict_original_channels_not_substituted_in_2025(tmp_path, monkeypatch):
    store = CompatibleSourceStore(tmp_path, progress=None)
    monkeypatch.setattr(store, "_download", lambda *a, **k: pytest.fail("Structural absence must not fetch SGPS or integral substitutes"))
    series = store.range("P4", "2025-01-01", "2025-01-02")
    quality = store.quality_range("P4", "2025-01-01", "2025-01-02")
    assert len(series) == 288
    assert series.isna().all()
    assert series.attrs["structural_missing"].all()
    assert quality.channel_status.eq("outside_original_channel_coverage").all()
    assert quality.source_satellite.eq("unavailable").all()


def test_no_harmonization_or_different_channel_prefixes(tmp_path):
    assert FEATURES == ("Vx", "flow_speed", "proton_density", "T", "F", "Xs", "Xl", "P4", "P5", "P6")
    with pytest.raises(ValueError, match="strict"):
        CompatibleSourceStore(tmp_path, channel_policy="harmonized")
    with pytest.raises(KeyError):
        CompatibleSourceStore(tmp_path, progress=None).range("ProtonFlux_GT10", "2025-01-01", "2025-01-02")


def test_hapi_enforces_server_parameter_order_preserves_satellite_and_fill(tmp_path, monkeypatch):
    metadata = {"parameters": [
        {"name": "Time", "type": "isotime"},
        {"name": "P10", "type": "double", "fill": "-100000", "units": "pfu"},
        {"name": "P60", "type": "double", "fill": "null", "units": "pfu"},
        {"name": "satelliteProton", "type": "string"},
    ]}
    store = CompatibleSourceStore(tmp_path, progress=None)

    def download(url, params=None, **kwargs):
        if url.endswith("/info"):
            return json.dumps(metadata).encode()
        assert params["parameters"] == "P10,P60,satelliteProton"
        return b"2025-01-01T00:00:00Z,12,-100000,GOES-18\n"

    monkeypatch.setattr(store, "_download", download)
    frame = store._hapi("https://example.test/hapi", "goesp_part_flux_P5M", ["satelliteProton", "P60", "P10"], pd.Timestamp("2025-01-01T00:00Z"), pd.Timestamp("2025-01-01T00:05Z"))
    assert frame.P10.iloc[0] == 12
    assert pd.isna(frame.P60.iloc[0])
    assert frame.satelliteProton.iloc[0] == "GOES-18"


def test_hapi_splits_month_plus_trailing_boundary_at_31_day_limit(tmp_path, monkeypatch):
    store = CompatibleSourceStore(tmp_path, progress=None)
    requests_made = []
    metadata = {"parameters": [{"name": "Time", "type": "isotime"}, {"name": "F", "type": "double"}]}

    def download(url, params=None, **kwargs):
        if url.endswith("/info"):
            return json.dumps(metadata).encode()
        start = pd.Timestamp(params["time.min"])
        stop = pd.Timestamp(params["time.max"])
        assert stop - start <= pd.Timedelta(days=31)
        requests_made.append(params)
        return f"{start.isoformat()},1.0\n".encode()

    monkeypatch.setattr(store, "_download", download)
    frame = store._hapi("https://example.test/hapi", "test", ["F"], pd.Timestamp("2025-11-30T23:55Z"), pd.Timestamp("2026-01-01T00:00Z"))
    assert len(requests_made) == 2
    assert len(frame) == 2
    assert not frame.index.has_duplicates


def test_xrs_resamples_native_measurements_without_assumed_rescaling(tmp_path, monkeypatch):
    store = CompatibleSourceStore(tmp_path, progress=None)
    raw = pd.DataFrame({"Short_Wave": [1., 2., 3., 4., 5., 6.], "Long_Wave": [2., 4., 6., 8., 10., 12.], "satellite": ["GOES-18"] * 6}, index=pd.date_range("2025-01-01T00:00Z", periods=6, freq="min", name="Time"))
    monkeypatch.setattr(store, "_hapi", lambda *a, **k: raw)
    series = store.range("Xs", "2025-01-01T00:00Z", "2025-01-01T00:10Z")
    assert series.tolist() == [1., 4.]
    assert "no assumed rescaling" in series.attrs["calibration"].lower()
    assert series.attrs["bin_lookahead_minutes"] == 0


def test_native_five_minute_predictors_expose_conservative_lookahead(tmp_path):
    store = CompatibleSourceStore(tmp_path, progress=None)
    for feature in ("Vx", "flow_speed", "proton_density", "T", "F", "P4", "P5", "P6"):
        assert store.metadata(feature)["bin_lookahead_minutes"] == 5


def test_unexpected_download_failure_raises_and_never_caches(tmp_path, monkeypatch):
    calls = []

    def get(*args, **kwargs):
        calls.append(args)
        raise requests.Timeout("source did not respond")

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr("sep_dataset.compatible_sources.time.sleep", lambda *a: None)
    store = CompatibleSourceStore(tmp_path, retries=2, progress=None)
    with pytest.raises(SourceDataError, match="after 2 attempts"):
        store._download("https://example.test/data", kind="hapi_csv")
    assert len(calls) == 2
    assert not list(tmp_path.rglob("*.part"))
    assert not (tmp_path / "raw").exists()


def test_malformed_http200_error_never_enters_raw_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(
        content=b',\n"status":{"code":1411,"message":"Parameter out of order"}', raise_for_status=lambda: None))
    store = CompatibleSourceStore(tmp_path, retries=1, progress=None)
    with pytest.raises(SourceDataError, match="Invalid HAPI CSV"):
        store._download("https://example.test/data", kind="hapi_csv")
    assert not (tmp_path / "raw").exists()


def test_2020_transition_preserves_real_legacy_values_and_structural_missing(tmp_path, monkeypatch):
    store = CompatibleSourceStore(tmp_path, progress=None)
    # Real final monthly archive files contain only March 1–4 records.
    index = pd.date_range("2020-03-01T00:00Z", LEGACY_END, freq="5min", inclusive="left", name="Time")
    frame = pd.DataFrame({"P4": 1., "P5": 2., "P6": 3., "channel_status": "measured_legacy_channel", "source_product": "NOAA", "source_satellite": "GOES-15"}, index=index)
    monkeypatch.setattr(store, "_legacy", lambda *a, **k: frame)
    series = store.range("P4", "2020-03-04T23:55Z", "2020-03-05T00:05Z")
    assert series.iloc[0] == 1
    assert pd.isna(series.iloc[1])
    assert series.attrs["structural_missing"].tolist() == [False, True]


def test_structural_cache_survives_new_store_without_network(tmp_path, monkeypatch):
    original = CompatibleSourceStore(tmp_path, progress=None).range("P6", "2025-01-01", "2025-01-02")
    second = CompatibleSourceStore(tmp_path, progress=None)
    monkeypatch.setattr(second, "_download", lambda *a, **k: pytest.fail("Should use parsed cache"))
    reloaded = second.range("P6", "2025-01-01", "2025-01-02")
    assert reloaded.isna().all()
    assert reloaded.attrs["structural_missing"].all()
    assert second.source_records
    assert len(reloaded) == len(original)


def test_tampered_or_interrupted_parsed_cache_is_rebuilt_with_provenance(tmp_path, monkeypatch):
    first = CompatibleSourceStore(tmp_path, progress=None)
    first.range("P6", "2025-01-01", "2025-01-02")
    parsed = next((tmp_path / "parsed").glob("*.csv"))
    manifest = parsed.with_suffix(".json")
    parsed.write_text(parsed.read_text().replace("outside_original_channel_coverage", "measured", 1))
    second = CompatibleSourceStore(tmp_path, progress=None)
    monkeypatch.setattr(second, "_download", lambda *a, **k: pytest.fail("Structural absence needs no network"))
    recovered = second.range("P6", "2025-01-01", "2025-01-02")
    assert recovered.attrs["structural_missing"].all()
    assert json.loads(manifest.read_text())["sources"]
    manifest.write_text('{"sources":')
    third = CompatibleSourceStore(tmp_path, progress=None)
    assert third.range("P6", "2025-01-01", "2025-01-02").isna().all()
    assert json.loads(manifest.read_text())["parsed_sha256"]
    assert not list((tmp_path / "parsed").glob("*.part"))
