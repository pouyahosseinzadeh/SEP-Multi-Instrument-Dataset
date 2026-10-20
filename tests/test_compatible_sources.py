import json
import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests

from sep_dataset.compatible_sources import (
    FEATURES, INTEGRAL_COLUMNS, ISWA, LEGACY_END, CompatibleSourceStore,
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


NO_DATA_PAYLOAD = {"HAPI": "2.0", "status": {"code": 1406, "message": "No data available for the specified ID and time range"}, "data": []}
XRS_METADATA = {"parameters": [
    {"name": "Time", "type": "isotime"},
    {"name": "Short_Wave", "type": "double", "fill": "0", "units": "W/m^2"},
    {"name": "Long_Wave", "type": "double", "fill": "0", "units": "W/m^2"},
    {"name": "satellite", "type": "string"},
]}


def _http_response(status, body, url="https://example.test/hapi/data"):
    response = requests.Response()
    response.status_code = status
    response._content = body if isinstance(body, bytes) else json.dumps(body).encode()
    response.url = url
    return response


def test_actual_iswa_404_no_data_returns_typed_empty_and_caches_provenance(tmp_path, monkeypatch):
    requests_made = []

    def get(url, params=None, **kwargs):
        requests_made.append((url, params))
        if url.endswith("/info"):
            return _http_response(200, XRS_METADATA, url)
        return _http_response(404, NO_DATA_PAYLOAD, url)

    monkeypatch.setattr(requests, "get", get)
    store = CompatibleSourceStore(tmp_path, retries=1, progress=None)
    start, stop = pd.Timestamp("2014-12-31T23:55Z"), pd.Timestamp("2015-01-01T00:00Z")
    wanted = ["Short_Wave", "Long_Wave", "satellite"]
    frame = store._hapi(ISWA, "goesp_xray_flux_P1M", wanted, start, stop)
    assert frame.empty
    assert frame.columns.tolist() == wanted
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert str(frame.index.tz) == "UTC"
    assert pd.api.types.is_numeric_dtype(frame.Short_Wave)
    assert pd.api.types.is_numeric_dtype(frame.Long_Wave)
    assert not pd.api.types.is_numeric_dtype(frame.satellite)
    body = json.dumps(NO_DATA_PAYLOAD).encode()
    assert any(p.read_bytes() == body for p in (tmp_path / "raw").iterdir())
    records = [r for r in store.source_records if r.get("dataset") == "goesp_xray_flux_P1M"]
    assert records
    assert records[-1]["query"]["time.min"] == "2014-12-31T23:55:00Z"
    assert records[-1]["sha256"] == hashlib.sha256(body).hexdigest()
    before = len(requests_made)
    assert store._hapi(ISWA, "goesp_xray_flux_P1M", wanted, start, stop).empty
    assert len(requests_made) == before


def test_31_day_xrs_chunk_plus_actual_empty_remainder_keeps_valid_measurements(tmp_path, monkeypatch):
    requested_chunks = []

    def get(url, params=None, **kwargs):
        if url.endswith("/info"):
            return _http_response(200, XRS_METADATA, url)
        requested_chunks.append(params)
        if params["time.min"] == "2014-12-31T23:55:00Z":
            return _http_response(404, NO_DATA_PAYLOAD, url)
        return _http_response(200, b"2014-12-01T00:00:00Z,1e-8,2e-7,GOES-15\n2014-12-31T23:54:00Z,3e-8,4e-7,GOES-15\n", url)

    monkeypatch.setattr(requests, "get", get)
    store = CompatibleSourceStore(tmp_path, retries=1, progress=None)
    frame = store._hapi(ISWA, "goesp_xray_flux_P1M", ["Short_Wave", "Long_Wave", "satellite"], pd.Timestamp("2014-11-30T23:55Z"), pd.Timestamp("2015-01-01T00:00Z"))
    assert len(requested_chunks) == 2
    assert frame.index.tolist() == [pd.Timestamp("2014-12-01T00:00Z"), pd.Timestamp("2014-12-31T23:54Z")]
    np.testing.assert_array_equal(frame.Short_Wave, [1e-8, 3e-8])
    assert not frame.index.has_duplicates
    assert frame.satellite.tolist() == ["GOES-15", "GOES-15"]


def test_entire_empty_xrs_month_becomes_audited_nan_grid_not_failed_build(tmp_path, monkeypatch):
    def get(url, params=None, **kwargs):
        return _http_response(200, XRS_METADATA, url) if url.endswith("/info") else _http_response(404, NO_DATA_PAYLOAD, url)

    monkeypatch.setattr(requests, "get", get)
    store = CompatibleSourceStore(tmp_path, retries=1, progress=None)
    frame = store._load_month("xrs", pd.Timestamp("2014-12-01T00:00Z"))
    assert len(frame) == 31 * 288
    assert frame[["Xs", "Xl"]].isna().all().all()
    assert str(frame.index.tz) == "UTC"
    quality = store.quality_range("Xs", "2014-12-31T23:55Z", "2015-01-01T00:00Z")
    assert quality.measurement_missing.all()
    assert not quality.channel_status.eq("outside_original_channel_coverage").any()
    assert store.source_records


@pytest.mark.parametrize("http_status,payload", [
    (404, {"HAPI": "2.0", "status": {"code": 1406, "message": "Unknown dataset ID"}, "data": []}),
    (401, NO_DATA_PAYLOAD),
    (403, NO_DATA_PAYLOAD),
    (500, {"HAPI": "2.0", "status": {"code": 1201, "message": "OK - no data for time range"}, "data": []}),
    (404, {"HAPI": "2.0", "status": {"code": 1406, "message": "No data available for the specified ID and time range"}}),
    (404, {"HAPI": "2.0", "status": {"code": 1406, "message": "No data available for the specified ID and time range"}, "data": [123]}),
])
def test_unknown_id_auth_and_malformed_no_data_still_fail_without_cache(tmp_path, monkeypatch, http_status, payload):
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: _http_response(http_status, payload, url))
    store = CompatibleSourceStore(tmp_path, retries=1, progress=None)
    with pytest.raises(SourceDataError):
        store._download(ISWA + "/data", {"id": "unknown"}, kind="hapi_csv")
    assert not (tmp_path / "raw").exists()


def test_iswa_nonstandard_404_exception_is_not_applied_to_other_servers(tmp_path, monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: _http_response(404, NO_DATA_PAYLOAD, url))
    store = CompatibleSourceStore(tmp_path, retries=1, progress=None)
    with pytest.raises(SourceDataError):
        store._download("https://other-hapi.example.test/data", kind="hapi_csv")
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize("body", [
    b"",
    json.dumps({"HAPI": "2.0", "status": {"code": 1200, "message": "OK"}, "data": []}).encode(),
    json.dumps({"HAPI": "2.0", "status": {"code": 1201, "message": "OK - no data for time range"}, "data": []}).encode(),
])
def test_standard_hapi_empty_intervals_are_supported_on_other_servers(tmp_path, monkeypatch, body):
    def get(url, **kwargs):
        return _http_response(200, XRS_METADATA if url.endswith("/info") else body, url)

    monkeypatch.setattr(requests, "get", get)
    store = CompatibleSourceStore(tmp_path, retries=1, progress=None)
    frame = store._hapi("https://example.test/hapi", "fixture", ["Short_Wave", "Long_Wave", "satellite"], pd.Timestamp("2014-12-31T23:55Z"), pd.Timestamp("2015-01-01T00:00Z"))
    assert frame.empty
    assert str(frame.index.tz) == "UTC"
    assert store.source_records[-1]["source_status"] == "no_data_available"


def test_integral_no_data_never_creates_false_nsep_labels(tmp_path, monkeypatch):
    from sep_dataset.compatible_catalogs import label_flare_events, parse_hek_flares

    metadata = {"parameters": [{"name": "Time", "type": "isotime"}] + [
        {"name": f"P{energy}", "type": "double", "fill": "-100000", "units": "pfu"}
        for energy in (5, 10, 30, 60, 100)
    ] + [{"name": "satelliteProton", "type": "string"}]}

    def get(url, params=None, **kwargs):
        return _http_response(200, metadata, url) if url.endswith("/info") else _http_response(404, NO_DATA_PAYLOAD, url)

    monkeypatch.setattr(requests, "get", get)
    store = CompatibleSourceStore(tmp_path, retries=1, progress=None)
    flux = store.integral_range("2025-01-01", "2025-01-04")
    assert flux[list(INTEGRAL_COLUMNS)].isna().all().all()
    flares = parse_hek_flares([{
        "frm_name": "SWPC", "obs_observatory": "GOES", "event_starttime": "2025-01-02T00:00:00",
        "event_peaktime": "2025-01-02T00:00:00", "event_endtime": "2025-01-02T00:00:00",
        "fl_goescls": "C1.0", "kb_archivid": "fixture-unknown-label",
    }])
    labels, audit = label_flare_events(flares, flux)
    assert labels.empty
    assert audit.event_type.isna().all()
    assert audit.status.tolist() == ["excluded_missing_label_flux"]
