import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sep_dataset import magnetic_sources as ms


def record(time="2025-01-01T23:00:00Z", *, ar=13950, members="13950", patch="10", quality="0x00000000", flux=1e22, rval=3.5, spec=ms.SHARP):
    return {"T_REC": time, "T_OBS": time, spec.patch_key: patch, "NOAA_AR": ar,
            "NOAA_ARS": members, "QUALITY": quality, "USFLUXL": flux, "R_VALUE": rval}


def store_with_rows(tmp_path, monkeypatch, rows, **kwargs):
    store = ms.MagneticStore(tmp_path, progress=None, **kwargs)
    calls = []

    def fetch(spec, start, stop):
        calls.append((spec.series, start, stop))
        return pd.DataFrame(rows)

    monkeypatch.setattr(store, "_fetch", fetch)
    return store, calls


def test_tai_to_utc_uses_historical_leap_second_offsets():
    values = ms.parse_jsoc_times(["1998.01.01_00:00:31_TAI", "2025.01.01_00:00:37_TAI", "2025-01-01T00:00:00Z", "2025-01-01 00:00:00"])
    assert values[0] == pd.Timestamp("1998-01-01T00:00:00Z")
    assert values[1] == values[2] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert pd.isna(values[3])


def test_native_quality_thresholds_and_zero_r_value_are_preserved():
    rows = [record("1998-01-01T00:00:00Z", ar=8124, members="8124", quality="0x00000200", rval=0, spec=ms.SMARP),
            record("1998-01-01T01:36:00Z", ar=8124, members="8124", quality="0x00040000", spec=ms.SMARP)]
    data = ms.normalize_native(pd.DataFrame(rows), ms.SMARP)
    assert data.quality_ok.tolist() == [True, False]
    assert data.R_VALUE.iloc[0] == 0
    assert data.native_quality.iloc[0] == 512


def test_later_t_obs_and_integration_guard_control_availability():
    row = record("1998.01.01_00:00:00_TAI", ar=8124, members="8124", spec=ms.SMARP)
    row["T_OBS"] = "1998.01.01_00:03:30_TAI"
    data = ms.normalize_native(pd.DataFrame([row]), ms.SMARP)
    assert data.available_at.iloc[0] == pd.Timestamp("1998-01-01T00:07:59Z")


def test_backward_holds_never_interpolate_toward_future_records(tmp_path, monkeypatch):
    rows = [record(flux=1e22), record("2025-01-01T23:12:00Z", flux=9e22)]
    store, calls = store_with_rows(tmp_path, monkeypatch, rows)
    values, q = store.window("USFLUXL", "2025-01-02T00:00Z", 13950)
    assert len(values) == len(q) == 288
    assert values.index[0] == pd.Timestamp("2025-01-01T00:00Z")
    assert values.index[-1] == pd.Timestamp("2025-01-01T23:55Z")
    assert pd.isna(values.loc["2025-01-01T23:10Z"])
    assert values.loc["2025-01-01T23:15Z"] == 1e22
    assert values.loc["2025-01-01T23:20Z"] == 1e22
    assert values.loc["2025-01-01T23:25Z"] == 9e22
    assert q.loc["2025-01-01T23:15Z", "point_status"] == "O"
    assert q.loc["2025-01-01T23:20Z", "point_status"] == "H"
    assert pd.isna(values.loc["2025-01-01T23:40Z"])
    valid = values.notna()
    assert (q.loc[valid, "available_at"] <= values.index[valid]).all()
    assert (q.loc[valid, "native_observation_time"] <= values.index[valid]).all()
    other, _ = store.window("R_VALUE", "2025-01-02T00:00Z", 13950)
    assert other.loc["2025-01-01T23:15Z"] == 3.5
    assert len(calls) == 2  # December/January once, shared by both features


@pytest.mark.parametrize("requested", [13951, 1395])
def test_no_assignment_from_unrelated_or_substring_region(tmp_path, monkeypatch, requested):
    store, _ = store_with_rows(tmp_path, monkeypatch, [record()])
    values, q = store.window("USFLUXL", "2025-01-02T00:00Z", requested)
    assert values.isna().all()
    assert q.point_status.eq("G").all()


def test_missing_region_never_triggers_unrelated_solar_disk_download(tmp_path, monkeypatch):
    store, calls = store_with_rows(tmp_path, monkeypatch, [record()])
    values, q = store.window("R_VALUE", "2025-01-02T00:00Z", np.nan)
    assert values.isna().all() and q.point_status.eq("M").all()
    assert not calls


def test_multiple_region_native_patch_is_not_presented_as_isolated_region(tmp_path, monkeypatch):
    rows = [record(members="13950,13951")]
    store, _ = store_with_rows(tmp_path / "strict", monkeypatch, rows)
    values, q = store.window("R_VALUE", "2025-01-02T00:00Z", 13950)
    assert pd.isna(values.loc["2025-01-01T23:15Z"])
    assert q.loc["2025-01-01T23:15Z", "point_status"] == "A"
    permissive, _ = store_with_rows(tmp_path / "flagged", monkeypatch, rows, allow_multi_region_patch=True)
    values, q = permissive.window("R_VALUE", "2025-01-02T00:00Z", 13950)
    assert values.loc["2025-01-01T23:15Z"] == 3.5
    assert q.loc["2025-01-01T23:15Z", "region_count"] == 2


def test_conflicting_patches_for_one_region_are_ambiguous(tmp_path, monkeypatch):
    store, _ = store_with_rows(tmp_path, monkeypatch, [record(patch="10"), record(patch="11")])
    values, q = store.window("USFLUXL", "2025-01-02T00:00Z", 13950)
    assert pd.isna(values.loc["2025-01-01T23:15Z"])
    assert q.loc["2025-01-01T23:15Z", "point_status"] == "A"


def test_rejected_latest_quality_does_not_silently_fall_back_to_older_good_row(tmp_path, monkeypatch):
    store, _ = store_with_rows(tmp_path, monkeypatch, [record(), record("2025-01-01T23:12:00Z", quality="0x00010000")])
    values, q = store.window("R_VALUE", "2025-01-02T00:00Z", 13950)
    assert values.loc["2025-01-01T23:15Z"] == 3.5
    assert pd.isna(values.loc["2025-01-01T23:25Z"])
    assert q.loc["2025-01-01T23:25Z", "point_status"] == "Q"


def test_zero_r_value_survives_window_selection(tmp_path, monkeypatch):
    store, _ = store_with_rows(tmp_path, monkeypatch, [record(rval=0)])
    values, _ = store.window("R_VALUE", "2025-01-02T00:00Z", 13950)
    assert values.loc["2025-01-01T23:15Z"] == 0
    assert "log10" in store.metadata("R_VALUE")["units"]


def test_count_mismatch_is_an_error_not_a_partial_month():
    row = record()
    payload = {"status": 0, "count": 2, "keywords": [{"name": k, "values": [v]} for k, v in row.items()]}
    with pytest.raises(ms.MagneticSourceError, match="count disagrees"):
        ms.parse_keyword_payload(payload, ms.SHARP)
    assert ms.parse_keyword_payload({"status": 0, "count": 0, "keywords": []}, ms.SHARP).empty


def test_offline_native_csv_uses_same_quality_and_causal_selection(tmp_path):
    path = tmp_path / "native.csv"
    rows = pd.DataFrame([record(rval=0)])
    rows["series"] = ms.SHARP.series
    rows.to_csv(path, index=False)
    store = ms.MagneticStore(tmp_path / "cache", native_csv=path, progress=None)
    values, q = store.window("R_VALUE", "2025-01-02T00:00Z", 13950)
    assert values.loc["2025-01-01T23:15Z"] == 0
    assert q.loc["2025-01-01T23:15Z", "native_cadence_minutes"] == 12
    assert any(r.get("source") == "offline_native_keyword_csv" for r in store.source_records)


def test_successful_request_cache_is_verified_and_reused(tmp_path, monkeypatch):
    body = json.dumps({"status": 0, "count": 0, "keywords": []}).encode()
    calls = []
    monkeypatch.setattr(ms.requests, "get", lambda *args, **kwargs: calls.append(kwargs) or SimpleNamespace(content=body, raise_for_status=lambda: None))
    store = ms.MagneticStore(tmp_path, progress=None)
    params = {"op": "rs_list", "ds": "test"}
    assert store._request(params)["count"] == 0
    assert store._request(params)["count"] == 0
    assert len(calls) == 1
    next((tmp_path / "raw").glob("*.json")).write_bytes(b"corrupt")
    assert store._request(params)["count"] == 0
    assert len(calls) == 2


def test_network_failure_raises_instead_of_creating_filled_measurements(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ms.requests.ConnectionError("unreachable")
    monkeypatch.setattr(ms.requests, "get", fail)
    store = ms.MagneticStore(tmp_path, progress=None, retries=1)
    with pytest.raises(ms.MagneticSourceError, match="no measurements have been fabricated"):
        store._request({"op": "series_struct", "ds": ms.SHARP.series})


def test_large_jsoc_queries_are_bounded_and_split_without_losing_native_records(tmp_path, monkeypatch):
    store = ms.MagneticStore(tmp_path, progress=None)
    requested = []

    def request(params):
        if params["op"] == "series_struct":
            return {"status": 0, "keywords": [{"name": name} for name in record()]}
        interval = params["ds"].split("[][")[1].rstrip("]")
        low, high = interval.split("-")
        start, stop = ms.parse_jsoc_times([low, high])
        requested.append((start, stop))
        if stop-start > pd.Timedelta(days=4):
            raise ms.MagneticQueryError(6, "query too large")
        row = record(low)
        return {"status": 0, "count": 1, "keywords": [{"name": k, "values": [v]} for k, v in row.items()]}

    monkeypatch.setattr(store, "_request", request)
    frame = store._fetch(ms.SHARP, pd.Timestamp("2025-01-01T00:00Z"), pd.Timestamp("2025-01-15T00:00Z"))
    assert len(frame) == 4
    assert max(stop-start for start, stop in requested) <= pd.Timedelta(days=7)
    assert len(requested) == 6  # two failed weeks, four successful half-weeks


def test_failed_daily_jsoc_query_is_not_relabelled_as_empty_month(tmp_path, monkeypatch):
    store = ms.MagneticStore(tmp_path, progress=None)

    def request(params):
        if params["op"] == "series_struct":
            return {"status": 0, "keywords": [{"name": name} for name in record()]}
        raise ms.MagneticQueryError(6, "query failed")

    monkeypatch.setattr(store, "_request", request)
    with pytest.raises(ms.MagneticQueryError):
        store._fetch(ms.SHARP, pd.Timestamp("2025-01-01T00:00Z"), pd.Timestamp("2025-01-02T00:00Z"))


def test_historical_four_digit_region_expands_only_with_native_confirmation(tmp_path, monkeypatch):
    rows = [record('2013-01-01T23:00:00Z',ar=11775,members='11775')]
    store,_=store_with_rows(tmp_path,monkeypatch,rows)
    s,q=store.window('R_VALUE','2013-01-02T00:00:00Z',1775)
    assert s.notna().any()
    assert q.loc[s.notna(),'noaa_identifier_expanded'].all()
    assert q.loc[s.notna(),'association_noaa_ar'].eq(11775).all()
    missing,_=store.window('R_VALUE','2013-01-02T00:00:00Z',1776)
    assert missing.isna().all()


@pytest.mark.parametrize("empty_fields", [{}, {"keywords": None}, {"keywords": []}], ids=["omitted", "null", "empty-array"])
def test_confirmed_zero_record_response_retains_empty_native_schema(empty_fields):
    payload = {"status": 0, "count": 0, "runtime": 0.02, **empty_fields}
    frame = ms.parse_keyword_payload(payload, ms.SMARP)
    assert frame.empty
    assert set(frame.columns) == set(record(spec=ms.SMARP))
    normalized = ms.normalize_native(frame, ms.SMARP)
    assert normalized.empty
    assert str(normalized.record_time.dtype) == "datetime64[ns, UTC]"


@pytest.mark.parametrize("fields", [
    {"count": 0, "keywords": {}},
    {"count": 0, "keywords": ""},
    {"count": 1},
    {"count": False},
    {"count": "0"},
    {"count": 0.0},
    {},
], ids=["malformed-object", "malformed-string", "missing-positive-arrays", "boolean-count", "string-count", "float-count", "missing-count"])
def test_ambiguous_or_malformed_response_is_not_relabelled_as_empty(fields):
    with pytest.raises(ms.MagneticSourceError):
        ms.parse_keyword_payload({"status": 0, **fields}, ms.SMARP)


def test_empty_week_between_populated_weeks_preserves_only_real_native_records(tmp_path, monkeypatch):
    store = ms.MagneticStore(tmp_path, progress=None)
    requested = []

    def request(params):
        if params["op"] == "series_struct":
            return {"status": 0, "keywords": [{"name": name} for name in record(spec=ms.SMARP)]}
        low, high = params["ds"].split("[][")[1].rstrip("]").split("-")
        requested.append((low, high))
        if len(requested) == 2:
            return {"status": 0, "count": 0, "runtime": 0.02}
        row = record(low, ar=8124, members="8124", spec=ms.SMARP)
        return {"status": 0, "count": 1, "keywords": [{"name": name, "values": [value]} for name, value in row.items()]}

    monkeypatch.setattr(store, "_request", request)
    frame = store._fetch(ms.SMARP, pd.Timestamp("1998-01-01T00:00Z"), pd.Timestamp("1998-01-22T00:00Z"))
    assert len(requested) == 3
    assert frame.T_REC.tolist() == [requested[0][0], requested[2][0]]
    assert frame.USFLUXL.eq(1e22).all()
    assert frame.R_VALUE.eq(3.5).all()


def test_empty_month_marks_both_features_missing_and_reuses_raw_cache(tmp_path, monkeypatch):
    requests = []

    def get(url, *, params, timeout):
        requests.append(params.copy())
        payload = ({"status": 0, "keywords": [{"name": name} for name in record(spec=ms.SMARP)]}
                   if params["op"] == "series_struct" else {"status": 0, "count": 0, "runtime": 0.02})
        return SimpleNamespace(content=json.dumps(payload).encode(), raise_for_status=lambda: None)

    monkeypatch.setattr(ms.requests, "get", get)
    store = ms.MagneticStore(tmp_path, progress=None)
    for feature in ms.FEATURES:
        values, quality = store.window(feature, "1998-07-15T00:00Z", 8124)
        assert len(values) == len(quality) == 288
        assert values.isna().all()
        assert quality.point_status.eq("G").all()
        assert quality.reason.eq("no_matching_native_record").all()
    records = [item for item in store.source_records if "native_records" in item]
    assert len(records) == 1 and records[0]["native_records"] == 0
    first_request_count = len(requests)
    assert first_request_count > 1
    resumed = ms.MagneticStore(tmp_path, progress=None)
    values, quality = resumed.window("USFLUXL", "1998-07-15T00:00Z", 8124)
    assert values.isna().all() and quality.point_status.eq("G").all()
    assert len(requests) == first_request_count
