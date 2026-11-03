import hashlib
import json

import netCDF4
import numpy as np
import pandas as pd
import pytest
import requests

from sep_dataset.sgps_sources import (
    METHOD, SGPSStore, SGPSSourceError, TARGET_ENERGIES,
    interpolate_energy, read_l1b, read_l2,
)


ENERGY = np.array([1., 2., 3., 5., 8., 16., 31., 55., 90., 109., 130., 196., 330.])


def _l2(path, *, date="2025-12-01", unused_effective=False, legacy_lut=False):
    with netCDF4.Dataset(path, "w") as d:
        for name, length in [("time", 4), ("sensor_units", 2), ("diff_channels", 13)]:
            d.createDimension(name, length)
        t = d.createVariable("time", "f8", ("time",))
        t.units = f"seconds since {date} 00:00:00 UTC"
        t[:] = np.arange(4) * 300
        yaw = d.createVariable("yaw_flip_flag", "i4", ("time",))
        yaw[:] = [0, 0, 2, 1]
        dims = ("time", "sensor_units", "diff_channels")
        f = d.createVariable("AvgDiffProtonFlux", "f8", dims, fill_value=-1e31)
        f.units = "protons/(cm^2 sr keV s)"
        flux = np.broadcast_to(ENERGY ** -2 / 1000, (4, 2, 13)).copy()
        flux[:, 1] *= 10
        f[:] = flux
        error = d.createVariable("AvgDiffProtonFluxUncert", "f8", dims)
        error.units = f.units
        error[:] = flux * 0.1
        energy = d.createVariable("DiffProtonEffectiveEnergy", "f8", ("sensor_units", "diff_channels"), fill_value=-9999.)
        energy.units = "keV"
        energy[:] = -9999 if unused_effective else np.broadcast_to(ENERGY * 1000, (2, 13))
        for name, factor in [("DiffProtonLowerEnergy", 1 / np.sqrt(2)), ("DiffProtonUpperEnergy", np.sqrt(2))]:
            value = d.createVariable(name, "f8", ("sensor_units", "diff_channels"))
            value.units = "keV"
            value[:] = np.broadcast_to(ENERGY * 1000 * factor, (2, 13))
        counts = d.createVariable("DiffValidL1bSamplesInAvg", "i4", dims)
        counts[:] = 300
        for name in ["DiffDQFdtcSum", "DiffDQFoobSum", "DiffDQFerrSum", "DiffProtonIgnoredL1bDQFs"]:
            q = d.createVariable(name, "i4", dims)
            q[:] = 60 if name == "DiffDQFerrSum" else 0
        d["DiffProtonIgnoredL1bDQFs"][1, 0, 5] = 1
        lut = d.createVariable("ExpectedLUTNotFound", "i4")
        good = 0 if legacy_lut else 1
        lut.long_name = f"A boolean indicator: {good} if all input files' specified lookup tables matched the tables that were used, {1-good} otherwise."
        lut.assignValue(good)
    return path


def _l1b(path):
    labels = ["P1:1-1.9MeV", "P2A:1.9-2.3MeV", "P2B:2.3-3.4MeV", "P3:3.4-6.5MeV", "P4:6.5-12MeV", "P5:12-25MeV", "P6:25-40MeV", "P7:40-80MeV", "P8A:83-99MeV", "P8B:99-118MeV", "P8C:118-150MeV", "P9:150-275MeV", "P10:275-500MeV"]
    bounds = [(1, 1.9), (1.9, 2.3), (2.3, 3.4), (3.4, 6.5), (6.5, 12), (12, 25), (25, 40), (40, 80), (83, 99), (99, 118), (118, 150), (150, 275), (275, 500)]
    energy = np.sqrt(np.prod(bounds, axis=1))
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("record", 900)
        d.createDimension("sensor", 2)
        d.createDimension("label_length", 30)
        timestamp = d.createVariable("L1a_SciData_TimeStamp", "f8", ("record", "sensor"))
        timestamp.units = "seconds since 2020-03-01 00:00:00"
        timestamp[:] = np.broadcast_to(np.arange(900)[:, None], (900, 2))
        d.createVariable("yaw_flip_flag", "i4", ("record",))[:] = 0
        offset = 0
        for telescope, width in [("T1", 6), ("T2", 2), ("T3", 5)]:
            dim = f"energy_{telescope}"
            d.createDimension(dim, width)
            variable = d.createVariable(f"{telescope}_DifferentialProtonFluxes", "f8", ("record", "sensor", dim))
            variable.units = "cm-2 sr-1 s-1 keV-1"
            flux = np.broadcast_to(energy[offset:offset+width] ** -2 / 1000, (900, 2, width)).copy()
            variable[:] = flux
            error = d.createVariable(f"{telescope}_DifferentialProtonFluxUncertainties", "f8", ("record", "sensor", dim))
            error.units = variable.units
            error[:] = flux * 0.1
            flags = d.createVariable(f"{telescope}_DifferentialProtonFluxDQFs", "i4", ("record", "sensor", dim))
            flags[:] = 16  # Dynamic-error advisory is NOT an invalid bit.
            flags[751:, :, :] = 1  # Last bin only 151 valid samples.
            flag_labels = d.createVariable(f"energy_{telescope}_label", "S1", (dim, "label_length"))
            encoded = np.full((width, 30), b" ", dtype="S1")
            for i, label in enumerate(labels[offset:offset+width]):
                encoded[i, :len(label)] = np.array(list(label), dtype="S1")
            flag_labels[:] = encoded
            offset += width
        d["T1_DifferentialProtonFluxDQFs"][300:600, 0, 5] = 1
    return path


def test_log_log_interpolation_recovers_power_law_and_propagates_error():
    energies = np.array([10., 100.])
    flux = np.array([[100., 1.]])
    result = interpolate_energy(energies, flux, np.sqrt(1000), flux * 0.1)
    assert result["value"][0] == pytest.approx(10.)
    assert result["uncertainty"][0] == pytest.approx(10 * 0.1 / np.sqrt(2))
    assert result["weight"][0] == pytest.approx(0.5)
    assert result["lower_energy"][0] == 10
    assert result["upper_energy"][0] == 100


@pytest.mark.parametrize("flux,target", [([1., 0., 0.01], 40), ([1., np.nan, 0.01], 40), ([1., -1., 0.01], 40), ([1., 0.1, 0.01], 1), ([1., 0.1, 0.01], 1000)])
def test_no_energy_extrapolation_zero_replacement_or_skipping_bad_adjacent_channel(flux, target):
    result = interpolate_energy([10., 30., 100.], [flux], target)
    assert np.isnan(result["value"][0])


def test_l2_units_yaw_qc_and_advisory_uncertainty(tmp_path):
    frame, details = read_l2(_l2(tmp_path / "source.nc"), 18)
    for feature, target in TARGET_ENERGIES.items():
        assert frame[feature].iloc[0] == pytest.approx(target ** -2)
        assert frame[feature].iloc[2] == pytest.approx(10 * target ** -2)
        assert pd.isna(frame[feature].iloc[3])
        assert frame[f"{feature}_advisory_fraction"].iloc[0] == pytest.approx(0.2)
        assert frame[f"{feature}_uncertainty"].iloc[0] > 0
    assert pd.isna(frame.P4.iloc[1])  # invalid physical bracket must not be bridged
    assert frame.P5.iloc[1] > 0
    assert frame.source_detector.iloc[2] == "SGPS+X physical west"
    assert details["native_flux_units"] == "protons/(cm^2 sr keV s)"
    assert str(frame.index.tz) == "UTC"
    assert frame.P4_channel_status.iloc[0] == METHOD


def test_september2017_unused_effective_energy_and_opposite_lut_boolean(tmp_path):
    frame, details = read_l2(_l2(tmp_path / "2017.nc", date="2017-09-01", unused_effective=True, legacy_lut=True), 16)
    assert frame.P4.iloc[0] == pytest.approx(TARGET_ENERGIES["P4"] ** -2)
    assert details["energy_coordinate_basis"] == "geometric_mean_of_native_L2_band_bounds_effective_field_unused"


def test_l2_insufficient_sample_coverage_or_failed_lut_stays_missing(tmp_path):
    path = _l2(tmp_path / "source.nc")
    with netCDF4.Dataset(path, "a") as d:
        d["DiffValidL1bSamplesInAvg"][0, :, :] = 149
    frame, _ = read_l2(path, 18)
    assert frame[["P4", "P5", "P6"]].iloc[0].isna().all()
    with netCDF4.Dataset(path, "a") as d:
        d["ExpectedLUTNotFound"].assignValue(0)
    frame, _ = read_l2(path, 18)
    assert frame[["P4", "P5", "P6"]].isna().all().all()


def test_old_nonzero_aggregate_yaw_not_assumed_to_be_new_inverted_enum(tmp_path):
    path = _l2(tmp_path / "source.nc")
    with netCDF4.Dataset(path, "a") as d:
        d.renameVariable("yaw_flip_flag", "YawFlipFlag")
        d["YawFlipFlag"][:] = [0, 300, 2, 1]
    frame, _ = read_l2(path, 16)
    assert frame.P5.iloc[0] > 0
    assert frame.P5.iloc[1:].isna().all()


def test_l1b_real_bridge_averages_valid_native_flux_with_flagged_nominal_energies(tmp_path):
    path = _l1b(tmp_path / "source.nc")
    frame, details = read_l1b(path, 16)
    assert len(frame) == 3
    for feature, target in TARGET_ENERGIES.items():
        assert frame[feature].iloc[0] == pytest.approx(target ** -2)
        assert frame[f"{feature}_advisory_fraction"].iloc[0] == 1
    assert pd.isna(frame.P4.iloc[1])
    assert frame.P5.iloc[1] > 0
    assert "NO L2 temperature correction" in details["processing"]
    assert "nominal" in frame.energy_coordinate_basis.iloc[0]
    assert frame.native_valid_samples_6.iloc[2] == 151
    strict, _ = read_l1b(path, 16, min_valid_fraction=0.9)
    assert strict[["P4", "P5", "P6"]].iloc[2].isna().all()


def _response(code, body, url):
    response = requests.Response()
    response.status_code = code
    response._content = body if isinstance(body, bytes) else body.encode()
    response.url = url
    return response


def test_archive_fallback_highest_version_cache_provenance_and_bounded_ram(tmp_path, monkeypatch):
    body = _l2(tmp_path / "fixture.nc").read_bytes()
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        if "/goes16/" in url:
            return _response(404, "Not found", url)
        if url.endswith("/"):
            return _response(200, '<title>Index of /NOAA</title><a href="sci_sgps-l2-avg5m_g18_d20251201_v1-0-1.nc">old</a><a href="sci_sgps-l2-avg5m_g18_d20251201_v3-0-2.nc">current</a>', url)
        assert url.endswith("_v3-0-2.nc")
        return _response(200, body, url)

    monkeypatch.setattr(requests, "get", get)
    store = SGPSStore(tmp_path / "cache", progress=None, satellites=(16, 18), memory_days=1)
    first = store.range("P4", "2025-12-01", "2025-12-02")
    assert len(first) == 288
    assert first.notna().sum() == 2  # remaining rows are measured-invalid/missing, not filled
    q = store.quality_range("P4", "2025-12-01", "2025-12-02")
    assert q.source_satellite.iloc[0] == "GOES-18"
    assert q.channel_status.iloc[0] == METHOD
    assert q.measurement_missing.iloc[1]
    assert q.target_energy_MeV.iloc[0] == 30.6
    assert store.source_records[0]["sha256"] == hashlib.sha256(body).hexdigest()
    json.dumps(store.source_records, allow_nan=False)
    original_native = store.native_range("2025-12-01", "2025-12-02")
    assert "native_flux_6" in original_native
    old_calls = len(calls)
    reopened = SGPSStore(tmp_path / "cache", progress=None, satellites=(16, 18), memory_days=1)
    pd.testing.assert_series_equal(reopened.range("P4", "2025-12-01", "2025-12-02"), first, check_exact=True)
    pd.testing.assert_frame_equal(reopened.native_range("2025-12-01", "2025-12-02"), original_native, check_exact=True)
    assert len(calls) == old_calls
    assert reopened.source_records
    parsed = next((tmp_path / "cache" / "parsed").rglob("*.csv"))
    parsed.write_text(parsed.read_text().replace("source_sgps_energy_interpolated", "tampered", 1))
    fresh = SGPSStore(tmp_path / "cache", progress=None, satellites=(16, 18), memory_days=1)
    assert fresh.quality_range("P4", "2025-12-01", "2025-12-02").channel_status.iloc[0] == METHOD
    assert len(calls) == old_calls


def test_missing_archive_directory_not_blindly_retried_and_never_invented(tmp_path, monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return _response(404, "Not found", url)

    monkeypatch.setattr(requests, "get", get)
    store = SGPSStore(tmp_path, progress=None, satellites=(16,), memory_days=1)
    result = store.range("P6", "2018-01-01", "2018-01-04")
    assert len(calls) == 2  # one monthly L2 and one monthly L1b index, not daily retries
    assert result.isna().all()
    assert len(store._day_memory) == 1
    q = store.quality_range("P6", "2018-01-01", "2018-01-02")
    assert q.channel_status.eq("sgps_no_source_coverage").all()
    assert q.measurement_missing.all()
    assert len(calls) == 2


@pytest.mark.parametrize("code", [401, 403, 500])
def test_auth_server_or_network_failure_never_becomes_empty_coverage(tmp_path, monkeypatch, code):
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: _response(code, "failure", url))
    store = SGPSStore(tmp_path, progress=None, retries=1)
    with pytest.raises(SGPSSourceError, match="not converted"):
        store.range("P4", "2025-12-01", "2025-12-02")
    assert not list(tmp_path.rglob("*.csv"))


def test_invalid_netcdf_not_published_as_successful_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, **kwargs: _response(200, "<html>upstream failure</html>", url))
    store = SGPSStore(tmp_path, progress=None)
    with pytest.raises(SGPSSourceError, match="non-netCDF"):
        store._download_netcdf("https://noaa.example.test/data.nc")
    assert not list(tmp_path.rglob("*.nc"))


def test_metadata_exposes_energy_matching_uncertainty_and_causal_availability(tmp_path):
    store = SGPSStore(tmp_path, progress=None)
    for feature, target in TARGET_ENERGIES.items():
        info = store.metadata(feature)
        assert info["target_energy_MeV"] == target
        assert info["method"] == METHOD
        assert info["bin_lookahead_minutes"] == 5
        assert "not proven" in info["calibration_caveat"]
        assert "not quantified" in info["uncertainty"]
    with pytest.raises(KeyError):
        store.metadata("SGPS_P4")
