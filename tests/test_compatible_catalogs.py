import numpy as np
import pandas as pd
import pytest

from sep_dataset import compatible_catalogs as cc


def _record(onset="2025-01-02T00:00:00", cls="C1.0", **extra):
    return {
        "frm_name": "SWPC", "obs_observatory": "GOES",
        "event_starttime": onset, "event_peaktime": onset,
        "event_endtime": onset, "fl_goescls": cls,
        "kb_archivid": f"hek-{onset}", **extra,
    }


def _flare(onset="2025-01-02T00:00:00", cls="C1.0", **extra):
    frame = cc.parse_hek_flares([_record(onset, cls)])
    for key, value in extra.items():
        frame[key] = value
    return frame


def _flux(start="2024-12-30", stop="2025-01-06", value=0.5):
    index = pd.date_range(start, stop, freq="5min", inclusive="left", tz="UTC")
    return pd.DataFrame(value, index=index, columns=cc.INTEGRAL_COLUMNS)


def test_flare_cohort_is_swpc_goes_c_m_x_and_keeps_missing_position_missing():
    rows = [_record(cls=cls, onset=f"2025-01-02T0{i}:00:00") for i, cls in enumerate(["B9.9", "C1.0", "M1.0", "X1.0"])]
    rows += [_record("2025-01-02T05:00:00", "M2.0", frm_name="OTHER")]
    rows[1].update(noposition="true", event_coord1=0, event_coord2=0)
    result = cc.parse_hek_flares(rows)
    assert result.xrsclass.tolist() == ["C1.0", "M1.0", "X1.0"]
    assert pd.isna(result.iloc[0].goes_location)
    assert cc.goes_class_flux("C9.9") < cc.goes_class_flux("M1.0")
    assert result.iloc[0]["Peak Flux"] == 1e-6


def test_exact_onset_retains_conflicting_known_regions_for_identity_audit():
    rows = [
        _record(cls="C1.0", ar_noaanum=13944),
        _record(cls="C2.0", ar_noaanum=13944, kb_archivid="revision-2", revision="2"),
        _record(cls="C3.0", ar_noaanum=0, kb_archivid="missing-region", revision="3"),
        _record(cls="C4.0", ar_noaanum=13945, kb_archivid="different-region"),
    ]
    flares = cc.parse_hek_flares(rows)
    assert set(flares.noaa_active_region) == {13944, 13945}
    assert set(flares.xrsclass) == {"C2.0", "C4.0"}
    labels, audit = cc.label_flare_events(flares, _flux())
    assert labels.empty
    assert audit.status.eq("excluded_ambiguous_physical_flare_identity").all()


def test_observed_three_point_enhancement_produces_positive_and_actual_metrics():
    flux = _flux()
    flux.loc["2025-01-02T01:00:00Z":"2025-01-02T01:55:00Z", "GT10"] = 8.0
    labels, audit = cc.label_flare_events(_flare(), flux)
    assert labels.event_type.tolist() == [1]
    assert labels.iloc[0].sep_start == pd.Timestamp("2025-01-02T01:00:00Z")
    assert labels.iloc[0].sep_end == pd.Timestamp("2025-01-02T02:00:00Z")
    assert labels.iloc[0].sep_rise == 1.0
    assert labels.iloc[0].sep_duration == 1.0
    assert labels.iloc[0].sep_peak_2 == 8.0
    assert pd.isna(labels.iloc[0].sep_fluence_2)
    assert audit.iloc[0].GT10_fluence_pfu_seconds == 8.0 * 3600


def test_high_existing_background_is_not_mistaken_for_new_sep():
    labels, audit = cc.label_flare_events(_flare(), _flux(value=9.0))
    assert labels.event_type.tolist() == [0]
    assert audit.iloc[0].background_max_gt10_pfu == 9.0
    assert labels.iloc[0].sep_rise == -9999
    assert labels.iloc[0].sep_end - labels.iloc[0].sep_start == pd.Timedelta(hours=6)


def test_only_two_samples_above_threshold_remains_negative():
    flux = _flux()
    flux.loc["2025-01-02T01:00:00Z":"2025-01-02T01:05:00Z", "GT10"] = 8
    labels, _ = cc.label_flare_events(_flare(), flux)
    assert labels.event_type.tolist() == [0]


def test_missing_background_or_forecast_never_becomes_negative():
    for missing in ["2025-01-01T23:55:00Z", "2025-01-02T02:00:00Z"]:
        flux = _flux().drop(pd.Timestamp(missing))
        labels, audit = cc.label_flare_events(_flare(), flux)
        assert labels.empty
        assert audit.iloc[0].status == "excluded_missing_label_flux"
        assert pd.isna(audit.iloc[0].event_type)


def test_incomplete_forecast_excludes_positive_in_strict_mode():
    flux = _flux()
    flux.loc["2025-01-02T01:00:00Z":"2025-01-02T01:55:00Z", "GT10"] = 8
    flux = flux.drop(pd.Timestamp("2025-01-02T04:00:00Z"))
    assert cc.label_flare_events(_flare(), flux)[0].empty
    labels, audit = cc.label_flare_events(_flare(), flux, strictcoverage=False)
    assert labels.event_type.tolist() == [1]
    assert audit.iloc[0].forecast_valid_fraction < 1


def test_competing_flares_assign_one_enhancement_to_largest_xray_flare():
    flares = pd.concat([_flare(), _flare("2025-01-02T00:30:00", "M1.0")], ignore_index=True)
    flux = _flux()
    flux.loc["2025-01-02T01:00:00Z":"2025-01-02T01:55:00Z", "GT10"] = 8
    labels, audit = cc.label_flare_events(flares, flux)
    assert labels.event_type.tolist() == [0, 1]
    assert audit.status.tolist() == ["negative_competing_flare", "positive"]
    assert audit.competing_winner_id.nunique() == 1


def test_competing_halo_across_year_boundary_does_not_export_halo_rows():
    flares = pd.concat([
        _flare("2024-12-31T23:50:00", "M1.0", _in_requested_range=False),
        _flare("2025-01-01T00:00:00", "C1.0", _in_requested_range=True),
    ], ignore_index=True)
    flux = _flux()
    flux.loc["2025-01-01T01:00:00Z":"2025-01-01T01:55:00Z", "GT10"] = 8
    labels, audit = cc.label_flare_events(flares, flux)
    assert len(labels) == len(audit) == 1
    assert labels.event_type.tolist() == [0]
    assert labels.iloc[0].FlrOnset.year == 2025


def test_missing_other_energy_band_is_not_substituted_or_fabricated():
    flux = _flux()
    flux.loc["2025-01-02T01:00:00Z":"2025-01-02T01:55:00Z", "GT10"] = 8
    flux["GT60"] = np.nan
    labels, audit = cc.label_flare_events(_flare(), flux)
    assert labels.event_type.tolist() == [1]
    assert pd.isna(labels.iloc[0].sep_peak_4)
    assert audit.iloc[0].GT60_metric_valid_fraction == 0


def test_subcadence_flare_uses_only_points_inside_true_background_window():
    flux = _flux()
    # This point is outside the exact three-hour window for 00:03, and must
    # not falsely raise its background maximum.
    flux.loc["2025-01-01T21:00:00Z", "GT10"] = 100
    flux.loc["2025-01-02T01:00:00Z":"2025-01-02T01:55:00Z", "GT10"] = 8
    labels, audit = cc.label_flare_events(_flare("2025-01-02T00:03:00"), flux)
    assert labels.event_type.tolist() == [1]
    assert audit.iloc[0].background_max_gt10_pfu == 0.5


def test_fetch_includes_entire_last_day_and_boundary_halo(monkeypatch, tmp_path):
    calls = []

    def page(params, cache_path):
        calls.append(params)
        return {"result": [
            _record("2025-12-31T23:59:00", "C2.0"),
            _record("2026-01-01T01:00:00", "M1.0"),
        ], "overmax": False}

    monkeypatch.setattr(cc, "_hek_json", page)
    flares = cc.fetch_goes_flares("2025-12-31", "2025-12-31", tmp_path)
    assert flares["_in_requested_range"].tolist() == [True, False]
    assert len(calls) == 2  # December/January halo requests


def test_empty_or_truncated_flare_catalog_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "_hek_json", lambda *args: {"result": [], "overmax": False})
    with pytest.raises(RuntimeError, match="no usable"):
        cc.fetch_goes_flares("2025-01-01", "2025-01-02", tmp_path)
    monkeypatch.setattr(cc, "_hek_json", lambda *args: {"result": [_record()], "overmax": True})
    with pytest.raises(RuntimeError, match="repeated page"):
        cc.fetch_goes_flares("2025-01-01", "2025-01-02", tmp_path)


def test_provider_gets_background_and_future_padding():
    calls = []

    def provider(start, stop):
        calls.append((start, stop))
        return _flux()

    labels, _ = cc.label_flare_events(_flare(), flux_provider=provider)
    assert labels.event_type.tolist() == [0]
    assert calls == [(pd.Timestamp("2024-12-31T21:00:00Z"), pd.Timestamp("2025-03-03T06:00:00Z"))]


def test_collector_quarantines_physical_flare_alias_with_conflicting_provisional_labels():
    flares = pd.concat([
        _flare("2025-01-02T00:00:00", "M1.0", noaa_active_region=13944,
               Flrmaxtime=pd.Timestamp("2025-01-02T00:10:00Z")),
        _flare("2025-01-02T00:04:00", "C1.0", noaa_active_region=13944,
               Flrmaxtime=pd.Timestamp("2025-01-02T00:10:00Z")),
    ], ignore_index=True)
    flux = _flux()
    flux.loc["2025-01-02T01:00:00Z":"2025-01-02T01:55:00Z", "GT10"] = 8
    labels, audit = cc.label_flare_events(flares, flux)
    assert labels.empty
    assert set(audit.classification_event_type) == {0, 1}
    assert audit.event_type.isna().all()
    assert audit.status.eq("excluded_conflicting_physical_flare_labels").all()
    assert audit.canonical_flare_id.nunique() == 1
    assert audit.physical_flare_group_size.eq(2).all()
    assert set(audit.classification_status) == {"positive", "negative_competing_flare"}


def test_collector_keeps_one_same_label_catalog_duplicate_with_audit():
    one = _flare("2025-01-02T00:00:00", noaa_active_region=13944)
    labels, audit = cc.label_flare_events(pd.concat([one, one], ignore_index=True), _flux())
    assert len(labels) == 1
    assert labels.event_type.tolist() == [0]
    assert audit.status.tolist().count("excluded_duplicate_physical_flare") == 1
    assert audit.classification_event_type.eq(0).all()
    assert audit.event_type.notna().sum() == 1


def test_collector_halo_positive_cannot_reappear_as_negative_alias_inside_requested_dates():
    flares = pd.concat([
        _flare("2024-12-31T23:58:00", "M1.0", _in_requested_range=False,
               noaa_active_region=13944, Flrmaxtime=pd.Timestamp("2025-01-01T00:08:00Z")),
        _flare("2025-01-01T00:02:00", "C1.0", _in_requested_range=True,
               noaa_active_region=13944, Flrmaxtime=pd.Timestamp("2025-01-01T00:08:00Z")),
    ], ignore_index=True)
    flux = _flux()
    flux.loc["2025-01-01T01:00:00Z":"2025-01-01T01:55:00Z", "GT10"] = 8
    labels, audit = cc.label_flare_events(flares, flux)
    assert labels.empty
    assert len(audit) == 1
    assert audit.iloc[0].classification_event_type == 0
    assert audit.iloc[0].status == "excluded_conflicting_physical_flare_labels"
    assert pd.isna(audit.iloc[0].event_type)
