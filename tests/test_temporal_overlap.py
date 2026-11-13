import numpy as np
import pandas as pd
import pytest

from sep_dataset.compatible_builder import window_index
from sep_dataset.temporal_overlap import assert_no_temporal_overlap, audit_temporal_overlap


def test_grid_overlap_touching_boundary_and_mixed_utc_onsets():
    metadata = pd.DataFrame({
        "FlrOnset": [
            "2025-05-03 00:00:00Z",
            "2025-05-01T18:01:00-06:00",
            "2025-05-03 00:05:00",
            "2025-05-01 00:05:00",
            "2025-05-01 00:10:00",
            "2025-05-01 00:00:00",
        ],
        "event_type": [0, 1, 0, 0, 0, 0],
    }, index=[100, 104, 102, 200, 108, 108])
    mask, audit = audit_temporal_overlap(metadata)
    assert mask.tolist() == [True, False, False, False, True, False]
    assert audit.source_row_index.tolist() == [0, 4]
    assert audit.witness_sep_source_row_index.tolist() == [1, 1]
    assert audit.overlap_bins.tolist() == [1, 1]
    for row in audit.itertuples():
        nsep_grid = window_index(metadata.FlrOnset.iloc[row.source_row_index])
        sep_grid = window_index(metadata.FlrOnset.iloc[row.witness_sep_source_row_index])
        assert row.overlap_bins == len(nsep_grid.intersection(sep_grid))
        assert row.nsep_window_start_utc == nsep_grid[0]
        assert row.witness_sep_window_start_utc == sep_grid[0]
    assert_no_temporal_overlap(metadata.loc[~mask])
    with pytest.raises(ValueError, match="2 NSEP predictor windows"):
        assert_no_temporal_overlap(metadata)


def test_audit_matches_brute_force_and_uses_deterministic_witness():
    rng = np.random.default_rng(17)
    onset_minutes = rng.integers(0, 9000, size=60)
    onset_minutes[[14, 22]] = 1700
    labels = rng.integers(0, 2, size=60)
    labels[[14, 22]] = 1
    metadata = pd.DataFrame({
        "FlrOnset": pd.Timestamp("2017-09-01", tz="UTC") + pd.to_timedelta(onset_minutes, unit="min"),
        "event_type": labels,
    })
    mask, audit = audit_temporal_overlap(metadata)
    grids = [window_index(onset) for onset in metadata.FlrOnset]
    expected = []
    for i in np.flatnonzero(labels == 0):
        witnesses = [(grids[j][-1], j, len(grids[i].intersection(grids[j]))) for j in np.flatnonzero(labels == 1)]
        matches = sorted(row for row in witnesses if row[2])
        if matches:
            _, j, bins = matches[0]
            expected.append((i, j, bins))
    assert np.flatnonzero(mask).tolist() == [row[0] for row in expected]
    assert list(audit[["source_row_index", "witness_sep_source_row_index", "overlap_bins"]].itertuples(index=False, name=None)) == expected
    assert not mask[labels == 1].any()


@pytest.mark.parametrize("label", [0, 1])
def test_one_class_has_empty_audit(label):
    metadata = pd.DataFrame({"FlrOnset": ["2009-02-01"], "event_type": [label]})
    mask, audit = audit_temporal_overlap(metadata)
    assert mask.tolist() == [False]
    assert audit.empty
    assert "overlap_bins" in audit


def test_overlap_audit_rejects_unknown_labels_and_onsets():
    with pytest.raises(ValueError, match="binary"):
        audit_temporal_overlap(pd.DataFrame({"FlrOnset": ["2009-02-01"], "event_type": [2]}))
    with pytest.raises(ValueError, match="onset for every sample"):
        audit_temporal_overlap(pd.DataFrame({"FlrOnset": [None], "event_type": [0]}))


def test_preparsed_onsets_normalize_timestamp_resolution():
    metadata = pd.DataFrame({
        "FlrOnset": pd.date_range("2025-01-01", periods=2, freq="12h", tz="UTC").as_unit("ms"),
        "event_type": [1, 0],
    })
    mask, audit = audit_temporal_overlap(metadata)
    assert mask.tolist() == [False, True]
    assert audit.overlap_bins.tolist() == [144]
