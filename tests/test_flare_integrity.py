"""Independent flare records cannot cross the SEP/NSEP boundary."""
import numpy as np
import pandas as pd
import pytest

from sep_dataset.flare_integrity import (
    IDENTITY_VERSION, assert_flare_label_integrity, audit_flare_identities,
    filter_nsep_candidates, normalize_noaa_region,
)


def frame(onsets, labels, **values):
    return pd.DataFrame({"FlrOnset": onsets, "event_type": labels, **values})


def test_exact_utc_onset_conflict_quarantines_both_classes_without_changing_labels():
    original = frame(["2025-01-01 12:00:00", "2025-01-01T05:00:00-07:00", "2025-01-02 12:00"], [1, 0, 0])
    before = original.copy(deep=True)
    result = audit_flare_identities(original)
    assert result["keep"].tolist() == [False, False, True]
    assert result["canonical_ids"].iloc[0] == result["canonical_ids"].iloc[1]
    assert result["audit"].exclusion_reason.tolist() == ["conflicting_sep_nsep_labels", "conflicting_sep_nsep_labels", ""]
    assert result["summary"]["conflicting_label_groups"] == 1
    assert result["summary"]["conflicting_label_rows"] == 2
    assert result["summary"]["cross_class_flare_overlap_after"] == 0
    pd.testing.assert_frame_equal(original, before)


def test_same_label_duplicates_keep_complete_record_and_have_stable_ids_after_reordering():
    original = frame(["2025-01-01 12:00"] * 3, [1, 1, 1],
                     noaa_active_region=[13944.0, np.nan, "13944"],
                     peak_time=["2025-01-01 12:10", None, "2025-01-01 12:10"],
                     catalog_id=["B", "A", "C"])
    result = audit_flare_identities(original)
    assert result["keep"].tolist() == [True, False, False]
    assert result["summary"]["same_label_duplicate_rows_removed"] == 2
    reordered = original.iloc[[2, 1, 0]].reset_index(drop=True)
    rerun = audit_flare_identities(reordered)
    kept = reordered.loc[rerun["keep"]]
    assert kept.catalog_id.tolist() == ["B"]
    assert result["canonical_ids"].iloc[0] == rerun["canonical_ids"].iloc[0]
    assert_flare_label_integrity(original.loc[result["keep"]])


def test_near_onsets_need_same_region_and_shared_peak_to_find_cross_catalog_conflict():
    data = frame(["2025-01-01 12:00", "2025-01-01 12:05", "2025-01-01 14:00"], [1, 0, 0],
                 noaa_active_region=["13944.0", 13944, 13944],
                 Flrmaxtime=["2025-01-01 12:10", None, "2025-01-01 14:10"],
                 peak_time=[None, "2025-01-01T12:10:00Z", None])
    result = audit_flare_identities(data)
    assert result["keep"].tolist() == [False, False, True]
    assert "same_peak_and_region" in result["audit"].identity_evidence.iloc[0]
    assert result["summary"]["retained_rows"] == 1


def test_nearby_or_overlapping_flares_are_not_duplicates_without_identity_evidence():
    data = frame(["2025-01-01 12:00", "2025-01-01 12:05", "2025-01-01 12:08"], [1, 0, 0],
                 noaa_active_region=[13944, 13944, 13945],
                 peak_time=["2025-01-01 12:20", "2025-01-01 12:25", "2025-01-01 12:20"],
                 sep_start=["2025-01-01 12:30"] * 3, sep_end=["2025-01-02 12:30"] * 3)
    result = audit_flare_identities(data)
    assert result["keep"].all()
    assert result["canonical_ids"].nunique() == 3
    assert_flare_label_integrity(data)


def test_unknown_or_truncated_region_does_not_prove_nearby_flare_identity():
    data = frame(["2025-01-01 12:00", "2025-01-01 12:02", "2025-01-01 12:04"], [1, 0, 0],
                 noaa_active_region=[13944, 3944, np.nan],
                 peak_time=["2025-01-01 12:10"] * 3)
    assert audit_flare_identities(data)["keep"].all()
    assert normalize_noaa_region(13944.0) == "13944"
    for value in (None, np.nan, "", "0", "-99999", "13944.3", "13944,13945"):
        assert normalize_noaa_region(value) is None


def test_exact_onset_region_ambiguity_is_reported_instead_of_selecting_one_region():
    data = frame(["2025-01-01 12:00"] * 2, [1, 1], noaa_active_region=[13944, 13945])
    result = audit_flare_identities(data)
    assert not result["keep"].any()
    assert result["summary"]["ambiguous_region_groups"] == 1
    assert result["audit"].region_ambiguity.all()
    assert result["audit"].exclusion_reason.eq("ambiguous_region_identity").all()


def test_clock_only_peak_is_anchored_to_flare_date_and_can_cross_midnight():
    data = frame(["2025-01-01 23:58", "2025-01-02 00:01"], [1, 0],
                 noaa_active_region=[13944, 13944], peak_time=["00:03", "00:03:00"])
    result = audit_flare_identities(data)
    assert not result["keep"].any()
    assert result["summary"]["conflicting_label_groups"] == 1


def test_label_conflict_and_region_ambiguity_are_both_counted_when_present():
    data = frame(["2025-01-01 12:00"] * 2, [1, 0], noaa_active_region=[13944, 13945])
    result = audit_flare_identities(data)
    assert not result["keep"].any()
    assert result["summary"]["conflicting_label_groups"] == 1
    assert result["summary"]["ambiguous_region_groups"] == 1


def test_date_only_or_invalid_auxiliary_timestamps_do_not_merge_flares():
    data = frame(["2025-01-01 12:00", "2025-01-01 12:05"], [1, 0],
                 noaa_active_region=[13944, 13944], peak_time=["2025-01-01", "2025-01-01"],
                 end_time=["-99999", "-99999"])
    assert audit_flare_identities(data)["keep"].all()


def test_near_identity_does_not_chain_across_unbounded_onset_span():
    data = frame(["2025-01-01 12:00", "2025-01-01 12:09", "2025-01-01 12:18"], [1, 1, 1],
                 noaa_active_region=[13944] * 3, peak_time=["2025-01-01 12:20"] * 3)
    result = audit_flare_identities(data)
    assert result["keep"].tolist() == [True, False, True]
    assert result["summary"]["same_label_duplicate_rows_removed"] == 1


@pytest.mark.parametrize("labels", [[1, 0], [1, 1]])
def test_release_assertion_rejects_both_conflicting_and_same_class_duplicates(labels):
    with pytest.raises(ValueError, match="Flare identity validation failed"):
        assert_flare_label_integrity(frame(["2025-01-01"] * 2, labels))


def test_collection_rejects_negatives_matching_known_positive_and_deduplicates_remaining_candidates():
    sep = frame(["2025-01-01 12:00"], [1], noaa_active_region=[13944], peak_time=["2025-01-01 12:10"])
    candidates = frame(["2025-01-01 12:04", "2025-01-02 12:00", "2025-01-02 12:00"], [0, 0, 0],
                       noaa_active_region=[13944, 13944, 13944],
                       peak_time=["2025-01-01 12:10", "2025-01-02 12:10", "2025-01-02 12:10"])
    sep_before, candidates_before = sep.copy(deep=True), candidates.copy(deep=True)
    retained, audit = filter_nsep_candidates(candidates, sep)
    assert retained.FlrOnset.tolist() == ["2025-01-02 12:00"]
    assert audit.retained.tolist() == [False, True, False]
    assert audit.source_row_position.tolist() == [0, 1, 2]
    assert audit.exclusion_reason.iloc[0] == "conflicting_sep_nsep_labels"
    assert audit.representative_source_row_position.iloc[1] == 1
    assert_flare_label_integrity(pd.concat([sep, retained], ignore_index=True))
    pd.testing.assert_frame_equal(sep, sep_before)
    pd.testing.assert_frame_equal(candidates, candidates_before)


def test_minimal_unique_metadata_and_empty_frames_are_supported_without_mutation():
    data = frame(["2025-01-01", "2025-01-02"], ["0", "1"])
    data.index = [5, 5]
    result = audit_flare_identities(data)
    assert result["keep"].all()
    assert result["canonical_ids"].index.tolist() == [5, 5]
    assert result["summary"]["identity_version"] == IDENTITY_VERSION
    assert assert_flare_label_integrity(data)["retained_rows"] == 2
    empty = audit_flare_identities(data.iloc[:0])
    assert len(empty["audit"]) == 0
    assert empty["summary"]["retained_rows"] == 0


@pytest.mark.parametrize("data", [frame(["bad"], [0]), frame(["2025-01-01"], [2]), pd.DataFrame({"FlrOnset": ["2025-01-01"]})])
def test_missing_identity_or_invalid_labels_fail_closed(data):
    with pytest.raises(ValueError):
        audit_flare_identities(data)


def test_near_duplicate_canonical_id_remains_stable_after_deterministic_representative_selection():
    data = frame(["2025-01-01 12:04", "2025-01-01 12:00"], [1, 1],
                 noaa_active_region=[13944, 13944], peak_time=["2025-01-01 12:10"] * 2)
    result = audit_flare_identities(data)
    retained = data.iloc[np.flatnonzero(result["keep"])]
    reduced = audit_flare_identities(retained)
    assert result["canonical_ids"].iloc[0] == reduced["canonical_ids"].iloc[0]
