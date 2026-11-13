"""Audit overlap between the exported SEP and NSEP predictor windows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .compatible_baseline import STEPS


OVERLAP_POLICY = "shared_five_minute_predictor_bins_against_all_source_sep_v1"
BIN = pd.Timedelta(minutes=5)
WINDOW = STEPS * BIN


def audit_temporal_overlap(metadata):
    """Return an exclusion mask and the earliest SEP witness per flagged NSEP.

    The half-open interval from the first stored bin to the end of the last
    stored bin contains all 288 grid slots in compatible_builder.window_index.
    All supplied positive windows protect those slots, even if later filtering
    drops a positive row. Original input positions, rather than DataFrame
    index labels, identify each NSEP and SEP witness.
    """
    if not {"FlrOnset", "event_type"}.issubset(metadata):
        raise ValueError("Time overlap audit requires FlrOnset and event_type")
    onsets = pd.to_datetime(metadata.FlrOnset, utc=True, format="mixed", errors="raise").astype("datetime64[ns, UTC]")
    if onsets.isna().any():
        raise ValueError("Time overlap audit requires an onset for every sample")
    labels = pd.to_numeric(metadata.event_type, errors="coerce")
    if not labels.isin([0, 1]).all():
        raise ValueError("Time overlap audit requires binary SEP/NSEP labels")

    stops = onsets.dt.ceil("5min")
    starts = stops - WINDOW
    stop_ns, start_ns = stops.array.asi8, starts.array.asi8
    positives = np.flatnonzero(labels.eq(1).to_numpy())
    negatives = np.flatnonzero(labels.eq(0).to_numpy())
    ordered = positives[np.argsort(stop_ns[positives], kind="stable")]

    # All windows have the same duration. Find the first SEP whose end
    # is strictly after the NSEP start; if it starts after the NSEP ends,
    # all subsequent SEP windows do too. A touching boundary is allowed.
    first = np.searchsorted(stop_ns[ordered], start_ns[negatives], side="right")
    possible = first < len(ordered)
    nsep_rows, sep_rows = negatives[possible], ordered[first[possible]]
    overlaps = start_ns[sep_rows] < stop_ns[nsep_rows]
    nsep_rows, sep_rows = nsep_rows[overlaps], sep_rows[overlaps]

    exclude = np.zeros(len(metadata), dtype=bool)
    exclude[nsep_rows] = True
    overlap_ns = np.minimum(stop_ns[nsep_rows], stop_ns[sep_rows]) - np.maximum(start_ns[nsep_rows], start_ns[sep_rows])
    audit = pd.DataFrame({
        "source_row_index": nsep_rows,
        "nsep_onset_utc": onsets.array[nsep_rows],
        "nsep_window_start_utc": starts.array[nsep_rows],
        "nsep_window_stop_exclusive_utc": stops.array[nsep_rows],
        "witness_sep_source_row_index": sep_rows,
        "witness_sep_onset_utc": onsets.array[sep_rows],
        "witness_sep_window_start_utc": starts.array[sep_rows],
        "witness_sep_window_stop_exclusive_utc": stops.array[sep_rows],
        "overlap_bins": overlap_ns // BIN.value,
    })
    return exclude, audit


def assert_no_temporal_overlap(metadata):
    """Reject a release whose SEP and NSEP predictor intervals intersect."""
    exclude, _ = audit_temporal_overlap(metadata)
    if exclude.any():
        raise ValueError(f"{exclude.sum()} NSEP predictor windows overlap a SEP predictor window")
