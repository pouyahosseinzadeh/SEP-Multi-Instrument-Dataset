"""Conservative physical-flare identity checks for SEP/NSEP samples.

Exact UTC flare onsets are a primary identity key. Nearby onsets are joined
only when the same known NOAA active-region number and an exact shared flare
start, peak or end timestamp provide additional evidence. Merely overlapping
flare intervals, proximity, class or a shared SEP episode never identify two
flares as the same physical eruption.

Conflicting binary labels are quarantined, not resolved by guessing which
catalog is correct. This module never changes a supplied label or metadata.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import re

import numpy as np
import pandas as pd


IDENTITY_VERSION = "physical-flare-identity-v1"
_ANCHORS = {
    "start": ("start_time", "flare_start_time"),
    "peak": ("Flrmaxtime", "peak_time", "flare_peak_time"),
    "end": ("Flrendtime", "end_time", "flare_end_time"),
}
_REGION_FIELDS = ("noaa_active_region", "noaa_region", "active_region")
_AUDIT_COLUMNS = (
    "source_row_position", "source_index", "canonical_flare_id", "event_type",
    "FlrOnset_utc", "noaa_region_normalized", "group_size", "group_label_count",
    "region_ambiguity", "identity_evidence", "retained", "exclusion_reason",
    "representative_source_row_position",
)


def normalize_noaa_region(value):
    """Normalize an integer-valued NOAA identifier; never infer a new century.

    Unknown, zero, fractional, multi-region and negative/fill values remain
    unknown. In particular 3944 and 13944 are not silently treated as aliases.
    """
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not re.fullmatch(r"[+]?\d+(?:\.0+)?", text):
        return None
    number = int(text.split(".")[0])
    return str(number) if 0 < number < 1000000 else None


def _onsets(metadata):
    if "FlrOnset" not in metadata or "event_type" not in metadata:
        raise ValueError("Flare identity requires FlrOnset and event_type")
    labels = pd.to_numeric(metadata["event_type"], errors="coerce")
    if not labels.isin([0, 1]).all():
        raise ValueError("Flare labels must be 0 (NSEP) or 1 (SEP)")
    # Parse as UTC even when catalogs use different offsets or mix naive UTC
    # with timezone-bearing strings. A missing onset is never an identity key.
    onsets = pd.to_datetime(metadata["FlrOnset"], utc=True, format="mixed", errors="coerce")
    if onsets.isna().any():
        raise ValueError("Invalid or missing FlrOnset prevents flare identity validation")
    return onsets.astype("datetime64[ns, UTC]").reset_index(drop=True), labels.to_numpy(dtype=int)


def _regions(metadata):
    result = [None] * len(metadata)
    for name in _REGION_FIELDS:
        if name not in metadata:
            continue
        for i, value in enumerate(metadata[name]):
            if result[i] is None:
                result[i] = normalize_noaa_region(value)
    return result


def _anchor_times(metadata, aliases, onsets, *, stage):
    """Take the first usable alias; never parse an isolated clock on today."""
    result = pd.Series(pd.NaT, index=range(len(metadata)), dtype="datetime64[ns, UTC]")
    day = onsets.dt.normalize()
    for name in aliases:
        if name not in metadata:
            continue
        text = metadata[name].reset_index(drop=True).astype("string").str.strip()
        clock = text.str.fullmatch(r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?", na=False)
        # A date-only placeholder is not evidence of a shared peak/end/start.
        date_only = text.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
        numeric = text.str.fullmatch(r"[+-]?\d+(?:\.\d+)?", na=False)
        value = pd.to_datetime(text.where(~(clock | date_only | numeric)), utc=True,
                               format="mixed", errors="coerce")
        if clock.any():
            clock_text = text.where(clock).str.replace(r"^(\d{1,2}:\d{2})$", r"\1:00", regex=True)
            delta = pd.to_timedelta(clock_text, errors="coerce")
            clock_values = day + delta
            if stage != "start":
                clock_values = clock_values.where(clock_values >= onsets, clock_values + pd.Timedelta(days=1))
            value = value.where(~clock, clock_values)
        # Grossly inconsistent auxiliary times cannot merge unrelated events.
        good = (value >= onsets - pd.Timedelta(minutes=10)) & (value <= onsets + pd.Timedelta(days=1))
        value = value.where(good)
        result = result.fillna(value)
    return result


def _rank_value(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def audit_flare_identities(metadata, near_tolerance_minutes=10):
    """Return an audited selection with at most one row per physical flare.

    ``keep`` is a positional bool array. ``canonical_ids`` has the input index.
    ``audit`` contains one record per input row and explicit exclusion reasons.
    Conflicting labels and exact-onset region ambiguities quarantine every row
    in that group. Same-label duplicates keep the earliest onset, then the most
    complete metadata record, then a stable content tie-break. Retained rows
    preserve their original order. Identity comparisons require no network.
    """
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame")
    tolerance = float(near_tolerance_minutes)
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 60:
        raise ValueError("near_tolerance_minutes must be between zero and 60")
    onsets, labels = _onsets(metadata)
    n = len(metadata)
    regions = _regions(metadata)
    anchors = {name: _anchor_times(metadata, aliases, onsets, stage=name)
               for name, aliases in _ANCHORS.items()}
    parent = list(range(n))
    low = onsets.astype("int64").to_numpy(copy=True)
    high = low.copy()
    known_regions = [{r} if r is not None else set() for r in regions]
    evidence = [set() for _ in range(n)]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    limit_ns = int(pd.Timedelta(minutes=tolerance).value)

    def join(a, b, rule, exact=False):
        a, b = find(a), find(b)
        if a == b:
            evidence[a].add(rule)
            return True
        first, last = min(low[a], low[b]), max(high[a], high[b])
        combined_regions = known_regions[a] | known_regions[b]
        if not exact and (last - first > limit_ns or len(combined_regions) > 1):
            return False
        # Index-independent group statistics are retained even though the
        # internal union root is an implementation detail.
        if a > b:
            a, b = b, a
        parent[b] = a
        low[a], high[a] = first, last
        known_regions[a] = combined_regions
        evidence[a] |= evidence[b] | {rule}
        return True

    exact_buckets = defaultdict(list)
    for i, onset in enumerate(onsets):
        exact_buckets[onset.value].append(i)
    for bucket in exact_buckets.values():
        for i in bucket[1:]:
            join(bucket[0], i, "same_utc_onset", exact=True)

    if limit_ns:
        for stage, values in anchors.items():
            buckets = defaultdict(list)
            for i, value in enumerate(values):
                if regions[i] is not None and pd.notna(value):
                    buckets[(regions[i], value.value)].append(i)
            for bucket in buckets.values():
                bucket.sort(key=lambda i: (onsets.iloc[i].value, i))
                previous = bucket[0]
                for i in bucket[1:]:
                    if onsets.iloc[i].value - onsets.iloc[previous].value <= limit_ns:
                        join(previous, i, "same_" + stage + "_and_region")
                    previous = i

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    keep = np.zeros(n, dtype=bool)
    canonical = np.empty(n, dtype=object)
    rows = []
    conflict_groups = conflict_rows = ambiguous_groups = duplicate_groups = duplicate_removed = 0
    # Exclude measurement blocks from completeness and tie-break logic if a
    # caller accidentally supplies a whole feature file rather than metadata.
    fields = sorted(c for c in metadata.columns if not re.fullmatch(r".+_\d{1,3}", str(c)))
    for root, members in groups.items():
        group_labels = set(labels[members].tolist())
        region_set = known_regions[find(root)]
        region_ambiguity = len(region_set) > 1

        def rank(i):
            row = metadata.iloc[i]
            values = [_rank_value(row[c]) for c in fields]
            completeness = sum(bool(v) and v.lower() not in ("nan", "nat", "none", "<na>") for v in values)
            return (onsets.iloc[i].value, -completeness,
                    json.dumps(values, ensure_ascii=True, separators=(",", ":")), i)

        representative = min(members, key=rank)
        if region_ambiguity:
            ambiguous_groups += 1
        if len(group_labels) > 1:
            reason = "conflicting_sep_nsep_labels"
            conflict_groups += 1
            conflict_rows += len(members)
            representative_retained = -1
        elif region_ambiguity:
            reason = "ambiguous_region_identity"
            representative_retained = -1
        else:
            keep[representative] = True
            reason = "duplicate_same_label_flare"
            representative_retained = representative
            if len(members) > 1:
                duplicate_groups += 1
                duplicate_removed += len(members) - 1
        # The representative's earliest onset and normalized region define a
        # stable key, independent of source row order and metadata aliases.
        canonical_region = next(iter(region_set)) if len(region_set) == 1 else "|".join(sorted(region_set))
        payload = onsets.iloc[representative].isoformat() + "|" + canonical_region
        flare_id = "flare-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        canonical[members] = flare_id
        for i in members:
            rows.append({"source_row_position": i, "source_index": _rank_value(metadata.index[i]),
                         "canonical_flare_id": flare_id, "event_type": int(labels[i]),
                         "FlrOnset_utc": onsets.iloc[i].isoformat(),
                         "noaa_region_normalized": regions[i] or "unknown", "group_size": len(members),
                         "group_label_count": len(group_labels), "region_ambiguity": region_ambiguity,
                         "identity_evidence": " | ".join(sorted(evidence[find(root)])) or "unique_onset",
                         "retained": bool(keep[i]), "exclusion_reason": "" if keep[i] else reason,
                         "representative_source_row_position": representative_retained})
    audit = pd.DataFrame(rows, columns=_AUDIT_COLUMNS).sort_values("source_row_position").reset_index(drop=True)
    kept_ids = canonical[keep]
    if len(set(kept_ids.tolist())) != int(keep.sum()):
        raise AssertionError("Physical flare selection retained a duplicate identity")
    cross_class_after = len(set(canonical[keep & (labels == 1)].tolist())
                            & set(canonical[keep & (labels == 0)].tolist()))
    if cross_class_after:
        raise AssertionError("A physical flare remains in both SEP and NSEP classes")
    canonical_series = pd.Series(canonical, index=metadata.index, name="canonical_flare_id", dtype=object)
    summary = {"identity_version": IDENTITY_VERSION, "near_tolerance_minutes": tolerance,
               "input_rows": n, "retained_rows": int(keep.sum()), "excluded_rows": int((~keep).sum()),
               "identity_groups": len(groups), "conflicting_label_groups": conflict_groups,
               "conflicting_label_rows": conflict_rows, "ambiguous_region_groups": ambiguous_groups,
               "same_label_duplicate_groups": duplicate_groups, "same_label_duplicate_rows_removed": duplicate_removed,
               "sep_rows_before": int((labels == 1).sum()), "nsep_rows_before": int((labels == 0).sum()),
               "sep_rows_after": int((keep & (labels == 1)).sum()), "nsep_rows_after": int((keep & (labels == 0)).sum()),
               "cross_class_flare_overlap_after": cross_class_after}
    return {"keep": keep, "canonical_ids": canonical_series, "audit": audit, "summary": summary}


def assert_flare_label_integrity(metadata, near_tolerance_minutes=10):
    """Fail if a release repeats a physical flare or assigns conflicting labels."""
    result = audit_flare_identities(metadata, near_tolerance_minutes=near_tolerance_minutes)
    summary = result["summary"]
    if summary["excluded_rows"]:
        raise ValueError(
            "Flare identity validation failed: {} conflicting SEP/NSEP groups, {} ambiguous region groups, "
            "{} duplicate same-label rows; inspect the flare identity audit before releasing samples".format(
                summary["conflicting_label_groups"], summary["ambiguous_region_groups"],
                summary["same_label_duplicate_rows_removed"]))
    return summary


def filter_nsep_candidates(candidates, sep_samples, near_tolerance_minutes=10):
    """Exclude proposed negatives matching any known positive physical flare.

    Existing positive metadata is not changed. Proposed negatives are also
    deduplicated among themselves. Return a retained candidate copy and a
    candidate-only audit with source positions relative to ``candidates``.
    """
    _, negative_labels = _onsets(candidates)
    _, positive_labels = _onsets(sep_samples)
    if not (negative_labels == 0).all() or not (positive_labels == 1).all():
        raise ValueError("NSEP candidates must all have label 0 and known SEP samples label 1")
    combined = pd.concat([sep_samples, candidates], ignore_index=True, sort=False)
    result = audit_flare_identities(combined, near_tolerance_minutes=near_tolerance_minutes)
    offset = len(sep_samples)
    keep = result["keep"][offset:]
    audit = result["audit"].iloc[offset:].copy().reset_index(drop=True)
    audit["source_row_position"] -= offset
    representative = audit["representative_source_row_position"]
    audit["representative_source_row_position"] = representative.where(representative < 0, representative - offset)
    audit["source_index"] = [_rank_value(i) for i in candidates.index]
    return candidates.iloc[np.flatnonzero(keep)].copy(), audit
