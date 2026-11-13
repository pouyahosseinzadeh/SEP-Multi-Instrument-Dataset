"""Leakage-controlled univariate RF and genuine Time Series Forest benchmarks.

One case is one flare's 288-point pre-flare window from one physical feature.
The same chronological split is used for every feature and both estimators.
Decision thresholds are selected on validation data only. No dataset values or
labels are changed, and the held-out test set is never resampled or tuned.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


FEATURES = ("Vx", "flow_speed", "proton_density", "T", "F", "Xs", "Xl", "P4", "P5", "P6")
MODELS = ("RandomForest", "TimeSeriesForest")
DISPLAY_MODELS = {"RandomForest": "Random Forest", "TimeSeriesForest": "Time Series Forest"}
ALLOWED_WORKFLOWS = ("repaired_ten_feature_release_v2",)
VERSION = "univariate-purged-chronological-v2"
PARTITIONS = ("train", "validation", "test")
LABEL_POLICY = "All release rows with binary event_type labels enter the split audit. SEP is event_type == 1. No proton energy or peak-flux cutoff is applied."
SUMMARY_METRICS = ("TSS", "HSS", "F1", "Precision", "Recall", "ROC_AUC", "AveragePrecision", "BalancedAccuracy")
METRIC_LABELS = ("TSS", "HSS", "F1", "Precision", "Recall", "ROC AUC", "AP", "Balanced acc.")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, data):
    """Write strict JSON; unavailable numerical results are null, never NaN."""
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    part = Path(str(path) + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")
    part.replace(path)


def _csv(path, data):
    part = Path(str(path) + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(part, index=False)
    part.replace(path)


def _dates(series):
    return pd.to_datetime(series, utc=True, format="mixed", errors="coerce")


def _boundary(value):
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.normalize() + pd.Timedelta(days=1)


def _episode_groups(metadata):
    """Group positive windows sharing known episode starts or ends.

    Empty timestamps never become a common group. Labels are not inferred from
    these groups. Episode intervals crossing a split boundary are purged below.
    """
    n = len(metadata)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    positive = pd.to_numeric(metadata.event_type, errors="raise").eq(1).to_numpy()
    for col in ("sep_episode_id", "sep_start", "sep_end"):
        if col not in metadata:
            continue
        values = metadata[col].astype(str) if col == "sep_episode_id" else _dates(metadata[col]).astype(str)
        seen = {}
        for i, value in enumerate(values):
            if not positive[i] or value.lower() in ("nat", "nan", "none", "", "-9999", "-99999"):
                continue
            if value in seen:
                union(i, seen[value])
            else:
                seen[value] = i
    return np.asarray([find(i) for i in range(n)], dtype=int)


def chronological_splits(metadata, sample_ids=None, canonical_ids=None, *,
                         train_end="2014-12-31", validation_end="2019-12-31", purge_hours=30,
                         calibration_dates=()):
    """A fixed temporal holdout, with input/label buffers and episode purging.

    Default 30 h = the 24 h input window plus the 6 h label search horizon.
    Groups crossing a boundary are quarantined rather than reassigned. The
    split includes every source row, with explicit reasons for excluded rows.
    """
    if purge_hours < 30:
        raise ValueError("The purge must be at least 30 hours (24 h input + 6 h label horizon)")
    dates = _dates(metadata.FlrOnset)
    labels = pd.to_numeric(metadata.event_type, errors="raise")
    if dates.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Benchmark inputs require valid UTC flare times and binary SEP/NSEP labels")
    boundaries = (_boundary(train_end), _boundary(validation_end))
    if boundaries[0] >= boundaries[1]:
        raise ValueError("Training must finish before validation finishes")
    n = len(metadata)
    ids = pd.Series(sample_ids if sample_ids is not None else ["sample-{}".format(i) for i in range(n)])
    canonical = pd.Series(canonical_ids if canonical_ids is not None else dates.astype(str))
    if len(ids) != n or len(canonical) != n or ids.isna().any() or ids.duplicated().any():
        raise ValueError("Sample identities must be present, unique and aligned")
    if canonical.isna().any() or canonical.duplicated().any():
        raise ValueError("The same canonical flare cannot appear twice in benchmark inputs")
    ids, canonical = ids.astype(str), canonical.astype(str)
    if ids.str.strip().eq("").any() or canonical.str.strip().eq("").any():
        raise ValueError("Sample and canonical flare identities cannot be blank")
    split = np.where(dates < boundaries[0], "train", np.where(dates < boundaries[1], "validation", "test"))
    initial = split.copy()
    reason = np.full(n, "", dtype=object)
    for edge in boundaries:
        selected = ((dates >= edge) & (dates < edge + pd.Timedelta(hours=purge_hours))).to_numpy()
        split[selected], reason[selected] = "purged", "boundary_input_and_label_buffer"
    episodes = _episode_groups(metadata)
    for group in np.unique(episodes):
        members = episodes == group
        if len(set(initial[members])) > 1:
            split[members], reason[members] = "purged", "SEP_episode_crosses_split"
    # Also reject a known ongoing episode which straddles a boundary, even if
    # only one member flare happens to be present in the sampled cohort.
    if {"sep_start", "sep_end"}.issubset(metadata):
        starts, stops = _dates(metadata.sep_start), _dates(metadata.sep_end)
        for edge in boundaries:
            selected = (labels.eq(1) & starts.lt(edge) & (stops.ge(edge) | (stops.isna() & dates.ge(edge)))).to_numpy()
            for group in np.unique(episodes[selected]):
                members = episodes == group
                split[members], reason[members] = "purged", "SEP_episode_interval_crosses_split"
    calibration_days = _normalise_calibration_days(calibration_dates)
    overlap_dates = [set() for _ in range(n)]
    direct_input_embargo = np.zeros(n, dtype=bool)
    starts, ends = dates - pd.Timedelta(hours=24), dates + pd.Timedelta(hours=6)
    episode_starts = _dates(metadata.sep_start) if "sep_start" in metadata else pd.Series(pd.NaT, index=metadata.index, dtype="datetime64[ns, UTC]")
    episode_ends = _dates(metadata.sep_end) if "sep_end" in metadata else pd.Series(pd.NaT, index=metadata.index, dtype="datetime64[ns, UTC]")
    # An unresolved episode is known to extend at least to this sampled flare's
    # label-search horizon; do not assume that it continued indefinitely.
    episode_ends = episode_ends.fillna(ends)
    for day in calibration_days:
        edge = pd.Timestamp(day, tz="UTC")
        direct = ((starts < edge + pd.Timedelta(days=1)) & (ends >= edge)).to_numpy()
        direct_input_embargo |= direct
        episode = (labels.eq(1) & episode_starts.lt(edge + pd.Timedelta(days=1)) & episode_ends.ge(edge)).to_numpy()
        for i in np.flatnonzero(direct | episode):
            overlap_dates[i].add(day)
    calibration_embargo = np.asarray([bool(days) for days in overlap_dates], dtype=bool)
    direct_embargo = direct_input_embargo
    # A known positive episode is one physical source-development event, even
    # when a later member window no longer overlaps the calibration day itself.
    for group in np.unique(episodes[calibration_embargo & labels.eq(1).to_numpy()]):
        members = np.flatnonzero(episodes == group)
        group_days = set().union(*(overlap_dates[i] for i in members))
        for i in members:
            calibration_embargo[i] = True
            overlap_dates[i].update(group_days)
    split[calibration_embargo] = "purged"
    reason[calibration_embargo] = "SEP_episode_calibration_embargo"
    reason[direct_embargo] = "calibration_observation_day_embargo"
    result = pd.DataFrame({"dataset_row_index": np.arange(n), "sample_id": ids.to_numpy(),
                           "canonical_flare_id": canonical.to_numpy(), "FlrOnset": dates.astype(str).to_numpy(),
                           "event_type": labels.to_numpy(dtype=int), "episode_group": episodes,
                           "calendar_partition": initial, "split": split, "exclusion_reason": reason,
                           "calibration_embargo": calibration_embargo,
                           "calibration_overlap_dates": [" | ".join(sorted(days)) for days in overlap_dates]})
    for earlier, later in (("train", "validation"), ("validation", "test"), ("train", "test")):
        a, b = dates[result.split.eq(earlier)], dates[result.split.eq(later)]
        if len(a) and len(b) and b.min() - a.max() < pd.Timedelta(hours=purge_hours):
            raise AssertionError("Temporal input/label windows overlap across partitions")
    return result


def _normalise_calibration_days(values):
    if not isinstance(values, (list, tuple, set)):
        raise ValueError("Calibration observation dates must be a list of YYYY-MM-DD days")
    result = set()
    for value in values:
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("Calibration observation dates must use YYYY-MM-DD")
        try:
            parsed = pd.Timestamp(value)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid calibration observation day") from error
        if parsed.strftime("%Y-%m-%d") != value:
            raise ValueError("Invalid calibration observation day")
        result.add(value)
    return sorted(result)


def _calibration_observation_days(root, manifest):
    path = root / "calibration_bank.json"
    if not path.is_file():
        raise ValueError("Benchmark requires the copied calibration bank to embargo its development observations")
    digest = _sha(path)
    expected = manifest.get("audit_files_sha256", {}).get(path.name)
    if expected and digest != expected:
        raise ValueError("Benchmark calibration-bank checksum mismatch")
    bank = json.loads(path.read_text())
    if not isinstance(bank.get("artifacts"), list):
        raise ValueError("Calibration bank must provide its artifact list")
    days = set()
    for artifact in bank["artifacts"]:
        values = artifact.get("provenance", {}).get("calibration_dates")
        if not isinstance(values, list) or not values:
            raise ValueError("A calibration artifact lacks its actual fitting/held-out observation dates; update the calibration bank before benchmarking")
        days.update(_normalise_calibration_days(values))
    return sorted(days), digest


def classification_metrics(y, probability, threshold=0.5):
    """Binary scores with explicit undefined cases; no zero-filled AUCs."""
    from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score
    y, probability = np.asarray(y, int), np.asarray(probability, float)
    if len(y) != len(probability) or not np.isin(y, [0, 1]).all() or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Scores require aligned binary labels and finite probabilities in [0, 1]")
    pred = probability >= threshold
    tp, fp = int(((y == 1) & pred).sum()), int(((y == 0) & pred).sum())
    tn, fn = int(((y == 0) & ~pred).sum()), int(((y == 1) & ~pred).sum())
    divide = lambda a, b: float(a / b) if b else np.nan
    recall, fpr, specificity = divide(tp, tp + fn), divide(fp, fp + tn), divide(tn, fp + tn)
    both = bool((tp + fn) and (fp + tn))
    scores = {"n": len(y), "SEP": tp + fn, "NSEP": fp + tn, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
              "threshold": float(threshold), "TSS": recall - fpr if both else np.nan,
              "HSS": divide(2 * (tp * tn - fp * fn), (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)),
              "F1": divide(2 * tp, 2 * tp + fp + fn), "Precision": divide(tp, tp + fp),
              "Recall": recall, "Specificity": specificity, "FalseAlarmRate": fpr,
              "BalancedAccuracy": (recall + specificity) / 2 if both else np.nan,
              "Accuracy": divide(tp + tn, len(y)), "ROC_AUC": np.nan, "PR_AUC": np.nan,
              "AveragePrecision": np.nan, "BrierScore": float(np.mean((probability - y) ** 2)) if len(y) else np.nan,
              "MCC": divide(tp * tn - fp * fn, np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))}
    if both:
        scores["ROC_AUC"] = float(roc_auc_score(y, probability))
        scores["AveragePrecision"] = float(average_precision_score(y, probability))
        precision, recall_curve, _ = precision_recall_curve(y, probability)
        scores["PR_AUC"] = float(auc(recall_curve, precision))
    undefined = [key for key in SUMMARY_METRICS if not np.isfinite(scores[key])]
    scores["undefined_metrics"] = ",".join(undefined)
    scores["status"] = "ok" if not undefined else "undefined_metrics; empty/single-class cohort or no positive predictions"
    return scores


def select_threshold(y_validation, probability_validation):
    """Maximize validation TSS; ties nearest .5, then higher threshold."""
    from sklearn.metrics import roc_curve
    y, probability = np.asarray(y_validation, int), np.asarray(probability_validation, float)
    if len(y) != len(probability) or not np.isin(y, [0, 1]).all() or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid validation labels/probabilities")
    if len(np.unique(y)) < 2:
        return {"threshold": 0.5, "validation_tss": np.nan, "policy": "fixed_0.5_validation_has_one_or_zero_classes"}
    fpr, tpr, thresholds = roc_curve(y, probability, drop_intermediate=False)
    thresholds[~np.isfinite(thresholds)] = np.nextafter(1., 2.)
    candidates = list(zip(thresholds, tpr - fpr))
    candidates.append((0.5, classification_metrics(y, probability, 0.5)["TSS"]))
    threshold, score = min(candidates, key=lambda pair: (-pair[1], abs(pair[0] - 0.5), -pair[0]))
    return {"threshold": float(threshold), "validation_tss": float(score), "policy": "validation_only_maximum_TSS_ties_nearest_0.5_then_higher"}


def _versions():
    result = {}
    for package in ("numpy", "pandas", "scikit-learn", "sktime"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not_installed"
    return result


def _estimators(n_estimators, n_jobs, random_state):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sktime.classification.interval_based import TimeSeriesForestClassifier
    except ImportError as error:
        raise RuntimeError("Install benchmark dependencies in the project environment: .venv/bin/python -m pip install -r requirements-benchmark.txt") from error
    return {"RandomForest": lambda: RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced", n_jobs=n_jobs, random_state=random_state),
            "TimeSeriesForest": lambda: TimeSeriesForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state)}


def _load_inputs(root, features):
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Benchmark requires a completed repaired dataset manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("workflow") not in ALLOWED_WORKFLOWS:
        raise ValueError("Benchmark requires the repaired ten-feature release; old cleaned files do not establish proton consistency")
    entries = {e["feature"]: e for e in manifest["feature_files"]}
    metadata_columns = manifest["metadata_columns"]
    metadata, sample_ids, canonical_ids = None, None, None
    paths, eras, hashes = {}, {}, {}
    for feature in features:
        entry = entries.get(feature)
        if entry is None or Path(entry["filename"]).name != entry["filename"]:
            raise ValueError("Missing or invalid dataset path for {}".format(feature))
        path, quality_path = root / entry["filename"], root / "quality" / (feature + "_quality.csv")
        if not entry.get("sha256") or _sha(path) != entry["sha256"]:
            raise ValueError("Benchmark input checksum mismatch: {}".format(feature))
        if not entry.get("quality_sha256") or _sha(quality_path) != entry["quality_sha256"]:
            raise ValueError("Benchmark quality checksum mismatch: {}".format(feature))
        expected = metadata_columns + ["{}_{}".format(feature, i) for i in range(288, 0, -1)]
        if list(pd.read_csv(path, nrows=0).columns) != expected:
            raise ValueError("Benchmark input needs exactly its metadata and 288 time columns")
        current = pd.read_csv(path, usecols=metadata_columns, float_precision="round_trip", low_memory=False)
        quality = pd.read_csv(quality_path, dtype={"sample_id": str, "canonical_flare_id": str}, low_memory=False)
        if len(current) != manifest["total_rows_per_feature"] or len(quality) != len(current) or quality.row_index.tolist() != list(range(len(current))):
            raise ValueError("Misaligned benchmark metadata/quality rows")
        if metadata is None:
            metadata, sample_ids = current, quality.sample_id.copy()
            canonical_ids = quality.canonical_flare_id.copy() if "canonical_flare_id" in quality else None
        else:
            pd.testing.assert_frame_equal(metadata, current, check_dtype=False, check_exact=True)
            if not sample_ids.equals(quality.sample_id):
                raise ValueError("Benchmark feature sample IDs differ")
            if canonical_ids is not None and ("canonical_flare_id" not in quality or not canonical_ids.equals(quality.canonical_flare_id)):
                raise ValueError("Benchmark canonical flare identities differ")
        paths[feature] = path
        eras[feature] = quality.source_era.fillna("unknown").astype(str) if "source_era" in quality else pd.Series("unknown", index=range(len(current)))
        hashes[feature] = {"data": entry["sha256"], "quality": entry["quality_sha256"]}
    from .flare_integrity import assert_flare_label_integrity, audit_flare_identities
    assert_flare_label_integrity(metadata)
    audited = audit_flare_identities(metadata)["canonical_ids"].astype(str).reset_index(drop=True)
    if canonical_ids is None:
        canonical_ids = audited
    # Both stored and independently derived identities must describe one flare
    # per row; exact identity text may use different versioned naming schemes.
    if audited.duplicated().any() or canonical_ids.duplicated().any():
        raise ValueError("Duplicate physical flares cannot enter model evaluation")
    return manifest, metadata, sample_ids, canonical_ids, paths, eras, hashes


def _positive_probability(estimator, x):
    classes = np.asarray(estimator.classes_)
    if len(classes) != 2 or 1 not in classes:
        raise ValueError("A fitted benchmark model must contain both classes")
    return np.asarray(estimator.predict_proba(x), dtype=float)[:, int(np.flatnonzero(classes == 1)[0])]


def _save_plot(fig, path, plt):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_results(metrics, predictions, features, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), squeeze=False)
    for ax, model in zip(axes[0], MODELS):
        table = metrics[metrics.model.eq(model)].set_index("feature").reindex(features)
        matrix = table[list(SUMMARY_METRICS)].to_numpy(dtype=float)
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        for i in range(len(features)):
            for j in range(len(SUMMARY_METRICS)):
                ax.text(j, i, "—" if not np.isfinite(matrix[i, j]) else "{:.2f}".format(matrix[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if np.isfinite(matrix[i, j]) and abs(matrix[i, j]) > .65 else "black")
        ax.set(yticks=np.arange(len(features)), yticklabels=features, xticks=np.arange(len(SUMMARY_METRICS)), xticklabels=METRIC_LABELS, title=DISPLAY_MODELS[model])
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.subplots_adjust(left=.10, bottom=.22, wspace=.40, right=.89, top=.92)
    colorbar_axis = fig.add_axes([.925, .22, .014, .70])
    fig.colorbar(image, cax=colorbar_axis, label="Held-out test score")
    for ext in ("png", "pdf"):
        fig.savefig(outdir / ("univariate_metrics." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)
    colors = plt.get_cmap("tab10")
    for kind in ("ROC", "PR"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, model in zip(axes, MODELS):
            for i, feature in enumerate(features):
                data = predictions[(feature, model)]
                if len(data) == 0 or data.observed_label.nunique() < 2:
                    continue
                if kind == "ROC":
                    x, y, _ = roc_curve(data.observed_label, data.probability)
                else:
                    y, x, _ = precision_recall_curve(data.observed_label, data.probability)
                ax.plot(x, y, lw=1.2, color=colors(i), label=feature)
            if kind == "ROC":
                ax.plot([0, 1], [0, 1], "--", color=".5", lw=.8)
                ax.set(xlabel="False positive rate", ylabel="True positive rate")
            else:
                data = predictions[(features[0], model)]
                if len(data):
                    ax.axhline(data.observed_label.mean(), ls="--", color=".5", lw=.8)
                ax.set(xlabel="Recall", ylabel="Precision")
            ax.set(xlim=(0, 1), ylim=(0, 1.02), title=DISPLAY_MODELS[model])
            ax.grid(alpha=.15)
            handles, names = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, names, fontsize=7, ncol=2, frameon=False, loc="best")
        _save_plot(fig, outdir / ("univariate_" + kind.lower()), plt)


def _partition_counts(splits):
    rows = []
    for partition in (*PARTITIONS, "all"):
        group = splits if partition == "all" else splits[splits.calendar_partition.eq(partition)]
        purged = group.split.eq("purged")
        row = {"partition": partition}
        for label, name in ((0, "NSEP"), (1, "SEP")):
            selected = group.event_type.eq(label)
            row["input_" + name] = int(selected.sum())
            row["used_" + name] = int((selected & ~purged).sum())
            row["purged_" + name] = int((selected & purged).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _exclusion_counts(splits):
    purged = splits[splits.split.eq("purged")]
    keys = ["calendar_partition", "exclusion_reason"]
    if purged.empty:
        return pd.DataFrame(columns=[*keys, "NSEP", "SEP"])
    return (purged.groupby([*keys, "event_type"], observed=True).size().unstack(fill_value=0)
            .reindex(columns=[0, 1], fill_value=0)
            .rename(columns={0: "NSEP", 1: "SEP"}).reset_index())


def _write_analysis(metrics, splits, counts, exclusions, outdir, calibration_dates=()):
    lines = ["# Univariate chronological model evaluation", "",
             "Each model uses one feature's 288 pre-flare samples. The same saved split is used for every case.", "",
             LABEL_POLICY, "",
             "| Calendar partition | NSEP candidates | SEP candidates | NSEP used | SEP used | NSEP purged | SEP purged |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in counts.itertuples(index=False):
        lines.append(f"| {row.partition} | {row.input_NSEP} | {row.input_SEP} | {row.used_NSEP} | {row.used_SEP} | {row.purged_NSEP} | {row.purged_SEP} |")
    total = counts.iloc[-1]
    test = counts[counts.partition.eq("test")].iloc[0]
    lines += ["", f"The headline test scores use {test.used_SEP} held-out SEP samples. The dataset contains {total.input_SEP} SEP samples across all calendar partitions, including {total.purged_SEP} excluded from model evaluation."]
    if not exclusions.empty:
        lines += ["", "| Calendar partition | Exclusion reason | NSEP | SEP |", "|---|---|---:|---:|"]
        for row in exclusions.itertuples(index=False):
            lines.append(f"| {row.calendar_partition} | `{row.exclusion_reason}` | {row.NSEP} | {row.SEP} |")
    lines += ["", "Thresholds maximize validation TSS. Test labels do not choose thresholds, models or feature transforms. Fixed-0.5 results are supplied separately.", "",
              "Calibration-development embargo: {} observation days; {} flare windows/known-episode members excluded. Input/label intervals touching fitting or held-out instrument-calibration days cannot enter classifier training, validation or testing. See calibration_embargo.csv.".format(len(calibration_dates), int(splits.calibration_embargo.sum())), ""]
    for model in MODELS:
        data = metrics[metrics.model.eq(model)].dropna(subset=["TSS"]).sort_values(["TSS", "ROC_AUC"], ascending=False)
        lines.append("{}: {}".format(DISPLAY_MODELS[model], "; ".join("{} TSS={:.3f}, HSS={:.3f}, F1={:.3f}, ROC AUC={:.3f}".format(r.feature, r.TSS, r.HSS, r.F1, r.ROC_AUC) for r in data.head(3).itertuples()) if len(data) else "No defined test TSS; inspect class counts and status columns."))
        lines.append("")
    test_rows = splits[splits.split.eq("test")]
    base = float(test_rows.event_type.mean()) if len(test_rows) else np.nan
    lines += ["The test SEP prevalence is {:.4f}; this is the reference average precision for an uninformative ranking. ROC AUC=0.5, TSS=0 and HSS=0 are useful no-skill references. Accuracy alone is misleading in this imbalanced cohort.".format(base), "", "The benchmark checks for cross-class temporal overlap before model fitting. Same-class windows within each partition can still overlap. These retrospective scores assess discrimination on the stated held-out cohort. Consult the provenance counts and measurement-validation reports for instrument calibration and source-era coverage.", "", "Response bridges fitted using 2024/2025 observations may postdate earlier test inputs. Embargoing calibration observations prevents those observations from entering the classifier evaluation. Operational forecasting would need to use only information available at each forecast date.", "", "Random Forest uses balanced class weights. The reference sktime Time Series Forest uses its documented unweighted tree fitting. Test prevalence is preserved. Undefined metrics are left unavailable with reasons in the CSV files."]
    (outdir / "evaluation_summary.md").write_text("\n".join(lines) + "\n")


def run_benchmarks(dataset_dir, outdir, *, train_end="2014-12-31", validation_end="2019-12-31",
                   n_estimators=200, n_jobs=1, random_state=42, features=FEATURES, purge_hours=30):
    """Train and evaluate two separate univariate models per selected feature."""
    root, outdir = Path(dataset_dir), Path(outdir)
    features = tuple(features)
    if not features or len(set(features)) != len(features) or any(f not in FEATURES for f in features):
        raise ValueError("Choose distinct original features")
    if n_estimators < 1 or n_jobs == 0:
        raise ValueError("Estimator count must be positive and n_jobs cannot be zero")
    manifest, metadata, ids, canonical, paths, eras, hashes = _load_inputs(root, features)
    from .temporal_overlap import assert_no_temporal_overlap
    assert_no_temporal_overlap(metadata)
    calibration_dates, bank_hash = _calibration_observation_days(root, manifest)
    splits = chronological_splits(metadata, ids, canonical, train_end=train_end, validation_end=validation_end, purge_hours=purge_hours, calibration_dates=calibration_dates)
    outdir.mkdir(parents=True, exist_ok=True)
    _csv(outdir / "splits.csv", splits)
    _csv(outdir / "calibration_embargo.csv", splits[splits.calibration_embargo].copy())
    counts, exclusions = _partition_counts(splits), _exclusion_counts(splits)
    _csv(outdir / "split_counts.csv", counts)
    _csv(outdir / "split_exclusion_counts.csv", exclusions)
    split_indices = {name: np.flatnonzero(splits.split.eq(name).to_numpy()) for name in PARTITIONS}
    labels = metadata.event_type.to_numpy(dtype=int)
    if any(not len(split_indices[name]) for name in split_indices):
        raise ValueError("Every chronological partition must contain samples; inspect splits.csv and choose appropriate date boundaries")
    for name, idx in split_indices.items():
        if len(np.unique(labels[idx])) != 2:
            raise ValueError("{} requires both SEP and NSEP after timing/episode/calibration embargoes; inspect splits.csv. No model is fitted and no score or threshold is fabricated. Any revised split protocol must be specified before evaluating test predictions.".format(name))
    factories = _estimators(n_estimators, n_jobs, random_state)
    config = {"version": VERSION, "features": features, "train_end": train_end, "validation_end": validation_end,
              "purge_hours": purge_hours, "n_estimators": n_estimators, "n_jobs": n_jobs, "random_state": random_state,
              "input_manifest_sha256": _sha(root / "dataset_manifest.json"), "inputs": hashes,
              "calibration_bank_sha256": bank_hash, "calibration_embargo_dates": calibration_dates,
              "split_sha256": _sha(outdir / "splits.csv"), "package_versions": _versions()}
    config_signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    scores, fixed_scores, threshold_rows, predictions, cases = [], [], [], {}, []
    era_counts = []
    for feature in features:
        cols = ["{}_{}".format(feature, i) for i in range(288, 0, -1)]
        x = pd.read_csv(paths[feature], usecols=cols, float_precision="round_trip").to_numpy(dtype=np.float32)
        if not np.isfinite(x).all() or np.isin(x, [-9999., -99999.]).any():
            raise ValueError("Benchmark does not fill missing values; invalid input in {}".format(feature))
        for name, idx in split_indices.items():
            for era, count in eras[feature].iloc[idx].value_counts().items():
                era_counts.append({"feature": feature, "split": name, "source_era": era, "samples": int(count)})
        for model in MODELS:
            print("[BENCHMARK] {} / {}".format(feature, model), flush=True)
            case_dir = outdir / "cases" / feature / model
            case_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = case_dir / "case_manifest.json"
            saved = json.loads(checkpoint.read_text()) if checkpoint.is_file() else {}
            reusable = saved.get("config_signature") == config_signature and all((case_dir / (name + "_predictions.csv")).is_file() and _sha(case_dir / (name + "_predictions.csv")) == saved.get("predictions_sha256", {}).get(name) for name in ("validation", "test"))
            if reusable:
                tables = {name: pd.read_csv(case_dir / (name + "_predictions.csv"), float_precision="round_trip") for name in ("validation", "test")}
                print("[RESUME] Predictions verified", flush=True)
            else:
                estimator = factories[model]()
                estimator.fit(x[split_indices["train"]], labels[split_indices["train"]])
                tables = {}
                for name in ("validation", "test"):
                    idx = split_indices[name]
                    table = splits.iloc[idx].drop(columns=["exclusion_reason"]).copy()
                    table["observed_label"] = labels[idx]
                    table["probability"] = _positive_probability(estimator, x[idx])
                    table["source_era"] = eras[feature].iloc[idx].to_numpy()
                    tables[name] = table
                    _csv(case_dir / (name + "_predictions.csv"), table)
                saved = {"feature": feature, "model": model, "config_signature": config_signature,
                         "predictions_sha256": {name: _sha(case_dir / (name + "_predictions.csv")) for name in ("validation", "test")}}
                _json(checkpoint, saved)
                del estimator
            for name, table in tables.items():
                expected = labels[split_indices[name]]
                if not np.array_equal(table.observed_label.to_numpy(), expected):
                    raise ValueError("Cached prediction labels are inconsistent with the immutable split")
            threshold = select_threshold(tables["validation"].observed_label, tables["validation"].probability)
            threshold_rows.append({"feature": feature, "model": model, **threshold})
            test = tables["test"]
            score = classification_metrics(test.observed_label, test.probability, threshold["threshold"])
            fixed = classification_metrics(test.observed_label, test.probability, .5)
            train_prevalence = labels[split_indices["train"]].mean()
            baseline_brier = float(np.mean((test.observed_label.to_numpy() - train_prevalence) ** 2))
            score["BrierSkillAgainstTrainingPrior"] = 1 - score["BrierScore"] / baseline_brier if baseline_brier else np.nan
            scores.append({"feature": feature, "model": model, **score})
            fixed_scores.append({"feature": feature, "model": model, **fixed})
            predictions[(feature, model)] = test
            cases.append({**saved, "threshold_selection": threshold})
        del x
    metrics = pd.DataFrame(scores)
    _csv(outdir / "univariate_test_metrics.csv", metrics)
    _csv(outdir / "univariate_test_metrics_fixed_0_5.csv", pd.DataFrame(fixed_scores))
    _csv(outdir / "validation_thresholds.csv", pd.DataFrame(threshold_rows))
    _csv(outdir / "source_era_counts.csv", pd.DataFrame(era_counts))
    plot_benchmark_results(metrics, predictions, features, outdir)
    _write_analysis(metrics, splits, counts, exclusions, outdir, calibration_dates)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "workflow": VERSION, "config": config,
              "config_signature": config_signature, "input_workflow": manifest["workflow"], "case_count": len(scores),
              "split_counts": {name: {"NSEP": int((labels[idx] == 0).sum()), "SEP": int((labels[idx] == 1).sum())} for name, idx in split_indices.items()},
              "cohort_counts": counts.to_dict("records"), "split_exclusion_counts": exclusions.to_dict("records"),
              "label_policy": LABEL_POLICY,
              "purged_samples": int(splits.split.eq("purged").sum()),
              "calibration_embargo_samples": int(splits.calibration_embargo.sum()),
              "calibration_embargo_dates": calibration_dates, "cases": cases,
              "input_definition": "One original physical feature's 288 chronological pre-flare time samples; metadata and other feature files never enter a model's predictors.",
              "threshold_policy": "Validation-only maximum TSS; fixed .5 test results supplied separately. No test-set optimization.",
              "model_policies": {"RandomForest": "scikit-learn RF on 288 lag values; balanced class weights", "TimeSeriesForest": "sktime TimeSeriesForestClassifier; random intervals with mean, standard deviation and slope; unweighted fitting"},
              "leakage_controls": "Canonical flare uniqueness and cross-class temporal overlap checked before fitting. Chronological train/validation/test split with a 30-hour boundary buffer. Shared SEP episodes and known episodes straddling boundaries are purged. Calibration observation days are embargoed against input windows, label horizons and known positive episodes. All cases share the same split.",
              "evaluation_scope": "Retrospectively calibrated-data discrimination; not a strict as-of-time operational forecast replay. Bridge response coefficients may use 2024/2025 instrument observations after earlier classifier test inputs.",
              "limitations": ["Model discrimination is not a validation of physical calibration", "This retrospective reprocessed-data analysis does not recreate the calibration information available at each historical forecast date", "Correlated and overlapping same-class windows within a partition remain", "Performance applies to these held-out years and source eras; consult source_era_counts.csv", "Unknown SEP end timestamps can limit episode grouping; explicit timing buffers and shared known identities still apply", "No trained models are retained by default; checksum-protected predictions support bounded-storage resume"]}
    _json(outdir / "benchmark_manifest.json", result)
    print("[DONE] {} univariate model cases: {}".format(len(scores), outdir), flush=True)
    return result
