"""Additional dataset-paper diagnostics; one flare window is one observation.

This module supplements, and never replaces, the original 288-point PCA and
profile workflow. Measurements and labels are read-only. No values are imputed
during analysis. Correlations are descriptive, without significance claims.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


FEATURES = ("Vx", "flow_speed", "proton_density", "T", "F", "Xs", "Xl", "P4", "P5", "P6", "USFLUXL", "R_VALUE")
LOG_FEATURES = frozenset(("Xs", "Xl", "P4", "P5", "P6", "USFLUXL"))
FEATURE_UNITS = {"Vx": "km/s", "flow_speed": "km/s", "proton_density": "cm^-3", "T": "K", "F": "nT",
                 "Xs": "W/m^2", "Xl": "W/m^2", "P4": "cm^-2 s^-1 sr^-1 MeV^-1", "P5": "cm^-2 s^-1 sr^-1 MeV^-1",
                 "P6": "cm^-2 s^-1 sr^-1 MeV^-1", "USFLUXL": "Mx", "R_VALUE": "catalog logarithmic quantity"}
STATUS_CODES = "OIEMSGUDH"
STEPS = 288


def _variable(values):
    return bool(len(values) and np.ptp(values) > 1e-12 * max(float(np.max(np.abs(values))), np.finfo(float).tiny))


def transform_summaries(frame: pd.DataFrame, mode="physical_log") -> pd.DataFrame:
    """Log10 positive window summaries for skewed flux channels only.

    R_VALUE is already logarithmic and Vx is signed: neither is logged.
    Nonpositive summaries become unavailable only in the log variant.
    """
    if mode not in ("raw", "physical_log"):
        raise ValueError("transform must be raw or physical_log")
    result = frame.copy()
    if mode == "physical_log":
        for feature in result.columns:
            if feature in LOG_FEATURES:
                result[feature] = np.log10(result[feature].where(result[feature] > 0))
    return result


def pairwise_correlations(frame, min_pairs=5) -> pd.DataFrame:
    """Pairwise-complete Pearson/Spearman; a sample row is a flare window."""
    if min_pairs < 3:
        raise ValueError("min_pairs must be at least three")
    rows = []
    columns = list(frame.columns)
    for i, a in enumerate(columns):
        for b in columns[i:]:
            x, y = frame[a].to_numpy(dtype=float), frame[b].to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            n = int(valid.sum())
            variable = n >= min_pairs and _variable(x) and _variable(y)
            if variable:
                pearson = float(pearsonr(x, y)[0])
                spearman = float(spearmanr(x, y)[0])
            else:
                pearson = spearman = np.nan
            rows.append({"feature_a": a, "feature_b": b, "n_pairwise": n,
                         "pearson_r": pearson, "spearman_rho": spearman,
                         "status": "ok" if variable else "too_few_or_constant"})
    return pd.DataFrame(rows)


def _matrix(table, features, column):
    result = pd.DataFrame(np.nan, index=features, columns=features)
    for row in table.itertuples():
        result.loc[row.feature_a, row.feature_b] = getattr(row, column)
        result.loc[row.feature_b, row.feature_a] = getattr(row, column)
    return result


def _reduce(x, min_points=1):
    valid = np.isfinite(x)
    n = valid.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(x, axis=1)
        median = np.nanmedian(x, axis=1)
        std = np.nanstd(x, axis=1, ddof=1)
        quartiles = np.nanpercentile(x, [25, 75], axis=1)
    result = {"mean": mean, "median": median, "std": std, "iqr": quartiles[1]-quartiles[0], "valid_points": n}
    for key in ("mean", "median", "std", "iqr"):
        result[key][n < min_points] = np.nan
    return result


def load_feature(path: Path, quality_path: Path, feature: str, native_min_points=3):
    frame = pd.read_csv(path, low_memory=False, float_precision="round_trip")
    columns = ["{}_{}".format(feature, i) for i in range(STEPS, 0, -1)]
    if any(column not in frame for column in columns):
        raise ValueError("Missing 288-point block for {}".format(feature))
    metadata = frame.drop(columns=columns)
    if not {"FlrOnset", "event_type"}.issubset(metadata):
        raise ValueError("Missing FlrOnset/event_type metadata")
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    values[(values == -9999) | (~np.isfinite(values)) | (np.abs(values) >= 1e29)] = np.nan
    quality = pd.DataFrame(index=range(len(frame)))
    if quality_path.exists():
        source = pd.read_csv(quality_path, dtype={"point_status": str, "causality_mask": str}, low_memory=False)
        if "row_index" not in source or source.row_index.duplicated().any():
            raise ValueError("Quality sidecar requires unique row_index: {}".format(quality_path))
        if not source.row_index.isin(range(len(frame))).all():
            raise ValueError("Quality row_index is outside feature dataset")
        quality = source.set_index("row_index").reindex(range(len(frame)))
    status = quality["point_status"].fillna("U"*STEPS) if "point_status" in quality else pd.Series("U"*STEPS, index=quality.index)
    if not status.str.fullmatch("[{}]{{{}}}".format(STATUS_CODES, STEPS)).all():
        raise ValueError("Invalid point_status mask for {}".format(feature))
    masks = np.array([list(value) for value in status], dtype="U1")
    # Structural absence and remaining gaps are unavailable even if an invalid
    # upstream file accidentally contains finite placeholders in those cells.
    values[np.isin(masks, ["S", "G"])] = np.nan
    native_values = np.where(masks == "O", values, np.nan)
    filled = _reduce(values)
    native = _reduce(native_values, min_points=native_min_points)
    counts = {code: (masks == code).sum(axis=1) for code in STATUS_CODES}
    eras = quality["source_era"].fillna("unknown") if "source_era" in quality else pd.Series("unknown", index=quality.index)
    summary = pd.DataFrame({"dataset_row_index": range(len(frame)), "FlrOnset": metadata["FlrOnset"],
                            "event_type": metadata["event_type"]})
    for key in filled:
        summary["filled_"+key] = filled[key]
        summary["native_"+key] = native[key]
    for code in STATUS_CODES:
        summary["points_"+code] = counts[code]
    summary["source_era"] = eras.to_numpy()
    for col in ("sample_id", "origin", "source_method", "source_row_index", "cleaning_missing_points_before",
                "cleaning_interpolated_points", "cleaning_edge_filled_points", "source_fill_points"):
        if col in quality:
            summary[col] = quality[col].to_numpy()
    return metadata, values, native_values, summary, counts


def _label_association(values, labels, min_pairs):
    """Point-biserial association per bin; no p-values or independence claims."""
    rows = []
    for j in range(STEPS):
        valid = np.isfinite(values[:, j]) & np.isfinite(labels)
        x, y = values[valid, j], labels[valid]
        n0, n1 = int((y == 0).sum()), int((y == 1).sum())
        ok = len(x) >= min_pairs and min(n0, n1) >= 3 and _variable(x)
        r = float(pearsonr(x, y)[0]) if ok else np.nan
        rows.append({"step_before_flare": STEPS-j, "hours_before_aligned_flare": (STEPS-j)/12,
                     "n_pairwise": len(x), "n_nsep": n0, "n_sep": n1, "pearson_label_r": r})
    return pd.DataFrame(rows)


def _plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, base, plt, artifacts):
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        path = base.with_suffix(suffix)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        artifacts.append(str(path))
    plt.close(fig)


def _heatmap(ax, matrix, title, fig):
    image = ax.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm", aspect="equal")
    ax.set(xticks=np.arange(len(matrix)), yticks=np.arange(len(matrix)), xticklabels=matrix.columns,
           yticklabels=matrix.index, title=title)
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Descriptive correlation")


def _multi_heatmaps(tables, features, outdir, name, plt, artifacts):
    fig, axes = plt.subplots(1, len(tables), figsize=(6*len(tables), 6), squeeze=False)
    for ax, (title, table, statistic) in zip(axes[0], tables):
        _heatmap(ax, _matrix(table, features, statistic), title, fig)
    fig.text(0.5, 0.01, "One row = one flare window. Flux-log: positive Xs/Xl/P4/P5/P6/USFLUXL summaries use log10; R_VALUE is unchanged.",
             ha="center", fontsize=8)
    _save(fig, outdir / name, plt, artifacts)


def _distribution_panels(values_by_feature, labels, outdir, name, ylabel, plt, artifacts, log_scale=False):
    ncols = 4
    nrows = int(np.ceil(len(values_by_feature)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.2*nrows), squeeze=False)
    rows = []
    for ax, (feature, values) in zip(axes.ravel(), values_by_feature.items()):
        groups, ns = [], []
        for label, group in ((0, "NSEP"), (1, "SEP")):
            valid = np.isfinite(values) & (labels == label)
            if log_scale and feature in LOG_FEATURES:
                valid &= values > 0
            data = values[valid]
            groups.append(data)
            ns.append(len(data))
            rows.append({"feature": feature, "event_type": label, "n": len(data),
                         "median": float(np.median(data)) if len(data) else np.nan,
                         "q25": float(np.percentile(data, 25)) if len(data) else np.nan,
                         "q75": float(np.percentile(data, 75)) if len(data) else np.nan})
        # Empty groups are shown as unavailable rather than a zero value.
        ax.boxplot(groups, positions=[1, 2], showfliers=False, widths=0.55)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["NSEP n={}".format(ns[0]), "SEP n={}".format(ns[1])])
        if log_scale and feature in LOG_FEATURES and any(ns):
            ax.set_yscale("log")
        suffix = " (already log)" if feature == "R_VALUE" else " (log axis; positive only)" if log_scale and feature in LOG_FEATURES else ""
        ax.set(title=feature + suffix, ylabel=ylabel.replace("native units", FEATURE_UNITS[feature]))
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.2)
    for ax in axes.ravel()[len(values_by_feature):]:
        ax.axis("off")
    _save(fig, outdir / name, plt, artifacts)
    pd.DataFrame(rows).to_csv(outdir / (name + "_statistics.csv"), index=False)


def run_paper_figures(dataset_dir, outdir, features=FEATURES, min_pairs=5, native_min_points=3):
    """Add publication diagnostics without modifying datasets or old figures.

    Default requires all twelve feature files. Passing a feature subset enables
    focused inspection. Each data row, never each five-minute cell, is a sample
    for feature correlations. Native-only sensitivity uses O-coded points only.
    """
    dataset_dir, outdir = Path(dataset_dir), Path(outdir)
    input_manifest_path = dataset_dir / "dataset_manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text()) if input_manifest_path.exists() else {}
    cleaned = input_manifest.get("workflow") == "cleaned_twelve_feature_release"
    features = tuple(features)
    if not features or len(set(features)) != len(features) or any(f not in FEATURES for f in features):
        raise ValueError("Choose distinct supported paper features")
    if min_pairs < 3 or native_min_points < 1 or native_min_points > STEPS:
        raise ValueError("Invalid minimum pair/native observation count")
    paths = {}
    for feature in features:
        found = sorted(dataset_dir.glob("*_with_{}_filled.csv".format(feature)))
        if len(found) != 1:
            raise ValueError("Expected exactly one filled dataset for {}".format(feature))
        paths[feature] = found[0]
    outdir.mkdir(parents=True, exist_ok=True)
    plt, artifacts = _plot(), []
    time_tables, coverage_rows, method_rows, era_series = [], [], [], {}
    reference = None
    matrices = {(policy, aggregate): pd.DataFrame() for policy in ("filled", "native_only") for aggregate in ("mean", "median")}
    iqr_values = {}
    for feature in features:
        print("[PAPER FIGURES] {}".format(feature), flush=True)
        metadata, values, native_values, summary, counts = load_feature(paths[feature], dataset_dir/"quality"/(feature+"_quality.csv"), feature, native_min_points)
        if reference is None:
            reference = metadata.copy()
            labels = pd.to_numeric(metadata["event_type"], errors="coerce").to_numpy(dtype=float)
            if not np.isin(labels, [0, 1]).all():
                raise ValueError("event_type must be 0 (NSEP) or 1 (SEP)")
            dates = pd.to_datetime(metadata["FlrOnset"], utc=True, format="mixed", errors="coerce")
            if dates.isna().any():
                raise ValueError("Invalid FlrOnset timestamps")
        elif not metadata.equals(reference):
            raise ValueError("All feature files must share metadata, labels and sample order")
        summary.to_csv(outdir/(feature+"_window_summary.csv"), index=False)
        era_series[feature] = summary["source_era"]
        for policy, prefix in (("filled", "filled"), ("native_only", "native")):
            for aggregate in ("mean", "median"):
                matrices[(policy, aggregate)][feature] = summary[prefix+"_"+aggregate].to_numpy()
        iqr_values[feature] = summary["filled_iqr"].to_numpy()
        for year in sorted(dates.dt.year.unique()):
            selected = dates.dt.year.eq(year).to_numpy()
            count = int(selected.sum())
            total = count*STEPS
            count_codes = {code: int(counts[code][selected].sum()) for code in STATUS_CODES}
            known = total-count_codes["U"]
            available_pre_fill = sum(count_codes[code] for code in "ODH")
            coverage_rows.append({"feature": feature, "year": int(year), "sample_windows": count,
                                  "known_provenance_points": known, "unknown_historical_points": count_codes["U"],
                                  "fraction_missing_before_gap_fill_known_only": (known-available_pre_fill)/known if known else np.nan,
                                  "fraction_missing_after_filling": float((~np.isfinite(values[selected])).mean()),
                                  "native_fraction_all_points": count_codes["O"]/total,
                                  "derived_fraction_all_points": count_codes["D"]/total,
                                  "held_fraction_all_points": count_codes["H"]/total})
            if "cleaning_missing_points_before" in summary:
                coverage_rows[-1]["fraction_missing_before_offline_cleaning_retained_rows"] = float(pd.to_numeric(summary.loc[selected, "cleaning_missing_points_before"], errors="raise").sum()/total)
                coverage_rows[-1]["fraction_filled_in_offline_cleaning"] = float((pd.to_numeric(summary.loc[selected, "cleaning_interpolated_points"], errors="raise") + pd.to_numeric(summary.loc[selected, "cleaning_edge_filled_points"], errors="raise")).sum()/total)
            for code in STATUS_CODES:
                method_rows.append({"feature": feature, "year": int(year), "status_code": code,
                                    "points": count_codes[code], "total_points": total, "fraction": count_codes[code]/total})
        for policy, source in (("filled", values), ("native_only", native_values)):
            transformed = source.copy()
            if feature in LOG_FEATURES:
                transformed[transformed <= 0] = np.nan
                transformed = np.log10(transformed)
            table = _label_association(transformed, labels, min_pairs)
            table["feature"], table["policy"] = feature, policy
            time_tables.append(table)
        del values, native_values, source, transformed
    coverage, methods = pd.DataFrame(coverage_rows), pd.DataFrame(method_rows)
    coverage.to_csv(outdir/"feature_year_coverage.csv", index=False)
    methods.to_csv(outdir/"point_methods_by_feature_year.csv", index=False)
    time_associations = pd.concat(time_tables, ignore_index=True)
    time_associations.to_csv(outdir/"time_resolved_label_association.csv", index=False)
    periods = np.select([dates.dt.year.le(2013), dates.dt.year.lt(2020)], ["1998_2013_published", "2014_2019_extension"], default="2020_onward_extension")
    groups = {"all": np.ones(len(reference), dtype=bool), "SEP": labels == 1, "NSEP": labels == 0}
    groups.update({str(period): periods == period for period in np.unique(periods)})
    correlations, cache, transform_rows = [], {}, []
    for (policy, aggregate), matrix in matrices.items():
        for transform in ("raw", "physical_log"):
            data = transform_summaries(matrix, transform)
            for feature in features:
                transform_rows.append({"feature": feature, "policy": policy, "aggregate": aggregate, "transform": transform,
                                       "finite_summaries_before": int(np.isfinite(matrix[feature]).sum()),
                                       "nonpositive_excluded": int((np.isfinite(matrix[feature]) & matrix[feature].le(0)).sum()) if transform == "physical_log" and feature in LOG_FEATURES else 0,
                                       "finite_summaries_after": int(np.isfinite(data[feature]).sum())})
            for group, selection in groups.items():
                result = pairwise_correlations(data.loc[selection], min_pairs)
                result["policy"], result["aggregate"], result["transform"], result["group"] = policy, aggregate, transform, group
                correlations.append(result)
                cache[(policy, aggregate, transform, group)] = result
    all_correlations = pd.concat(correlations, ignore_index=True)
    all_correlations.to_csv(outdir/"feature_correlations.csv", index=False)
    pd.DataFrame(transform_rows).to_csv(outdir/"transform_sample_counts.csv", index=False)
    # Actual feature-specific instrument eras, not calendar labels alone.
    era_rows = []
    for policy in ("filled", "native_only"):
        data = transform_summaries(matrices[(policy, "mean")])
        for i, feature_a in enumerate(features):
            for feature_b in features[i+1:]:
                era_a, era_b = era_series[feature_a].astype(str), era_series[feature_b].astype(str)
                pair_eras = pd.DataFrame({"a": era_a, "b": era_b}).drop_duplicates()
                for era in pair_eras.itertuples(index=False):
                    for group in ("all", "SEP", "NSEP"):
                        selected = era_a.eq(era.a).to_numpy() & era_b.eq(era.b).to_numpy() & groups[group]
                        if not selected.any():
                            continue
                        result = pairwise_correlations(data.loc[selected, [feature_a, feature_b]], min_pairs)
                        row = result[result.feature_a.ne(result.feature_b)].iloc[0].to_dict()
                        row.update({"era_a": era.a, "era_b": era.b, "group": group, "policy": policy, "aggregate": "mean", "transform": "physical_log"})
                        era_rows.append(row)
    pd.DataFrame(era_rows, columns=["feature_a", "feature_b", "n_pairwise", "pearson_r", "spearman_rho", "status",
                                   "era_a", "era_b", "group", "policy", "aggregate", "transform"]).to_csv(outdir/"correlations_by_instrument_era.csv", index=False)
    get = lambda policy="filled", transform="physical_log", group="all": cache[(policy, "mean", transform, group)]
    _multi_heatmaps([("Window means: flux-log Pearson", get(), "pearson_r"), ("Window means: flux-log Spearman", get(), "spearman_rho")], features, outdir, "correlations_primary", plt, artifacts)
    _multi_heatmaps([("Filled values: Spearman", get(), "spearman_rho"), ("Native O-coded points only: Spearman", get("native_only"), "spearman_rho")], features, outdir, "correlations_native_vs_filled", plt, artifacts)
    _multi_heatmaps([("SEP window means: Spearman", get(group="SEP"), "spearman_rho"), ("NSEP window means: Spearman", get(group="NSEP"), "spearman_rho")], features, outdir, "correlations_sep_vs_nsep", plt, artifacts)
    _multi_heatmaps([("Raw window means: Pearson", get(transform="raw"), "pearson_r"), ("Flux-log window means: Pearson", get(), "pearson_r")], features, outdir, "correlations_raw_vs_flux_log", plt, artifacts)
    era_plots = [(str(period), get(group=str(period)), "spearman_rho") for period in np.unique(periods)]
    _multi_heatmaps(era_plots, features, outdir, "correlations_by_calendar_cohort", plt, artifacts)
    _distribution_panels({f: matrices[("filled", "mean")][f].to_numpy() for f in features}, labels, outdir, "window_mean_distributions", "Window mean (native units)", plt, artifacts, log_scale=True)
    _distribution_panels(iqr_values, labels, outdir, "window_variability_distributions", "Within-window IQR (native units)", plt, artifacts, log_scale=True)
    sensitivity_rows = []
    for feature in features:
        filled = matrices[("filled", "mean")][feature].to_numpy()
        native = matrices[("native_only", "mean")][feature].to_numpy()
        finite = np.isfinite(filled)
        spread = np.percentile(filled[finite], 75)-np.percentile(filled[finite], 25) if finite.any() else np.nan
        for label in (0, 1):
            valid = np.isfinite(filled) & np.isfinite(native) & (labels == label)
            d = (filled[valid]-native[valid])/spread if spread > 0 else np.array([])
            sensitivity_rows.append({"feature": feature, "event_type": label, "n_paired": int(valid.sum()),
                                     "median_difference_over_filled_IQR": float(np.median(d)) if len(d) else np.nan,
                                     "q25": float(np.percentile(d, 25)) if len(d) else np.nan, "q75": float(np.percentile(d, 75)) if len(d) else np.nan})
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(outdir/"native_vs_filled_sensitivity.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, offset, color in ((0, -0.12, "#75899c"), (1, 0.12, "#d85b56")):
        subset = sensitivity[sensitivity.event_type.eq(label)].set_index("feature").reindex(features)
        y = subset.median_difference_over_filled_IQR.to_numpy()
        valid = np.isfinite(y) & np.isfinite(subset.q25.to_numpy()) & np.isfinite(subset.q75.to_numpy())
        ax.errorbar(np.arange(len(features))[valid]+offset, y[valid],
                    yerr=np.vstack([y[valid]-subset.q25.to_numpy()[valid], subset.q75.to_numpy()[valid]-y[valid]]),
                    fmt="o", color=color, label="NSEP" if label == 0 else "SEP")
    ax.axhline(0, color="0.5", lw=1)
    ax.set(xticks=np.arange(len(features)), xticklabels=features, ylabel="(Filled mean − native-only mean) / filled IQR", title="Paired sensitivity to filling and derived/held values")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False)
    _save(fig, outdir/"native_vs_filled_sensitivity", plt, artifacts)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    before_column = "fraction_missing_before_offline_cleaning_retained_rows" if cleaned and "fraction_missing_before_offline_cleaning_retained_rows" in coverage else "fraction_missing_before_gap_fill_known_only"
    before_title = "Before offline filling: retained cohort" if before_column.startswith("fraction_missing_before_offline") else "Before gap fill: known provenance only"
    for ax, column, title in ((axes[0], before_column, before_title), (axes[1], "fraction_missing_after_filling", "After filling: all saved rows")):
        pivot = coverage.pivot(index="feature", columns="year", values=column).reindex(features)
        im = ax.imshow(pivot, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
        ax.set(yticks=np.arange(len(features)), yticklabels=features, xticks=np.arange(len(pivot.columns)), xticklabels=pivot.columns.astype(str), title=title)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        fig.colorbar(im, ax=ax, label="Fraction missing; blank = unavailable audit")
    _save(fig, outdir/"missingness_before_after_by_year", plt, artifacts)
    aggregated = methods.groupby(["feature", "status_code"]).points.sum().unstack("status_code").reindex(features).reindex(columns=list(STATUS_CODES), fill_value=0)
    fractions = aggregated.div(aggregated.sum(axis=1), axis=0)
    fractions.to_csv(outdir/"point_method_fractions.csv")
    fig, ax = plt.subplots(figsize=(12, 6))
    fractions.plot.barh(stacked=True, ax=ax, colormap="tab10")
    ax.set(xlim=(0, 1), xlabel="Fraction of saved feature points", title="Observed, derived, held, filled and unavailable provenance")
    ax.legend(title="Point-status code (see methods manifest)", bbox_to_anchor=(1.01, 1), loc="upper left")
    _save(fig, outdir/"point_method_fractions", plt, artifacts)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, column, title in zip(axes, ("native_fraction_all_points", "derived_fraction_all_points", "held_fraction_all_points"), ("Native O-coded", "SGPS-derived D-coded", "Causal held H-coded")):
        pivot = coverage.pivot(index="feature", columns="year", values=column).reindex(features)
        im = ax.imshow(pivot, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax.set(yticks=np.arange(len(features)), yticklabels=features, xticks=np.arange(len(pivot.columns)), xticklabels=pivot.columns.astype(str), title=title)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        fig.colorbar(im, ax=ax, label="Fraction of all points")
    _save(fig, outdir/"native_derived_held_fractions_by_year", plt, artifacts)
    nrows = int(np.ceil(len(features)/4))
    fig, axes = plt.subplots(nrows, 4, figsize=(15, nrows*3.2), squeeze=False)
    for ax, feature in zip(axes.ravel(), features):
        for policy, color in (("filled", "#bb5b52"), ("native_only", "#447c88")):
            subset = time_associations[time_associations.feature.eq(feature) & time_associations.policy.eq(policy)]
            ax.plot(subset.hours_before_aligned_flare, subset.pearson_label_r, label=policy, color=color, lw=1.3)
        ax.axhline(0, color="0.6", lw=0.7)
        ax.set(xlim=(24, 0), ylim=(-1, 1), title=feature + (" (log10 points)" if feature in LOG_FEATURES else ""), xlabel="Hours before aligned flare", ylabel="Label association r")
    axes.ravel()[0].legend(fontsize=8, frameon=False)
    for ax in axes.ravel()[len(features):]:
        ax.axis("off")
    _save(fig, outdir/"time_resolved_label_association", plt, artifacts)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "features": list(features), "sample_windows": len(reference),
        "input_workflow": input_manifest.get("workflow", "original_or_external"),
        "inputs": {f: str(paths[f]) for f in features}, "figure_files": artifacts,
        "correlation_unit": "one flare window per row; never pooled five-minute observations",
        "correlation_statistics": ["Pearson r", "Spearman rho", "pairwise complete n"], "minimum_pairs": min_pairs,
        "constant_guard": "Range <=1e-12 times maximum absolute value is treated as numerically constant.",
        "aggregates": ["window mean", "window median"],
        "spearman_transform_note": "A monotonic log10 preserves ranks for a fixed positive cohort; differences can arise when nonpositive summaries are excluded. See transform_sample_counts.csv.",
        "transforms": {f: "log10 of positive window summary; nonpositive excluded" if f in LOG_FEATURES else "unchanged (already logarithmic)" if f == "R_VALUE" else "unchanged" for f in features},
        "time_association_transform": "log10 positive instantaneous flux points; other features unchanged; distinct from log10 of window summaries",
        "native_only_policy": "only finite O-coded values; minimum {} per window; excludes I/E/M/S/G/U/D/H".format(native_min_points),
        "filled_policy": "finite saved values except structural S/remaining gaps G; includes flagged interpolation, median fill, derived SGPS, bounded causal holds and unknown-provenance historical values",
        "status_codes": {"O": "native observation", "I": "linear interpolation", "E": "edge fill", "M": "historical median fallback", "S": "structural absence", "G": "remaining gap", "U": "unknown historical provenance", "D": "SGPS-derived reference-energy point-flux estimate", "H": "bounded causal hold of a magnetic observation or proton estimate"},
        "prefill_quality": "known provenance only; O/D/H available before gap fill; U excluded from denominator; derived and held are separately reported and are not native observations",
        "instrument_era_analysis": "actual pairwise source_era strings in correlations_by_instrument_era.csv; calendar cohorts are additional descriptive partitions, not instrument-equivalence claims",
        "limitations": ["No significance tests or forecast-accuracy claims", "Overlapping windows, temporal dependence and repeated flare associations may violate independence", "Unknown historical U provenance cannot support observed-only analysis", "Derived proton bands and held magnetogram values are not independent native measurements", "Pairwise missingness changes the cohort between feature pairs; consult n_pairwise", "Class balance, calibration era and median imputation can influence correlations", "Time-resolved correlations are descriptive repeated-bin associations without independence or multiple-testing claims"],
    }
    if cleaned:
        manifest["cleaning_policy"] = "Retrospective linear interpolation within each pre-flare feature row; nearest valid edge extension; no cross-row/label filling. Shared cohort excludes a sample if any feature is entirely unavailable after source-fill masking."
        manifest["cohort_report"] = str(dataset_dir / "cohort_counts.csv")
        manifest["prefill_quality"] += "; the before-offline-cleaning panel describes recognized invalid cells in retained source rows, not historical acquisition missingness; exclusions are reported separately"
        manifest["limitations"].append("Global feature-availability filtering changes class/year composition; these figures describe the retained cohort and interpolated points are not native measurements.")
    manifest_path = outdir/"paper_figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    print("[DONE] Publication diagnostics: {}".format(outdir), flush=True)
    return {"outdir": str(outdir), "manifest": str(manifest_path), "feature_count": len(features), "figure_files": artifacts}
