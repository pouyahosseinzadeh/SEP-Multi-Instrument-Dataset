"""Reproduce the original ten-file MEMSEP figure workflow.

All class comparisons use the original flare rows, FlrOnset alignment, 288 time
samples, energy columns and peak thresholds. PCA is fitted to the complete,
column-standardized time series, never to window summary statistics. This
module reads datasets; it does not change measurements or impute new values.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


FEATURES = ("Vx", "flow_speed", "proton_density", "T", "F", "Xs", "Xl", "P4", "P5", "P6")
DISPLAY_NAMES = {
    "Vx": "Vx", "flow_speed": "FlowSpeed", "proton_density": "ProtonDensity",
    "T": "ProtonTemp", "F": "F", "Xs": "Xs", "Xl": "Xl", "P4": "P4", "P5": "P5", "P6": "P6",
}
UNITS = {"Vx": "km/s", "flow_speed": "km/s", "proton_density": "cm$^{-3}$",
         "T": "K", "F": "nT", "Xs": "W/m$^2$", "Xl": "W/m$^2$",
         "P4": "protons/(cm$^2$ s sr MeV)", "P5": "protons/(cm$^2$ s sr MeV)",
         "P6": "protons/(cm$^2$ s sr MeV)"}
# Verified against the original notebook AND its four exported summary tables.
BAND_SELECTION = {10: ("sep_peak_2", 10.0), 30: ("sep_peak_3", 5.0),
                  60: ("sep_peak_4", 5.0), 100: ("sep_peak_5", 5.0)}
NORMAL_95 = 1.959963984540054
STEP_MINUTES = 5
STEPS = 288


def timeseries_columns(feature: str) -> list:
    return ["{}_{}".format(feature, i) for i in range(STEPS, 0, -1)]


def band_masks(frame: pd.DataFrame, band: int) -> Tuple[np.ndarray, np.ndarray]:
    """Original selection: threshold-qualified SEP rows versus ALL NSEP rows."""
    if band not in BAND_SELECTION:
        raise ValueError("Supported energy bands are 10, 30, 60 and 100 MeV")
    peak_column, threshold = BAND_SELECTION[band]
    for col in ("event_type", peak_column):
        if col not in frame:
            raise ValueError("Missing required metadata column: {}".format(col))
    labels = pd.to_numeric(frame["event_type"], errors="coerce")
    if not labels.isin([0, 1]).all():
        raise ValueError("event_type must contain only 0 (NSEP) or 1 (SEP)")
    peak = pd.to_numeric(frame[peak_column], errors="coerce")
    return (labels.eq(1) & peak.ge(threshold)).to_numpy(), labels.eq(0).to_numpy()


def adjust_pvalues_bh(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q values, preserving unavailable tests as NaN."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p.ravel()))
    if not len(valid):
        return out
    values = p.ravel()[valid]
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    out.ravel()[valid[order]] = np.minimum(adjusted, 1.0)
    return out


def mean_ci(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Original normal-approximation CI; not a simultaneous confidence band."""
    values = np.asarray(values, dtype=float)
    n = np.isfinite(values).sum(axis=0)
    total = np.nansum(values, axis=0)
    mean = np.divide(total, n, out=np.full(values.shape[1], np.nan), where=n > 0)
    squared = np.nansum((values - mean) ** 2, axis=0)
    var = np.divide(squared, n - 1, out=np.full(values.shape[1], np.nan), where=n > 1)
    ci = NORMAL_95 * np.sqrt(np.divide(var, n, out=np.full_like(var, np.nan), where=n > 0))
    return mean, ci, n


def profile_statistics(sep: np.ndarray, nsep: np.ndarray, feature: str, band: int) -> pd.DataFrame:
    """One row per original time bin; raw and corrected tests are both saved."""
    ms, cs, ns = mean_ci(sep)
    mn, cn, nn = mean_ci(nsep)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = ttest_ind(sep, nsep, axis=0, equal_var=False, nan_policy="omit")
    p = np.asarray(result.pvalue, dtype=float)
    p[(ns < 2) | (nn < 2)] = np.nan
    q = adjust_pvalues_bh(p)
    return pd.DataFrame({
        "feature": feature, "energy_mev": band,
        "column": timeseries_columns(feature),
        "step_before_flare": np.arange(STEPS, 0, -1),
        "hours_before_aligned_flare": np.arange(STEPS, 0, -1) * STEP_MINUTES / 60.0,
        # Retained separately so published legacy CSVs can be compared exactly.
        "legacy_display_hours": np.linspace(24.0, 0.0, STEPS),
        "sep_n": ns, "nsep_n": nn,
        "sep_mean": ms, "sep_ci95_halfwidth": cs,
        "nsep_mean": mn, "nsep_ci95_halfwidth": cn,
        "difference": ms - mn, "welch_t": np.asarray(result.statistic),
        "p_value": p, "q_value_bh_within_feature_band": q,
        "raw_p_lt_0_05": p < 0.05, "bh_q_lt_0_05": q < 0.05,
    })


def full_window_pca(values: np.ndarray, n_components: int = 10) -> dict:
    """Exact PCA equivalent to StandardScaler + PCA on all 288 columns.

    Covariance eigendecomposition keeps this operation bounded in memory even
    for tens of thousands of flare rows. Eigenvector signs are deterministic.
    Any unfinished/nonfinite input rows are excluded explicitly, not imputed.
    """
    x = np.asarray(values, dtype=float)
    keep = np.isfinite(x).all(axis=1)
    clean = x[keep]
    if len(clean) < 2:
        raise ValueError("PCA needs at least two completely filled time series")
    center = clean.mean(axis=0)
    scale = clean.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    z = (clean - center) / scale
    covariance = z.T.dot(z) / (len(z) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = eigenvectors[:, order].T
    n = min(n_components, len(clean), x.shape[1])
    components = components[:n]
    for component in components:
        if component[np.argmax(np.abs(component))] < 0:
            component *= -1
    total = float(eigenvalues.sum())
    ratio = eigenvalues[:n] / total if total else np.zeros(n)
    return {"scores": z.dot(components.T), "components": components,
            "explained_variance_ratio": ratio, "retained_rows": keep,
            "mean": center, "scale": scale, "zero_total_variance": total == 0}


def _find_dataset(dataset_dir: Path, feature: str) -> Path:
    paths = sorted(dataset_dir.glob("*_with_{}_filled.csv".format(feature)))
    if len(paths) != 1:
        raise ValueError("Expected exactly one *_with_{}_filled.csv in {}; found {}".format(feature, dataset_dir, len(paths)))
    return paths[0]


def _read_feature(path: Path, feature: str) -> Tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path, low_memory=False)
    required = timeseries_columns(feature)
    missing = [c for c in required if c not in frame]
    if missing:
        raise ValueError("{} is missing {} expected time-series columns".format(path.name, len(missing)))
    # Numerical flags are preserved as unavailable rather than treated as physics.
    x = frame[required].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    x[(x == -9999.0) | (~np.isfinite(x)) | (np.abs(x) >= 1e29)] = np.nan
    if "FlrOnset" not in frame:
        raise ValueError("FlrOnset is required for compatibility figures")
    return frame, x


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, base: Path, plt, artifacts: list) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = base.with_suffix("." + ext)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        artifacts.append(str(path))
    plt.close(fig)


def _plot_profile(stats: pd.DataFrame, feature: str, band: int, outdir: Path, plt, artifacts: list) -> None:
    x = stats["hours_before_aligned_flare"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5))
    # Retain the legacy pointwise test, explicitly distinguish it from FDR.
    ax.fill_between(x, 0, 1, where=stats["raw_p_lt_0_05"].to_numpy(),
                    transform=ax.get_xaxis_transform(), color="0.90", zorder=0,
                    label="Raw Welch p < 0.05 (pointwise)")
    for group, color, label in (("sep", "#d94b4b", "SEP >{} MeV".format(band)), ("nsep", "#60758a", "NSEP")):
        mean = stats[group + "_mean"].to_numpy()
        ci = stats[group + "_ci95_halfwidth"].to_numpy()
        ax.plot(x, mean, color=color, lw=1.7, label="{} (n={})".format(label, int(stats[group + "_n"].max())))
        ax.fill_between(x, mean-ci, mean+ci, color=color, alpha=0.22)
    sig = stats["bh_q_lt_0_05"].to_numpy()
    ax.plot(x[sig], np.full(sig.sum(), 0.025), "|", color="purple", transform=ax.get_xaxis_transform(), label="BH q < 0.05")
    ax.set(xlim=(24, 0), xlabel="Hours before 5-minute aligned FlrOnset", ylabel="{} ({})".format(DISPLAY_NAMES[feature], UNITS[feature]),
           title="{}: mean ± 95% CI".format(DISPLAY_NAMES[feature]))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.text(0.5, 0.005, "Pointwise CIs; BH correction across 288 tests within this feature and energy band.", ha="center", fontsize=8)
    _save(fig, outdir / "curves_{}_gt{}".format(DISPLAY_NAMES[feature], band), plt, artifacts)


def _plot_examples(frame: pd.DataFrame, values: np.ndarray, sep_mask: np.ndarray, nsep_mask: np.ndarray,
                   feature: str, band: int, outdir: Path, plt, artifacts: list) -> None:
    """Show real individual windows as in the original sample-24h figures."""
    dates = pd.to_datetime(frame["FlrOnset"], utc=True, format="mixed")
    x = np.arange(STEPS, 0, -1)*STEP_MINUTES/60.0
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    rows = []
    for ax, mask, color, name in ((axes[0], sep_mask, "#d94b4b", "SEP"), (axes[1], nsep_mask, "#60758a", "NSEP")):
        candidates = np.flatnonzero(mask & np.isfinite(values).all(axis=1))
        extended = candidates[(dates.iloc[candidates].dt.year > 2013).to_numpy()]
        if not len(candidates):
            ax.text(0.5, 0.5, "No complete {} window".format(name), ha="center", transform=ax.transAxes)
            continue
        row = int(extended[0] if len(extended) else candidates[0])
        ax.plot(x, values[row], color=color, lw=1.4)
        ax.set(ylabel="{} ({})".format(DISPLAY_NAMES[feature], UNITS[feature]), title="{}: FlrOnset {} (dataset row {})".format(name, frame.iloc[row]["FlrOnset"], row))
        ax.grid(alpha=0.25)
        rows.append({"feature": feature, "energy_mev": band, "class": name, "dataset_row_index": row,
                     "FlrOnset": frame.iloc[row]["FlrOnset"], "selection": "first complete post-2013 row, or first complete baseline row if unavailable"})
    axes[-1].set(xlim=(24, 0), xlabel="Hours before 5-minute aligned FlrOnset")
    pd.DataFrame(rows).to_csv(outdir / (DISPLAY_NAMES[feature] + "_example_windows.csv"), index=False)
    _save(fig, outdir / "{}_sample24h_gt{}_MeV".format(DISPLAY_NAMES[feature], band), plt, artifacts)


def _plot_pca(pca: dict, labels: np.ndarray, metadata: pd.DataFrame, feature: str, band: int,
              outdir: Path, plt, artifacts: list, row_audit=None) -> None:
    name = DISPLAY_NAMES[feature]
    variance = pca["explained_variance_ratio"]
    pd.DataFrame({"pc": np.arange(1, len(variance)+1), "explained_variance_ratio": variance,
                  "cumulative": np.cumsum(variance)}).to_csv(outdir / (name + "_pca_explained_variance.csv"), index=False)
    loadings = pd.DataFrame(pca["components"], columns=timeseries_columns(feature))
    loadings.insert(0, "pc", np.arange(1, len(loadings)+1))
    loadings.to_csv(outdir / (name + "_pca_loadings.csv"), index=False)
    keep = pca["retained_rows"]
    scores = pd.DataFrame(pca["scores"], columns=["PC{}".format(i+1) for i in range(len(variance))])
    scores.insert(0, "FlrOnset", metadata.loc[keep, "FlrOnset"].to_numpy())
    scores.insert(0, "event_type", labels[keep])
    scores.insert(0, "dataset_row_index", metadata.index.to_numpy()[keep])
    if row_audit is not None:
        for column in ("sample_id", "origin", "source_era", "source_method"):
            if column in row_audit:
                scores[column] = row_audit[column].reindex(scores["dataset_row_index"]).to_numpy()
    scores.to_csv(outdir / (name + "_pca_scores.csv"), index=False)
    pd.DataFrame({"column": timeseries_columns(feature), "mean": pca["mean"], "scale": pca["scale"]}).to_csv(outdir / (name + "_pca_scaling.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(np.arange(1, len(variance)+1), variance, color="#53848e", label="Individual")
    ax.plot(np.arange(1, len(variance)+1), np.cumsum(variance), "o-", color="#b36b28", label="Cumulative")
    ax.set(xlabel="Principal component", ylabel="Explained variance ratio", ylim=(0, 1.03),
           title="{}: full 288-point PCA, >{} MeV selection".format(name, band))
    ax.legend(frameon=False)
    _save(fig, outdir / (name + "_pca_scree"), plt, artifacts)
    if len(variance) >= 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, color, title in ((0, "#7c8d9f", "NSEP"), (1, "#d94b4b", "SEP >{} MeV".format(band))):
            select = labels[keep] == label
            ax.scatter(pca["scores"][select, 0], pca["scores"][select, 1], s=13, alpha=0.6,
                       c=color, linewidths=0, rasterized=True, label="{} (n={})".format(title, int(select.sum())))
        ax.set(xlabel="PC1 ({:.2%})".format(variance[0]), ylabel="PC2 ({:.2%})".format(variance[1]),
               title="{}: PCA of complete time series".format(name))
        ax.legend(frameon=False)
        _save(fig, outdir / (name + "_pca_scatter_pc1_pc2"), plt, artifacts)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(pca["components"], aspect="auto", cmap="coolwarm", extent=[24, 0, len(variance)+0.5, 0.5])
    ax.set(xlabel="Hours before aligned FlrOnset", ylabel="Principal component", title=name + ": PCA component weights")
    fig.colorbar(im, ax=ax, label="Component weight (eigenvector coefficient)")
    _save(fig, outdir / (name + "_pca_loadings_heatmap"), plt, artifacts)


def _plot_difference_heatmap(effect_tables: Dict[str, pd.DataFrame], band: int, outdir: Path, plt, artifacts: list) -> None:
    names = list(effect_tables)
    d = np.vstack([effect_tables[n]["difference"].to_numpy() for n in names])
    mean = np.nanmean(d, axis=1, keepdims=True)
    std = np.nanstd(d, axis=1, ddof=1, keepdims=True)
    normalized = np.divide(d-mean, std, out=np.zeros_like(d), where=std > 0)
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(normalized, aspect="auto", origin="lower", cmap="coolwarm", extent=[24, 0, 0, len(names)], vmin=-2.5, vmax=2.5)
    x = np.arange(STEPS, 0, -1) * STEP_MINUTES / 60.0
    for i, name in enumerate(names):
        raw = effect_tables[name]["raw_p_lt_0_05"].to_numpy()
        corrected = effect_tables[name]["bh_q_lt_0_05"].to_numpy()
        ax.scatter(x[raw], np.full(raw.sum(), i+0.40), marker="|", s=14, color="black")
        ax.scatter(x[corrected], np.full(corrected.sum(), i+0.65), marker="|", s=14, color="purple")
    ax.set(yticks=np.arange(len(names))+0.5, yticklabels=[DISPLAY_NAMES[n] for n in names],
           xlabel="Hours before aligned FlrOnset",
           title="SEP >{} MeV − NSEP: mean differences normalized across time".format(band))
    fig.colorbar(im, ax=ax, label="Within-row z score of mean difference (not Cohen's d)")
    fig.text(0.5, 0.005, "Black: raw Welch p < 0.05; purple: BH q < 0.05 within each feature and band.", ha="center", fontsize=8)
    # Historical filename retained for comparison with the original project.
    _save(fig, outdir / "effectsize_heatmap_rownorm_gt{}".format(band), plt, artifacts)


def _plot_counts(frame: pd.DataFrame, outdir: Path, plt, artifacts: list) -> None:
    dates = pd.to_datetime(frame["FlrOnset"], utc=True, errors="coerce", format="mixed")
    if dates.isna().any():
        raise ValueError("Invalid FlrOnset values prevent yearly coverage validation")
    counts = pd.crosstab(dates.dt.year, pd.to_numeric(frame["event_type"]))
    counts = counts.reindex(columns=[0, 1], fill_value=0)
    counts.columns = ["NSEP", "SEP"]
    counts.index.name = "flare_year"
    counts.to_csv(outdir / "label_counts_by_year.csv")
    fig, ax = plt.subplots(figsize=(13, 4.5))
    counts.plot.bar(stacked=True, ax=ax, color=["#7c8d9f", "#d94b4b"], width=0.8)
    ax.set(xlabel="FlrOnset year", ylabel="Flare samples", title="SEP and NSEP flare samples by year")
    ax.legend(title=None, frameon=False)
    _save(fig, outdir / "label_counts_by_year", plt, artifacts)


def _quality_report(dataset_dir: Path, summaries: list, outdir: Path, plt, artifacts: list) -> dict:
    """Do not infer before-interpolation missingness from filled datasets."""
    after = pd.DataFrame(summaries)
    before_path = dataset_dir / "quality_summary.csv"
    quality = after
    prefill_available = False
    if before_path.exists():
        before = pd.read_csv(before_path)
        if "feature" in before:
            quality = after.merge(before, on="feature", how="left", suffixes=("_measured", ""))
            prefill_available = bool("missing_fraction_before" in quality and pd.to_numeric(quality["missing_fraction_before"], errors="coerce").notna().any())
    audit_rows, origin_rows = [], []
    for feature in after["feature"]:
        path = dataset_dir / "quality" / (feature + "_quality.csv")
        if not path.exists():
            continue
        audit = pd.read_csv(path)
        if "observed_points" not in audit:
            continue
        observed = pd.to_numeric(audit["observed_points"], errors="coerce")
        known = observed.notna()
        row = {"feature": feature, "prefill_rows_with_audit": int(known.sum()),
               "prefill_rows_without_audit": int((~known).sum()),
               "audited_missing_fraction_before": float((STEPS-observed[known]).sum() / (STEPS*known.sum())) if known.any() else np.nan}
        if "unavailable_channel" in audit:
            unavailable = audit["unavailable_channel"].astype(str).str.lower().isin(["true", "1", "1.0"])
            row["structurally_unavailable_windows"] = int(unavailable.sum())
        else:
            unavailable = pd.Series(False, index=audit.index)
        for column in ("interpolated_points", "edge_filled_points", "median_filled_points"):
            if column in audit:
                values = pd.to_numeric(audit[column], errors="coerce")
                row[column + "_audited_total"] = float(values.sum(min_count=1))
                if column == "median_filled_points":
                    row["wholly_median_imputed_audited_windows"] = int(values.eq(STEPS).sum())
        audit_rows.append(row)
        if "origin" in audit:
            for origin in audit["origin"].dropna().unique():
                group = audit["origin"].eq(origin)
                supported_known = group & known & ~unavailable
                origin_rows.append({"feature": feature, "origin": origin, "rows": int(group.sum()),
                                    "prefill_rows_with_audit": int((group & known).sum()),
                                    "prefill_rows_without_audit": int((group & ~known).sum()),
                                    "structurally_unavailable_windows": int((group & unavailable).sum()),
                                    "missing_fraction_before_supported_audited": float((STEPS-observed[supported_known]).sum() / (STEPS*supported_known.sum())) if supported_known.any() else np.nan})
    if audit_rows:
        quality = quality.merge(pd.DataFrame(audit_rows), on="feature", how="left")
    quality.to_csv(outdir / "feature_quality_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(quality))
    if prefill_available:
        audited_only = "prefill_rows_without_audit" in quality and quality["prefill_rows_without_audit"].fillna(0).gt(0).any()
        before_values = quality["audited_missing_fraction_before"] if audited_only else quality["missing_fraction_before"]
        ax.barh(y-0.15, before_values, height=0.3, label="Before filling{}".format(" (audited rows only)" if audited_only else ""), color="#c28b4c")
        ax.barh(y+0.15, quality["missing_fraction_after_measured"] if "missing_fraction_after_measured" in quality else quality["missing_fraction_after"], height=0.3, label="After filling", color="#548f72")
        ax.legend(frameon=False)
    else:
        ax.barh(y, quality["missing_fraction_after"], color="#548f72")
    ax.set(yticks=y, yticklabels=quality["feature"], xlim=(0, 1), xlabel="Fraction of time-series cells missing",
           title="Feature missingness{}".format(" before and after filling" if prefill_available else " after filling (prefill audit unavailable)"))
    _save(fig, outdir / "feature_missingness", plt, artifacts)
    if origin_rows:
        by_origin = pd.DataFrame(origin_rows)
        by_origin.to_csv(outdir / "feature_quality_by_origin.csv", index=False)
        pivot = by_origin.pivot(index="feature", columns="origin", values="structurally_unavailable_windows")
        fig, ax = plt.subplots(figsize=(11, 5))
        pivot.plot.bar(ax=ax)
        ax.set(xlabel="Feature", ylabel="Windows with structurally unavailable channels",
               title="Channel availability by preserved baseline and extension")
        ax.legend(title="Row origin", frameon=False)
        _save(fig, outdir / "structural_channel_availability", plt, artifacts)
    return {"prefill_quality_available": prefill_available, "source": str(before_path) if before_path.exists() else None,
            "per_sample_audit_features": [row["feature"] for row in audit_rows],
            "baseline_prefill_note": "Preserved filled baseline rows have unknown original missingness unless an audit is supplied; unknown rows are excluded from prefill fractions."}


def _run_tsne(values: np.ndarray, labels: np.ndarray, metadata: pd.DataFrame, feature: str, band: int,
              outdir: Path, plt, artifacts: list, random_state: int, max_samples: int, row_audit=None) -> dict:
    from sklearn.manifold import TSNE
    valid = np.flatnonzero(np.isfinite(values).all(axis=1))
    if len(valid) < 4:
        return {"status": "skipped", "reason": "fewer than four complete windows"}
    rng = np.random.RandomState(random_state)
    if max_samples and len(valid) > max_samples:
        # Reproducible class-stratified display subsample, stated in the manifest.
        positive = valid[labels[valid] == 1]
        negative = valid[labels[valid] == 0]
        npos = min(len(positive), max(1, round(max_samples * len(positive)/len(valid))))
        nneg = min(len(negative), max_samples-npos)
        valid = np.sort(np.concatenate([rng.choice(positive, npos, replace=False), rng.choice(negative, nneg, replace=False)]))
    selected = values[valid]
    scale = selected.std(axis=0)
    scale[scale == 0] = 1
    selected = (selected-selected.mean(axis=0))/scale
    # Pre-PCA is descriptive dimensionality reduction; no class labels enter fit.
    pca = full_window_pca(selected, n_components=min(50, len(selected)-1))
    import inspect
    kwargs = {"n_components": 2, "perplexity": min(30.0, (len(valid)-1)/3.0), "random_state": random_state, "init": "pca", "learning_rate": 200.0}
    # Support the Python 3.8-compatible scikit-learn release on the server.
    kwargs["max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"] = 1000
    embedding = TSNE(**kwargs).fit_transform(pca["scores"])
    table = pd.DataFrame({"dataset_row_index": metadata.index.to_numpy()[valid], "FlrOnset": metadata.iloc[valid]["FlrOnset"].to_numpy(),
                          "event_type": labels[valid], "TSNE1": embedding[:, 0], "TSNE2": embedding[:, 1]})
    if row_audit is not None:
        for column in ("sample_id", "origin", "source_era", "source_method"):
            if column in row_audit:
                table[column] = row_audit[column].reindex(table["dataset_row_index"]).to_numpy()
    table.to_csv(outdir / (DISPLAY_NAMES[feature] + "_tsne_coordinates.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, color, title in ((0, "#7c8d9f", "NSEP"), (1, "#d94b4b", "SEP")):
        mask = labels[valid] == label
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=12, alpha=0.65, c=color, linewidths=0, label="{} n={}".format(title, int(mask.sum())), rasterized=True)
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title="{}: descriptive t-SNE >{} MeV{}".format(DISPLAY_NAMES[feature], band, " (display subset)" if len(valid) < len(values) else ""))
    ax.legend(frameon=False)
    _save(fig, outdir / (DISPLAY_NAMES[feature] + "_tsne_2d"), plt, artifacts)
    return {"status": "complete", "plotted_samples": len(valid), "input_samples": len(values), "random_state": random_state,
            "perplexity": kwargs["perplexity"], "pre_pca_components": pca["scores"].shape[1]}


def run_compatible_figures(dataset_dir, outdir, bands: Sequence[int] = (10, 30, 60, 100),
                           features: Sequence[str] = FEATURES, tsne: bool = False,
                           tsne_max_samples: int = 3000, random_state: int = 42) -> dict:
    """Write PNG/PDF figures and audit tables for all ten extended files.

    Original energy cuts: >10 MeV peak >=10 pfu; >30, >60 and >100 MeV peak
    >=5 pfu. The > sign refers to the energy channel, not the peak threshold.
    CIs and Welch tests reproduce the notebook. FDR is additional and explicit.
    All PCA models are descriptive fits to the chosen SEP and NSEP rows jointly.
    t-SNE is opt-in because it is slow and not a validation of forecast skill.
    """
    dataset_dir, outdir = Path(dataset_dir), Path(outdir)
    bands, features = tuple(int(b) for b in bands), tuple(features)
    if not bands or len(set(bands)) != len(bands) or any(b not in BAND_SELECTION for b in bands):
        raise ValueError("Choose distinct bands from 10, 30, 60 and 100")
    if not features or len(set(features)) != len(features) or any(f not in FEATURES for f in features):
        raise ValueError("Unknown, empty or duplicate feature selection")
    if tsne_max_samples and tsne_max_samples < 4:
        raise ValueError("tsne_max_samples must be zero (all samples) or at least four")
    paths = {feature: _find_dataset(dataset_dir, feature) for feature in features}
    outdir.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    artifacts, sample_counts, quality_rows, availability_rows = [], [], [], []
    effects = {band: {} for band in bands}
    summaries = {band: [] for band in bands}
    reference_metadata = None
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": "original ten-feature MEMSEP full-window comparisons",
        "step_minutes": STEP_MINUTES, "steps": STEPS,
        "alignment": "FlrOnset ceil to 5 minutes; bins from -24h to -5min relative to aligned onset",
        "time_axis_note": "Legacy displayed linspace(24,0,288) is retained in statistics only; last physical bin is -5 min.",
        "band_selection": {str(b): {"peak_column": BAND_SELECTION[b][0], "minimum_peak_pfu": BAND_SELECTION[b][1]} for b in bands},
        "control_selection": "all event_type==0 flare rows, as in the original workflow",
        "statistics": {"ci": "normal 1.959963984540054*SEM, pointwise", "test": "two-sided Welch t test per time column",
                       "fdr": "Benjamini-Hochberg independently within 288 tests per feature and band",
                       "caveat": "Descriptive comparisons; temporal dependence, imputation and repeated flare associations can affect inference. These plots do not establish forecast accuracy."},
        "pca": "all 288 time columns standardized by combined selected SEP+NSEP mean and population SD; exact covariance eigenvectors; no extra imputation; first ten PCs",
        "features": {}, "warnings": [], "tsne_enabled": bool(tsne),
    }
    for feature, path in paths.items():
        print("[FIGURES] {}: reading {}".format(feature, path.name), flush=True)
        frame, x = _read_feature(path, feature)
        audit_path = dataset_dir / "quality" / (feature + "_quality.csv")
        row_audit = None
        if audit_path.exists():
            row_audit = pd.read_csv(audit_path)
            if "row_index" in row_audit and not row_audit["row_index"].duplicated().any():
                row_audit = row_audit.set_index("row_index")
            else:
                row_audit = None
        metadata = frame.drop(columns=timeseries_columns(feature))
        if reference_metadata is None:
            reference_metadata = metadata.copy()
            _plot_counts(frame, outdir, plt, artifacts)
        elif not metadata.equals(reference_metadata):
            raise ValueError("Feature files do not share identical metadata, sample order and labels: {}".format(path.name))
        missing_cells = ~np.isfinite(x)
        dates = pd.to_datetime(frame["FlrOnset"], utc=True, format="mixed")
        for year in sorted(dates.dt.year.unique()):
            for label in (0, 1):
                group = dates.dt.year.eq(year).to_numpy() & frame["event_type"].eq(label).to_numpy()
                if group.any():
                    available = ~missing_cells[group]
                    availability_rows.append({"feature": feature, "year": int(year), "event_type": label,
                                              "sample_windows": int(group.sum()), "windows_with_any_filled_values": int(available.any(axis=1).sum()),
                                              "complete_filled_windows": int(available.all(axis=1).sum()),
                                              "fraction_cells_available_after_filling": float(available.mean())})
        quality_rows.append({"feature": feature, "samples": len(frame), "time_points": STEPS,
                             "missing_fraction_after": float(missing_cells.mean()),
                             "entirely_missing_windows_after": int(missing_cells.all(axis=1).sum())})
        manifest["features"][feature] = {"path": str(path), "rows": len(frame), "columns": len(frame.columns), "bands": {}}
        for band in bands:
            sep_mask, nsep_mask = band_masks(frame, band)
            selected = sep_mask | nsep_mask
            band_dir = outdir / "figs_gt{}".format(band)
            band_dir.mkdir(exist_ok=True)
            nsep_count, sep_count = int(nsep_mask.sum()), int(sep_mask.sum())
            observed_window = np.isfinite(x).any(axis=1)
            sep_available = int((sep_mask & observed_window).sum())
            nsep_available = int((nsep_mask & observed_window).sum())
            sample_counts.append({"feature": feature, "energy_mev": band, "SEP": sep_count, "NSEP": nsep_count,
                                  "SEP_with_feature_values": sep_available, "NSEP_with_feature_values": nsep_available,
                                  "SEP_without_feature_values": sep_count-sep_available, "NSEP_without_feature_values": nsep_count-nsep_available,
                                  "excluded_SEP_below_threshold_or_unknown": int((frame["event_type"].eq(1) & ~sep_mask).sum())})
            info = {"sep_samples": sep_count, "nsep_samples": nsep_count,
                    "sep_with_feature_values": sep_available, "nsep_with_feature_values": nsep_available,
                    "sep_without_feature_values": sep_count-sep_available, "nsep_without_feature_values": nsep_count-nsep_available}
            manifest["features"][feature]["bands"][str(band)] = info
            if not sep_count or not nsep_count:
                message = "{} >{} MeV: empty SEP or NSEP group; comparison skipped".format(feature, band)
                manifest["warnings"].append(message)
                print("[WARN] " + message, flush=True)
                continue
            if not sep_available or not nsep_available:
                message = "{} >{} MeV: no measurements in SEP or NSEP feature windows; comparison skipped".format(feature, band)
                manifest["warnings"].append(message)
                info["comparison_skipped"] = "no measurements in one or both classes"
                print("[WARN] " + message, flush=True)
                continue
            stats = profile_statistics(x[sep_mask], x[nsep_mask], feature, band)
            stats.to_csv(band_dir / (DISPLAY_NAMES[feature] + "_profile_statistics.csv"), index=False)
            effects[band][feature] = stats
            _plot_profile(stats, feature, band, band_dir, plt, artifacts)
            _plot_examples(frame, x, sep_mask, nsep_mask, feature, band, band_dir, plt, artifacts)
            for hour in (24, 12, 6, 0):
                # Preserve old summary selection exactly; physical time is added.
                index = int(np.argmin(np.abs(stats["legacy_display_hours"].to_numpy()-hour)))
                row = stats.iloc[index]
                summaries[band].append({"variable": DISPLAY_NAMES[feature], "hours_before": row["legacy_display_hours"],
                                        "hours_before_aligned_flare": row["hours_before_aligned_flare"],
                                        "sep_gt{}_mean".format(band): row["sep_mean"], "nsep_mean": row["nsep_mean"],
                                        "difference": row["difference"], "p_value": row["p_value"],
                                        "q_value_bh_within_feature_band": row["q_value_bh_within_feature_band"]})
            try:
                pca = full_window_pca(x[selected])
            except ValueError as error:
                info["pca_skipped"] = str(error)
                manifest["warnings"].append("{} >{} MeV PCA: {}".format(feature, band, error))
                continue
            info["pca_excluded_incomplete_rows"] = int((~pca["retained_rows"]).sum())
            info["pca_zero_total_variance"] = pca["zero_total_variance"]
            labels = pd.to_numeric(frame.loc[selected, "event_type"]).to_numpy()
            _plot_pca(pca, labels, frame.loc[selected], feature, band, band_dir, plt, artifacts, row_audit)
            if tsne:
                info["tsne"] = _run_tsne(x[selected], labels, frame.loc[selected], feature, band, band_dir,
                                         plt, artifacts, random_state, tsne_max_samples, row_audit)
            print("[FIGURES] {} >{} MeV: SEP={} NSEP={} complete".format(feature, band, sep_count, nsep_count), flush=True)
    for band in bands:
        if effects[band]:
            band_dir = outdir / "figs_gt{}".format(band)
            _plot_difference_heatmap(effects[band], band, band_dir, plt, artifacts)
            pd.DataFrame(summaries[band]).to_csv(band_dir / "summary_key_times_gt{}.csv".format(band), index=False)
    counts = pd.DataFrame(sample_counts)
    counts.to_csv(outdir / "sample_counts_by_feature_band.csv", index=False)
    availability = pd.DataFrame(availability_rows)
    availability.to_csv(outdir / "feature_availability_by_year_label.csv", index=False)
    if len(availability):
        summed = availability.assign(available_cells=lambda f: f["fraction_cells_available_after_filling"] * f["sample_windows"] * STEPS).groupby(["feature", "year"]).agg({"sample_windows": "sum", "available_cells": "sum"})
        matrix = (summed["available_cells"] / (summed["sample_windows"]*STEPS)).unstack("year").reindex(features)
        fig, ax = plt.subplots(figsize=(14, 5))
        im = ax.imshow(matrix.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")
        ax.set(yticks=np.arange(len(matrix)), yticklabels=matrix.index, xticks=np.arange(len(matrix.columns)),
               xticklabels=matrix.columns.astype(str), xlabel="FlrOnset year", title="Saved feature availability by year (after filling)")
        ax.tick_params(axis="x", rotation=90)
        fig.colorbar(im, ax=ax, label="Fraction available in filled time-series cells")
        _save(fig, outdir / "feature_availability_by_year", plt, artifacts)
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(counts):
        first = counts[counts["feature"].eq(features[0])].set_index("energy_mev")[["SEP", "NSEP"]]
        first.plot.bar(ax=ax, color=["#d94b4b", "#7c8d9f"])
        ax.set_yscale("symlog", linthresh=10)
        ax.set(xlabel="Integral proton energy threshold (MeV)", ylabel="Flare sample count (symlog scale)", title="Original energy-band sample selection")
    _save(fig, outdir / "sample_counts_by_energy", plt, artifacts)
    manifest["quality"] = _quality_report(dataset_dir, quality_rows, outdir, plt, artifacts)
    if not manifest["quality"]["prefill_quality_available"]:
        manifest["warnings"].append("No prefill audit summary was available; missingness figure reports after filling only.")
    manifest["figure_files"] = artifacts
    manifest_path = outdir / "figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("[DONE] Figures and tables: {}".format(outdir), flush=True)
    return {"outdir": str(outdir), "manifest": str(manifest_path), "figure_files": artifacts,
            "feature_count": len(features), "bands": list(bands), "warnings": manifest["warnings"]}
