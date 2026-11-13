"""Yearly P4/P5/P6 checks for the instrument transition."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .compatible_sources import LEGACY_END, PROTON_FEATURES


POINT_FRACTIONS = {
    "O": "observed_fraction",
    "D": "estimated_fraction",
    "H": "held_fraction",
    "I": "interpolated_fraction",
    "E": "edge_filled_fraction",
    "M": "median_filled_fraction",
    "U": "historical_provenance_unknown_fraction",
    "S": "structurally_unavailable_fraction",
    "G": "remaining_gap_fraction",
}
CLASSES = ((0, "NSEP", "#246482"), (1, "SEP", "#a84e30"))


def summarize_proton_years(feature, years, summary):
    """Summarize each year's flare windows, retaining empty classes and years."""
    if feature not in PROTON_FEATURES:
        raise ValueError("Yearly proton diagnostics support P4, P5 and P6")
    years = np.asarray(years, dtype=int)
    labels = summary.event_type.to_numpy()
    if not len(years) or len(years) != len(summary) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Yearly proton diagnostics require aligned years and binary labels")
    means = summary.filled_mean.to_numpy(dtype=float)
    complete = summary.filled_valid_points.to_numpy() == 288
    counts = summary[["points_" + code for code in POINT_FRACTIONS]].to_numpy(dtype=int)
    if (counts < 0).any() or not (counts.sum(axis=1) == 288).all():
        raise ValueError("Yearly proton diagnostics require one point status per five-minute bin")
    rows = []
    for year in range(int(years.min()), int(years.max()) + 1):
        for label, name, _ in CLASSES:
            selected = (years == year) & (labels == label)
            n_windows = int(selected.sum())
            data = means[selected & complete]
            data = data[np.isfinite(data) & (data >= 0)]
            q25, median, q75 = np.percentile(data, [25, 50, 75]) if len(data) else (np.nan,) * 3
            fractions = counts[selected].sum(axis=0) / (288 * n_windows) if n_windows else np.full(len(POINT_FRACTIONS), np.nan)
            rows.append({"feature": feature, "year": year, "class": name, "sample_windows": n_windows,
                         "valid_window_means": len(data), "zero_window_means": int((data == 0).sum()),
                         "mean_window_mean": data.mean() if len(data) else np.nan,
                         "q25_window_mean": q25, "median_window_mean": median, "q75_window_mean": q75,
                         **dict(zip(POINT_FRACTIONS.values(), fractions))})
    return pd.DataFrame(rows)


def plot_proton_years(table, plt):
    """Plot yearly medians and interquartile ranges of 24-hour window means."""
    from matplotlib.ticker import MaxNLocator

    features = [feature for feature in PROTON_FEATURES if feature in set(table.feature)]
    if not features:
        raise ValueError("No proton channels to plot")
    fig, axes = plt.subplots(len(features), 1, sharex=True, figsize=(9, 2.25 * len(features) + 1), squeeze=False)
    transition = LEGACY_END.year + (LEGACY_END.dayofyear - 1) / (366 if LEGACY_END.is_leap_year else 365)
    for feature, ax in zip(features, axes[:, 0]):
        channel = table[table.feature.eq(feature)]
        for _, name, color in CLASSES:
            subset = channel[channel["class"].eq(name)].sort_values("year")
            year = subset.year.to_numpy()
            med = subset.median_window_mean.to_numpy(dtype=float)
            ax.plot(year, med, color=color, marker="o", markersize=3, lw=1.25, label=name)
            multiple = subset.valid_window_means.to_numpy() >= 2
            ax.fill_between(year, subset.q25_window_mean.to_numpy(dtype=float), subset.q75_window_mean.to_numpy(dtype=float),
                            where=multiple, color=color, alpha=.16, linewidth=0)
        bounds = channel[["q25_window_mean", "q75_window_mean"]].to_numpy(dtype=float)
        finite = bounds[np.isfinite(bounds)]
        positive = finite[finite > 0]
        if len(positive) and (finite > 0).all():
            ax.set_yscale("log")
        elif len(positive):
            ax.set_yscale("symlog", linthresh=float(positive.min()))
        if channel.year.min() <= transition <= channel.year.max():
            ax.axvline(transition, color="0.35", lw=.8, linestyle="--", label="March 2020 transition")
        ax.set_title(feature, loc="left", fontweight="bold", fontsize=10)
        ax.grid(axis="y", alpha=.22, linewidth=.6)
        ax.set_axisbelow(True)
    axes[0, 0].legend(title="Median and interquartile range", loc="best", frameon=False, fontsize=8, title_fontsize=8)
    axes[-1, 0].set_xlabel("Flare year (UTC)")
    axes[-1, 0].xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
    fig.supylabel(r"24-hour mean differential proton flux (cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)", fontsize=10)
    return fig
