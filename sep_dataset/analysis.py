from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .utils import ensure_dir


WINDOW_RE = re.compile(r"^(?P<feature>.+)_(?P<step>\d+)$")


def window_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    groups: dict[str, list[tuple[int, str]]] = {}
    for col in df.columns:
        match = WINDOW_RE.match(str(col))
        if not match:
            continue
        groups.setdefault(match.group("feature"), []).append((int(match.group("step")), col))
    return {feature: [col for _, col in sorted(items, reverse=True)] for feature, items in groups.items() if len(items) >= 6}


def profile_features(df: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for feature, cols in groups.items():
        values = df[cols].apply(pd.to_numeric, errors="coerce")
        values = values.replace(-9999.0, np.nan)
        rows.append(
            {
                "feature": feature,
                "n_window_columns": len(cols),
                "missing_fraction": float(values.isna().mean().mean()),
                "mean": float(values.stack().mean()) if values.notna().any().any() else np.nan,
                "std": float(values.stack().std()) if values.notna().any().any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("missing_fraction")


def build_summary_matrix(df: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    summary = pd.DataFrame(index=df.index)
    for feature, cols in groups.items():
        values = df[cols].apply(pd.to_numeric, errors="coerce").replace(-9999.0, np.nan)
        summary[f"{feature}_mean"] = values.mean(axis=1)
        summary[f"{feature}_std"] = values.std(axis=1)
        summary[f"{feature}_min"] = values.min(axis=1)
        summary[f"{feature}_max"] = values.max(axis=1)
        summary[f"{feature}_last"] = values.iloc[:, -1]
        summary[f"{feature}_slope"] = values.iloc[:, -1] - values.iloc[:, 0]
    return summary


def plot_label_counts(df: pd.DataFrame, outdir: Path) -> None:
    work = df.copy()
    work["anchor_time"] = pd.to_datetime(work["anchor_time"], utc=True, errors="coerce")
    work["year"] = work["anchor_time"].dt.year
    counts = work.groupby(["year", "event_type"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", stacked=True, figsize=(13, 4), color=["#7a8799", "#d64b4b"])
    plt.xlabel("Anchor year")
    plt.ylabel("Samples")
    plt.title("SEP and non-SEP sample coverage by year")
    plt.tight_layout()
    plt.savefig(outdir / "label_counts_by_year.png", dpi=220)
    plt.close()


def plot_source_counts(df: pd.DataFrame, outdir: Path) -> None:
    positives = df[df["event_type"] == 1].copy()
    if positives.empty or "source" not in positives.columns:
        return
    exploded = positives["source"].fillna("").str.split("+").explode()
    counts = exploded[exploded != ""].value_counts()
    if counts.empty:
        return
    plt.figure(figsize=(8, 4))
    sns.barplot(x=counts.values, y=counts.index, color="#4c78a8")
    plt.xlabel("Merged event count")
    plt.ylabel("Catalog source")
    plt.title("Catalog source coverage")
    plt.tight_layout()
    plt.savefig(outdir / "catalog_source_coverage.png", dpi=220)
    plt.close()


def plot_missingness(profile: pd.DataFrame, outdir: Path) -> None:
    if profile.empty:
        return
    plt.figure(figsize=(9, max(3, 0.35 * len(profile))))
    sns.barplot(data=profile, x="missing_fraction", y="feature", color="#59a14f")
    plt.xlabel("Fraction missing before interpolation")
    plt.ylabel("Feature")
    plt.title("Window missingness by feature")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(outdir / "feature_missingness.png", dpi=220)
    plt.close()


def plot_pca(summary: pd.DataFrame, labels: pd.Series, outdir: Path) -> pd.DataFrame:
    if summary.shape[1] < 2 or summary.shape[0] < 3:
        return pd.DataFrame()
    imputed = SimpleImputer(strategy="median").fit_transform(summary)
    scaled = StandardScaler().fit_transform(imputed)
    pca = PCA(n_components=min(10, scaled.shape[1], scaled.shape[0]))
    transformed = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(transformed[:, :2], columns=["PC1", "PC2"], index=summary.index)
    pca_df["event_type"] = labels.values

    plt.figure(figsize=(7, 5.5))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="event_type", palette={0: "#7a8799", 1: "#d64b4b"}, alpha=0.75, s=28)
    plt.title("PCA projection of window summary features")
    plt.tight_layout()
    plt.savefig(outdir / "pca_projection.png", dpi=240)
    plt.close()

    variance = pd.DataFrame({"component": np.arange(1, len(pca.explained_variance_ratio_) + 1), "explained_variance_ratio": pca.explained_variance_ratio_})
    plt.figure(figsize=(7, 4))
    sns.barplot(data=variance, x="component", y="explained_variance_ratio", color="#f28e2b")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title("PCA explained variance")
    plt.tight_layout()
    plt.savefig(outdir / "pca_variance.png", dpi=220)
    plt.close()
    return variance


def plot_mean_profiles(df: pd.DataFrame, groups: dict[str, list[str]], outdir: Path) -> None:
    max_features = 12
    selected = list(groups.items())[:max_features]
    if not selected:
        return
    ncols = 2
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.0 * nrows), squeeze=False)
    for ax, (feature, cols) in zip(axes.ravel(), selected):
        x = np.linspace(24, 0, len(cols))
        for label, color, name in [(1, "#d64b4b", "SEP"), (0, "#7a8799", "NSEP")]:
            values = df.loc[df["event_type"] == label, cols].apply(pd.to_numeric, errors="coerce").replace(-9999.0, np.nan)
            if values.empty:
                continue
            mean = values.mean(axis=0).to_numpy(dtype=float)
            ax.plot(x, mean, label=name, color=color, linewidth=1.8)
        ax.set_title(feature)
        ax.set_xlim(24, 0)
        ax.set_xlabel("Hours before anchor")
        ax.grid(True, alpha=0.25)
    for ax in axes.ravel()[len(selected) :]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Mean pre-event profiles", y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "mean_profiles_sep_vs_nsep.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_correlation(summary: pd.DataFrame, outdir: Path) -> None:
    if summary.shape[1] < 2:
        return
    corr = summary.corr(numeric_only=True)
    plt.figure(figsize=(min(16, max(7, 0.25 * corr.shape[1])), min(14, max(6, 0.25 * corr.shape[0]))))
    sns.heatmap(corr, cmap="vlag", center=0, square=False, cbar_kws={"shrink": 0.7})
    plt.title("Correlation of window summary features")
    plt.tight_layout()
    plt.savefig(outdir / "feature_correlation.png", dpi=220)
    plt.close()


def run_plots(dataset_csv: str | Path, outdir: str | Path = "reports/figures") -> dict[str, Path]:
    outdir = ensure_dir(outdir)
    df = pd.read_csv(dataset_csv)
    groups = window_columns(df)
    if "event_type" not in df.columns:
        raise ValueError("dataset must include event_type")

    profile = profile_features(df, groups)
    profile_path = outdir / "feature_profile.csv"
    profile.to_csv(profile_path, index=False)

    summary = build_summary_matrix(df, groups)
    summary_path = outdir / "window_summary_features.csv"
    summary.to_csv(summary_path, index=False)

    plot_label_counts(df, outdir)
    plot_source_counts(df, outdir)
    plot_missingness(profile, outdir)
    variance = plot_pca(summary, df["event_type"], outdir)
    if not variance.empty:
        variance.to_csv(outdir / "pca_explained_variance.csv", index=False)
    plot_mean_profiles(df, groups, outdir)
    plot_correlation(summary, outdir)
    return {"profile": profile_path, "summary": summary_path}
