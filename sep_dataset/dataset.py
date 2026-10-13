from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .catalogs import collect_event_catalogs
from .hapi import DEFAULT_HAPI_SPECS, HapiClient, HapiDatasetSpec, HapiMonthlyStore
from .utils import ensure_dir, utc_timestamp


@dataclass
class BuildResult:
    raw_catalog_path: Path
    merged_catalog_path: Path
    samples_path: Path
    dataset_path: Path | None


def anchor_for_event(row: pd.Series, anchor: str) -> pd.Timestamp:
    if anchor == "flare_time" and pd.notna(row.get("flare_time")):
        return utc_timestamp(row["flare_time"])
    if anchor == "max_time" and pd.notna(row.get("max_time")):
        return utc_timestamp(row["max_time"])
    return utc_timestamp(row["start_time"])


def make_samples(
    events: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    anchor: str = "start_time",
    negative_ratio: float = 3.0,
    negative_cadence: str = "12h",
    blackout_days: float = 3.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i, (_, event) in enumerate(events.iterrows(), start=1):
        anchor_time = anchor_for_event(event, anchor)
        if pd.isna(anchor_time):
            continue
        record = event.to_dict()
        record.update(
            {
                "sample_id": f"SEP_{i:05d}",
                "event_type": 1,
                "label": 1,
                "anchor_time": anchor_time,
                "sample_source": "catalog_positive",
            }
        )
        rows.append(record)

    if negative_ratio > 0 and rows:
        rng = np.random.default_rng(random_seed)
        start_ts = utc_timestamp(start)
        end_ts = utc_timestamp(end)
        candidates = pd.date_range(start_ts, end_ts, freq=negative_cadence, inclusive="both")
        event_times = pd.to_datetime(events["start_time"], utc=True, errors="coerce").dropna()
        blackout = pd.Timedelta(days=blackout_days)
        keep = []
        for candidate in candidates:
            if event_times.empty:
                keep.append(True)
            else:
                keep.append(bool(((event_times - candidate).abs() > blackout).all()))
        quiet = candidates[np.array(keep, dtype=bool)]
        target_n = min(len(quiet), int(round(len(rows) * negative_ratio)))
        if target_n > 0:
            chosen = np.sort(rng.choice(np.arange(len(quiet)), size=target_n, replace=False))
            for j, ts in enumerate(quiet[chosen], start=1):
                rows.append(
                    {
                        "sample_id": f"NSEP_{j:05d}",
                        "event_type": 0,
                        "label": 0,
                        "source": "synthetic_quiet_time",
                        "start_time": pd.NaT,
                        "anchor_time": ts,
                        "sample_source": "quiet_time_negative",
                    }
                )

    samples = pd.DataFrame(rows)
    samples["anchor_time"] = pd.to_datetime(samples["anchor_time"], utc=True, errors="coerce")
    return samples.sort_values(["event_type", "anchor_time"], ascending=[False, True]).reset_index(drop=True)


def extract_window(
    series: pd.Series,
    anchor_time: pd.Timestamp,
    lookback_hours: float,
    cadence: str,
    steps: int,
    fill_value: float,
    interpolate: bool = True,
) -> tuple[np.ndarray, float]:
    anchor_time = utc_timestamp(anchor_time).ceil(cadence)
    freq = pd.Timedelta(cadence)
    index = pd.date_range(end=anchor_time - freq, periods=steps, freq=cadence, tz="UTC")
    if series.empty:
        return np.full(steps, fill_value, dtype=float), 1.0
    series = pd.to_numeric(series, errors="coerce")
    series = series.sort_index()
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    aligned = series.reindex(index)
    missing_before_fill = float(aligned.isna().mean())
    if interpolate:
        aligned = aligned.interpolate(limit_direction="both").ffill().bfill()
    values = aligned.fillna(fill_value).to_numpy(dtype=float)
    return values, missing_before_fill


def build_feature_windows(
    samples: pd.DataFrame,
    specs: list[HapiDatasetSpec] | None = None,
    cache_dir: str | Path = "data/cache/hapi",
    lookback_hours: float = 24.0,
    cadence: str = "5min",
    fill_value: float = -9999.0,
    selected_dataset_names: set[str] | None = None,
) -> pd.DataFrame:
    specs = specs or DEFAULT_HAPI_SPECS
    specs = [spec for spec in specs if spec.enabled and (selected_dataset_names is None or spec.name in selected_dataset_names)]
    steps = int(pd.Timedelta(hours=lookback_hours) / pd.Timedelta(cadence))
    output = samples.copy()
    client = HapiClient(cache_dir=cache_dir)

    for spec in specs:
        print(f"[INFO] Preparing HAPI source: {spec.name} ({spec.dataset})")
        try:
            info = client.info(spec)
            selected = client.select_parameters(info, spec.feature_aliases)
        except Exception as exc:
            print(f"[WARN] Skipping {spec.name}: {exc}")
            continue
        if not selected:
            print(f"[WARN] No configured parameters found for {spec.name}; skipping")
            continue
        print("[INFO] Selected parameters: " + ", ".join(f"{item.feature}={item.parameter}" for item in selected))
        store = HapiMonthlyStore(client=client, spec=spec, selected=selected, info=info)

        feature_values: dict[str, list[np.ndarray]] = {item.feature: [] for item in selected}
        missing_values: dict[str, list[float]] = {item.feature: [] for item in selected}
        for _, sample in samples.iterrows():
            anchor_time = utc_timestamp(sample["anchor_time"])
            start = anchor_time - pd.Timedelta(hours=lookback_hours)
            stop = anchor_time
            df = store.range(start, stop)
            for item in selected:
                series = df[item.parameter] if not df.empty and item.parameter in df.columns else pd.Series(dtype=float)
                values, missing = extract_window(series, anchor_time, lookback_hours, cadence, steps, fill_value)
                feature_values[item.feature].append(values)
                missing_values[item.feature].append(missing)

        for feature, arrays in feature_values.items():
            matrix = np.vstack(arrays) if arrays else np.empty((0, steps))
            columns = [f"{feature}_{i}" for i in range(steps, 0, -1)]
            output = pd.concat([output, pd.DataFrame(matrix, columns=columns, index=output.index)], axis=1)
            output[f"{feature}_missing_fraction"] = missing_values[feature]
    return output


def build_extended_dataset(
    start: str = "1998-01-01",
    end: str = "2024-12-31",
    output_dir: str | Path = "data/processed",
    cache_dir: str | Path = "data/cache",
    memsep_csv: str | Path | None = None,
    catalog_only: bool = False,
    anchor: str = "start_time",
    negative_ratio: float = 3.0,
    negative_cadence: str = "12h",
    blackout_days: float = 3.0,
    lookback_hours: float = 24.0,
    cadence: str = "5min",
    fill_value: float = -9999.0,
    datasets: list[str] | None = None,
    include_noaa: bool = True,
    include_cdaw: bool = True,
    include_donki: bool = True,
    donki_api_key: str = "DEMO_KEY",
) -> BuildResult:
    output_dir = ensure_dir(output_dir)
    cache_dir = ensure_dir(cache_dir)
    raw, merged = collect_event_catalogs(
        start=start,
        end=end,
        cache_dir=cache_dir / "catalogs",
        memsep_csv=memsep_csv,
        include_noaa=include_noaa,
        include_cdaw=include_cdaw,
        include_donki=include_donki,
        donki_api_key=donki_api_key,
    )
    raw_path = output_dir / "sep_catalog_raw_sources.csv"
    merged_path = output_dir / "sep_catalog_merged.csv"
    raw.to_csv(raw_path, index=False)
    merged.to_csv(merged_path, index=False)

    samples = make_samples(
        merged,
        start=start,
        end=end,
        anchor=anchor,
        negative_ratio=negative_ratio,
        negative_cadence=negative_cadence,
        blackout_days=blackout_days,
    )
    samples_path = output_dir / "sep_samples.csv"
    samples.to_csv(samples_path, index=False)

    if catalog_only:
        return BuildResult(raw_path, merged_path, samples_path, None)

    selected = set(datasets) if datasets else None
    dataset = build_feature_windows(
        samples=samples,
        specs=DEFAULT_HAPI_SPECS,
        cache_dir=cache_dir / "hapi",
        lookback_hours=lookback_hours,
        cadence=cadence,
        fill_value=fill_value,
        selected_dataset_names=selected,
    )
    dataset_path = output_dir / f"extended_sep_dataset_{start[:4]}_{end[:4]}.csv"
    dataset.to_csv(dataset_path, index=False)
    return BuildResult(raw_path, merged_path, samples_path, dataset_path)
