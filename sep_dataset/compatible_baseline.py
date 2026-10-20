"""Frozen copies of the ten published historical feature datasets.

The extension must append to these records, never rebuild or relabel them.
Only the extra energy-bucket ``Label`` column in the Zenodo copies is dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import requests


FEATURES = ("Vx", "flow_speed", "proton_density", "T", "F", "Xs", "Xl", "P4", "P5", "P6")
ZENODO_RECORD = "17595580"
HISTORICAL_END = "2013-12-31"
STEPS = 288


@dataclass(frozen=True)
class BaselineSpec:
    feature: str
    published_name: str
    size: int
    md5: str

    @property
    def url(self) -> str:
        return f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{self.published_name}/content"


BASELINES = {
    x.feature: x for x in (
        BaselineSpec("Vx", "Vx.csv", 97371010, "9d69e5b4911699a2f8b1824295ac4b98"),
        BaselineSpec("flow_speed", "V.csv", 92657375, "d26db8784a354e8e0e60999f4d81d1c3"),
        BaselineSpec("proton_density", "Np.csv", 102313281, "1d5c90b8b208e996de62e1918d29c052"),
        BaselineSpec("T", "Tp.csv", 56026173, "3e7f5f2897aa6eb19293b56d67df6187"),
        BaselineSpec("F", "F.csv", 100762551, "f56b8fa3a6daf00df62d17993cf4e0a3"),
        BaselineSpec("Xs", "Xs.csv", 62867345, "ae19071faf60f5713a20fe0d0b10a10a"),
        BaselineSpec("Xl", "Xl.csv", 63455644, "1a3575afc5ac050b3a0acf99723e616c"),
        BaselineSpec("P4", "P4.csv", 49202583, "87e930b78446d587c9c6ec06997a9cd3"),
        BaselineSpec("P5", "P5.csv", 51512335, "78fddb725a8c57ee565e1b2937f17278"),
        BaselineSpec("P6", "P6.csv", 55152914, "38162f5553065f6eabb16f0539160db1"),
    )
}


def window_columns(feature: str) -> list[str]:
    if feature not in FEATURES:
        raise ValueError(f"Unknown original feature: {feature}")
    return [f"{feature}_{i}" for i in range(STEPS, 0, -1)]


def original_metadata_path() -> Path:
    return Path(__file__).resolve().parents[1] / "1998_2013_MEMSEP_dataset.csv"


def read_original_metadata(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path is not None else original_metadata_path()
    frame = pd.read_csv(p, float_precision="round_trip", low_memory=False)
    if "FlrOnset" not in frame or "event_type" not in frame:
        raise ValueError(f"Historical metadata lacks FlrOnset/event_type: {p}")
    return frame


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_baseline(spec: BaselineSpec, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == spec.size and file_md5(target) == spec.md5:
        print(f"[CACHE] Original {spec.feature}: {target}", flush=True)
        return target
    part = target.with_suffix(target.suffix + ".part")
    last_error = None
    for attempt in range(3):
        try:
            if part.exists() and part.stat().st_size >= spec.size:
                if part.stat().st_size == spec.size and file_md5(part) == spec.md5:
                    os.replace(part, target)
                    return target
                part.unlink()
            offset = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            print(f"[DOWNLOAD] Original {spec.feature}: attempt {attempt + 1}/3, {offset}/{spec.size} bytes", flush=True)
            with requests.get(spec.url, headers=headers, stream=True, timeout=(20, 120)) as response:
                if response.status_code == 416:
                    part.unlink(missing_ok=True)
                    raise ValueError("Server rejected the partial-file range; restarting the download")
                response.raise_for_status()
                if response.status_code == 206:
                    content_range = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if (content_range is None or int(content_range[1]) != offset
                            or int(content_range[2]) < offset or int(content_range[2]) >= spec.size
                            or int(content_range[3]) != spec.size):
                        part.unlink(missing_ok=True)
                        raise ValueError("Invalid Content-Range for published baseline; restarting the download")
                append = bool(offset and response.status_code == 206)
                if not append:
                    offset = 0
                written, last_report = offset, offset
                with part.open("ab" if append else "wb") as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        stream.write(chunk)
                        written += len(chunk)
                        if written - last_report >= 16 * 1024 * 1024:
                            print(f"[DOWNLOAD] Original {spec.feature}: {written}/{spec.size} bytes", flush=True)
                            last_report = written
            if part.stat().st_size < spec.size:
                raise ValueError(f"Incomplete baseline download for {spec.feature}; retaining valid partial bytes")
            if part.stat().st_size != spec.size or file_md5(part) != spec.md5:
                part.unlink(missing_ok=True)
                raise ValueError(f"Published baseline checksum/size mismatch for {spec.feature}")
            os.replace(part, target)
            return target
        except (requests.RequestException, OSError, ValueError) as exc:
            last_error = exc
            print(f"[RETRY] Original {spec.feature}: {exc}", flush=True)
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Cannot download frozen original {spec.feature}: {last_error}")


def locate_baseline(feature: str, cache_dir: str | Path, baseline_dir: str | Path | None = None) -> Path:
    spec = BASELINES[feature]
    if baseline_dir is not None:
        root = Path(baseline_dir).expanduser()
        candidates = [root / f"1998_2013_MEMSEP_dataset_with_{feature}_filled.csv", root / spec.published_name]
        existing = [p for p in candidates if p.is_file()]
        if len(existing) != 1:
            raise FileNotFoundError(f"Expected one original {feature} baseline in {root}; checked {candidates}")
        return existing[0]
    return _download_baseline(spec, Path(cache_dir) / "original_baselines" / spec.published_name)


def load_baseline(feature: str, path: str | Path, metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    expected = (metadata if metadata is not None else read_original_metadata()).reset_index(drop=True)
    frame = pd.read_csv(path, float_precision="round_trip", low_memory=False)
    if "Label" in frame:
        # Published Label is an auxiliary energy bucket, not a second binary
        # target: 83 original event_type=1 records are also marked NSEP here.
        # The original event_type remains authoritative and is checked below.
        allowed = {"NSEP": {0, 1}, "SEP": {1}, "gt10": {1}, "gt30": {1}, "gt60": {1}, "gt100": {1}}
        if not all(label in allowed and event in allowed[label]
                   for label, event in zip(frame["Label"].astype(str), frame["event_type"])):
            raise ValueError(f"Unrecognized or incompatible auxiliary baseline Label in {path}")
        frame = frame.drop(columns="Label")
    columns = list(expected.columns) + window_columns(feature)
    if list(frame.columns) != columns:
        raise ValueError(f"Baseline {feature} must have original metadata followed by its 288 ordered values: {path}")
    if len(frame) != len(expected):
        raise ValueError(f"Baseline {feature} has {len(frame)} rows; original cohort has {len(expected)}")
    onset = pd.to_datetime(frame["FlrOnset"], utc=True, errors="coerce")
    original_onset = pd.to_datetime(expected["FlrOnset"], utc=True, errors="coerce")
    if onset.isna().any() or not onset.equals(original_onset):
        raise ValueError(f"Baseline {feature} changes the historical flare cohort/order")
    if not np.array_equal(frame["event_type"].to_numpy(), expected["event_type"].to_numpy()):
        raise ValueError(f"Baseline {feature} changes the historical SEP/NSEP labels")
    # The frozen filled files are authoritative for their exact stored values.
    # They contain tiny CSV-roundtrip differences from the metadata-only CSV;
    # validate correspondence without replacing or rounding the frozen values.
    for column in expected.columns:
        original = expected[column]
        observed = frame[column]
        if pd.api.types.is_numeric_dtype(original):
            numeric = pd.to_numeric(observed, errors="coerce")
            invalid = observed.notna() & numeric.isna()
            matches = np.isclose(numeric.to_numpy(dtype=float), original.to_numpy(dtype=float), rtol=1e-12, atol=0, equal_nan=True)
            if invalid.any() or not matches.all():
                raise ValueError(f"Baseline {feature} changes historical metadata {column!r}")
        else:
            same_missing = observed.isna().equals(original.isna())
            present = original.notna()
            if not same_missing or not observed[present].astype(str).equals(original[present].astype(str)):
                raise ValueError(f"Baseline {feature} changes historical metadata {column!r}")
    values = frame[window_columns(feature)].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(f"Baseline {feature} contains infinite measurements")
    return frame


def baseline_fingerprint(frame: pd.DataFrame, feature: str) -> str:
    """Hash all historical fields, with stable integer/float and NaN encoding.

    Numeric columns use canonical float64 bytes, so concatenating missing values
    (which promotes integer metadata to float) does not change the fingerprint.
    String fields retain their exact text. Column order and row order are hashed.
    """
    if not set(window_columns(feature)).issubset(frame.columns):
        raise ValueError(f"Historical {feature} window columns are missing")
    digest = hashlib.sha256()
    digest.update(str(len(frame)).encode("ascii") + b"\0")
    for column in frame.columns:
        encoded_name = str(column).encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "little") + encoded_name)
        values = frame[column]
        numeric = pd.to_numeric(values, errors="coerce")
        if pd.api.types.is_numeric_dtype(values) or numeric.notna().sum() == values.notna().sum():
            digest.update(b"N")
            data = numeric.to_numpy(dtype="<f8", copy=True)
            data[np.isnan(data)] = np.nan
            data[data == 0] = 0.0  # normalize negative zero
            digest.update(np.ascontiguousarray(data).tobytes())
        else:
            digest.update(b"S")
            for value in values:
                if pd.isna(value):
                    digest.update(b"\0")
                else:
                    data = str(value).encode("utf-8")
                    digest.update(b"\1" + len(data).to_bytes(8, "little") + data)
    return digest.hexdigest()
