"""Read-only repair adapter: raw historical availability and calibrated protons."""
from __future__ import annotations

from collections import OrderedDict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cleaned_builder import _load_quality, _read_feature
from .compatible_builder import file_sha256, window_index
from .enhanced_builder import LEGACY_END, feature_columns
from .historical_masks import load_mask_bundle, apply_historical_masks, DEFAULT_BUNDLE
from .proton_calibration import KEY_FIELDS, load_calibration_bank, apply_calibration_bank

DEFAULT_BANK = Path(__file__).resolve().parents[1] / "calibration" / "proton_calibration_bank.json"
VERSION = "raw-masks-spectral-response-v2"
BAD_CALIBRATION = {"unsupported_source", "unsupported_energy_grid", "outside_validated_spectral_range"}


def source_key(row):
    return json.dumps({key: str(row[key]) for key in KEY_FIELDS}, sort_keys=True, separators=(",", ":"))


class RepairAdapter:
    def __init__(self, root, original_manifest, calibration_bank=None, historical_masks=None, cache_dir=None):
        self.root, self.manifest = Path(root), original_manifest
        self.bank_path = Path(calibration_bank or DEFAULT_BANK)
        self.mask_path = Path(historical_masks or DEFAULT_BUNDLE)
        self.bank = load_calibration_bank(self.bank_path)
        self.masks = load_mask_bundle(self.mask_path)
        self.bank_sha = file_sha256(self.bank_path)
        self.cache_dir = Path(cache_dir or self.root.parent / ".proton_response_cache")
        self.native_paths = {p.name: p for p in (self.root / "continuation_audit").glob("native_sgps_*.csv")}
        self.native_hashes = {name: file_sha256(path) for name, path in self.native_paths.items()}
        self.expected_hashes = {}
        for entry in original_manifest.get("feature_files", []):
            self.expected_hashes.update(entry.get("audit_files_sha256", {}))
        self.validated_keys = set()
        for artifact in self.bank["artifacts"]:
            key = artifact["applicability"]["source_key"]
            for basis in artifact["applicability"]["allowed_energy_bases"]:
                self.validated_keys.add(source_key({**key, "energy_coordinate_basis": basis}))
        self.encountered, self.processed, self.unsupported_rows = set(), set(), set()
        self.historical_reports, self.feature_reports = {}, {}
        self.memory = OrderedDict()
        self.config = {"adapter_version": VERSION, "calibration_bank_sha256": self.bank_sha,
                       "historical_raw_masks_sha256": self.masks["bundle_sha256"],
                       "native_audit_files": self.native_hashes,
                       "calibration_quantity": "legacy instrument-equivalent estimates; not native EPEAD observations",
                       "unsupported_policy": "quarantine a whole window if any native point has an unsupported source, energy grid or tested spectral range",
                       "whole_median_policy": "wholly median-filled source rows have no observed anchors and are unavailable"}

    def _remember_keys(self, mapped):
        keys = set(mapped.get("source_calibration_key", pd.Series(dtype=str)).dropna().astype(str)) - {""}
        self.encountered.update(keys)
        self.processed.update(keys & self.validated_keys)

    def _month(self, month):
        if month in self.memory:
            self.memory.move_to_end(month)
            return self.memory[month]
        name = f"native_sgps_{month}.csv"
        if name not in self.native_paths:
            raise ValueError(f"Missing {self.root / 'continuation_audit' / name}. Restore the native continuation audit from the completed enhanced archive; old uncalibrated proton values cannot substitute for it.")
        path = self.native_paths[name]
        expected = self.expected_hashes.get("continuation_audit/" + name)
        if not expected:
            raise ValueError("Native proton audit has no checksum in the enhanced manifest: " + name)
        if self.native_hashes[name] != expected or file_sha256(path) != expected:
            raise ValueError("Native proton audit checksum mismatch: " + name)
        signature = hashlib.sha256((VERSION+self.bank_sha+self.native_hashes[name]).encode()).hexdigest()
        cache = self.cache_dir / (month+"_"+signature[:20]+".csv.gz")
        record = cache.with_suffix(".json")
        mapped = None
        if cache.is_file() and record.is_file():
            saved = json.loads(record.read_text())
            if saved.get("signature") == signature and file_sha256(cache) == saved.get("sha256"):
                mapped = pd.read_csv(cache, index_col="Time", float_precision="round_trip")
                mapped.index = pd.to_datetime(mapped.index, utc=True, format="mixed")
        if mapped is None:
            native = pd.read_csv(path, low_memory=False, float_precision="round_trip")
            if "Time" not in native:
                raise ValueError("Native proton audit lacks Time: " + name)
            native.index = pd.to_datetime(native.pop("Time"), utc=True, format="mixed")
            if native.index.has_duplicates or native.index.isna().any():
                raise ValueError("Native proton timestamps are not unique and valid: " + name)
            if any(key not in native for key in KEY_FIELDS):
                raise ValueError("Native proton audit lacks source identity: " + name)
            present = native.source_product.astype(str).str.startswith("NOAA_SGPS_")
            mapped = pd.DataFrame(index=native.index)
            for feature in ("P4", "P5", "P6"):
                mapped[feature] = np.nan
                mapped[feature+"_calibration_status"] = "missing_or_invalid_native_spectrum"
            mapped["calibration_id"] = ""
            mapped["calibration_reference_kind"] = "unavailable"
            mapped["source_calibration_key"] = ""
            if present.any():
                estimates = apply_calibration_bank(native.loc[present], self.bank, strict_source=False)
                for column in estimates:
                    mapped.loc[present, column] = estimates[column]
                mapped.loc[present, "source_calibration_key"] = native.loc[present, list(KEY_FIELDS)].apply(source_key, axis=1)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            content = gzip.compress(mapped.to_csv(index_label="Time").encode(), mtime=0)
            part = cache.with_suffix(".part")
            part.write_bytes(content)
            part.replace(cache)
            record.write_text(json.dumps({"signature": signature, "sha256": file_sha256(cache)})+"\n")
        if mapped.index.has_duplicates:
            raise ValueError("Duplicate parsed calibration timestamps")
        self._remember_keys(mapped)
        self.memory[month] = mapped
        while len(self.memory) > 2:
            self.memory.popitem(last=False)
        return mapped

    def read_feature(self, entry, metadata_columns, rows):
        feature = entry["feature"]
        frame = _read_feature(self.root, entry, metadata_columns, rows)
        q = _load_quality(self.root/"quality"/f"{feature}_quality.csv", rows)
        q["pre_repair_point_status"] = q.point_status
        frame, q, historical = apply_historical_masks(frame, q, feature, self.masks)
        self.historical_reports[feature] = historical
        columns = feature_columns(feature)
        values = frame[columns].to_numpy(dtype=float)
        values[values == -99999.0] = -9999.0
        # Physical fluxes/magnitudes/densities/speeds/temperatures cannot be
        # negative; signed Vx and valid zero measurements are preserved.
        if feature != "Vx":
            values[values < 0] = -9999.0
        whole_median = q.point_status.eq("M"*288).to_numpy()
        values[whole_median] = -9999.0
        q.loc[whole_median, "point_status"] = "G"*288
        q["source_calibration_keys"] = "[]"
        q["calibration_ids"] = "[]"
        q["unsupported_calibration_row"] = False
        q["calibration_out_of_domain_points"] = 0
        onsets = pd.to_datetime(frame.FlrOnset, utc=True, format="mixed")
        eligible = np.flatnonzero(onsets.gt(LEGACY_END).to_numpy()) if feature in ("P4", "P5", "P6") else []
        for number, row in enumerate(eligible):
            onset = onsets.iloc[row]
            if number % 500 == 0:
                print(f"[REPAIR PROTON] {feature}: {number}/{len(eligible)}", flush=True)
            grid = window_index(onset)
            replace = grid >= LEGACY_END
            source = self._month(onset.strftime("%Y-%m")).reindex(grid)
            status_values = source[feature+"_calibration_status"].fillna("missing_or_invalid_native_spectrum")
            unsupported = status_values.isin(BAD_CALIBRATION).to_numpy() & replace
            if unsupported.any():
                values[row] = -9999.0
                q.loc[row, "point_status"] = "G"*288
                q.loc[row, "unsupported_calibration_row"] = True
                q.loc[row, "calibration_out_of_domain_points"] = int(unsupported.sum())
                self.unsupported_rows.add(int(row))
                continue
            candidate = source[feature].to_numpy(dtype=float)
            candidate[grid+pd.Timedelta(minutes=5) > onset] = np.nan
            letters = np.array(list(str(q.loc[row, "point_status"])), dtype="U1")
            letters[replace] = "G"
            values[row, replace] = -9999.0
            last_value, last_time, last_key, last_id = None, None, None, None
            used_keys, used_ids = set(), set()
            for j in range(288):
                if np.isfinite(candidate[j]) and candidate[j] >= 0:
                    last_value, last_time = candidate[j], grid[j]
                    last_key, last_id = source.iloc[j].source_calibration_key, source.iloc[j].calibration_id
                    if replace[j]:
                        values[row, j], letters[j] = candidate[j], "D"
                        used_keys.add(str(last_key))
                        used_ids.add(str(last_id))
                elif replace[j] and last_time is not None and grid[j]-last_time <= pd.Timedelta(minutes=10):
                    values[row, j], letters[j] = last_value, "H"
                    used_keys.add(str(last_key))
                    used_ids.add(str(last_id))
            q.loc[row, "source_calibration_keys"] = json.dumps(sorted(used_keys-{ "", "nan", "None" }))
            q.loc[row, "calibration_ids"] = json.dumps(sorted(used_ids-{ "", "nan", "None" }))
            q.loc[row, "point_status"] = "".join(letters)
            q.loc[row, "source_method"] = "validated_nonnegative_spectral_response_legacy_equivalent; bounded_causal_hold_10min"
        frame[columns] = values
        self.feature_reports[feature] = {"rows": rows, "modern_rows_recomputed": len(eligible),
                                         "whole_median_rows_unavailable": int(whole_median.sum()),
                                         "unsupported_windows": int(q.unsupported_calibration_row.sum())}
        return frame, q

    @property
    def reports(self):
        return {"historical_mask_report": {"status": "passed", "bundle_sha256": self.masks["bundle_sha256"],
                                           "features": self.historical_reports},
                "proton_calibration_report": {"status": "passed", "quantity": self.config["calibration_quantity"],
                    "calibration_bank_sha256": self.bank_sha, "encountered_native_keys": sorted(self.encountered),
                    "processed_native_keys": sorted(self.processed), "validated_native_keys": sorted(self.validated_keys),
                    "unsupported_rows_excluded": len(self.unsupported_rows), "features": self.feature_reports,
                    "limitations": "All retained modern proton anchors are tested source-scoped legacy-equivalent estimates, not native EPEAD measurements. Unsupported spectral windows are quarantined; genuine observation gaps may be interpolated later."}}

    def write_audits(self, outdir):
        outdir = Path(outdir)
        paths = []
        for name, report in self.reports.items():
            path = outdir/(name+".json")
            path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            paths.append(path.name)
        (outdir/"calibration_bank.json").write_bytes(self.bank_path.read_bytes())
        (outdir/"historical_raw_masks.json.gz").write_bytes(self.mask_path.read_bytes())
        (outdir/"historical_raw_masks.json.manifest.json").write_bytes(self.mask_path.with_suffix(".manifest.json").read_bytes())
        return paths + ["calibration_bank.json", "historical_raw_masks.json.gz", "historical_raw_masks.json.manifest.json"]

    def verify_sources(self):
        if file_sha256(self.bank_path) != self.bank_sha or file_sha256(self.mask_path) != self.masks["bundle_sha256"]:
            raise ValueError("Calibration or historical mask inputs changed during repair")
        for name, digest in self.native_hashes.items():
            if file_sha256(self.native_paths[name]) != digest:
                raise ValueError("Native proton source changed during repair: " + name)
