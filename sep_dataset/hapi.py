from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .utils import cache_key, clean_numeric_series, ensure_dir, isoformat_z, iter_month_starts, next_month, normalize_name


@dataclass
class HapiDatasetSpec:
    name: str
    server: str
    dataset: str
    feature_aliases: dict[str, list[str]]
    cadence: str = "5min"
    enabled: bool = True
    notes: str = ""


DEFAULT_HAPI_SPECS = [
    HapiDatasetSpec(
        name="omni_hro2_5min",
        server="https://cdaweb.gsfc.nasa.gov/hapi",
        dataset="OMNI_HRO2_5MIN",
        cadence="5min",
        notes="Near-Earth OMNI 5-minute solar wind, IMF, indices, and GOES proton flux where available.",
        feature_aliases={
            "Bmag": ["F", "ABS_B", "Magnitude", "B_TOT", "Bt", "IMF Magnitude Avg"],
            "Bx_GSE": ["BX_GSE", "Bx", "BX", "Bx GSE"],
            "By_GSM": ["BY_GSM", "By GSM", "BY"],
            "Bz_GSM": ["BZ_GSM", "Bz GSM", "BZ"],
            "Vx_GSE": ["VX_GSE", "Vx", "V_X_GSE", "Vx Velocity"],
            "FlowSpeed": ["flow_speed", "Speed", "V", "Vsw", "Flow Speed"],
            "ProtonDensity": ["proton_density", "Np", "Density", "Proton Density"],
            "ProtonTemp": ["T", "Temperature", "Proton Temperature"],
            "FlowPressure": ["Pressure", "Flow Pressure"],
            "Ey": ["E", "Ey", "Electric Field"],
            "PlasmaBeta": ["Beta", "Plasma Beta"],
            "AlfvenMach": ["Mach", "Alfven Mach"],
            "SYM_H": ["SYM_H", "SYM/H", "SYM-H"],
            "ProtonFlux_GT10": ["PR-FLX_10", "P10", "Proton Flux >10", "Proton_flux_>10_Mev"],
            "ProtonFlux_GT30": ["PR-FLX_30", "P30", "Proton Flux >30", "Proton_flux_>30_Mev"],
            "ProtonFlux_GT60": ["PR-FLX_60", "P60", "Proton Flux >60", "Proton_flux_>60_Mev"],
        },
    ),
    HapiDatasetSpec(
        name="iswa_goes_p10_5min",
        server="https://iswa.ccmc.gsfc.nasa.gov/hapi",
        dataset="goesp_part_flux_P5M",
        cadence="5min",
        notes="ISWA GOES integral proton flux. CDAW uses this after OMNIWeb GOES flux ends in 2020.",
        feature_aliases={"GOES_P10": ["P10", "proton_flux_gt10", ">=10 MeV", ">10 MeV"]},
    ),
]


@dataclass
class SelectedParameter:
    feature: str
    parameter: str
    fill: object | None = None
    units: str = ""


class HapiClient:
    def __init__(self, cache_dir: str | Path = "data/cache/hapi", timeout: int = 120):
        self.cache_dir = ensure_dir(cache_dir)
        self.timeout = timeout

    def _request_json(self, server: str, endpoint: str, params: dict[str, object]) -> dict:
        url = server.rstrip("/") + "/" + endpoint.lstrip("/")
        key = cache_key(url, sorted(params.items()))
        cache_path = self.cache_dir / "metadata" / f"{key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        import requests

        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        ensure_dir(cache_path.parent)
        cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def info(self, spec: HapiDatasetSpec) -> dict:
        errors: list[str] = []
        for params in ({"id": spec.dataset}, {"dataset": spec.dataset}):
            try:
                payload = self._request_json(spec.server, "info", params)
                if "parameters" in payload:
                    return payload
                errors.append(str(payload.get("status", payload))[:200])
            except Exception as exc:  # pragma: no cover - live network only
                errors.append(str(exc))
        raise RuntimeError(f"HAPI info failed for {spec.name}/{spec.dataset}: {'; '.join(errors)}")

    @staticmethod
    def time_parameter(info: dict) -> str:
        for param in info.get("parameters", []):
            if str(param.get("type", "")).lower() == "isotime":
                return str(param["name"])
        return str(info.get("parameters", [{}])[0].get("name", "Time"))

    @staticmethod
    def select_parameters(info: dict, feature_aliases: dict[str, list[str]]) -> list[SelectedParameter]:
        params = [p for p in info.get("parameters", []) if str(p.get("type", "")).lower() != "isotime"]
        selected: list[SelectedParameter] = []
        used: set[str] = set()
        indexed = []
        for param in params:
            text = " ".join(str(param.get(k, "")) for k in ["name", "label", "description", "units"])
            indexed.append((param, normalize_name(param.get("name", "")), normalize_name(text)))

        for feature, aliases in feature_aliases.items():
            alias_norm = [normalize_name(alias) for alias in aliases]
            chosen = None
            for param, name_norm, text_norm in indexed:
                if param.get("name") in used:
                    continue
                if any(name_norm == alias for alias in alias_norm):
                    chosen = param
                    break
            if chosen is None:
                for param, name_norm, text_norm in indexed:
                    if param.get("name") in used:
                        continue
                    if any(alias and alias in text_norm for alias in alias_norm):
                        chosen = param
                        break
            if chosen is not None:
                used.add(str(chosen.get("name")))
                selected.append(
                    SelectedParameter(
                        feature=feature,
                        parameter=str(chosen.get("name")),
                        fill=chosen.get("fill"),
                        units=str(chosen.get("units", "")),
                    )
                )
        # HAPI servers may require the subset in the same order as /info.
        # Feature configuration order is independent of the server column order.
        parameter_order = {str(param.get("name")): i for i, param in enumerate(info.get("parameters", []))}
        return sorted(selected, key=lambda item: parameter_order[item.parameter])

    def data(
        self,
        spec: HapiDatasetSpec,
        selected: Iterable[SelectedParameter],
        start: pd.Timestamp,
        stop: pd.Timestamp,
        info: dict | None = None,
    ) -> pd.DataFrame:
        selected = list(selected)
        if not selected:
            return pd.DataFrame()
        info = info or self.info(spec)
        parameter_order = {str(param.get("name")): i for i, param in enumerate(info.get("parameters", []))}
        selected.sort(key=lambda item: parameter_order[item.parameter])
        time_col = self.time_parameter(info)
        params = ",".join(item.parameter for item in selected)
        cache_path = self.cache_dir / "data" / f"{cache_key(spec.server, spec.dataset, params, isoformat_z(start), isoformat_z(stop))}.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path, parse_dates=[time_col])

        query_options = [
            {
                "id": spec.dataset,
                "parameters": params,
                "time.min": isoformat_z(start),
                "time.max": isoformat_z(stop),
                "format": "csv",
            },
            {
                "dataset": spec.dataset,
                "parameters": params,
                "start": isoformat_z(start),
                "stop": isoformat_z(stop),
                "format": "csv",
            },
        ]
        errors: list[str] = []
        for query in query_options:
            try:
                import requests

                response = requests.get(spec.server.rstrip("/") + "/data", params=query, timeout=self.timeout)
                response.raise_for_status()
                text = response.text
                # CDAWeb has also returned malformed JSON errors beginning
                # with a comma and a "status" key while using HTTP 200.
                content_start = text.lstrip("\ufeff \t\r\n,")
                if content_start.startswith(("{", "[", "<", '"status"')):
                    errors.append(text[:300])
                    continue
                names = [time_col] + [item.parameter for item in selected]
                from io import StringIO

                if not text.strip():
                    return pd.DataFrame(columns=names)
                df = pd.read_csv(StringIO(text), comment="#", header=None)
                if df.empty:
                    return df
                if df.shape[1] != len(names):
                    raise ValueError(f"HAPI CSV has {df.shape[1]} columns; expected {len(names)}")
                df.columns = names
                df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
                if df[time_col].isna().any():
                    raise ValueError("HAPI CSV contains invalid timestamps")
                for item in selected:
                    df[item.parameter] = clean_numeric_series(df[item.parameter], fill_values=[item.fill])
                ensure_dir(cache_path.parent)
                df.to_csv(cache_path, index=False)
                return df
            except Exception as exc:  # pragma: no cover - live network only
                errors.append(str(exc))
        raise RuntimeError(f"HAPI data failed for {spec.name} {isoformat_z(start)}..{isoformat_z(stop)}: {'; '.join(errors)}")


@dataclass
class HapiMonthlyStore:
    client: HapiClient
    spec: HapiDatasetSpec
    selected: list[SelectedParameter]
    info: dict
    _month_cache: dict[str, pd.DataFrame] = field(default_factory=dict)

    def get_month(self, month: pd.Timestamp) -> pd.DataFrame:
        start = pd.Timestamp(year=month.year, month=month.month, day=1, tz="UTC")
        key = start.strftime("%Y-%m")
        if key in self._month_cache:
            return self._month_cache[key]
        stop = next_month(start)
        coverage_start = pd.to_datetime(self.info.get("startDate"), utc=True, errors="coerce")
        coverage_stop = pd.to_datetime(self.info.get("stopDate"), utc=True, errors="coerce")
        if ((pd.notna(coverage_start) and stop <= coverage_start)
                or (pd.notna(coverage_stop) and start > coverage_stop)):
            self._month_cache[key] = pd.DataFrame()
            return self._month_cache[key]
        try:
            df = self.client.data(self.spec, self.selected, start, stop, info=self.info)
        except Exception as exc:
            print(f"[WARN] HAPI month failed for {self.spec.name} {key}: {exc}")
            df = pd.DataFrame()
        time_col = self.client.time_parameter(self.info)
        if not df.empty:
            df = df.set_index(time_col).sort_index()
            df = df[~df.index.duplicated(keep="last")]
        self._month_cache[key] = df
        return df

    def range(self, start: pd.Timestamp, stop: pd.Timestamp) -> pd.DataFrame:
        frames = [self.get_month(month) for month in iter_month_starts(start, stop)]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, axis=0).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df[(df.index >= start) & (df.index < stop)]
