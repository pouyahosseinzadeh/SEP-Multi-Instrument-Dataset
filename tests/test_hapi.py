from types import SimpleNamespace

import pandas as pd
import pytest
import requests

from sep_dataset.hapi import DEFAULT_HAPI_SPECS, HapiClient, HapiDatasetSpec, HapiMonthlyStore, SelectedParameter


INFO = {
    "parameters": [
        {"name": "Time", "type": "isotime"},
        {"name": "flow_speed", "type": "double", "fill": "99999.9"},
        {"name": "Vx", "type": "double", "fill": "99999.9"},
    ]
}


def test_parameter_selection_preserves_server_order_and_feature_mapping():
    selected = HapiClient.select_parameters(
        INFO, {"Vx_GSE": ["Vx"], "FlowSpeed": ["flow_speed"]}
    )
    assert [(item.feature, item.parameter) for item in selected] == [
        ("FlowSpeed", "flow_speed"),
        ("Vx_GSE", "Vx"),
    ]


def test_omni_bmag_selects_field_magnitude_instead_of_observation_interval():
    info = {"parameters": [
        {"name": "Time", "type": "isotime"},
        {"name": "Time_btwn_obs", "type": "double", "description": "Time between observations (seconds)"},
        {"name": "F", "type": "double", "description": "Magnitude of avg. field vector (nT)"},
    ]}
    selected = HapiClient.select_parameters(
        info, {"Bmag": DEFAULT_HAPI_SPECS[0].feature_aliases["Bmag"]}
    )
    assert [(item.feature, item.parameter) for item in selected] == [("Bmag", "F")]


def test_data_requests_and_labels_columns_in_server_order(tmp_path, monkeypatch):
    def get(url, params, timeout):
        assert params["parameters"] == "flow_speed,Vx"
        return SimpleNamespace(
            text="2024-05-10T00:00:00Z,450.0,-449.0\n",
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(requests, "get", get)
    client = HapiClient(cache_dir=tmp_path)
    spec = HapiDatasetSpec("test", "https://example.test/hapi", "test", {})
    selected = [SelectedParameter("Vx_GSE", "Vx"), SelectedParameter("FlowSpeed", "flow_speed")]
    df = client.data(
        spec, selected, pd.Timestamp("2024-05-10T00:00:00Z"),
        pd.Timestamp("2024-05-10T00:05:00Z"), info=INFO,
    )
    assert df["flow_speed"].iloc[0] == 450.0
    assert df["Vx"].iloc[0] == -449.0
    assert len(list((tmp_path / "data").glob("*.csv"))) == 1


@pytest.mark.parametrize(
    "body, error",
    [
        (',\n"status": {"code": 1411, "message": "HAPI error 1411: Parameter out of order"}\n}\n', "Parameter out of order"),
        ('{"status":{"code":1400,"message":"HAPI error"}}', "HAPI error"),
        ("not-a-time,12\n", "invalid timestamps"),
        ("2024-05-10T00:00:00Z,12,13\n", "expected 2"),
    ],
)
def test_invalid_data_response_is_never_cached(tmp_path, monkeypatch, body, error):
    monkeypatch.setattr(
        requests, "get", lambda *args, **kwargs: SimpleNamespace(
            text=body, raise_for_status=lambda: None
        ),
    )
    client = HapiClient(cache_dir=tmp_path)
    spec = HapiDatasetSpec("test", "https://example.test/hapi", "test", {})
    with pytest.raises(RuntimeError, match=error):
        client.data(
            spec, [SelectedParameter("FlowSpeed", "flow_speed")],
            pd.Timestamp("2024-05-10T00:00:00Z"),
            pd.Timestamp("2024-05-10T00:05:00Z"), info=INFO,
        )
    assert not list((tmp_path / "data").glob("*.csv"))


@pytest.mark.parametrize("month", ["1998-01-01", "2025-01-01"])
def test_month_outside_metadata_coverage_is_not_requested(tmp_path, monkeypatch, capsys, month):
    client = HapiClient(cache_dir=tmp_path)

    def unexpected_data(*args, **kwargs):
        pytest.fail("Months completely outside metadata coverage must not be downloaded")

    monkeypatch.setattr(client, "data", unexpected_data)
    store = HapiMonthlyStore(
        client, HapiDatasetSpec("test", "https://example.test/hapi", "test", {}), [],
        {**INFO, "startDate": "2010-04-14T00:00:00Z", "stopDate": "2024-12-31T23:55:00Z"},
    )
    assert store.get_month(pd.Timestamp(month, tz="UTC")).empty
    assert not capsys.readouterr().out


def test_month_at_metadata_end_still_requests_available_boundary_sample(tmp_path, monkeypatch):
    client = HapiClient(cache_dir=tmp_path)
    calls = []

    def data(*args, **kwargs):
        calls.append(args)
        return pd.DataFrame({"Time": [pd.Timestamp("2024-02-01T00:00:00Z")], "flow_speed": [400.0]})

    monkeypatch.setattr(client, "data", data)
    store = HapiMonthlyStore(
        client, HapiDatasetSpec("test", "https://example.test/hapi", "test", {}), [],
        {**INFO, "startDate": "2024-01-01T00:00:00Z", "stopDate": "2024-02-01T00:00:00Z"},
    )
    assert len(store.get_month(pd.Timestamp("2024-02-01T00:00:00Z"))) == 1
    assert len(calls) == 1


def test_failed_month_inside_metadata_coverage_still_warns(tmp_path, monkeypatch, capsys):
    client = HapiClient(cache_dir=tmp_path)

    def data(*args, **kwargs):
        raise RuntimeError("temporary source failure")

    monkeypatch.setattr(client, "data", data)
    store = HapiMonthlyStore(
        client, HapiDatasetSpec("test", "https://example.test/hapi", "test", {}), [],
        {**INFO, "startDate": "2024-01-01T00:00:00Z", "stopDate": "2024-12-31T23:55:00Z"},
    )
    assert store.get_month(pd.Timestamp("2024-05-01T00:00:00Z")).empty
    assert "temporary source failure" in capsys.readouterr().out
