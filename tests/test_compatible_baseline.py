import hashlib

import numpy as np
import pandas as pd
import pytest
import requests

from sep_dataset import compatible_baseline as cb


class Response:
    def __init__(self, data=b"", status=200, content_range=None, failure=None):
        self.data, self.status_code, self.failure = data, status, failure
        self.headers = {"Content-Range": content_range} if content_range else {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size):
        yield self.data
        if self.failure:
            raise self.failure


@pytest.fixture
def download(tmp_path, monkeypatch):
    content = b"abcdefghij"
    spec = cb.BaselineSpec("Vx", "Vx.csv", len(content), hashlib.md5(content).hexdigest())
    target = tmp_path / "Vx.csv"
    monkeypatch.setattr(cb.time, "sleep", lambda *args: None)
    return spec, target, target.with_suffix(".csv.part"), content


def test_complete_partial_is_promoted_without_network(download, monkeypatch):
    spec, target, part, content = download
    part.write_bytes(content)
    monkeypatch.setattr(cb.requests, "get", lambda *a, **kw: pytest.fail("complete verified .part must be reused"))
    assert cb._download_baseline(spec, target) == target
    assert target.read_bytes() == content
    assert not part.exists()


@pytest.mark.parametrize("bad", [b"too many corrupt bytes", b"0123456789"])
def test_oversize_or_complete_corrupt_partial_restarts(download, monkeypatch, bad):
    spec, target, part, content = download
    part.write_bytes(bad)

    def get(*args, **kwargs):
        assert kwargs["headers"] == {}
        return Response(content)

    monkeypatch.setattr(cb.requests, "get", get)
    cb._download_baseline(spec, target)
    assert target.read_bytes() == content


def test_valid_partial_response_resumes(download, monkeypatch):
    spec, target, part, content = download
    part.write_bytes(content[:4])

    def get(*args, **kwargs):
        assert kwargs["headers"] == {"Range": "bytes=4-"}
        return Response(content[4:], 206, "bytes 4-9/10")

    monkeypatch.setattr(cb.requests, "get", get)
    cb._download_baseline(spec, target)
    assert target.read_bytes() == content


@pytest.mark.parametrize("first", [
    Response(status=416), Response(b"wrong", 206, "bytes 0-4/10"),
    Response(b"wrong", 206, "bytes 4-8/11"), Response(b"wrong", 206),
])
def test_rejected_or_invalid_range_discards_partial_before_retry(download, monkeypatch, first):
    spec, target, part, content = download
    part.write_bytes(content[:4])
    calls = []

    def get(*args, **kwargs):
        calls.append(kwargs["headers"])
        return first if len(calls) == 1 else Response(content)

    monkeypatch.setattr(cb.requests, "get", get)
    cb._download_baseline(spec, target)
    assert calls == [{"Range": "bytes=4-"}, {}]
    assert target.read_bytes() == content


def test_server_ignoring_range_replaces_instead_of_appending(download, monkeypatch):
    spec, target, part, content = download
    part.write_bytes(content[:4])
    monkeypatch.setattr(cb.requests, "get", lambda *a, **kw: Response(content))
    cb._download_baseline(spec, target)
    assert target.read_bytes() == content


@pytest.mark.parametrize("failure", [None, requests.ConnectionError("interrupted")])
def test_short_or_interrupted_download_resumes_saved_bytes(download, monkeypatch, failure):
    spec, target, part, content = download
    calls = []

    def get(*args, **kwargs):
        calls.append(kwargs["headers"])
        if len(calls) == 1:
            return Response(content[:4], failure=failure)
        return Response(content[4:], 206, "bytes 4-9/10")

    monkeypatch.setattr(cb.requests, "get", get)
    cb._download_baseline(spec, target)
    assert calls == [{}, {"Range": "bytes=4-"}]
    assert target.read_bytes() == content


def baseline_fixture():
    metadata = pd.DataFrame({
        "FlrOnset": ["1998-01-01 03:01:00", "1998-01-02 12:57:00"],
        "event_type": [0, 1], "Rise Time": [0.18333333333333332, 0.2],
        "xrsclass": ["C1.0", "M1.2"], "missing": [np.nan, np.nan],
    })
    windows = pd.DataFrame(np.arange(576, dtype=float).reshape(2, 288), columns=cb.window_columns("Vx"))
    return metadata, pd.concat([metadata, windows], axis=1)


def test_validation_accepts_reference_rounding_but_preserves_frozen_values(tmp_path):
    metadata, frozen = baseline_fixture()
    frozen.loc[0, "Rise Time"] = 0.1833333333333333
    frozen["Label"] = ["NSEP", "SEP"]
    path = tmp_path / "Vx.csv"
    frozen.to_csv(path, index=False)
    result = cb.load_baseline("Vx", path, metadata)
    assert result.loc[0, "Rise Time"] == 0.1833333333333333
    assert "Label" not in result
    assert len(result) == 2  # use supplied fixture metadata length


@pytest.mark.parametrize("positive_bucket", ["NSEP", "gt10", "gt30", "gt60", "gt100"])
def test_published_energy_bucket_does_not_replace_binary_event_type(tmp_path, positive_bucket):
    metadata, frozen = baseline_fixture()
    frozen["Label"] = ["NSEP", positive_bucket]
    path = tmp_path / "published.csv"
    frozen.to_csv(path, index=False)
    result = cb.load_baseline("Vx", path, metadata)
    assert result.event_type.tolist() == [0, 1]
    assert "Label" not in result


@pytest.mark.parametrize("negative_bucket", ["gt10", "unknown"])
def test_auxiliary_bucket_rejects_unknown_or_negative_high_energy_label(tmp_path, negative_bucket):
    metadata, frozen = baseline_fixture()
    frozen["Label"] = [negative_bucket, "gt10"]
    path = tmp_path / "published.csv"
    frozen.to_csv(path, index=False)
    with pytest.raises(ValueError, match="auxiliary baseline Label"):
        cb.load_baseline("Vx", path, metadata)


@pytest.mark.parametrize("column, value", [("Rise Time", 0.22), ("xrsclass", "X1.0"), ("missing", 1.0), ("event_type", 1)])
def test_validation_rejects_changes_in_all_historical_metadata(tmp_path, column, value):
    metadata, frozen = baseline_fixture()
    frozen.loc[0, column] = value
    path = tmp_path / "Vx.csv"
    frozen.to_csv(path, index=False)
    with pytest.raises(ValueError, match="historical"):
        cb.load_baseline("Vx", path, metadata)


def test_fingerprint_hashes_every_field_and_ignores_integer_float_promotion():
    _, frozen = baseline_fixture()
    expected = cb.baseline_fingerprint(frozen, "Vx")
    promoted = frozen.copy()
    promoted["event_type"] = promoted.event_type.astype(float)
    assert cb.baseline_fingerprint(promoted, "Vx") == expected
    for column, value in [("Rise Time", 0.22), ("xrsclass", "X1.0"), ("Vx_288", 999.0), ("missing", 1.0)]:
        changed = frozen.copy()
        changed.loc[0, column] = value
        assert cb.baseline_fingerprint(changed, "Vx") != expected


def test_fingerprint_survives_csv_roundtrip_and_object_numeric_promotion(tmp_path):
    _, frozen = baseline_fixture()
    path = tmp_path / "Vx.csv"
    frozen.to_csv(path, index=False)
    read = pd.read_csv(path, float_precision="round_trip")
    assert cb.baseline_fingerprint(read, "Vx") == cb.baseline_fingerprint(frozen, "Vx")
    read["event_type"] = read.event_type.astype(object)
    assert cb.baseline_fingerprint(read, "Vx") == cb.baseline_fingerprint(frozen, "Vx")
