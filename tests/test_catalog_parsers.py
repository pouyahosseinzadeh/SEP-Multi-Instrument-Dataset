import pandas as pd
import pytest

from sep_dataset import catalogs
from sep_dataset.catalogs import merge_event_catalogs, parse_cdaw_major_sep_events, parse_noaa_spe_events


def test_parse_noaa_spe_html_like_row():
    text = """
    <tr><td>2024</td></tr>
    <tr><td>May 10/1350</td><td>May 11/0245</td><td>207</td><td></td><td></td><td>May 10/0640</td><td>X5.8/--</td><td>N22E65</td><td>13664</td></tr>
    """
    df = parse_noaa_spe_events(text, source_url="test")
    assert len(df) == 1
    assert df.loc[0, "start_time"] == pd.Timestamp("2024-05-10T13:50:00Z")
    assert df.loc[0, "peak_pfu_gt10"] == 207
    assert df.loc[0, "flare_class"] == "X5.8/--"
    assert df.loc[0, "flare_location"] == "N22E65"
    assert df.loc[0, "active_region"] == "13664"


def test_parse_noaa_legacy_implicit_cells_and_multiline_rows():
    # NASA's legacy mirror omits closing TD and TR tags, including empty cells.
    text = """
    <table><tr><td><td><td><td><strong>2001</strong><td><td><td><td><td>
    <tr><td>June 15/1750<td>June 16/0005<td align=right>26<td><td>W /15 1648
      <td>June 15/1600<td>M6/1N<td>S08W61<td>9500
    <tr><td>Aug 10/1020<td>Aug 10/1155<td>17<td><td><td><td>none<td>Disk Center<td>
    </table>
    """
    df = parse_noaa_spe_events(text)
    assert len(df) == 2
    assert df.loc[0, "start_time"] == pd.Timestamp("2001-06-15T17:50Z")
    assert df.loc[0, "flare_time"] == pd.Timestamp("2001-06-15T16:00Z")
    assert df.loc[0, "flare_class"] == "M6/1N"
    assert df.loc[1, "flare_class"] == "none"
    assert df.loc[1, "active_region"] == ""


def test_parse_noaa_fixed_width_pre_preserves_separators():
    text = """<pre>
    2024
    May 10/1350  May 11/0245  1,207  --  Halo  May 10/0640  X5.8  N22E65  13664
    </pre>"""
    df = parse_noaa_spe_events(text)
    assert len(df) == 1
    assert df.loc[0, "peak_pfu_gt10"] == 1207
    assert df.loc[0, "active_region"] == "13664"


def test_parse_noaa_current_ncei_table_and_previous_year_flare():
    text = """
    <table><tr>
    <td>2024<br>01/03 2005</td><td>2024 01/04 0835</td><td>20</td>
    <td>13536</td><td>N04W90</td><td>X5/3B 12/31 2203</td>
    <td>Yes</td><td>Yes</td><td>N/A</td><td>N/A</td></tr></table>
    """
    df = parse_noaa_spe_events(text, source_url="current-noaa")
    assert len(df) == 1
    assert df.loc[0, "start_time"] == pd.Timestamp("2024-01-03T20:05Z")
    assert df.loc[0, "max_time"] == pd.Timestamp("2024-01-04T08:35Z")
    assert df.loc[0, "flare_time"] == pd.Timestamp("2023-12-31T22:03Z")
    assert df.loc[0, "flare_class"] == "X5/3B"
    assert df.loc[0, "active_region"] == "13536"
    assert df.loc[0, "source_url"] == "current-noaa"


def test_parse_noaa_legacy_maximum_rolls_into_next_year():
    df = parse_noaa_spe_events("2023\nDec 31/2300  Jan 01/0100  20")
    assert df.loc[0, "max_time"] == pd.Timestamp("2024-01-01T01:00Z")


def test_parse_noaa_keeps_event_with_invalid_optional_times():
    text = """<tr><td>2024 02/09 1530</td><td>2024 02/30 9999</td><td>187</td>
    <td>13575</td><td>Beyond SW limb</td><td>X3.3 02/31 9999</td></tr>"""
    df = parse_noaa_spe_events(text)
    assert len(df) == 1
    assert pd.isna(df.loc[0, "max_time"])
    assert pd.isna(df.loc[0, "flare_time"])


@pytest.mark.parametrize("start, flare, expected", [
    ("Jul 14/0900", "14/0209", "2017-07-14T02:09Z"),
    ("Sep 05/0000", "04/2033", "2017-09-04T20:33Z"),
    ("Jan 01/0300", "31/2300", "2016-12-31T23:00Z"),
])
def test_noaa_legacy_flare_date_with_omitted_month(start, flare, expected):
    text = f"<tr><td>2017</td></tr><tr><td>{start}<td>--<td>20<td><td><td>{flare}<td>M1<td>S10W30<td>12667"
    df = parse_noaa_spe_events(text)
    assert df.loc[0, "flare_time"] == pd.Timestamp(expected)


def test_noaa_mirror_failover(monkeypatch, tmp_path):
    monkeypatch.setattr(catalogs, "NOAA_SPE_URLS", ["timeout", "working"])
    calls = []

    def download(url, **kwargs):
        calls.append(url)
        if url == "timeout":
            raise RuntimeError("connection timeout")
        return "2024\nMay 10/1350  May 11/0245  207"

    monkeypatch.setattr(catalogs, "http_get_text", download)
    df = catalogs.fetch_noaa_spe_events(tmp_path)
    assert calls == ["timeout", "working"]
    assert df.loc[0, "source_url"] == "working"


def test_noaa_refreshes_cached_error_page(monkeypatch, tmp_path):
    monkeypatch.setattr(catalogs, "NOAA_SPE_URLS", ["current"])
    cache = tmp_path / f"noaa_spe_{catalogs.cache_key('current')}.txt"
    cache.write_text("<html>maintenance</html>")

    def download(url, cache_path=None, **kwargs):
        return cache.read_text() if cache_path else "2024\nMay 10/1350  May 11/0245  207"

    monkeypatch.setattr(catalogs, "http_get_text", download)
    assert len(catalogs.fetch_noaa_spe_events(tmp_path)) == 1
    assert "May 10/1350" in cache.read_text()


def test_noaa_all_mirrors_unavailable_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(catalogs, "NOAA_SPE_URLS", ["broken"])
    monkeypatch.setattr(catalogs, "http_get_text", lambda *args, **kwargs: "Maintenance")
    with pytest.raises(RuntimeError, match="NOAA SPE catalog unavailable.*no parseable rows"):
        catalogs.fetch_noaa_spe_events(tmp_path)


def test_donki_splits_ranges_and_accepts_quiet_intervals(monkeypatch, tmp_path):
    monkeypatch.setattr(catalogs, "DONKI_ENDPOINTS", ["current-donki"])
    calls = []

    def download(url, params, cache_path):
        calls.append(params.copy())
        start = pd.Timestamp(params["startDate"])
        end = pd.Timestamp(params["endDate"])
        assert 0 <= (end - start).days <= 59
        if len(calls) == 1:
            return []
        return [{"sepID": "test-sep", "eventTime": "2024-03-01T12:00Z"}]

    monkeypatch.setattr(catalogs, "_download_json", download)
    df = catalogs.fetch_donki_sep_events("2024-01-01", "2024-05-01", tmp_path)
    assert calls == [
        {"startDate": "2024-01-01", "endDate": "2024-02-29"},
        {"startDate": "2024-03-01", "endDate": "2024-04-29"},
        {"startDate": "2024-04-30", "endDate": "2024-05-01"},
    ]
    assert len(df) == 1  # Repeat event IDs from overlapping source responses deduplicate.


def test_donki_empty_is_success_and_error_payload_is_not(monkeypatch, tmp_path):
    monkeypatch.setattr(catalogs, "DONKI_ENDPOINTS", ["current"])
    monkeypatch.setattr(catalogs, "_download_json", lambda *args: [])
    assert catalogs.fetch_donki_sep_events("2024-01-01", "2024-01-02", tmp_path).empty
    monkeypatch.setattr(catalogs, "_download_json", lambda *args: {"error": "rate limit"})
    with pytest.raises(RuntimeError, match="rate limit"):
        catalogs.fetch_donki_sep_events("2024-01-01", "2024-01-02", tmp_path)


def test_parse_cdaw_major_sep_line():
    text = """
    2024/10/09 02:45  2024/10/10 15:15   1812  RELAT  10/10 03:25   1152  --   2024/10/09 02:12  1435     H    N13W08  10/09 01:25  10/09 01:56  X1.8/--   10/09 01:44  10/09 01:54   SOHO shock
    """
    df = parse_cdaw_major_sep_events(text)
    assert len(df) == 1
    assert df.loc[0, "start_time"] == pd.Timestamp("2024-10-09T02:45:00Z")
    assert df.loc[0, "peak_pfu_gt10"] == 1812
    assert df.loc[0, "cme_width_deg"] == 360


def test_merge_catalogs_deduplicates_close_starts():
    a = parse_cdaw_major_sep_events("2024/10/09 02:45  2024/10/10 15:15 1812 RELAT 10/10 03:25 1152 --")
    b = parse_noaa_spe_events("<tr><td>2024</td></tr><tr><td>Oct 09/0300</td><td>Oct 10/1515</td><td>1800</td></tr>")
    merged = merge_event_catalogs([a, b], tolerance_hours=6)
    assert len(merged) == 1
    assert "CDAW_MAJOR_SEP" in merged.loc[0, "source"]
    assert "NOAA_SWPC_SPE" in merged.loc[0, "source"]

