import pandas as pd

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
