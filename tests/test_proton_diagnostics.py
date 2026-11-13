import matplotlib
import numpy as np
import pandas as pd

from sep_dataset.proton_diagnostics import POINT_FRACTIONS, plot_proton_years, summarize_proton_years


def _windows():
    summary = pd.DataFrame({"event_type": [0, 0, 1, 1, 1], "filled_mean": [0., 4., 18., 26., np.nan],
                            "filled_valid_points": [288, 288, 288, 288, 0]})
    for code in POINT_FRACTIONS:
        summary["points_" + code] = 0
    summary.loc[0, ["points_U", "points_I"]] = [216, 72]
    summary.loc[1, "points_O"] = 288
    summary.loc[2, ["points_D", "points_H", "points_E"]] = [144, 72, 72]
    summary.loc[3, "points_D"] = 288
    summary.loc[4, "points_S"] = 288
    return summary


def test_yearly_proton_summary_keeps_zeros_missing_years_and_quality_fractions():
    table = summarize_proton_years("P4", [2019, 2019, 2021, 2021, 2021], _windows()).set_index(["year", "class"])
    nsep = table.loc[(2019, "NSEP")]
    assert nsep.valid_window_means == 2
    assert nsep.zero_window_means == 1
    assert nsep.median_window_mean == 2
    assert nsep.q25_window_mean == 1
    assert nsep.q75_window_mean == 3
    assert nsep.interpolated_fraction == .125
    assert nsep.historical_provenance_unknown_fraction == .375
    assert nsep.observed_fraction == .5
    missing = table.loc[(2020, "NSEP")]
    assert missing.sample_windows == 0
    assert np.isnan(missing.median_window_mean)
    sep = table.loc[(2021, "SEP")]
    assert sep.sample_windows == 3
    assert sep.valid_window_means == 2
    assert sep.mean_window_mean == 22
    assert sep.estimated_fraction == .5
    assert sep.held_fraction == 1 / 12
    assert sep.structurally_unavailable_fraction == 1 / 3


def test_incomplete_window_cannot_contribute_a_24_hour_mean():
    summary = _windows()
    summary.loc[1, "filled_valid_points"] = 287
    table = summarize_proton_years("P4", [2019] * len(summary), summary)
    assert table.iloc[0].sample_windows == 2
    assert table.iloc[0].valid_window_means == 1
    assert table.iloc[0].mean_window_mean == 0


def test_plot_preserves_zero_and_marks_instrument_boundary(tmp_path):
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    tables = [summarize_proton_years(feature, [2019, 2020, 2021, 2021, 2021], _windows())
              for feature in ("P4", "P5", "P6")]
    figure = plot_proton_years(pd.concat(tables), plt)
    assert [ax.get_title(loc="left") for ax in figure.axes] == ["P4", "P5", "P6"]
    assert all(ax.get_yscale() == "symlog" for ax in figure.axes)
    assert figure.axes[0].lines[0].get_ydata()[0] == 0
    assert 2020 < figure.axes[0].lines[-1].get_xdata()[0] < 2021
    path = tmp_path / "proton.pdf"
    figure.savefig(path)
    plt.close(figure)
    assert path.read_bytes().startswith(b"%PDF-")
