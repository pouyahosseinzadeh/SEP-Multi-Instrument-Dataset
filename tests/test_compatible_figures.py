"""Scientific compatibility checks for the original per-feature workflow."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from sep_dataset.compatible_figures import (
    BAND_SELECTION, adjust_pvalues_bh, band_masks, full_window_pca,
    mean_ci, profile_statistics, run_compatible_figures, timeseries_columns,
)


class CompatibleFigureTests(unittest.TestCase):
    def test_original_energy_cuts_use_correct_peak_columns(self):
        # A >10-MeV NOAA event is not automatically a >100-MeV event.
        frame = pd.DataFrame({"event_type": [0, 1, 1, 1],
                              "sep_peak_2": [999, 10, 9.99, 100],
                              "sep_peak_3": [999, 4.99, 5, np.nan],
                              "sep_peak_4": [999, 5, 0.1, 4],
                              "sep_peak_5": [999, np.nan, 5, 4.99]})
        expected = {10: [False, True, False, True], 30: [False, False, True, False],
                    60: [False, True, False, False], 100: [False, False, True, False]}
        for band in BAND_SELECTION:
            sep, nsep = band_masks(frame, band)
            np.testing.assert_array_equal(sep, expected[band])
            np.testing.assert_array_equal(nsep, [True, False, False, False])

    def test_original_normal_ci_and_welch_test_are_preserved(self):
        rng = np.random.RandomState(5)
        sep, nsep = rng.normal(size=(8, 288)), rng.normal(0.5, 2, size=(15, 288))
        stats = profile_statistics(sep, nsep, "Vx", 10)
        np.testing.assert_allclose(stats.sep_mean, sep.mean(axis=0))
        np.testing.assert_allclose(stats.sep_ci95_halfwidth, 1.959963984540054 * sep.std(axis=0, ddof=1)/np.sqrt(8))
        np.testing.assert_allclose(stats.p_value, ttest_ind(sep, nsep, axis=0, equal_var=False).pvalue)
        self.assertEqual(stats.hours_before_aligned_flare.iloc[-1], 5/60)
        self.assertEqual(stats.legacy_display_hours.iloc[-1], 0)
        self.assertEqual(stats.column.iloc[0], "Vx_288")
        self.assertEqual(stats.column.iloc[-1], "Vx_1")

    def test_bh_correction_has_known_values_and_handles_unavailable_tests(self):
        result = adjust_pvalues_bh(np.array([0.01, 0.04, 0.03, np.nan, 0.9]))
        np.testing.assert_allclose(result, [0.04, 0.05333333333333334, 0.05333333333333334, np.nan, 0.9], equal_nan=True)

    def test_pca_matches_full_window_standardized_svd(self):
        rng = np.random.RandomState(9)
        # Strongly unequal units per column expose any missing standardization.
        x = rng.normal(size=(35, 288))*np.linspace(1, 500, 288)
        model = full_window_pca(x, n_components=8)
        z = (x-x.mean(axis=0))/x.std(axis=0)
        _, s, components = np.linalg.svd(z, full_matrices=False)
        np.testing.assert_allclose(model["explained_variance_ratio"], s[:8]**2/np.sum(s**2), rtol=1e-11)
        np.testing.assert_allclose(np.abs(model["components"]), np.abs(components[:8]), atol=1e-11)
        self.assertEqual(model["components"].shape, (8, 288))
        self.assertEqual(model["scores"].shape, (35, 8))

    def test_no_extra_imputation_occurs_during_pca(self):
        x = np.arange(6*288, dtype=float).reshape(6, 288)
        x[0, 0] = np.nan
        model = full_window_pca(x)
        self.assertFalse(model["retained_rows"][0])
        self.assertEqual(model["scores"].shape[0], 5)

    def test_structurally_missing_channel_is_not_fabricated_in_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = pd.DataFrame({"FlrOnset": pd.date_range("2024-01-01", periods=4, freq="D").astype(str),
                                  "event_type": [0, 0, 1, 1], "sep_peak_2": [0, 0, 12, 12]})
            frame = pd.concat([frame, pd.DataFrame(-9999., index=range(4), columns=timeseries_columns("P4"))], axis=1)
            frame.to_csv(root / "1998_2025_MEMSEP_dataset_with_P4_filled.csv", index=False)
            result = run_compatible_figures(root, root / "figures", bands=(10,), features=("P4",))
            manifest = json.loads(Path(result["manifest"]).read_text())
            info = manifest["features"]["P4"]["bands"]["10"]
            self.assertEqual(info["sep_without_feature_values"], 2)
            self.assertIn("comparison_skipped", info)
            self.assertFalse((root / "figures" / "figs_gt10" / "P4_pca_scores.csv").exists())

    def test_mean_ci_single_observation_has_no_spurious_precision(self):
        mean, ci, n = mean_ci(np.array([[1, np.nan], [np.nan, np.nan]]))
        self.assertEqual(mean[0], 1)
        self.assertTrue(np.isnan(ci).all())
        np.testing.assert_array_equal(n, [1, 0])

    def test_complete_single_feature_workflow_and_missingness_audit(self):
        rng = np.random.RandomState(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "data"
            dataset.mkdir()
            # Preserved historical naive timestamps coexist with new UTC ISO
            # timestamps. Every plotting date parser must accept both formats.
            mixed_onsets = list(pd.date_range("2013-01-01", periods=8, freq="D").astype(str))
            mixed_onsets += list(pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC").astype(str))
            frame = pd.DataFrame({"FlrOnset": mixed_onsets,
                                  "event_type": [0]*8+[1]*4, "sep_peak_2": [0]*8+[12]*4})
            frame = pd.concat([frame, pd.DataFrame(rng.normal(size=(12, 288)), columns=timeseries_columns("Vx"))], axis=1)
            frame.to_csv(dataset / "1998_2025_MEMSEP_dataset_with_Vx_filled.csv", index=False)
            pd.DataFrame({"feature": ["Vx"], "missing_fraction_before": [0.5], "missing_fraction_after": [0.0]}).to_csv(dataset / "quality_summary.csv", index=False)
            (dataset / "quality").mkdir()
            pd.DataFrame({"row_index": np.arange(12), "sample_id": ["sample{}".format(i) for i in range(12)],
                          "origin": ["baseline"]*8+["extension"]*4, "unavailable_channel": [False]*12,
                          "observed_points": [np.nan]*8+[144]*4, "median_filled_points": [np.nan]*8+[0]*4}).to_csv(dataset / "quality" / "Vx_quality.csv", index=False)
            result = run_compatible_figures(dataset, root / "figures", bands=(10,), features=("Vx",))
            manifest = json.loads(Path(result["manifest"]).read_text())
            self.assertEqual(manifest["features"]["Vx"]["bands"]["10"]["sep_samples"], 4)
            self.assertTrue(manifest["quality"]["prefill_quality_available"])
            self.assertTrue((root / "figures" / "figs_gt10" / "curves_Vx_gt10.png").is_file())
            self.assertTrue((root / "figures" / "figs_gt10" / "Vx_pca_loadings_heatmap.pdf").is_file())
            quality = pd.read_csv(root / "figures" / "feature_quality_summary.csv")
            self.assertEqual(quality.prefill_rows_without_audit.iloc[0], 8)
            self.assertEqual(quality.audited_missing_fraction_before.iloc[0], 0.5)
            self.assertEqual(len(pd.read_csv(root / "figures" / "figs_gt10" / "Vx_profile_statistics.csv")), 288)
            scores = pd.read_csv(root / "figures" / "figs_gt10" / "Vx_pca_scores.csv")
            self.assertEqual(scores.sample_id.iloc[0], "sample0")
            self.assertEqual(scores.origin.iloc[-1], "extension")
            self.assertTrue((root / "figures" / "feature_quality_by_origin.csv").exists())


if __name__ == "__main__":
    unittest.main()
