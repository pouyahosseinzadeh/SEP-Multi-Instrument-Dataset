import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sep_dataset.paper_figures import (
    LOG_FEATURES, _label_association, load_feature, pairwise_correlations,
    run_paper_figures, transform_summaries,
)


class PaperFigureTests(unittest.TestCase):
    def test_pairwise_counts_use_window_rows_and_exclude_missing_pairs(self):
        frame = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [2, 4, np.nan, 8, 10, 12]})
        rows = pairwise_correlations(frame, min_pairs=5)
        pair = rows[rows.feature_a.ne(rows.feature_b)].iloc[0]
        self.assertEqual(pair.n_pairwise, 5)
        self.assertAlmostEqual(pair.pearson_r, 1.)
        self.assertAlmostEqual(pair.spearman_rho, 1.)

    def test_small_n_and_constant_columns_have_no_correlation_claim(self):
        rows = pairwise_correlations(pd.DataFrame({"a": [1, 1, 1, 1], "b": [1, 2, 3, 4]}))
        self.assertTrue(rows.pearson_r.isna().all())
        self.assertTrue(rows.spearman_rho.isna().all())

    def test_flux_transform_never_double_logs_r_or_signed_velocity(self):
        raw = pd.DataFrame({"Vx": [-400, -300], "P4": [100., 0], "USFLUXL": [1e20, -1], "R_VALUE": [4., 0]})
        result = transform_summaries(raw)
        np.testing.assert_array_equal(result.Vx, raw.Vx)
        np.testing.assert_array_equal(result.R_VALUE, raw.R_VALUE)
        self.assertEqual(result.P4.iloc[0], 2)
        self.assertTrue(np.isnan(result.P4.iloc[1]))
        self.assertEqual(result.USFLUXL.iloc[0], 20)
        self.assertTrue(np.isnan(result.USFLUXL.iloc[1]))
        pd.testing.assert_frame_equal(transform_summaries(raw, "raw"), raw)

    def _dataset(self, root, feature, n=12):
        metadata = pd.DataFrame({"FlrOnset": ["2013-01-01 00:00:00"]*6+["2025-01-01 00:00:00+00:00"]*6,
                                 "event_type": [0]*6+[1]*6})
        x = np.arange(n*288, dtype=float).reshape(n, 288)/1000+1
        if feature == "Vx":
            x *= -100
        frame = pd.concat([metadata, pd.DataFrame(x, columns=["{}_{}".format(feature, i) for i in range(288, 0, -1)])], axis=1)
        path = root/("1998_2025_MEMSEP_dataset_with_{}_filled.csv".format(feature))
        frame.to_csv(path, index=False)
        q = root/"quality"
        q.mkdir(exist_ok=True)
        pd.DataFrame({"row_index": np.arange(n), "point_status": ["U"*288, "M"*288, "D"*288, "H"*288, "S"*288]+["O"*288]*(n-5),
                      "source_era": ["historical_unknown"]*6+["modern"]*6}).to_csv(q/(feature+"_quality.csv"), index=False)
        return path, q/(feature+"_quality.csv")

    def test_native_policy_excludes_median_structural_unknown_derived_and_held(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, quality = self._dataset(Path(tmp), "P4")
            metadata, filled, native, summary, counts = load_feature(path, quality, "P4")
            self.assertTrue(np.isnan(native[:5]).all())
            self.assertTrue(np.isnan(filled[4]).all())
            self.assertTrue(np.isfinite(filled[:4]).all())
            self.assertTrue(np.isfinite(native[5:]).all())
            self.assertTrue(summary.native_mean.iloc[:5].isna().all())
            self.assertEqual(summary.points_D.iloc[2], 288)
            self.assertEqual(summary.points_H.iloc[3], 288)

    def test_native_minimum_points_guards_single_observation_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, quality = self._dataset(Path(tmp), "P4")
            q = pd.read_csv(quality, dtype={"point_status": str})
            q.loc[5, "point_status"] = "O"+"I"*287
            q.to_csv(quality, index=False)
            _, _, _, summary, _ = load_feature(path, quality, "P4", native_min_points=3)
            self.assertTrue(np.isnan(summary.native_mean.iloc[5]))
            self.assertEqual(summary.native_valid_points.iloc[5], 1)

    def test_invalid_quality_mask_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, quality = self._dataset(Path(tmp), "P4")
            q = pd.read_csv(quality)
            q.loc[0, "point_status"] = "O"*287
            q.to_csv(quality, index=False)
            with self.assertRaisesRegex(ValueError, "point_status"):
                load_feature(path, quality, "P4")

    def test_time_label_correlation_requires_both_classes(self):
        x = np.arange(12.)[:, None]*np.ones((1, 288))
        rows = _label_association(x, np.zeros(12), 5)
        self.assertTrue(rows.pearson_label_r.isna().all())
        self.assertEqual(rows.n_pairwise.iloc[0], 12)

    def test_structurally_unavailable_feature_still_produces_honest_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, quality = self._dataset(root, "P4")
            q = pd.read_csv(quality, dtype={"point_status": str})
            q["point_status"] = "S"*288
            q.to_csv(quality, index=False)
            result = run_paper_figures(root, root/"figures", features=("P4",))
            correlations = pd.read_csv(root/"figures"/"feature_correlations.csv")
            self.assertTrue(correlations.pearson_r.isna().all())
            self.assertEqual(correlations.n_pairwise.max(), 0)
            self.assertEqual(len(result["figure_files"]), 24)
            # Even empty era tables retain a reusable, documented CSV schema.
            eras = pd.read_csv(root/"figures"/"correlations_by_instrument_era.csv")
            self.assertIn("n_pairwise", eras.columns)

    def test_complete_diagnostics_export_methods_counts_and_all_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for feature in ("Vx", "P4", "Xs"):
                self._dataset(root, feature)
            out = root/"figures"
            result = run_paper_figures(root, out, features=("Vx", "P4", "Xs"))
            self.assertEqual(len(result["figure_files"]), 24)
            self.assertTrue(all(Path(path).is_file() for path in result["figure_files"]))
            manifest = json.loads(Path(result["manifest"]).read_text())
            self.assertEqual(manifest["sample_windows"], 12)
            self.assertIn("log10", manifest["transforms"]["Xs"])
            correlations = pd.read_csv(out/"feature_correlations.csv")
            self.assertLessEqual(correlations.n_pairwise.max(), 12)
            self.assertTrue((out/"correlations_by_instrument_era.csv").is_file())
            coverage = pd.read_csv(out/"feature_year_coverage.csv")
            self.assertTrue(coverage.unknown_historical_points.gt(0).any())
            source = pd.read_csv(root/"1998_2025_MEMSEP_dataset_with_P4_filled.csv")
            self.assertEqual(source.shape, (12, 290))


if __name__ == "__main__":
    unittest.main()
