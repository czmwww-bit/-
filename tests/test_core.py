import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from demo_data import DEMO_METADATA, DEMO_PATH, DEMO_SCORES, DemoModel, demo_dataset
from detection_service import (filter_results, prepare_features, read_dataset, run_configuration,
                               run_detection, validate_scores)
from evaluation import (METRICS, build_metrics_table, category_summary, confusion_matrix,
                        evaluate_results, metrics_from_counts, model_disagreements, normalize_labels)
from export_results import csv_bytes
from model_registry import discover_real_models


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.dataset = demo_dataset()

    def detect(self, loader=DemoModel, models=None, batch_size=3, dataset=None, metadata=None):
        return run_detection(dataset or self.dataset, models or ["rf", "mlp"], metadata or DEMO_METADATA,
                             loader, "debug", batch_size)

    def test_reference_metrics(self):
        metrics = metrics_from_counts(40, 50, 5, 5)
        for name, expected in zip(METRICS, [.9, 40 / 45, 40 / 45, 80 / 90, 5 / 55]):
            self.assertAlmostEqual(metrics[name], expected)

    def test_zero_denominators_and_extreme_classes(self):
        for counts, unavailable in [((0, 10, 0, 0), ["Precision", "Recall", "F1"]),
                                    ((10, 0, 0, 0), ["FPR"]),
                                    ((0, 0, 0, 10), ["Precision", "FPR"]),
                                    ((0, 0, 0, 0), METRICS)]:
            with self.subTest(counts=counts):
                metrics = metrics_from_counts(*counts)
                for name in unavailable:
                    self.assertIsNone(metrics[name])
                    self.assertTrue(metrics[f"{name}_reason"])

    def test_label_missing_invalid_and_no_label(self):
        labels, status = normalize_labels(pd.Series([0, 1, "0", "1", None, "", " ", "NA", 2, -.1, np.inf]))
        self.assertEqual(status.tolist(), ["valid"] * 4 + ["missing"] * 3 + ["invalid"] * 4)
        results, _ = self.detect()
        metrics = evaluate_results(results.loc[results.model_id.eq("rf")])
        self.assertEqual((metrics["evaluated_records"], metrics["missing_labels"], metrics["invalid_labels"]), (10, 1, 1))
        dataset = read_dataset(b"proto,dur,sbytes,dbytes,service\ntcp,1,2,3,http\n", "no_labels.csv")
        results, failures = self.detect(dataset=dataset, models=["rf"])
        self.assertFalse(failures)
        self.assertEqual(results.error_type.tolist(), ["missing_label"])
        self.assertEqual(evaluate_results(results)["evaluated_records"], 0)
        self.assertIsNone(evaluate_results(results)["Accuracy"])

    def test_stable_ids_duplicate_original_id_and_line_numbers(self):
        a = read_dataset(b'id,proto\n7,tcp\n\n7,"multi\nline"\n', "a.csv")
        b = read_dataset(b'id,proto\n7,tcp\n\n7,"multi\nline"\n', "renamed.csv")
        self.assertEqual(a.dataset_id, b.dataset_id)
        self.assertEqual(a.records.record_id.tolist(), b.records.record_id.tolist())
        self.assertTrue(a.records.record_id.is_unique)
        self.assertEqual(a.records.source_row.tolist(), [2, 4])

    def test_malformed_csv_and_empty(self):
        for data in [b"", b"id,id\n1,2\n", b"a,b\n1,2,3\n", b'a,b\n"oops,2\n']:
            with self.subTest(data=data), self.assertRaises(ValueError):
                read_dataset(data, "bad.csv")
        empty = read_dataset(b"proto,dur\n", "empty.csv")
        self.assertTrue(empty.raw.empty)
        with self.assertRaisesRegex(ValueError, "不能为空"):
            self.detect(dataset=empty)
        results, _ = self.detect()
        self.assertIsNone(evaluate_results(results.iloc[:0])["Accuracy"])

    def test_threshold_and_batch_alignment(self):
        progress = []
        results, failures = run_detection(self.dataset, ["rf", "mlp"], DEMO_METADATA, DemoModel, "debug", 5,
                                          lambda done, total, text: progress.append((done, total)))
        self.assertFalse(failures)
        self.assertEqual(progress, [(5, 24), (10, 24), (12, 24), (17, 24), (22, 24), (24, 24)])
        for key, scores in DEMO_SCORES.items():
            frame = results.loc[results.model_id.eq(key)]
            np.testing.assert_array_equal(frame.attack_score, scores)
            self.assertEqual(frame.record_id.tolist(), self.dataset.records.record_id.tolist())
        self.assertEqual(results.loc[(results.model_id == "rf") & (results.source_row == 6), "predicted_label"].iloc[0], 1)

    def test_invalid_model_outputs(self):
        bad = [np.array([[.2]]), np.array([]), np.array([np.nan]), np.array([np.inf]),
               np.array([-.01]), np.array([1.01]), np.array(["0.2"]), np.array([.1j]), [.2]]
        for values in bad:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_scores(values, 1)
        np.testing.assert_array_equal(validate_scores(np.array([0., .5, 1.]), 3), [0., .5, 1.])

    def test_feature_isolation_order_types_and_missing(self):
        raw = b"id,proto,dur,label,attack_cat,record_id,Accuracy,attack_score\n1,tcp,,1,DoS,spoof,1,1\n"
        dataset = read_dataset(raw, "features.csv")
        features = prepare_features(dataset, {"feature_names": ["dur", "proto"], "feature_dtypes": {"dur": "numeric"}})
        self.assertEqual(features.columns.tolist(), ["dur", "proto"])
        self.assertTrue(pd.isna(features.dur.iloc[0]))
        self.assertEqual(prepare_features(dataset, {}).columns.tolist(), ["proto", "dur"])
        for meta in [{"feature_names": ["sbytes"]}, {"feature_names": ["label"]}]:
            with self.assertRaises(ValueError):
                prepare_features(dataset, meta)
        invalid = read_dataset(b"dur\nabc\n", "invalid.csv")
        with self.assertRaisesRegex(ValueError, "CSV 行号.*2"):
            prepare_features(invalid, {"feature_dtypes": {"dur": "numeric"}})

    def test_partial_failure_not_normal_and_fair_comparison(self):
        class SometimesFails(DemoModel):
            def predict_attack_scores(self, features):
                if self.model_id == "rf" and features.index.min() == 3:
                    return np.array([np.nan] * len(features))
                return super().predict_attack_scores(features)
        results, failures = self.detect(loader=SometimesFails)
        self.assertEqual(len(failures), 1)
        self.assertEqual((failures[0]["source_row_start"], failures[0]["source_row_end"]), (5, 7))
        failed = results.loc[results.prediction_status.eq("failed")]
        self.assertTrue(failed.predicted_label.isna().all())
        self.assertTrue(failed.error_type.eq("failed").all())
        metrics = build_metrics_table(results)
        self.assertEqual(metrics.loc[metrics.scope.eq("individual"), "evaluated_records"].tolist(), [7, 10])
        common = metrics.loc[metrics.scope.eq("common")]
        self.assertEqual(common.evaluated_records.tolist(), [7, 7])
        self.assertTrue(common.common_evaluation_records.eq(7).all())

    def test_loader_failure_and_missing_columns(self):
        def missing_loader(key):
            raise FileNotFoundError("缺少 models/test-model")
        results, failures = self.detect(loader=missing_loader)
        self.assertEqual(len(failures), 2)
        self.assertTrue(results.predicted_label.isna().all())
        self.assertEqual(evaluate_results(results)["failed_records"], 24)
        dataset = read_dataset(b"proto\ntcp\n", "missing.csv")
        results, failures = self.detect(dataset=dataset)
        self.assertEqual(len(failures), 2)
        self.assertTrue(all("缺少模型必需列" in failure["reason"] for failure in failures))

    def test_confusion_categories_disagreements(self):
        results, _ = self.detect()
        frame = results.loc[results.model_id.eq("rf")]
        np.testing.assert_array_equal(confusion_matrix(frame), [[3, 1], [2, 4]])
        summary = category_summary(frame).set_index("attack_category")
        self.assertEqual(summary.loc["DoS", "漏报数"], 2)
        differences = model_disagreements(results)
        self.assertEqual(len(differences), 5)
        self.assertTrue(differences.record_id.is_unique)

    def test_exports_match_filter_and_configuration(self):
        results, _ = self.detect()
        filtered = filter_results(results, ["rf"], "1", "false_positive", "", "source_row", False)
        self.assertEqual(len(filtered), 1)
        queried = filter_results(results, ["rf", "mlp"], "all", "all", "3", "source_row", False)
        self.assertEqual(queried.source_row.tolist(), [3, 3])
        exported = csv_bytes(filtered)
        self.assertTrue(exported.startswith(b"\xef\xbb\xbf"))
        restored = pd.read_csv(io.BytesIO(exported))
        self.assertEqual(restored.record_id.tolist(), filtered.record_id.tolist())
        self.assertEqual(restored.data_notice.tolist(), ["界面调试数据"])
        self.assertEqual(restored.threshold.tolist(), [.5])
        metrics = build_metrics_table(results)
        restored_metrics = pd.read_csv(io.BytesIO(csv_bytes(metrics)))
        self.assertAlmostEqual(restored_metrics.Accuracy.iloc[0], metrics.Accuracy.iloc[0])
        config = run_configuration(self.dataset, "debug", ["rf"], DEMO_METADATA, 3)
        self.assertNotEqual(config, run_configuration(self.dataset, "real", ["rf"], DEMO_METADATA, 3))
        self.assertNotEqual(config, run_configuration(self.dataset, "debug", ["mlp"], DEMO_METADATA, 3))
        changed_data = read_dataset(DEMO_PATH.read_bytes() + b"\n", "changed.csv")
        self.assertNotEqual(config, run_configuration(changed_data, "debug", ["rf"], DEMO_METADATA, 3))
        changed_metadata = {key: {**value, "version": "new-version"} for key, value in DEMO_METADATA.items()}
        self.assertNotEqual(config, run_configuration(self.dataset, "debug", ["rf"], changed_metadata, 3))

    def test_registry_missing_and_metadata_validation(self):
        with patch("model_registry.importlib.import_module", side_effect=ModuleNotFoundError("model_adapter")):
            metadata, issues = discover_real_models()
            self.assertFalse(metadata)
            self.assertEqual(set(issues), {"rf", "mlp"})
        adapter = SimpleNamespace(load_model=lambda *args: None,
                                  get_model_metadata=lambda *args: {"rf": DEMO_METADATA["rf"], "mlp": {}})
        with patch("model_registry.importlib.import_module", return_value=adapter):
            metadata, issues = discover_real_models()
            self.assertEqual(set(metadata), {"rf"})
            self.assertEqual(set(issues), {"mlp"})


if __name__ == "__main__":
    unittest.main()
