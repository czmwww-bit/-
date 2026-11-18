"""深度学习分析验证：手算曲线、同分边界、训练记录和页面交互。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import detection_service
from demo_data import demo_dataset
from evaluation import score_curves, scored_records
from training_report import load_training_report, validate_training_report


def prediction_frame(labels, scores):
    """用可手算的标签和得分建立最小结果表。"""
    return pd.DataFrame({"true_label": labels, "attack_score": scores, "prediction_status": "success", "threshold": .5})


def training_fixture():
    """仅用于验证展示功能的人工测试记录，不是项目训练结果。"""
    return {"schema_version": 1, "report_type": "training_run", "model_id": "mlp", "model_version": "test-v1",
            "framework": "PyTorch", "architecture": "测试结构 5 → 32 → 16 → 1",
            "training": {"dataset_name": "人工测试数据", "train_samples": 8, "validation_samples": 4,
                         "seed": 42, "device": "cpu", "optimizer": "Adam", "learning_rate": .001,
                         "batch_size": 4, "loss": "BCEWithLogitsLoss"},
            "history": [{"epoch": 1, "train_loss": .8, "val_loss": .9, "val_f1": .4, "val_accuracy": .5},
                        {"epoch": 2, "train_loss": .6, "val_loss": .7, "val_f1": .6, "val_accuracy": .75}]}


class CurveTests(unittest.TestCase):
    """核对 ROC-AUC、非插值 AP、单类和非法记录的处理。"""
    def test_hand_calculated_curves(self):
        """四条记录的标准手算案例：AUC=0.75，AP=5/6。"""
        curves = score_curves(prediction_frame([0, 0, 1, 1], [.1, .4, .35, .8]))
        self.assertAlmostEqual(curves["summary"]["ROC_AUC"], .75)
        self.assertAlmostEqual(curves["summary"]["Average_Precision"], 5 / 6)
        np.testing.assert_allclose(curves["points"].TPR, [0, .5, .5, 1, 1])
        np.testing.assert_allclose(curves["points"].FPR, [0, 0, .5, .5, 1])

    def test_perfect_reverse_and_tied_scores(self):
        """完美排序、反向排序和全部同分的曲线应有确定结果。"""
        for scores, auc, ap in [([.1, .2, .8, .9], 1, 1), ([.9, .8, .2, .1], 0, 5 / 12), ([.5] * 4, .5, .5)]:
            with self.subTest(scores=scores):
                frame = prediction_frame([0, 0, 1, 1], scores)
                summary = score_curves(frame)["summary"]
                self.assertAlmostEqual(summary["ROC_AUC"], auc)
                self.assertAlmostEqual(summary["Average_Precision"], ap)
                self.assertEqual(summary, score_curves(frame.iloc[::-1])["summary"])

    def test_single_class_empty_and_exclusions(self):
        """无标签、单类、非法得分或失败时，不强行给出 AUC。"""
        for labels, scores in [([], []), ([0, 0], [.2, .8]), ([None, "bad"], [.2, .8])]:
            summary = score_curves(prediction_frame(labels, scores))["summary"]
            self.assertIsNone(summary["ROC_AUC"])
            self.assertIsNone(summary["Average_Precision"])
        attacks = score_curves(prediction_frame([1, 1], [.2, .8]))["summary"]
        self.assertIsNone(attacks["ROC_AUC"])
        self.assertEqual(attacks["Average_Precision"], 1)
        frame = prediction_frame([0, 1, 1, 1, 0, 0], [.1, .9, np.nan, np.inf, -.1, .6])
        frame.loc[5, "prediction_status"] = "failed"
        summary = score_curves(frame)["summary"]
        self.assertEqual(summary["ranking_evaluated_records"], 2)
        self.assertEqual(summary["ranking_excluded_records"], 4)

    def test_near_threshold_and_label_groups(self):
        """边界分析可包含缺失标签，但不得包含失败记录或非法得分。"""
        frame = prediction_frame([0, 1, None, "bad", 1], [.4, .5, .6, .55, np.nan])
        scored = scored_records(frame)
        self.assertEqual(len(scored), 4)
        self.assertEqual(scored.score_group.tolist(), ["真实攻击", "标签非法", "真实正常", "标签缺失"])
        self.assertEqual(len(scored.loc[scored.distance_to_threshold.le(.1 + 1e-12)]), 4)


class TrainingReportTests(unittest.TestCase):
    """训练报告只接受同一模型版本的实际记录，模板和非法曲线明确拒绝。"""
    def test_report_validation(self):
        """验证可选指标、空记录、模板、错版本、重复轮次和非有限数值。"""
        report = training_fixture()
        self.assertEqual(validate_training_report(report, "test-v1"), report)
        cases = []
        for key, value in [("schema_version", True), ("report_type", "template"), ("model_version", "wrong"), ("history", [])]:
            invalid = copy.deepcopy(report)
            invalid[key] = value
            cases.append(invalid)
        for key, value in [("epoch", 1), ("train_loss", np.nan), ("val_loss", -.1), ("val_f1", 90)]:
            invalid = copy.deepcopy(report)
            invalid["history"][1][key] = value
            cases.append(invalid)
        for invalid in cases:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_training_report(invalid, "test-v1")

    def test_load_modes_missing_and_bad_json(self):
        """验证模式隔离、缺少报告、JSON 损坏和 BOM 编码读取。"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            self.assertIn("尚未交付", load_training_report(path, "test-v1", "real")[1])
            path.write_text(json.dumps(training_fixture(), ensure_ascii=False), encoding="utf-8-sig")
            self.assertIsNotNone(load_training_report(path, "test-v1", "real")[0])
            self.assertIsNone(load_training_report(path, "test-v1", "debug")[0])
            self.assertIn("不一致", load_training_report(path, "wrong", "real")[1])
            path.write_text("broken", encoding="utf-8")
            self.assertIsNone(load_training_report(path, "test-v1", "real")[0])

    def test_training_report_view(self):
        """用人工测试记录验证损失图、验证指标图与下载按钮的显示分支。"""
        app = AppTest.from_string("import streamlit as st\nfrom deep_learning_view import render_training_report\nrender_training_report(st.session_state['test_run'])")
        app.session_state["test_run"] = {"training_report": training_fixture(), "dataset": demo_dataset()}
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("plotly_chart")), 2)
        self.assertEqual(len(app.get("download_button")), 2)


class LearningPageTests(unittest.TestCase):
    """验证 MLP 默认优先，以及新增分析交互复用已有预测。"""
    def test_primary_model_and_no_reinference(self):
        """边界观察范围变化只改变分析视图，预测标签与 0.5 阈值保持不变。"""
        with patch("detection_service.run_detection", wraps=detection_service.run_detection) as inference:
            app = AppTest.from_file("app.py", default_timeout=30).run()
            self.assertEqual(app.multiselect(key="models_debug").value[0], "mlp")
            app.button(key="start").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key="focus").value, "mlp")
            before = app.session_state["run"]["results"].copy()
            app.slider(key="mlp_band").set_value(.05).run()
            self.assertFalse(app.exception)
            self.assertEqual(inference.call_count, 1)
            pd.testing.assert_frame_equal(before, app.session_state["run"]["results"])
            self.assertTrue(before.threshold.eq(.5).all())
            self.assertIsNone(app.session_state["run"]["training_report"])


if __name__ == "__main__":
    unittest.main()
