"""使用 Streamlit 官方 AppTest 验证状态，不需要真实模型。"""
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
import detection_service
from demo_data import DemoModel, demo_dataset


class AppTests(unittest.TestCase):
    def test_failure_feedback(self):
        with patch("demo_data.DemoModel", side_effect=FileNotFoundError("缺少测试模型包")):
            app = AppTest.from_file("app.py", default_timeout=30).run()
            app.button(key="start").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("全部检测失败" in item.value for item in app.error))
            self.assertTrue(app.session_state["run"]["results"].predicted_label.isna().all())
        class PartialModel(DemoModel):
            def predict_attack_scores(self, features):
                if self.model_id == "rf" and features.index.min() == 3:
                    raise ValueError("测试批次失败")
                return super().predict_attack_scores(features)
        with patch("demo_data.DemoModel", PartialModel):
            app = AppTest.from_file("app.py", default_timeout=30).run()
            app.number_input(key="batch_size").set_value(3).run()
            app.button(key="start").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("部分检测失败" in item.value for item in app.warning))
            self.assertEqual(len(app.session_state["run"]["failures"]), 1)

    def test_unlabelled_input(self):
        dataset = demo_dataset()
        dataset.raw = dataset.raw.drop(columns=["label", "attack_cat"])
        dataset.records["true_label"] = None
        dataset.records["attack_category"] = None
        with patch("demo_data.demo_dataset", return_value=dataset):
            app = AppTest.from_file("app.py", default_timeout=30).run()
            app.button(key="start").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("没有预测成功且标签有效" in item.value for item in app.info))
            self.assertTrue(app.session_state["run"]["results"].error_type.eq("missing_label").all())

    def test_full_debug_flow_and_no_reinference(self):
        with patch("detection_service.run_detection", wraps=detection_service.run_detection) as inference:
            app = AppTest.from_file("app.py", default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertEqual(inference.call_count, 0)
            app.button(key="start").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(inference.call_count, 1)
            self.assertEqual(len(app.session_state["run"]["results"]), 24)
            app.selectbox(key="overview_error").select("false_positive").run()
            app.text_input(key="overview_query").input("3").run()
            app.selectbox(key="analysis_model").select("mlp").run()
            app.run()  # 页签为原生前端切换；脚本重跑也不会重新推理。
            self.assertFalse(app.exception)
            self.assertEqual(inference.call_count, 1)
            self.assertEqual(app.session_state["run_count"], 1)
            app.multiselect(key="models_debug").set_value(["rf"]).run()
            self.assertTrue(any("配置已改变" in item.value for item in app.warning))
            self.assertEqual(len(app.get("download_button")), 1)  # 只剩固定示例下载。
            app.button(key="start").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(inference.call_count, 2)
            self.assertEqual(len(app.session_state["run"]["results"]), 12)
            app.radio(key="mode").set_value("真实检测").run()
            self.assertFalse(app.exception)
            self.assertTrue(app.button(key="start").disabled)
            self.assertTrue(any("配置已改变" in item.value for item in app.warning))
            self.assertEqual(inference.call_count, 2)


if __name__ == "__main__":
    unittest.main()
