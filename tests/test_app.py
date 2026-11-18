"""使用 Streamlit 官方 AppTest 验证状态，不需要真实模型。"""
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
import detection_service
from demo_data import DemoModel, demo_dataset


class AppTests(unittest.TestCase):
    """模拟用户点击页面，验证提示、结果保存和推理次数。"""
    def test_failure_feedback(self):
        """验证模型缺失和单批失败时，界面都能说明原因。"""
        # 用 patch 临时替换模型，主动制造可重复的故障场景。
        with patch("demo_data.DemoModel", side_effect=FileNotFoundError("缺少测试模型包")):
            app = AppTest.from_file("app.py", default_timeout=30).run()
            app.button(key="start").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("全部检测失败" in item.value for item in app.error))
            self.assertTrue(app.session_state["run"]["results"].predicted_label.isna().all())
        class PartialModel(DemoModel):
            """只让 RF 的第二批出错，其余批次仍返回固定得分。"""
            def predict_attack_scores(self, features):
                """输入从第 4 条记录开始时，模拟本批预测异常。"""
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
        """验证没有真实标签时仍可预测，但不能计算评估指标。"""
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
        """验证完整操作流程，重点检查交互不重复推理、配置变化不复用旧结果。"""
        # wraps 保留真实调用，同时统计调用次数，不用假结果替代检测。
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
            # 换模型后必须重跑；重跑结果替换旧结果，不追加重复记录。
            app.multiselect(key="models_debug").set_value(["rf"]).run()
            self.assertTrue(any("配置已改变" in item.value for item in app.warning))
            self.assertEqual(len(app.get("download_button")), 1)  # 只剩固定示例下载。
            app.button(key="start").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(inference.call_count, 2)
            self.assertEqual(len(app.session_state["run"]["results"]), 12)
            # 切换真实模式时，不能把调试结果当作真实模型输出。
            app.radio(key="mode").set_value("真实检测").run()
            self.assertFalse(app.exception)
            self.assertTrue(app.button(key="start").disabled)
            self.assertTrue(any("配置已改变" in item.value for item in app.warning))
            self.assertEqual(inference.call_count, 2)


if __name__ == "__main__":
    # 直接执行此文件时，启动 unittest 测试入口。
    unittest.main()
