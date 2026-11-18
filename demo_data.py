"""固定示例，不按标签生成预测，不应用于真实实验。"""
from pathlib import Path
import numpy as np
from detection_service import read_dataset

# 示例数据与固定得分按原始行顺序一一对应，不读取真实标签来生成得分。
DEMO_PATH = Path(__file__).parent / "data" / "demo_flows.csv"
DEMO_SCORES = {
    "rf": np.array([.08, .76, .91, .24, .50, .16, .84, .32, .67, .42, .71, .12]),
    "mlp": np.array([.10, .21, .88, .72, .49, .19, .80, .68, .81, .63, .51, .14]),
}
# 模型说明沿用真实模式的结构，用于验证同一套检测调用流程。
DEMO_METADATA = {key: {"name": name, "version": "demo-fixed-v1", "feature_names": ["proto", "service", "dur", "sbytes", "dbytes"],
                       "feature_dtypes": {"proto": "categorical", "service": "categorical", "dur": "numeric", "sbytes": "numeric", "dbytes": "numeric"}}
                 for key, name in [("rf", "随机森林 · 示例"), ("mlp", "MLP · 示例")]}


def demo_dataset():
    """读取固定 CSV，生成与真实导入相同的数据对象。"""
    return read_dataset(DEMO_PATH.read_bytes(), DEMO_PATH.name)


class DemoModel:
    """模拟真实模型的调用接口，提供固定得分供界面演示。"""
    def __init__(self, model_id):
        """记录选用哪个示例模型及其展示名称。"""
        self.model_id = model_id
        self.model_name = DEMO_METADATA[model_id]["name"]

    def predict_attack_scores(self, features):
        """按输入行的原始位置取分数，支持多批检测且不改变顺序。"""
        # 原始位置索引由 B 保留，分批与整批得到完全相同的固定得分。
        return DEMO_SCORES[self.model_id][features.index.to_numpy()].copy()
