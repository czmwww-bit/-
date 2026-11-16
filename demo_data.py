"""固定示例，不按标签生成预测，不应用于真实实验。"""
from pathlib import Path
import numpy as np
from detection_service import read_dataset

DEMO_PATH = Path(__file__).parent / "data" / "demo_flows.csv"
DEMO_SCORES = {
    "rf": np.array([.08, .76, .91, .24, .50, .16, .84, .32, .67, .42, .71, .12]),
    "mlp": np.array([.10, .21, .88, .72, .49, .19, .80, .68, .81, .63, .51, .14]),
}
DEMO_METADATA = {key: {"name": name, "version": "demo-fixed-v1", "feature_names": ["proto", "service", "dur", "sbytes", "dbytes"],
                       "feature_dtypes": {"proto": "categorical", "service": "categorical", "dur": "numeric", "sbytes": "numeric", "dbytes": "numeric"}}
                 for key, name in [("rf", "随机森林 · 示例"), ("mlp", "MLP · 示例")]}


def demo_dataset():
    return read_dataset(DEMO_PATH.read_bytes(), DEMO_PATH.name)


class DemoModel:
    def __init__(self, model_id):
        self.model_id = model_id
        self.model_name = DEMO_METADATA[model_id]["name"]

    def predict_attack_scores(self, features):
        # 原始位置索引由 B 保留，分批与整批得到完全相同的固定得分。
        return DEMO_SCORES[self.model_id][features.index.to_numpy()].copy()
