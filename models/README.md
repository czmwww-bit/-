# A 的真实模型交接

本目录当前没有真实模型。B 不推测文件名、joblib 内容或 MLP 网络结构，也不会自行拟合预处理器。

请 A 提供 README 约定的 `model_adapter.py`，以及它引用的完整模型产物。B 唯一桥接位置是根目录 `model_registry.py`。以下两个函数是 **B 为待交付接口预留的接入方案**，并非声称 A 已实现；若 A 已有不同命名，在桥接处映射即可，不必修改 A。

```python
from pathlib import Path
import numpy as np
import pandas as pd

def get_model_metadata(models_dir: Path) -> dict:
    # 不做预测；检查真实文件，不存在时抛出包含缺失路径的异常。
    # 若仅交付 RF，可以只返回 rf；mlp 会显示不可用。
    return {
        "rf": {
            "name": "随机森林",  # 必须等于返回对象的 model_name
            "version": "完整模型包的内容哈希或唯一版本",
            "feature_names": ["proto", "dur"],  # 换成实际、完整、有序的输入特征
            "feature_dtypes": {"proto": "categorical", "dur": "numeric"},
        },
        # "mlp": {...相同结构...},
    }

def load_model(model_id: str, models_dir: Path):
    # 在此调用 A 实际的加载入口，返回实现下述接口的对象。
    # 加载已拟合处理器和模型，不在检测阶段 fit。
    ...

class DetectionModel:
    model_name: str

    def predict_attack_scores(self, features: pd.DataFrame) -> np.ndarray:
        # 列按 feature_names 排序；numeric 已转数值，空值保持 NaN；
        # categorical 保留原始值及缺失值。由 A 已拟合处理器处理。
        # 输入 index 保留原始位置；返回顺序必须与输入一致。
        # 必须返回 shape=(len(features),) 的有限实数 NumPy 数组，范围 [0,1]。
        ...
```

`version` 必须在权重、预处理器、特征清单或编码规则改变时更新，推荐覆盖全部模型产物的哈希。B 会将版本及特征约定写入本次运行配置；更换配置后隐藏旧结果。

需要交付的材料：

1. 随机森林模型、已拟合预处理器、类别编码规则、输入列名和顺序、缺失值与未知类别约定。
2. MLP 权重、重建网络所需结构、编码后输入维度、已拟合预处理器，以及 CPU 下 eval/no_grad 的加载实现。
3. `model_adapter.py` 和最小调用示例；RF 根据 `classes_` 取标签 1 的概率，MLP 将 logits 经 sigmoid 转为得分。
4. 小型验证 CSV、真实标签、攻击类别、按顺序可核对的两模型真实得分与版本。
5. Python、scikit-learn、joblib、PyTorch 的实际版本、训练划分和随机种子记录。B 当前依赖不包含训练库，待 A 交付后合并、验证并固定版本。

首次联调应核对：输入行数与输出长度、0/1 语义、阈值等于 0.5 的攻击判定、缺失值、未知类别、重启加载一致性，以及 B 导出的指标与 A 结果相符。

## MLP 训练报告展示约定（可选增补）

模型加载接口保持不变。A 若要让 B 展示实际训练过程，请另交付 `models/mlp_training_report.json`。缺少报告不影响预测，只显示“尚未交付训练记录”。B 在点击开始检测时读取一次并保存快照；更新报告后需重新开始检测。

可从 [空模板](../data/mlp_training_report.example.json) 填写，所有数值必须来自实际训练。模板的 `report_type=template` 会被明确拒绝作为真实记录；填完真实内容后设置为 `training_run`。

| 字段 | 内容与检查 |
| --- | --- |
| `schema_version` | 当前为 1 |
| `report_type` | 实际报告为 `training_run` |
| `model_id` | 必须是 `mlp` |
| `model_version` | 必须与 `get_model_metadata()` 中本次 MLP 的版本完全相同 |
| `framework`、`architecture` | 实际框架与网络结构说明，不能照抄计划结构冒充实际结构 |
| `training.dataset_name` | 训练/验证数据来源与划分说明 |
| `training.train_samples`、`validation_samples`、`batch_size` | 正整数 |
| `training.seed`、`device`、`optimizer`、`learning_rate`、`loss` | 实际随机种子、设备、优化器、正学习率和损失函数 |
| `history` | 非空列表，每行含递增正整数 `epoch`、非负有限数值 `train_loss`、`val_loss` |
| `history` 可选列 | `val_f1`、`val_accuracy` 使用 0～1 小数，允许缺失；不提供时不画对应曲线 |

不要写入 NaN、Infinity、重复轮次或百分数 90。模型训练、最佳检查点选择和报告生成由 A 完成；B 不在检测阶段重新训练或推断网络结构。版本不符的报告不会与当前 MLP 结果混用。训练配置与原始报告可以展开查看和下载，以便核对曲线来源。
