"""读取 A 提供的 MLP 训练记录；只核对与展示，不训练、不加载权重。"""
import json
import math
from pathlib import Path


def validate_training_report(report: dict, model_version: str) -> dict:
    """验证报告结构、模型版本及训练数值，防止展示其他模型或模板的记录。"""
    if not isinstance(report, dict) or type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError("训练报告 schema_version 必须为 1")
    if report.get("report_type") != "training_run":
        raise ValueError("训练报告是模板或未知类型，不能作为实际训练记录")
    if report.get("model_id") != "mlp" or report.get("model_version") != model_version:
        raise ValueError("训练报告的模型标识或版本与本次 MLP 不一致")
    for field in ["framework", "architecture"]:
        if not isinstance(report.get(field), str) or not report[field].strip():
            raise ValueError(f"训练报告缺少 {field}")
    training, history = report.get("training"), report.get("history")
    if not isinstance(training, dict) or not isinstance(history, list) or not history:
        raise ValueError("训练报告需要 training 配置和非空 history 轮次记录")
    for field in ["dataset_name", "optimizer", "loss", "device"]:
        if not isinstance(training.get(field), str) or not training[field].strip():
            raise ValueError(f"训练配置缺少 {field}")
    for field in ["train_samples", "validation_samples", "batch_size"]:
        if type(training.get(field)) is not int or training[field] <= 0:
            raise ValueError(f"训练配置 {field} 必须为正整数")
    if type(training.get("seed")) is not int:
        raise ValueError("训练配置 seed 必须为整数")
    learning_rate = training.get("learning_rate")
    if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("训练配置 learning_rate 必须为正的有限数值")

    # 轮次必须递增；损失不可为负，可选验证指标限定为 0～1。
    previous_epoch = 0
    for row in history:
        if not isinstance(row, dict) or type(row.get("epoch")) is not int or row["epoch"] <= previous_epoch:
            raise ValueError("训练轮次 epoch 必须为递增的正整数")
        previous_epoch = row["epoch"]
        for field in ["train_loss", "val_loss", "val_f1", "val_accuracy"]:
            value = row.get(field)
            if field in ("val_f1", "val_accuracy") and value is None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"第 {previous_epoch} 轮 {field} 缺失或不是有效非负数值")
            if field in ("val_f1", "val_accuracy") and value > 1:
                raise ValueError(f"第 {previous_epoch} 轮 {field} 应使用 0～1 小数")
    return report


def load_training_report(path: Path, model_version: str, mode: str) -> tuple[dict | None, str]:
    """检测开始时读取一次报告；模式不符、缺失或非法都不影响正常预测。"""
    if mode != "real":
        return None, "界面调试模式使用固定得分，不加载真实训练记录，也不生成示例损失曲线。"
    try:
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        return validate_training_report(report, model_version), ""
    except FileNotFoundError:
        return None, f"尚未交付训练记录：{path.name}。请 A 导出与当前 MLP 版本一致的报告。"
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        return None, f"训练记录未展示：{exc}"
