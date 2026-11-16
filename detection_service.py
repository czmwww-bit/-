"""B 调用层：CSV 身份、特征隔离、批量调用与按原始位置回填。"""
import csv
import hashlib
import io
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd
from evaluation import METRICS, classify_results

THRESHOLD = 0.5
EXCLUDED = {"label", "attack_cat", "id", "record_id", "true_label", "attack_category", "source_row",
            "dataset_id", "run_mode", "data_notice", "model_id", "model_name", "model_version", "threshold",
            "attack_score", "predicted_label", "prediction_status", "failure_reason", "error_type", "label_status",
            "tp", "tn", "fp", "fn", "correct", "false_positive", "false_negative", "evaluation_status",
            "total_records", "successful_records", "failed_records", "evaluated_records", "missing_labels",
            "invalid_labels", "excluded_records", "scope", "common_evaluation_records", "metric_definition",
            "错误类型说明", "真实标签", "真实类别", "预测标签", "预测类别"}
EXCLUDED.update(name.lower() for name in METRICS)
EXCLUDED.update(f"{name.lower()}_reason" for name in METRICS)


class DetectionModel(Protocol):
    model_name: str

    def predict_attack_scores(self, features: pd.DataFrame) -> np.ndarray:
        ...


@dataclass
class Dataset:
    dataset_id: str
    name: str
    raw: pd.DataFrame
    records: pd.DataFrame


def read_dataset(content: bytes, name: str) -> Dataset:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = content.decode("gb18030")
        except UnicodeDecodeError as exc:
            raise ValueError("CSV 编码无法识别，请另存为 UTF-8 CSV。") from exc
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader, [])
        if not header or any(not field.strip() for field in header):
            raise ValueError("CSV 为空或列名为空。")
        if len(set(header)) != len(header):
            raise ValueError("CSV 含重复列名，请先修正表头。")
        source_rows = []
        rows = []
        while True:
            start_line = reader.line_num + 1
            row = next(reader, None)
            if row is None:
                break
            if not row:  # 空白物理行跳过，但保留后续真实行号。
                continue
            if len(row) != len(header):
                raise ValueError(f"CSV 第 {start_line} 行字段数为 {len(row)}，应为 {len(header)}。")
            source_rows.append(start_line)
            rows.append(row)
    except csv.Error as exc:
        raise ValueError(f"CSV 第 {reader.line_num} 行语法错误：{exc}") from exc
    # 仅空字段作为缺失；NA 等文本原样保留，避免非法标签被悄悄记作缺失。
    raw = pd.DataFrame(rows, columns=header)
    raw = raw.where(raw.ne(""), np.nan)
    dataset_id = hashlib.sha256(content).hexdigest()
    records = pd.DataFrame({"record_id": [f"{dataset_id}:{i + 1}" for i in range(len(raw))],
                            "source_row": source_rows})
    records["true_label"] = raw["label"] if "label" in raw else raw.get("true_label", pd.Series(np.nan, index=raw.index))
    records["attack_category"] = raw["attack_cat"] if "attack_cat" in raw else raw.get("attack_category", pd.Series(np.nan, index=raw.index))
    return Dataset(dataset_id, name, raw, records)


def is_feature(column: str) -> bool:
    lowered = column.strip().lower()
    return lowered not in EXCLUDED and not lowered.startswith(("eval_", "metric_", "prediction_", "input_"))


def prepare_features(dataset: Dataset, metadata: dict) -> pd.DataFrame:
    allowed = [column for column in dataset.raw if is_feature(column)]
    required = metadata.get("feature_names")
    if required is not None:
        leaked = [name for name in required if not is_feature(name)]
        if leaked:
            raise ValueError(f"模型特征清单含标签或评估字段：{', '.join(leaked)}")
        missing = [name for name in required if name not in allowed]
        if missing:
            raise ValueError(f"缺少模型必需列：{', '.join(missing)}")
        allowed = list(required)
    if not allowed:
        raise ValueError("没有可用的模型输入特征。")
    features = dataset.raw[allowed].copy()
    # 类型约定由 A 的元数据给出；不填补缺失、不拟合、不编码、不标准化。
    for name, kind in metadata.get("feature_dtypes", {}).items():
        if name not in features:
            raise ValueError(f"类型清单字段不在输入中：{name}")
        if kind == "numeric":
            numeric = pd.to_numeric(features[name], errors="coerce")
            bad = features[name].notna() & (numeric.isna() | ~np.isfinite(numeric))
            if bad.any():
                lines = dataset.records.loc[bad, "source_row"].head(8).tolist()
                raise ValueError(f"特征 {name} 存在非法数值，CSV 行号：{lines}")
            features[name] = numeric
        elif kind != "categorical":
            raise ValueError(f"不支持的类型约定 {name}={kind}，请在 B 加载桥接处适配。")
    return features


def validate_scores(scores: np.ndarray, expected_count: int) -> np.ndarray:
    if not isinstance(scores, np.ndarray) or scores.ndim != 1:
        raise ValueError("模型必须返回一维 NumPy 数组。")
    if len(scores) != expected_count:
        raise ValueError(f"模型返回长度 {len(scores)} 与输入 {expected_count} 不一致。")
    if not np.issubdtype(scores.dtype, np.number) or np.iscomplexobj(scores):
        raise ValueError("模型得分必须为实数。")
    if not np.isfinite(scores).all():
        raise ValueError("模型得分包含 NaN 或无穷值。")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("模型得分超出 0～1 范围。")
    return scores.astype(float)


def run_detection(dataset: Dataset, model_ids: list[str], metadata: dict, loader: Callable,
                  mode: str, batch_size: int = 256, progress: Callable | None = None) -> tuple[pd.DataFrame, list[dict]]:
    if not model_ids or len(set(model_ids)) != len(model_ids):
        raise ValueError("请选择不重复的模型。")
    if batch_size < 1 or dataset.raw.empty:
        raise ValueError("批量大小须大于零，数据集不能为空。")
    frames, failures = [], []
    completed, total = 0, len(dataset.raw) * len(model_ids)
    for model_id in model_ids:
        meta = metadata[model_id]
        result = dataset.records.copy()
        result["dataset_id"], result["run_mode"] = dataset.dataset_id, mode
        result["data_notice"] = "界面调试数据" if mode == "debug" else "真实检测"
        result["model_id"], result["model_name"], result["model_version"] = model_id, meta["name"], meta["version"]
        result["threshold"] = THRESHOLD
        result["attack_score"] = np.nan
        result["predicted_label"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
        result["prediction_status"], result["failure_reason"] = "failed", ""
        try:
            features = prepare_features(dataset, meta)
            model = loader(model_id)
            if not callable(getattr(model, "predict_attack_scores", None)) or model.model_name != meta["name"]:
                raise ValueError("模型未实现预测接口或 model_name 与元数据不一致。")
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            result["failure_reason"] = reason
            failures.append({"model_id": model_id, "source_row_start": int(result.source_row.iloc[0]),
                             "source_row_end": int(result.source_row.iloc[-1]), "reason": reason})
            completed += len(result)
            if progress:
                progress(completed, total, f"{meta['name']} 加载或输入检查失败")
            frames.append(result)
            continue
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            positions = result.index[start:stop]
            try:
                scores = validate_scores(model.predict_attack_scores(features.iloc[start:stop].copy()), stop - start)
                # 使用位置回填；不使用原始 id 或模型输出的索引进行连接。
                result.loc[positions, "attack_score"] = scores
                result.loc[positions, "predicted_label"] = (scores >= THRESHOLD).astype(int)
                result.loc[positions, "prediction_status"] = "success"
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                result.loc[positions, "failure_reason"] = reason
                failures.append({"model_id": model_id, "source_row_start": int(result.loc[positions[0], "source_row"]),
                                 "source_row_end": int(result.loc[positions[-1], "source_row"]), "reason": reason})
            completed += stop - start
            if progress:
                progress(completed, total, f"{meta['name']} 已尝试 {stop}/{len(features)} 条")
        frames.append(result)
    combined = classify_results(pd.concat(frames, ignore_index=True))
    # 展示原始字段时，冲突字段加 input_ 前缀，防止覆盖运行标识或预测。
    displayed = dataset.raw.rename(columns={col: f"input_{col}" for col in dataset.raw if col in combined.columns})
    displayed = displayed.copy()
    # 极端情况下 input_x 本身已存在，不进行含糊的重复列导出。
    if displayed.columns.duplicated().any():
        displayed = dataset.raw.copy()
        displayed.columns = [f"raw_{i}_{name}" for i, name in enumerate(displayed.columns)]
    displayed["record_id"] = dataset.records.record_id
    return combined.merge(displayed, on="record_id", how="left", validate="many_to_one"), failures


def run_configuration(dataset: Dataset, mode: str, model_ids: list[str], metadata: dict, batch_size: int) -> dict:
    return {"dataset_id": dataset.dataset_id, "mode": mode, "threshold": THRESHOLD, "batch_size": batch_size,
            "models": [{"id": key, **metadata[key]} for key in sorted(model_ids)]}


def filter_results(results: pd.DataFrame, models: list[str], prediction: str, error: str, query: str,
                   sort_by: str, descending: bool) -> pd.DataFrame:
    filtered = results.loc[results.model_id.isin(models)]
    if prediction != "all":
        filtered = filtered.loc[filtered.predicted_label.eq(int(prediction)).fillna(False)]
    if error != "all":
        filtered = filtered.loc[filtered.error_type.eq(error)]
    if query.strip():
        # 纯数字按原始 id/行号精确查询，避免命中文件哈希中的数字导致全表匹配。
        mask = pd.Series(False, index=filtered.index) if query.strip().isdecimal() else filtered.record_id.str.contains(query.strip(), regex=False)
        if "id" in filtered:
            mask |= filtered["id"].astype(str).eq(query.strip())
        mask |= filtered.source_row.astype(str).eq(query.strip())
        filtered = filtered.loc[mask]
    return filtered.sort_values(sort_by, ascending=not descending, kind="stable", na_position="last")
