"""攻击为正类；只用预测成功且标签有效的记录计算指标。"""
import numpy as np
import pandas as pd

METRICS = ["Accuracy", "Precision", "Recall", "F1", "FPR"]
ERROR_NAMES = {"correct": "正确", "false_positive": "误报", "false_negative": "漏报",
               "missing_label": "缺失标签", "invalid_label": "非法标签", "failed": "处理失败"}


def normalize_labels(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    text = values.astype("string").str.strip()
    missing = values.isna() | text.eq("").fillna(False)
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.isin([0, 1]) & ~missing
    status = pd.Series("invalid", index=values.index, dtype="string")
    status.loc[missing] = "missing"
    status.loc[valid] = "valid"
    return numeric.where(valid).astype("Int64"), status


def classify_results(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    labels, status = normalize_labels(frame["true_label"])
    frame["label_status"] = status
    frame["error_type"] = status.map({"missing": "missing_label", "invalid": "invalid_label", "valid": "correct"})
    success = frame["prediction_status"].eq("success")
    frame.loc[success & labels.eq(0).fillna(False) & frame["predicted_label"].eq(1), "error_type"] = "false_positive"
    frame.loc[success & labels.eq(1).fillna(False) & frame["predicted_label"].eq(0), "error_type"] = "false_negative"
    # 失败记录保留空预测，不能算作正常或漏报。
    frame.loc[~success, "error_type"] = "failed"
    return frame


def metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict:
    fractions = {"Accuracy": (tp + tn, tp + tn + fp + fn), "Precision": (tp, tp + fp),
                 "Recall": (tp, tp + fn), "F1": (2 * tp, 2 * tp + fp + fn), "FPR": (fp, fp + tn)}
    result = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
    for name, (numerator, denominator) in fractions.items():
        result[name] = numerator / denominator if denominator else None
        result[f"{name}_reason"] = "" if denominator else "分母为零，无可计算样本"
    return result


def evaluate_results(results: pd.DataFrame) -> dict:
    labels, status = normalize_labels(results["true_label"])
    success = results["prediction_status"].eq("success")
    eligible = success & status.eq("valid")
    truth = labels.loc[eligible]
    pred = results.loc[eligible, "predicted_label"]
    counts = [int(((truth == a) & (pred == b)).sum()) for a, b in [(1, 1), (0, 0), (0, 1), (1, 0)]]
    result = metrics_from_counts(*counts)
    result.update(total_records=len(results), successful_records=int(success.sum()),
                  failed_records=int((~success).sum()), evaluated_records=int(eligible.sum()),
                  missing_labels=int(status.eq("missing").sum()), invalid_labels=int(status.eq("invalid").sum()),
                  excluded_records=int((~eligible).sum()))
    return result


def build_metrics_table(results: pd.DataFrame) -> pd.DataFrame:
    """保留各模型独立口径，同时生成共同成功且标签有效的公平比较口径。"""
    groups = list(results.groupby("model_id", sort=False))
    eligible_sets = []
    for _, frame in groups:
        _, status = normalize_labels(frame["true_label"])
        eligible_sets.append(set(frame.loc[frame.prediction_status.eq("success") & status.eq("valid"), "record_id"]))
    common = set.intersection(*eligible_sets) if eligible_sets else set()
    rows = []
    for _, frame in groups:
        metadata = frame.iloc[0][["dataset_id", "run_mode", "data_notice", "model_id", "model_name", "model_version", "threshold"]].to_dict()
        for scope in (["individual", "common"] if len(groups) > 1 else ["individual"]):
            subset = frame if scope == "individual" else frame.loc[frame.record_id.isin(common)]
            rows.append({**metadata, "scope": scope, "common_evaluation_records": len(common),
                         "metric_definition": "攻击=1；仅成功且标签为0/1；指标为0~1；空值原因见*_reason",
                         **evaluate_results(subset)})
    return pd.DataFrame(rows)


def confusion_matrix(results: pd.DataFrame) -> np.ndarray:
    metrics = evaluate_results(results)
    return np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])


def category_summary(results: pd.DataFrame) -> pd.DataFrame:
    labels, status = normalize_labels(results["true_label"])
    attacks = results.loc[status.eq("valid") & labels.eq(1).fillna(False)].copy()
    attacks["attack_category"] = attacks.attack_category.fillna("未提供类别").replace("", "未提供类别")
    attacks["detected"] = attacks.prediction_status.eq("success") & attacks.predicted_label.eq(1)
    attacks["missed"] = attacks.error_type.eq("false_negative")
    attacks["failed"] = attacks.prediction_status.eq("failed")
    return attacks.groupby("attack_category", dropna=False).agg(
        样本数=("record_id", "size"), 检出数=("detected", "sum"), 漏报数=("missed", "sum"), 失败数=("failed", "sum")).reset_index()


def model_disagreements(results: pd.DataFrame) -> pd.DataFrame:
    models = results.model_id.unique()
    if len(models) != 2:
        return pd.DataFrame()
    frames = [results.loc[results.model_id.eq(model) & results.prediction_status.eq("success"),
                          ["record_id", "model_name", "attack_score", "predicted_label"]] for model in models]
    aligned = frames[0].merge(frames[1], on="record_id", suffixes=("_left", "_right"), validate="one_to_one")
    return aligned.loc[aligned.predicted_label_left != aligned.predicted_label_right]
