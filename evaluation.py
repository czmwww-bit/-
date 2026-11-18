"""攻击为正类；只用预测成功且标签有效的记录计算指标。"""
import numpy as np
import pandas as pd

# 公共名称：五项指标用于计算和绘图，错误类型用于筛选和中文展示。
METRICS = ["Accuracy", "Precision", "Recall", "F1", "FPR"]
RANKING_METRICS = ["ROC_AUC", "Average_Precision"]
ERROR_NAMES = {"correct": "正确", "false_positive": "误报", "false_negative": "漏报",
               "missing_label": "缺失标签", "invalid_label": "非法标签", "failed": "处理失败"}


def normalize_labels(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """把标签分为有效、缺失、非法三类，返回数值标签和状态。"""
    # 空值和空白字符串算缺失；其他非 0/1 内容算非法。
    text = values.astype("string").str.strip()
    missing = values.isna() | text.eq("").fillna(False)
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.isin([0, 1]) & ~missing
    status = pd.Series("invalid", index=values.index, dtype="string")
    status.loc[missing] = "missing"
    status.loc[valid] = "valid"
    # Int64 可以同时保存整数和空值；非法标签不会被当成正常标签 0。
    return numeric.where(valid).astype("Int64"), status


def classify_results(results: pd.DataFrame) -> pd.DataFrame:
    """为每条结果标记正确、误报、漏报、缺失/非法标签或处理失败。"""
    frame = results.copy()
    labels, status = normalize_labels(frame["true_label"])
    frame["label_status"] = status
    frame["error_type"] = status.map({"missing": "missing_label", "invalid": "invalid_label", "valid": "correct"})
    # 有效标签先视为正确，再用真实值与预测值找出两种错误。
    # eq 是逐项比较；多个条件用 & 连接，表示必须同时满足。
    success = frame["prediction_status"].eq("success")
    frame.loc[success & labels.eq(0).fillna(False) & frame["predicted_label"].eq(1), "error_type"] = "false_positive"
    frame.loc[success & labels.eq(1).fillna(False) & frame["predicted_label"].eq(0), "error_type"] = "false_negative"
    # 失败记录保留空预测，不能算作正常或漏报。
    frame.loc[~success, "error_type"] = "failed"
    return frame


def metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict:
    """由 TP/TN/FP/FN 计算五项指标，并记录无法计算的原因。"""
    # Accuracy：总体判对比例；Precision：告警中真正攻击的比例。
    # Recall：真实攻击的检出比例；F1：综合 Precision 与 Recall。
    # FPR：正常流量被误报的比例，分母只包含真实正常流量。
    fractions = {"Accuracy": (tp + tn, tp + tn + fp + fn), "Precision": (tp, tp + fp),
                 "Recall": (tp, tp + fn), "F1": (2 * tp, 2 * tp + fp + fn), "FPR": (fp, fp + tn)}
    result = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
    for name, (numerator, denominator) in fractions.items():
        # 分母为零时不能当成 0%，留空并让界面显示 N/A。
        result[name] = numerator / denominator if denominator else None
        result[f"{name}_reason"] = "" if denominator else "分母为零，无可计算样本"
    return result


def evaluate_results(results: pd.DataFrame) -> dict:
    """统计混淆矩阵四格、五项指标和各类排除数量。"""
    labels, status = normalize_labels(results["true_label"])
    success = results["prediction_status"].eq("success")
    # 同时满足“预测成功”和“标签有效”的记录才参与评估。
    eligible = success & status.eq("valid")
    truth = labels.loc[eligible]
    pred = results.loc[eligible, "predicted_label"]
    # (真实值, 预测值) 按 TP、TN、FP、FN 的顺序逐项计数。
    counts = [int(((truth == a) & (pred == b)).sum()) for a, b in [(1, 1), (0, 0), (0, 1), (1, 0)]]
    result = metrics_from_counts(*counts)
    # 缺失标签、非法标签和预测失败可能重叠；excluded_records 按记录去重计数。
    result.update(total_records=len(results), successful_records=int(success.sum()),
                  failed_records=int((~success).sum()), evaluated_records=int(eligible.sum()),
                  missing_labels=int(status.eq("missing").sum()), invalid_labels=int(status.eq("invalid").sum()),
                  excluded_records=int((~eligible).sum()))
    return result


def build_metrics_table(results: pd.DataFrame) -> pd.DataFrame:
    """保留各模型独立口径，同时生成共同成功且标签有效的公平比较口径。"""
    # 先收集每个模型可评估的 record_id，再取交集，保证比较同一批流量。
    groups = list(results.groupby("model_id", sort=False))
    eligible_sets = []
    for _, frame in groups:
        _, status = normalize_labels(frame["true_label"])
        eligible_sets.append(set(frame.loc[frame.prediction_status.eq("success") & status.eq("valid"), "record_id"]))
    common = set.intersection(*eligible_sets) if eligible_sets else set()
    # individual 报各自表现；common 只报共同样本上的表现。
    rows = []
    for _, frame in groups:
        metadata = frame.iloc[0][["dataset_id", "run_mode", "data_notice", "model_id", "model_name", "model_version", "threshold"]].to_dict()
        for scope in (["individual", "common"] if len(groups) > 1 else ["individual"]):
            subset = frame if scope == "individual" else frame.loc[frame.record_id.isin(common)]
            rows.append({**metadata, "scope": scope, "common_evaluation_records": len(common),
                         "metric_definition": "攻击=1；仅成功且标签为0/1；指标为0~1；空值原因见*_reason",
                         **evaluate_results(subset), **score_curves(subset)["summary"]})
    return pd.DataFrame(rows)


def confusion_matrix(results: pd.DataFrame) -> np.ndarray:
    """返回二维矩阵：行是真实类别，列是预测类别，类别顺序均为 0、1。"""
    metrics = evaluate_results(results)
    return np.array([[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]])


def category_summary(results: pd.DataFrame) -> pd.DataFrame:
    """按真实攻击类别汇总样本、检出、漏报和失败数量。"""
    labels, status = normalize_labels(results["true_label"])
    # 只分析真实攻击；正常流量不进入攻击类别统计。
    attacks = results.loc[status.eq("valid") & labels.eq(1).fillna(False)].copy()
    attacks["attack_category"] = attacks.attack_category.fillna("未提供类别").replace("", "未提供类别")
    attacks["detected"] = attacks.prediction_status.eq("success") & attacks.predicted_label.eq(1)
    attacks["missed"] = attacks.error_type.eq("false_negative")
    # 失败无法判断是否漏报，所以单独列出。
    attacks["failed"] = attacks.prediction_status.eq("failed")
    return attacks.groupby("attack_category", dropna=False).agg(
        样本数=("record_id", "size"), 检出数=("detected", "sum"), 漏报数=("missed", "sum"), 失败数=("failed", "sum")).reset_index()


def model_disagreements(results: pd.DataFrame) -> pd.DataFrame:
    """按 record_id 对齐两个模型，找出双方成功但预测类别不同的记录。"""
    models = results.model_id.unique()
    if len(models) != 2:
        return pd.DataFrame()
    frames = [results.loc[results.model_id.eq(model) & results.prediction_status.eq("success"),
                          ["record_id", "model_name", "attack_score", "predicted_label"]] for model in models]
    # 一对一连接防止重复编号放大记录数；左右后缀区分两个模型的字段。
    aligned = frames[0].merge(frames[1], on="record_id", suffixes=("_left", "_right"), validate="one_to_one")
    return aligned.loc[aligned.predicted_label_left != aligned.predicted_label_right]


def score_curves(results: pd.DataFrame) -> dict:
    """根据已有得分计算 ROC、PR、ROC-AUC 和 AP，不改变 0.5 判定阈值。"""
    labels, status = normalize_labels(results.true_label)
    scores = pd.to_numeric(results.attack_score, errors="coerce")
    valid = results.prediction_status.eq("success") & status.eq("valid") & scores.between(0, 1) & np.isfinite(scores)
    truth = labels.loc[valid].to_numpy(dtype=int)
    values = scores.loc[valid].to_numpy(dtype=float)
    positives, negatives = int(truth.sum()), int(len(truth) - truth.sum())
    summary = {"ranking_evaluated_records": len(truth), "ranking_excluded_records": int((~valid).sum()),
               "positive_records": positives, "negative_records": negatives,
               "ROC_AUC": None, "ROC_AUC_reason": "需要有效的正常和攻击两类记录",
               "Average_Precision": None, "Average_Precision_reason": "没有有效的真实攻击记录"}
    columns = ["score_threshold", "FPR", "TPR", "Precision", "Recall"]
    if not len(truth):
        return {"points": pd.DataFrame(columns=columns), "summary": summary}

    # 从高分到低分累计告警；同分必须一起处理，避免排序顺序影响曲线。
    order = np.argsort(-values, kind="stable")
    sorted_scores, sorted_truth = values[order], truth[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(truth) - 1]
    tp = np.r_[0, np.cumsum(sorted_truth)[ends]]
    fp = np.r_[0, ends + 1 - tp[1:]]
    recall = tp / positives if positives else np.full(len(tp), np.nan)
    fpr = fp / negatives if negatives else np.full(len(fp), np.nan)
    # 起点表示尚无任何告警，PR 约定以 Precision=1 作为绘图端点。
    precision = np.r_[1.0, tp[1:] / (tp[1:] + fp[1:])]
    points = pd.DataFrame({"score_threshold": np.r_[np.inf, sorted_scores[ends]],
                           "FPR": fpr, "TPR": recall, "Precision": precision, "Recall": recall})
    if positives and negatives:
        summary.update(ROC_AUC=float(np.trapezoid(recall, fpr)), ROC_AUC_reason="")
    if positives:
        # AP 按召回率增量加权，不用梯形面积冒充 AP。
        summary.update(Average_Precision=float(np.sum(np.diff(recall) * precision[1:])), Average_Precision_reason="")
    return {"points": points, "summary": summary}


def scored_records(results: pd.DataFrame) -> pd.DataFrame:
    """取成功且得分有效的记录，补充距固定阈值的距离，供分布与边界分析。"""
    scores = pd.to_numeric(results.attack_score, errors="coerce")
    valid = results.prediction_status.eq("success") & scores.between(0, 1) & np.isfinite(scores)
    frame = results.loc[valid].copy()
    frame["attack_score"] = scores.loc[valid]
    labels, status = normalize_labels(frame.true_label)
    frame["score_group"] = status.map({"missing": "标签缺失", "invalid": "标签非法", "valid": "真实正常"})
    frame.loc[labels.eq(1).fillna(False), "score_group"] = "真实攻击"
    frame["distance_to_threshold"] = (frame.attack_score - frame.threshold).abs()
    return frame.sort_values("distance_to_threshold", kind="stable")
