"""MLP 主模型分析页：只读取本次结果和训练报告快照，不调用模型。"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from evaluation import evaluate_results, score_curves, scored_records
from export_results import csv_bytes


def decimal(value):
    """AUC 和 AP 显示为四位小数，无有效结果时显示 N/A。"""
    return "N/A" if value is None or pd.isna(value) else f"{value:.4f}"


def curve_figure(x_title, y_title):
    """统一 ROC 和 PR 图的坐标范围及样式，方便并排比较。"""
    figure = go.Figure()
    figure.update_layout(height=330, margin=dict(l=20, r=20, t=20, b=30),
                         xaxis=dict(title=x_title, range=[0, 1]), yaxis=dict(title=y_title, range=[0, 1.05]),
                         paper_bgcolor="white", plot_bgcolor="white", legend=dict(orientation="h", y=1.17))
    return figure


def render_training_report(run):
    """展示与当前 MLP 版本匹配的真实结构、训练配置和逐轮曲线。"""
    st.subheader("MLP 结构与训练记录", anchor=False)
    report = run.get("training_report")
    if report is None:
        st.info(run.get("training_issue") or "本次没有可用的 MLP 训练记录。")
        st.caption("README 的计划结构为 输入 → 32 → 16 → 1，隐藏层 ReLU；实际结构以 A 的交付为准。")
        st.code("流量特征 → 已拟合预处理器 → MLP → sigmoid 攻击得分 → 0.5 阈值判定", language=None)
        st.caption("训练时利用标签计算损失、反向传播和更新权重；检测时只使用已训练好的权重。")
        return

    # 训练记录来自开始检测时的快照，交互时不会读入其他版本的曲线。
    st.caption(f"报告来源：A 的训练记录 · 模型版本：{report['model_version']} · 框架：{report['framework']}")
    st.code(report["architecture"], language=None)
    config = pd.DataFrame([{"配置项": key, "实际记录": str(value)} for key, value in report["training"].items()])
    with st.expander("查看实际训练配置与数据划分", expanded=False):
        st.dataframe(config, hide_index=True, width="stretch")
    history = pd.DataFrame(report["history"])
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**训练与验证损失**")
        figure = go.Figure()
        for field, name, color in [("train_loss", "训练损失", "#0d9488"), ("val_loss", "验证损失", "#ea580c")]:
            figure.add_scatter(x=history.epoch, y=history[field], name=name, mode="lines+markers", line_color=color)
        figure.update_layout(height=300, xaxis_title="训练轮次 Epoch", yaxis_title="Loss", legend_orientation="h")
        st.plotly_chart(figure, width="stretch", key="training_loss")
    with right, st.container(border=True):
        st.markdown("**验证集指标**")
        figure = go.Figure()
        for field, name in [("val_f1", "验证 F1"), ("val_accuracy", "验证 Accuracy")]:
            if field in history and history[field].notna().any():
                figure.add_scatter(x=history.epoch, y=history[field], name=name, mode="lines+markers")
        if figure.data:
            figure.update_layout(height=300, xaxis_title="训练轮次 Epoch", yaxis=dict(title="指标", range=[0, 1.05], tickformat=".0%"))
            st.plotly_chart(figure, width="stretch", key="training_validation")
        else:
            st.info("A 未提供逐轮验证指标，当前只展示实际损失记录。")
    st.caption("训练曲线只描述报告中的训练/验证过程，不能替代当前检测数据上的评估，也不自动判定是否过拟合。")
    exported = history.assign(model_version=report["model_version"], dataset_id=run["dataset"].dataset_id,
                              run_mode="real", data_notice="真实训练记录")
    st.download_button("下载训练记录 CSV", csv_bytes(exported), file_name="mlp_training_history.csv", on_click="ignore")
    st.download_button("下载原始训练报告 JSON", json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name="mlp_training_report.json", mime="application/json", on_click="ignore")


def render_deep_learning(run):
    """以 MLP 为主展示排序能力、得分分布和边界样本，保留基线对照。"""
    results = run["results"]
    mlp = results.loc[results.model_id.eq("mlp")]
    st.subheader("深度学习主模型 · MLP", anchor=False)
    if mlp.empty:
        st.info("本次没有 MLP 结果。请在侧边栏选择 MLP 并重新开始检测；随机森林仅作为对照基线。")
        return
    st.caption(f"当前分析：{mlp.model_name.iloc[0]} · 版本 {mlp.model_version.iloc[0]} · 固定判定阈值 0.50")
    curves = score_curves(mlp)
    points, summary = curves["points"], curves["summary"]
    # 大数据时只对绘图点限量，AUC/AP 和 CSV 仍使用全部曲线点。
    plot_points = points.iloc[np.unique(np.linspace(0, len(points) - 1, min(1000, len(points)), dtype=int))] if len(points) else points
    scored = scored_records(mlp)
    width = st.slider("阈值附近观察范围（不改变检测阈值）", .01, .20, .10, .01, key="mlp_band")
    near = scored.loc[scored.distance_to_threshold.le(width + 1e-12)]
    for column, title, value in zip(st.columns(4), ["ROC-AUC", "AP · 平均精确率", "有效评估记录", "阈值附近记录"],
                                     [decimal(summary["ROC_AUC"]), decimal(summary["Average_Precision"]),
                                      str(summary["ranking_evaluated_records"]), str(len(near))]):
        column.metric(title, value)
    st.caption(f"曲线使用 MLP 自身预测成功、得分有效且标签为 0/1 的 {summary['ranking_evaluated_records']} 条记录，排除 {summary['ranking_excluded_records']} 条。")
    if summary["positive_records"] and not summary["negative_records"]:
        st.info("当前只有攻击记录。AP 可以计算，但不能据此判断模型区分正常流量与攻击的能力；ROC-AUC 显示 N/A。")

    # 曲线遍历现有得分的不同切分点，不会修改保存的预测标签或执行新预测。
    fixed = evaluate_results(mlp)
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**ROC · 误报与检出的权衡**")
        if summary["ROC_AUC"] is None:
            st.info(summary["ROC_AUC_reason"])
        else:
            figure = curve_figure("误报率 FPR", "检出率 TPR / Recall")
            figure.add_scatter(x=plot_points.FPR, y=plot_points.TPR, name="MLP ROC", mode="lines", line_color="#0d9488")
            figure.add_scatter(x=[0, 1], y=[0, 1], name="随机排序参考", mode="lines", line=dict(dash="dash", color="#94a3b8"))
            figure.add_scatter(x=[fixed["FPR"]], y=[fixed["Recall"]], name="当前阈值 0.5", mode="markers", marker=dict(color="#ea580c", size=10))
            st.plotly_chart(figure, width="stretch", key="mlp_roc")
    with right, st.container(border=True):
        st.markdown("**PR · 告警精确率与召回率**")
        if summary["Average_Precision"] is None:
            st.info(summary["Average_Precision_reason"])
        else:
            figure = curve_figure("召回率 Recall", "精确率 Precision")
            figure.add_scatter(x=plot_points.Recall, y=plot_points.Precision, name="MLP PR", mode="lines+markers", line_color="#0d9488", line_shape="vh")
            if fixed["Precision"] is not None:
                figure.add_scatter(x=[fixed["Recall"]], y=[fixed["Precision"]], name="当前阈值 0.5", mode="markers", marker=dict(color="#ea580c", size=10))
            st.plotly_chart(figure, width="stretch", key="mlp_pr")
    st.caption("ROC-AUC 衡量得分排序的区分能力；AP 按召回率增量加权汇总精确率，采用非插值计算。五项检测指标仍以固定阈值 0.5 为准。")
    if len(points) > len(plot_points):
        st.caption(f"绘图从 {len(points)} 个实际曲线点中采样 {len(plot_points)} 个；指标和下载仍使用完整数据。")
    if not points.empty:
        exported = points.assign(dataset_id=mlp.dataset_id.iloc[0], run_mode=mlp.run_mode.iloc[0],
                                 data_notice=mlp.data_notice.iloc[0], model_id="mlp", model_version=mlp.model_version.iloc[0],
                                 threshold=.5, curve_scope="individual")
        st.download_button("下载 MLP 曲线点 CSV", csv_bytes(exported), file_name=f"{mlp.run_mode.iloc[0]}_mlp_curves.csv", on_click="ignore")

    st.markdown("**攻击得分分布与边界样本**")
    if scored.empty:
        st.info("没有预测成功且得分有效的记录，暂不能分析得分分布。")
    else:
        figure = go.Figure()
        colors = {"真实正常": "#16a34a", "真实攻击": "#ea580c", "标签缺失": "#94a3b8", "标签非法": "#475569"}
        for group, frame in scored.groupby("score_group", sort=False):
            # 固定分箱包含最右端 1.0，避免满分记录落在图外。
            counts, edges = np.histogram(frame.attack_score, bins=np.linspace(0, 1, 21))
            figure.add_bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=.048,
                           name=group, marker_color=colors[group], opacity=.65)
        figure.add_vline(x=.5, line_dash="dash", line_color="#0d9488", annotation_text="固定阈值 0.5")
        figure.update_layout(height=280, barmode="overlay", xaxis=dict(title="攻击得分", range=[0, 1]), yaxis_title="记录数", legend_orientation="h")
        st.plotly_chart(figure, width="stretch", key="mlp_scores")
    st.caption(f"筛选 |得分 − 0.5| ≤ {width:.2f} 的 {len(near)} 条记录，按距离从近到远展示。靠近阈值只表示判定接近边界，得分不是已校准的攻击概率。")
    columns = ["record_id", "source_row", "attack_score", "distance_to_threshold", "predicted_label", "true_label", "error_type"]
    st.dataframe(near[columns].head(100), hide_index=True, width="stretch")
    st.download_button("下载阈值附近记录 CSV", csv_bytes(near), file_name=f"{mlp.run_mode.iloc[0]}_mlp_near_threshold.csv", on_click="ignore")

    # 基线表严格沿用共同有效记录，不能拿各自不同样本上的 AUC 作比较。
    common = run["metrics"].loc[run["metrics"].scope.eq("common")]
    if not common.empty:
        with st.expander("MLP 与随机森林基线对照（共同样本）", expanded=False):
            st.caption(f"共同成功且标签有效的记录数：{int(common.common_evaluation_records.iloc[0])}。")
            shown = common[["model_name", "evaluated_records", "Accuracy", "Recall", "F1", "FPR", "ROC_AUC", "Average_Precision"]].copy()
            for name in ["Accuracy", "Recall", "F1", "FPR"]:
                shown[name] = shown[name].map(lambda value: "N/A" if pd.isna(value) else f"{value:.2%}")
            for name in ["ROC_AUC", "Average_Precision"]:
                shown[name] = shown[name].map(decimal)
            st.dataframe(shown, hide_index=True, width="stretch")
    render_training_report(run)
