"""界面入口：配置检测 → 点击执行 → 展示分析 → 下载结果。

业务计算交给其他模块，本文件负责界面和用户操作。
Streamlit 交互时会重跑脚本，因此用 session_state 保留已完成的结果。
"""
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demo_data import DEMO_METADATA, DEMO_PATH, DemoModel, demo_dataset
from deep_learning_view import render_deep_learning
from detection_service import (THRESHOLD, filter_results, read_dataset, run_configuration, run_detection)
from evaluation import (ERROR_NAMES, METRICS, build_metrics_table, category_summary,
                        confusion_matrix, evaluate_results, model_disagreements)
from export_results import csv_bytes
from model_registry import MODEL_DIR, REAL_NAMES, discover_real_models, load_real_model
from training_report import load_training_report

# 1. 页面设置：统一标题、宽屏布局和卡片配色。
st.set_page_config(page_title="网络流量检测分析工作台", page_icon="🛡️", layout="wide")
st.markdown("""<style>
.block-container {max-width:1480px; padding-top:3.5rem; padding-bottom:2rem;}
[data-testid="stMetric"] {background:white; border:1px solid #e2e8f0;
 border-radius:12px; padding:18px 20px; border-top:3px solid #0d9488;}
[data-testid="stMetricLabel"] {color:#64748b;}
[data-testid="stMetricValue"] {font-weight:650;}
.st-key-attack_card [data-testid="stMetric"] {border-top-color:#ea580c;}
.st-key-attack_card [data-testid="stMetricValue"] {color:#c2410c;}
.st-key-normal_card [data-testid="stMetric"] {border-top-color:#16a34a;}
.st-key-normal_card [data-testid="stMetricValue"] {color:#15803d;}
[data-testid="stSidebar"] {border-right:1px solid #e2e8f0;}
div[data-testid="stVerticalBlockBorderWrapper"] > div {border-radius:12px;}
.workbench-kicker {color:#0d9488; font-size:12px; letter-spacing:2px; font-weight:700;}
.legend {font-size:13px; color:#64748b; margin-bottom:18px;}
.legend .normal {color:#15803d;} .legend .attack {color:#ea580c;}
</style>""", unsafe_allow_html=True)


# 2. 展示函数：将结果整理成指标表、图表和可筛选的流量表。
def percent(value):
    """把小数显示为两位百分比；无法计算时显示 N/A。"""
    return "N/A" if value is None or pd.isna(value) else f"{value:.2%}"


def display_metrics(frame):
    """展示指标及有效样本数，只格式化副本，保留原始计算精度。"""
    shown = frame.copy()
    for metric in METRICS:
        shown[metric] = shown[metric].map(percent)
    columns = ["model_name", "successful_records", "failed_records", "evaluated_records", "missing_labels", "invalid_labels", *METRICS]
    st.dataframe(shown[columns].rename(columns={"model_name": "模型", "successful_records": "成功预测",
                 "failed_records": "处理失败", "evaluated_records": "参与评估", "missing_labels": "缺失标签", "invalid_labels": "非法标签"}),
                 hide_index=True, width="stretch")


def plot_confusion(frame, key):
    """画真实类别与预测类别的对照图，横轴预测、纵轴真实。"""
    matrix = confusion_matrix(frame)
    # 四格依次为 TN、FP、FN、TP，可直观看出误报和漏报。
    figure = go.Figure(go.Heatmap(z=matrix, x=["正常 0", "攻击 1"], y=["正常 0", "攻击 1"],
                                text=[[f"TN · {matrix[0, 0]}", f"FP 误报 · {matrix[0, 1]}"],
                                      [f"FN 漏报 · {matrix[1, 0]}", f"TP · {matrix[1, 1]}"]],
                                texttemplate="%{text}", colorscale=[[0, "#f0fdfa"], [1, "#0d9488"]],
                                showscale=False, hovertemplate="真实 %{y}<br>预测 %{x}<br>数量 %{z}<extra></extra>"))
    figure.update_layout(height=315, margin=dict(l=20, r=20, t=15, b=25),
                         xaxis_title="预测类别", yaxis_title="真实类别", yaxis_autorange="reversed",
                         paper_bgcolor="white", font=dict(color="#334155"))
    st.plotly_chart(figure, width="stretch", key=key)
    st.caption(f"仅使用预测成功且标签有效的 {int(matrix.sum())} 条记录。")


def plot_comparison(metrics, key):
    """用分组柱状图比较模型；每种颜色代表一个模型。"""
    figure = go.Figure()
    for index, (_, row) in enumerate(metrics.iterrows()):
        values = [row[name] for name in METRICS]
        figure.add_bar(name=row.model_name, x=METRICS, y=values,
                       marker_color=["#0d9488", "#64748b"][index % 2],
                       hovertemplate="%{x}: %{y:.2%}<extra>%{fullData.name}</extra>")
    figure.update_layout(barmode="group", height=315, margin=dict(l=15, r=15, t=15, b=25),
                         yaxis=dict(range=[0, 1.08], tickformat=".0%", gridcolor="#edf2f7"),
                         legend=dict(orientation="h", y=1.14), paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(figure, width="stretch", key=key)
    st.caption("召回率越高表示漏报越少；FPR 越低表示正常流量受误报影响越小。N/A 不绘制柱形。")


def traffic_table(results, dataset, prefix):
    """展示筛选、分页、下载和详情；全程使用已有预测结果。"""
    st.subheader("流量记录", anchor=False)
    # prefix 区分两个页签中的同类控件，避免控件 key 重复。
    controls = st.columns([1.3, 1, 1.1, 1.5])
    available = results.model_id.unique().tolist()
    names = results.drop_duplicates("model_id").set_index("model_id").model_name.to_dict()
    models = controls[0].multiselect("模型筛选", available, default=available,
                                    format_func=lambda key: names[key], key=f"{prefix}_models")
    prediction = controls[1].selectbox("预测类别", ["all", "1", "0"],
                                    format_func=lambda key: {"all": "全部预测", "1": "攻击告警", "0": "正常"}[key], key=f"{prefix}_prediction")
    error = controls[2].selectbox("判断状态", ["all", *ERROR_NAMES],
                               format_func=lambda key: "全部状态" if key == "all" else ERROR_NAMES[key], key=f"{prefix}_error")
    query = controls[3].text_input("编号查询", placeholder="record_id / 原始 id / 行号", key=f"{prefix}_query")
    # 排序与分页：filtered 是全部匹配结果，view 只是当前页。
    sorting = st.columns([1.5, 1, 1, 1])
    sort_by = sorting[0].selectbox("排序字段", ["source_row", "attack_score", "record_id", "model_name"], key=f"{prefix}_sort")
    descending = sorting[1].checkbox("降序排列", key=f"{prefix}_descending")
    page_size = sorting[2].selectbox("每页条数", [25, 50, 100], key=f"{prefix}_size")
    filtered = filter_results(results, models, prediction, error, query, sort_by, descending)
    pages = max(1, math.ceil(len(filtered) / page_size))
    page = sorting[3].number_input("页码", min_value=1, value=1, step=1, key=f"{prefix}_page")
    page = min(page, pages)
    view = filtered.iloc[(page - 1) * page_size:page * page_size]
    st.caption(f"筛选后 {len(filtered):,} 条预测记录 · 第 {page}/{pages} 页；导出包含所有符合筛选条件的记录。")
    # 表格只选实际存在的列，并将状态转成中文文字。
    columns = ["record_id", "source_row", "id", "proto", "service", "model_name", "attack_score", "predicted_label", "true_label", "attack_category", "error_type", "failure_reason"]
    shown = view[[col for col in columns if col in view]].copy()
    shown["error_type"] = shown.error_type.map(ERROR_NAMES)
    shown["predicted_label"] = shown.predicted_label.map({0: "正常 · 0", 1: "攻击告警 · 1"}).fillna("处理失败")
    st.dataframe(shown, hide_index=True, width="stretch", height=330,
                 column_config={"attack_score": st.column_config.NumberColumn("攻击得分", format="%.4f"),
                                "source_row": "CSV 原始行号", "record_id": st.column_config.TextColumn("record_id", width="medium")})
    # 下载所有匹配记录，而非仅下载当前页；下载不触发重新推理。
    st.download_button("下载当前筛选结果 CSV", csv_bytes(filtered),
                       file_name=f"{results.run_mode.iloc[0]}_{dataset.dataset_id[:12]}_filtered.csv", mime="text/csv",
                       key=f"{prefix}_download", on_click="ignore")
    if view.empty:
        st.info("当前筛选条件下没有记录，可调整筛选条件。")
        return
    # 详情通过结果索引选取，再用 record_id 找回原始记录。
    selected = st.selectbox("选择记录查看详情（当前页）", view.index.tolist(),
                            format_func=lambda idx: f"行 {results.loc[idx, 'source_row']} · {results.loc[idx, 'model_name']} · {ERROR_NAMES[results.loc[idx, 'error_type']]}",
                            key=f"{prefix}_detail")
    row = results.loc[selected]
    with st.expander("展开单条流量详情", expanded=False):
        st.code(row.record_id, language=None)
        cols = st.columns(3)
        cols[0].metric("攻击得分", "N/A" if pd.isna(row.attack_score) else f"{row.attack_score:.6f}")
        cols[1].metric("固定阈值", f"{row.threshold:.2f}")
        cols[2].metric("判断状态", ERROR_NAMES[row.error_type])
        # 失败只展示原因；成功才展示得分与阈值的比较。
        if row.prediction_status == "failed":
            st.error(row.failure_reason)
        else:
            text = f"攻击得分 {row.attack_score:.6f} {'≥' if row.predicted_label == 1 else '<'} {row.threshold:.2f}，预测为{'攻击' if row.predicted_label == 1 else '正常'}。"
            (st.warning if row.predicted_label == 1 else st.success)(text)
        st.write(f"原始标签：{row.true_label if pd.notna(row.true_label) else '缺失'}；标签状态：{row.label_status}；攻击类别：{row.attack_category if pd.notna(row.attack_category) else '未提供'}。")
        st.caption("以下为导入时的原始字段；得分与标签对照只说明本次判定，不能据此推断攻击原因。")
        position = dataset.records.index[dataset.records.record_id.eq(row.record_id)][0]
        original = dataset.raw.loc[position].rename("原始值").rename_axis("字段").reset_index()
        original["原始值"] = original["原始值"].astype("string")
        st.dataframe(original, hide_index=True, width="stretch")


# 3. 页面标题：说明系统用途，并用文字和颜色区分正常与攻击。
st.markdown('<div class="workbench-kicker">NETWORK SECURITY / 中期项目</div>', unsafe_allow_html=True)
st.title("网络流量检测分析工作台", anchor=False)
st.caption("深度学习 MLP 主模型 · 随机森林基线对照 · 流量检测与评估")
st.markdown('<div class="legend"><span class="normal">● 正常流量</span>　<span class="attack">● 攻击告警</span>　固定判定阈值 0.50</div>', unsafe_allow_html=True)

# 4. 侧边栏：准备数据、运行模式和模型，不在这里执行预测。
with st.sidebar:
    st.subheader("检测配置", anchor=False)
    mode_text = st.radio("运行模式", ["界面调试模式", "真实检测"], key="mode")
    mode = "debug" if mode_text == "界面调试模式" else "real"
    upload = st.file_uploader("导入流量 CSV", type=["csv"], disabled=mode == "debug", key="upload")
    dataset = None
    # 调试模式只用固定示例；真实模式读取上传文件并检查 A 的接口。
    if mode == "debug":
        st.warning("界面调试数据：固定示例得分，不能用于真实实验报告。")
        dataset = demo_dataset()
        metadata, issues = DEMO_METADATA, {}
        st.download_button("下载固定示例 CSV", DEMO_PATH.read_bytes(), file_name="demo_flows.csv", mime="text/csv", on_click="ignore")
    else:
        metadata, issues = discover_real_models()
        if upload is not None:
            try:
                dataset = read_dataset(upload.getvalue(), upload.name)
            except ValueError as exc:
                st.error(str(exc))
    # 只有通过接口检查的模型才进入可选列表，缺失原因另行提示。
    # 默认优先排列 MLP，保留随机森林作为基线；仍允许同时检测和公平比较。
    choices = [key for key in ["mlp", "rf"] if key in metadata]
    model_ids = st.multiselect("选择检测模型", choices, default=choices,
                               format_func=lambda key: f"{metadata[key]['name']} · {'主模型' if key == 'mlp' else '基线'}", key=f"models_{mode}")
    st.caption("MLP 是深度学习主模型；随机森林用于相同数据上的基线对照。")
    for key in REAL_NAMES:
        if key in metadata:
            st.caption(f"● {metadata[key]['name']} 可用 · {metadata[key]['version']}")
        elif mode == "real":
            st.error(f"{REAL_NAMES[key]} 不可用：{issues[key]}")
    batch_size = int(st.number_input("每批记录数", min_value=1, max_value=8192, value=256, step=1, key="batch_size"))
    # 显示当前数据身份，防止选错文件；空数据不能开始检测。
    if dataset is not None:
        st.write(f"数据集：{dataset.name}")
        st.write(f"样本数量：{len(dataset.raw):,}")
        st.caption(f"文件标识：{dataset.dataset_id[:16]}…")
        if dataset.raw.empty:
            st.error("CSV 只有表头，没有可检测记录。")
    else:
        st.caption("请上传 CSV；标签可缺失，模型必需特征必须完整。")
    start = st.button("开始检测", type="primary", width="stretch",
                      disabled=dataset is None or not model_ids or dataset.raw.empty, key="start")

# 5. 执行检测：记录配置，只有按钮被点击时才调用检测服务。
current_config = run_configuration(dataset, mode, model_ids, metadata, batch_size) if dataset else None
# 始终保留相同的主区占位，防止首次筛选重跑后原生页签位置变化、跳回首页。
inference_progress = st.empty()
inference_spinner = st.empty()
inference_error = st.empty()
if start:
    progress = inference_progress.progress(0, text="正在校验输入并加载模型…")
    def update_progress(completed, total, message):
        """接收检测服务的回调，按真实已尝试任务数更新进度。"""
        progress.progress(completed / total, text=f"{message} · {completed}/{total} 条模型任务已尝试")
    # 一次提交整体替换旧运行，重跑失败模型时选择该模型即可，不追加重复结果。
    st.session_state.pop("run", None)
    try:
        with inference_spinner.container(), st.spinner("正在执行批量检测…"):
            results, failures = run_detection(dataset, model_ids, metadata, DemoModel if mode == "debug" else load_real_model,
                                               mode, batch_size, update_progress)
        # 训练报告只读取一次并核对版本；报告缺失不会阻止预测或评估。
        training_report, training_issue = (None, "本次未选择 MLP 主模型。")
        if "mlp" in model_ids:
            training_report, training_issue = load_training_report(
                MODEL_DIR / "mlp_training_report.json", metadata["mlp"]["version"], mode)
        # 将配置、原始数据、结果、错误、指标和训练报告保存为同一次运行。
        st.session_state["run"] = {"config": current_config, "dataset": dataset, "results": results,
                                   "failures": failures, "metrics": build_metrics_table(results),
                                   "training_report": training_report, "training_issue": training_issue}
        st.session_state["run_count"] = st.session_state.get("run_count", 0) + 1
    except Exception as exc:
        inference_error.error(f"本次运行未完成：{type(exc).__name__}: {exc}。可修正配置后重新检测。")
    finally:
        progress.empty()
        inference_spinner.empty()

# 6. 结果检查：没有结果或配置已变化时，停止展示和导出。
run = st.session_state.get("run")
if run is None:
    st.info("准备开始：在侧边栏选择模型并点击“开始检测”。默认调试模式可直接体验完整流程。")
    st.stop()
if current_config != run["config"]:
    st.warning("检测配置已改变，需要重新点击“开始检测”。旧结果已隐藏，当前不能查看或导出。")
    st.stop()

# 7. 检测摘要：提示成功/部分失败/全部失败，汇总顶部四张卡片。
results, dataset, metrics = run["results"], run["dataset"], run["metrics"]
if mode == "debug":
    st.warning("界面调试数据 · 以下全部卡片、图表与导出均来自固定示例，不能作为真实模型性能结论。")
success = results.prediction_status.eq("success")
failed_count = int((~success).sum())
if failed_count == len(results):
    st.error("全部检测失败。请查看错误范围及原因，修正后重新运行。")
elif failed_count:
    st.warning(f"部分检测失败：{failed_count} 条模型任务失败，其他结果已保留；失败记录不参与指标。")
else:
    st.success("检测完成，当前结果已保存；筛选、页签切换、查看详情和下载均复用本次结果。")
st.progress(1.0, text=f"已完成尝试 {len(results)}/{len(results)} 条模型任务 · 预测成功 {int(success.sum())} 条")
st.caption(f"当前运行：{dataset.name} · {len(dataset.raw)} 条独立流量 · {results.model_id.nunique()} 个模型 · 以下按“流量 × 模型”计数。")
# 两模型会各预测一次，因此卡片按“流量 × 模型”计数。
cards = st.columns(4)
for col, title, value, card_key in zip(cards, ["已处理记录数", "攻击告警数", "正常预测数", "处理失败数"],
                              [int(success.sum()), int(results.predicted_label.eq(1).sum()), int(results.predicted_label.eq(0).sum()), failed_count],
                              ["processed_card", "attack_card", "normal_card", "failed_card"]):
    with col, st.container(key=card_key):
        st.metric(title, f"{value:,}")
if run["failures"]:
    with st.expander("查看失败模型、记录范围及原因", expanded=True):
        st.dataframe(pd.DataFrame(run["failures"]), hide_index=True, width="stretch")

# 8. 深度学习分析放在首个页签，原有三个操作页签继续复用本次结果。
learning_tab, overview, records_tab, analysis = st.tabs(["深度学习分析", "检测概览", "流量记录", "误报与漏报分析"])
with learning_tab:
    render_deep_learning(run)

with overview:
    # 概览先报告各模型的有效数量，再展示所选模型的五项指标。
    individual = metrics.loc[metrics.scope.eq("individual")]
    st.subheader("模型评估", anchor=False)
    display_metrics(individual)
    focus = st.selectbox("查看模型指标与混淆矩阵", results.model_id.unique().tolist(),
                         format_func=lambda key: metadata[key]["name"], key="focus")
    selected_results = results.loc[results.model_id.eq(focus)]
    selected_metrics = evaluate_results(selected_results)
    if selected_metrics["evaluated_records"] == 0:
        st.info("当前模型没有预测成功且标签有效的记录。仍可查看预测与导出，评估指标显示 N/A。")
    for col, name in zip(st.columns(5), METRICS):
        col.metric(name, percent(selected_metrics[name]))
    st.caption(f"参与评估 {selected_metrics['evaluated_records']} · 缺失标签 {selected_metrics['missing_labels']} · 非法标签 {selected_metrics['invalid_labels']} · 处理失败 {selected_metrics['failed_records']}。N/A 表示该指标分母为零。")
    # 双图并排：比较图使用共同评估样本，矩阵使用当前所选模型。
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**模型指标比较**")
        common = metrics.loc[metrics.scope.eq("common")]
        if len(individual) > 1:
            st.caption(f"公平比较使用共同成功且标签有效的 {int(common.common_evaluation_records.iloc[0])} 条记录。")
            plot_comparison(common, "comparison")
        else:
            st.info("选择两个模型完成检测后，可在共同评估记录上比较指标。")
            plot_comparison(individual, "single_metrics")
    with right, st.container(border=True):
        st.markdown(f"**混淆矩阵 · {metadata[focus]['name']}**")
        plot_confusion(selected_results, "overview_confusion")
    if len(individual) > 1:
        st.markdown("**共同评估样本上的指标比较表**")
        display_metrics(common)
        st.caption("上方模型评估表报告各自样本数；本表和比较图使用同一批 record_id，避免不同成功集合造成不公平比较。")
    traffic_table(results, dataset, "overview")

with records_tab:
    # 复用表格函数，提供独立筛选和单条记录详情。
    traffic_table(results, dataset, "records")

with analysis:
    # 先选模型，再查看矩阵、攻击类别和误报/漏报记录。
    model = st.selectbox("分析模型", results.model_id.unique().tolist(), format_func=lambda key: metadata[key]["name"], key="analysis_model")
    model_results = results.loc[results.model_id.eq(model)]
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**真实与预测对照**")
        plot_confusion(model_results, "analysis_confusion")
    with right, st.container(border=True):
        st.markdown("**按真实攻击类别统计**")
        categories = category_summary(model_results)
        if categories.empty:
            st.info("没有标签有效的攻击记录，暂不能分析攻击类别。")
        else:
            st.dataframe(categories, hide_index=True, width="stretch")
        st.caption("只统计真实标签为 1 的记录；失败单独计数，不计作漏报。缺失类别显示“未提供类别”。")
    # 误报：正常被告警；漏报：攻击被判正常。只展示有有效标签的错误。
    error = st.radio("错误样本筛选", ["false_positive", "false_negative"], format_func=lambda key: ERROR_NAMES[key], horizontal=True)
    errors = model_results.loc[model_results.error_type.eq(error)]
    st.write(f"{ERROR_NAMES[error]}：{len(errors)} 条")
    st.dataframe(errors[["record_id", "source_row", "attack_score", "threshold", "predicted_label", "true_label", "attack_category"]], hide_index=True, width="stretch")
    # 分歧只说明判断不同，不依赖标签，也不代表谁一定正确。
    st.markdown("**两模型判断不一致的记录**")
    disagreements = model_disagreements(results)
    st.caption("按 record_id 一对一对齐，只比较两个模型均预测成功的记录；无标签也可比较预测差异。")
    if disagreements.empty:
        st.info("当前无可展示的分歧记录，或尚未同时检测两个模型。")
    else:
        st.dataframe(disagreements, hide_index=True, width="stretch")

# 9. 完整导出：分别下载本次全部预测和指标，保留运行标识。
st.divider()
st.subheader("本次运行导出", anchor=False)
left, right = st.columns(2)
left.download_button("下载本次完整预测结果 CSV", csv_bytes(results), file_name=f"{mode}_{dataset.dataset_id[:12]}_predictions.csv", mime="text/csv", on_click="ignore")
right.download_button("下载模型指标比较 CSV", csv_bytes(metrics), file_name=f"{mode}_{dataset.dataset_id[:12]}_metrics.csv", mime="text/csv", on_click="ignore")
st.caption("CSV 使用 UTF-8 BOM。指标文件另含 ROC_AUC 与 Average_Precision（AP）；保留完整精度，空指标的原因写入 *_reason 字段。")
