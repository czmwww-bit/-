# B 部分验证与交接记录

日期：2026-11-16。最初项目仅有 README；未发现 A 的实现或模型。因此以下性能数字全部为明确标识的界面调试数据，不能作为真实实验结论。

## 环境

Windows，Python 3.12.14（64 位），项目内 `.venv`。最初终端没有 `python` 命令，找到本机可用解释器后创建独立虚拟环境；没有修改其他 Python 环境。

| 直接依赖 | 实际安装、验证版本 |
| --- | --- |
| Streamlit | 1.51.0 |
| pandas | 2.3.3 |
| NumPy | 2.3.5 |
| Plotly | 6.4.0 |

完整间接依赖见 `requirements-lock.txt`。`python -m pip check` 返回 `No broken requirements found.`。安装时使用 `--no-cache-dir` 避免项目外缓存写入问题。

## 已执行的检查

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q app.py evaluation.py detection_service.py export_results.py model_registry.py demo_data.py tests
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501
```

16 项测试全部通过（13 项核心测试、3 项 AppTest 测试方法，其中包含多个场景）。模块编译通过。服务健康端点 `http://127.0.0.1:8501/_stcore/health` 返回 `ok`。

| 验证范围 | 结果与证据 |
| --- | --- |
| 手算 TP=40、TN=50、FP=5、FN=5 | Accuracy=0.9；Precision/Recall/F1=8/9；FPR=1/11 |
| 没有告警、全正常、全攻击、空结果 | 分母为零返回 None，并生成原因；空 CSV/仅表头明确拒绝检测 |
| 缺失/非法标签、无标签数据 | 独立计数；不将缺失或非法标签判为正确/误报/漏报；AppTest 无标签界面可检测、显示 N/A |
| 阈值边界 | 0.5 判为攻击；固定阈值写入结果和导出 |
| 数据身份与 CSV | 同内容改名身份不变、重复原始 id 不影响 record_id、空行与多行字段物理行号正确；重复表头、字段数不一致、非法引号拒绝导入 |
| 特征输入 | 排除标签/id/预测/评估字段；遵循必需列与列顺序；保留缺失；非法数字提供原始行号 |
| 输出校验 | 拒绝二维、长度不符、非数组、文本/复数、NaN、无穷、低于 0 或高于 1 的得分 |
| 分批与失败 | 实际任务数更新进度；按位置回填 record_id；部分失败继续其他批次；失败标签保持空；全部失败有反馈 |
| 公平比较 | 故意让 RF 某一批失败：独立评估数量 7/10，共同比较均为 7 |
| 筛选、指标、类别与分歧 | 精确核对 RF 混淆矩阵、DoS 漏报、5 条模型分歧、筛选导出与原结果对应 |
| 会话状态 | 检测前调用数 0；检测后筛选/查询/重跑页面仍为 1；改模型隐藏旧结果和结果下载；再次检测替换成 12 行单模型结果，未追加；切真实模式不混用调试结果 |
| 配置身份 | 数据内容、模式、模型、版本变化会产生不同运行配置；配置包含阈值、特征约定和分批大小 |
| 真实模型缺失 | 未交付 `model_adapter.py` 时，两个模型均显示缺失接口；没有可用模型时开始按钮禁用 |

## 浏览器实测

使用本机 Chrome 的无头模式与 DevTools 协议检查真实页面，未安装额外前端组件库。检查了约 1440 像素窗口和 1280×800 CSS 视口：无页面横向溢出；宽流量表在自己的容器内滚动。

完成“调试开始 → 概览与双图 → 流量页签 → 编号查询 → 展开详情 → 三类下载 → 错误分析 → 返回概览”。页面没有 Streamlit 异常。浏览器实测发现并修复两处问题：纯数字查询会命中文件哈希；首次筛选因临时进度组件改变布局位置而跳回概览。现在纯数字精确匹配原始 id/行号，推理状态使用固定占位；实测筛选后页签保留（报告 `filterTabPreserved=true`）。

下载的文件已使用 pandas 重新读取并断言内容：

- 筛选 CSV：查询原始行号 3，2 条记录；RF 得分 0.76、预测攻击、真实正常、误报；MLP 得分 0.21、预测正常、判断正确。
- 完整预测 CSV：24 行，对应 12 条独立流量 × 2 模型；`record_id + model_id` 无重复；13 次攻击告警、11 次正常预测、0 次失败。
- 指标 CSV：4 行，两个模型各有 individual/common；两模型共同参与评估 10 条；独立口径各有 1 条缺失标签、1 条非法标签。
- 三类 CSV 均以 `EF BB BF` 开头，包含 `run_mode=debug`、`data_notice=界面调试数据`；预测阈值为 0.5。

| 固定调试模型 | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 | FPR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RF 示例 | 4 | 3 | 1 | 2 | 70.00% | 80.00% | 66.67% | 72.73% | 25.00% |
| MLP 示例 | 5 | 3 | 1 | 1 | 80.00% | 83.33% | 83.33% | 83.33% | 25.00% |

截图、实际下载文件与浏览器 JSON 报告位于 `results/browser/`（运行产物，已被 gitignore 排除）：

- [1280 视口概览](results/browser/overview_1280.png)
- [概览截图](results/browser/overview_1440.png)
- [模型比较与混淆矩阵](results/browser/charts_1440.png)
- [筛选结果与展开详情](results/browser/records_1440.png)
- [浏览器检查报告](results/browser/browser_report.json)

## 尚待真实模型联调

未验证真实 RF/MLP 模型加载、预处理与训练版本兼容、真实 UNSW-NB15 验证集的特征契约、实际推理结果、重启后模型结果一致性、真实规模下的耗时/内存与模型性能。尚未在 Windows Excel 应用中手工打开 CSV；已验证 BOM、中文字段与 pandas 回读。

当前 A 接入契约是 B 的预留桥接方案，未假设已经存在。A 的交接文件、调用示例和版本要求见 [models/README.md](models/README.md)。交付后首先映射 `model_registry.py` 的元数据和加载两个调用，再用 A 提供的固定小型验证 CSV 和真实得分核对；预处理、RF 和 MLP 实现仍归 A。

中期演示按 README 的 3 分钟脚本执行。调试样例用于证明 B 的操作与评估链路，汇报中应明确真实模型仍待交付。
