"""B 的唯一真实加载桥接处。A 未交付前不猜测权重、处理器或网络结构。"""
import importlib
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
REAL_NAMES = {"rf": "随机森林", "mlp": "PyTorch MLP"}


def discover_real_models() -> tuple[dict, dict]:
    metadata, issues = {}, {}
    try:
        adapter = importlib.import_module("model_adapter")
        if not callable(getattr(adapter, "load_model", None)):
            raise ValueError("缺少 model_adapter.load_model(model_id, models_dir) 加载入口")
        if not callable(getattr(adapter, "get_model_metadata", None)):
            raise ValueError("缺少 model_adapter.get_model_metadata(models_dir) 元数据入口")
        supplied = adapter.get_model_metadata(MODEL_DIR)
        if not isinstance(supplied, dict):
            raise ValueError("get_model_metadata 必须返回按 rf / mlp 索引的字典")
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}。待 A 提供 model_adapter.py、完整模型包及特征清单。"
        return metadata, {key: reason for key in REAL_NAMES}
    for key in REAL_NAMES:
        try:
            meta = supplied[key]
            if not isinstance(meta.get("name"), str) or not meta["name"].strip():
                raise ValueError("缺少模型名称 name")
            if not isinstance(meta.get("version"), str) or not meta["version"].strip():
                raise ValueError("缺少模型版本 version（建议使用完整模型包内容哈希）")
            names, dtypes = meta.get("feature_names"), meta.get("feature_dtypes")
            if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
                raise ValueError("缺少有序且不重复的 feature_names")
            if not isinstance(dtypes, dict) or set(dtypes) != set(names) or any(t not in ("numeric", "categorical") for t in dtypes.values()):
                raise ValueError("feature_dtypes 必须覆盖每个输入字段，取值 numeric / categorical")
            metadata[key] = {"name": meta["name"], "version": meta["version"], "feature_names": names, "feature_dtypes": dtypes}
        except Exception as exc:
            issues[key] = f"{type(exc).__name__}: {exc}"
    return metadata, issues


def load_real_model(model_id):
    # A 的函数命名若不同，只需修改本文件两个桥接调用，不修改 A 的实现。
    adapter = importlib.import_module("model_adapter")
    return adapter.load_model(model_id, MODEL_DIR)
