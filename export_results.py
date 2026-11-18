"""所有下载复用本次结果，UTF-8 BOM 便于 Windows Excel 打开。"""
import pandas as pd
from evaluation import ERROR_NAMES


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """把当前结果表转成可下载的 CSV 字节，不做预测或指标计算。"""
    # 使用副本添加中文错误说明，避免改变会话中的原始结果。
    exported = frame.copy()
    if "error_type" in exported:
        exported["错误类型说明"] = exported.error_type.map(ERROR_NAMES)
    # 不导出 DataFrame 索引；空值留空；utf-8-sig 自带 BOM，便于 Excel 识别中文。
    return exported.to_csv(index=False, na_rep="").encode("utf-8-sig")
