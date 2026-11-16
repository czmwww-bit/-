"""所有下载复用本次结果，UTF-8 BOM 便于 Windows Excel 打开。"""
import pandas as pd
from evaluation import ERROR_NAMES


def csv_bytes(frame: pd.DataFrame) -> bytes:
    exported = frame.copy()
    if "error_type" in exported:
        exported["错误类型说明"] = exported.error_type.map(ERROR_NAMES)
    return exported.to_csv(index=False, na_rep="").encode("utf-8-sig")
