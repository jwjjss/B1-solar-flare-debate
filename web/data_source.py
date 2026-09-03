"""
前端/API 共享的数据源解析器。

设计原则（见 WEB 任务说明）：
  1. 只读静态 JSON，绝不调用 LLM、绝不联网——保证评委离线可复现、零 API 费用；
  2. 数据源优先级：outputs/real_final（真实模式产物，canonical）
     → outputs/real_final_v2（修复重建版）→ shared/mock_data（兜底，防白屏）；
  3. 字段定义以 shared/contracts.py 为准，本模块只读不写、不做任何修改。

api.py 以 ``uvicorn web.api:app`` 从仓库根启动（根目录在 sys.path）；
app.py 以 ``streamlit run web/app.py`` 启动（web/ 目录在 sys.path）。
两种导入路径都用 try/except 兼容。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

# 仓库根目录 = web/ 的上一级
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 数据源候选目录（按优先级）
REAL_DIR = os.path.join(BASE_DIR, "outputs", "real_final")
REAL_V2_DIR = os.path.join(BASE_DIR, "outputs", "real_final_v2")
MOCK_DIR = os.path.join(BASE_DIR, "shared", "mock_data")

# 6 个标准产物文件名
FILE_NAMES = [
    "literature_facts.json",
    "evidence_report.json",
    "transcript.json",
    "causal_graph.json",
    "hypothesis_report.json",
    "run_metadata.json",
]

# 判定一个目录「可用」的最小文件集
_REQUIRED = ("causal_graph.json", "hypothesis_report.json")


class DataSourceError(RuntimeError):
    """数据源缺失或文件不存在时抛出，携带面向用户的友好提示。"""


def candidate_dirs() -> List[str]:
    """按优先级返回候选数据目录列表。"""
    return [REAL_DIR, REAL_V2_DIR, MOCK_DIR]


def resolve_data_dir() -> Optional[str]:
    """返回第一个「存在且含最小文件集」的数据目录；都不可用时返回 None。"""
    for d in candidate_dirs():
        if os.path.isdir(d) and all(os.path.isfile(os.path.join(d, f)) for f in _REQUIRED):
            return d
    return None


def data_dir_label() -> str:
    """返回当前数据源的可读标签，供前端展示「正在看哪份数据」。"""
    d = resolve_data_dir()
    if d is None:
        return "（无可用数据源）"
    if d == REAL_DIR:
        return "真实模式产物 outputs/real_final"
    if d == REAL_V2_DIR:
        return "修复重建版 outputs/real_final_v2"
    if d == MOCK_DIR:
        return "mock 兜底数据 shared/mock_data"
    return os.path.relpath(d, BASE_DIR)


def is_real() -> bool:
    """当前数据源是否为真实模式产物（而非 mock）。"""
    d = resolve_data_dir()
    return d in (REAL_DIR, REAL_V2_DIR)


def available_files() -> Dict[str, bool]:
    """返回每个标准产物文件在当前数据源中是否存在的映射。"""
    d = resolve_data_dir()
    if d is None:
        return {name: False for name in FILE_NAMES}
    return {name: os.path.isfile(os.path.join(d, name)) for name in FILE_NAMES}


def load(name: str) -> Any:
    """从解析出的数据目录加载指定 JSON 文件。

    Raises:
        DataSourceError: 无任何可用数据源，或该文件在当前数据源中缺失。
    """
    d = resolve_data_dir()
    if d is None:
        raise DataSourceError(
            "未找到任何数据源。请先运行 `python infer.py --real --data_root .. "
            "--out outputs/real_final` 生成真实产物，或确认 shared/mock_data 存在。"
        )
    path = os.path.join(d, name)
    if not os.path.isfile(path):
        raise DataSourceError(
            f"当前数据源（{os.path.relpath(d, BASE_DIR)}）缺少 {name}。"
            f"请重新运行 infer.py 补全产物。"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all() -> Dict[str, Any]:
    """一次性加载全部存在的产物文件，缺失的键对应 None。"""
    out: Dict[str, Any] = {}
    for name in FILE_NAMES:
        key = name.replace(".json", "")
        try:
            out[key] = load(name)
        except DataSourceError:
            out[key] = None
    return out
