"""
可调用测试 API（FastAPI）· B-1 耀斑前兆因果链系统

端点（对应系统 5 个环节 + 探活 + 元数据）：
  GET /              列出全部可用端点与当前数据源（探活）
  GET /literature    文献因素（literature_facts.json）
  GET /evidence      数据证据（evidence_report.json）
  GET /debate        三角色辩论记录（transcript.json）
  GET /causal_graph  因果图谱（causal_graph.json）
  GET /hypothesis    11 字段科学假设报告（hypothesis_report.json）
  GET /metadata      运行元数据（run_metadata.json）

数据源：只读 outputs/real_final（真实产物）静态 JSON，缺失时回退 mock_data；
绝不调用 LLM、绝不联网，保证评委离线可复现。

运行：uvicorn web.api:app --reload --port 8000
文档：http://localhost:8000/docs
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException

try:  # uvicorn web.api:app（仓库根在 sys.path）
    from web.data_source import (
        DataSourceError, available_files, data_dir_label, is_real, load,
    )
except ImportError:  # 直接以 web/ 为工作目录启动
    from data_source import (  # type: ignore
        DataSourceError, available_files, data_dir_label, is_real, load,
    )

app = FastAPI(
    title="B-1 耀斑前兆因果链 API",
    description="多智能体辩论科学假设生成系统 · 只读真实运行产物的展示接口",
    version="2.0",
)

# 端点 → (产物文件名, 中文说明)
_ENDPOINTS: Dict[str, tuple[str, str]] = {
    "/literature": ("literature_facts.json", "文献因素：arXiv 检索抽取的前兆候选因子"),
    "/evidence": ("evidence_report.json", "数据证据：JW-FD 统计关联分析结果"),
    "/debate": ("transcript.json", "辩论记录：物理学家/质疑者/方法论审查三角色全过程"),
    "/causal_graph": ("causal_graph.json", "因果图谱：节点-边-根因-辩论轮数"),
    "/hypothesis": ("hypothesis_report.json", "科学假设报告：11 字段标准输出"),
    "/metadata": ("run_metadata.json", "运行元数据：模式/科学问题/产物清单"),
}


def _serve(filename: str) -> Any:
    """统一的数据加载 + 404 友好错误处理。"""
    try:
        return load(filename)
    except DataSourceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/")
def index() -> Dict[str, Any]:
    """探活：返回当前数据源、各产物可用性与全部端点列表。"""
    files = available_files()
    return {
        "service": "B-1 耀斑前兆因果链 API",
        "data_source": data_dir_label(),
        "is_real_mode": is_real(),
        "endpoints": [
            {"path": path, "file": fname, "description": desc,
             "available": files.get(fname, False)}
            for path, (fname, desc) in _ENDPOINTS.items()
        ],
        "docs": "/docs",
    }


@app.get("/literature")
def get_literature() -> Any:
    """文献因素（前兆候选因子，每个带真实 arXiv ID）。"""
    return _serve("literature_facts.json")


@app.get("/evidence")
def get_evidence() -> Any:
    """数据证据（JW-FD 统计验证：相关性/p 值/样本量/反例）。"""
    return _serve("evidence_report.json")


@app.get("/debate")
def get_debate() -> Any:
    """三角色辩论全过程记录（transcript 数组）。"""
    return _serve("transcript.json")


@app.get("/causal_graph")
def get_causal_graph() -> Any:
    """因果图谱（nodes/edges/root_cause/debate_rounds）。"""
    return _serve("causal_graph.json")


@app.get("/hypothesis")
def get_hypothesis() -> Any:
    """11 字段《科学假设与研究计划》。"""
    return _serve("hypothesis_report.json")


@app.get("/metadata")
def get_metadata() -> Any:
    """运行元数据（mode/query/artifacts）。"""
    return _serve("run_metadata.json")
