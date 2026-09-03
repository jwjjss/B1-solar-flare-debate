"""
B-1 项目 · 4 个 JSON 接口契约（全项目唯一耦合点）

任何人改这些结构，必须走「例会提 → 确认 → 改本文档 → 改代码」流程。
契约使用 dataclass 定义，to_dict / from_dict 做序列化；同时提供 JSON_SCHEMA
字符串供前端/文档参考。mock 数据见 shared/mock_data/。
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ───────────────────────── 1. 文献因素 (literature_facts) ─────────────────────────
@dataclass
class LiteratureFactor:
    id: str
    name: str
    description: str
    source: str            # 论文出处，必须真实可查（arXiv id / DOI）
    confidence: float      # 0~1，文献对该前兆的支持强度
    ftype: str = "precursor"   # precursor=前兆, mechanism=机制, condition=条件


@dataclass
class LiteratureFacts:
    query: str
    papers_searched: int
    factors: List[LiteratureFactor] = field(default_factory=list)


# ───────────────────────── 2. 证据报告 (evidence_report) ─────────────────────────
@dataclass
class EvidenceItem:
    factor_id: str
    supported: bool
    correlation: float     # 与耀斑爆发的统计相关性 -1~1
    p_value: float
    sample_size: int
    counter_examples: int
    note: str = ""


@dataclass
class EvidenceReport:
    dataset: str
    evidences: List[EvidenceItem] = field(default_factory=list)
    n_features: int = 0        # JW-FD 实际特征列数；0 表示未提供


# ───────────────────────── 3. 因果图谱 (causal_graph) ─────────────────────────
@dataclass
class GraphNode:
    id: str
    label: str
    ntype: str             # precursor / mechanism / event


@dataclass
class GraphEdge:
    src: str
    dst: str
    relation: str          # triggers / enables / correlates
    confidence: float
    evidence_ref: str      # 关联的 factor_id 或 evidence 来源


@dataclass
class CausalGraph:
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    root_cause: str = ""
    debate_rounds: int = 0


# ──────────────────── 4. 科学假设报告 (hypothesis_report) ────────────────────
# 11 个标准字段（评审硬性要求，缺字段扣分）
@dataclass
class HypothesisReport:
    problem_statement: str
    rationale: str
    technical_details: str
    datasets: str
    source: str
    target: str
    paper_title: str
    abstract: str
    methods: str
    experiments: str
    results: str
    references: List[str] = field(default_factory=list)


# ───────────────────────── 序列化工具 ─────────────────────────
def _wrap(obj):
    return {obj.__class__.__name__: asdict(obj)}


def to_json(obj) -> dict:
    return asdict(obj)


def literature_facts_from(d: dict) -> LiteratureFacts:
    return LiteratureFacts(
        query=d["query"],
        papers_searched=d["papers_searched"],
        factors=[LiteratureFactor(**f) for f in d["factors"]],
    )


def evidence_report_from(d: dict) -> EvidenceReport:
    return EvidenceReport(
        dataset=d["dataset"],
        evidences=[EvidenceItem(**e) for e in d["evidences"]],
        n_features=int(d.get("n_features", 0) or 0),
    )


def causal_graph_from(d: dict) -> CausalGraph:
    return CausalGraph(
        nodes=[GraphNode(**n) for n in d["nodes"]],
        edges=[GraphEdge(**e) for e in d["edges"]],
        root_cause=d.get("root_cause", ""),
        debate_rounds=d.get("debate_rounds", 0),
    )


def hypothesis_report_from(d: dict) -> HypothesisReport:
    return HypothesisReport(
        problem_statement=d["problem_statement"],
        rationale=d["rationale"],
        technical_details=d["technical_details"],
        datasets=d["datasets"],
        source=d["source"],
        target=d["target"],
        paper_title=d["paper_title"],
        abstract=d["abstract"],
        methods=d["methods"],
        experiments=d["experiments"],
        results=d["results"],
        references=d.get("references", []),
    )


# ───────────────────────── 文档用 JSON Schema（摘要） ─────────────────────────
JSON_SCHEMA = {
    "LiteratureFacts": {
        "query": "string", "papers_searched": "int",
        "factors[]": {"id": "str", "name": "str", "description": "str",
                       "source": "str(真实出处)", "confidence": "float 0~1", "ftype": "precursor|mechanism|condition"}
    },
    "EvidenceReport": {
        "dataset": "str",
        "evidences[]": {"factor_id": "str", "supported": "bool", "correlation": "float -1~1",
                         "p_value": "float", "sample_size": "int", "counter_examples": "int", "note": "str"}
    },
    "CausalGraph": {
        "nodes[]": {"id": "str", "label": "str", "ntype": "precursor|mechanism|event"},
        "edges[]": {"src": "str", "dst": "str", "relation": "triggers|enables|correlates",
                     "confidence": "float 0~1", "evidence_ref": "str"},
        "root_cause": "str", "debate_rounds": "int"
    },
    "HypothesisReport": {
        "problem_statement": "str", "rationale": "str", "technical_details": "str",
        "datasets": "str", "source": "str", "target": "str", "paper_title": "str",
        "abstract": "str", "methods": "str", "experiments": "str", "results": "str",
        "references[]": "str(真实文献)"
    }
}
