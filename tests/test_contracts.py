"""
B-1 项目契约验证与链路测试

运行：python -m pytest tests/ -v
或：  python tests/test_contracts.py（直接运行）
"""
from __future__ import annotations
import asyncio
import json, os, sys

# Windows 终端 GBK → UTF-8，防止 ✓✗ 等 Unicode 字符编码报错
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.contracts import (
    LiteratureFacts, LiteratureFactor,
    EvidenceReport, EvidenceItem,
    CausalGraph, GraphNode, GraphEdge,
    HypothesisReport,
    literature_facts_from, evidence_report_from,
    causal_graph_from, hypothesis_report_from,
    to_json,
)


MOCK_DIR = os.path.join(os.path.dirname(__file__), "..", "shared", "mock_data")


def _load_mock(name: str) -> dict:
    with open(os.path.join(MOCK_DIR, name), encoding="utf-8") as f:
        return json.load(f)


# ─────────────────── 契约 1: LiteratureFacts ───────────────────

def test_literature_facts_required_fields():
    """LiteratureFacts 必须包含 query, papers_searched, factors[]"""
    data = _load_mock("literature_facts.json")
    obj = literature_facts_from(data)
    assert isinstance(obj.query, str) and len(obj.query) > 0
    assert isinstance(obj.papers_searched, int) and obj.papers_searched > 0
    assert len(obj.factors) >= 1, "至少需要 1 个前兆因素"


def test_literature_factor_fields():
    """每个 LiteratureFactor 必须有 id, name, description, source, confidence, ftype"""
    data = _load_mock("literature_facts.json")
    obj = literature_facts_from(data)
    for f in obj.factors:
        assert f.id, f"factor 缺少 id"
        assert f.name, f"factor {f.id} 缺少 name"
        assert f.description, f"factor {f.id} 缺少 description"
        assert f.source, f"factor {f.id} 缺少 source（必须真实文献来源）"
        assert 0 <= f.confidence <= 1, f"factor {f.id} confidence 必须在 [0,1]"
        assert f.ftype in ("precursor", "mechanism", "condition"), \
            f"factor {f.id} ftype 必须为 precursor/mechanism/condition"


def test_literature_factor_source_format():
    """文献来源必须是真实的 arXiv ID 或 DOI"""
    data = _load_mock("literature_facts.json")
    obj = literature_facts_from(data)
    for f in obj.factors:
        assert f.source.startswith("arXiv:") or f.source.startswith("doi:"), \
            f"factor {f.id} 的 source '{f.source}' 格式不正确"


# ─────────────────── 契约 2: EvidenceReport ───────────────────

def test_evidence_report_required_fields():
    """EvidenceReport 必须包含 dataset 和 evidences[]"""
    data = _load_mock("evidence_report.json")
    obj = evidence_report_from(data)
    assert isinstance(obj.dataset, str) and len(obj.dataset) > 0
    assert len(obj.evidences) >= 1, "至少需要 1 条证据"


def test_evidence_item_fields():
    """每个 EvidenceItem 必须有完整的统计信息"""
    data = _load_mock("evidence_report.json")
    obj = evidence_report_from(data)
    for e in obj.evidences:
        assert e.factor_id, "evidence 缺少 factor_id"
        assert isinstance(e.supported, bool), f"evidence {e.factor_id} supported 必须是 bool"
        assert -1 <= e.correlation <= 1, f"evidence {e.factor_id} correlation 必须在 [-1,1]"
        assert 0 <= e.p_value <= 1, f"evidence {e.factor_id} p_value 必须在 [0,1]"
        assert e.sample_size > 0, f"evidence {e.factor_id} sample_size 必须 > 0"
        assert e.counter_examples >= 0, f"evidence {e.factor_id} counter_examples 必须 >= 0"


def test_evidence_factor_ids_match():
    """EvidenceReport 中的 factor_id 必须能在 LiteratureFacts 中找到"""
    facts_data = _load_mock("literature_facts.json")
    evidence_data = _load_mock("evidence_report.json")
    facts = literature_facts_from(facts_data)
    evidence = evidence_report_from(evidence_data)
    fact_ids = {f.id for f in facts.factors}
    for e in evidence.evidences:
        assert e.factor_id in fact_ids, \
            f"evidence 的 factor_id '{e.factor_id}' 不在 literature_facts 中"


# ─────────────────── 契约 3: CausalGraph ───────────────────

def test_causal_graph_structure():
    """CausalGraph 必须有节点、边和根因"""
    data = _load_mock("causal_graph.json")
    obj = causal_graph_from(data)
    assert len(obj.nodes) >= 2, "因果图谱至少需要 2 个节点"
    assert len(obj.edges) >= 1, "因果图谱至少需要 1 条边"
    assert obj.root_cause, "因果图谱缺少 root_cause"


def test_causal_graph_node_types():
    """节点类型必须是 precursor / mechanism / event"""
    data = _load_mock("causal_graph.json")
    obj = causal_graph_from(data)
    valid_types = {"precursor", "mechanism", "event"}
    for n in obj.nodes:
        assert n.ntype in valid_types, \
            f"节点 {n.id} 类型 '{n.ntype}' 不合法，应为 {valid_types}"


def test_causal_graph_edge_fields():
    """边的字段必须符合规范"""
    data = _load_mock("causal_graph.json")
    obj = causal_graph_from(data)
    node_ids = {n.id for n in obj.nodes}
    valid_relations = {"triggers", "enables", "correlates"}
    for e in obj.edges:
        assert e.src in node_ids, f"边的 src '{e.src}' 不在节点列表中"
        assert e.dst in node_ids, f"边的 dst '{e.dst}' 不在节点列表中"
        assert e.relation in valid_relations, \
            f"边的 relation '{e.relation}' 不合法，应为 {valid_relations}"
        assert 0 <= e.confidence <= 1, f"边的 confidence {e.confidence} 不在 [0,1]"


def test_causal_graph_has_event_node():
    """因果图谱必须包含至少一个 event 类型节点（耀斑爆发）"""
    data = _load_mock("causal_graph.json")
    obj = causal_graph_from(data)
    event_nodes = [n for n in obj.nodes if n.ntype == "event"]
    assert len(event_nodes) >= 1, "因果图谱缺少 event 类型节点"


# ─────────────────── 契约 4: HypothesisReport ───────────────────

def test_hypothesis_report_11_fields():
    """HypothesisReport 必须包含全部 11 个标准字段"""
    data = _load_mock("hypothesis_report.json")
    obj = hypothesis_report_from(data)
    required = [
        "problem_statement", "rationale", "technical_details",
        "datasets", "source", "target", "paper_title",
        "abstract", "methods", "experiments", "results",
    ]
    for field in required:
        val = getattr(obj, field, None)
        assert val and len(str(val).strip()) > 0, f"假设报告缺少字段: {field}"


def test_hypothesis_report_references_not_empty():
    """参考文献列表不能为空（严禁虚构）"""
    data = _load_mock("hypothesis_report.json")
    obj = hypothesis_report_from(data)
    assert len(obj.references) >= 1, "references 不能为空"
    for ref in obj.references:
        assert ref.startswith("arXiv:") or ref.startswith("doi:"), \
            f"参考文献格式可疑: {ref}"


# ─────────────────── 端到端链路测试 ───────────────────

def test_data_evidence_agent_real_mode():
    """DataEvidenceAgent 真实模式：JW-FD 加载 + 统计验证 + 契约2合规"""
    from data_check.agent import DataEvidenceAgent, _find_dataset

    ds_path = _find_dataset()
    if ds_path is None:
        print("  (JW-FD 不可用，跳过真实模式测试)")
        return

    facts_data = _load_mock("literature_facts.json")
    facts = literature_facts_from(facts_data)

    agent = DataEvidenceAgent(dataset_path=ds_path)
    report = agent.run(facts, use_real=True)

    assert report.dataset == "JW-FD"
    assert len(report.evidences) == len(facts.factors)
    for e in report.evidences:
        assert e.factor_id, "factor_id 不能为空"
        assert isinstance(e.supported, bool), "supported 必须是 bool"
        assert -1 <= e.correlation <= 1, f"correlation {e.correlation} 越界"
        assert 0 <= e.p_value <= 1, f"p_value {e.p_value} 越界"
        assert e.sample_size > 0, "sample_size 必须 > 0"
        assert e.counter_examples >= 0, "counter_examples 必须 >= 0"
        assert e.note, "note 不能为空"


def test_end_to_end_mock():
    """完整 mock 链路：文献→证据→辩论→因果图→假设报告"""
    from literature.agent import LiteratureAgent
    from data_check.agent import DataEvidenceAgent
    from debate.scheduler import run_debate
    from hypothesis_generator import generate_hypothesis

    facts = asyncio.run(LiteratureAgent().run())
    evidence = DataEvidenceAgent().run(facts)
    transcript, graph = run_debate(facts, evidence, client=None)
    report = generate_hypothesis(graph, facts, evidence, client=None, transcript=transcript)

    # 辩论记录非空
    assert len(transcript) > 0
    # 因果图谱有内容
    assert len(graph.nodes) >= 2
    assert len(graph.edges) >= 1
    # 报告完整
    assert report.paper_title
    assert len(report.references) >= 1


# ─────────────────── 运行入口 ───────────────────

if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"  ✓ {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {test_fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n结果: {passed} 通过, {failed} 失败")
    sys.exit(1 if failed else 0)
