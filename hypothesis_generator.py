"""
假设生成 Agent · 钟菁泽 / 吴宜俊

输入：因果图谱 + 文献因素 + 数据证据 + （可选）辩论记录
输出：HypothesisReport（11 个标准字段，评审硬性要求）

mock 模式：基于因果图谱与证据填充模板（结构完整、内容真实可查）。
LLM 模式：调用 Qwen 生成更自然的学术综述（需 client）。

11 字段（缺一不可）：
  problem_statement / rationale / technical_details / datasets /
  source / target / paper_title / abstract / methods /
  experiments / results / references
"""
from __future__ import annotations
from typing import List, Optional

from shared.contracts import (
    LiteratureFacts, EvidenceReport, CausalGraph, HypothesisReport,
)
from debate.model_client import is_mock


def generate_hypothesis(graph: CausalGraph, facts: LiteratureFacts,
                         evidence: EvidenceReport, client=None,
                         transcript: Optional[List[dict]] = None) -> HypothesisReport:
    if is_mock(client):
        return _template(graph, facts, evidence, transcript)
    return _run_llm(graph, facts, evidence, client, transcript)


def _template(graph: CausalGraph, facts: LiteratureFacts,
              evidence: EvidenceReport,
              transcript: Optional[List[dict]] = None) -> HypothesisReport:
    ev_by_id = {e.factor_id: e for e in evidence.evidences}
    supported = [f for f in facts.factors if ev_by_id.get(f.id) and ev_by_id[f.id].supported]
    refs = [f.source for f in facts.factors]

    # 构建结果摘要（含因果链描述）
    result_parts = []
    for f in supported:
        ev = ev_by_id[f.id]
        result_parts.append(
            f"{f.name} 被识别为显著前兆（corr={ev.correlation}, p={ev.p_value}, "
            f"n={ev.sample_size}）"
        )
    unsupported = [f for f in facts.factors
                   if ev_by_id.get(f.id) and not ev_by_id[f.id].supported]
    for f in unsupported:
        ev = ev_by_id[f.id]
        result_parts.append(
            f"{f.name} 经质疑者挑战后被降级（corr={ev.correlation}, p={ev.p_value}，"
            f"反例{ev.counter_examples}个，可能为伴随现象）"
        )
    result_summary = "；".join(result_parts) or "未识别出足够支撑的前兆"

    # 从因果图谱提取因果链描述
    chain_desc = _describe_causal_chain(graph, facts)

    # 构建方法描述（含辩论机制）
    debate_info = ""
    if transcript:
        n_rounds = max((t.get("round", 1) for t in transcript), default=1)
        debate_info = (
            f"三角色（物理学家/质疑者/方法论审查）经过 {n_rounds} 轮结构化辩论，"
            f"通过假设-证伪-审查范式收敛出因果共识。"
        )

    # 构建验证计划描述
    next_steps = "建议后续使用 SDO/HMI 高分辨率磁场数据进行时序追踪验证"
    if graph.root_cause and "未识别" not in graph.root_cause:
        next_steps += f"，重点验证主前兆「{_extract_main_precursor(graph.root_cause)}」的时序稳定性"

    return HypothesisReport(
        problem_statement="哪些可观测前兆能稳定预测太阳耀斑爆发？这些前兆之间的因果链如何？",
        rationale=(
            "耀斑爆发深刻影响近地空间环境与卫星通信，但触发机制尚未完全厘清。"
            "传统方法依赖单一模型拟合，难以揭示多因素间的因果链条。"
            "本研究通过多智能体辩论系统，整合文献证据与观测统计，"
            "系统发掘耀斑触发的可验证因果假设。"
        ),
        technical_details=(
            f"{chain_desc}\n\n"
            f"主因结论：{graph.root_cause}"
        ),
        datasets=(
            f"{evidence.dataset}（太阳耀斑预报，"
            f"{max((e.sample_size for e in evidence.evidences), default=0) or '?'} 样本 × "
            f"{evidence.n_features or '未知'} 特征，多时间窗口耀斑等级标签）"
        ),
        source="arXiv 太阳物理文献抽取 + JW-FD 观测数据统计关联",
        target="输出可验证的耀斑触发前兆因果链图谱与科学假设报告",
        paper_title="A Multi-Agent Debate Framework for Discovering Solar Flare Precursor Causal Chains",
        abstract=(
            "本文提出一种基于多智能体辩论的科学假设生成系统，"
            "自动整合太阳物理文献与 JW-FD 观测数据集，"
            "通过物理学家-质疑者-方法论审查三角色的结构化辩论，"
            "发现耀斑触发的可验证前兆因果链。"
            f"{debate_info}"
        ),
        methods=(
            "系统由 5 个模块组成：\n"
            "1) 文献 Agent：从 arXiv 检索太阳耀斑相关论文，抽取前兆候选因素\n"
            "2) 数据证据 Agent：在 JW-FD 数据集上对每个因素做统计关联分析（相关性、p值、时序）\n"
            "3) 辩论调度器：三角色（物理学家/质疑者/方法论审查）进行多轮结构化辩论\n"
            "4) 因果图谱构建：从辩论共识中提取因果关系，构建链式因果图\n"
            "5) 假设生成器：综合所有证据输出 11 字段标准科学假设报告"
        ),
        experiments=(
            f"在 {evidence.dataset} 数据集上对 {len(facts.factors)} 个候选前兆因素"
            f"做相关性分析和 p 值检验（显著水平 α=0.05），"
            f"结果作为辩论证据输入。同时引入 {len(unsupported)} 个被挑战因素作为反例验证。"
        ),
        results=f"{result_summary}\n\n{debate_info}",
        references=refs,
    )


def _describe_causal_chain(graph: CausalGraph, facts: LiteratureFacts) -> str:
    """从因果图谱中提取可读的因果链描述。"""
    if not graph.edges:
        return "未检测到显著因果链。"

    # 按拓扑顺序描述因果链
    # 找到中间机制节点（ntype == "mechanism"）
    mechanism_nodes = {n.id: n for n in graph.nodes if n.ntype == "mechanism"}
    precursor_nodes = {n.id: n for n in graph.nodes if n.ntype == "precursor"}

    parts = []
    # 描述链式关系
    for edge in graph.edges:
        src_label = next((n.label for n in graph.nodes if n.id == edge.src), edge.src)
        dst_label = next((n.label for n in graph.nodes if n.id == edge.dst), edge.dst)
        if edge.relation == "enables":
            parts.append(f"{src_label} → {dst_label}（{edge.relation}, conf={edge.confidence}）")
        elif edge.relation == "triggers":
            parts.append(f"{src_label} → {dst_label}（直接触发, conf={edge.confidence}）")

    if parts:
        return "发现以下因果链：\n" + "\n".join(f"  • {p}" for p in parts)
    return "因果链结构待进一步分析。"


def _extract_main_precursor(root_cause: str) -> str:
    """从 root_cause 字符串中提取主前兆名称。"""
    if "主前兆为 " in root_cause:
        name = root_cause.split("主前兆为 ")[1].split("（")[0].split("(")[0]
        return name.strip()
    return root_cause


def _run_llm(graph, facts, evidence, client,
             transcript=None) -> HypothesisReport:  # pragma: no cover
    """LLM 模式：调用 Qwen 生成更自然的学术文本。

    当前回退到模板，后续吴宜俊接入后替换。
    """
    # TODO(吴宜俊): 用 Qwen 生成各字段，保证学术表达质量
    return _template(graph, facts, evidence, transcript)
