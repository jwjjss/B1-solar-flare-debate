"""
因果图谱构建 Agent · 钟菁泽负责

从辩论记录中提取因果关系，构建因果图谱。

策略（双模式）：
  1. 如果辩论记录中包含结构化结论块（---CAUSAL_CLAIMS--- 等），
     则优先从辩论共识中构建因果链（体现辩论的价值）。
  2. 如果没有结构化输出（如早期 mock），回退到纯证据驱动的规则构建。

核心改进（相比纯规则版本）：
  - 引入中间机制节点（如「自由磁能积累」），形成链式因果而非星形
  - 被质疑者成功挑战的因素降级为弱节点（虚线边）
  - 方法论审查判定 insufficient 的因素被排除
"""
from __future__ import annotations
import re
from typing import List, Dict, Optional, Tuple

from shared.contracts import (
    LiteratureFacts, EvidenceReport, CausalGraph,
    GraphNode, GraphEdge,
)


# ───────────────────────── 已知物理因果链模板 ─────────────────────────
# 当文献因素匹配已知物理机制时，插入中间节点以展示链式因果
# 这些是太阳物理学中的经典因果路径，可由物理学家角色在辩论中引用
KNOWN_CHAINS = {
    "光球剪切运动增强": {
        "intermediate": "自由磁能积累",
        "intermediate_desc": "剪切运动持续向日冕注入自由磁能",
        "next_factor": "磁通量绳形成",
        "relation": "enables",
    },
    "磁通量绳形成": {
        "intermediate": None,
        "relation": "triggers",
        "direct_target": "耀斑爆发",
    },
}


def _edge_confidence(ev, factor, kind: str) -> float:
    """把数据证据映射为科学上可解释的边置信度。

    旧公式 ``1 - p_value`` 会把 p=0.21（未通过显著性）映成 0.79，
    与「证据不足」的语义完全相反，因此这里改为基于相关系数与
    supported 标志的门控映射：

    - ``trigger``（主前兆）：|r| 若 ev.supported 且 p<=0.05，否则退回
      文献置信度 * 0.5 作为保守下界；
    - ``enable``（条件性/调制）：0.6 * |r|；
    - ``correlate``（弱节点/被驳斥）：固定 0.1，只保留可视线索。
    """
    if kind == "correlate":
        return 0.1
    if ev is None:
        # 无数据证据时，只能靠文献置信度打折
        base = float(getattr(factor, "confidence", 0.0))
        return round(max(0.0, min(1.0, base * (1.0 if kind == "trigger" else 0.6))), 3)
    r = abs(float(ev.correlation))
    if kind == "trigger":
        if ev.supported and ev.p_value <= 0.05:
            return round(max(0.0, min(1.0, r)), 3)
        # 数据未通过但被误分到 trigger 时的兜底（不应发生）
        return round(max(0.0, min(1.0, r * 0.5)), 3)
    if kind == "enable":
        return round(max(0.0, min(1.0, 0.6 * r)), 3)
    return round(max(0.0, min(1.0, r)), 3)


def build_causal_graph(facts: LiteratureFacts, evidence: EvidenceReport,
                        transcript: List[dict], rounds: int) -> CausalGraph:
    """从辩论结果构建因果图谱。

    优先使用辩论共识（结构化输出块），回退到证据驱动规则。
    """
    # 1. 尝试从辩论记录中提取结构化结论
    debate_claims = _parse_causal_claims(transcript)
    debate_challenges = _parse_challenges(transcript)
    debate_verdict = _parse_verdict(transcript)

    # 2. 如果有辩论结论，用辩论结果驱动图谱构建
    if debate_claims or debate_verdict:
        return _build_from_debate(
            facts, evidence, transcript, rounds,
            debate_claims, debate_challenges, debate_verdict,
        )
    # 3. 否则回退到纯证据规则
    return _build_from_evidence(facts, evidence, rounds)


# ───────────────────────── 从辩论共识构建 ─────────────────────────

def _build_from_debate(
    facts: LiteratureFacts, evidence: EvidenceReport,
    transcript: List[dict], rounds: int,
    claims: Dict[str, dict], challenges: Dict[str, dict],
    verdict: Optional[dict],
) -> CausalGraph:
    """基于辩论共识构建因果图谱，包含中间节点和链式因果。"""
    ev_by_id = {e.factor_id: e for e in evidence.evidences}
    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []
    node_ids = set()

    # 添加耀斑爆发事件节点
    nodes.append(GraphNode(id="E1", label="耀斑爆发", ntype="event"))
    node_ids.add("E1")

    # 根据辩论结论分类因素
    validated = set()
    if verdict and "validated" in verdict:
        validated = set(verdict["validated"])

    # 收集被接受的因素（debate 共识 + 证据支持）
    accepted_factors = []
    dependent_factors = []   # LLM 可能判为 "dependent"（条件性因素）
    challenged_factors = []

    # 检查辩论是否达成共识（最后一轮 skeptic 是否输出 NO_NEW_CHALLENGES）
    consensus_reached = any(
        "NO_NEW_CHALLENGES" in t.get("content", "")
        for t in transcript if t.get("speaker") == "skeptic"
    )

    for f in facts.factors:
        ev = ev_by_id.get(f.id)
        claim = claims.get(f.id, {})
        challenge = challenges.get(f.id, {})
        verdict_str = claim.get("verdict", "").lower()
        challenge_status = challenge.get("status", "").lower()

        # ── 数据证据门控（不可绕过）───────────────────────────────
        # 无论辩论用什么措辞，只有 ev.supported=True 且 p<=0.05 的因素
        # 才可能被升为 accepted。这直接杜绝了「LLM 用 STATUS: accepted
        # 表示接受物理学家的驳斥」被误读为「该因素是有效前兆」的语义反转。
        data_ok = bool(ev and ev.supported and ev.p_value <= 0.05)

        # ── 辩论显式否决 ────────────────────────────────────────
        debate_rejected = (
            verdict_str in ("refuted", "rejected", "unsupported", "not a precursor")
            or challenge_status in ("refuted", "rejected")
        )

        # ── 辩论无条件支持（仅这两种情况才升 accepted）──────────
        # 1) 方法论审查在 VALIDATED_CHAINS 里点名
        # 2) 物理学家给出无保留的 "supported" 判定
        debate_unconditional = (
            f.id in validated
            or verdict_str == "supported"
        )

        # ── 辩论有条件支持（路由到 dependent，而非 accepted）────
        # "conditionally supported" / "conditional" / "questionable" /
        # "dependent" 都表示「辩论认可其作用，但需补充验证才能当独立触发器」
        debate_conditional = verdict_str in (
            "conditionally supported", "conditional",
            "questionable", "dependent",
        )

        # 综合判断（数据门控优先，辩论用于分级）
        if data_ok and not debate_rejected:
            if debate_unconditional:
                is_accepted = True
                is_dependent = False
            elif debate_conditional:
                # 数据通过但辩论只给有条件支持 → 条件性因素
                is_accepted = False
                is_dependent = True
            elif challenge_status in ("accepted", "partially accepted", ""):
                # 没有 CAUSAL_CLAIMS 但质疑者接受 → 视为 accepted
                is_accepted = True
                is_dependent = False
            else:
                # 数据通过但辩论持续 challenged → 条件性因素
                is_accepted = False
                is_dependent = True
        elif data_ok and debate_rejected:
            # 罕见：数据显著但辩论给出强否决（例如判为伴随现象）→ 条件性
            is_accepted = False
            is_dependent = True
        else:
            # 数据未通过：无论辩论如何措辞，都不进入 accepted / dependent
            is_accepted = False
            is_dependent = False

        if is_accepted:
            accepted_factors.append(f)
        elif is_dependent:
            dependent_factors.append(f)
        else:
            # 数据未通过或辩论否决 → 弱节点（correlates）
            challenged_factors.append(f)

    # 构建因果链（含中间节点）
    # 排序键：数据相关性 |r| 为主，文献置信度为辅助 tie-breaker。
    # 之前仅按 f.confidence 排序，导致「文献置信度高但数据未通过」的因素
    # 被误选为主因；现改为证据驱动。
    def _composite_score(f):
        ev = ev_by_id.get(f.id)
        r = abs(ev.correlation) if ev else 0.0
        return (r, f.confidence)

    sorted_factors = sorted(accepted_factors, key=_composite_score, reverse=True)

    # 检查是否可以形成链式因果
    chain_factors = _detect_chain(sorted_factors)

    if chain_factors:
        # 链式因果：A → 中间节点 → B → 耀斑
        prev_id = None
        for i, f in enumerate(chain_factors):
            ev = ev_by_id.get(f.id)
            if f.id not in node_ids:
                nodes.append(GraphNode(id=f.id, label=f.name, ntype=getattr(f, "ftype", "") or "precursor"))
                node_ids.add(f.id)

            chain_info = KNOWN_CHAINS.get(f.name, {})
            intermediate = chain_info.get("intermediate")

            if intermediate and i < len(chain_factors) - 1:
                # 添加中间机制节点
                mid_id = f"M{i}"
                if mid_id not in node_ids:
                    nodes.append(GraphNode(id=mid_id, label=intermediate, ntype="mechanism"))
                    node_ids.add(mid_id)

                # 前驱 → 中间节点
                src_id = prev_id if prev_id else f.id
                edges.append(GraphEdge(
                    src=f.id, dst=mid_id, relation="enables",
                    confidence=_edge_confidence(ev, f, "enable"),
                    evidence_ref=f.source,
                ))
                # 中间节点 → 下一个因素
                next_f = chain_factors[i + 1]
                next_ev = ev_by_id.get(next_f.id)
                edges.append(GraphEdge(
                    src=mid_id, dst=next_f.id, relation="enables",
                    confidence=_edge_confidence(next_ev, next_f, "enable"),
                    evidence_ref=f.source,
                ))
                prev_id = mid_id
            elif i == len(chain_factors) - 1:
                # 最后一个因素 → 耀斑爆发
                edges.append(GraphEdge(
                    src=f.id, dst="E1", relation="triggers",
                    confidence=_edge_confidence(ev, f, "trigger"),
                    evidence_ref=f.source,
                ))
                # Note: prev_id→f.id edge was already created when the
                # intermediate node was processed in the previous iteration.
    else:
        # 星形因果：各因素直接指向耀斑
        for f in sorted_factors:
            ev = ev_by_id.get(f.id)
            if f.id not in node_ids:
                nodes.append(GraphNode(id=f.id, label=f.name, ntype=getattr(f, "ftype", "") or "precursor"))
                node_ids.add(f.id)
            edges.append(GraphEdge(
                src=f.id, dst="E1", relation="triggers",
                confidence=_edge_confidence(ev, f, "trigger"),
                evidence_ref=f.source,
            ))

    # 条件性因素（dependent）作为辅助节点添加（enables 边，中等置信度）
    for f in dependent_factors:
        ev = ev_by_id.get(f.id)
        if f.id not in node_ids:
            nodes.append(GraphNode(id=f.id, label=f.name, ntype=getattr(f, "ftype", "") or "precursor"))
            node_ids.add(f.id)
        edges.append(GraphEdge(
            src=f.id, dst="E1", relation="enables",
            confidence=_edge_confidence(ev, f, "enable"),
            evidence_ref=f.source,
        ))

    # 被挑战的因素作为弱节点添加（虚线，低置信度）
    for f in challenged_factors:
        if f.id not in node_ids:
            nodes.append(GraphNode(id=f.id, label=f.name, ntype=getattr(f, "ftype", "") or "precursor"))
            node_ids.add(f.id)
        edges.append(GraphEdge(
            src=f.id, dst="E1", relation="correlates",
            confidence=_edge_confidence(None, f, "correlate"),
            evidence_ref=f"f.challenged:{f.source}",
        ))

    # 确定主因（基于数据相关性，而非文献置信度）
    if sorted_factors:
        main = sorted_factors[0]
        main_ev = ev_by_id.get(main.id)
        if main_ev is not None:
            root = (
                f"主前兆为 {main.name}"
                f"（辩论共识 + 数据支撑最强：corr={main_ev.correlation}, "
                f"p={main_ev.p_value}, n={main_ev.sample_size}）"
            )
        else:
            root = f"主前兆为 {main.name}（辩论共识，缺少直接数据证据）"
    else:
        # 没有 accepted 因素时，从数据证据里挑最强的作为「候选待验证」
        best_ev = None
        best_f = None
        for f in facts.factors:
            ev = ev_by_id.get(f.id)
            if ev and ev.supported and ev.p_value <= 0.05:
                if best_ev is None or abs(ev.correlation) > abs(best_ev.correlation):
                    best_ev, best_f = ev, f
        if best_f is not None:
            root = (
                f"辩论未达成充分共识；数据层面最强候选为 {best_f.name}"
                f"（corr={best_ev.correlation}, p={best_ev.p_value}），"
                f"需按 NEXT_STEPS 补充对照实验后再定主因"
            )
        else:
            root = "辩论未达成充分共识，且无因素通过数据显著性门控，需进一步研究"

    return CausalGraph(
        nodes=nodes, edges=edges,
        root_cause=root,
        debate_rounds=rounds,
    )


def _detect_chain(factors: list) -> Optional[list]:
    """检测因素之间是否存在已知物理因果链。

    如果检测到链式关系，返回按因果顺序排列的因素列表；否则返回 None。
    """
    names = [f.name for f in factors]
    # 检查已知的链式关系
    for chain_start, info in KNOWN_CHAINS.items():
        next_name = info.get("next_factor")
        if next_name and chain_start in names and next_name in names:
            # 找到了链式关系，按因果顺序排列
            start_factor = next(f for f in factors if f.name == chain_start)
            next_factor = next(f for f in factors if f.name == next_name)
            # 因果顺序：剪切运动 → 磁通量绳 → ...
            return [start_factor, next_factor] + [
                f for f in factors if f.name not in (chain_start, next_name)
            ]
    return None


# ───────────────────────── 回退：纯证据驱动 ─────────────────────────

def _build_from_evidence(facts: LiteratureFacts, evidence: EvidenceReport,
                          rounds: int) -> CausalGraph:
    """回退方案：仅基于证据数据构建因果图谱（不含辩论信息）。"""
    ev_by_id = {e.factor_id: e for e in evidence.evidences}
    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []

    nodes.append(GraphNode(id="E1", label="耀斑爆发", ntype="event"))

    best = None  # (factor, |correlation|)
    for f in facts.factors:
        ev = ev_by_id.get(f.id)
        if not ev or not ev.supported or ev.p_value > 0.05 or ev.sample_size < 50:
            continue
        nodes.append(GraphNode(id=f.id, label=f.name,
                                ntype=getattr(f, "ftype", "") or "precursor"))
        edges.append(GraphEdge(
            src=f.id, dst="E1", relation="triggers",
            confidence=_edge_confidence(ev, f, "trigger"),
            evidence_ref=f.id,
        ))
        score = abs(ev.correlation)
        if best is None or score > best[1]:
            best = (f, score)

    if best:
        best_ev = ev_by_id[best[0].id]
        root = (
            f"主前兆为 {best[0].name}"
            f"（证据支撑最强：corr={best_ev.correlation}, p={best_ev.p_value}）"
        )
    else:
        root = "未识别出足够支撑的前兆"
    return CausalGraph(nodes=nodes, edges=edges, root_cause=root, debate_rounds=rounds)


# ───────────────────────── 辩论记录解析 ─────────────────────────

def _parse_causal_claims(transcript: List[dict]) -> Dict[str, dict]:
    """从辩论记录中解析物理学家的 ---CAUSAL_CLAIMS--- 块。"""
    claims = {}
    for entry in transcript:
        if entry.get("speaker") != "physicist":
            continue
        block = _extract_block(entry["content"], "CAUSAL_CLAIMS")
        if not block:
            continue
        for line in block.splitlines():
            m = re.match(
                r"-\s*FACTOR:\s*(\S+)\s*\|\s*VERDICT:\s*([^|]+?)\s*\|\s*MECHANISM:\s*(.+?)\s*\|\s*EVIDENCE:\s*(.+)",
                line.strip(),
            )
            if m:
                claims[m.group(1)] = {
                    "verdict": m.group(2).strip(),
                    "mechanism": m.group(3).strip(),
                    "evidence": m.group(4).strip(),
                }
    return claims


def _parse_challenges(transcript: List[dict]) -> Dict[str, dict]:
    """从辩论记录中解析质疑者的 ---CHALLENGES--- 块。"""
    challenges = {}
    for entry in transcript:
        if entry.get("speaker") != "skeptic":
            continue
        block = _extract_block(entry["content"], "CHALLENGES")
        if not block:
            continue
        if "NO_NEW_CHALLENGES" in block:
            continue
        for line in block.splitlines():
            m = re.match(
                r"-\s*FACTOR:\s*(\S+)\s*\|\s*STATUS:\s*(\S+)\s*\|\s*REASON:\s*(.+?)(?:\s*\|\s*NEED:\s*(.+))?$",
                line.strip(),
            )
            if m:
                challenges[m.group(1)] = {
                    "status": m.group(2),
                    "reason": m.group(3).strip(),
                    "need": (m.group(4) or "").strip(),
                }
    return challenges


def _parse_verdict(transcript: List[dict]) -> Optional[dict]:
    """从辩论记录中解析方法论审查的 ---VERDICT--- 块。"""
    for entry in reversed(transcript):
        if entry.get("speaker") != "methodologist":
            continue
        block = _extract_block(entry["content"], "VERDICT")
        if not block:
            continue
        result = {}
        for line in block.splitlines():
            if line.startswith("VALIDATED_CHAINS:"):
                chains = line.split(":", 1)[1].strip()
                # 提取因素 ID（支持 "F1→(...)", "F1, F2", "F1→chain" 等格式）
                ids = re.findall(r"(F\d+)", chains)
                result["validated"] = list(dict.fromkeys(ids))  # 去重保序
            elif line.startswith("WEAK_POINTS:"):
                result["weak"] = [
                    s.strip() for s in line.split(":", 1)[1].split(",") if s.strip() and s.strip() != "无"
                ]
            elif line.startswith("NEXT_STEPS:"):
                result["next_steps"] = line.split(":", 1)[1].strip()
        if result:
            return result
    return None


def _extract_block(text: str, tag: str) -> Optional[str]:
    """从文本中提取 ---TAG--- ... ---END_TAG--- 之间的内容。

    兼容别名：LLM 有时会把 ``---END_CAUSAL_CLAIMS---`` 简写成
    ``---END_CLAIMS---``（roles.py 的模板即如此），这里同时接受两种收尾，
    避免因标签不匹配而丢失整段结构化结论。
    """
    aliases = [tag]
    if tag == "CAUSAL_CLAIMS":
        aliases.append("CLAIMS")
    for alias in aliases:
        # 起始标签固定用 tag 本身；结束标签允许 tag 或 alias。
        pattern = rf"---{tag}---\s*\n(.*?)\n\s*---END_{alias}---"
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return m.group(1)
    return None
