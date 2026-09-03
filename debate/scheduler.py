"""
辩论调度器（系统大脑）· 钟菁泽负责

对外唯一入口：run_debate(facts, evidence, client=None, rounds=2)
  - client=None  → mock 模式（无需 Key/依赖，本地直接跑通，用于验证链路）
  - client=Qwen  → 真实多智能体辩论（AutoGen GroupChat）

输出：(transcript, causal_graph)
  - transcript: List[{"speaker","role","content"}]  辩论记录，也是系统的「实验过程」
  - causal_graph: 由辩论结果 + 证据驱动构建的因果图谱（见 causal_graph_builder）

终止条件（二选一，先到先停）：
  1. 达到 max_rounds 上限
  2. 质疑者连续 2 轮输出 NO_NEW_CHALLENGES（说明辩论已达成共识）
"""
from __future__ import annotations
import re
from typing import List, Tuple

from shared.contracts import LiteratureFacts, EvidenceReport, CausalGraph
from debate.roles import ROLE_ORDER, ROLE_SYSTEM, ROLE_TITLE
from debate.model_client import is_mock
from causal_graph_builder import build_causal_graph


def run_debate(facts: LiteratureFacts, evidence: EvidenceReport,
               client=None, config: dict | None = None,
               rounds: int | None = None,
               max_rounds: int | None = None) -> Tuple[List[dict], CausalGraph]:
    cfg = config or {}
    d = cfg.get("debate", {})
    if rounds is None:
        rounds = d.get("rounds", 2)
    if max_rounds is None:
        max_rounds = d.get("max_rounds", 8)

    if is_mock(client):
        print("[debate] mock 模式：用模板逻辑模拟三角色辩论（无 API 调用）")
        transcript = _simulate(facts, evidence, rounds, max_rounds)
    else:
        print("[debate] LLM 模式：AutoGen GroupChat 多角色辩论")
        transcript = _run_llm(facts, evidence, client, rounds, max_rounds)

    # 辩论轮次 = transcript 中最大的 round 编号，而非消息条数。
    # 之前传 len(transcript) 会让 causal_graph.debate_rounds 出现 8/9 之类的
    # 消息计数，与报告里 "3 轮辩论" 的表述矛盾。
    actual_rounds = max((t.get("round", 0) for t in transcript), default=0)
    graph = build_causal_graph(facts, evidence, transcript, actual_rounds)
    return transcript, graph


# ───────────────────────── mock 模拟（验证链路用） ─────────────────────────

def _simulate(facts: LiteratureFacts, evidence: EvidenceReport,
              target_rounds: int, max_rounds: int) -> List[dict]:
    """模拟多轮辩论。

    特性：
      - 第 1 轮：各角色独立陈述初始观点
      - 第 2+ 轮：回应前轮其他角色的论点（物理学家回应质疑，质疑者深挖）
      - 质疑者第 2 轮起无新质疑 → 提前终止（达成共识）
    """
    ev_by_id = {e.factor_id: e for e in evidence.evidences}
    transcript: List[dict] = []

    for r in range(max_rounds):
        print(f"  [debate] 第 {r + 1} 轮辩论...")
        early_stop = False
        for key in ROLE_ORDER:
            content = _role_speak(key, facts, evidence, ev_by_id, r, transcript)
            transcript.append({
                "round": r + 1,
                "speaker": key,
                "role": ROLE_TITLE[key],
                "content": content,
            })
            # 质疑者发言后立即检查：若共识已达成，让方法论审查完成本轮后终止
            if key == "skeptic" and "NO_NEW_CHALLENGES" in content and r >= 1:
                early_stop = True

        if early_stop:
            print(f"  [debate] 质疑者无新质疑，辩论达成共识，终止于第 {r + 1} 轮。")
            break

        if r + 1 >= target_rounds:
            break

    return transcript


def _role_speak(key: str, facts: LiteratureFacts, evidence: EvidenceReport,
                ev_by_id: dict, round_idx: int,
                transcript: List[dict]) -> str:
    """生成单个角色在某一轮的发言内容。"""

    if key == "physicist":
        return _physicist_speak(facts, evidence, ev_by_id, round_idx, transcript)
    elif key == "skeptic":
        return _skeptic_speak(facts, evidence, ev_by_id, round_idx, transcript)
    else:
        return _methodologist_speak(facts, evidence, ev_by_id, round_idx, transcript)


# ───────────────────────── 物理学家 ─────────────────────────

def _physicist_speak(facts, evidence, ev_by_id, round_idx, transcript) -> str:
    if round_idx == 0:
        lines = [
            "【物理学家·第1轮】从磁流体力学（MHD）角度分析各前兆因素的物理机制：\n"
        ]
        for f in facts.factors:
            ev = ev_by_id.get(f.id)
            if ev and ev.supported and ev.p_value <= 0.05:
                lines.append(
                    f"- {f.id} {f.name}：从磁重联理论看，{f.description}。"
                    f"观测数据显示 corr={ev.correlation}, p={ev.p_value}，"
                    f"机制上成立。"
                )
            elif ev and ev.supported:
                lines.append(
                    f"- {f.id} {f.name}：物理机制有一定合理性，"
                    f"但统计证据偏弱（corr={ev.correlation}, p={ev.p_value}），"
                    f"标记为「机制上部分成立」，建议补充时序观测数据。"
                )
            else:
                lines.append(
                    f"- {f.id} {f.name}：物理机制存在跳跃。{f.description}"
                    f"虽然理论上可能影响活动区稳定性，"
                    f"但缺乏直接因果证据（corr={ev.correlation}, p={ev.p_value}），暂标记为「存疑」。"
                )
        # 补充因果链推理
        supported_ids = [
            f.id for f in facts.factors
            if ev_by_id.get(f.id) and ev_by_id[f.id].supported and ev_by_id[f.id].p_value <= 0.05
        ]
        if len(supported_ids) >= 2:
            lines.append(
                "\n值得注意的是，多个前兆因素之间可能存在递进关系："
                f"{supported_ids} 可能构成一条因果链，"
                "即前一个因素为后一个因素创造条件，最终触发耀斑。"
            )
    else:
        # 后续轮次：回应质疑者的挑战
        lines = [f"【物理学家·第{round_idx + 1}轮】回应质疑者的挑战：\n"]
        # 找到上一轮质疑者的发言
        prev_skeptic = None
        for t in reversed(transcript):
            if t["speaker"] == "skeptic" and t.get("round") == round_idx:
                prev_skeptic = t
                break
        if prev_skeptic and "NO_NEW_CHALLENGES" not in prev_skeptic["content"]:
            for f in facts.factors:
                ev = ev_by_id.get(f.id)
                if ev and (not ev.supported or ev.p_value > 0.05 or ev.counter_examples >= 5):
                    lines.append(
                        f"- 关于 {f.id} {f.name}：质疑者指出的反例问题值得重视。"
                        f"从物理机制看，{f.description}，但需承认在当前样本下"
                        f"（{ev.counter_examples}个反例）该因素确实不够稳健。"
                        f"我同意将其从主因果链降级为「待验证因素」。"
                    )
                elif ev and ev.supported:
                    lines.append(
                        f"- 关于 {f.id} {f.name}：质疑者未提出有效反驳，"
                        f"我维持「机制上成立」的判断。"
                    )
        else:
            lines.append("- 质疑者本轮未提出新挑战，我维持此前所有判断。")

    # 输出结构化结论块
    lines.append("\n---CAUSAL_CLAIMS---")
    for f in facts.factors:
        ev = ev_by_id.get(f.id)
        if ev and ev.supported and ev.p_value <= 0.05:
            verdict = "supported"
            mechanism = f"物理机制清晰，{f.description}"
        elif ev and ev.supported:
            verdict = "questionable"
            mechanism = "物理机制部分成立但统计证据偏弱"
        else:
            verdict = "unsupported"
            mechanism = "因果跳跃过大，缺乏充分证据"
        lines.append(
            f"- FACTOR: {f.id} | VERDICT: {verdict} "
            f"| MECHANISM: {mechanism} | EVIDENCE: {f.source}"
        )
    lines.append("---END_CLAIMS---")
    return "\n".join(lines)


# ───────────────────────── 质疑者 ─────────────────────────

def _skeptic_speak(facts, evidence, ev_by_id, round_idx, transcript) -> str:
    if round_idx == 0:
        lines = ["【质疑者·第1轮】对物理学家的因果论断提出挑战：\n"]
        challenged = False
        for f in facts.factors:
            ev = ev_by_id.get(f.id)
            if ev and (not ev.supported or ev.p_value > 0.05 or ev.counter_examples >= 5):
                lines.append(
                    f"- {f.id} {f.name}：p_value={ev.p_value}（{'> 0.05' if ev.p_value > 0.05 else '边缘'}），"
                    f"反例 {ev.counter_examples} 个（占比 "
                    f"{ev.counter_examples / ev.sample_size * 100:.0f}%），"
                    f"相关性仅 {ev.correlation}。该因素更可能是伴随现象而非因果前兆。"
                    f"因果链在此处存在跳跃。"
                )
                challenged = True
            elif ev and ev.counter_examples > 0:
                lines.append(
                    f"- {f.id} {f.name}：虽有统计显著性，但存在 {ev.counter_examples} 个反例"
                    f"（{ev.counter_examples / ev.sample_size * 100:.0f}%），"
                    f"需要解释这些反例的物理原因。"
                )
                challenged = True
        if not challenged:
            lines.append("- 当前证据整体可支撑主因果链，暂未找到强反例。")
    else:
        # 后续轮次：检查物理学家是否已接受质疑（降级了问题因素）
        lines = [f"【质疑者·第{round_idx + 1}轮】深化质疑：\n"]
        prev_physicist = None
        for t in reversed(transcript):
            if t["speaker"] == "physicist" and t.get("round") == round_idx:
                prev_physicist = t
                break

        # 检查物理学家是否已承认某些因素需降级
        conceded = prev_physicist and "降级" in (prev_physicist.get("content", "") if prev_physicist else "")

        if conceded and round_idx >= 2:
            # 物理学家已接受降级，质疑者不再有新论点 → 共识达成
            lines.append("- 物理学家已接受对有争议因素的降级处理，各方达成共识，无新质疑。")
            challenged = False
        else:
            remaining_issues = []
            for f in facts.factors:
                ev = ev_by_id.get(f.id)
                if ev and (not ev.supported or ev.p_value > 0.05):
                    remaining_issues.append(f)

            if remaining_issues:
                for f in remaining_issues:
                    ev = ev_by_id[f.id]
                    lines.append(
                        f"- {f.id} {f.name}：维持质疑。物理学家已同意将其降级，"
                        f"但我补充一点：该因素的 counter_examples 比例"
                        f"（{ev.counter_examples}/{ev.sample_size}）表明"
                        f"它不适合作为通用前兆指标。建议从因果链中移除。"
                    )
                challenged = True
            else:
                lines.append("- 所有剩余因果链已有充分数据支撑，无新质疑。")
                challenged = False

    # 输出结构化结论块
    lines.append("\n---CHALLENGES---")
    any_challenged = False
    for f in facts.factors:
        ev = ev_by_id.get(f.id)
        if ev and (not ev.supported or ev.p_value > 0.05 or ev.counter_examples >= 5):
            lines.append(
                f"- FACTOR: {f.id} | STATUS: challenged "
                f"| REASON: p_value={ev.p_value}, 反例{ev.counter_examples}个, "
                f"相关性仅{ev.correlation} "
                f"| NEED: 需更精细的时序分析来区分因果与伴随"
            )
            any_challenged = True
        elif ev and ev.counter_examples > 2:
            lines.append(
                f"- FACTOR: {f.id} | STATUS: noted "
                f"| REASON: 统计显著但存在{ev.counter_examples}个反例需解释 "
                f"| NEED: 反例的物理成因分析"
            )
            any_challenged = True
        elif ev and ev.supported and ev.p_value <= 0.05:
            lines.append(
                f"- FACTOR: {f.id} | STATUS: accepted "
                f"| REASON: 物理机制清晰，数据支撑充分"
            )
    if not any_challenged:
        lines.append("NO_NEW_CHALLENGES")
    lines.append("---END_CHALLENGES---")
    return "\n".join(lines)


# ───────────────────────── 方法论审查 ─────────────────────────

def _methodologist_speak(facts, evidence, ev_by_id, round_idx, transcript) -> str:
    lines = [f"【方法论审查·第{round_idx + 1}轮】证据质量审查：\n"]

    sufficient_factors = []
    insufficient_factors = []

    for f in facts.factors:
        ev = ev_by_id.get(f.id)
        if not ev:
            continue
        sample_ok = ev.sample_size >= 50
        pvalue_ok = ev.p_value <= 0.05
        counter_ratio = ev.counter_examples / max(ev.sample_size, 1)

        if sample_ok and pvalue_ok and counter_ratio < 0.1:
            quality = "sufficient"
            note = f"样本充足(n={ev.sample_size})，统计显著(p={ev.p_value})，反例比例低({counter_ratio:.0%})"
            sufficient_factors.append(f.id)
        elif pvalue_ok and sample_ok:
            quality = "borderline"
            note = f"统计显著但反例比例偏高({counter_ratio:.0%})，需谨慎解读"
            insufficient_factors.append(f.id)
        else:
            quality = "insufficient"
            reasons = []
            if not sample_ok:
                reasons.append(f"样本量过小(n={ev.sample_size})")
            if not pvalue_ok:
                reasons.append(f"p值不显著(p={ev.p_value})")
            if counter_ratio >= 0.1:
                reasons.append(f"反例比例过高({counter_ratio:.0%})")
            note = "，".join(reasons)
            insufficient_factors.append(f.id)

        lines.append(
            f"- {f.id} {f.name}：样本量 {ev.sample_size}、"
            f"p={ev.p_value}、反例 {ev.counter_examples} → {quality}。{note}"
        )

    # 检查跨活动区数据泄露风险
    lines.append(
        f"\n数据泄露检查：JW-FD 数据集共 {evidence.evidences[0].sample_size if evidence.evidences else '?'} 个样本，"
        f"当前分析基于同一批样本，未发现明显的跨活动区泄露风险。"
    )

    # 最终结论
    lines.append("\n---EVIDENCE_QUALITY---")
    for f in facts.factors:
        ev = ev_by_id.get(f.id)
        if not ev:
            continue
        if f.id in sufficient_factors:
            q = "sufficient"
        elif f.id in insufficient_factors and ev.p_value <= 0.05:
            q = "borderline"
        else:
            q = "insufficient"
        lines.append(
            f"- FACTOR: {f.id} | sample_size: {ev.sample_size} "
            f"| p_value: {ev.p_value} | QUALITY: {q} | NOTE: 见上方分析"
        )
    lines.append("---END_QUALITY---")

    lines.append("\n---VERDICT---")
    valid_str = ", ".join(sufficient_factors) if sufficient_factors else "无"
    weak_str = ", ".join(insufficient_factors) if insufficient_factors else "无"
    lines.append(f"VALIDATED_CHAINS: {valid_str}")
    lines.append(f"WEAK_POINTS: {weak_str}")

    if insufficient_factors:
        lines.append(
            f"NEXT_STEPS: 建议用 SDO/HMI 高分辨率磁场数据对 "
            f"{', '.join(insufficient_factors)} 进行时序追踪验证，"
            f"补充更多活动区样本以提高统计效力"
        )
    else:
        lines.append("NEXT_STEPS: 所有因果链通过审查，可输出最终假设报告")
    lines.append("---END_VERDICT---")
    return "\n".join(lines)


# ───────────────────────── 真实 LLM 辩论（需 autogen + Key） ─────────────────────────

def _format_facts_for_prompt(facts: LiteratureFacts) -> str:
    """将文献因素格式化为可读文本，供 LLM 辩论使用。"""
    lines = []
    for f in facts.factors:
        lines.append(f"  [{f.id}] {f.name}：{f.description}（来源: {f.source}, 置信度: {f.confidence}）")
    return "\n".join(lines)


def _format_evidence_for_prompt(evidence: EvidenceReport) -> str:
    """将证据报告格式化为可读文本，供 LLM 辩论使用。"""
    lines = []
    for e in evidence.evidences:
        verdict = "支持" if e.supported else "不支持"
        lines.append(
            f"  [{e.factor_id}] {verdict} | 相关性={e.correlation}, "
            f"p={e.p_value}, 样本量={e.sample_size}, "
            f"反例={e.counter_examples} | {e.note}"
        )
    return "\n".join(lines)


def _run_llm(facts, evidence, client, rounds, max_rounds) -> List[dict]:
    try:
        import asyncio
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
        from autogen_agentchat.teams import RoundRobinGroupChat

        agents = [
            AssistantAgent("physicist", model_client=client,
                           system_message=ROLE_SYSTEM["physicist"]),
            AssistantAgent("skeptic", model_client=client,
                           system_message=ROLE_SYSTEM["skeptic"]),
            AssistantAgent("methodologist", model_client=client,
                           system_message=ROLE_SYSTEM["methodologist"]),
        ]
        # 终止条件：出现"辩论达成共识"标记 或 达到最大消息数
        termination = (
            TextMentionTermination("NO_NEW_CHALLENGES")
            | MaxMessageTermination(max_rounds * 3 + 2)
        )
        team = RoundRobinGroupChat(agents, termination_condition=termination)

        facts_text = _format_facts_for_prompt(facts)
        evidence_text = _format_evidence_for_prompt(evidence)

        task = (
            "请围绕以下文献因素与数据证据，开展「太阳耀斑触发前兆」因果辩论。\n\n"
            f"## 文献因素\n{facts_text}\n\n"
            f"## 数据证据（来自 {evidence.dataset}）\n{evidence_text}\n\n"
            "请按照各自角色的输出格式要求，进行多轮辩论，"
            "最终由方法论审查给出可验证性结论。"
        )

        transcript: List[dict] = []
        round_counter = [0]

        async def _run():
            async for msg in team.run_stream(task=task):
                speaker = getattr(msg, "source", "unknown")
                if speaker in ROLE_TITLE:
                    # 计算当前轮次（每 3 条消息一轮）
                    current_round = len(transcript) // 3 + 1
                    transcript.append({
                        "round": current_round,
                        "speaker": speaker,
                        "role": ROLE_TITLE[speaker],
                        "content": getattr(msg, "content", str(msg)),
                    })

        asyncio.run(_run())
        return transcript

    except Exception as e:  # pragma: no cover
        print(f"[warn] LLM 辩论失败，回退 mock：{e}")
        return _simulate(facts, evidence, rounds, max_rounds)
