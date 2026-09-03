"""
端到端演示（钟菁泽验证链路用）
无需任何 Key / 第三方依赖，mock 模式直接跑通：
  文献 Agent(mock) → 数据证据 Agent(mock) → 辩论调度 → 因果图谱 → 假设报告

运行：python run_demo.py
真实模式：在 .env 配置 DASHSCOPE_API_KEY 后重跑，自动切换 Qwen 辩论。
"""
from __future__ import annotations
import asyncio
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows 终端 GBK 编码兼容：避免非 ASCII 字符（如 Å）导致 UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from literature.agent import LiteratureAgent
from data_check.agent import DataEvidenceAgent
from debate.scheduler import run_debate
from hypothesis_generator import generate_hypothesis
from debate.model_client import get_model_client
from shared.config import load_config
from shared.contracts import to_json

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def main():
    os.makedirs(OUT, exist_ok=True)
    config = load_config()

    print("=" * 60)
    print("  B-1 耀斑触发前兆因果链发现 — 端到端演示")
    print("=" * 60)

    # Step 1: 文献 Agent
    print("\n[1/5] 文献 Agent：抽取耀斑前兆因素...")
    facts = asyncio.run(LiteratureAgent().run())
    print(f"      检索 {facts.papers_searched} 篇论文，抽取 {len(facts.factors)} 个前兆因素")
    for f in facts.factors:
        print(f"      - {f.id} {f.name} (confidence={f.confidence}, source={f.source})")

    # Step 2: 数据证据 Agent
    use_real_data = "--real-data" in sys.argv
    if not use_real_data:
        # 自动检测 JW-FD 是否可用
        from data_check.agent import _find_dataset
        ds_path = _find_dataset()
        if ds_path:
            use_real_data = True
            print(f"  （自动检测到 JW-FD 数据集，切换真实模式）")

    print(f"\n[2/5] 数据证据 Agent：统计验证各前兆因素...")
    print(f"      模式: {'真实 JW-FD' if use_real_data else 'mock'}")
    evidence = DataEvidenceAgent().run(facts, use_real=use_real_data)
    print(f"      数据集: {evidence.dataset}, {len(evidence.evidences)} 个证据项")
    for e in evidence.evidences:
        status = "✓ 支持" if e.supported else "✗ 不支持"
        print(f"      - {e.factor_id}: {status} (corr={e.correlation}, p={e.p_value}, 反例={e.counter_examples})")

    # Step 3: 辩论调度
    print("\n[3/5] 辩论调度器：三角色多轮辩论...")
    client = get_model_client(config)
    transcript, graph = run_debate(facts, evidence, client=client, config=config)
    print(f"      辩论完成：{len(transcript)} 条发言")
    for t in transcript:
        round_label = f"R{t.get('round', '?')}"
        preview = t["content"][:80].replace("\n", " ")
        print(f"      [{round_label}] [{t['role']}] {preview}...")

    # Step 4: 因果图谱
    print(f"\n[4/5] 因果图谱构建...")
    print(f"      {len(graph.nodes)} 个节点, {len(graph.edges)} 条边")
    print(f"      根因: {graph.root_cause}")
    print(f"      辩论轮数: {graph.debate_rounds}")
    for n in graph.nodes:
        print(f"      节点: [{n.ntype}] {n.label}")
    for e in graph.edges:
        print(f"      边: {e.src} --{e.relation}({e.confidence})--> {e.dst}")

    # Step 5: 假设报告
    print(f"\n[5/5] 假设报告生成...")
    report = generate_hypothesis(graph, facts, evidence, client=client, transcript=transcript)
    print(f"      标题: {report.paper_title}")

    # 保存产物
    transcript_path = os.path.join(OUT, "transcript.json")
    graph_path = os.path.join(OUT, "causal_graph.json")
    report_path = os.path.join(OUT, "hypothesis_report.json")

    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(to_json(graph), f, ensure_ascii=False, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(to_json(report), f, ensure_ascii=False, indent=2)

    # 验证输出
    print("\n" + "=" * 60)
    print("  输出验证")
    print("=" * 60)
    errors = _validate_output(facts, evidence, transcript, graph, report)
    if errors:
        print(f"  ⚠ {len(errors)} 个问题:")
        for err in errors:
            print(f"    - {err}")
    else:
        print("  ✓ 所有验证通过")

    print(f"\n  产物已写入: {OUT}")
    print(f"    - transcript.json     ({len(transcript)} 条发言)")
    print(f"    - causal_graph.json   ({len(graph.nodes)} 节点, {len(graph.edges)} 边)")
    print(f"    - hypothesis_report.json (11 字段)")
    print("=" * 60)

    return len(errors) == 0


def _validate_output(facts, evidence, transcript, graph, report) -> list:
    """验证所有输出是否符合契约规范。"""
    errors = []

    # 1. 检查辩论记录格式
    for i, t in enumerate(transcript):
        if "speaker" not in t or "role" not in t or "content" not in t:
            errors.append(f"transcript[{i}] 缺少 speaker/role/content 字段")
        if "round" not in t:
            errors.append(f"transcript[{i}] 缺少 round 字段")

    # 2. 检查因果图谱
    if not graph.nodes:
        errors.append("因果图谱没有节点")
    if not graph.edges:
        errors.append("因果图谱没有边")
    event_nodes = [n for n in graph.nodes if n.ntype == "event"]
    if not event_nodes:
        errors.append("因果图谱缺少事件节点（耀斑爆发）")
    for e in graph.edges:
        if e.confidence < 0 or e.confidence > 1:
            errors.append(f"边 {e.src}->{e.dst} 置信度 {e.confidence} 不在 [0,1] 范围")

    # 3. 检查假设报告 11 字段
    report_fields = [
        "problem_statement", "rationale", "technical_details",
        "datasets", "source", "target", "paper_title",
        "abstract", "methods", "experiments", "results", "references",
    ]
    for field in report_fields:
        val = getattr(report, field, None)
        if val is None or (isinstance(val, str) and not val.strip()):
            errors.append(f"假设报告缺少字段: {field}")
    if not report.references:
        errors.append("假设报告 references 为空（严禁虚构文献）")

    # 4. 检查文献来源真实性格式（arXiv ID 格式）
    for ref in report.references:
        if not ref.startswith("arXiv:") and not ref.startswith("doi:"):
            errors.append(f"参考文献格式可疑: {ref}（应以 arXiv: 或 doi: 开头）")

    return errors


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
