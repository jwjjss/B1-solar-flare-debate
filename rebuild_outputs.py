"""
用现有 outputs/real_final/ 的产物（literature_facts / evidence_report / transcript）
重新运行 causal_graph_builder + hypothesis_generator，得到修复后的一致性产物。

不重跑 Qwen / arXiv / JW-FD 加载，纯本地重建，零 API 成本。

用法：
    python rebuild_outputs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.contracts import (  # noqa: E402
    literature_facts_from,
    evidence_report_from,
    to_json,
)
from causal_graph_builder import build_causal_graph  # noqa: E402
from hypothesis_generator import generate_hypothesis  # noqa: E402


SRC = Path(__file__).resolve().parent / "outputs" / "real_final"
DST = Path(__file__).resolve().parent / "outputs" / "real_final_v2"
DST.mkdir(parents=True, exist_ok=True)

# JW-FD 实际特征列数（data_check/agent.py: 前 29 列纯数值特征）
JWFD_N_FEATURES = 29


def main() -> None:
    facts = literature_facts_from(json.loads((SRC / "literature_facts.json").read_text(encoding="utf-8")))
    ev_dict = json.loads((SRC / "evidence_report.json").read_text(encoding="utf-8"))
    # 旧产物没有 n_features 字段，补上真实值
    ev_dict.setdefault("n_features", JWFD_N_FEATURES)
    evidence = evidence_report_from(ev_dict)
    transcript = json.loads((SRC / "transcript.json").read_text(encoding="utf-8"))

    actual_rounds = max((t.get("round", 0) for t in transcript), default=0)
    print(f"[rebuild] 辩论实际轮次: {actual_rounds} (消息数 {len(transcript)})")
    print(f"[rebuild] 文献因素: {[f.id for f in facts.factors]}")
    print(f"[rebuild] 数据证据: {[(e.factor_id, e.supported, e.correlation, e.p_value) for e in evidence.evidences]}")

    graph = build_causal_graph(facts, evidence, transcript, actual_rounds)
    report = generate_hypothesis(graph, facts, evidence, client=None, transcript=transcript)

    # 写新产物
    (DST / "literature_facts.json").write_text(
        json.dumps(to_json(facts), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DST / "evidence_report.json").write_text(
        json.dumps(to_json(evidence), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DST / "transcript.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DST / "causal_graph.json").write_text(
        json.dumps(to_json(graph), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DST / "hypothesis_report.json").write_text(
        json.dumps(to_json(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 拷贝 run_metadata 并标注 v2
    meta_src = SRC / "run_metadata.json"
    if meta_src.exists():
        meta = json.loads(meta_src.read_text(encoding="utf-8"))
        meta["rebuild_note"] = (
            "v2: 修复 causal_graph_builder 边置信度公式(1-p→|r|)、"
            "root_cause 排序键(literature conf→data corr)、"
            "debate_rounds(len→max round)、ntype(硬编码→f.ftype)、"
            "hypothesis_generator 特征数(55→实际)、"
            "_extract_block 兼容 ---END_CLAIMS--- 别名。"
            "基于 v1 的 transcript/evidence/facts 本地重建，未重跑 Qwen/arXiv。"
        )
        (DST / "run_metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print("\n[rebuild] === causal_graph.json ===")
    print(json.dumps(to_json(graph), ensure_ascii=False, indent=2))
    print("\n[rebuild] === hypothesis_report.technical_details ===")
    print(report.technical_details)
    print("\n[rebuild] === hypothesis_report.datasets ===")
    print(report.datasets)
    print(f"\n[rebuild] 新产物写入: {DST}")


if __name__ == "__main__":
    main()
