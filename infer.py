"""标准批量推理入口（B-1 智能体路线）。"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_check.agent import DataEvidenceAgent
from debate.model_client import get_model_client
from debate.scheduler import run_debate
from hypothesis_generator import generate_hypothesis
from literature.agent import LiteratureAgent
from shared.config import load_config
from shared.contracts import to_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B-1 多智能体科学假设批量推理")
    parser.add_argument("--data_root", type=Path, default=None,
                        help="JW-FD zip 文件路径；未指定时自动搜索")
    parser.add_argument("--out", type=Path, required=True,
                        help="输出目录，将写入标准 JSON 结果")
    parser.add_argument("--query", default="太阳耀斑触发前兆因果链发现",
                        help="待研究科学问题")
    parser.add_argument("--real", action="store_true",
                        help="启用 arXiv + 百炼 Qwen；默认使用离线 mock 模式")
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    config = load_config()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # arXiv 是英文文献库：若 --query 是中文科学问题（含非 ASCII 字符），
    # 不能直接当检索词（会返回 0 篇论文），而应传空字符串触发
    # LiteratureAgent 内部用 Qwen 生成英文检索关键词。
    # 纯 ASCII 的英文检索词才直接透传。args.query 仍写入 metadata 作为科学问题记录。
    search_query = args.query if args.query.isascii() else ""
    facts = asyncio.run(LiteratureAgent().run(query=search_query, use_real=args.real))
    dataset_path = None
    if args.data_root is not None:
        candidate = args.data_root.expanduser().resolve()
        if candidate.is_file():
            dataset_path = str(candidate)
        elif candidate.is_dir():
            archives = sorted(candidate.glob("JW-FD*.zip"))
            if archives:
                dataset_path = str(archives[0])

    evidence = DataEvidenceAgent(dataset_path=dataset_path).run(
        facts, use_real=args.real or dataset_path is not None
    )
    client = get_model_client(config) if args.real else None
    transcript, graph = run_debate(facts, evidence, client=client, config=config)
    report = generate_hypothesis(graph, facts, evidence, client=client, transcript=transcript)

    artifacts = {
        "literature_facts.json": to_json(facts),
        "evidence_report.json": to_json(evidence),
        "transcript.json": transcript,
        "causal_graph.json": to_json(graph),
        "hypothesis_report.json": to_json(report),
    }
    for filename, payload in artifacts.items():
        with (out_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    metadata = {
        "task_id": "XH-202619-B1",
        "query": args.query,
        "mode": "real" if args.real else "mock",
        "data_root": str(args.data_root.resolve()) if args.data_root else None,
        "artifacts": list(artifacts),
    }
    with (out_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return out_dir


def main() -> int:
    args = _parse_args()
    try:
        output = run(args)
    except Exception as exc:
        print(f"推理失败: {exc}", file=sys.stderr)
        return 1
    print(f"推理完成，结果写入: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
