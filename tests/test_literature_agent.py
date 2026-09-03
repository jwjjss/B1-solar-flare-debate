"""``LiteratureAgent.run`` 的真实集成测试。

``use_real=True`` 的测试会实际调用已配置的 Qwen、arXiv 和 ChromaDB；因此需要
有效的模型 API 配置与网络连接。mock JSON 仅作为项目约定的数据结构模板，不能被用作
真实 LLM 输出的固定预期值。
"""
from __future__ import annotations

from dataclasses import asdict
import asyncio
import json
from pathlib import Path
import re
from typing import Any

import pytest

import literature.agent as agent_module
from literature.agent import LiteratureAgent
from literature.storage import FactorVectorStore
from shared.contracts import LiteratureFactor, LiteratureFacts, literature_facts_from


@pytest.fixture
def mock_literature_facts() -> dict[str, Any]:
    """读取项目实际使用的文献因素 mock 契约数据。"""

    with open(agent_module.MOCK_PATH, "r", encoding="utf-8") as mock_file:
        return json.load(mock_file)


@pytest.fixture
def output_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """将真实输出写入 pytest 临时目录，避免改动项目输出文件。"""

    path = tmp_path / "literature_facts.json"
    monkeypatch.setattr(agent_module, "OUTPUT_PATH", str(path))
    return path


def _read_output(path: Path) -> list[dict[str, Any]]:
    assert path.is_file(), "run() 未生成输出文件"
    with path.open("r", encoding="utf-8") as output_file:
        return json.load(output_file)


def _assert_matches_mock_structure(
    actual: Any,
    expected: Any,
    *,
    location: str,
) -> None:
    """按实际 mock 数据递归验证 JSON 字段和数据类型，而非固定真实内容。"""

    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{location} 必须是字典"
        assert set(actual) == set(expected), f"{location} 的字段不符合 mock 数据契约"
        for key, expected_value in expected.items():
            _assert_matches_mock_structure(
                actual[key],
                expected_value,
                location=f"{location}.{key}",
            )
        return

    if isinstance(expected, list):
        assert isinstance(actual, list), f"{location} 必须是列表"
        assert expected, f"{location} 的 mock 数据模板不能为空"
        for index, item in enumerate(actual):
            _assert_matches_mock_structure(
                item,
                expected[0],
                location=f"{location}[{index}]",
            )
        return

    assert type(actual) is type(expected), (
        f"{location} 的类型应为 {type(expected).__name__}，"
        f"实际为 {type(actual).__name__}"
    )


def _assert_factor_contract(factor: dict[str, Any], template: dict[str, Any]) -> None:
    """验证单个因素符合 mock JSON 结构及 ``LiteratureFactor`` 契约。"""

    _assert_matches_mock_structure(factor, template, location="factor")
    literature_factor = LiteratureFactor(**factor)

    assert re.fullmatch(r"F[1-9]\d*", literature_factor.id)
    assert literature_factor.name.strip()
    assert literature_factor.description.strip()
    assert literature_factor.source.startswith(("arXiv:", "DOI:"))
    assert 0.0 < literature_factor.confidence < 1.0
    assert literature_factor.ftype in {"precursor", "mechanism", "condition"}


def _assert_result_contract(
    results: list[dict[str, Any]],
    mock_literature_facts: dict[str, Any],
) -> None:
    """验证真实运行结果中的每个输出项都符合 mock 数据契约。"""

    assert isinstance(results, list)
    assert results, "run() 未返回任何检索结果"

    factor_template = mock_literature_facts["factors"][0]
    for result in results:
        _assert_matches_mock_structure(
            result,
            mock_literature_facts,
            location="literature_result",
        )
        assert result["query"].strip()
        assert result["papers_searched"] >= 0
        for factor in result["factors"]:
            _assert_factor_contract(factor, factor_template)

    assert any(result["factors"] for result in results), "真实流程未抽取到任何文献因素"


def _assert_factors_exist_in_chromadb(
    results: list[dict[str, Any]],
    store: FactorVectorStore,
) -> None:
    """使用 ``FactorVectorStore.query`` 验证输出因素已真实持久化到 ChromaDB。"""

    for result in results:
        for output_factor in result["factors"]:
            stored_factors = store.query(where={"source": output_factor["source"]})
            assert len(stored_factors) == 1
            assert asdict(stored_factors[0]) == output_factor


def test_default_constructor_falls_back_to_default_store_path() -> None:
    agent = LiteratureAgent()
    assert agent._persist_path == agent_module.DEFAULT_STORE_PATH
    assert agent._store is None


def test_constructor_rejects_blank_persist_path() -> None:
    with pytest.raises(ValueError, match="非空字符串"):
        LiteratureAgent(persist_path="   ")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": None},
        {"query": 42},
        {"use_real": "False"},
        {"use_real": 1},
        {"max_results_per_keyword": 0},
        {"max_results_per_keyword": True},
        {"min_queries": 0},
        {"min_queries": 1.5},
        {"max_queries": []},
        {"min_queries": 4, "max_queries": 3},
    ],
    ids=[
        "query-none",
        "query-not-string",
        "use-real-not-bool",
        "use-real-integer",
        "max-results-zero",
        "max-results-bool",
        "min-queries-zero",
        "min-queries-not-integer",
        "max-queries-not-integer",
        "minimum-exceeds-maximum",
    ],
)
def test_run_rejects_invalid_arguments(tmp_path: Path, kwargs: dict[str, Any]) -> None:
    agent = LiteratureAgent(persist_path=str(tmp_path / "chroma_db"))

    with pytest.raises(ValueError):
        asyncio.run(agent.run(**kwargs))


def test_run_without_real_returns_mock_contract(
    tmp_path: Path,
    output_path: Path,
    mock_literature_facts: dict[str, Any],
) -> None:
    agent = LiteratureAgent(persist_path=str(tmp_path / "chroma_db"))

    facts = asyncio.run(agent.run(use_real=False))

    assert isinstance(facts, LiteratureFacts)
    assert facts == literature_facts_from(mock_literature_facts)
    assert _read_output(output_path) == [mock_literature_facts]


def test_run_with_real_llm_is_stable_and_persists_every_factor(
    tmp_path: Path,
    output_path: Path,
    mock_literature_facts: dict[str, Any],
) -> None:
    """连续执行三次真实流程，并分别验证契约返回值、输出文件与 ChromaDB 持久化。

    ``run`` 返回聚合后的 ``LiteratureFacts``；每个检索关键词的明细写入输出文件，
    因此关键词数量与逐条结构从输出文件校验。``max_results_per_keyword=2`` 用于
    限制单次运行的 Qwen 调用量。
    """

    persist_path = str(tmp_path / "literature_chroma")
    agent = LiteratureAgent(persist_path=persist_path)
    scenarios = (
        (
            "explicit-query",
            {"query": "solar flare precursor"},
            1,
            1,
        ),
        (
            "default-query-count",
            {},
            3,
            6,
        ),
        (
            "custom-query-count",
            {"min_queries": 6, "max_queries": 9},
            6,
            9,
        ),
    )

    for scenario_name, kwargs, minimum_results, maximum_results in scenarios:
        facts = asyncio.run(
            agent.run(use_real=True, max_results_per_keyword=2, **kwargs)
        )

        # 返回值必须是聚合后的契约对象
        assert isinstance(facts, LiteratureFacts), scenario_name
        factor_template = mock_literature_facts["factors"][0]
        for factor in facts.factors:
            _assert_factor_contract(asdict(factor), factor_template)

        # 输出文件保存逐关键词明细，用于校验关键词数量与结构
        results = _read_output(output_path)
        assert minimum_results <= len(results) <= maximum_results, scenario_name
        _assert_result_contract(results, mock_literature_facts)

        # 契约中的因素数量不应超过输出文件中因素总数（来源去重）
        file_factor_count = sum(len(r["factors"]) for r in results)
        assert len(facts.factors) <= file_factor_count, scenario_name

        # 重新通过真实持久化路径打开存储，而不是使用 run() 内部持有的对象。
        store = FactorVectorStore(persist_path)
        _assert_factors_exist_in_chromadb(results, store)
