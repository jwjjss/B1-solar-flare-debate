"""太阳耀斑文献因素 Agent。

``LiteratureAgent.run()`` 是本模块的主入口：空查询时使用 Qwen 生成 arXiv
检索关键词，非空查询时直接使用该查询词；每个关键词的因素结果会写入输出文件。
mock 模式将固定数据包装为结果列表并写入输出文件。
"""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
import re
from typing import Any
from autogen_ext.models.openai import OpenAIChatCompletionClient

from debate.model_client import get_model_client, is_mock
from shared.contracts import LiteratureFactor, LiteratureFacts, literature_facts_from
from shared.prompts.literature import (
    EXTRACTION_SYSTEM_PROMPT,
    KEYWORD_DEFINITION_PROMPT,
)

from .LiteratureError import DuplicateError
from .arxiv_search import ArxivSearchError, search_arxiv
from .storage import FactorVectorStore


MOCK_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "shared",
    "mock_data",
    "literature_facts.json",
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "outputs",
    "literature_facts.json",
)
DEFAULT_STORE_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
# 默认每个关键词最多处理 5 篇论文：真实模式下每篇论文都会触发一次 Qwen 调用，
# 关键词数 × 该上限 即为单次运行的调用量上界，设小一些以控制 API 成本。
DEFAULT_MAX_RESULTS_PER_KEYWORD = 5

# 覆盖耀斑因果链中的可观测前兆、物理机制和发生条件。
SEARCH_KEYWORDS = (
    "solar flare precursor magnetic flux emergence shear",
    "solar flare triggering mechanism magnetic reconnection",
    "solar flare preflare condition magnetic topology",
    "solar flare hard X-ray ultraviolet precursor",
    "solar flare eruption flux rope instability",
)
_ALLOWED_FACTOR_TYPES = frozenset({"precursor", "mechanism", "condition"})
_ARXIV_VERSION_SUFFIX = re.compile(r"v\d+$", re.IGNORECASE)


class LiteratureAgentError(RuntimeError):
    """文献检索、模型抽取或持久化流程失败时抛出。"""


class LiteratureAgent:
    """检索文献、抽取单个因素并持久化到 ChromaDB。"""

    def __init__(
        self,
        persist_path: str | None = None,
        *,
        store: FactorVectorStore | None = None,
        model_client: Any | None = None,
    ) -> None:
        if persist_path is not None and (not isinstance(persist_path, str) or not persist_path.strip()):
            raise ValueError("persist_path 必须是非空字符串或 None")
        if persist_path is None and store is None:
            persist_path = DEFAULT_STORE_PATH

        self._persist_path = persist_path
        self._store = store
        self._model_client = model_client

    async def run(
        self,
        query: str = "",
        use_real: bool = False,
        *,
        max_results_per_keyword: int = DEFAULT_MAX_RESULTS_PER_KEYWORD,
        min_queries: int = 3,
        max_queries: int = 6,
    ) -> LiteratureFacts:
        """执行完整流程并返回文献因素契约对象。

        Args:
            query: 非空时作为唯一 arXiv 检索关键词；为空时由 Qwen 生成关键词。
            use_real: ``True`` 执行真实流程（Qwen + arXiv + ChromaDB）；
                ``False``（默认）读取固定 mock 数据，无需任何 API 配置。
            max_results_per_keyword: 每个实际 arXiv 搜索词最多处理的论文数。
            min_queries: ``query`` 为空时要求 Qwen 至少生成的检索关键词数量。
            max_queries: ``query`` 为空时允许 Qwen 最多生成的检索关键词数量。

        Returns:
            ``LiteratureFacts`` 契约对象：所有关键词抽取到的因素按来源去重后
            汇总，``papers_searched`` 为各关键词检索到的论文总数。
            每个关键词的明细仍以列表形式写入输出文件。

        Raises:
            ValueError: 输入参数不合法。
            LiteratureAgentError: Qwen、检索、存储或输出持久化失败。
        """
        if not isinstance(query, str):
            raise ValueError("query 必须是字符串")
        if not isinstance(use_real, bool):
            raise ValueError("use_real 必须是布尔值")
        if (
            isinstance(max_results_per_keyword, bool)
            or not isinstance(max_results_per_keyword, int)
            or max_results_per_keyword <= 0
        ):
            raise ValueError("max_results_per_keyword 必须是正整数")
        if isinstance(min_queries, bool) or not isinstance(min_queries, int) or min_queries <= 0:
            raise ValueError("min_queries 必须是正整数")
        if isinstance(max_queries, bool) or not isinstance(max_queries, int) or max_queries <= 0:
            raise ValueError("max_queries 必须是正整数")
        if min_queries > max_queries:
            raise ValueError("min_queries 不能大于 max_queries")

        if not use_real:
            mock_result = self._load_mock_result()
            self._write_results([mock_result])
            return literature_facts_from(mock_result)

        explicit_query = query.strip()
        model_client = self._get_model_client()
        keywords = (
            (explicit_query,)
            if explicit_query
            else await self._generate_search_keywords(model_client, min_queries, max_queries)
        )

        results: list[dict[str, Any]] = []
        store: FactorVectorStore | None = None
        analyzed_sources: set[str] = set()
        for keyword in keywords:
            papers = self._search_unique_papers((keyword,), max_results_per_keyword)
            factors: list[dict[str, Any]] = []

            for source, paper in papers:
                if source in analyzed_sources and store is not None:
                    stored_factor = self._get_stored_factor(store, source)
                    if stored_factor is not None:
                        factors.append(asdict(stored_factor))
                        continue

                extracted_factor = await self._extract_factor_from_paper(
                    model_client,
                    source,
                    paper,
                )
                if extracted_factor is None:
                    continue

                metadata: dict[str, str | float] = {
                    "name": extracted_factor["name"],
                    "description": extracted_factor["description"],
                    "source": source,
                    "confidence": extracted_factor["confidence"],
                    "ftype": extracted_factor["ftype"],
                }

                if store is None:
                    store = self._get_store()

                try:
                    factor_id = store.insert(metadata)
                    analyzed_sources.add(source)
                except DuplicateError:
                    stored_factor = self._get_stored_factor(store, source)
                    if stored_factor is not None:
                        factors.append(asdict(stored_factor))
                    continue
                except Exception as exc:
                    raise LiteratureAgentError(f"写入文献因素失败（{source}）：{exc}") from exc

                if not isinstance(factor_id, str) or not factor_id.strip():
                    raise LiteratureAgentError("存储层未返回有效的 LiteratureFactor ID")

                factors.append(
                    asdict(
                        LiteratureFactor(
                            id=factor_id,
                            name=metadata["name"],
                            description=metadata["description"],
                            source=source,
                            confidence=float(metadata["confidence"]),
                            ftype=metadata["ftype"],
                        )
                    )
                )

            results.append(
                {
                    "query": keyword,
                    "papers_searched": len(papers),
                    "factors": factors,
                }
            )

        self._write_results(results)
        return self._aggregate_to_contract(results, explicit_query)

    @staticmethod
    def _aggregate_to_contract(
        results: list[dict[str, Any]],
        explicit_query: str,
    ) -> LiteratureFacts:
        """把多关键词结果汇总为单个 ``LiteratureFacts`` 契约对象。

        因素按来源去重（同一篇论文可能被多个关键词检索到），``papers_searched``
        取各关键词检索到的论文总数，``query`` 为显式查询或全部关键词的拼接。
        """
        factors: list[LiteratureFactor] = []
        seen_sources: set[str] = set()
        total_papers = 0
        for result in results:
            total_papers += int(result.get("papers_searched", 0))
            for factor in result.get("factors", []):
                source = factor.get("source")
                if source in seen_sources:
                    continue
                seen_sources.add(source)
                factors.append(LiteratureFactor(**factor))

        query = explicit_query or "; ".join(r["query"] for r in results)
        return LiteratureFacts(
            query=query,
            papers_searched=total_papers,
            factors=factors,
        )

    @staticmethod
    def _load_mock_result() -> dict[str, Any]:
        """读取显式 mock 模式所需的固定契约数据。"""
        with open(MOCK_PATH, "r", encoding="utf-8") as mock_file:
            result = json.load(mock_file)
        if not isinstance(result, dict):
            raise LiteratureAgentError("mock 文献数据必须是字典")
        return result

    @staticmethod
    def _write_results(results: list[dict[str, Any]]) -> None:
        """在所有关键词处理成功后覆盖输出结果文件。"""
        output_dir = os.path.dirname(OUTPUT_PATH)
        temporary_path = f"{OUTPUT_PATH}.tmp"
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(temporary_path, "w", encoding="utf-8") as output_file:
                json.dump(results, output_file, ensure_ascii=False, indent=2)
                output_file.write("\n")
            os.replace(temporary_path, OUTPUT_PATH)
        except Exception as exc:
            try:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
            except OSError:
                pass
            raise LiteratureAgentError(f"写入文献因素结果失败：{exc}") from exc

    @staticmethod
    def _get_stored_factor(
        store: FactorVectorStore,
        source: str,
    ) -> LiteratureFactor | None:
        """按来源读取唯一的已存储因素，供重复论文复用。"""
        try:
            factors = store.query(where={"source": source})
        except Exception as exc:
            raise LiteratureAgentError(f"读取已存储文献因素失败（{source}）：{exc}") from exc
        if len(factors) > 1:
            raise LiteratureAgentError(f"数据库中存在重复文献因素（{source}）")
        return factors[0] if factors else None

    def _get_model_client(self) -> Any:
        """延迟创建 Qwen 客户端，避免 mock 模式依赖真实配置。"""
        if self._model_client is None:
            self._model_client = get_model_client()

        if is_mock(self._model_client):
            raise LiteratureAgentError(
                "未能初始化 Qwen 客户端；请设置 DASHSCOPE_API_KEY 或 BAILIAN_API_KEY"
            )
        return self._model_client

    def _get_store(self) -> FactorVectorStore:
        """仅在产生可写入因素时初始化 ChromaDB。"""
        if self._store is None:
            try:
                self._store = FactorVectorStore(self._persist_path)
            except Exception as exc:
                raise LiteratureAgentError(f"无法初始化 ChromaDB：{exc}") from exc
        return self._store

    @staticmethod
    def _paper_source(paper: dict[str, Any]) -> str | None:
        """将 arXiv 论文记录规范化为可验证且稳定的来源字符串。"""
        arxiv_id = paper.get("arxiv_id")
        if isinstance(arxiv_id, str) and arxiv_id.strip():
            normalized_id = _ARXIV_VERSION_SUFFIX.sub("", arxiv_id.strip())
            return f"arXiv:{normalized_id}"

        doi = paper.get("doi")
        if isinstance(doi, str) and doi.strip():
            return f"DOI:{doi.strip()}"
        return None

    def _search_unique_papers(
        self,
        keywords: tuple[str, ...],
        max_results_per_keyword: int,
    ) -> list[tuple[str, dict[str, Any]]]:
        """按给定关键词检索，并按照来源去除单个关键词内的重复论文。"""
        papers: list[tuple[str, dict[str, Any]]] = []
        seen_sources: set[str] = set()

        # 按生成顺序遍历关键词
        for keyword in keywords:
            try:
                search_result = search_arxiv(keyword, max_results=max_results_per_keyword)
            except ArxivSearchError as exc:
                raise LiteratureAgentError(f"arXiv 检索失败（{keyword}）：{exc}") from exc

            raw_papers = search_result.get("papers")
            if not isinstance(raw_papers, list):
                raise LiteratureAgentError(f"arXiv 返回的 papers 字段非法（{keyword}）")

            # 遍历每一篇搜索到的论文
            for paper in raw_papers:
                if not isinstance(paper, dict):
                    raise LiteratureAgentError(f"arXiv 返回了非字典论文记录（{keyword}）")

                source = self._paper_source(paper)

                # 判重
                if source is None or source in seen_sources:
                    continue

                seen_sources.add(source)
                papers.append((source, paper))

        return papers

    async def _generate_search_keywords(
        self,
        model_client: OpenAIChatCompletionClient,
        min_queries: int,
        max_queries: int,
    ) -> tuple[str, ...]:
        """调用 Qwen 生成并验证 arXiv 检索关键词。"""
        messages = self._build_keyword_messages(min_queries, max_queries)
        try:
            response = await self._create_completion(model_client, messages)
        except Exception as exc:
            raise LiteratureAgentError(f"Qwen 生成文献检索关键词失败：{exc}") from exc

        content = response if isinstance(response, str) else getattr(response, "content", None)
        if not isinstance(content, str):
            raise LiteratureAgentError("Qwen 关键词响应不是文本")

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LiteratureAgentError("Qwen 未返回合法的关键词 JSON") from exc

        return self._validate_keyword_payload(payload, min_queries, max_queries)

    @staticmethod
    def _build_keyword_messages(min_queries: int, max_queries: int) -> list[Any]:
        """创建关键词生成所需的 AutoGen 消息。"""
        try:
            from autogen_core.models import SystemMessage, UserMessage
        except ImportError as exc:
            raise LiteratureAgentError("未安装 AutoGen 消息依赖，无法生成检索关键词") from exc

        request = json.dumps(
            {"min_queries": min_queries, "max_queries": max_queries},
            ensure_ascii=False,
        )
        return [
            SystemMessage(content=KEYWORD_DEFINITION_PROMPT),
            UserMessage(content=request, source="literature_agent"),
        ]

    @staticmethod
    def _validate_keyword_payload(
        payload: Any,
        min_queries: int,
        max_queries: int,
    ) -> tuple[str, ...]:
        """验证 Qwen 返回的关键词列表。"""
        if not isinstance(payload, dict) or "queries" not in payload:
            raise LiteratureAgentError("Qwen 关键词输出缺少 queries 字段")

        queries = payload["queries"]
        if not isinstance(queries, list):
            raise LiteratureAgentError("Qwen 的 queries 字段必须是数组")
        if not min_queries <= len(queries) <= max_queries:
            raise LiteratureAgentError(
                f"Qwen 返回的关键词数量必须位于 {min_queries} 到 {max_queries} 之间"
            )

        normalized: list[str] = []
        seen: set[str] = set()
        for index, query in enumerate(queries, start=1):
            if not isinstance(query, str) or not query.strip():
                raise LiteratureAgentError(f"Qwen 返回的第 {index} 个关键词非法")

            cleaned = " ".join(query.split())
            terms = cleaned.split(" ")
            normalized_terms = [term.casefold() for term in terms]
            lowered = cleaned.casefold()
            if not any(
                normalized_terms[position : position + 2] == ["solar", "flare"]
                for position in range(len(normalized_terms) - 1)
            ):
                raise LiteratureAgentError(
                    f"Qwen 返回的第 {index} 个关键词必须包含 solar flare"
                )
            if not 3 <= len(terms) <= 8:
                raise LiteratureAgentError(
                    f"Qwen 返回的第 {index} 个关键词必须包含 3 到 8 个词"
                )
            if {"and", "or", "not"}.intersection(normalized_terms):
                raise LiteratureAgentError(f"Qwen 返回的第 {index} 个关键词包含布尔运算符")
            if any(character in cleaned for character in {'"', "'", "(", ")", "*", ":"}):
                raise LiteratureAgentError(f"Qwen 返回的第 {index} 个关键词包含禁用语法")
            if lowered in seen:
                raise LiteratureAgentError("Qwen 返回了重复的文献检索关键词")

            seen.add(lowered)
            normalized.append(cleaned)

        return tuple(normalized)

    async def _extract_factor_from_paper(
        self,
        model_client: Any,
        source: str,
        paper: dict[str, Any],
    ) -> dict[str, Any] | None:
        """调用 Qwen 并验证其 ``{"factor": ...}`` JSON 输出。"""

        # 构建 AutoGen 消息
        messages = self._build_messages(source, paper)

        try:
            response = await self._create_completion(model_client, messages)
        except Exception as exc:
            raise LiteratureAgentError(f"Qwen 分析论文失败（{source}）：{exc}") from exc

        content = response if isinstance(response, str) else getattr(response, "content", None)
        if not isinstance(content, str):
            raise LiteratureAgentError(f"Qwen 返回了非文本内容（{source}）")

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LiteratureAgentError(f"Qwen 未返回合法 JSON（{source}）") from exc

        # 验证合法性后输出
        return self._validate_extraction_payload(payload, source)

    @staticmethod
    def _build_messages(source: str, paper: dict[str, Any]) -> list[Any]:
        """创建 AutoGen 消息，论文原文仅作为待分析证据。"""

        # AutoGen消息依赖检测
        try:
            from autogen_core.models import SystemMessage, UserMessage
        except ImportError as exc:
            raise LiteratureAgentError("未安装 AutoGen 消息依赖，无法调用 Qwen") from exc

        title = paper.get("title")
        abstract = paper.get("abstract")
        introduction = paper.get("introduction")
        if not isinstance(title, str) or not title.strip():
            raise LiteratureAgentError(f"论文标题缺失（{source}）")
        if not isinstance(abstract, str) or not abstract.strip():
            raise LiteratureAgentError(f"论文摘要缺失（{source}）")

        paper_text = (
            f"Paper source: {source}\n"
            f"Title: {title.strip()}\n\n"
            f"Abstract:\n{abstract.strip()}"
        )
        if isinstance(introduction, str) and introduction.strip():
            paper_text += f"\n\nIntroduction:\n{introduction.strip()}"

        # 返回系统消息（预置prompt）+ 用户消息（论文来源、论文标题、论文摘要）
        return [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            UserMessage(content=paper_text, source="literature_agent"),
        ]

    @staticmethod
    async def _create_completion(
        model_client: OpenAIChatCompletionClient,
        messages: list[Any],
    ) -> Any:
        """调用客户端的异步 create 接口。"""
        try:
            return await model_client.create(
                messages,
                json_output=True,
                extra_create_args={"temperature": 0},
            )
        except TypeError:
            # 兼容不支持 JSON/额外参数的旧版客户端；系统提示词仍要求 JSON。
            return await model_client.create(messages)

    @staticmethod
    def _validate_extraction_payload(payload: Any, source: str) -> dict[str, Any] | None:
        """验证模型单因素信封，拒绝不符合契约的内容。"""
        if not isinstance(payload, dict) or "factor" not in payload:
            raise LiteratureAgentError(f"Qwen 输出缺少 factor 字段（{source}）")

        factor = payload["factor"]
        if factor is None:
            return None
        if not isinstance(factor, dict):
            raise LiteratureAgentError(f"Qwen 的 factor 字段必须是对象或 null（{source}）")

        name = factor.get("name")
        description = factor.get("description")
        confidence = factor.get("confidence")
        ftype = factor.get("ftype")

        if not isinstance(name, str) or not name.strip():
            raise LiteratureAgentError(f"Qwen 返回的 name 非法（{source}）")
        if not isinstance(description, str) or not description.strip():
            raise LiteratureAgentError(f"Qwen 返回的 description 非法（{source}）")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise LiteratureAgentError(f"Qwen 返回的 confidence 非法（{source}）")
        if not math.isfinite(float(confidence)) or not 0.0 < float(confidence) < 1.0:
            raise LiteratureAgentError(f"Qwen 返回的 confidence 必须位于 0 到 1 之间（{source}）")
        if not isinstance(ftype, str) or ftype not in _ALLOWED_FACTOR_TYPES:
            raise LiteratureAgentError(f"Qwen 返回的 ftype 非法（{source}）")

        return {
            "name": name.strip(),
            "description": description.strip(),
            "confidence": float(confidence),
            "ftype": ftype,
        }

__all__ = ["LiteratureAgent", "LiteratureAgentError", "SEARCH_KEYWORDS"]
