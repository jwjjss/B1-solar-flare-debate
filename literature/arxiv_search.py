"""通过 arXiv Atom API 检索论文。"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


ARXIV_API_URL = "https://export.arxiv.org/api/query"
_DEFAULT_MAX_RESULTS = 10
_DEFAULT_TIMEOUT = 30.0
_USER_AGENT = "challenge-cup-literature-agent/1.0"

_NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}


class ArxivSearchError(RuntimeError):
    """arXiv 请求或响应解析失败时抛出的异常。"""


@dataclass(frozen=True)
class _ApiResponse:
    body: bytes


def _clean_text(value: str | None) -> str | None:
    """去掉 Atom 文本中的首尾空白，并合并换行。"""
    if value is None:
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def _text(element: ET.Element, path: str) -> str | None:
    return _clean_text(element.findtext(path, default=None, namespaces=_NAMESPACES))


def _local_name(tag: str) -> str:
    """返回 XML 标签的本地名，兼容带命名空间和不带命名空间的标签。"""
    return tag.rsplit("}", 1)[-1]


def _optional_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _extract_doi(entry: ET.Element) -> str | None:
    doi = _text(entry, "arxiv:doi")
    if doi:
        return doi

    # 某些代理会把 DOI 放在 title="doi" 的 link 中，而不是 arxiv:doi 节点。
    for link in entry.findall("atom:link", _NAMESPACES):
        title = (link.get("title") or "").strip().lower()
        href = _clean_text(link.get("href"))
        if title == "doi" and href:
            return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", href, flags=re.IGNORECASE)
    return None


def _extract_links(entry: ET.Element) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    for link in entry.findall("atom:link", _NAMESPACES):
        href = _clean_text(link.get("href"))
        if not href:
            continue
        item: dict[str, str] = {"url": href}
        for key in ("rel", "title", "type", "href"):
            value = _clean_text(link.get(key))
            if value:
                item[key] = value
        # href 已经作为 url 保存，保留其他元数据但避免重复字段。
        item.pop("href", None)
        links.append(item)
    return links


def _extract_introduction(entry: ET.Element) -> str | None:
    """仅在 API 确实提供 introduction 字段时返回它。

    标准 arXiv Atom API 通常只提供 summary（摘要），不提供论文引言，
    因此正常情况下该函数返回 None，调用方不会添加 introduction 键。
    """
    for child in entry.iter():
        if child is entry:
            continue
        if _local_name(child.tag).lower() in {"introduction", "intro"}:
            introduction = _clean_text(child.text)
            if introduction:
                return introduction
    return None


def _parse_entry(entry: ET.Element) -> dict[str, Any]:
    entry_url = _text(entry, "atom:id")
    paper: dict[str, Any] = {
        "title": _text(entry, "atom:title") or "",
        "doi": _extract_doi(entry),
        "abstract": _text(entry, "atom:summary") or "",
        "authors": [
            author_name
            for author in entry.findall("atom:author", _NAMESPACES)
            if (author_name := _text(author, "atom:name"))
        ],
        "arxiv_id": entry_url.rsplit("/", 1)[-1] if entry_url else None,
        "published": _text(entry, "atom:published"),
        "updated": _text(entry, "atom:updated"),
        "categories": [
            term
            for category in entry.findall("atom:category", _NAMESPACES)
            if (term := _clean_text(category.get("term")))
        ],
        "links": _extract_links(entry),
    }

    primary_category = entry.find("arxiv:primary_category", _NAMESPACES)
    if primary_category is not None:
        primary_term = _clean_text(primary_category.get("term"))
        if primary_term:
            paper["primary_category"] = primary_term

    for field, path in (
        ("comment", "arxiv:comment"),
        ("journal_ref", "arxiv:journal_ref"),
    ):
        value = _text(entry, path)
        if value:
            paper[field] = value

    introduction = _extract_introduction(entry)
    if introduction:
        # arXiv API 没有引言时，完全不创建该键，而不是返回空值。
        paper["introduction"] = introduction

    return paper


def _request(query: str, start: int, max_results: int, timeout: float) -> _ApiResponse:
    params = urlencode(
        {
            "search_query": f"all:{query}",
            "start": start,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    request = Request(
        f"{ARXIV_API_URL}?{params}",
        headers={"User-Agent": _USER_AGENT},
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            return _ApiResponse(response.read())
    except HTTPError as exc:
        raise ArxivSearchError(f"arXiv API 请求失败（HTTP {exc.code}）") from exc
    except URLError as exc:
        raise ArxivSearchError(f"无法连接 arXiv API：{exc.reason}") from exc
    except TimeoutError as exc:
        raise ArxivSearchError("arXiv API 请求超时") from exc


def search_arxiv(
    query: str,
    *,
    start: int = 0,
    max_results: int = _DEFAULT_MAX_RESULTS,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """按关键字搜索 arXiv 并返回论文信息。

    Args:
        query: 搜索关键字，例如 ``"太阳耀斑触发前兆"``。
        max_results: 本次 API 请求最多返回的论文数。
        start: 从匹配结果中的第几篇开始，适用于分页。
        timeout: HTTP 请求超时时间（秒）。

    Returns:
        包含以下字段的字典：

        ``query``
            原始搜索关键字。
        ``total_results``
            arXiv API 报告的匹配总数；若响应未提供该字段，则使用本次返回数。
        ``paper_count``
            本次实际返回的论文数。
        ``papers``
            论文信息列表。标准 API 没有引言字段，因此论文字典中不会包含
            ``introduction``；只有 API 明确返回该字段时才会添加它。

    Raises:
        ValueError: 参数不合法。
        ArxivSearchError: 网络请求失败或 API 返回了无效 XML。
    """
    # 类型判断
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串")
    if not isinstance(max_results, int) or isinstance(max_results, bool) or max_results <= 0:
        raise ValueError("max_results 必须是正整数")
    if not isinstance(start, int) or isinstance(start, bool) or start < 0:
        raise ValueError("start 必须是非负整数")
    if timeout <= 0:
        raise ValueError("timeout 必须大于 0")

    response = _request(query.strip(), start, max_results, timeout)
    try:
        root = ET.fromstring(response.body)
    except ET.ParseError as exc:
        raise ArxivSearchError("arXiv API 返回的内容不是有效 XML") from exc

    entries = root.findall("atom:entry", _NAMESPACES)
    papers = [_parse_entry(entry) for entry in entries]

    total_results = _optional_int(
        root.findtext(
            "opensearch:totalResults",
            default=None,
            namespaces=_NAMESPACES,
        )
    )
    if total_results is None:
        total_results = len(papers)

    return {
        "query": query.strip(),
        "total_results": total_results,
        "paper_count": len(papers),
        "papers": papers,
    }


__all__ = ["ArxivSearchError", "search_arxiv"]
