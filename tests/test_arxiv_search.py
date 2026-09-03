"""literature.arxiv_search 的真实 arXiv 联网集成测试。

测试不会构造或模拟 arXiv 响应。正常场景直接调用真实 arXiv API；若 arXiv
暂时不可达、限流或拒绝连接，依赖正常响应的测试会被跳过。

用法：python -m unittest tests.test_arxiv_search
"""
from __future__ import annotations

import os
import sys
import time
import unittest
from uuid import uuid4

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from literature import arxiv_search


LIVE_QUERY = "solar flare"
REQUEST_TIMEOUT_SECONDS = 30.0
REQUEST_INTERVAL_SECONDS = 3.0
INVALID_ARXIV_API_URL = "https://export.arxiv.org/api/query/not-a-real-endpoint"
NON_ATOM_ARXIV_URL = "https://arxiv.org/abs/1706.03762"


class SearchArxivLiveTests(unittest.TestCase):
    """覆盖参数校验以及真实 arXiv 请求的正常、失败和异常响应场景。"""

    _last_request_started_at = 0.0

    @classmethod
    def _search_live(cls, query: str, **kwargs):
        """发起真实请求，并遵守 arXiv 建议的请求间隔。"""
        elapsed = time.monotonic() - cls._last_request_started_at
        if elapsed < REQUEST_INTERVAL_SECONDS:
            time.sleep(REQUEST_INTERVAL_SECONDS - elapsed)

        cls._last_request_started_at = time.monotonic()
        return arxiv_search.search_arxiv(
            query,
            timeout=REQUEST_TIMEOUT_SECONDS,
            **kwargs,
        )

    def _search_or_skip_if_arxiv_unavailable(self, query: str, **kwargs):
        """正常联网场景中，arXiv 不可用时跳过而非误报功能失败。"""
        try:
            return self._search_live(query, **kwargs)
        except arxiv_search.ArxivSearchError as exc:
            self.skipTest(f"arXiv 暂时不可用、限流或拒绝连接：{exc}")

    def test_invalid_input_types_raise_exception(self) -> None:
        """参数类型或取值错误时，应在请求前抛出相应异常。"""
        invalid_calls = (
            ((None,), {}, ValueError),
            ((123,), {}, ValueError),
            ((["solar", "flare"],), {}, ValueError),
            (("",), {}, ValueError),
            (("   ",), {}, ValueError),
            ((LIVE_QUERY,), {"max_results": "3"}, ValueError),
            ((LIVE_QUERY,), {"max_results": 1.5}, ValueError),
            ((LIVE_QUERY,), {"max_results": True}, ValueError),
            ((LIVE_QUERY,), {"start": "0"}, ValueError),
            ((LIVE_QUERY,), {"start": 0.5}, ValueError),
            ((LIVE_QUERY,), {"start": True}, ValueError),
            ((LIVE_QUERY,), {"timeout": "30"}, TypeError),
            ((LIVE_QUERY,), {"timeout": None}, TypeError),
        )

        for args, kwargs, expected_exception in invalid_calls:
            with self.subTest(args=args, kwargs=kwargs):
                with self.assertRaises(expected_exception):
                    arxiv_search.search_arxiv(*args, **kwargs)

    def test_zero_results_are_returned_normally(self) -> None:
        """真实 API 对每次生成的唯一关键词应正常返回零篇论文。"""
        query = f"challengecupzerosearch{uuid4().hex}"
        result = self._search_or_skip_if_arxiv_unavailable(query, max_results=1)

        self.assertEqual(result["query"], query)
        self.assertEqual(result["total_results"], 0)
        self.assertEqual(result["paper_count"], 0)
        self.assertEqual(result["papers"], [])

    def test_multiple_real_results_are_parsed(self) -> None:
        """真实 API 返回多篇论文时，应返回符合模块结构的动态论文信息。"""
        result = self._search_or_skip_if_arxiv_unavailable(LIVE_QUERY, max_results=3)

        self.assertEqual(result["query"], LIVE_QUERY)
        self.assertGreaterEqual(result["total_results"], 2)
        self.assertGreaterEqual(result["paper_count"], 2)
        self.assertLessEqual(result["paper_count"], 3)
        self.assertEqual(result["paper_count"], len(result["papers"]))

        required_paper_fields = {
            "title",
            "doi",
            "abstract",
            "authors",
            "arxiv_id",
            "published",
            "updated",
            "categories",
            "links",
        }
        for paper in result["papers"]:
            with self.subTest(arxiv_id=paper.get("arxiv_id")):
                self.assertTrue(required_paper_fields.issubset(paper))
                self.assertIsInstance(paper["title"], str)
                self.assertTrue(paper["title"])
                self.assertIsInstance(paper["abstract"], str)
                self.assertTrue(paper["abstract"])
                self.assertIsInstance(paper["doi"], (str, type(None)))
                self.assertIsInstance(paper["authors"], list)
                self.assertIsInstance(paper["arxiv_id"], str)
                self.assertTrue(paper["arxiv_id"])
                self.assertIsInstance(paper["categories"], list)
                self.assertIsInstance(paper["links"], list)
                if "introduction" in paper:
                    self.assertIsInstance(paper["introduction"], str)
                    self.assertTrue(paper["introduction"])

    def test_real_arxiv_api_failure_raises_exception(self) -> None:
        """请求真实 arXiv 的不存在 API 路径时，应抛出 ArxivSearchError。"""
        original_api_url = arxiv_search.ARXIV_API_URL
        try:
            arxiv_search.ARXIV_API_URL = INVALID_ARXIV_API_URL
            with self.assertRaises(arxiv_search.ArxivSearchError):
                self._search_live(LIVE_QUERY, max_results=1)
        finally:
            arxiv_search.ARXIV_API_URL = original_api_url

    def test_real_non_atom_response_raises_exception(self) -> None:
        """真实 arXiv 论文 HTML 不是 Atom XML，应触发异常响应处理。"""
        original_api_url = arxiv_search.ARXIV_API_URL
        try:
            arxiv_search.ARXIV_API_URL = NON_ATOM_ARXIV_URL
            try:
                self._search_live(LIVE_QUERY, max_results=1)
            except arxiv_search.ArxivSearchError as exc:
                if "不是有效 XML" not in str(exc):
                    self.skipTest(f"未能取得预期的 arXiv HTML 异常响应：{exc}")
            else:
                self.fail("arXiv 论文 HTML 被错误地当作有效 Atom API 响应")
        finally:
            arxiv_search.ARXIV_API_URL = original_api_url


if __name__ == "__main__":
    unittest.main()
