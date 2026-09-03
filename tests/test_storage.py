"""FactorVectorStore 的输入校验、CRUD 与检索流程测试。"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import literature.storage as storage_module
from literature.storage import DuplicateError, FactorVectorStore, MetadataError
from shared.contracts import LiteratureFactor


class _FakeEmbeddingFunction:
    """测试用 embedding function；不下载或加载真实模型。"""


class _FakeCollection:
    """一个只实现 FactorVectorStore 所需接口的内存版 Chroma collection。"""

    _metadata_fields = {
        "id",
        "name",
        "description",
        "source",
        "confidence",
        "ftype",
    }
    _operators = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"}

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def _matches(self, metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
        if where is None or where == {}:
            return True
        if not isinstance(where, dict):
            raise TypeError("where 必须是字典")

        logical_keys = [key for key in where if key in {"$and", "$or"}]
        if logical_keys:
            if len(where) != 1:
                raise ValueError("逻辑条件不能与其他条件混用")
            clauses = where[logical_keys[0]]
            if not isinstance(clauses, list) or not clauses:
                raise ValueError("逻辑条件必须是非空列表")
            matched = [self._matches(metadata, clause) for clause in clauses]
            return all(matched) if logical_keys[0] == "$and" else any(matched)

        for field, condition in where.items():
            if field not in self._metadata_fields:
                raise ValueError(f"未知 metadata 字段: {field}")

            actual = metadata.get(field)
            if isinstance(condition, dict):
                if len(condition) != 1:
                    raise ValueError("单个字段只能有一个比较条件")
                operator, expected = next(iter(condition.items()))
                if operator not in self._operators:
                    raise ValueError(f"不支持的过滤操作符: {operator}")
                if operator in {"$in", "$nin"}:
                    if not isinstance(expected, list) or not expected:
                        raise ValueError("$in/$nin 的值必须是非空列表")
                    result = actual in expected
                    if operator == "$nin":
                        result = not result
                else:
                    try:
                        if operator == "$eq":
                            result = actual == expected
                        elif operator == "$ne":
                            result = actual != expected
                        elif operator == "$gt":
                            result = actual > expected
                        elif operator == "$gte":
                            result = actual >= expected
                        elif operator == "$lt":
                            result = actual < expected
                        else:  # $lte
                            result = actual <= expected
                    except TypeError as error:
                        raise ValueError("过滤值类型不合法") from error
            elif isinstance(condition, (str, int, float, bool)):
                result = actual == condition
            else:
                raise ValueError("过滤值类型不合法")

            if not result:
                return False
        return True

    def _selected(
        self,
        ids: list[str] | str | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        requested = None
        if ids is not None:
            requested_values = ids if isinstance(ids, list) else [ids]
            requested = {str(value) for value in requested_values}

        selected = []
        for factor_id, record in self._records.items():
            if requested is not None and factor_id not in requested:
                continue
            if self._matches(record["metadata"], where):
                selected.append((factor_id, record))
        return selected

    def add(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for factor_id, document, metadata in zip(ids, documents, metadatas):
            factor_id = str(factor_id)
            if factor_id in self._records:
                raise ValueError(f"ID 已存在: {factor_id}")
            self._records[factor_id] = {
                "document": document,
                "metadata": dict(metadata),
            }

    def get(
        self,
        *,
        ids: list[str] | str | None = None,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, list[Any]]:
        selected = self._selected(ids=ids, where=where)
        result: dict[str, list[Any]] = {
            "ids": [factor_id for factor_id, _ in selected],
            "metadatas": [dict(record["metadata"]) for _, record in selected],
        }
        if include and "documents" in include:
            result["documents"] = [record["document"] for _, record in selected]
        if include and "embeddings" in include:
            result["embeddings"] = [None for _ in selected]
        return result

    def update(
        self,
        *,
        ids: list[str],
        documents: list[str] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        for index, factor_id in enumerate(ids):
            factor_id = str(factor_id)
            if factor_id not in self._records:
                raise ValueError(f"ID 不存在: {factor_id}")
            if documents is not None:
                self._records[factor_id]["document"] = documents[index]
            if metadatas is not None:
                self._records[factor_id]["metadata"] = dict(metadatas[index])

    def delete(self, *, ids: list[str]) -> None:
        for factor_id in ids:
            self._records.pop(str(factor_id), None)

    def count(self) -> int:
        return len(self._records)

    @staticmethod
    def _similarity(text: str, document: str) -> float:
        query_words = set(text.lower().split())
        document_words = set(document.lower().split())
        if not query_words:
            return 0.0
        return len(query_words & document_words) / len(query_words)

    def query(
        self,
        *,
        query_texts: list[str],
        n_results: int,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, list[list[Any]]]:
        del include
        text = query_texts[0]
        ranked: list[tuple[float, str, dict[str, Any]]] = []
        for factor_id, record in self._selected(where=where):
            similarity = self._similarity(text, record["document"])
            distance = 2.0 * (1.0 - similarity)
            ranked.append((distance, factor_id, record["metadata"]))
        ranked.sort(key=lambda item: (item[0], item[1]))
        ranked = ranked[:n_results]
        return {
            "ids": [[item[1] for item in ranked]],
            "metadatas": [[dict(item[2]) for item in ranked]],
            "distances": [[item[0] for item in ranked]],
        }


class _FakePersistentClient:
    def __init__(self, path: str) -> None:
        self.path = path
        self.collection = _FakeCollection()

    def get_or_create_collection(self, **_: Any) -> _FakeCollection:
        return self.collection


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """使用确定性的内存 collection，避免测试依赖真实 embedding 模型。"""

    monkeypatch.setattr(storage_module, "HAS_CHROMADB", True)
    monkeypatch.setattr(
        storage_module,
        "chromadb",
        SimpleNamespace(PersistentClient=_FakePersistentClient),
        raising=False,
    )
    monkeypatch.setattr(
        storage_module,
        "embedding_functions",
        SimpleNamespace(DefaultEmbeddingFunction=_FakeEmbeddingFunction),
        raising=False,
    )
    return FactorVectorStore(str(tmp_path / "vector_db"))


def _metadata(
    *,
    name: str = "UV brightening",
    description: str = "Enhanced UV emission before solar flares",
    source: str = "arxiv:paper-1",
    confidence: float = 0.82,
    ftype: str = "precursor",
) -> dict[str, str | float]:
    return {
        "name": name,
        "description": description,
        "source": source,
        "confidence": confidence,
        "ftype": ftype,
    }


def _sample_metadata() -> list[dict[str, str | float]]:
    return [
        _metadata(
            name="UV brightening",
            description="Enhanced UV emission before solar flares",
            source="arxiv:paper-1",
            confidence=0.82,
            ftype="precursor",
        ),
        _metadata(
            name="Magnetic shear",
            description="Magnetic shear flows along the polarity inversion line",
            source="arxiv:paper-2",
            confidence=0.78,
            ftype="mechanism",
        ),
        _metadata(
            name="Delta spot",
            description="Delta spot configuration before major solar flares",
            source="arxiv:paper-3",
            confidence=0.91,
            ftype="condition",
        ),
    ]


@pytest.fixture
def seeded_store(store: FactorVectorStore) -> FactorVectorStore:
    for metadata in _sample_metadata():
        store.insert(metadata)
    return store


# ---------------------------------------------------------------------------
# insert
# ---------------------------------------------------------------------------


def test_insert_rejects_metadata_that_is_not_a_dict(store: FactorVectorStore):
    for metadata in (None, [], "metadata", 1):
        with pytest.raises(TypeError):
            store.insert(metadata)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("metadata", "error"),
    [
        ({"name": "not complete"}, TypeError),
        ({**_metadata(), "name": ""}, ValueError),
        ({**_metadata(), "description": 123}, TypeError),
        ({**_metadata(), "source": "   "}, ValueError),
        ({**_metadata(), "ftype": None}, TypeError),
        ({**_metadata(), "confidence": "0.5"}, MetadataError),
        ({**_metadata(), "confidence": 1.01}, MetadataError),
    ],
)
def test_insert_rejects_invalid_metadata_values(
    store: FactorVectorStore,
    metadata: dict[str, Any],
    error: type[Exception],
):
    with pytest.raises(error):
        store.insert(metadata)


def test_insert_rejects_duplicate_source(store: FactorVectorStore):
    store.insert(_metadata(source=" arxiv:duplicate "))

    duplicate = _metadata(
        name="A different factor",
        source="arxiv:duplicate",
    )
    with pytest.raises(DuplicateError):
        store.insert(duplicate)
    assert len(store.query()) == 1


def test_insert_accepts_valid_metadata_and_generates_a_factor_id(
    store: FactorVectorStore,
):
    metadata = _metadata(
        name="  UV brightening  ",
        source="  arxiv:valid  ",
    )

    returned_id = store.insert(metadata)
    assert returned_id == "F1"
    factors = store.query(where={"id": "F1"})

    assert len(factors) == 1
    assert isinstance(factors[0], LiteratureFactor)
    assert factors[0].id == "F1"
    assert factors[0].name == "UV brightening"
    assert factors[0].source == "arxiv:valid"


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factor_id", [1, None, object()])
def test_delete_non_string_id_is_treated_as_missing(
    store: FactorVectorStore,
    factor_id: object,
):
    assert store.delete(factor_id) is False  # type: ignore[arg-type]


def test_delete_empty_id_is_treated_as_missing(store: FactorVectorStore):
    assert store.delete("") is False


def test_delete_overlong_id_is_treated_as_missing(store: FactorVectorStore):
    assert store.delete("F" * 4096) is False


def test_delete_returns_false_when_database_does_not_contain_id(
    seeded_store: FactorVectorStore,
):
    assert seeded_store.delete("F999") is False
    assert seeded_store.query() and len(seeded_store.query()) == 3


def test_delete_removes_the_record_for_a_valid_id(
    seeded_store: FactorVectorStore,
):
    before = seeded_store.query(where={"id": "F2"})
    assert len(before) == 1

    assert seeded_store.delete("F2") is True
    assert seeded_store.query(where={"id": "F2"}) == []
    assert {factor.id for factor in seeded_store.query()} == {"F1", "F3"}
    assert seeded_store.delete("F2") is False


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factor_id", [1, None, "", "F" * 4096, "F999"])
def test_update_rejects_non_string_empty_overlong_or_missing_id(
    seeded_store: FactorVectorStore,
    factor_id: object,
):
    with pytest.raises(Exception, match="不存在此ID"):
        seeded_store.update(factor_id, {"confidence": 0.9})  # type: ignore[arg-type]


def test_update_rejects_updates_that_are_not_a_dict(
    seeded_store: FactorVectorStore,
):
    for updates in (None, [], "updates", 1):
        with pytest.raises(TypeError):
            seeded_store.update("F1", updates)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"name": ""}, ValueError),
        ({"description": None}, TypeError),
        ({"confidence": "high"}, MetadataError),
        ({"confidence": 2.0}, MetadataError),
        ({"ftype": ""}, ValueError),
    ],
)
def test_update_rejects_invalid_values(
    seeded_store: FactorVectorStore,
    updates: dict[str, Any],
    error: type[Exception],
):
    with pytest.raises(error):
        seeded_store.update("F1", updates)


def test_update_rejects_attempt_to_change_id(seeded_store: FactorVectorStore):
    with pytest.raises(ValueError, match="不支持修改字段"):
        seeded_store.update("F1", {"id": "F999"})


def test_update_rejects_source_that_would_be_duplicate(
    seeded_store: FactorVectorStore,
):
    with pytest.raises(DuplicateError):
        seeded_store.update("F1", {"source": "arxiv:paper-2"})

    unchanged = seeded_store.query(where={"id": "F1"})[0]
    assert unchanged.source == "arxiv:paper-1"


def test_update_accepts_valid_updates_and_preserves_id(
    seeded_store: FactorVectorStore,
):
    updated = seeded_store.update(
        "F1",
        {
            "name": "UV footpoint brightening",
            "description": "Updated UV emission description",
            "confidence": 0.95,
            "ftype": "mechanism",
            "source": "arxiv:paper-1-updated",
        },
    )

    assert isinstance(updated, LiteratureFactor)
    assert updated.id == "F1"
    assert updated.name == "UV footpoint brightening"
    assert updated.description == "Updated UV emission description"
    assert updated.confidence == 0.95
    assert updated.ftype == "mechanism"
    assert updated.source == "arxiv:paper-1-updated"
    assert seeded_store.query(where={"id": "F1"})[0] == updated


def test_update_accepts_a_valid_id_and_empty_updates_are_a_noop(
    seeded_store: FactorVectorStore,
):
    before = seeded_store.query(where={"id": "F1"})[0]
    after = seeded_store.update("F1", {})
    assert after == before


def test_update_with_empty_id_is_rejected(seeded_store: FactorVectorStore):
    with pytest.raises(Exception, match="不存在此ID"):
        seeded_store.update("", {"confidence": 0.5})


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


def test_query_rejects_where_that_is_not_a_dict(seeded_store: FactorVectorStore):
    for where in ("where", [], 1):
        with pytest.raises(TypeError):
            seeded_store.query(where=where)  # type: ignore[arg-type]


def test_query_rejects_unexpected_arguments(seeded_store: FactorVectorStore):
    with pytest.raises(TypeError):
        seeded_store.query({"source": "arxiv:paper-1"})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        seeded_store.query(where={"source": "arxiv:paper-1"}, limit=1)  # type: ignore[call-arg]


def test_query_rejects_invalid_where_values(seeded_store: FactorVectorStore):
    with pytest.raises(ValueError):
        seeded_store.query(where={"unknown": "value"})
    with pytest.raises(ValueError):
        seeded_store.query(where={"confidence": {"$gte": "high"}})
    with pytest.raises(ValueError):
        seeded_store.query(where={"$and": {"source": "arxiv:paper-1"}})


def test_query_accepts_normal_metadata_and_id_filters(
    seeded_store: FactorVectorStore,
):
    by_source = seeded_store.query(where={"source": "arxiv:paper-1"})
    by_id = seeded_store.query(where={"id": "F2"})
    by_condition = seeded_store.query(
        where={
            "$and": [
                {"ftype": "precursor"},
                {"confidence": {"$gte": 0.8}},
            ]
        }
    )

    assert [factor.id for factor in by_source] == ["F1"]
    assert [factor.id for factor in by_id] == ["F2"]
    assert [factor.id for factor in by_condition] == ["F1"]


def test_query_without_where_returns_all_records(seeded_store: FactorVectorStore):
    assert len(seeded_store.query()) == 3
    assert len(seeded_store.query(where=None)) == 3
    assert len(seeded_store.query(where={})) == 3


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("where", ["where", [], 1])
def test_search_rejects_where_that_is_not_a_dict(
    seeded_store: FactorVectorStore,
    where: object,
):
    with pytest.raises(TypeError):
        seeded_store.search("magnetic shear", where=where)  # type: ignore[arg-type]


def test_search_rejects_unexpected_arguments(seeded_store: FactorVectorStore):
    with pytest.raises(TypeError):
        seeded_store.search("magnetic shear", {"ftype": "mechanism"})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        seeded_store.search(
            "magnetic shear",
            where={"ftype": "mechanism"},
            unknown=True,
        )  # type: ignore[call-arg]


def test_search_rejects_invalid_where_values(seeded_store: FactorVectorStore):
    with pytest.raises(ValueError):
        seeded_store.search("magnetic shear", where={"unknown": "value"})
    with pytest.raises(ValueError):
        seeded_store.search(
            "magnetic shear",
            where={"confidence": {"$gte": "high"}},
        )
    with pytest.raises(ValueError):
        seeded_store.search(
            "magnetic shear",
            where={"$and": {"ftype": "mechanism"}},
        )


def test_search_rejects_empty_or_non_string_text(seeded_store: FactorVectorStore):
    for text in ("", "   ", None, 123):
        with pytest.raises(ValueError, match="不能为空"):
            seeded_store.search(text)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"limit": 0}, ValueError),
        ({"limit": -1}, ValueError),
        ({"min_similarity": -0.1}, ValueError),
        ({"min_similarity": 1.1}, ValueError),
    ],
)
def test_search_rejects_invalid_search_parameters(
    seeded_store: FactorVectorStore,
    kwargs: dict[str, Any],
    error: type[Exception],
):
    with pytest.raises(error):
        seeded_store.search("magnetic shear", **kwargs)


def test_search_accepts_normal_text_where_limit_and_similarity(
    seeded_store: FactorVectorStore,
):
    results = seeded_store.search(
        "magnetic shear",
        where={"ftype": "mechanism"},
        limit=2,
        min_similarity=0.5,
    )

    assert len(results) == 1
    assert results[0].id == "F2"
    assert isinstance(results[0], LiteratureFactor)


def test_search_with_empty_where_returns_ranked_results(
    seeded_store: FactorVectorStore,
):
    without_where = seeded_store.search("solar flares", limit=3)
    with_none = seeded_store.search("solar flares", where=None, limit=3)
    with_empty_mapping = seeded_store.search("solar flares", where={}, limit=3)

    assert without_where
    assert [factor.id for factor in without_where] == [factor.id for factor in with_none]
    assert [factor.id for factor in without_where] == [factor.id for factor in with_empty_mapping]


# ---------------------------------------------------------------------------
# end-to-end flow
# ---------------------------------------------------------------------------


def test_insert_query_search_update_delete_flow(store: FactorVectorStore):
    for metadata in _sample_metadata():
        store.insert(metadata)

    assert len(store.query()) == 3

    exact = store.query(where={"id": "F2"})
    assert len(exact) == 1
    assert exact[0].source == "arxiv:paper-2"

    fuzzy = store.search(
        "magnetic shear",
        where={"ftype": "mechanism"},
        limit=1,
        min_similarity=0.9,
    )
    assert [factor.id for factor in fuzzy] == ["F2"]

    updated = store.update(
        exact[0].id,
        {
            "description": "Updated magnetic shear description",
            "confidence": 0.88,
        },
    )
    assert updated.id == "F2"
    assert updated.description == "Updated magnetic shear description"
    assert updated.confidence == 0.88

    assert store.delete(updated.id) is True
    assert store.query(where={"id": "F2"}) == []
    assert {factor.id for factor in store.query()} == {"F1", "F3"}
