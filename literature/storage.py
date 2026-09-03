"""
面向文献因素 ChromaDB 存储封装。

本文件**不考虑**项目的规则，只考虑如何为下游提供数据库的封装，限制条件由下游自行管理

``query`` 用于按 ID 或 metadata 精确查询；``search`` 用于自然语言语义检索。
下游模块只需使用 ``insert``、``delete``、``update``、``query`` 和 ``search``。
"""
from __future__ import annotations

import json
import os
from numbers import Real
from typing import Any

from shared.contracts import LiteratureFactor
from .LiteratureError import MetadataError, DuplicateError

try:
    import chromadb
    from chromadb.utils import embedding_functions
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


_ALLOWED_UPDATE_FIELDS = {
    "name",
    "description",
    "source",
    "confidence",
    "ftype",
}
_STRING_FIELDS = ("name", "description", "source", "ftype")

class FactorVectorStore:
    """将 LiteratureFactor 映射到 ChromaDB 的业务存储类。"""

    def __init__(self, persist_path: str):
        if not HAS_CHROMADB:
            raise ImportError("未检测到 ChromaDB")

        os.makedirs(persist_path, exist_ok=True)
        self._config_path = os.path.join(persist_path, "config")
        if os.path.exists(self._config_path):
            with open(self._config_path, "r", encoding="utf-8") as config_file:
                self.literature_counter = json.load(config_file).get("id_counter", 0)
        else:
            self.literature_counter = 0
            with open(self._config_path, "w", encoding="utf-8") as config_file:
                json.dump({"id_counter": self.literature_counter}, config_file)

        self._client = chromadb.PersistentClient(path=persist_path)
        self._embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name='factors',
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def insert(self, metadata: dict[str, str | float]) -> str:
        """插入单个元素，并返回自动分配的因素 ID。"""

        self._validateMetadata(metadata)
        self._validateDuplicate(metadata["source"])

        target = self._metadata2Factor(metadata)
        target_metadata = self._factor2Dict(target)
        self._collection.add(
            ids=[target.id],
            documents=[target.description],
            metadatas=[target_metadata],
        )
        with open(self._config_path, "w", encoding="utf-8") as config_file:
            json.dump({"id_counter": self.literature_counter}, config_file)
        return target.id

    def delete(self, id: str) -> bool:
        """根据文献 ID 删除因素；不存在时返回 ``False``。"""

        factor_id = str(id)
        raw = self._collection.get(
            ids=[factor_id],
            include=["metadatas"],
        )
        ids = raw.get("ids") if raw else None
        if not ids:
            return False

        self._collection.delete(ids=[factor_id])
        return True

    def update(self, id: str, updates: dict[str, object]) -> LiteratureFactor:
        """按 ID 原位更新一条因素，ID 本身不可修改。"""

        if not isinstance(updates, dict):
            raise TypeError("updates 必须是字典")

        unsupported = sorted(set(updates) - _ALLOWED_UPDATE_FIELDS)
        if unsupported:
            raise ValueError(f"不支持修改字段: {unsupported}")

        factor_id = str(id)
        raw = self._collection.get(
            ids=[factor_id],
            include=["metadatas"],
        )
        ids = raw.get("ids") or []
        metadatas = raw.get("metadatas") or []
        if not ids or not metadatas or metadatas[0] is None:
            raise Exception("不存在此ID")

        metadata = metadatas[0]
        values = {
            "id": factor_id,
            "name": metadata.get("name", ""),
            "description": metadata.get("description", ""),
            "source": metadata.get("source", ""),
            "confidence": metadata.get("confidence", 0.0),
            "ftype": metadata.get("ftype", "precursor"),
        }
        values.update(updates)
        self._validateMetadata(values)
        updated = self._dict2Factor(values)
        self._validateDuplicate(updated.source, exclude_id=factor_id)

        updated_metadata = self._factor2Dict(updated)
        if "description" in updates:
            self._collection.update(
                ids=[factor_id],
                documents=[updated.description],
                metadatas=[updated_metadata],
            )
        else:
            self._collection.update(
                ids=[factor_id],
                metadatas=[updated_metadata],
            )
        return updated

    # 精确查询
    def query(self, *, where: dict[str, Any] | None = None) -> list[LiteratureFactor]:
        """精确查询因素。

        按照精确值进行查询，暂时不支持除了“等值”与“且”以外的混合条件。

        - ``where`` 使用 ChromaDB metadata filter 语法，可表达等值、比较和逻辑条件。
        """

        if where is not None and not isinstance(where, dict):
            raise TypeError("where 必须是字典")

        raw = self._collection.get(where=where) if where is not None else self._collection.get()
        ids = raw.get("ids") or []
        metadatas = raw.get("metadatas") or []
        factors: list[LiteratureFactor] = []
        for index, factor_id in enumerate(ids):
            metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            metadata["id"] = str(factor_id)
            factors.append(self._dict2Factor(metadata))
        return factors

    # 模糊查询（语义查询）
    def search(
        self,
        text: str,
        *,
        where: dict[str, Any] | None = None,
        limit: int = 5,
        min_similarity: float = 0.0,
    ) -> list[LiteratureFactor]:
        """按自然语言语义检索因素，并可使用 metadata filter 过滤。"""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("检索文本不能为空")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit 必须是正整数")
        if (
            isinstance(min_similarity, bool) or
            not isinstance(min_similarity, Real) or
            not 0.0 <= float(min_similarity) <= 1.0
        ):
            raise ValueError("min_similarity 必须位于 0~1")
        if where is not None and not isinstance(where, dict):
            raise TypeError("where 必须是字典")

        record_count = self._collection.count()
        if record_count == 0:
            return []

        kwargs: dict[str, Any] = {
            "query_texts": [text.strip()],
            "n_results": min(limit, record_count),
            "include": ["metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)
        ids = (raw.get("ids") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results: list[LiteratureFactor] = []
        for index, factor_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else 0.0
            # ChromaDB 的 cosine distance 为 0~2，转换为 0~1 的相似度。
            similarity = 1.0 - distance / 2.0
            if similarity < float(min_similarity):
                continue
            metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            metadata["id"] = str(factor_id)
            results.append(self._dict2Factor(metadata))
        return results


    # region 数据验证
    @staticmethod
    def _validateMetadata(metadata: dict[str, str | float]) -> None:
        """验证 metadata 合法性。"""

        if not isinstance(metadata, dict):
            raise TypeError("metadata 类型错误")

        # 除了 confidence 字段外，各字段必须均为非空字符串。
        for field_name in _STRING_FIELDS:
            value = metadata.get(field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} 必须是字符串")
            if not value.strip():
                raise ValueError(f"{field_name} 不能为空")

        # confidence 字段必须为 0~1 的实数。
        confidence = metadata.get("confidence")
        if not isinstance(confidence, Real):
            raise MetadataError("confidence 必须是数值")
        if not 0.0 <= float(confidence) <= 1.0:
            raise MetadataError("confidence 必须位于 0~1")

    def _validateDuplicate(self, source: str, exclude_id: str | None = None) -> None:
        """检验 source 是否重复，可排除指定 ID。"""

        raw = self._collection.get(
            where={"source": source.strip()},
            include=["metadatas"],
        )
        ids = raw.get("ids") or []
        if exclude_id is None or any(str(factor_id) != exclude_id for factor_id in ids):
            if ids:
                raise DuplicateError("重复文献")
    # endregion

    # region 数据转换
    @staticmethod
    def _factor2Dict(factor: LiteratureFactor) -> dict[str, (str|float)]:
        """将LiteratureFactor转换为格式化的字典"""

        return {
            "id": factor.id,
            "name": factor.name.strip(),
            "description": factor.description,
            "source": factor.source.strip(),
            "confidence": float(factor.confidence),
            "ftype": factor.ftype,
        }

    @staticmethod
    def _dict2Factor(dictdata: dict[str, (str|float)]) -> LiteratureFactor:
        """将字典转换为格式化的LiteratureFactor"""

        return LiteratureFactor(
            id=dictdata.get("id", "ERROR"),
            name=str(dictdata.get("name", "")).strip(),
            description=str(dictdata.get("description", "")),
            source=str(dictdata.get("source", "")).strip(),
            confidence=float(dictdata.get("confidence", 0.0)),
            ftype=str(dictdata.get("ftype", "precursor")),
        )
    
    def _metadata2Factor(self, metadata: dict[str, (str|float)]) -> LiteratureFactor:
        """将元数据转换为格式化的LiteratureFactor"""

        self.literature_counter += 1
        return LiteratureFactor(
            id=f"F{self.literature_counter}",
            name=str(metadata.get("name", "")).strip(),
            description=str(metadata.get("description", "")),
            source=str(metadata.get("source", "")).strip(),
            confidence=float(metadata.get("confidence", 0.0)),
            ftype=str(metadata.get("ftype", "precursor")),
        )
    # endregion
