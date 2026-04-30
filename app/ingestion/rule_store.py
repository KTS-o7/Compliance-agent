from __future__ import annotations
import uuid
import os
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchAny
)
from sentence_transformers import SentenceTransformer
from app.models.schemas import Rule

COLLECTION = "compliance_rules"
VECTOR_SIZE = 768   # BAAI/bge-base-en-v1.5
_EMBEDDER: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("BAAI/bge-base-en-v1.5")
    return _EMBEDDER


def _rule_id_to_uuid(rule_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, rule_id))


class RuleStore:
    def __init__(self, qdrant_url: str | None = None):
        if qdrant_url is None:
            cfg_path = os.environ.get("CONFIG_PATH", "config.yaml")
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            qdrant_url = os.environ.get("QDRANT_URL", cfg["qdrant"]["url"])
        self.client = QdrantClient(url=qdrant_url, timeout=10)
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION not in existing:
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    def add_rule(self, rule: Rule) -> None:
        vector = _get_embedder().encode(rule.text).tolist()
        point = PointStruct(
            id=_rule_id_to_uuid(rule.id),
            vector=vector,
            payload=rule.model_dump(),
        )
        self.client.upsert(collection_name=COLLECTION, points=[point])

    def rule_exists(self, rule_id: str) -> bool:
        results = self.client.retrieve(
            collection_name=COLLECTION,
            ids=[_rule_id_to_uuid(rule_id)],
        )
        return len(results) > 0

    def semantic_search(self, query: str, top_k: int = 5, role: str = "senior") -> list[Rule]:
        role_tags = ["all"] if role == "junior" else ["junior", "senior", "all"]
        prefixed_query = f"Represent this sentence for searching relevant passages: {query}"
        vector = _get_embedder().encode(prefixed_query).tolist()
        results = self.client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=top_k,
            query_filter=Filter(
                must=[FieldCondition(key="role_tag", match=MatchAny(any=role_tags))]
            ),
        ).points
        return [Rule(**r.payload) for r in results]

    def get_by_ids(self, rule_ids: list[str], role: str = "senior") -> list[Rule]:
        if not rule_ids:
            return []
        role_tags = ["all"] if role == "junior" else ["junior", "senior", "all"]
        points = self.client.retrieve(
            collection_name=COLLECTION,
            ids=[_rule_id_to_uuid(rid) for rid in rule_ids],
            with_payload=True,
        )
        return [Rule(**p.payload) for p in points if p.payload.get("role_tag") in role_tags]

    def count(self) -> int:
        return self.client.count(collection_name=COLLECTION).count
