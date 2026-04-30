from __future__ import annotations
from typing import TYPE_CHECKING
from app.models.schemas import Rule

if TYPE_CHECKING:
    from app.ingestion.rule_store import RuleStore


class SemanticRetriever:
    def __init__(self, rule_store: RuleStore, top_k: int = 5):
        self.store = rule_store
        self.top_k = top_k

    def search(self, query: str, role: str = "senior") -> list[Rule]:
        return self.store.semantic_search(query, top_k=self.top_k, role=role)
