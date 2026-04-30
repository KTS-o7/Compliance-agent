from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from app.models.schemas import Rule
from app.retrieval.deterministic import DeterministicRetriever
from app.retrieval.semantic import SemanticRetriever

if TYPE_CHECKING:
    from app.ingestion.rule_store import RuleStore

logger = logging.getLogger(__name__)


class CorrectiveRetriever:
    def __init__(
        self,
        rule_store: RuleStore,
        threshold: float = 0.5,
        top_k: int = 5,
    ):
        self.det = DeterministicRetriever()
        self.sem = SemanticRetriever(rule_store=rule_store, top_k=top_k)
        self.store = rule_store
        self.threshold = threshold

    def retrieve(
        self, triggers: list[str], transcript: str, role: str = "senior"
    ) -> tuple[list[Rule], bool]:
        det_ids = set(self.det.get_rule_ids(triggers))
        sem_rules = self.sem.search(transcript, role=role)
        sem_ids = {r.id for r in sem_rules}

        overlap = det_ids & sem_ids

        # Disagreement = deterministic rules that semantic completely missed.
        # Semantic returning *extra* rules is not disagreement — it's enrichment.
        # Only flag when semantic misses a significant fraction of det_ids.
        if det_ids:
            missed_ratio = len(det_ids - sem_ids) / len(det_ids)
        else:
            missed_ratio = 0.0
        disagreed = missed_ratio > self.threshold

        if disagreed:
            logger.warning(
                "Corrective RAG disagreement | det=%s | sem_missed=%s | missed_ratio=%.2f",
                sorted(det_ids), sorted(det_ids - sem_ids), missed_ratio
            )

        det_rules = self.store.get_by_ids(list(det_ids), role=role)
        all_rules: dict[str, Rule] = {r.id: r for r in det_rules}
        for r in sem_rules:
            all_rules.setdefault(r.id, r)

        return list(all_rules.values()), disagreed
