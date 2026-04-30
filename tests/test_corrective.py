import pytest
from unittest.mock import MagicMock
from app.retrieval.corrective import CorrectiveRetriever
from app.models.schemas import Rule


def make_rule(rule_id: str, role_tag: str = "all") -> Rule:
    return Rule(id=rule_id, citation="x", severity="high",
                agent_must=[], agent_must_not=[],
                role_tag=role_tag, trigger_labels=[], text="x")


@pytest.fixture
def store():
    return MagicMock()


def test_no_disagreement_when_sets_match(store):
    store.semantic_search.return_value = [make_rule("A"), make_rule("B")]
    store.get_by_ids.return_value = [make_rule("A"), make_rule("B")]
    cr = CorrectiveRetriever(rule_store=store)
    cr.det.get_rule_ids = lambda triggers: ["A", "B"]
    rules, disagreed = cr.retrieve(["Debt Dispute"], "text", "senior")
    assert not disagreed


def test_disagreement_flagged_when_sets_differ(store):
    store.semantic_search.return_value = [make_rule("C"), make_rule("D")]
    store.get_by_ids.return_value = [make_rule("A"), make_rule("B")]
    cr = CorrectiveRetriever(rule_store=store)
    cr.det.get_rule_ids = lambda triggers: ["A", "B"]
    rules, disagreed = cr.retrieve(["Debt Dispute"], "text", "senior")
    assert disagreed


def test_union_includes_both_paths(store):
    store.semantic_search.return_value = [make_rule("C")]
    store.get_by_ids.return_value = [make_rule("A")]
    cr = CorrectiveRetriever(rule_store=store)
    cr.det.get_rule_ids = lambda triggers: ["A"]
    rules, _ = cr.retrieve(["Debt Dispute"], "text", "senior")
    ids = {r.id for r in rules}
    assert "A" in ids
    assert "C" in ids


def test_empty_triggers_returns_semantic_only(store):
    store.semantic_search.return_value = [make_rule("X")]
    store.get_by_ids.return_value = []
    cr = CorrectiveRetriever(rule_store=store)
    cr.det.get_rule_ids = lambda triggers: []
    rules, _ = cr.retrieve([], "text", "senior")
    assert any(r.id == "X" for r in rules)
