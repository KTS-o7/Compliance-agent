import pytest
from unittest.mock import MagicMock
from app.models.schemas import Rule
from app.ingestion.rule_store import RuleStore, _rule_id_to_uuid

SAMPLE_RULE = Rule(
    id="FDCPA-809", citation="15 U.S.C. § 1692g", severity="critical",
    agent_must=["send notice"], agent_must_not=["ignore dispute"],
    role_tag="all", trigger_labels=["Debt Dispute"],
    text="Must send validation notice within 5 days."
)


@pytest.fixture
def store(mocker):
    mocker.patch("app.ingestion.rule_store.QdrantClient")
    mocker.patch("app.ingestion.rule_store._get_embedder")
    s = RuleStore(qdrant_url="http://localhost:6333")
    s.client = MagicMock()
    s.client.get_collections.return_value = MagicMock(collections=[])
    return s


def test_add_rule_calls_upsert(store, mocker):
    mock_vector = MagicMock()
    mock_vector.tolist.return_value = [0.1] * 768
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = mock_vector
    mocker.patch("app.ingestion.rule_store._get_embedder", return_value=mock_embedder)
    store.add_rule(SAMPLE_RULE)
    assert store.client.upsert.called


def test_rule_id_to_uuid_deterministic():
    a = _rule_id_to_uuid("FDCPA-809")
    b = _rule_id_to_uuid("FDCPA-809")
    assert a == b


def test_get_by_ids_empty_returns_empty(store):
    result = store.get_by_ids([])
    assert result == []


def test_semantic_search_calls_query_points(store, mocker):
    mock_vector = MagicMock()
    mock_vector.tolist.return_value = [0.1] * 768
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = mock_vector
    mocker.patch("app.ingestion.rule_store._get_embedder", return_value=mock_embedder)
    mock_result = MagicMock()
    mock_result.points = []
    store.client.query_points.return_value = mock_result
    results = store.semantic_search("debt dispute", top_k=3, role="senior")
    assert store.client.query_points.called
    assert results == []


def test_seed_rules_count():
    from app.ingestion.seed_rules import SEED_RULES
    assert len(SEED_RULES) == 10
    ids = [r.id for r in SEED_RULES]
    assert "FDCPA-809" in ids
    assert "BANKRUPTCY-362" in ids
