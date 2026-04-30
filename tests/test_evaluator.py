import pytest
from unittest.mock import MagicMock
from app.evaluation.evaluator import Evaluator, EvaluationResponse
from app.models.schemas import Rule, Verdict

RULE = Rule(
    id="FDCPA-809", citation="15 U.S.C. § 1692g", severity="critical",
    agent_must=["send validation notice"],
    agent_must_not=["ignore dispute"],
    role_tag="all", trigger_labels=["Debt Dispute"],
    text="Must send validation notice within 5 days."
)

TRANSCRIPT = (
    "Turn 1 — Agent: I see you're disputing this debt.\n"
    "Turn 2 — Customer: Yes I dispute it.\n"
    "Turn 3 — Agent: We'll look into it."
)

SAMPLE_VERDICT = Verdict(
    rule_id="FDCPA-809", verdict="FAIL",
    reasoning="Agent did not send validation notice.",
    citation="Turn 3", severity="critical"
)


@pytest.fixture
def evaluator(mocker):
    mocker.patch("app.evaluation.evaluator.get_client")
    mocker.patch("app.evaluation.evaluator.get_models",
                 return_value={"trigger": "test", "evaluator": "test"})
    e = Evaluator()
    e.client = MagicMock()
    return e


def _mock_parsed_response(verdicts: list[Verdict]):
    """Mock client.beta.chat.completions.parse response."""
    parsed = EvaluationResponse(verdicts=verdicts)
    msg = MagicMock()
    msg.parsed = parsed
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_returns_verdicts(evaluator):
    evaluator.client.beta.chat.completions.parse.return_value = _mock_parsed_response([SAMPLE_VERDICT])
    verdicts = evaluator.evaluate(TRANSCRIPT, [RULE])
    assert len(verdicts) == 1
    assert verdicts[0].verdict == "FAIL"
    assert verdicts[0].rule_id == "FDCPA-809"


def test_empty_rules_returns_empty(evaluator):
    verdicts = evaluator.evaluate(TRANSCRIPT, [])
    assert verdicts == []


def test_bad_parse_returns_empty(evaluator):
    evaluator.client.beta.chat.completions.parse.side_effect = Exception("parse error")
    verdicts = evaluator.evaluate(TRANSCRIPT, [RULE])
    assert verdicts == []


def test_none_parsed_returns_empty(evaluator):
    msg = MagicMock(); msg.parsed = None
    choice = MagicMock(); choice.message = msg
    resp = MagicMock(); resp.choices = [choice]
    evaluator.client.beta.chat.completions.parse.return_value = resp
    verdicts = evaluator.evaluate(TRANSCRIPT, [RULE])
    assert verdicts == []
