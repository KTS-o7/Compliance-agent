import json
import pytest
from unittest.mock import MagicMock
from app.evaluation.evaluator import Evaluator
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


@pytest.fixture
def evaluator(mocker):
    mocker.patch("app.evaluation.evaluator.get_client")
    mocker.patch("app.evaluation.evaluator.get_models",
                 return_value={"trigger": "test", "evaluator": "test"})
    e = Evaluator()
    e.client = MagicMock()
    return e


def _mock_response(content: str):
    msg = MagicMock(); msg.content = content
    choice = MagicMock(); choice.message = msg
    resp = MagicMock(); resp.choices = [choice]
    return resp


def test_returns_verdicts(evaluator):
    payload = json.dumps([{
        "rule_id": "FDCPA-809", "verdict": "FAIL",
        "reasoning": "Agent did not send validation notice.",
        "citation": "Turn 3", "severity": "critical"
    }])
    evaluator.client.chat.completions.create.return_value = _mock_response(payload)
    verdicts = evaluator.evaluate(TRANSCRIPT, [RULE])
    assert len(verdicts) == 1
    assert verdicts[0].verdict == "FAIL"
    assert verdicts[0].rule_id == "FDCPA-809"


def test_empty_rules_returns_empty(evaluator):
    verdicts = evaluator.evaluate(TRANSCRIPT, [])
    assert verdicts == []


def test_bad_json_returns_empty(evaluator):
    evaluator.client.chat.completions.create.return_value = _mock_response(
        "I cannot evaluate this transcript."
    )
    verdicts = evaluator.evaluate(TRANSCRIPT, [RULE])
    assert verdicts == []


def test_markdown_fenced_response(evaluator):
    payload = '```json\n[{"rule_id":"FDCPA-809","verdict":"PASS","reasoning":"ok","citation":"Turn 1","severity":"critical"}]\n```'
    evaluator.client.chat.completions.create.return_value = _mock_response(payload)
    verdicts = evaluator.evaluate(TRANSCRIPT, [RULE])
    assert verdicts[0].verdict == "PASS"
