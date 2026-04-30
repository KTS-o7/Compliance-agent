import pytest
from unittest.mock import MagicMock
from app.detection.trigger_detector import TriggerDetector

TRANSCRIPT = (
    "Agent: I see you're disputing this debt.\n"
    "Customer: Yes, I don't believe I owe this.\n"
    "Agent: I understand, we'll need to validate."
)


@pytest.fixture
def detector(mocker):
    mocker.patch("app.detection.trigger_detector.get_client")
    mocker.patch("app.detection.trigger_detector.get_models",
                 return_value={"trigger": "test-model", "evaluator": "test-model"})
    d = TriggerDetector()
    d.client = MagicMock()
    return d


def _mock_response(content: str):
    msg = MagicMock(); msg.content = content
    choice = MagicMock(); choice.message = msg
    resp = MagicMock(); resp.choices = [choice]
    return resp


def test_detects_valid_labels(detector):
    detector.client.chat.completions.create.return_value = _mock_response(
        '["Debt Dispute", "Right to Validation Request"]'
    )
    labels = detector.detect(TRANSCRIPT)
    assert "Debt Dispute" in labels
    assert "Right to Validation Request" in labels


def test_filters_invalid_labels(detector):
    detector.client.chat.completions.create.return_value = _mock_response(
        '["Debt Dispute", "Made Up Label"]'
    )
    labels = detector.detect(TRANSCRIPT)
    assert "Made Up Label" not in labels
    assert "Debt Dispute" in labels


def test_bad_json_returns_empty(detector):
    detector.client.chat.completions.create.return_value = _mock_response(
        "I cannot determine triggers from this transcript."
    )
    labels = detector.detect(TRANSCRIPT)
    assert labels == []


def test_markdown_fenced_json_parsed(detector):
    detector.client.chat.completions.create.return_value = _mock_response(
        '```json\n["Debt Dispute"]\n```'
    )
    labels = detector.detect(TRANSCRIPT)
    assert "Debt Dispute" in labels
