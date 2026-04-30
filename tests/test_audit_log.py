import os
import pytest
from app.audit.audit_log import AuditLog
from app.models.schemas import AuditEntry, Verdict


def make_entry(tid="t001"):
    return AuditEntry(
        transcript_id=tid,
        rules_evaluated=["FDCPA-809"],
        verdicts=[Verdict(rule_id="FDCPA-809", verdict="PASS",
                          reasoning="ok", citation="Turn 1", severity="critical")],
        model_used="test-model",
        latency_seconds=1.2,
        timestamp="2026-04-30T10:00:00",
        user_role="senior",
    )


@pytest.fixture
def log(tmp_path):
    return AuditLog(db_path=str(tmp_path / "audit.db"))


def test_append_and_read(log):
    log.append(make_entry())
    entries = log.read_all()
    assert len(entries) == 1
    assert entries[0]["transcript_id"] == "t001"


def test_append_only_no_overwrite(log):
    log.append(make_entry("t001"))
    log.append(make_entry("t002"))
    assert log.count() == 2


def test_disagreement_stored(log):
    e = make_entry()
    e.corrective_disagreement = True
    log.append(e)
    entries = log.read_all()
    assert entries[0]["corrective_disagreement"] == 1
