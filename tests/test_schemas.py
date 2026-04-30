from app.models.schemas import Rule, Verdict, RegulatoryCard, AuditEntry

def test_rule_creation():
    r = Rule(id="FDCPA-809", citation="15 U.S.C. § 1692g", severity="critical",
             agent_must=["send notice"], agent_must_not=["ignore dispute"],
             role_tag="all", trigger_labels=["Debt Dispute"], text="Must send notice.")
    assert r.id == "FDCPA-809"
    assert r.severity == "critical"

def test_verdict():
    v = Verdict(rule_id="FDCPA-809", verdict="PASS",
                reasoning="Agent complied.", citation="Turn 3", severity="critical")
    assert v.verdict == "PASS"

def test_regulatory_card_counts():
    verdicts = [
        Verdict(rule_id="A", verdict="PASS", reasoning="ok", citation="T1", severity="high"),
        Verdict(rule_id="B", verdict="FAIL", reasoning="missed", citation="T2", severity="critical"),
    ]
    card = RegulatoryCard(transcript_id="t001", verdicts=verdicts,
                          evaluated_at="2026-04-30T10:00:00")
    assert card.pass_count == 1
    assert card.fail_count == 1

def test_audit_entry():
    v = Verdict(rule_id="FDCPA-809", verdict="PASS",
                reasoning="ok", citation="T1", severity="critical")
    e = AuditEntry(transcript_id="t001", rules_evaluated=["FDCPA-809"],
                   verdicts=[v], model_used="sonnet-4-6",
                   latency_seconds=1.5, timestamp="2026-04-30T10:00:00",
                   user_role="senior")
    assert e.transcript_id == "t001"
