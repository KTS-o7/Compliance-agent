from app.retrieval.deterministic import DeterministicRetriever

def test_known_trigger():
    r = DeterministicRetriever()
    assert "FDCPA-809" in r.get_rule_ids(["Debt Dispute"])

def test_unknown_trigger():
    r = DeterministicRetriever()
    assert r.get_rule_ids(["Unknown"]) == []

def test_multiple_triggers_union():
    r = DeterministicRetriever()
    ids = r.get_rule_ids(["Debt Dispute", "Bankruptcy Notification"])
    assert "FDCPA-809" in ids
    assert "BANKRUPTCY-362" in ids

def test_no_duplicates():
    r = DeterministicRetriever()
    ids = r.get_rule_ids(["Bankruptcy Notification", "Cease and Desist Request"])
    assert ids.count("FDCPA-805") == 1
