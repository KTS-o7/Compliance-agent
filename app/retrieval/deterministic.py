TRIGGER_RULE_MAP: dict[str, list[str]] = {
    "Debt Dispute":                  ["FDCPA-809", "FDCPA-807"],
    "Financial Hardship":            ["CFPB-HARDSHIP-01", "FDCPA-806"],
    "Bankruptcy Notification":       ["FDCPA-805", "BANKRUPTCY-362"],
    "Payment Plan Request":          ["CFPB-HARDSHIP-01", "FDCPA-808"],
    "Cease and Desist Request":      ["FDCPA-805", "FDCPA-806"],
    "Account Closure Request":       ["FCRA-623"],
    "Fraud Claim":                   ["FCRA-605", "FCRA-623", "FDCPA-807"],
    "Identity Verification Failure": ["FCRA-605"],
    "Complaint Escalation":          ["CFPB-COMPLAINT-01"],
    "Right to Validation Request":   ["FDCPA-809"],
}


class DeterministicRetriever:
    def get_rule_ids(self, triggers: list[str]) -> list[str]:
        ids: set[str] = set()
        for trigger in triggers:
            ids.update(TRIGGER_RULE_MAP.get(trigger, []))
        return list(ids)
