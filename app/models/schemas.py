from __future__ import annotations
from pydantic import BaseModel
from typing import Literal


class Rule(BaseModel):
    id: str
    citation: str
    severity: Literal["critical", "high", "medium", "low"]
    agent_must: list[str]
    agent_must_not: list[str]
    role_tag: Literal["junior", "senior", "all"]
    trigger_labels: list[str]
    text: str


class Verdict(BaseModel):
    rule_id: str
    verdict: Literal["PASS", "FAIL"]
    reasoning: str
    citation: str
    severity: str


class RegulatoryCard(BaseModel):
    transcript_id: str
    verdicts: list[Verdict]
    evaluated_at: str
    model_used: str = ""
    latency_seconds: float = 0.0
    corrective_disagreement: bool = False

    @property
    def pass_count(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == "PASS")

    @property
    def fail_count(self) -> int:
        return sum(1 for v in self.verdicts if v.verdict == "FAIL")


class AuditEntry(BaseModel):
    transcript_id: str
    rules_evaluated: list[str]
    verdicts: list[Verdict]
    model_used: str
    latency_seconds: float
    timestamp: str
    user_role: str
    corrective_disagreement: bool = False
