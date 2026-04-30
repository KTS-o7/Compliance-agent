from __future__ import annotations
import json
from app.models.schemas import Rule, Verdict
from app.llm_client import get_client, get_models

SYSTEM_PROMPT = """\
You are a strict compliance auditor evaluating a debt collection agent's conversation.

For each rule provided, determine if the agent PASSED or FAILED compliance.

Return ONLY a valid JSON array. Each element must have exactly these fields:
- "rule_id": the rule ID string
- "verdict": "PASS" or "FAIL"
- "reasoning": one concise sentence explaining the verdict
- "citation": the specific turn(s) cited as evidence (e.g., "Turn 3", "Turns 2-4", "Turn 5 — Agent said: '...'")
- "severity": the severity from the rule

Evaluate ONLY the rules provided. Do not add rules or flag issues not in the list.
Do not add commentary. Do not use markdown. Return only the JSON array.

Rules to evaluate:
{rules_json}
"""


class Evaluator:
    def __init__(self):
        self.client = get_client()
        self.model = get_models()["evaluator"]

    def evaluate(self, transcript: str, rules: list[Rule]) -> list[Verdict]:
        if not rules:
            return []

        rules_json = json.dumps(
            [{"id": r.id, "citation": r.citation, "severity": r.severity,
              "agent_must": r.agent_must, "agent_must_not": r.agent_must_not,
              "text": r.text}
             for r in rules],
            indent=2
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(rules_json=rules_json)},
                    {"role": "user", "content": f"Transcript:\n{transcript}"},
                ],
                max_tokens=2048,
                temperature=0.0,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            raw = json.loads(content)
            return [Verdict(**item) for item in raw]
        except Exception:
            return []
