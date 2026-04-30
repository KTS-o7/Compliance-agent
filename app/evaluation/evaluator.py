from __future__ import annotations
import json
from typing import Generator
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

    def _build_messages(self, transcript: str, rules: list[Rule]) -> tuple[list[dict], str]:
        rules_json = json.dumps(
            [{"id": r.id, "citation": r.citation, "severity": r.severity,
              "agent_must": r.agent_must, "agent_must_not": r.agent_must_not,
              "text": r.text}
             for r in rules],
            indent=2
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(rules_json=rules_json)},
            {"role": "user", "content": f"Transcript:\n{transcript}"},
        ]
        return messages, rules_json

    def _parse_content(self, content: str) -> list[Verdict]:
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        raw = json.loads(content)
        return [Verdict(**item) for item in raw]

    def evaluate(self, transcript: str, rules: list[Rule]) -> list[Verdict]:
        """Non-streaming evaluation — used by run_eval.py."""
        if not rules:
            return []
        messages, _ = self._build_messages(transcript, rules)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,
                temperature=0.0,
            )
            return self._parse_content(response.choices[0].message.content)
        except Exception:
            return []

    def evaluate_stream(self, transcript: str, rules: list[Rule]) -> Generator[str, None, None]:
        """Streaming evaluation — yields token chunks as they arrive."""
        if not rules:
            return
        messages, _ = self._build_messages(transcript, rules)
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,
                temperature=0.0,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception:
            return

    def evaluate_stream_parsed(self, transcript: str, rules: list[Rule]) -> tuple[list[Verdict], str]:
        """Stream tokens, collect full content, then parse. Returns (verdicts, raw_content)."""
        chunks = list(self.evaluate_stream(transcript, rules))
        content = "".join(chunks)
        try:
            return self._parse_content(content), content
        except Exception:
            return [], content
