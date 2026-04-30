from __future__ import annotations
import json
from typing import Generator
from pydantic import BaseModel
from app.models.schemas import Rule, Verdict
from app.llm_client import get_client, get_models

SYSTEM_PROMPT = """\
You are a strict compliance auditor evaluating a debt collection agent's conversation.

For each rule provided, determine if the agent PASSED or FAILED compliance.

Evaluate ONLY the rules provided. Do not add rules or flag issues not in the list.

Rules to evaluate:
{rules_json}
"""

# Wrapper model so client.beta.chat.completions.stream can parse the full response
class EvaluationResponse(BaseModel):
    verdicts: list[Verdict]


class Evaluator:
    def __init__(self):
        self.client = get_client()
        self.model = get_models()["evaluator"]

    def _build_messages(self, transcript: str, rules: list[Rule]) -> list[dict]:
        rules_json = json.dumps(
            [{"id": r.id, "citation": r.citation, "severity": r.severity,
              "agent_must": r.agent_must, "agent_must_not": r.agent_must_not,
              "text": r.text}
             for r in rules],
            indent=2
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT.format(rules_json=rules_json)},
            {"role": "user", "content": f"Transcript:\n{transcript}"},
        ]

    def evaluate(self, transcript: str, rules: list[Rule]) -> list[Verdict]:
        """Non-streaming evaluation — used by run_eval.py."""
        if not rules:
            return []
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=self._build_messages(transcript, rules),
                response_format=EvaluationResponse,
                max_tokens=2048,
                temperature=0.0,
            )
            result = completion.choices[0].message.parsed
            return result.verdicts if result else []
        except Exception:
            return []

    def evaluate_stream_verdicts(self, transcript: str, rules: list[Rule]) -> Generator[Verdict, None, None]:
        """
        Production streaming using client.beta.chat.completions.stream +
        Pydantic structured output. The SDK fires content.delta events with
        event.snapshot — a fully-parsed EvaluationResponse that grows as tokens
        arrive. We diff consecutive snapshots to yield each new Verdict the
        moment it completes.
        """
        if not rules:
            return
        try:
            seen = 0
            with self.client.beta.chat.completions.stream(
                model=self.model,
                messages=self._build_messages(transcript, rules),
                response_format=EvaluationResponse,
                max_tokens=2048,
                temperature=0.0,
            ) as stream:
                for event in stream:
                    if event.type == "content.delta" and event.parsed is not None:
                        current: EvaluationResponse = event.parsed
                        current_verdicts = current.verdicts or []
                        # yield each newly completed verdict
                        while seen < len(current_verdicts) - 1:
                            yield current_verdicts[seen]
                            seen += 1
                # yield the final verdict once stream completes
                final = stream.get_final_completion()
                if final:
                    result = final.choices[0].message.parsed
                    if result:
                        while seen < len(result.verdicts):
                            yield result.verdicts[seen]
                            seen += 1
        except Exception:
            return
