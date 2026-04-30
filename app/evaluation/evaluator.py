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
        True streaming via client.beta.chat.completions.stream + Pydantic structured output.
        - content.delta: partial dict, yield completed verdicts as they accumulate
        - content.done: fully parsed EvaluationResponse, flush any remaining verdicts
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
                    if event.type == "content.delta" and event.parsed:
                        # bifrost returns partial dict on content.delta
                        partial = event.parsed if isinstance(event.parsed, dict) else {}
                        partial_verdicts = partial.get("verdicts", [])
                        # yield all complete verdicts except the last (still streaming)
                        while seen < len(partial_verdicts) - 1:
                            try:
                                yield Verdict(**partial_verdicts[seen])
                            except Exception:
                                pass
                            seen += 1

                    elif event.type == "content.done" and event.parsed is not None:
                        # Final fully-validated EvaluationResponse object
                        parsed = event.parsed
                        if isinstance(parsed, EvaluationResponse):
                            verdicts_list = parsed.verdicts
                        elif isinstance(parsed, dict):
                            verdicts_list = [Verdict(**v) for v in parsed.get("verdicts", [])]
                        else:
                            verdicts_list = []
                        # flush all remaining verdicts including the last one
                        while seen < len(verdicts_list):
                            yield verdicts_list[seen]
                            seen += 1
        except Exception:
            return
