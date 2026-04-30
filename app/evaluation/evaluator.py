from __future__ import annotations
import json
import re
from typing import Generator
from pydantic import BaseModel
from app.models.schemas import Rule, Verdict
from app.llm_client import get_client, get_models, use_structured_output, get_no_think_kwargs

SYSTEM_PROMPT_CLOUD = """\
You are a strict compliance auditor evaluating a debt collection agent's conversation.

For each rule provided, determine if the agent PASSED or FAILED compliance.

IMPORTANT: Only mark FAIL if there is clear, direct, explicit evidence in the transcript that the agent violated the specific rule. Do NOT infer violations from ambiguous or borderline language. When in doubt, mark PASS.

Evaluate ONLY the rules provided. Do not add rules or flag issues not in the list.

Rules to evaluate:
{rules_json}
"""

SYSTEM_PROMPT_JSON = """\
You are a strict compliance auditor evaluating a debt collection agent's conversation.

For each rule provided, determine if the agent PASSED or FAILED compliance.

IMPORTANT: Only mark FAIL if there is clear, direct, explicit evidence in the transcript that the agent violated the specific rule. Do NOT infer violations from ambiguous or borderline language. When in doubt, mark PASS.

Evaluate ONLY the rules provided. Do not add rules or flag issues not in the list.

Return a JSON object in this exact format:
{{"verdicts": [{{"rule_id": "...", "verdict": "PASS or FAIL", "reasoning": "...", "citation": "...", "severity": "..."}}]}}

No markdown, no code fences, no explanation — only the raw JSON object.

Rules to evaluate:
{rules_json}
"""

# Wrapper model for cloud beta structured output
class EvaluationResponse(BaseModel):
    verdicts: list[Verdict]


def _parse_verdicts(content: str) -> list[Verdict]:
    """Parse verdicts from raw LLM content, stripping fences/think blocks."""
    if not content:
        return []
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-z]*\n?", "", content).rstrip("```").strip()
    try:
        data = json.loads(content)
    except Exception:
        return []
    results = []
    for v in data.get("verdicts", []):
        try:
            results.append(Verdict(**v))
        except Exception:
            pass
    return results


class Evaluator:
    def __init__(self):
        self.client = get_client()
        self.model = get_models()["evaluator"]
        self.structured = use_structured_output()
        self._no_think = get_no_think_kwargs()

    def _messages(self, transcript: str, rules: list[Rule]) -> list[dict]:
        rules_json = json.dumps(
            [{"id": r.id, "citation": r.citation, "severity": r.severity,
              "agent_must": r.agent_must, "agent_must_not": r.agent_must_not,
              "text": r.text}
             for r in rules],
            indent=2
        )
        prompt = SYSTEM_PROMPT_CLOUD if self.structured else SYSTEM_PROMPT_JSON
        return [
            {"role": "system", "content": prompt.format(rules_json=rules_json)},
            {"role": "user",   "content": f"Transcript:\n{transcript}"},
        ]

    def evaluate(self, transcript: str, rules: list[Rule]) -> list[Verdict]:
        """Non-streaming evaluation — used by run_eval.py."""
        if not rules:
            return []
        try:
            if self.structured:
                # cloud — beta.parse with JSON schema
                completion = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=self._messages(transcript, rules),
                    response_format=EvaluationResponse,
                    max_tokens=2048,
                    temperature=0.0,
                )
                result = completion.choices[0].message.parsed
                return result.verdicts if result else []
            else:
                # ollama — plain create + manual JSON parse
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=self._messages(transcript, rules),
                    max_tokens=2048,
                    temperature=0.0,
                    **self._no_think,
                )
                return _parse_verdicts(completion.choices[0].message.content or "")
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Evaluation failed: %s", e, exc_info=True)
            return []

    def evaluate_stream_verdicts(self, transcript: str, rules: list[Rule]) -> Generator[Verdict, None, None]:
        """Streaming evaluation."""
        if not rules:
            return
        try:
            if self.structured:
                # cloud — beta.stream with structured output
                seen = 0
                with self.client.beta.chat.completions.stream(
                    model=self.model,
                    messages=self._messages(transcript, rules),
                    response_format=EvaluationResponse,
                    max_tokens=2048,
                    temperature=0.0,
                ) as stream:
                    for event in stream:
                        if event.type == "content.delta" and event.parsed:
                            partial = event.parsed if isinstance(event.parsed, dict) else {}
                            partial_verdicts = partial.get("verdicts", [])
                            while seen < len(partial_verdicts) - 1:
                                try:
                                    yield Verdict(**partial_verdicts[seen])
                                except Exception:
                                    pass
                                seen += 1
                        elif event.type == "content.done" and event.parsed is not None:
                            parsed = event.parsed
                            if isinstance(parsed, EvaluationResponse):
                                verdicts_list = parsed.verdicts
                            elif isinstance(parsed, dict):
                                verdicts_list = [Verdict(**v) for v in parsed.get("verdicts", [])]
                            else:
                                verdicts_list = []
                            while seen < len(verdicts_list):
                                yield verdicts_list[seen]
                                seen += 1
            else:
                # ollama — stream and accumulate full response then parse
                full_content = ""
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=self._messages(transcript, rules),
                    max_tokens=2048,
                    temperature=0.0,
                    stream=True,
                    **self._no_think,
                )
                for chunk in stream:
                    full_content += chunk.choices[0].delta.content or ""
                for verdict in _parse_verdicts(full_content):
                    yield verdict
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Streaming evaluation failed: %s", e, exc_info=True)
            return
