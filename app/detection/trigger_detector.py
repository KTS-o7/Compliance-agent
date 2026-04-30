from __future__ import annotations
import json
import re
from app.detection.labels import TRIGGER_LABELS
from app.llm_client import get_client, get_models, get_no_think_kwargs

SYSTEM_PROMPT = """\
You are a regulatory trigger classifier for debt collection conversations.
Analyze the transcript and return ONLY a JSON array of triggered regulatory situations.
Choose ONLY from this exact list:
{labels}

Rules:
- Return [] if no triggers apply
- Return only labels from the list above — no variations, no new labels
- No explanation, no markdown, no code fences — just the raw JSON array

Example output: ["Debt Dispute", "Right to Validation Request"]
"""


class TriggerDetector:
    def __init__(self):
        self.client = get_client()
        self.model = get_models()["trigger"]
        self._no_think = get_no_think_kwargs()

    def detect(self, transcript: str) -> list[str]:
        prompt = SYSTEM_PROMPT.format(labels=json.dumps(TRIGGER_LABELS, indent=2))
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user",   "content": f"Transcript:\n{transcript}"},
                ],
                max_tokens=256,
                temperature=0.0,
                **self._no_think,
            )
            content = response.choices[0].message.content.strip()
            # Strip any residual think blocks
            if "<think>" in content:
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            labels = json.loads(content.strip())
            return [l for l in labels if l in TRIGGER_LABELS]
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Trigger detection failed: %s", e, exc_info=True)
            return []
