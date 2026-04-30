# Compliance Evaluator Agent — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an AI auditor that evaluates agent-customer transcripts against compliance rules and produces a Regulatory Card with PASS/FAIL verdicts, reasoning, and citations.

**Architecture:** A Streamlit frontend with two views (admin rule ingestion, reviewer transcript evaluation) backed by Ollama-served local models (small for trigger detection, large for evaluation), a Qdrant vector store in a separate Docker container, a deterministic trigger→rule lookup table, corrective RAG that unions both retrieval paths, and an append-only SQLite audit log.

**Tech Stack:** Python 3.11, Streamlit, Ollama (qwen2.5:7b + qwen2.5:14b), Qdrant (Docker), sentence-transformers, SQLite, Docker Compose, pytest

---

## Timebox Strategy (16 hours)

| Hours | Focus |
|-------|-------|
| 0–1   | Repo scaffold, Docker Compose, Ollama + Qdrant up |
| 1–3   | Rule ingestion backend + vector store |
| 3–5   | Trigger detector (small model) |
| 5–7   | Rule retrieval — deterministic + semantic + corrective check |
| 7–10  | Evaluator (large model) + Regulatory Card schema |
| 10–12 | Streamlit UI (admin + reviewer views) + auth/role filter |
| 12–14 | Audit log + QoS eval set (10 transcripts) |
| 14–16 | README, architecture sketch, screen recording |

---

## Project Structure

```
compliance-agent/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── config.yaml                  # model names, Qdrant URL, thresholds
├── run.sh                       # one-command start
├── app/
│   ├── main.py                  # Streamlit entrypoint
│   ├── auth.py                  # hardcoded users + roles
│   ├── models/
│   │   └── schemas.py           # Rule, Verdict, RegulatoryCard, AuditEntry
│   ├── ingestion/
│   │   └── rule_store.py        # chunk, embed, upsert to Qdrant + trigger table
│   ├── detection/
│   │   └── trigger_detector.py  # small model → trigger labels
│   ├── retrieval/
│   │   ├── deterministic.py     # trigger → reg IDs lookup table
│   │   ├── semantic.py          # Qdrant semantic search
│   │   └── corrective.py        # union + disagreement logging
│   ├── evaluation/
│   │   └── evaluator.py         # large model → PASS/FAIL per rule
│   ├── audit/
│   │   └── audit_log.py         # SQLite append-only log
│   └── ui/
│       ├── admin.py             # rule ingestion page
│       └── reviewer.py          # transcript → Regulatory Card page
├── eval/
│   ├── transcripts/             # 10 test transcripts (.txt)
│   ├── ground_truth.json        # expected verdicts per transcript
│   └── run_eval.py              # measures accuracy + p95 latency
├── tests/
│   ├── test_schemas.py
│   ├── test_rule_store.py
│   ├── test_trigger_detector.py
│   ├── test_deterministic.py
│   ├── test_semantic.py
│   ├── test_corrective.py
│   ├── test_evaluator.py
│   └── test_audit_log.py
└── docs/
    └── plans/
        └── 2026-04-30-compliance-evaluator.md
```

---

## Trigger Label Set (define upfront)

```python
TRIGGER_LABELS = [
    "Debt Dispute",
    "Financial Hardship",
    "Bankruptcy Notification",
    "Payment Plan Request",
    "Cease and Desist Request",
    "Account Closure Request",
    "Fraud Claim",
    "Identity Verification Failure",
    "Complaint Escalation",
    "Right to Validation Request",
]
```

---

## Rule Schema

```python
@dataclass
class Rule:
    id: str                    # e.g. "FDCPA-809"
    citation: str              # e.g. "15 U.S.C. § 1692g"
    severity: str              # "critical" | "high" | "medium" | "low"
    agent_must: list[str]      # required agent behaviours
    agent_must_not: list[str]  # prohibited agent behaviours
    role_tag: str              # "junior" | "senior" | "all"
    trigger_labels: list[str]  # which triggers activate this rule
    text: str                  # full rule text for embedding
```

---

## Deterministic Trigger → Rule ID Table (seed data)

```python
TRIGGER_RULE_MAP = {
    "Debt Dispute":              ["FDCPA-809", "FDCPA-805"],
    "Financial Hardship":        ["CFPB-HARDSHIP-01", "FDCPA-806"],
    "Bankruptcy Notification":   ["FDCPA-805", "BANKRUPTCY-362"],
    "Payment Plan Request":      ["CFPB-HARDSHIP-01", "FDCPA-808"],
    "Cease and Desist Request":  ["FDCPA-805", "FDCPA-806"],
    "Account Closure Request":   ["FCRA-623", "CFPB-CLOSURE-01"],
    "Fraud Claim":               ["FCRA-605", "FCRA-623"],
    "Identity Verification Failure": ["FCRA-605", "FDCPA-809"],
    "Complaint Escalation":      ["CFPB-COMPLAINT-01", "FDCPA-813"],
    "Right to Validation Request": ["FDCPA-809", "FDCPA-810"],
}
```

---

## Hardcoded Users + Roles

```python
USERS = {
    "junior": {"password": "junior123", "role": "junior"},
    "senior": {"password": "senior123", "role": "senior"},
}
# junior → role_tag in ("junior", "all")
# senior → role_tag in ("junior", "senior", "all")
```

---

## Task 1: Repo Scaffold + Docker Compose

**Files:**
- Create: `docker-compose.yml`
- Create: `Dockerfile`
- Create: `requirements.txt`
- Create: `config.yaml`
- Create: `run.sh`

**Step 1: Write `docker-compose.yml`**

```yaml
version: "3.9"
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  app:
    build: .
    ports:
      - "8501:8501"
    depends_on:
      - qdrant
      - ollama
    environment:
      - QDRANT_URL=http://qdrant:6333
      - OLLAMA_URL=http://ollama:11434
    volumes:
      - ./app:/app/app
      - ./eval:/app/eval
      - audit_data:/app/data

volumes:
  qdrant_data:
  ollama_data:
  audit_data:
```

**Step 2: Write `Dockerfile`**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Step 3: Write `requirements.txt`**

```
streamlit==1.35.0
qdrant-client==1.9.1
sentence-transformers==3.0.1
ollama==0.2.1
pydantic==2.7.1
pyyaml==6.0.1
pytest==8.2.0
```

**Step 4: Write `config.yaml`**

```yaml
models:
  trigger_model: "qwen2.5:7b"
  evaluator_model: "qwen2.5:14b"

qdrant:
  url: "http://localhost:6333"
  collection: "compliance_rules"
  vector_size: 384

ollama:
  url: "http://localhost:11434"

retrieval:
  semantic_top_k: 5
  disagreement_threshold: 0.5   # fraction of IDs that must differ to flag

audit:
  db_path: "data/audit.db"
```

**Step 5: Write `run.sh`**

```bash
#!/bin/bash
set -e
echo "Starting compliance evaluator..."
docker compose up --build -d
echo "Pulling models (first run only)..."
docker compose exec ollama ollama pull qwen2.5:7b
docker compose exec ollama ollama pull qwen2.5:14b
echo "App ready at http://localhost:8501"
```

**Step 6: Make run.sh executable and commit**

```bash
chmod +x run.sh
git add .
git commit -m "feat: repo scaffold, Docker Compose, Ollama + Qdrant"
```

---

## Task 2: Data Schemas

**Files:**
- Create: `app/models/schemas.py`
- Create: `tests/test_schemas.py`

**Step 1: Write failing test**

```python
# tests/test_schemas.py
from app.models.schemas import Rule, Verdict, RegulatoryCard, AuditEntry

def test_rule_creation():
    r = Rule(
        id="FDCPA-809", citation="15 U.S.C. § 1692g", severity="critical",
        agent_must=["send validation notice"], agent_must_not=["ignore dispute"],
        role_tag="all", trigger_labels=["Debt Dispute"], text="Agent must..."
    )
    assert r.id == "FDCPA-809"
    assert r.severity == "critical"

def test_verdict_defaults():
    v = Verdict(rule_id="FDCPA-809", verdict="PASS", reasoning="Agent complied.", citation="Turn 3", severity="critical")
    assert v.verdict == "PASS"

def test_regulatory_card():
    card = RegulatoryCard(transcript_id="t001", verdicts=[], evaluated_at="2026-04-30T10:00:00")
    assert card.transcript_id == "t001"
```

**Step 2: Run test — expect FAIL**

```bash
pytest tests/test_schemas.py -v
```

**Step 3: Implement `app/models/schemas.py`**

```python
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

class AuditEntry(BaseModel):
    transcript_id: str
    rules_evaluated: list[str]
    verdicts: list[Verdict]
    model_used: str
    latency_seconds: float
    timestamp: str
    user_role: str
```

**Step 4: Run test — expect PASS**

```bash
pytest tests/test_schemas.py -v
```

**Step 5: Commit**

```bash
git add app/models/schemas.py tests/test_schemas.py
git commit -m "feat: data schemas (Rule, Verdict, RegulatoryCard, AuditEntry)"
```

---

## Task 3: Rule Store (Ingestion + Vector Store)

**Files:**
- Create: `app/ingestion/rule_store.py`
- Create: `tests/test_rule_store.py`

**Step 1: Write failing tests**

```python
# tests/test_rule_store.py
from unittest.mock import MagicMock, patch
from app.models.schemas import Rule
from app.ingestion.rule_store import RuleStore

SAMPLE_RULE = Rule(
    id="FDCPA-809", citation="15 U.S.C. § 1692g", severity="critical",
    agent_must=["send validation notice"], agent_must_not=["ignore dispute"],
    role_tag="all", trigger_labels=["Debt Dispute"], text="Agent must send validation notice within 5 days."
)

def test_rule_store_add(mock_qdrant):
    store = RuleStore(qdrant_url="http://localhost:6333")
    store.client = MagicMock()
    store.add_rule(SAMPLE_RULE)
    assert store.client.upsert.called

def test_rule_store_get_by_ids(mock_qdrant):
    store = RuleStore(qdrant_url="http://localhost:6333")
    store.client = MagicMock()
    store.client.retrieve.return_value = []
    results = store.get_by_ids(["FDCPA-809"])
    assert isinstance(results, list)
```

**Step 2: Run test — expect FAIL**

```bash
pytest tests/test_rule_store.py -v
```

**Step 3: Implement `app/ingestion/rule_store.py`**

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from app.models.schemas import Rule
import uuid

COLLECTION = "compliance_rules"
VECTOR_SIZE = 384

class RuleStore:
    def __init__(self, qdrant_url: str):
        self.client = QdrantClient(url=qdrant_url)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION not in existing:
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    def add_rule(self, rule: Rule):
        vector = self.embedder.encode(rule.text).tolist()
        point = PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, rule.id)),
            vector=vector,
            payload={
                "rule_id": rule.id,
                "citation": rule.citation,
                "severity": rule.severity,
                "agent_must": rule.agent_must,
                "agent_must_not": rule.agent_must_not,
                "role_tag": rule.role_tag,
                "trigger_labels": rule.trigger_labels,
                "text": rule.text,
            }
        )
        self.client.upsert(collection_name=COLLECTION, points=[point])

    def semantic_search(self, query: str, top_k: int = 5, role: str = "senior") -> list[Rule]:
        vector = self.embedder.encode(query).tolist()
        role_filter = ["all"] if role == "junior" else ["junior", "senior", "all"]
        results = self.client.search(
            collection_name=COLLECTION,
            query_vector=vector,
            limit=top_k,
            query_filter={"must": [{"key": "role_tag", "match": {"any": role_filter}}]},
        )
        return [self._payload_to_rule(r.payload) for r in results]

    def get_by_ids(self, rule_ids: list[str]) -> list[Rule]:
        points = self.client.retrieve(
            collection_name=COLLECTION,
            ids=[str(uuid.uuid5(uuid.NAMESPACE_DNS, rid)) for rid in rule_ids],
            with_payload=True,
        )
        return [self._payload_to_rule(p.payload) for p in points]

    def _payload_to_rule(self, payload: dict) -> Rule:
        return Rule(**{k: payload[k] for k in Rule.model_fields})
```

**Step 4: Run tests — expect PASS**

```bash
pytest tests/test_rule_store.py -v
```

**Step 5: Commit**

```bash
git add app/ingestion/rule_store.py tests/test_rule_store.py
git commit -m "feat: rule store with Qdrant upsert and semantic search"
```

---

## Task 4: Trigger Detector (Small Model)

**Files:**
- Create: `app/detection/trigger_detector.py`
- Create: `app/detection/labels.py`
- Create: `tests/test_trigger_detector.py`

**Step 1: Write failing tests**

```python
# tests/test_trigger_detector.py
from unittest.mock import patch, MagicMock
from app.detection.trigger_detector import TriggerDetector

TRANSCRIPT = """
Agent: I see you're disputing this debt.
Customer: Yes, I don't believe I owe this amount.
Agent: I understand. We'll need to validate this.
"""

def test_returns_list_of_labels():
    detector = TriggerDetector(model="qwen2.5:7b", ollama_url="http://localhost:11434")
    with patch("ollama.chat") as mock_chat:
        mock_chat.return_value = {"message": {"content": '["Debt Dispute", "Right to Validation Request"]'}}
        labels = detector.detect(TRANSCRIPT)
    assert isinstance(labels, list)
    assert "Debt Dispute" in labels

def test_invalid_json_returns_empty():
    detector = TriggerDetector(model="qwen2.5:7b", ollama_url="http://localhost:11434")
    with patch("ollama.chat") as mock_chat:
        mock_chat.return_value = {"message": {"content": "I cannot determine triggers."}}
        labels = detector.detect(TRANSCRIPT)
    assert labels == []
```

**Step 2: Run test — expect FAIL**

```bash
pytest tests/test_trigger_detector.py -v
```

**Step 3: Implement `app/detection/labels.py`**

```python
TRIGGER_LABELS = [
    "Debt Dispute",
    "Financial Hardship",
    "Bankruptcy Notification",
    "Payment Plan Request",
    "Cease and Desist Request",
    "Account Closure Request",
    "Fraud Claim",
    "Identity Verification Failure",
    "Complaint Escalation",
    "Right to Validation Request",
]
```

**Step 4: Implement `app/detection/trigger_detector.py`**

```python
import json
import ollama
from app.detection.labels import TRIGGER_LABELS

SYSTEM_PROMPT = """You are a compliance trigger detector. Given a conversation transcript, 
identify which regulatory situations are present. Return ONLY a valid JSON array of labels 
from the allowed set. No explanation, no markdown, just the JSON array.

Allowed labels: {labels}"""

class TriggerDetector:
    def __init__(self, model: str, ollama_url: str):
        self.model = model
        self.client = ollama.Client(host=ollama_url)

    def detect(self, transcript: str) -> list[str]:
        prompt = SYSTEM_PROMPT.format(labels=json.dumps(TRIGGER_LABELS))
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Transcript:\n{transcript}"},
                ],
            )
            content = response["message"]["content"].strip()
            labels = json.loads(content)
            return [l for l in labels if l in TRIGGER_LABELS]
        except (json.JSONDecodeError, KeyError):
            return []
```

**Step 5: Run tests — expect PASS**

```bash
pytest tests/test_trigger_detector.py -v
```

**Step 6: Commit**

```bash
git add app/detection/ tests/test_trigger_detector.py
git commit -m "feat: trigger detector using small model with JSON output"
```

---

## Task 5: Rule Retrieval — Deterministic Path

**Files:**
- Create: `app/retrieval/deterministic.py`
- Create: `tests/test_deterministic.py`

**Step 1: Write failing tests**

```python
# tests/test_deterministic.py
from app.retrieval.deterministic import DeterministicRetriever

def test_known_trigger_returns_ids():
    r = DeterministicRetriever()
    ids = r.get_rule_ids(["Debt Dispute"])
    assert "FDCPA-809" in ids

def test_unknown_trigger_returns_empty():
    r = DeterministicRetriever()
    ids = r.get_rule_ids(["Unknown Trigger"])
    assert ids == []

def test_multiple_triggers_union():
    r = DeterministicRetriever()
    ids = r.get_rule_ids(["Debt Dispute", "Bankruptcy Notification"])
    assert "FDCPA-809" in ids
    assert "BANKRUPTCY-362" in ids
```

**Step 2: Run test — expect FAIL**

```bash
pytest tests/test_deterministic.py -v
```

**Step 3: Implement `app/retrieval/deterministic.py`**

```python
TRIGGER_RULE_MAP: dict[str, list[str]] = {
    "Debt Dispute":                  ["FDCPA-809", "FDCPA-805"],
    "Financial Hardship":            ["CFPB-HARDSHIP-01", "FDCPA-806"],
    "Bankruptcy Notification":       ["FDCPA-805", "BANKRUPTCY-362"],
    "Payment Plan Request":          ["CFPB-HARDSHIP-01", "FDCPA-808"],
    "Cease and Desist Request":      ["FDCPA-805", "FDCPA-806"],
    "Account Closure Request":       ["FCRA-623", "CFPB-CLOSURE-01"],
    "Fraud Claim":                   ["FCRA-605", "FCRA-623"],
    "Identity Verification Failure": ["FCRA-605", "FDCPA-809"],
    "Complaint Escalation":          ["CFPB-COMPLAINT-01", "FDCPA-813"],
    "Right to Validation Request":   ["FDCPA-809", "FDCPA-810"],
}

class DeterministicRetriever:
    def get_rule_ids(self, triggers: list[str]) -> list[str]:
        ids: set[str] = set()
        for trigger in triggers:
            ids.update(TRIGGER_RULE_MAP.get(trigger, []))
        return list(ids)
```

**Step 4: Run tests — expect PASS**

```bash
pytest tests/test_deterministic.py -v
```

**Step 5: Commit**

```bash
git add app/retrieval/deterministic.py tests/test_deterministic.py
git commit -m "feat: deterministic trigger→rule ID lookup table"
```

---

## Task 6: Rule Retrieval — Semantic Path + Corrective RAG

**Files:**
- Create: `app/retrieval/semantic.py`
- Create: `app/retrieval/corrective.py`
- Create: `tests/test_semantic.py`
- Create: `tests/test_corrective.py`

**Step 1: Write failing tests**

```python
# tests/test_semantic.py
from unittest.mock import MagicMock
from app.retrieval.semantic import SemanticRetriever
from app.models.schemas import Rule

def test_semantic_search_returns_rules():
    store = MagicMock()
    store.semantic_search.return_value = [
        Rule(id="FDCPA-809", citation="x", severity="critical",
             agent_must=[], agent_must_not=[], role_tag="all",
             trigger_labels=[], text="test")
    ]
    retriever = SemanticRetriever(rule_store=store)
    rules = retriever.search("debt dispute validation", role="senior")
    assert len(rules) == 1
    assert rules[0].id == "FDCPA-809"
```

```python
# tests/test_corrective.py
from app.retrieval.corrective import CorrectiveRetriever
from app.models.schemas import Rule
from unittest.mock import MagicMock

def make_rule(id): 
    return Rule(id=id, citation="x", severity="high", agent_must=[], 
                agent_must_not=[], role_tag="all", trigger_labels=[], text="x")

def test_union_when_agreement():
    det = MagicMock(); det.get_rule_ids.return_value = ["A", "B"]
    sem = MagicMock(); sem.search.return_value = [make_rule("A"), make_rule("B")]
    store = MagicMock(); store.get_by_ids.return_value = [make_rule("A"), make_rule("B")]
    cr = CorrectiveRetriever(deterministic=det, semantic=sem, rule_store=store)
    rules, disagreed = cr.retrieve(["Debt Dispute"], "transcript text", "senior")
    assert not disagreed

def test_union_and_flag_when_disagreement():
    det = MagicMock(); det.get_rule_ids.return_value = ["A", "B"]
    sem = MagicMock(); sem.search.return_value = [make_rule("C"), make_rule("D")]
    store = MagicMock(); store.get_by_ids.return_value = [make_rule("A"), make_rule("B")]
    cr = CorrectiveRetriever(deterministic=det, semantic=sem, rule_store=store)
    rules, disagreed = cr.retrieve(["Debt Dispute"], "transcript text", "senior")
    assert disagreed
    rule_ids = {r.id for r in rules}
    assert "A" in rule_ids and "C" in rule_ids  # union
```

**Step 2: Run tests — expect FAIL**

```bash
pytest tests/test_semantic.py tests/test_corrective.py -v
```

**Step 3: Implement `app/retrieval/semantic.py`**

```python
from app.ingestion.rule_store import RuleStore
from app.models.schemas import Rule

class SemanticRetriever:
    def __init__(self, rule_store: RuleStore, top_k: int = 5):
        self.store = rule_store
        self.top_k = top_k

    def search(self, query: str, role: str = "senior") -> list[Rule]:
        return self.store.semantic_search(query, top_k=self.top_k, role=role)
```

**Step 4: Implement `app/retrieval/corrective.py`**

```python
import logging
from app.models.schemas import Rule

logger = logging.getLogger(__name__)

class CorrectiveRetriever:
    def __init__(self, deterministic, semantic, rule_store, threshold: float = 0.5):
        self.det = deterministic
        self.sem = semantic
        self.store = rule_store
        self.threshold = threshold

    def retrieve(self, triggers: list[str], transcript: str, role: str) -> tuple[list[Rule], bool]:
        det_ids = set(self.det.get_rule_ids(triggers))
        sem_rules = self.sem.search(transcript, role=role)
        sem_ids = {r.id for r in sem_rules}

        overlap = det_ids & sem_ids
        union_ids = det_ids | sem_ids
        disagreement_ratio = 1 - (len(overlap) / max(len(union_ids), 1))
        disagreed = disagreement_ratio > self.threshold

        if disagreed:
            logger.warning(
                "Corrective RAG disagreement: det=%s sem=%s overlap=%s ratio=%.2f",
                det_ids, sem_ids, overlap, disagreement_ratio
            )

        # Union: fetch deterministic IDs from store + merge semantic results
        det_rules = self.store.get_by_ids(list(det_ids))
        all_rules = {r.id: r for r in det_rules + sem_rules}
        return list(all_rules.values()), disagreed
```

**Step 5: Run tests — expect PASS**

```bash
pytest tests/test_semantic.py tests/test_corrective.py -v
```

**Step 6: Commit**

```bash
git add app/retrieval/ tests/test_semantic.py tests/test_corrective.py
git commit -m "feat: semantic retriever + corrective RAG with union and disagreement logging"
```

---

## Task 7: Evaluator (Large Model)

**Files:**
- Create: `app/evaluation/evaluator.py`
- Create: `tests/test_evaluator.py`

**Step 1: Write failing tests**

```python
# tests/test_evaluator.py
import json
from unittest.mock import patch, MagicMock
from app.evaluation.evaluator import Evaluator
from app.models.schemas import Rule, Verdict

RULE = Rule(id="FDCPA-809", citation="15 U.S.C. § 1692g", severity="critical",
            agent_must=["send validation notice"],
            agent_must_not=["ignore dispute"],
            role_tag="all", trigger_labels=["Debt Dispute"],
            text="Agent must send validation notice within 5 days of initial contact.")

TRANSCRIPT = "Turn 1 - Agent: I see you dispute this.\nTurn 2 - Customer: Yes."

def test_evaluator_returns_verdicts():
    evaluator = Evaluator(model="qwen2.5:14b", ollama_url="http://localhost:11434")
    mock_response = json.dumps([
        {"rule_id": "FDCPA-809", "verdict": "FAIL",
         "reasoning": "Agent did not send validation notice.",
         "citation": "Turn 1", "severity": "critical"}
    ])
    with patch("ollama.Client.chat") as mock_chat:
        mock_chat.return_value = {"message": {"content": mock_response}}
        verdicts = evaluator.evaluate(TRANSCRIPT, [RULE])
    assert len(verdicts) == 1
    assert verdicts[0].verdict == "FAIL"
    assert verdicts[0].rule_id == "FDCPA-809"

def test_evaluator_handles_bad_json():
    evaluator = Evaluator(model="qwen2.5:14b", ollama_url="http://localhost:11434")
    with patch("ollama.Client.chat") as mock_chat:
        mock_chat.return_value = {"message": {"content": "Sorry, cannot evaluate."}}
        verdicts = evaluator.evaluate(TRANSCRIPT, [RULE])
    assert verdicts == []
```

**Step 2: Run test — expect FAIL**

```bash
pytest tests/test_evaluator.py -v
```

**Step 3: Implement `app/evaluation/evaluator.py`**

```python
import json
import ollama
from app.models.schemas import Rule, Verdict

SYSTEM_PROMPT = """You are a compliance evaluator auditing an agent conversation.
For each rule provided, evaluate whether the agent PASSED or FAILED.
Return ONLY a valid JSON array. Each element must have:
  - rule_id: the rule ID
  - verdict: "PASS" or "FAIL"
  - reasoning: one sentence explaining the verdict
  - citation: the specific turn(s) from the transcript you are citing (e.g. "Turn 3")
  - severity: the rule severity

Rules to evaluate:
{rules}
"""

class Evaluator:
    def __init__(self, model: str, ollama_url: str):
        self.model = model
        self.client = ollama.Client(host=ollama_url)

    def evaluate(self, transcript: str, rules: list[Rule]) -> list[Verdict]:
        rules_text = json.dumps([
            {
                "id": r.id,
                "citation": r.citation,
                "severity": r.severity,
                "agent_must": r.agent_must,
                "agent_must_not": r.agent_must_not,
                "text": r.text,
            }
            for r in rules
        ], indent=2)

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(rules=rules_text)},
                    {"role": "user", "content": f"Transcript:\n{transcript}"},
                ],
            )
            content = response["message"]["content"].strip()
            raw = json.loads(content)
            return [Verdict(**item) for item in raw]
        except (json.JSONDecodeError, KeyError, Exception):
            return []
```

**Step 4: Run tests — expect PASS**

```bash
pytest tests/test_evaluator.py -v
```

**Step 5: Commit**

```bash
git add app/evaluation/ tests/test_evaluator.py
git commit -m "feat: evaluator using large model, returns PASS/FAIL verdicts with citations"
```

---

## Task 8: Audit Log

**Files:**
- Create: `app/audit/audit_log.py`
- Create: `tests/test_audit_log.py`

**Step 1: Write failing tests**

```python
# tests/test_audit_log.py
import os, tempfile, json
from app.audit.audit_log import AuditLog
from app.models.schemas import AuditEntry, Verdict

def make_entry():
    return AuditEntry(
        transcript_id="t001",
        rules_evaluated=["FDCPA-809"],
        verdicts=[Verdict(rule_id="FDCPA-809", verdict="PASS",
                          reasoning="ok", citation="Turn 1", severity="critical")],
        model_used="qwen2.5:14b",
        latency_seconds=1.5,
        timestamp="2026-04-30T10:00:00",
        user_role="senior",
    )

def test_append_and_read():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        log = AuditLog(db_path)
        log.append(make_entry())
        entries = log.read_all()
        assert len(entries) == 1
        assert entries[0]["transcript_id"] == "t001"
    finally:
        os.unlink(db_path)

def test_append_only_no_delete():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        log = AuditLog(db_path)
        log.append(make_entry())
        log.append(make_entry())
        entries = log.read_all()
        assert len(entries) == 2
    finally:
        os.unlink(db_path)
```

**Step 2: Run test — expect FAIL**

```bash
pytest tests/test_audit_log.py -v
```

**Step 3: Implement `app/audit/audit_log.py`**

```python
import sqlite3
import json
from app.models.schemas import AuditEntry

class AuditLog:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transcript_id TEXT NOT NULL,
                    rules_evaluated TEXT NOT NULL,
                    verdicts TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    latency_seconds REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_role TEXT NOT NULL
                )
            """)

    def append(self, entry: AuditEntry):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO audit_log
                   (transcript_id, rules_evaluated, verdicts, model_used,
                    latency_seconds, timestamp, user_role)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.transcript_id,
                    json.dumps(entry.rules_evaluated),
                    json.dumps([v.model_dump() for v in entry.verdicts]),
                    entry.model_used,
                    entry.latency_seconds,
                    entry.timestamp,
                    entry.user_role,
                )
            )

    def read_all(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM audit_log ORDER BY id").fetchall()
            return [dict(r) for r in rows]
```

**Step 4: Run tests — expect PASS**

```bash
pytest tests/test_audit_log.py -v
```

**Step 5: Commit**

```bash
git add app/audit/ tests/test_audit_log.py
git commit -m "feat: append-only SQLite audit log"
```

---

## Task 9: Auth Module

**Files:**
- Create: `app/auth.py`

**Step 1: Implement `app/auth.py`**

```python
from typing import Optional

USERS = {
    "junior": {"password": "junior123", "role": "junior"},
    "senior": {"password": "senior123", "role": "senior"},
}

ROLE_TAGS = {
    "junior": ["junior", "all"],
    "senior": ["junior", "senior", "all"],
}

def authenticate(username: str, password: str) -> Optional[str]:
    """Returns role if valid, None otherwise."""
    user = USERS.get(username)
    if user and user["password"] == password:
        return user["role"]
    return None

def allowed_role_tags(role: str) -> list[str]:
    return ROLE_TAGS.get(role, [])
```

**Step 2: Commit**

```bash
git add app/auth.py
git commit -m "feat: hardcoded auth with junior/senior roles"
```

---

## Task 10: Streamlit UI

**Files:**
- Create: `app/main.py`
- Create: `app/ui/admin.py`
- Create: `app/ui/reviewer.py`

**Step 1: Implement `app/ui/admin.py`**

```python
import streamlit as st
from app.models.schemas import Rule
from app.ingestion.rule_store import RuleStore
import yaml

def show_admin(rule_store: RuleStore):
    st.title("Rule Library — Admin")
    with st.form("add_rule"):
        rule_id   = st.text_input("Rule ID (e.g. FDCPA-809)")
        citation  = st.text_input("Citation (e.g. 15 U.S.C. § 1692g)")
        severity  = st.selectbox("Severity", ["critical", "high", "medium", "low"])
        role_tag  = st.selectbox("Role Tag", ["all", "junior", "senior"])
        triggers  = st.multiselect("Trigger Labels", options=[
            "Debt Dispute", "Financial Hardship", "Bankruptcy Notification",
            "Payment Plan Request", "Cease and Desist Request",
            "Account Closure Request", "Fraud Claim",
            "Identity Verification Failure", "Complaint Escalation",
            "Right to Validation Request",
        ])
        must      = st.text_area("Agent Must (one per line)")
        must_not  = st.text_area("Agent Must NOT (one per line)")
        text      = st.text_area("Full Rule Text")
        submitted = st.form_submit_button("Add Rule")

    if submitted and rule_id:
        rule = Rule(
            id=rule_id, citation=citation, severity=severity,
            agent_must=[l.strip() for l in must.splitlines() if l.strip()],
            agent_must_not=[l.strip() for l in must_not.splitlines() if l.strip()],
            role_tag=role_tag, trigger_labels=triggers, text=text,
        )
        rule_store.add_rule(rule)
        st.success(f"Rule {rule_id} added to vector store.")
```

**Step 2: Implement `app/ui/reviewer.py`**

```python
import streamlit as st
import time
from datetime import datetime, timezone
from app.detection.trigger_detector import TriggerDetector
from app.retrieval.deterministic import DeterministicRetriever
from app.retrieval.semantic import SemanticRetriever
from app.retrieval.corrective import CorrectiveRetriever
from app.evaluation.evaluator import Evaluator
from app.audit.audit_log import AuditLog
from app.models.schemas import RegulatoryCard, AuditEntry

def show_reviewer(role: str, rule_store, config: dict):
    st.title("Compliance Evaluator — Reviewer")
    st.caption(f"Logged in as: **{role}** reviewer")

    transcript = st.text_area("Paste transcript here", height=300)
    transcript_id = st.text_input("Transcript ID", value=f"t-{int(time.time())}")

    if st.button("Evaluate") and transcript.strip():
        with st.spinner("Detecting triggers..."):
            detector = TriggerDetector(
                model=config["models"]["trigger_model"],
                ollama_url=config["ollama"]["url"]
            )
            triggers = detector.detect(transcript)
        st.write("**Triggers detected:**", triggers)

        with st.spinner("Retrieving rules..."):
            det = DeterministicRetriever()
            sem = SemanticRetriever(rule_store=rule_store, top_k=config["retrieval"]["semantic_top_k"])
            corrective = CorrectiveRetriever(det, sem, rule_store,
                                             threshold=config["retrieval"]["disagreement_threshold"])
            rules, disagreed = corrective.retrieve(triggers, transcript, role)
            if disagreed:
                st.warning("Corrective RAG: deterministic and semantic paths disagreed. Union used.")

        with st.spinner("Evaluating compliance..."):
            evaluator = Evaluator(
                model=config["models"]["evaluator_model"],
                ollama_url=config["ollama"]["url"]
            )
            t0 = time.perf_counter()
            verdicts = evaluator.evaluate(transcript, rules)
            latency = time.perf_counter() - t0

        card = RegulatoryCard(
            transcript_id=transcript_id,
            verdicts=verdicts,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            model_used=config["models"]["evaluator_model"],
            latency_seconds=round(latency, 2),
        )

        st.subheader("Regulatory Card")
        st.metric("Latency", f"{card.latency_seconds}s")
        for v in card.verdicts:
            color = "🔴" if v.verdict == "FAIL" else "🟢"
            with st.expander(f"{color} {v.rule_id} — {v.verdict} ({v.severity})"):
                st.write(f"**Reasoning:** {v.reasoning}")
                st.write(f"**Citation:** {v.citation}")

        # Audit log
        log = AuditLog(config["audit"]["db_path"])
        log.append(AuditEntry(
            transcript_id=transcript_id,
            rules_evaluated=[r.id for r in rules],
            verdicts=verdicts,
            model_used=card.model_used,
            latency_seconds=card.latency_seconds,
            timestamp=card.evaluated_at,
            user_role=role,
        ))
        st.success("Evaluation logged to audit trail.")
```

**Step 3: Implement `app/main.py`**

```python
import streamlit as st
import yaml
import os
from app.auth import authenticate
from app.ingestion.rule_store import RuleStore

def load_config():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["qdrant"]["url"] = os.getenv("QDRANT_URL", cfg["qdrant"]["url"])
    cfg["ollama"]["url"] = os.getenv("OLLAMA_URL", cfg["ollama"]["url"])
    return cfg

config = load_config()
rule_store = RuleStore(qdrant_url=config["qdrant"]["url"])

st.set_page_config(page_title="Compliance Evaluator", layout="wide")

if "role" not in st.session_state:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        role = authenticate(username, password)
        if role:
            st.session_state["role"] = role
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid credentials")
else:
    role = st.session_state["role"]
    page = st.sidebar.radio("View", ["Reviewer"] + (["Admin"] if role == "senior" else []))

    if page == "Admin":
        from app.ui.admin import show_admin
        show_admin(rule_store)
    else:
        from app.ui.reviewer import show_reviewer
        show_reviewer(role=role, rule_store=rule_store, config=config)

    if st.sidebar.button("Logout"):
        del st.session_state["role"]
        st.rerun()
```

**Step 4: Commit**

```bash
git add app/main.py app/ui/ app/auth.py
git commit -m "feat: Streamlit UI with admin and reviewer views, role-based access"
```

---

## Task 11: QoS Eval Set

**Files:**
- Create: `eval/transcripts/t001.txt` through `t010.txt` (10 transcripts)
- Create: `eval/ground_truth.json`
- Create: `eval/run_eval.py`

**Step 1: Create 10 test transcripts**

Each transcript must clearly exercise specific trigger labels. Create a mix:
- t001–t003: Debt Dispute scenarios
- t004–t005: Financial Hardship
- t006: Bankruptcy Notification
- t007: Fraud Claim
- t008: Cease and Desist Request
- t009: Complaint Escalation
- t010: Right to Validation Request

**Step 2: Create `eval/ground_truth.json`**

```json
{
  "t001": {"triggers": ["Debt Dispute"], "expected_verdicts": {"FDCPA-809": "FAIL", "FDCPA-805": "PASS"}},
  "t002": {"triggers": ["Debt Dispute"], "expected_verdicts": {"FDCPA-809": "PASS"}},
  ...
}
```

**Step 3: Implement `eval/run_eval.py`**

```python
import json, time, os, sys, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.detection.trigger_detector import TriggerDetector
from app.retrieval.deterministic import DeterministicRetriever
from app.retrieval.semantic import SemanticRetriever
from app.retrieval.corrective import CorrectiveRetriever
from app.evaluation.evaluator import Evaluator
from app.ingestion.rule_store import RuleStore

with open("config.yaml") as f:
    config = yaml.safe_load(f)

with open("eval/ground_truth.json") as f:
    ground_truth = json.load(f)

rule_store = RuleStore(qdrant_url=config["qdrant"]["url"])
detector = TriggerDetector(model=config["models"]["trigger_model"], ollama_url=config["ollama"]["url"])
det = DeterministicRetriever()
sem = SemanticRetriever(rule_store=rule_store)
corrective = CorrectiveRetriever(det, sem, rule_store)
evaluator = Evaluator(model=config["models"]["evaluator_model"], ollama_url=config["ollama"]["url"])

latencies = []
correct = 0

for tid, gt in ground_truth.items():
    transcript_path = f"eval/transcripts/{tid}.txt"
    with open(transcript_path) as f:
        transcript = f.read()

    t0 = time.perf_counter()
    triggers = detector.detect(transcript)
    rules, _ = corrective.retrieve(triggers, transcript, role="senior")
    verdicts = evaluator.evaluate(transcript, rules)
    latency = time.perf_counter() - t0
    latencies.append(latency)

    verdict_map = {v.rule_id: v.verdict for v in verdicts}
    transcript_correct = all(
        verdict_map.get(rid) == expected
        for rid, expected in gt["expected_verdicts"].items()
    )
    if transcript_correct:
        correct += 1
    print(f"{tid}: {'PASS' if transcript_correct else 'FAIL'} | latency={latency:.2f}s")

latencies_sorted = sorted(latencies)
p95_idx = int(len(latencies_sorted) * 0.95) - 1
p95 = latencies_sorted[max(p95_idx, 0)]
accuracy = correct / len(ground_truth)

print(f"\n=== QoS Results ===")
print(f"Accuracy: {correct}/{len(ground_truth)} = {accuracy:.1%}")
print(f"p95 latency: {p95:.2f}s")
print(f"Target: accuracy >= 80%, p95 <= 5s")
```

**Step 4: Commit**

```bash
git add eval/
git commit -m "feat: QoS eval set — 10 transcripts, ground truth, run_eval script"
```

---

## Task 12: README + Architecture Sketch

**Files:**
- Create: `README.md`

**Content sections:**
1. Overview
2. How to run (one-command: `./run.sh`)
3. Models used + why
4. Trigger label set
5. Rule schema
6. How corrective RAG check works
7. Eval set rationale
8. Measured QoS numbers (fill in after running eval)
9. What you'd do differently with more time
10. Gen AI tools used + verification approach
11. Architecture diagram (embed sketch photo)

**Step 1: Commit README**

```bash
git add README.md
git commit -m "docs: README with architecture, QoS results, and trade-offs"
```

---

## Task 13: Final Integration Check

**Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

**Step 2: Start stack locally**

```bash
./run.sh
```

**Step 3: Smoke test the UI flow**
- Login as `senior` / `senior123`
- Add at least 3 rules via Admin page
- Switch to Reviewer, paste a transcript, evaluate
- Verify Regulatory Card appears
- Verify audit log has entry

**Step 4: Run QoS eval**

```bash
python eval/run_eval.py
```

**Step 5: Record 5-minute demo** covering:
1. Login (both roles)
2. Admin: add a rule
3. Reviewer: paste transcript, evaluate, show Regulatory Card
4. Show audit log entry
5. Show QoS eval output in terminal

**Step 6: Final commit + push**

```bash
git add -A
git commit -m "chore: final integration, README, QoS numbers"
git push origin feature/assignment-plan
```

---

## Cut Decisions (Explicit Scope Reductions)

| Cut | Reason |
|-----|--------|
| Audio transcription | Not in scope per assignment |
| File upload / OCR | Not in scope per assignment |
| PDF Regulatory Card export | Not in scope per assignment |
| OAuth / password hashing | Hardcoded users explicitly allowed |
| Multi-agent debate / ensembling | Not in scope per assignment |
| GPU tuning / vLLM | Not in scope per assignment |
| Load testing | 10-transcript single-user eval is sufficient |
| Separate EC2 for Qdrant | Docker container on same host is sufficient |

---

## Risk Log

| Risk | Mitigation |
|------|-----------|
| qwen2.5:14b too slow for p95 ≤ 5s | Fall back to qwen2.5:7b for both; note in README |
| Qdrant not reachable on first run | Health check in `run.sh`; retry logic in `RuleStore.__init__` |
| LLM returns malformed JSON | `try/except json.JSONDecodeError` in both detector and evaluator |
| Empty rule set for a trigger | Corrective union ensures at least deterministic rules are used |
