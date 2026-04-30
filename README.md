# Compliance Evaluator Agent

An AI auditor that evaluates debt collection agent transcripts against compliance rules, producing a **Regulatory Card** with PASS/FAIL verdicts, reasoning, and turn citations for each applicable rule.

---

## Quick Start

```bash
./run.sh
```

Then open **http://localhost:8502**

Credentials:
- `senior` / `senior123` — full access (Admin, Reviewer, Audit Log)
- `junior` / `junior123` — Reviewer only (Tier-1 rules only)

> **Requires:** Docker Desktop running, bifrost LLM proxy running on `localhost:24242`

---

## Architecture

```mermaid
graph TD
    User["👤 User (Browser)"] --> Streamlit["Streamlit App :8502"]

    Streamlit --> Login["Auth\njunior / senior"]
    Login -->|senior| AdminPage["Admin Page\nRule Ingestion"]
    Login -->|any| ReviewerPage["Reviewer Page\nTranscript Evaluation"]
    Login -->|senior| AuditPage["Audit Log Page"]

    AdminPage --> RuleStore["Rule Store\nrule_store.py"]
    RuleStore --> BGE["BAAI/bge-base-en-v1.5\nEmbedder (768-dim)"]
    BGE --> Qdrant["Qdrant :6333\n(Docker container)"]

    ReviewerPage --> TriggerDetector["Trigger Detector\nHaiku 4.5"]
    TriggerDetector --> Bifrost["bifrost :24242\n(OpenAI-compatible)"]
    Bifrost --> Bedrock["AWS Bedrock"]

    ReviewerPage --> CorrectiveRAG["Corrective RAG"]
    CorrectiveRAG --> DeterministicRetriever["Deterministic\nTRIGGER_RULE_MAP"]
    CorrectiveRAG --> SemanticRetriever["Semantic Search\nQdrant"]
    SemanticRetriever --> Qdrant

    CorrectiveRAG --> Evaluator["Evaluator\nSonnet 4.6"]
    Evaluator --> Bifrost

    Evaluator --> RegulatoryCard["Regulatory Card\nPASS/FAIL + Citations"]
    RegulatoryCard --> AuditLog["Audit Log\nSQLite data/audit.db"]
    AuditPage --> AuditLog
```

### Trigger Detection State Diagram

```mermaid
stateDiagram-v2
    [*] --> SendToHaiku: transcript received
    SendToHaiku --> ParseJSON: LLM response received
    ParseJSON --> FilterValidLabels: JSON parsed OK
    ParseJSON --> ReturnEmpty: JSONDecodeError
    FilterValidLabels --> ReturnLabels: labels validated against TRIGGER_LABELS
    ReturnEmpty --> [*]
    ReturnLabels --> [*]
```

### Corrective RAG State Diagram

```mermaid
stateDiagram-v2
    [*] --> DeterministicLookup: triggers received
    [*] --> SemanticSearch: transcript received (parallel)
    DeterministicLookup --> ComputeDisagreement: det_ids ready
    SemanticSearch --> ComputeDisagreement: sem_ids ready
    ComputeDisagreement --> LogWarning: ratio > 0.5
    ComputeDisagreement --> BuildUnion: ratio ≤ 0.5
    LogWarning --> BuildUnion: always union
    BuildUnion --> RoleFilter: merged rule set
    RoleFilter --> ReturnRules: role_tag filter applied
    ReturnRules --> [*]
```

### Evaluation State Diagram

```mermaid
stateDiagram-v2
    [*] --> CheckRules: rules + transcript received
    CheckRules --> ReturnEmpty: no rules
    CheckRules --> BuildPrompt: rules exist
    BuildPrompt --> SendToSonnet: rules serialised to JSON
    SendToSonnet --> ParseVerdicts: response received
    ParseVerdicts --> StripFences: markdown fences detected
    StripFences --> ParseVerdicts: fences removed
    ParseVerdicts --> ReturnVerdicts: parsed successfully
    ParseVerdicts --> ReturnEmpty: JSONDecodeError
    ReturnVerdicts --> [*]
    ReturnEmpty --> [*]
```

---

## Models Used

| Role | Model | Why |
|------|-------|-----|
| Trigger detection | `claude-haiku-4-5` (via bifrost) | Fast, cheap, structured JSON output |
| Evaluation | `claude-sonnet-4-6` (via bifrost) | Deep reasoning, high citation quality |
| Embeddings | `BAAI/bge-base-en-v1.5` (local) | 84.7% recall, 768-dim, ~420MB, English-optimised |

**LLM access:** Local [bifrost](https://github.com/promptfoo/bifrost) proxy at `localhost:24242` providing an OpenAI-compatible API over AWS Bedrock. To switch to self-hosted Ollama models, update `config.yaml`:

```yaml
bifrost:
  base_url: "http://localhost:11434/v1"
  api_key: "ollama"
models:
  trigger_model: "qwen2.5:3b"
  evaluator_model: "qwen2.5:7b"
```

Zero code changes required.

---

## Trigger Label Set

The trigger detector maps transcripts to one or more of these 10 regulatory situation labels:

| Label | Applicable Rules |
|-------|-----------------|
| Debt Dispute | FDCPA-809, FDCPA-807 |
| Financial Hardship | CFPB-HARDSHIP-01, FDCPA-806 |
| Bankruptcy Notification | FDCPA-805, BANKRUPTCY-362 |
| Payment Plan Request | CFPB-HARDSHIP-01, FDCPA-808 |
| Cease and Desist Request | FDCPA-805, FDCPA-806 |
| Account Closure Request | FCRA-623 |
| Fraud Claim | FCRA-605, FCRA-623, FDCPA-807 |
| Identity Verification Failure | FCRA-605 |
| Complaint Escalation | CFPB-COMPLAINT-01 |
| Right to Validation Request | FDCPA-809 |

---

## Rule Schema

```python
class Rule(BaseModel):
    id: str                      # "FDCPA-809"
    citation: str                # "15 U.S.C. § 1692g"
    severity: Literal["critical", "high", "medium", "low"]
    agent_must: list[str]        # required behaviours
    agent_must_not: list[str]    # prohibited behaviours
    role_tag: Literal["junior", "senior", "all"]  # role visibility
    trigger_labels: list[str]    # which triggers activate this rule
    text: str                    # full rule text — embedded in Qdrant
```

10 rules are pre-seeded at startup covering FDCPA, CFPB, FCRA, and Bankruptcy statutes. `BANKRUPTCY-362` and `FCRA-623` are `senior`-only rules.

---

## How Corrective RAG Works

For each transcript evaluation, rules are retrieved via **two parallel paths**:

1. **Deterministic path:** The detected trigger labels are looked up in a hardcoded `TRIGGER_RULE_MAP` dictionary → returns a list of rule IDs.

2. **Semantic path:** The full transcript is embedded with `BAAI/bge-base-en-v1.5` (with BGE asymmetric search prefix) and searched against the Qdrant vector store using cosine similarity → returns top-5 rules by semantic relevance.

**Corrective check:** The two sets are compared. If the Jaccard disagreement ratio exceeds 0.5 (i.e., less than 50% overlap relative to the union), a warning is logged and flagged on the Regulatory Card.

**Result:** The **union** of both paths is always used, filtered by the user's role tag. This ensures neither path's rules are silently dropped.

---

## Eval Set Rationale

10 transcripts, one per major trigger type, designed with **known ground truth**:

| ID | Trigger | Key Rule | Expected |
|----|---------|----------|----------|
| t001 | Debt Dispute | FDCPA-809 | FAIL — agent ignores dispute |
| t002 | Debt Dispute | FDCPA-809 | PASS — agent handles correctly |
| t003 | Financial Hardship | CFPB-HARDSHIP-01 | FAIL — agent denies programs |
| t004 | Payment Plan | FDCPA-808, CFPB-HARDSHIP-01 | PASS — agent offers plan + program |
| t005 | Bankruptcy | BANKRUPTCY-362, FDCPA-805 | FAIL — agent continues collecting |
| t006 | Cease and Desist | FDCPA-805, FDCPA-806 | PASS — agent stops immediately |
| t007 | Fraud Claim | FCRA-605 | FAIL — no identity verification |
| t008 | Complaint Escalation | CFPB-COMPLAINT-01 | PASS — escalates properly |
| t009 | Identity Verification | FCRA-605 | PASS — handles correctly |
| t010 | Right to Validation | FDCPA-809 | FAIL — continues without validating |

Transcripts were generated with Claude Sonnet 4.6 via bifrost to accelerate drafting, then manually reviewed to ensure each transcript unambiguously supports its ground truth verdict.

---

## Measured QoS Numbers

> Run `python3.11 eval/run_eval.py` to generate these numbers.

| Metric | Result | Target |
|--------|--------|--------|
| Accuracy | **TBD / 10** | ≥ 8/10 (80%) |
| p95 latency | **TBD s** | ≤ 5s |
| Mean latency | **TBD s** | — |

*Numbers will be filled in after running the eval against the deployed system.*

---

## What I'd Do Differently With More Time

1. **Self-hosted models from day 1** — Use Ollama with `qwen2.5:3b` + `qwen2.5:7b` as the default. I used bifrost (API-backed) for speed during development; the architecture is designed for a zero-code swap.

2. **Streaming UI** — Stream the evaluator's response token-by-token into the Regulatory Card for better UX.

3. **Proper auth** — bcrypt password hashing, session tokens, per-user audit history.

4. **PDF export** — Generate a signed, timestamped PDF Regulatory Card for compliance teams.

5. **More eval transcripts** — 10 is the minimum. A real benchmark would cover edge cases: multiple simultaneous triggers, ambiguous transcripts, multi-turn disputes.

6. **Retrieval tuning** — The `disagreement_threshold` of 0.5 is a reasonable default but should be calibrated against a larger ground-truth set.

7. **Structured citations** — Parse turn numbers from transcripts at ingestion time so citations are always in the format `Turn N` rather than free-text.

---

## Gen AI Tools Used

| Tool | How Used |
|------|----------|
| **OpenCode** (Claude Sonnet 4.6) | Primary coding assistant — architecture planning, implementation, test writing |
| **bifrost + Claude Haiku 4.5** | Trigger detection in the running system |
| **bifrost + Claude Sonnet 4.6** | Evaluation, QoS transcript generation |

### How LLM-Generated Code Was Verified

1. **Unit tests** — Every module has a test file with mocked dependencies. Tests were written alongside the code and must pass before committing.
2. **Manual code review** — Each generated file was read and verified for correctness before committing.
3. **QoS eval set** — 10 transcripts with known ground truth provide end-to-end validation of the full pipeline.
4. **Type annotations + Pydantic** — All data models are Pydantic v2 with `Literal` type constraints, catching schema errors at parse time.

---

## Project Structure

```
compliance-agent/
├── docker-compose.yml      # Qdrant + app containers
├── Dockerfile              # Python 3.11 + uv + pre-downloaded BGE model
├── requirements.txt        # Exact pinned versions
├── config.yaml             # Model IDs, URLs, thresholds
├── run.sh                  # One-command start
├── app/
│   ├── main.py             # Streamlit entrypoint
│   ├── auth.py             # Hardcoded users + role helpers
│   ├── llm_client.py       # OpenAI SDK → bifrost wrapper
│   ├── models/schemas.py   # Pydantic schemas
│   ├── ingestion/          # RuleStore + seed rules
│   ├── detection/          # Trigger detector (Haiku 4.5)
│   ├── retrieval/          # Deterministic + semantic + corrective RAG
│   ├── evaluation/         # Evaluator (Sonnet 4.6)
│   ├── audit/              # SQLite audit log
│   └── ui/                 # Streamlit pages
├── eval/
│   ├── transcripts/        # 10 ground-truth transcripts
│   ├── ground_truth.json   # Expected verdicts per transcript
│   └── run_eval.py         # QoS measurement script
└── tests/                  # Unit tests (27 passing, fully mocked)
```
