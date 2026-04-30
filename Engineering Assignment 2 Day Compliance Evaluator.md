**Compliance Evaluator Agent: 2-Day Build**

| Title | Compliance Evaluator Agent  |
| :---- | :---- |
| **Timebox** | Please submit what you have in 16 hours  |
| **Submit** | Git repo \+ README \+ 5-minute screen recording or live demo. |

# **What You're Building**

An AI evaluator that audits other agents on compliance. Conceptually: a benchmark/judge.

Input: a completed conversation transcript (between any agent and a customer).

Output: a Regulatory Card listing every applicable compliance rule, whether the agent passed or failed each, with reasoning that cites specific turns in the transcript.

This is the auditor that sits on top of whatever agent the bank is running. 

# **How We Evaluate Your Submission**

* Does the architecture match the requirements? (separate vector store, two-model router, corrective RAG, role-filtered rule scope, audit log)

* Does the demo run end-to-end without you fixing it live?

* Are the QoS numbers honest and measured, not estimated?

* Is the Regulatory Card output usable — would a compliance reviewer trust it and act on it?

* Are the trade-offs you made the right ones for a 2-day timebox? We expect cuts. We want to see good ones.

# **Architecture (the pieces, in order)**

* **Frontend:** two views — (1) admin page for ingesting rules, (2) reviewer page that takes a pasted transcript and shows the Regulatory Card.

* **Auth layer:** two hardcoded users with different roles. Role determines which rules are in evaluation scope.

* **Trigger detector (small model):** reads the transcript, returns a list of regulatory situation labels (e.g., "Debt Dispute", "Financial Hardship", "Bankruptcy Notification"). You define the label set.

* **Rule retrieval:** two paths in parallel — (a) deterministic lookup against a Trigger→Regulation-IDs table, (b) semantic search against the vector store. Corrective check compares the two; if they disagree, log it and use the union.

* **Evaluator (larger model):** receives the transcript \+ retrieved rule objects. For each rule, outputs PASS/FAIL with reasoning and citation to the relevant turns.

* **Vector store (separate process):** stores embedded rules with metadata (severity, role tag, trigger labels). Runs as its own Docker container.

* **Audit log:** append-only record of every evaluation (transcript ID, rules evaluated, verdicts, model used, latency, timestamp).

# **What To Build (and What To Skip)**

Each feature has a minimum viable version. Anything in the right column is explicitly out of scope — don't burn hours on it.

| Feature | Minimum viable (must) | Skip for now (don't) |
| :---- | :---- | :---- |
| **Transcript intake** | UI lets a reviewer paste a transcript (any format — labelled turns are fine) and submit it for evaluation. Streamlit or Gradio is fine. | Audio transcription, file parsing, multi-format support, polish. |
| **Rule library ingestion** | Admin page where you can add a regulation: ID, citation, severity, agent\_must list, agent\_must\_not list, role tag. System chunks, embeds, stores. | File uploads, OCR, document parsing. |
| **Trigger detection (small model)** | LLM call labels the transcript with regulatory situations it contains (e.g., "Debt Dispute", "Financial Hardship"). Output a fixed set of labels — define them yourself. | Custom-trained classifier, hierarchical taxonomies. |
| **Rule retrieval (corrective RAG)** | For each detected trigger, fetch applicable rules. Two paths: (a) deterministic lookup from a Trigger→RegIDs table, (b) semantic search in the vector store. Run a corrective check: if the two paths disagree significantly, log it or re-query. | Multi-hop graph retrieval, full CRAG paper implementation. |
| **Evaluator (larger model)** | LLM receives transcript \+ retrieved rule objects. For each rule, returns: PASS / FAIL, one-sentence reasoning, severity. Cite the specific transcript turn(s). | Multi-agent debate, ensembling, fine-tuning. |
| **LLM router** | Small model for trigger detection. Larger model for the evaluator step. Configurable via a config file. That's the router. | Trained classifier router, dynamic routing, cost optimization. |
| **Self-hosted models** | Ollama serving two open-weight models (e.g., qwen2.5:7b for triggers \+ qwen2.5:14b for evaluation, or Mistral 7B \+ Mixtral). Local or one EC2 g5/g6 instance. | vLLM tuning, quantization experiments, GPU optimization. |
| **Vector store on separate process** | Vector DB runs as a separate Docker container or process — not in the agent process. Qdrant, Chroma, or pgvector all fine. | Separate EC2 instance, replication, managed service. |
| **Auth \+ roles** | Two hardcoded users with different roles (e.g., "junior reviewer" sees only Tier-1 rules; "senior reviewer" sees all). Role filters which rules are in scope when the evaluator runs. | OAuth, password hashing rigor, account UI. |
| **Regulatory Card output** | Per-transcript output: list of every rule evaluated, PASS/FAIL, reasoning, severity. Display in the UI and persist to an audit log (file or table). | PDF generation, dashboards, trend reporting. |
| **QoS measurement** | Build a 10-transcript eval set, each with known correct verdicts. Run the evaluator, compare to ground truth. Report measured p95 latency and accuracy. | Load testing, full benchmark suites. |

# **QoS Targets**

* **Latency:** p95 ≤ 5 seconds per transcript evaluation, measured over your 10-transcript eval set. Single user, no concurrency required.

* **Accuracy:** ≥ 8/10 transcripts where every rule verdict matches your ground-truth labels. You define the eval set; that's part of the deliverable.

If you miss a target, say so in the README and explain why. Honest numbers beat fudged ones every time.

# **Deliverables**

* Git repo with code, Dockerfiles or docker-compose, and a one-command run script.

* README covering: how to run it, what models you used, your trigger label set, your rule schema, how the corrective-RAG check works, your eval-set rationale, your measured QoS numbers, and what you'd do differently with more time.

* 5-minute screen recording walking through the demo flow above. Live demo acceptable as a substitute.

* Sketch of the architecture: a hand drawing or whiteboard photo is fine. Don't spend more than 15 minutes on this.

* List which gen AI tools you used to code and how do you verify LLM generated code?

