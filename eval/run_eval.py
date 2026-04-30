#!/usr/bin/env python3
"""
QoS evaluation: measures accuracy and p95 latency over 10 ground-truth transcripts.
Run from project root: python3.11 eval/run_eval.py
Requires Qdrant running: docker compose up qdrant -d
"""
from __future__ import annotations
import json
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env if present (for local runs outside Docker)
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

import yaml
from app.detection.trigger_detector import TriggerDetector
from app.retrieval.corrective import CorrectiveRetriever
from app.evaluation.evaluator import Evaluator
from app.ingestion.rule_store import RuleStore
from app.ingestion.seed_rules import seed

GT_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")
TX_DIR  = os.path.join(os.path.dirname(__file__), "transcripts")

with open(GT_PATH) as f:
    ground_truth: dict = json.load(f)

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

print("=== Compliance Evaluator — QoS Eval ===\n")
print("Connecting to Qdrant and seeding rules...")
store = RuleStore()
inserted = seed(store)
print(f"Seed: {inserted} rules inserted (0 = already seeded)\n")

detector  = TriggerDetector()
evaluator = Evaluator()

latencies: list[float] = []
correct = 0
results: list[dict] = []

for tid in sorted(ground_truth.keys()):
    gt = ground_truth[tid]
    tx_path = os.path.join(TX_DIR, f"{tid}.txt")
    with open(tx_path) as f:
        transcript = f.read()

    t0 = time.perf_counter()

    triggers = detector.detect(transcript)
    retriever = CorrectiveRetriever(
        rule_store=store,
        threshold=cfg["retrieval"]["disagreement_threshold"],
        top_k=cfg["retrieval"]["semantic_top_k"],
    )
    rules, disagreed = retriever.retrieve(triggers, transcript, role="senior")
    verdicts = evaluator.evaluate(transcript, rules)

    latency = round(time.perf_counter() - t0, 2)
    latencies.append(latency)

    verdict_map = {v.rule_id: v.verdict for v in verdicts}
    expected = gt["expected_verdicts"]
    transcript_correct = all(
        verdict_map.get(rid) == exp
        for rid, exp in expected.items()
    )
    if transcript_correct:
        correct += 1

    status = "✅" if transcript_correct else "❌"
    print(f"{tid}: {status} | latency={latency:.2f}s | triggers={triggers}")
    if not transcript_correct:
        for rid, exp in expected.items():
            got = verdict_map.get(rid, "NOT_FOUND")
            if got != exp:
                print(f"       {rid}: expected={exp} got={got}")
    if disagreed:
        print(f"       ⚠️  corrective RAG disagreement")

latencies_sorted = sorted(latencies)
p95_idx = max(int(len(latencies_sorted) * 0.95) - 1, 0)
p95 = latencies_sorted[p95_idx]
accuracy = correct / len(ground_truth)

print(f"\n{'='*50}")
print(f"Accuracy : {correct}/{len(ground_truth)} = {accuracy:.0%}   (target: >= 80%)")
print(f"p95 lat  : {p95:.2f}s                    (target: <= 5s)")
print(f"Mean lat : {sum(latencies)/len(latencies):.2f}s")
print(f"{'='*50}")

if accuracy >= 0.8 and p95 <= 5.0:
    print("✅ All QoS targets MET")
else:
    if accuracy < 0.8:
        print(f"❌ Accuracy below target ({accuracy:.0%} < 80%)")
    if p95 > 5.0:
        print(f"❌ p95 latency above target ({p95:.2f}s > 5s)")

# Write results to file for README
results_path = os.path.join(os.path.dirname(__file__), "qos_results.json")
with open(results_path, "w") as f:
    json.dump({
        "accuracy": accuracy,
        "correct": correct,
        "total": len(ground_truth),
        "p95_latency": p95,
        "mean_latency": round(sum(latencies)/len(latencies), 2),
        "latencies": latencies,
    }, f, indent=2)
print(f"\nResults saved to eval/qos_results.json")
