from __future__ import annotations
import time
import uuid
import streamlit as st
from datetime import datetime, timezone
from app.detection.trigger_detector import TriggerDetector
from app.retrieval.corrective import CorrectiveRetriever
from app.evaluation.evaluator import Evaluator
from app.audit.audit_log import AuditLog
from app.models.schemas import RegulatoryCard, AuditEntry
from app.ingestion.rule_store import RuleStore

SEVERITY_COLOUR = {
    "critical": "🔴",
    "high":     "🟠",
    "medium":   "🟡",
    "low":      "🟢",
}
VERDICT_COLOUR = {"PASS": "🟢", "FAIL": "🔴"}


def show_reviewer(role: str, rule_store: RuleStore, config: dict) -> None:
    st.title("Compliance Evaluator")
    st.caption(f"Logged in as **{role}** reviewer")

    col1, col2 = st.columns([3, 1])
    with col1:
        transcript = st.text_area(
            "Paste conversation transcript",
            height=280,
            placeholder="Paste any format — free-form text, labelled turns, or raw call notes."
        )
    with col2:
        default_tid = f"t-{uuid.uuid4().hex[:8]}"
        transcript_id = st.text_input("Transcript ID", value=default_tid)
        st.markdown("---")
        st.markdown("**Models**")
        st.caption("Trigger: `haiku-4-5`")
        st.caption("Evaluator: `sonnet-4-6`")

    evaluate = st.button("Evaluate Compliance", type="primary", disabled=not transcript.strip())

    if evaluate and transcript.strip():
        with st.status("Running compliance evaluation...", expanded=True) as status:
            try:
                # Step 1: Trigger detection
                st.write("Detecting regulatory triggers...")
                detector = TriggerDetector()
                triggers = detector.detect(transcript)
                st.write(f"Triggers detected: {triggers if triggers else 'none'}")

                # Step 2: Rule retrieval
                st.write("Retrieving applicable rules...")
                retriever = CorrectiveRetriever(
                    rule_store=rule_store,
                    threshold=config["retrieval"]["disagreement_threshold"],
                    top_k=config["retrieval"]["semantic_top_k"],
                )
                rules, disagreed = retriever.retrieve(triggers, transcript, role)

                if not rules:
                    status.update(label="No rules found", state="error")
                    st.warning(
                        "No rules retrieved for this transcript. "
                        "Check that trigger labels are defined and rules have been seeded/added."
                    )
                    st.stop()

                st.write(f"Rules retrieved: {len(rules)} | Corrective disagreement: {'⚠️ yes' if disagreed else 'no'}")

                # Step 3: Streaming evaluation
                st.write("Evaluating with Sonnet 4.6 (streaming)...")
                evaluator = Evaluator()
                t0 = time.perf_counter()

                # Stream tokens live into a code block so the user sees output immediately
                stream_placeholder = st.empty()
                collected = []
                for chunk in evaluator.evaluate_stream(transcript, rules):
                    collected.append(chunk)
                    # Show last 300 chars of streamed JSON so it doesn't overflow
                    preview = "".join(collected)[-300:]
                    stream_placeholder.code(preview, language="json")

                stream_placeholder.empty()  # clear stream preview once done
                raw_content = "".join(collected)
                latency = round(time.perf_counter() - t0, 2)

                # Parse verdicts from collected stream
                try:
                    verdicts = evaluator._parse_content(raw_content)
                except Exception:
                    verdicts = []

                st.write(f"Done in {latency}s")
                status.update(label="Evaluation complete", state="complete")

            except Exception as e:
                status.update(label="Evaluation failed", state="error")
                st.error(f"Evaluation failed: {e}")
                st.stop()

        card = RegulatoryCard(
            transcript_id=transcript_id,
            verdicts=verdicts,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            model_used=config["models"]["evaluator_model"],
            latency_seconds=latency,
            corrective_disagreement=disagreed,
        )

        st.subheader("Regulatory Card")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rules Evaluated", len(verdicts))
        m2.metric("PASS", card.pass_count)
        m3.metric("FAIL", card.fail_count)
        m4.metric("Latency", f"{latency}s")

        if disagreed:
            st.warning("Corrective RAG: deterministic and semantic paths disagreed — union of both used.")

        for v in verdicts:
            sc = SEVERITY_COLOUR.get(v.severity, "")
            vc = VERDICT_COLOUR.get(v.verdict, "")
            with st.expander(f"{vc} {v.rule_id} — **{v.verdict}** {sc} {v.severity.upper()}"):
                st.markdown(f"**Reasoning:** {v.reasoning}")
                st.markdown(f"**Citation:** {v.citation}")

        log = AuditLog(config["audit"]["db_path"])
        log.append(AuditEntry(
            transcript_id=transcript_id,
            rules_evaluated=[r.id for r in rules],
            verdicts=verdicts,
            model_used=card.model_used,
            latency_seconds=card.latency_seconds,
            timestamp=card.evaluated_at,
            user_role=role,
            corrective_disagreement=disagreed,
        ))
        st.info(f"Evaluation saved to audit log (entry #{log.count()})")
