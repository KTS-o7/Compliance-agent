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


def _render_verdict(verdict) -> None:
    sc = SEVERITY_COLOUR.get(verdict.severity.lower(), "")
    vc = VERDICT_COLOUR.get(verdict.verdict, "")
    with st.expander(
        f"{vc} {verdict.rule_id} — **{verdict.verdict}** {sc} {verdict.severity.upper()}",
        expanded=True,
    ):
        st.markdown(f"**Reasoning:** {verdict.reasoning}")
        st.markdown(f"**Citation:** {verdict.citation}")


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
        trigger_name = config["models"]["trigger_model"].split("/")[-1]
        evaluator_name = config["models"]["evaluator_model"].split("/")[-1]
        st.caption(f"Trigger: `{trigger_name}`")
        st.caption(f"Evaluator: `{evaluator_name}`")

    evaluate = st.button("Evaluate Compliance", type="primary", disabled=not transcript.strip())

    if evaluate and transcript.strip():
        all_verdicts = []
        rules = []
        disagreed = False
        latency = 0.0

        # Everything runs inside the status — cards stream live here
        with st.status("Running evaluation...", expanded=True) as status:
            try:
                # Step 1: triggers
                st.write("Detecting regulatory triggers...")
                detector = TriggerDetector()
                triggers = detector.detect(transcript)
                st.write(f"Triggers: {triggers if triggers else 'none'}")

                # Step 2: rules
                st.write("Retrieving applicable rules...")
                retriever = CorrectiveRetriever(
                    rule_store=rule_store,
                    threshold=config["retrieval"]["disagreement_threshold"],
                    top_k=config["retrieval"]["semantic_top_k"],
                )
                rules, disagreed = retriever.retrieve(triggers, transcript, role)

                if not rules:
                    status.update(label="No rules found", state="error")
                    st.warning("No rules retrieved for this transcript.")
                    st.stop()

                st.write(f"Rules retrieved: {len(rules)} | Disagreement: {'⚠️ yes' if disagreed else 'no'}")

                if disagreed:
                    st.warning("Corrective RAG: deterministic and semantic paths disagreed — union used.")

                # Step 3: stream verdicts — cards appear live inside status
                st.markdown("**Evaluating compliance...**")
                evaluator = Evaluator()
                t0 = time.perf_counter()

                for verdict in evaluator.evaluate_stream_verdicts(transcript, rules):
                    all_verdicts.append(verdict)
                    _render_verdict(verdict)

                latency = round(time.perf_counter() - t0, 2)
                status.update(label=f"Evaluation complete — {len(all_verdicts)} rules in {latency}s", state="complete")

            except Exception as e:
                status.update(label="Evaluation failed", state="error")
                st.error(f"Error: {e}")
                st.stop()

        # ── Persistent section below (visible after status collapses) ──
        card = RegulatoryCard(
            transcript_id=transcript_id,
            verdicts=all_verdicts,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            model_used=config["models"]["evaluator_model"],
            latency_seconds=latency,
            corrective_disagreement=disagreed,
        )

        st.subheader("Regulatory Card")

        if disagreed:
            st.warning("Corrective RAG: deterministic and semantic paths disagreed — union of both used.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rules Evaluated", len(all_verdicts))
        m2.metric("PASS", card.pass_count)
        m3.metric("FAIL", card.fail_count)
        m4.metric("Latency", f"{latency}s")

        st.markdown("---")

        for v in all_verdicts:
            _render_verdict(v)

        # Audit log
        log = AuditLog(config["audit"]["db_path"])
        log.append(AuditEntry(
            transcript_id=transcript_id,
            rules_evaluated=[r.id for r in rules],
            verdicts=all_verdicts,
            model_used=card.model_used,
            latency_seconds=card.latency_seconds,
            timestamp=card.evaluated_at,
            user_role=role,
            corrective_disagreement=disagreed,
        ))
        st.info(f"Evaluation saved to audit log (entry #{log.count()})")
