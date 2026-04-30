from __future__ import annotations
import json
import streamlit as st
from app.audit.audit_log import AuditLog
from app.models.schemas import Verdict

VERDICT_COLOUR = {"PASS": "🟢", "FAIL": "🔴"}
SEVERITY_COLOUR = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}


def show_audit(config: dict) -> None:
    st.title("Audit Log")
    st.caption("Append-only record of all evaluations (senior access only)")

    log = AuditLog(config["audit"]["db_path"])
    entries = log.read_all()

    if not entries:
        st.info("No evaluations recorded yet.")
        return

    st.metric("Total evaluations", len(entries))
    st.markdown("---")

    for entry in reversed(entries):
        verdicts = [Verdict(**v) for v in json.loads(entry["verdicts"])]
        rules_evaluated = json.loads(entry["rules_evaluated"])
        pass_count = sum(1 for v in verdicts if v.verdict == "PASS")
        fail_count = sum(1 for v in verdicts if v.verdict == "FAIL")
        disagreed = bool(entry.get("corrective_disagreement"))

        header = (
            f"**{entry['transcript_id']}** | "
            f"{entry['timestamp'][:19].replace('T', ' ')} | "
            f"Role: {entry['user_role']} | "
            f"🟢 {pass_count} PASS  🔴 {fail_count} FAIL | "
            f"{entry['latency_seconds']:.2f}s"
            + (" | ⚠️ corrective disagreement" if disagreed else "")
        )
        with st.expander(header):
            st.markdown(f"**Model:** `{entry['model_used']}`")
            st.markdown(f"**Rules evaluated:** {', '.join(rules_evaluated)}")
            st.markdown("**Verdicts:**")
            for v in verdicts:
                vc = VERDICT_COLOUR.get(v.verdict, "")
                sc = SEVERITY_COLOUR.get(v.severity, "")
                st.markdown(
                    f"- {vc} **{v.rule_id}** — {v.verdict} {sc} `{v.severity}`  \n"
                    f"  *{v.reasoning}*  \n"
                    f"  Citation: {v.citation}"
                )
