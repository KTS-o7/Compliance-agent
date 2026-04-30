from __future__ import annotations
import streamlit as st
from app.models.schemas import Rule
from app.ingestion.rule_store import RuleStore
from app.detection.labels import TRIGGER_LABELS


def show_admin(rule_store: RuleStore) -> None:
    st.title("Rule Library — Admin")
    st.caption(f"Total rules in store: **{rule_store.count()}**")

    with st.form("add_rule_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            rule_id  = st.text_input("Rule ID *", placeholder="e.g. FDCPA-809")
            citation = st.text_input("Citation *", placeholder="e.g. 15 U.S.C. § 1692g")
            severity = st.selectbox("Severity *", ["critical", "high", "medium", "low"])
        with col2:
            role_tag = st.selectbox("Role Tag *", ["all", "junior", "senior"])
            triggers = st.multiselect("Applicable Trigger Labels *", options=TRIGGER_LABELS)

        agent_must     = st.text_area("Agent Must (one per line) *",
                                      placeholder="provide written validation notice\ndocument call")
        agent_must_not = st.text_area("Agent Must NOT (one per line)",
                                      placeholder="ignore dispute\ncontinue collection after dispute")
        text           = st.text_area("Full Rule Text (will be embedded) *",
                                      placeholder="Upon receiving a written dispute...")

        submitted = st.form_submit_button("Add Rule to Library", type="primary")

    if submitted:
        if not all([rule_id, citation, triggers, text, agent_must]):
            st.error("Please fill in all required fields (*).")
            return
        rule = Rule(
            id=rule_id.strip(),
            citation=citation.strip(),
            severity=severity,
            agent_must=[l.strip() for l in agent_must.splitlines() if l.strip()],
            agent_must_not=[l.strip() for l in agent_must_not.splitlines() if l.strip()],
            role_tag=role_tag,
            trigger_labels=triggers,
            text=text.strip(),
        )
        rule_store.add_rule(rule)
        st.success(f"Rule **{rule_id}** added. Total: {rule_store.count()}")

    # --- Rule Library Browser ---
    st.markdown("---")
    st.subheader("Existing Rules")

    SEVERITY_COLOUR = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
    ROLE_ICON = {"all": "👥", "junior": "🔵", "senior": "🟣"}

    all_rules = rule_store.get_all_rules()
    if not all_rules:
        st.info("No rules in library yet.")
    else:
        for rule in sorted(all_rules, key=lambda r: r.id):
            sc = SEVERITY_COLOUR.get(rule.severity, "")
            rc = ROLE_ICON.get(rule.role_tag, "")
            with st.expander(f"{sc} **{rule.id}** — {rule.citation} {rc} `{rule.role_tag}`"):
                st.markdown(f"**Severity:** {rule.severity.upper()}")
                st.markdown(f"**Triggers:** {', '.join(rule.trigger_labels)}")
                st.markdown(f"**Agent Must:**")
                for item in rule.agent_must:
                    st.markdown(f"  - {item}")
                if rule.agent_must_not:
                    st.markdown(f"**Agent Must NOT:**")
                    for item in rule.agent_must_not:
                        st.markdown(f"  - {item}")
                st.markdown(f"**Rule Text:** {rule.text}")
