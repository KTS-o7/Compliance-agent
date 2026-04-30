from __future__ import annotations
import streamlit as st
import yaml
import os

st.set_page_config(
    page_title="Compliance Evaluator",
    page_icon="⚖️",
    layout="wide",
)


@st.cache_resource
def load_config() -> dict:
    cfg_path = os.environ.get("CONFIG_PATH", "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if os.environ.get("QDRANT_URL"):
        cfg["qdrant"]["url"] = os.environ["QDRANT_URL"]
    if os.environ.get("BIFROST_URL"):
        cfg["bifrost"]["base_url"] = os.environ["BIFROST_URL"]
    return cfg


def load_rule_store(qdrant_url: str):
    if "rule_store" not in st.session_state:
        from app.ingestion.rule_store import RuleStore, prefetch_embedder
        from app.ingestion.seed_rules import seed
        store = RuleStore(qdrant_url=qdrant_url)
        seed(store)
        prefetch_embedder()
        st.session_state["rule_store"] = store
    return st.session_state["rule_store"]


def login_page() -> None:
    st.title("⚖️ Compliance Evaluator")
    st.subheader("Login")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary")
    if submitted:
        from app.auth import authenticate
        role = authenticate(username, password)
        if role:
            st.session_state["role"] = role
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid username or password.")


def main() -> None:
    if "role" not in st.session_state:
        login_page()
        return

    config = load_config()
    rule_store = load_rule_store(config["qdrant"]["url"])
    role = st.session_state["role"]

    from app.auth import can_access_admin
    pages = ["Reviewer"]
    if can_access_admin(role):
        pages.extend(["Admin", "Audit Log"])

    with st.sidebar:
        st.markdown(f"**User:** {st.session_state['username']} ({role})")
        st.markdown("---")
        page = st.radio("Page", pages)
        st.markdown("---")
        if st.button("Logout"):
            del st.session_state["role"]
            del st.session_state["username"]
            st.rerun()

    if page == "Admin":
        from app.ui.admin import show_admin
        show_admin(rule_store)
    elif page == "Audit Log":
        from app.ui.audit import show_audit
        show_audit(config)
    else:
        from app.ui.reviewer import show_reviewer
        show_reviewer(role=role, rule_store=rule_store, config=config)


main()
