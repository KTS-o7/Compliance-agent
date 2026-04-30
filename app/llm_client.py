from __future__ import annotations
import yaml
import os
from openai import OpenAI


def _load_config() -> dict:
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_client() -> OpenAI:
    cfg = _load_config()
    base_url = os.environ.get("BIFROST_URL", cfg["bifrost"]["base_url"])
    api_key = cfg["bifrost"].get("api_key", "bifrost")
    return OpenAI(base_url=base_url, api_key=api_key)


def get_models() -> dict[str, str]:
    cfg = _load_config()
    return {
        "trigger": cfg["models"]["trigger_model"],
        "evaluator": cfg["models"]["evaluator_model"],
    }
