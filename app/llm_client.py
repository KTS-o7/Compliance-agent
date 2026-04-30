from __future__ import annotations
import yaml
import os
from openai import OpenAI

# LLM_BACKEND controls which inference stack is used:
#   ollama — OpenAI client → Ollama /v1 directly (default)
#            Ollama 0.19+ is natively powered by MLX on Apple Silicon
#   cloud  — OpenAI client → bifrost → AWS Bedrock
BACKENDS = ("ollama", "cloud")

_OLLAMA_MODEL_DEFAULTS = {
    "trigger":   "qwen3.5:9b",
    "evaluator": "qwen3.5:9b",
}


def _load_config() -> dict:
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_backend() -> str:
    return os.environ.get("LLM_BACKEND", "ollama").lower()


def get_client() -> OpenAI:
    """OpenAI-compatible client for both backends."""
    backend = get_backend()
    if backend == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return OpenAI(base_url=f"{host}/v1", api_key="ollama")
    # cloud — route through bifrost
    cfg = _load_config()
    base_url = os.environ.get("BIFROST_URL", cfg["bifrost"]["base_url"])
    api_key = cfg["bifrost"].get("api_key", "dummy")
    return OpenAI(base_url=base_url, api_key=api_key)


def get_models() -> dict[str, str]:
    backend = get_backend()
    if backend == "ollama":
        return {
            "trigger":   os.environ.get("OLLAMA_TRIGGER_MODEL",   _OLLAMA_MODEL_DEFAULTS["trigger"]),
            "evaluator": os.environ.get("OLLAMA_EVALUATOR_MODEL", _OLLAMA_MODEL_DEFAULTS["evaluator"]),
        }
    # cloud — read from config.yaml
    cfg = _load_config()
    return {
        "trigger":   cfg["models"]["trigger_model"],
        "evaluator": cfg["models"]["evaluator_model"],
    }


def use_structured_output() -> bool:
    """
    Returns True when the backend supports OpenAI beta structured output.
    - cloud:  True  (bifrost → Bedrock supports response_format JSON schema)
    - ollama: False (use plain JSON + manual parse)
    Can be overridden via USE_STRUCTURED_OUTPUT env var.
    """
    backend = get_backend()
    default = "true" if backend == "cloud" else "false"
    val = os.environ.get("USE_STRUCTURED_OUTPUT", default).lower()
    return val not in ("false", "0", "no")


def get_no_think_kwargs() -> dict:
    """
    Disable thinking for Ollama thinking models via OpenAI-compatible API.
    Per https://docs.ollama.com/api/openai-compatibility:
      reasoning_effort: "none" disables thinking on /v1/chat/completions.
    Only applied for ollama backend.
    """
    if get_backend() == "ollama":
        return {"extra_body": {"reasoning_effort": "none"}}
    return {}
