"""Configuration of the local OpenAI-compatible LLM server (e.g. vLLM), read from environment variables.

    LOCAL_LLM_URL       full base url of the server, e.g. http://localhost:19000/v1  (or https://host/v1)
    LOCAL_LLM_KEY       api key sent to the server; default "EMPTY" (the openai client rejects an empty key,
                        vLLM ignores the key unless it was started with --api-key)
    LOCAL_LLM_MODEL_ID  served model name; the `--model-id` command line option takes precedence over it

Backward compatible: if LOCAL_LLM_URL is not set, the old LOCAL_LLM_PORT (-> http://localhost:<port>/v1) is used.
"""

import os

DEFAULT_LOCAL_LLM_URL = "http://localhost:8000/v1"
DEFAULT_LOCAL_LLM_KEY = "EMPTY"


def get_local_llm_url() -> str | None:
    """Base url of the local LLM server, or None if neither LOCAL_LLM_URL nor LOCAL_LLM_PORT is configured."""
    url = os.getenv("LOCAL_LLM_URL", "").strip()
    if url:
        return url.rstrip("/")
    port = os.getenv("LOCAL_LLM_PORT", "").strip()
    if port and int(port) > 0:
        return f"http://localhost:{port}/v1"
    return None


def is_local_llm_configured() -> bool:
    return get_local_llm_url() is not None


def get_local_llm_url_or_default() -> str:
    return get_local_llm_url() or DEFAULT_LOCAL_LLM_URL


def get_local_llm_key() -> str:
    return os.getenv("LOCAL_LLM_KEY", "").strip() or DEFAULT_LOCAL_LLM_KEY


def get_local_llm_model_id(model_id: str | None = None) -> str | None:
    """`model_id` (from --model-id) wins over LOCAL_LLM_MODEL_ID."""
    return model_id or os.getenv("LOCAL_LLM_MODEL_ID", "").strip() or None


def fetch_local_llm_model_id() -> str:
    """Ask the server (GET <url>/models) which model it serves; used when no model id was given."""
    import requests

    response = requests.get(
        f"{get_local_llm_url_or_default()}/models",
        headers={"Authorization": f"Bearer {get_local_llm_key()}"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["data"][0]["id"]
