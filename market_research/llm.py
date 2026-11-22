"""Thin wrapper around the Groq API: JSON-mode calls validated against Pydantic schemas."""
from __future__ import annotations

import json
from typing import Optional, TypeVar

from pydantic import BaseModel, ValidationError

from . import config
from .tracing import track

T = TypeVar("T", bound=BaseModel)

# Running totals of LLM usage, read by the workflow to attribute cost to each step.
_usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}


def usage_snapshot() -> dict[str, int]:
    return dict(_usage)


def _record_usage(handler) -> None:
    _usage["calls"] += 1
    for meta in (handler.usage_metadata or {}).values():
        _usage["input_tokens"] += int(meta.get("input_tokens", 0))
        _usage["output_tokens"] += int(meta.get("output_tokens", 0))


class LLMError(Exception):
    """Raised for problems the user can fix (missing key, bad output, rate limit)."""


def explain_error(e: Exception) -> str:
    """Turn any exception from an LLM or search call into one short message a student can act on.

    Classifies by exception type and HTTP status first. Only the start of the message is read as a fallback,
    because a failed call can echo the model's own output, which may contain words like "network".
    """
    name = type(e).__name__.lower()
    code = getattr(e, "status_code", None)
    head = str(e)[:80].lower()
    if code == 413 or "request too large" in head:
        return ("The request was too large for Groq's per-minute token limit (free tier: 8000 tokens). Retry in a "
                "minute, lower the gap-filling rounds, or raise GROQ_TPM_LIMIT in .env if your plan allows more.")
    if code == 429 or "ratelimit" in name or "rate limit" in head or "429" in head:
        return ("The LLM rate limit was hit on every Groq key. Wait a minute and try again, add more keys "
                "(GROQ_API_KEY_1, GROQ_API_KEY_2, ...) in .env, or pick a smaller model.")
    if code in (401, 403) or "authentication" in name or "permissiondenied" in name or "invalid api key" in head:
        return "The LLM rejected the API key. Check GROQ_API_KEY (and any GROQ_API_KEY_1, _2, ...) in your .env file."
    if (isinstance(e, (ConnectionError, TimeoutError)) or "connection" in name or "timeout" in name
            or any(w in head for w in ("connection", "timed out", "network", "unreachable", "resolve"))):
        return "Could not reach the LLM service (network problem or timeout). Check your internet connection and retry."
    return "The LLM call failed. Please retry; if it keeps failing, check the Agent trace tab and the README troubleshooting table."


def estimate_tokens(text: str) -> int:
    """Rough token count (about 3.5 characters per token for English and JSON). Errs on the high side."""
    return int(len(text) / 3.5) + 1


def output_budget(prompt_chars: int) -> int:
    """max_tokens that keeps prompt + answer under Groq's per-request TPM cap, within sane bounds.

    Groq rejects a request (HTTP 413) when prompt tokens plus max_tokens exceed the cap, even if the
    answer would have been short, so the answer budget shrinks as the prompt grows.
    """
    cap = config.groq_tpm_limit()
    if cap <= 0:
        return config.LLM_MAX_TOKENS
    room = cap - int(prompt_chars / 3.5) - 150  # 150 tokens of headroom for chat formatting
    return max(config.LLM_MIN_OUTPUT_TOKENS, min(config.LLM_MAX_TOKENS, room))


def _is_bad_json(e: Exception) -> bool:
    """Groq answers HTTP 400 json_validate_failed when the model's JSON does not fit the schema."""
    return getattr(e, "status_code", None) == 400 and "json" in str(e)[:300].lower()


# Index into config.groq_api_keys() of the key in use. Moves forward when a key is exhausted or rejected,
# so later calls start with the key that last worked.
_key_index = 0


def _should_rotate(e: Exception) -> bool:
    """Rate limit (429, includes daily token quota) or a rejected key (401/403): another key may work."""
    code = getattr(e, "status_code", None)
    name = type(e).__name__.lower()
    head = str(e)[:80].lower()
    return (code in (401, 403, 429) or "ratelimit" in name or "authentication" in name
            or "permissiondenied" in name or "rate limit" in head or "429" in head or "invalid api key" in head)


def _rotate_key() -> bool:
    """Switch to the next Groq key. False when every key has already been tried from this one."""
    global _key_index
    keys = config.groq_api_keys()
    if len(keys) < 2:
        return False
    _key_index = (_key_index + 1) % len(keys)
    return True


def current_key() -> str:
    keys = config.groq_api_keys()
    if not keys:
        raise LLMError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    return keys[_key_index % len(keys)]


def chat_model(model: Optional[str] = None, temperature: float = 0.3, max_tokens: int = config.LLM_MAX_TOKENS):
    """LangChain ChatGroq client using the active key (also reused by the RAGAS evaluator)."""
    from langchain_groq import ChatGroq

    key = current_key()
    # With spare keys, fail over fast instead of letting the client back off on a rate-limited key.
    retries = 1 if len(config.groq_api_keys()) > 1 else config.LLM_ATTEMPTS
    return ChatGroq(api_key=key, model=model or config.model_name(), temperature=temperature,
                    max_tokens=max_tokens, timeout=config.LLM_TIMEOUT, max_retries=retries)


@track("llm_call", "llm")
def ask_json(schema: type[T], system: str, user: str, model: Optional[str] = None,
             temperature: float = 0.3) -> T:
    """LangChain chain (prompt | ChatGroq structured output) validated against a Pydantic schema (one retry)."""
    from langchain_core.callbacks import UsageMetadataCallbackHandler
    from langchain_core.prompts import ChatPromptTemplate

    system_full = (
        system
        + "\n\nRespond with ONLY a single JSON object matching this JSON schema. "
        "No prose, no markdown fences.\n"
        + json.dumps(schema.model_json_schema())
    )
    prompt = ChatPromptTemplate.from_messages([("system", "{system}"), ("human", "{user}")])
    budget = output_budget(len(system_full) + len(user) + 400)  # +400: room for the retry note

    def build_chain():
        return prompt | chat_model(model, temperature, max_tokens=budget).with_structured_output(
            schema, method="json_mode")

    chain = build_chain()
    keys_left = len(config.groq_api_keys()) - 1  # other keys to fail over to
    text = user
    last_err: Exception | None = None
    attempt = 0
    while attempt < 2:
        handler = UsageMetadataCallbackHandler()
        try:
            try:
                result = chain.invoke({"system": system_full, "user": text}, config={"callbacks": [handler]})
            finally:
                _record_usage(handler)
            if isinstance(result, schema):
                return result
            return schema.model_validate(result)
        except LLMError:
            raise
        except (ValueError, ValidationError) as e:  # includes langchain OutputParserException
            last_err = e
            text = user + f"\n\nYour previous answer was invalid ({e}). Return ONLY valid JSON matching the schema."
        except Exception as e:  # network, auth, rate limit (after retries), bad model name...
            if _is_bad_json(e) and attempt == 0:  # same one retry as for invalid JSON
                last_err = e
                text = user + "\n\nYour previous answer was not valid JSON for the schema. Return ONLY valid JSON."
                attempt += 1
                continue
            if keys_left > 0 and _should_rotate(e) and _rotate_key():
                keys_left -= 1
                chain = build_chain()
                continue  # same attempt, next key
            raise LLMError(explain_error(e)) from e
        attempt += 1
    raise LLMError(f"Model did not return valid JSON: {last_err}")
