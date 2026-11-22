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
    if code == 429 or "ratelimit" in name or "rate limit" in head or "429" in head:
        return "The LLM rate limit was hit. Wait a minute and try again, or pick a smaller model in .env."
    if code in (401, 403) or "authentication" in name or "permissiondenied" in name or "invalid api key" in head:
        return "The LLM rejected the API key. Check GROQ_API_KEY in your .env file."
    if (isinstance(e, (ConnectionError, TimeoutError)) or "connection" in name or "timeout" in name
            or any(w in head for w in ("connection", "timed out", "network", "unreachable", "resolve"))):
        return "Could not reach the LLM service (network problem or timeout). Check your internet connection and retry."
    return "The LLM call failed. Please retry; if it keeps failing, check the Agent trace tab and the README troubleshooting table."


def _is_bad_json(e: Exception) -> bool:
    """Groq answers HTTP 400 json_validate_failed when the model's JSON does not fit the schema."""
    return getattr(e, "status_code", None) == 400 and "json" in str(e)[:300].lower()


def chat_model(model: Optional[str] = None, temperature: float = 0.3, max_tokens: int = config.LLM_MAX_TOKENS):
    """LangChain ChatGroq client (also reused by the RAGAS evaluator)."""
    from langchain_groq import ChatGroq

    key = config.groq_api_key()
    if not key:
        raise LLMError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    return ChatGroq(api_key=key, model=model or config.model_name(), temperature=temperature,
                    max_tokens=max_tokens, timeout=config.LLM_TIMEOUT, max_retries=config.LLM_ATTEMPTS)


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
    chain = prompt | chat_model(model, temperature).with_structured_output(schema, method="json_mode")
    text = user
    last_err: Exception | None = None
    for attempt in range(2):
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
                continue
            raise LLMError(explain_error(e)) from e
    raise LLMError(f"Model did not return valid JSON: {last_err}")
