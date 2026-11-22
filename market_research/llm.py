"""Thin wrapper around the Groq API: JSON-mode calls validated against Pydantic schemas."""
from __future__ import annotations

import json
from typing import Optional, Type, TypeVar

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


def chat_model(model: Optional[str] = None, temperature: float = 0.3, max_tokens: int = config.LLM_MAX_TOKENS):
    """LangChain ChatGroq client (also reused by the RAGAS evaluator)."""
    from langchain_groq import ChatGroq

    key = config.groq_api_key()
    if not key:
        raise LLMError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    return ChatGroq(api_key=key, model=model or config.model_name(), temperature=temperature,
                    max_tokens=max_tokens, timeout=config.LLM_TIMEOUT, max_retries=config.LLM_ATTEMPTS)


@track("llm_call", "llm")
def ask_json(schema: Type[T], system: str, user: str, model: Optional[str] = None,
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
    for _ in range(2):
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
            if "429" in str(e) or "rate" in str(e).lower():
                raise LLMError(f"Groq rate limit hit repeatedly, try again in a minute: {e}") from e
            raise LLMError(f"Groq API call failed: {e}") from e
    raise LLMError(f"Model did not return valid JSON: {last_err}")
