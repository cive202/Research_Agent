"""Keep every test offline: no real keys, no intake LLM call, default provider."""
import os

import pytest


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("GROQ_API_KEY", "TAVILY_API_KEY", "OPIK_API_KEY", "OPIK_URL_OVERRIDE", "USE_SAMPLE_DATA"):
        monkeypatch.setenv(name, "")
    for name in [n for n in os.environ if n.startswith("GROQ_API_KEY_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("INTAKE_CHECK", "0")
    from market_research import llm
    monkeypatch.setattr(llm, "_key_index", 0)
