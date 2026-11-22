"""Keep every test offline: no real keys, no intake LLM call, default provider."""
import pytest


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("GROQ_API_KEY", "TAVILY_API_KEY", "OPIK_API_KEY", "OPIK_URL_OVERRIDE", "USE_SAMPLE_DATA"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("INTAKE_CHECK", "0")
