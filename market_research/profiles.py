"""Static writing context: structure / terminology / character profiles and few-shot example briefs.

Profiles say *how* to write; the guideline says *what* to write. Both live in markdown so you can edit
them without touching code. Missing files are skipped, so the app still runs without them.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from . import config

PROFILE_NAMES = ("structure", "terminology", "character")


def _dir() -> Path:
    return Path(config.PROFILES_DIR)


@lru_cache(maxsize=8)
def _read(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8").strip() if p.is_file() else ""


def load_profiles() -> dict[str, str]:
    out = {name: _read(str(_dir() / f"{name}.md")) for name in PROFILE_NAMES}
    if out["terminology"]:
        out["terminology"] += "\n\nBanned phrases: " + ", ".join(f"'{p}'" for p in config.BANNED_PHRASES)
    return {k: v for k, v in out.items() if v}


def load_examples() -> list[str]:
    """Few-shot example briefs. Keep the number low: they are sent with every Analyst call."""
    d = _dir() / "examples"
    files = sorted(d.glob("*.md"))[: config.MAX_FEW_SHOT] if d.is_dir() else []
    return [_read(str(f)) for f in files]


def writing_context(include_examples: bool = True) -> str:
    """Profiles (and optionally examples) as one prompt block."""
    parts = [f"=== {name.upper()} PROFILE ===\n{text}" for name, text in load_profiles().items()]
    if include_examples:
        for n, ex in enumerate(load_examples(), 1):
            parts.append(f"=== EXAMPLE BRIEF {n} (style only, its facts are fictional) ===\n{ex}")
    return "\n\n".join(parts)
