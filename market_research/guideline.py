"""The human-written guideline: what the brief should cover (the 'what'; profiles are the 'how').

A guideline is a small markdown file with fixed headings, see `guidelines/TEMPLATE.md`:

    # <topic>
    ## Angle
    ## Audience
    ## Key questions      (bullets; each becomes a search query)
    ## Must include       (bullets)
    ## Seed links         (bullets with URLs the Researcher reads first)
    ## Constraints        (bullets; override the default profiles)

Only the title is required. The more specific the guideline, the better the brief.
"""
from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

_SECTIONS = {
    "angle": "angle", "audience": "audience", "key questions": "key_questions",
    "must include": "must_include", "seed links": "seed_links", "constraints": "constraints",
}
_LISTS = {"key_questions", "must_include", "seed_links", "constraints"}
_URL = re.compile(r"https?://[^\s)>\]]+")


class Guideline(BaseModel):
    topic: str
    angle: str = ""
    audience: str = ""
    key_questions: list[str] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    seed_links: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)

    def to_prompt(self) -> str:
        """Compact text block for the Analyst / Critic prompts. Empty sections are left out."""
        parts = [f"Topic: {self.topic}"]
        if self.angle:
            parts.append(f"Angle: {self.angle}")
        if self.audience:
            parts.append(f"Audience: {self.audience}")
        for title, items in (("Key questions to answer", self.key_questions),
                             ("Must include", self.must_include),
                             ("Constraints (override the profiles)", self.constraints)):
            if items:
                parts.append(title + ":\n" + "\n".join(f"- {i}" for i in items))
        return "\n".join(parts)


def parse_guideline(text: str) -> Guideline:
    topic, current, data = "", None, {}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("<!--") or not line:
            continue
        if line.startswith("# ") and not topic:
            topic, current = line[2:].strip(), None
        elif line.startswith("## "):
            current = _SECTIONS.get(line[3:].strip().lower())
        elif current:
            item = re.sub(r"^([-*]|\d+[.)])(\s+|$)", "", line).strip()
            if not item:  # empty template bullet
                continue
            if current in _LISTS:
                if current == "seed_links":
                    data.setdefault(current, []).extend(_URL.findall(item))
                elif line[:1] in "-*" or re.match(r"\d+[.)]", line):
                    data.setdefault(current, []).append(item)
                elif data.get(current):  # continuation of the previous bullet
                    data[current][-1] += " " + item
                else:
                    data.setdefault(current, []).append(item)
            else:
                data[current] = (data.get(current, "") + " " + item).strip()
    if not topic:
        raise ValueError("The guideline needs a '# <topic>' title line.")
    return Guideline(topic=topic, **data)


def load_guideline(path: str | Path) -> Guideline:
    return parse_guideline(Path(path).read_text(encoding="utf-8"))
