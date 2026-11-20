"""Tests for the deterministic evaluation metrics (no API keys needed)."""
from market_research.evaluation.rules import citation_validity, retrieval_check, structure_check
from market_research.evaluation.tamper import tamper
from market_research.models import Brief, Competitor, Finding, Research, Source, WorkflowState
from market_research.urls import normalize_url, snap_urls

SRCS = [
    Source(title="A", url="https://www.a.com/x/?utm=1", snippet="alpha " * 30),
    Source(title="B", url="https://b.org/y", snippet="beta " * 30),
    Source(title="C", url="https://c.net/z", snippet="gamma " * 30),
]


def make_state(brief: Brief, mode="web") -> WorkflowState:
    r = Research(mode=mode, findings=[Finding(claim="f")], sources=SRCS if mode == "web" else [])
    return WorkflowState(query="q", research=r, final=brief)


def good_brief() -> Brief:
    return Brief(
        summary="s", competitors=[Competitor(name="X", note="n")],
        key_observations=["a", "b", "c"], sources=[Source(title="A", url="https://a.com/x")],
        missing_information=["market size"],
    )


def test_normalize_url():
    assert normalize_url("https://www.A.com/x/?utm=1#f") == normalize_url("http://a.com/x")


def test_retrieval_check_pass_and_sample_fail():
    assert retrieval_check(make_state(good_brief())).passed
    assert not retrieval_check(make_state(good_brief(), mode="sample")).passed


def test_citation_valid_and_fake():
    st = make_state(good_brief())
    assert citation_validity(st).passed
    assert not citation_validity(st, tamper(good_brief())).passed


def test_structure():
    assert structure_check(make_state(good_brief())).passed
    bad = good_brief()
    bad.summary = ""
    assert not structure_check(make_state(bad)).passed


def test_competitors_capped_at_two():
    b = Brief(summary="s", competitors=[Competitor(name=str(i), note="n") for i in range(3)],
              key_observations=["a", "b", "c"])
    assert len(b.competitors) == 2


def test_snap_urls_repairs_mangled_but_not_fake():
    real = "https://www.facebook.com/naimanepal9/posts/as-nepals-mobility-sector-continues-to-evolve-how-can-the-country-address-the-re/122209008176864248"
    mangled = "https://www.facebook.com/naimanepal9/posts/as-nepals-mobility-sector-continues-to-evolve-how-can-the-re/122209008176864248"
    fake = "https://made-up-research.example/report-2024"
    items = [Source(title="a", url=mangled), Source(title="b", url=fake)]
    snap_urls(items, [Source(title="r", url=real)])
    assert items[0].url == real and items[1].url == fake
