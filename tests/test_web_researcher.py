"""
Web research: sources must actually be fetched, and evidence must be real text.

The researcher never researched. It took a `browser` argument and never used
it, issued no HTTP request, and returned `sources = ["heuristic"]`. The fields
named `supporting_yes` / `supporting_no` contained the question restated with
"supporting YES" in front of it. `confidence` was a literal 0.6 - or 0.65 if an
LLM answered - the same number whether ten credible sources were found or none.

Worst of all it split one block of LLM text by character offset:

    supporting_yes = text[:200]
    supporting_no  = text[200:400]

so the bear case was whatever happened to begin at character 200.
"""
import asyncio
from datetime import datetime, timezone

import pytest

from src.ptai.information.source_quality import SourceQualityEngine
from src.ptai.information.web_researcher import (
    FetchedSource, ResearchResult, WebResearcher,
)
from src.ptai.markets.base import Market, MarketSource

NOW = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)

REUTERS_URL = "https://www.reuters.com/markets/fed-rates"
AP_URL = "https://apnews.com/article/fed-rates"
BLOG_URL = "https://randomblog9.io/post"

REUTERS_HTML = (
    "<html><head><title>Fed</title><style>body{}</style></head><body>"
    "<nav>Home Markets Opinion</nav>"
    "<p>The Federal Reserve is expected to raise interest rates in June, according to "
    "officials who confirmed the plan was agreed by the committee. Markets have already "
    "priced in the move, traders said.</p>"
    "<script>var tracking = 'should not appear';</script>"
    "<footer>Copyright</footer></body></html>")

AP_HTML = (
    "<html><body><p>Federal Reserve officials denied any plan to raise interest rates and "
    "ruled out a June increase, saying the committee is unlikely to act before the autumn "
    "meeting according to two people familiar with the matter.</p></body></html>")

IRRELEVANT_HTML = (
    "<html><body><p>Local bakery announces new opening hours for the summer season and "
    "introduces a seasonal menu of pastries for customers visiting the town centre.</p>"
    "</body></html>")


@pytest.fixture
def market():
    return Market(id="m1", source=MarketSource.POLYMARKET,
                  question="Will the Federal Reserve raise interest rates in June?",
                  description="Resolves YES if the Fed raises the target rate.",
                  outcomes=["Yes", "No"], outcome_prices=[0.5, 0.5])


def researcher(pages, llm=None, browser=None):
    """A researcher whose HTTP layer is replaced by a canned page table."""
    def fetch(url):
        return pages.get(url, (0, ""))
    return WebResearcher(fetch=fetch, llm_router=llm, browser=browser,
                         source_engine=SourceQualityEngine(now=NOW))


GOOD_PAGES = {REUTERS_URL: (200, REUTERS_HTML), AP_URL: (200, AP_HTML)}


def run(coro):
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════════════════
# It actually fetches
# ═══════════════════════════════════════════════════════════════════════════

def test_sources_are_the_urls_actually_retrieved(market):
    """The old value was the literal list ['heuristic']."""
    res = run(researcher(GOOD_PAGES).research(
        market, candidate_urls=list(GOOD_PAGES)))
    assert res.sources == list(GOOD_PAGES)
    assert "heuristic" not in res.sources
    assert res.sources_retrieved == 2
    assert res.researched is True


def test_a_failed_fetch_is_reported_not_silently_dropped(market):
    pages = dict(GOOD_PAGES, **{BLOG_URL: (404, "")})
    res = run(researcher(pages).research(market, candidate_urls=list(pages)))
    assert res.sources_attempted == 3
    assert res.sources_retrieved == 2
    failed = [s for s in res.source_details if not s.ok]
    assert len(failed) == 1
    assert failed[0].url == BLOG_URL and "404" in failed[0].error


def test_nothing_retrievable_means_unresearched_not_confident(market):
    """
    The old code returned confidence 0.6 here, identical to a market where
    several credible sources were found.
    """
    res = run(researcher({}).research(market, candidate_urls=[BLOG_URL]))
    assert res.researched is False
    assert res.sources_retrieved == 0
    assert res.confidence <= 0.15
    assert res.provenance == "no_sources"
    assert any("not researched" in w or "unresearched" in w.lower()
               for w in res.warnings)


def test_evidence_fields_contain_retrieved_text_not_the_question(market):
    """
    The old bull case was `f"Polls, betting odds... supporting YES for {question}"`
    - the question restated, in a field the agent read as evidence.
    """
    res = run(researcher(GOOD_PAGES).research(market, candidate_urls=list(GOOD_PAGES)))
    assert "Federal Reserve" in res.supporting_yes
    assert res.supporting_yes != res.question
    assert "supporting YES for" not in res.supporting_yes


def test_evidence_is_attributed_to_the_page_it_came_from(market):
    res = run(researcher(GOOD_PAGES).research(market, candidate_urls=list(GOOD_PAGES)))
    assert "reuters.com" in res.supporting_yes
    assert "apnews.com" in res.supporting_no


def test_bull_and_bear_are_separated_by_what_the_pages_say(market):
    """Reuters leans yes, AP leans no; they must land on the correct side."""
    res = run(researcher(GOOD_PAGES).research(market, candidate_urls=list(GOOD_PAGES)))
    assert res.yes_evidence == 1 and res.no_evidence == 1
    assert "expected to raise" in res.supporting_yes
    assert "denied any plan" in res.supporting_no


def test_an_irrelevant_page_is_not_counted_as_evidence(market):
    pages = dict(GOOD_PAGES, **{"https://bakerynews.example/x": (200, IRRELEVANT_HTML)})
    res = run(researcher(pages).research(market, candidate_urls=list(pages)))
    irrelevant = [s for s in res.source_details if "bakery" in s.url][0]
    assert irrelevant.supports == "neither"
    assert res.yes_evidence + res.no_evidence == 2      # not 3


def test_a_missing_side_says_so_rather_than_implying_none_exists(market):
    res = run(researcher({REUTERS_URL: (200, REUTERS_HTML)}).research(
        market, candidate_urls=[REUTERS_URL]))
    assert res.no_evidence == 0
    assert res.supporting_no == "no retrieved source supports NO"


# ═══════════════════════════════════════════════════════════════════════════
# HTML extraction
# ═══════════════════════════════════════════════════════════════════════════

def test_scripts_styles_and_chrome_are_stripped():
    """
    Boilerplate left in the text would dominate the shingle comparison that
    novelty and corroboration depend on.
    """
    text = WebResearcher.extract_text(REUTERS_HTML)
    assert "tracking" not in text          # script
    assert "body{" not in text             # style
    assert "Home Markets Opinion" not in text   # nav
    assert "Copyright" not in text         # footer
    assert "Federal Reserve is expected to raise" in text


def test_extraction_handles_empty_and_tag_free_input():
    assert WebResearcher.extract_text("") == ""
    assert "plain words" in WebResearcher.extract_text("plain words")


def test_extraction_is_length_bounded():
    huge = "<p>" + ("word " * 5000) + "</p>"
    assert len(WebResearcher.extract_text(huge)) <= 4000


# ═══════════════════════════════════════════════════════════════════════════
# Section parsing - the character-offset bug
# ═══════════════════════════════════════════════════════════════════════════

def test_sections_are_parsed_by_marker_not_by_offset():
    text = ("1. Bull case FOR YES: Officials confirmed the plan.\n"
            "2. Bear case AGAINST YES: Officials denied it.\n"
            "3. Contradiction: The criterion may differ.\n"
            "4. Unknown: We don't know the threshold.\n"
            "5. Resolution risks: 'raise' vs 'announce'.")
    s = WebResearcher.parse_sections(text)
    assert s["supporting_yes"] == "Officials confirmed the plan."
    assert s["supporting_no"] == "Officials denied it."
    assert s["contradicting_both"] == "The criterion may differ."
    assert s["unknown"] == "We don't know the threshold."
    assert s["resolution_risks"] == "'raise' vs 'announce'."


def test_the_bear_case_is_never_a_slice_of_the_bull_case():
    """
    Regression guard on `text[:200]` / `text[200:400]`. A single-line answer
    with both cases must still separate them, where slicing would have cut
    the bear case out of the middle of the bull argument.
    """
    text = "Bull case: " + "A" * 300 + "\nBear case: " + "B" * 300
    s = WebResearcher.parse_sections(text)
    assert set(s["supporting_yes"]) == {"A"}
    assert set(s["supporting_no"]) == {"B"}


@pytest.mark.parametrize("text,expected", [
    ("Bull case: A\nBear case: B", ("A", "B")),
    ("**YES CASE** yes text\n**NO CASE** no text", ("yes text", "no text")),
    ("1) Supporting YES - alpha\n2) Supporting NO - beta", ("alpha", "beta")),
    ("BULL CASE FOR YES: a\nBEAR CASE AGAINST YES: b", ("a", "b")),
])
def test_label_formats_are_handled(text, expected):
    s = WebResearcher.parse_sections(text)
    assert s["supporting_yes"] == expected[0]
    assert s["supporting_no"] == expected[1]


def test_multiline_sections_are_joined():
    text = "Bull case: first line\nstill the bull case\nBear case: other"
    s = WebResearcher.parse_sections(text)
    assert "first line" in s["supporting_yes"]
    assert "still the bull case" in s["supporting_yes"]


def test_unlabelled_text_produces_no_sections():
    """
    Better to report the section missing than to fill it with neighbouring
    text, which is exactly what slicing did.
    """
    assert WebResearcher.parse_sections("Just prose with no labels at all.") == {}
    assert WebResearcher.parse_sections("") == {}


def test_a_repeated_marker_keeps_the_first_occurrence():
    text = "Bull case: first\nBear case: b\nBull case: second"
    assert WebResearcher.parse_sections(text)["supporting_yes"] == "first"


# ═══════════════════════════════════════════════════════════════════════════
# LLM synthesis
# ═══════════════════════════════════════════════════════════════════════════

class _Router:
    def __init__(self, text, raises=False):
        self.text, self.raises, self.prompts = text, raises, []

    async def generate(self, prompt, max_tokens=500):
        self.prompts.append(prompt)
        if self.raises:
            raise RuntimeError("router down")
        return self.text


def test_llm_sections_are_used_when_parsed(market):
    router = _Router("Bull case: Officials confirmed.\nBear case: Officials denied.\n"
                     "Contradiction: Criterion differs.\nUnknown: Threshold unclear.\n"
                     "Resolution risks: Ambiguous wording.")
    res = run(researcher(GOOD_PAGES, llm=router).research(
        market, candidate_urls=list(GOOD_PAGES)))
    assert res.used_llm is True
    assert res.supporting_no == "no retrieved source supports NO" or "denied" in res.supporting_no
    assert res.unknown == "Threshold unclear."
    assert res.resolution_risks == "Ambiguous wording."
    assert res.missing_sections == []


def test_the_llm_only_sees_retrieved_text(market):
    """The prompt must be grounded in what was fetched, not asked from nothing."""
    router = _Router("Bull case: x\nBear case: y")
    run(researcher(GOOD_PAGES, llm=router).research(market, candidate_urls=list(GOOD_PAGES)))
    assert len(router.prompts) == 1
    assert "Federal Reserve is expected to raise" in router.prompts[0]
    assert REUTERS_URL in router.prompts[0] or "reuters.com" in router.prompts[0]


def test_the_llm_is_not_called_when_nothing_was_retrieved(market):
    router = _Router("Bull case: x\nBear case: y")
    res = run(researcher({}, llm=router).research(market, candidate_urls=[BLOG_URL]))
    assert router.prompts == []
    assert res.used_llm is False
    assert "unresearched" in res.final_summary.lower()


def test_missing_llm_sections_are_reported_not_backfilled(market):
    router = _Router("Bull case: only this section exists.")
    res = run(researcher(GOOD_PAGES, llm=router).research(
        market, candidate_urls=list(GOOD_PAGES)))
    assert "supporting_no" in res.missing_sections
    assert "resolution_risks" in res.missing_sections
    # and the field says it was not established rather than borrowing text
    assert "not checked" in res.resolution_risks or "no source" in res.resolution_risks


def test_a_failing_router_falls_back_to_the_fetched_evidence(market):
    router = _Router("", raises=True)
    res = run(researcher(GOOD_PAGES, llm=router).research(
        market, candidate_urls=list(GOOD_PAGES)))
    assert res.used_llm is False
    assert res.researched is True          # the research still happened
    assert "source(s) retrieved" in res.final_summary


def test_a_router_without_generate_is_handled(market):
    class Bare:
        pass
    res = run(researcher(GOOD_PAGES, llm=Bare()).research(
        market, candidate_urls=list(GOOD_PAGES)))
    assert res.used_llm is False
    assert res.researched is True


# ═══════════════════════════════════════════════════════════════════════════
# Confidence
# ═══════════════════════════════════════════════════════════════════════════

def test_confidence_rises_with_more_and_better_sources(market):
    one = run(researcher({REUTERS_URL: (200, REUTERS_HTML)}).research(
        market, candidate_urls=[REUTERS_URL]))
    two = run(researcher(GOOD_PAGES).research(market, candidate_urls=list(GOOD_PAGES)))
    assert two.confidence > one.confidence


def test_confidence_is_not_a_constant(market):
    """The old value was a literal 0.6 whatever was found."""
    none = run(researcher({}).research(market, candidate_urls=[BLOG_URL]))
    some = run(researcher(GOOD_PAGES).research(market, candidate_urls=list(GOOD_PAGES)))
    assert none.confidence != some.confidence
    assert none.confidence < 0.2 < some.confidence


def test_one_sided_retrieval_scores_below_balanced(market):
    """
    Every source leaning one way usually means the search missed the other
    side, so it is weaker evidence than genuine disagreement.
    """
    yes_only = {REUTERS_URL: (200, REUTERS_HTML),
                "https://anotherwire.com/x": (200, REUTERS_HTML.replace("reuters", "wire"))}
    balanced = GOOD_PAGES
    a = run(researcher(yes_only).research(market, candidate_urls=list(yes_only)))
    b = run(researcher(balanced).research(market, candidate_urls=list(balanced)))
    assert a.no_evidence == 0
    assert b.confidence > a.confidence


def test_confidence_is_bounded(market):
    many = {f"https://w{i}.example/a": (200, REUTERS_HTML) for i in range(12)}
    res = run(researcher(many).research(market, candidate_urls=list(many)))
    assert 0.0 <= res.confidence <= 0.95


# ═══════════════════════════════════════════════════════════════════════════
# Source scoring integration
# ═══════════════════════════════════════════════════════════════════════════

def test_each_source_is_scored_for_quality(market):
    res = run(researcher(GOOD_PAGES).research(market, candidate_urls=list(GOOD_PAGES)))
    assert res.avg_source_quality > 0
    for s in res.source_details:
        if s.ok:
            assert s.assessment is not None
            assert s.assessment.tier in ("wire_service", "unknown")


def test_a_credible_source_outranks_an_unknown_one(market):
    pages = {REUTERS_URL: (200, REUTERS_HTML), BLOG_URL: (200, AP_HTML)}
    res = run(researcher(pages).research(market, candidate_urls=list(pages)))
    by_host = {s.url: s for s in res.source_details}
    assert by_host[REUTERS_URL].quality > by_host[BLOG_URL].quality


# ═══════════════════════════════════════════════════════════════════════════
# Discovery and browser
# ═══════════════════════════════════════════════════════════════════════════

def test_discovery_extracts_and_dedupes_urls():
    html = ('<a href="/l/?uddg=https%3A%2F%2Freuters.com%2Fa&rut=x">1</a>'
            '<a href="/l/?uddg=https%3A%2F%2Fapnews.com%2Fb&rut=x">2</a>'
            '<a href="/l/?uddg=https%3A%2F%2Freuters.com%2Fa&rut=x">3</a>'
            '<a href="/l/?uddg=https%3A%2F%2Fduckduckgo.com%2Fy">4</a>')
    r = WebResearcher(fetch=lambda u: (200, html))
    found = run(r.discover_sources("fed rates"))
    assert found == ["https://reuters.com/a", "https://apnews.com/b"]


def test_discovery_failure_returns_no_urls_not_invented_ones():
    r = WebResearcher(fetch=lambda u: (0, ""))
    assert run(r.discover_sources("fed rates")) == []


def test_discovery_respects_max_sources():
    links = "".join(f'<a href="/l/?uddg=https%3A%2F%2Fs{i}.com%2Fa">x</a>' for i in range(20))
    r = WebResearcher(fetch=lambda u: (200, links), max_sources=4)
    assert len(run(r.discover_sources("q"))) == 4


def test_a_supplied_browser_is_used_for_fetching(market):
    """The browser argument used to be accepted and ignored entirely."""
    calls = []

    class Browser:
        def get_page(self, url):
            calls.append(url)
            return (200, REUTERS_HTML) if "reuters" in url else (404, "")

    r = WebResearcher(browser=Browser(), source_engine=SourceQualityEngine(now=NOW))
    res = run(r.research(market, candidate_urls=[REUTERS_URL]))
    assert calls == [REUTERS_URL]
    assert res.sources_retrieved == 1
    assert r.get_report()["browser_used"] is True


def test_the_injected_fetch_takes_precedence_over_the_browser(market):
    class Browser:
        def get_page(self, url):
            raise AssertionError("browser should not be called")

    r = WebResearcher(browser=Browser(), fetch=lambda u: (200, REUTERS_HTML),
                      source_engine=SourceQualityEngine(now=NOW))
    assert run(r.research(market, candidate_urls=[REUTERS_URL])).sources_retrieved == 1


def test_time_budget_stops_the_scan(market):
    pages = {f"https://s{i}.example/a": (200, REUTERS_HTML) for i in range(50)}
    res = run(researcher(pages).research(market, max_time_seconds=0,
                                         candidate_urls=list(pages)))
    assert res.sources_attempted < 50


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def test_report_lists_the_removed_fabrications():
    report = WebResearcher().get_report()
    assert len(report["removed_fabrication"]) == 5
    assert any("browser" in f for f in report["removed_fabrication"])
    assert any("0.6" in f for f in report["removed_fabrication"])
    assert any("text[:200]" in f for f in report["removed_fabrication"])


def test_fetched_source_defaults_are_inert():
    fs = FetchedSource(url="https://x.example/a")
    assert fs.ok is False and fs.quality == 0.0 and fs.supports == "unknown"


def test_result_dataclass_defaults_to_unresearched():
    r = ResearchResult(market_id="m", question="q", supporting_yes="", supporting_no="",
                       contradicting_both="", unknown="", resolution_risks="",
                       final_summary="", sources=[], confidence=0.0,
                       time_spent_seconds=0.0)
    assert r.researched is False and r.sources_retrieved == 0
