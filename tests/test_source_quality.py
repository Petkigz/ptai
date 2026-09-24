"""
Source quality: every dimension must be computed, not assumed.

The module used to return `novelty = 0.6  # placeholder` and `recency = 0.7`
for every document, and decided corroboration with

    corroborated = len(other_sources) >= 2

which never compared any content - three sources on unrelated topics counted
as corroborating each other and added 0.14 to the quality score. Its
credibility lookup substring-matched the entire URL, so
`http://scam.example/?ref=reuters.com` scored as Reuters.
"""
from datetime import datetime, timedelta, timezone

import pytest

from src.ptai.information.source_quality import (
    CORROBORATION_SIMILARITY, NOVELTY_DERIVATIVE_THRESHOLD, NO_TIMESTAMP_RECENCY,
    SourceAssessment, SourceDocument, SourceQualityEngine,
)

NOW = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)

STORY = ("The Federal Reserve raised interest rates by 25 basis points on Wednesday "
         "citing persistent inflation concerns across the broader economy")
OTHER = "Shipping insurers are raising premiums for vessels transiting the Red Sea corridor"


@pytest.fixture
def engine():
    return SourceQualityEngine(now=NOW)


# ═══════════════════════════════════════════════════════════════════════════
# Domain parsing - the substring-match bug
# ═══════════════════════════════════════════════════════════════════════════

def test_domain_comes_from_the_host_not_a_url_substring():
    """The old match let ?ref=reuters.com score as Reuters."""
    d = SourceQualityEngine.registrable_domain
    assert d("http://scam.example/?ref=reuters.com") == "scam.example"
    assert d("https://reuters.com/fake-ref") == "reuters.com"
    assert d("https://notreuters.com.evil.io/x") == "evil.io"


def test_domain_handles_www_and_subdomains():
    d = SourceQualityEngine.registrable_domain
    assert d("https://www.reuters.com/world/x") == "reuters.com"
    assert d("https://sub.domain.bbc.co.uk/a") == "bbc.co.uk"
    assert d("reuters.com/path") == "reuters.com"


def test_domain_handles_multi_part_suffixes():
    d = SourceQualityEngine.registrable_domain
    assert d("https://news.co.uk/story") == "news.co.uk"
    assert d("https://www.abc.com.au/news") == "abc.com.au"
    assert d("https://bbc.co.uk/x") == "bbc.co.uk"


def test_free_text_is_not_a_domain():
    d = SourceQualityEngine.registrable_domain
    assert d("not a url") == ""
    assert d("") == ""
    assert d("http://localhost:8000/x") == ""       # no TLD
    assert d("just some words about markets") == ""


# ═══════════════════════════════════════════════════════════════════════════
# Credibility
# ═══════════════════════════════════════════════════════════════════════════

def test_known_tiers_are_recognised(engine):
    assert engine.credibility_for("https://reuters.com/a")[:2] == (0.95, "wire_service")
    assert engine.credibility_for("https://sec.gov/x")[:2] == (0.90, "government")
    assert engine.credibility_for("https://x.com/y")[:2] == (0.40, "social")
    assert engine.credibility_for("https://medium.com/@someone")[1] == "user_generated"


def test_unknown_domain_is_low_and_explained_not_a_confident_half(engine):
    """
    0.5 reads as "average, measured". Unknown means unmeasured, and a caller
    has to be able to tell those apart.
    """
    score, tier, why = engine.credibility_for("https://randomblog12345.io/post")
    assert tier == "unknown"
    assert score < 0.5
    assert "not in any credibility tier" in why


def test_a_spoofed_ref_parameter_does_not_inherit_reuters_trust(engine):
    score, tier, _ = engine.credibility_for("http://scam.example/?ref=reuters.com")
    assert tier == "unknown"
    assert score < 0.5


def test_an_unparseable_url_scores_zero_not_low(engine):
    score, tier, why = engine.credibility_for("not a url at all")
    assert tier == "invalid"
    assert score == 0.0
    assert "could not parse" in why


def test_deployment_can_add_its_own_trusted_domains():
    e = SourceQualityEngine(now=NOW, extra_credibility={"internal.example": 0.99})
    assert e.credibility_for("https://internal.example/x")[:2] == (0.99, "configured")
    # and it does not leak to a lookalike
    assert e.credibility_for("https://notinternal.example/x")[1] == "unknown"


def test_subdomain_inherits_the_parent_tier(engine):
    assert engine.credibility_for("https://www.bbc.co.uk/news")[1] == "public_broadcaster"


# ═══════════════════════════════════════════════════════════════════════════
# Novelty
# ═══════════════════════════════════════════════════════════════════════════

def test_first_document_is_fully_novel(engine):
    a = engine.assess("https://reuters.com/a", STORY, published_at=NOW)
    assert a.novelty == 1.0
    assert a.max_similarity == 0.0


def test_a_verbatim_copy_has_no_novelty(engine):
    """
    The flat 0.6 could not distinguish original reporting from the tenth
    syndicated copy, which is the only thing a novelty score is for.
    """
    engine.add_document("https://reuters.com/a", STORY)
    copy = engine.assess("https://randomblog1.io/p", STORY, published_at=NOW)
    assert copy.novelty == 0.0
    assert copy.max_similarity == pytest.approx(1.0)
    assert any("syndication" in w for w in copy.warnings)


def test_an_unrelated_story_stays_novel(engine):
    engine.add_document("https://reuters.com/a", STORY)
    a = engine.assess("https://ft.com/b", OTHER, published_at=NOW)
    assert a.novelty > 0.9
    assert a.max_similarity < NOVELTY_DERIVATIVE_THRESHOLD


def test_a_light_edit_of_the_same_story_is_caught_as_syndication(engine):
    """
    What syndication actually looks like: a couple of words swapped, or an
    attribution line bolted on the front. Both must score as derivatives.
    """
    engine.add_document("https://reuters.com/a", STORY)
    edited = engine.assess(
        "https://blog2.io/x",
        STORY.replace("raised", "lifted").replace("broader", "wider"),
        published_at=NOW)
    assert 0.0 < edited.novelty < 0.7
    assert edited.max_similarity > 0.4

    prefixed = engine.assess("https://blog3.io/y", "WASHINGTON (Reuters) - " + STORY,
                             published_at=NOW)
    assert prefixed.novelty < 0.25
    assert any("syndication" in w for w in prefixed.warnings)


def test_a_genuine_rewrite_is_treated_as_original_reporting(engine):
    """
    Novelty targets syndication, not paraphrase. A story told in different
    words is original reporting and must not be penalised for covering the
    same facts - otherwise the engine would punish independent journalism.
    """
    engine.add_document("https://reuters.com/a", STORY)
    rewrite = engine.assess(
        "https://blog2.io/x",
        "The Federal Reserve lifted rates 25 basis points Wednesday on inflation worries",
        published_at=NOW)
    assert rewrite.novelty > 0.9


def test_novelty_ignores_empty_corpus_documents(engine):
    engine.add_document("https://reuters.com/a", "")
    a = engine.assess("https://ft.com/b", STORY, published_at=NOW)
    assert a.novelty == 1.0


def test_empty_content_is_not_novel(engine):
    a = engine.assess("https://reuters.com/a", "", published_at=NOW)
    assert a.novelty == 0.0
    assert any("no content" in w for w in a.warnings)


# ═══════════════════════════════════════════════════════════════════════════
# Corroboration - the counting bug
# ═══════════════════════════════════════════════════════════════════════════

def test_corroboration_requires_shared_content(engine):
    engine.add_document("https://reuters.com/a", STORY)
    engine.add_document("https://apnews.com/b", STORY)
    a = engine.assess("https://reuters.com/a", STORY, published_at=NOW)
    assert a.corroborated is True
    assert a.corroborating_sources == ["https://apnews.com/b"]


def test_a_bare_url_list_is_not_corroboration(engine):
    """
    The exact bug: `len(other_sources) >= 2` counted URLs without ever
    comparing what they said.
    """
    a = engine.assess("https://newsite999.org/z", "A claim about a corporate merger",
                      other_sources=["https://a.com", "https://b.com"],
                      published_at=NOW)
    assert a.corroborated is False
    assert any("URL list is not corroboration" in w for w in a.warnings)


def test_unrelated_sources_do_not_corroborate(engine):
    engine.add_document("https://reuters.com/a", STORY)
    engine.add_document("https://apnews.com/b", OTHER)
    a = engine.assess("https://reuters.com/a", STORY, published_at=NOW)
    assert a.corroborated is False
    assert a.corroborating_sources == []


def test_same_domain_pages_are_not_two_witnesses(engine):
    """Two pages from one outlet are one source, not two."""
    engine.add_document("https://bbc.co.uk/a", STORY)
    engine.add_document("https://bbc.co.uk/b", STORY)
    a = engine.assess("https://bbc.co.uk/a", STORY, published_at=NOW)
    assert a.corroborated is False


def test_three_independent_outlets_do_corroborate(engine):
    engine.add_document("https://reuters.com/a", STORY)
    engine.add_document("https://apnews.com/b", STORY)
    engine.add_document("https://ft.com/c", STORY)
    a = engine.assess("https://reuters.com/a", STORY, published_at=NOW)
    assert a.corroborated is True
    assert len(a.corroborating_sources) == 2


# ═══════════════════════════════════════════════════════════════════════════
# Recency
# ═══════════════════════════════════════════════════════════════════════════

def test_recency_decays_from_a_real_timestamp(engine):
    fresh, _, age = engine.recency_of(NOW)
    day_old, _, _ = engine.recency_of(NOW - timedelta(hours=24))
    week_old, _, _ = engine.recency_of(NOW - timedelta(hours=168))
    assert fresh == pytest.approx(1.0)
    assert day_old == pytest.approx(0.5, abs=1e-3)   # 24h half-life
    assert week_old < 0.01
    assert age == 0.0


def test_recency_monotonically_decreases(engine):
    scores = [engine.recency_of(NOW - timedelta(hours=h))[0]
              for h in (0, 1, 6, 24, 72, 240)]
    assert scores == sorted(scores, reverse=True)


def test_a_missing_timestamp_is_stale_and_flagged_not_assumed_fresh(engine):
    """The old code returned 0.7 for every document regardless of date."""
    recency, has_ts, age = engine.recency_of(None)
    assert recency == NO_TIMESTAMP_RECENCY
    assert recency < 0.5
    assert has_ts is False and age is None
    a = engine.assess("https://reuters.com/a", STORY)
    assert a.has_timestamp is False
    assert any("no publication timestamp" in w for w in a.warnings)


def test_a_naive_timestamp_is_treated_as_utc(engine):
    naive = datetime(2026, 9, 23, 12, 0)      # no tzinfo
    recency, has_ts, age = engine.recency_of(naive)
    assert has_ts is True and recency == pytest.approx(1.0)


def test_a_future_timestamp_does_not_exceed_one(engine):
    recency, _, age = engine.recency_of(NOW + timedelta(hours=5))
    assert recency <= 1.0 and age == 0.0


def test_half_life_is_configurable():
    fast = SourceQualityEngine(now=NOW, half_life_hours=1.0)
    slow = SourceQualityEngine(now=NOW, half_life_hours=240.0)
    ts = NOW - timedelta(hours=1)
    assert fast.recency_of(ts)[0] < slow.recency_of(ts)[0]


# ═══════════════════════════════════════════════════════════════════════════
# Overall score
# ═══════════════════════════════════════════════════════════════════════════

def test_overall_is_in_unit_range(engine):
    engine.add_document("https://reuters.com/a", STORY)
    engine.add_document("https://apnews.com/b", STORY)
    a = engine.assess("https://reuters.com/a", STORY, published_at=NOW)
    assert 0.0 <= a.overall_quality <= 1.0


def test_a_corroborated_wire_service_outranks_an_unsourced_blog(engine):
    engine.add_document("https://reuters.com/a", STORY)
    engine.add_document("https://apnews.com/b", STORY)
    wire = engine.assess("https://reuters.com/a", STORY, published_at=NOW)
    blog = engine.assess("https://randomblog1.io/p", STORY, published_at=NOW - timedelta(days=9))
    assert wire.overall_quality > blog.overall_quality + 0.2


def test_stale_content_scores_below_fresh_content(engine):
    fresh = engine.assess("https://reuters.com/a", STORY, published_at=NOW)
    engine2 = SourceQualityEngine(now=NOW)
    stale = engine2.assess("https://reuters.com/a", STORY,
                           published_at=NOW - timedelta(days=10))
    assert fresh.overall_quality > stale.overall_quality


def test_custom_weights_are_normalised():
    e = SourceQualityEngine(now=NOW, weights={"credibility": 2.0, "novelty": 1.0,
                                              "corroboration": 1.0, "recency": 0.0})
    assert sum(e.weights.values()) == pytest.approx(1.0)
    assert e.weights["credibility"] == pytest.approx(0.5)
    assert e.weights["recency"] == 0.0


def test_a_zero_recency_weight_makes_age_irrelevant():
    """Proof the weights are applied rather than ignored."""
    stale = NOW - timedelta(days=30)

    # same engine, same URL assessed at two ages: with recency weighted to
    # zero the score must not move
    ignore_age = SourceQualityEngine(now=NOW, weights={"credibility": 1.0, "novelty": 1.0,
                                                       "corroboration": 1.0, "recency": 0.0})
    a = ignore_age.assess("https://reuters.com/a", STORY, published_at=stale)
    b = ignore_age.assess("https://reuters.com/a", STORY, published_at=NOW)
    assert a.overall_quality == pytest.approx(b.overall_quality)
    assert a.recency != b.recency, "recency itself must still be computed"

    # with the default weights age matters. Separate engines, because a second
    # document on the same engine is correctly flagged as a near-duplicate of
    # the first and would lose novelty for an unrelated reason.
    stale_engine = SourceQualityEngine(now=NOW)
    fresh_engine = SourceQualityEngine(now=NOW)
    s1 = stale_engine.assess("https://reuters.com/a", STORY, published_at=stale)
    s2 = fresh_engine.assess("https://reuters.com/a", STORY, published_at=NOW)
    assert s1.overall_quality < s2.overall_quality


def test_credibility_weight_dominates_when_weighted_heavily():
    heavy = SourceQualityEngine(now=NOW, weights={"credibility": 10.0, "novelty": 0.0,
                                                  "corroboration": 0.0, "recency": 0.0})
    wire = heavy.assess("https://reuters.com/a", STORY, published_at=NOW)
    blog = heavy.assess("https://randomblog1.io/p", STORY, published_at=NOW)
    assert wire.overall_quality > blog.overall_quality + 0.5


def test_invalid_weights_are_rejected():
    with pytest.raises(ValueError):
        SourceQualityEngine(weights={"credibility": 0.0, "novelty": 0.0,
                                     "corroboration": 0.0, "recency": 0.0})


def test_a_document_does_not_compare_against_itself(engine):
    """Registering then assessing the same URL must not report self-similarity."""
    engine.add_document("https://reuters.com/a", STORY)
    a = engine.assess("https://reuters.com/a", STORY, published_at=NOW)
    assert a.novelty == 1.0
    assert a.max_similarity == 0.0


def test_repeated_assessment_is_idempotent(engine):
    engine.add_document("https://reuters.com/a", STORY)
    first = engine.assess("https://reuters.com/a", STORY, published_at=NOW)
    second = engine.assess("https://reuters.com/a", STORY, published_at=NOW)
    assert first.overall_quality == second.overall_quality
    assert len(engine.documents) == 1


def test_assessment_reports_its_inputs(engine):
    """Without these a caller cannot tell which dimension carries the score."""
    a = engine.assess("https://reuters.com/a", STORY, published_at=NOW - timedelta(hours=3))
    assert a.domain == "reuters.com" and a.tier == "wire_service"
    assert a.content_words > 0
    assert a.age_hours == pytest.approx(3.0)
    assert isinstance(a.warnings, list)


def test_is_usable_requires_content_and_a_known_domain(engine):
    assert engine.assess("https://reuters.com/a", STORY, published_at=NOW).is_usable
    assert not engine.assess("https://reuters.com/a", "", published_at=NOW).is_usable
    assert not engine.assess("https://randomblog9.io/p", STORY,
                             published_at=NOW).is_usable


# ═══════════════════════════════════════════════════════════════════════════
# Ranking and reporting
# ═══════════════════════════════════════════════════════════════════════════

def test_rank_orders_best_first():
    e = SourceQualityEngine(now=NOW)
    ranked = e.rank([
        ("https://randomblog1.io/p", STORY),
        ("https://reuters.com/a", STORY),
        ("https://apnews.com/b", STORY),
    ], published={u: NOW for u in ("https://randomblog1.io/p", "https://reuters.com/a",
                                   "https://apnews.com/b")})
    assert len(ranked) == 3
    assert ranked[0].tier in ("wire_service",)
    assert ranked[-1].tier == "unknown"
    scores = [r.overall_quality for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_registers_the_whole_corpus_before_scoring():
    """Novelty must see all documents, not just the earlier ones."""
    e = SourceQualityEngine(now=NOW)
    ranked = e.rank([("https://reuters.com/a", STORY),
                     ("https://blog1.io/p", STORY)],
                    published={"https://reuters.com/a": NOW, "https://blog1.io/p": NOW})
    by_url = {r.url: r for r in ranked}
    assert by_url["https://blog1.io/p"].novelty == 0.0


def test_report_documents_what_was_removed(engine):
    report = engine.get_report()
    assert len(report["removed_placeholders"]) == 4
    assert any("0.6" in p for p in report["removed_placeholders"])
    assert any("0.7" in p for p in report["removed_placeholders"])
    assert set(report["weights"]) == {"credibility", "novelty", "corroboration", "recency"}


def test_corpus_grows_as_documents_are_added(engine):
    assert engine.get_report()["corpus_size"] == 0
    engine.add_document("https://reuters.com/a", STORY)
    assert engine.get_report()["corpus_size"] == 1
    assert isinstance(engine.documents[0], SourceDocument)
    assert engine.documents[0].tier == "wire_service"


def test_similarity_is_symmetric_and_bounded(engine):
    assert engine.similarity(STORY, STORY) == pytest.approx(1.0)
    assert engine.similarity(STORY, OTHER) < 0.2
    assert engine.similarity("", STORY) == 0.0
    assert engine.similarity(STORY, OTHER) == engine.similarity(OTHER, STORY)
