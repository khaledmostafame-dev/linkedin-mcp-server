"""Tests for free-text LinkedIn location resolution."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.geo_cache import CachedGeoResolution, GeoResolutionCache
from linkedin_mcp_server.scraping import geo_resolver as geo_resolver_module
from linkedin_mcp_server.scraping.contracts import FilterValidationError
from linkedin_mcp_server.scraping.geo_resolver import (
    GeoCandidate,
    GeoLocationResolver,
    GeoResolution,
    _find_exact_match,
    _normalize,
    _shorter_prefix,
)
from linkedin_mcp_server.scraping import session as session_module
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession


class _FakeJobsSearchPage:
    """A minimal double for the jobs-search page's location combobox.

    Modeled on what ``geo_resolver.py`` actually reads and writes through
    ``page.evaluate``/``page.wait_for_function``, identified by comparing
    against the module's own JS constants (never by call order), so a test
    can drive the resolver through several fills, clicks and reopens on one
    page exactly the way the refactored resolver does:

    * ``suggestions_by_query`` maps a query string (as the resolver would
      pass it, after its own ``.strip()``) to the labels LinkedIn's
      typeahead renders for it. A query missing from the map renders none.
    * ``geo_id_by_label`` maps a rendered label to the numeric id LinkedIn
      would put in the url once that suggestion is clicked. A label missing
      from the map is clickable but never carries an id, the same as a
      click LinkedIn's own page failed to complete.
    * ``stall_once_for`` names queries whose *first* fill renders no
      suggestions, so the listbox-not-populated-yet retry inside
      ``_fill_and_list`` is what makes the real (second) suggestions show.
    * ``lose_after_clicks`` forces the page to report itself "lost" (see
      ``_PAGE_LOST_JS``) starting right after that many suggestions have
      been clicked, until the next navigation -- simulating a client-side
      redirect mid-resolution.

    ``navigations`` counts every call the resolver made through
    ``PageNavigator._navigate_to_page`` (wired to :meth:`on_navigate` by
    ``_resolver`` below), which is what proves how many real page loads a
    resolution cost.
    """

    def __init__(
        self,
        suggestions_by_query: dict[str, list[str]] | None = None,
        geo_id_by_label: dict[str, str] | None = None,
        *,
        stall_once_for: set[str] | None = None,
        lose_after_clicks: int | None = None,
    ) -> None:
        self._suggestions_by_query = suggestions_by_query or {}
        self._geo_id_by_label = geo_id_by_label or {}
        self._stall_once_for = stall_once_for or set()
        self._stalled_already: set[str] = set()
        self._lose_after_clicks = lose_after_clicks
        self._clicks = 0
        self._forced_lost = False

        self.navigations = 0
        self._on_jobs_search = False
        self._current_labels: list[str] = []
        self._current_geo_id: str | None = None

    async def on_navigate(self, url: str) -> None:
        self.navigations += 1
        self._on_jobs_search = "/jobs/search/" in url
        self._current_labels = []
        self._current_geo_id = None
        self._forced_lost = False

    async def evaluate(self, script: str, arg: object = None) -> object:
        if script == geo_resolver_module._FILL_LOCATION_BOX_JS:
            query = arg
            if query in self._stall_once_for and query not in self._stalled_already:
                self._stalled_already.add(query)
                self._current_labels = []
            else:
                self._current_labels = list(self._suggestions_by_query.get(query, []))
            return True
        if script == geo_resolver_module._READ_SUGGESTION_LABELS_JS:
            return list(self._current_labels)
        if script == geo_resolver_module._CLICK_SUGGESTION_AT_INDEX_JS:
            index = arg
            if (
                not isinstance(index, int)
                or index < 0
                or index >= len(self._current_labels)
            ):
                return False
            label = self._current_labels[index]
            self._clicks += 1
            geo_id = self._geo_id_by_label.get(label)
            if geo_id is not None:
                self._current_geo_id = geo_id
            if (
                self._lose_after_clicks is not None
                and self._clicks == self._lose_after_clicks
            ):
                self._forced_lost = True
            return True
        if script == geo_resolver_module._URL_GEO_ID_JS:
            return self._current_geo_id
        if script == geo_resolver_module._PAGE_LOST_JS:
            return self._forced_lost or not self._on_jobs_search
        raise AssertionError(f"unexpected evaluate() script: {script!r}")

    async def wait_for_function(
        self, script: str, arg: object = None, timeout: float | None = None
    ) -> object:
        if script == geo_resolver_module._LISTBOX_HAS_OPTIONS_JS:
            if not self._current_labels:
                raise PlaywrightTimeoutError("listbox never populated")
            return True
        if script == geo_resolver_module._GEO_ID_CHANGED_JS:
            if self._current_geo_id is None or self._current_geo_id == arg:
                raise PlaywrightTimeoutError("geoId never changed")
            return True
        raise AssertionError(f"unexpected wait_for_function() script: {script!r}")


class _NullGeoCache:
    """Always misses and never remembers.

    Keeps every behavioral test in this file about the typeahead-driving
    logic only; caching itself is covered separately by
    :class:`TestGeoResolutionCache` and :class:`TestResolveWithCache`.
    """

    def get(self, key: str) -> CachedGeoResolution | None:
        return None

    def put(self, key: str, *, name: str, geo_urn_id: str) -> None:
        pass


class _RecordingCache:
    """An in-memory stand-in for the real cache, keyed exactly as
    :meth:`GeoLocationResolver.resolve` calls it (normalized query text --
    see its own use of ``_normalize``), recording every write so a test can
    assert exactly when one happened.
    """

    def __init__(self) -> None:
        self.puts: list[tuple[str, str, str]] = []
        self.store: dict[str, CachedGeoResolution] = {}

    def get(self, key: str) -> CachedGeoResolution | None:
        return self.store.get(key)

    def put(self, key: str, *, name: str, geo_urn_id: str) -> None:
        self.puts.append((key, name, geo_urn_id))
        self.store[key] = CachedGeoResolution(
            name=name, geo_urn_id=geo_urn_id, resolved_at=0.0
        )


def _resolver(
    page: _FakeJobsSearchPage, monkeypatch, cache: object | None = None
) -> GeoLocationResolver:
    """Wire a resolver against a fake jobs-search page.

    ``ScrapingSession`` is a frozen dataclass, so the module-level function
    it delegates rate-limit checks to is patched instead of an instance
    attribute. Navigation is wired to the fake page's own
    :meth:`_FakeJobsSearchPage.on_navigate` (rather than a bare no-op),
    which is what lets the fake reset its listbox/url state -- and count
    navigations -- exactly when the resolver actually navigates.
    """
    session = ScrapingSession(page)
    navigator = PageNavigator(session)
    resolver = GeoLocationResolver(session, navigator, cache=cache or _NullGeoCache())
    monkeypatch.setattr(
        navigator, "_navigate_to_page", AsyncMock(side_effect=page.on_navigate)
    )
    monkeypatch.setattr(session_module, "detect_rate_limit", AsyncMock())
    monkeypatch.setattr(geo_resolver_module, "NAV_DELAY", 0.0)
    return resolver


class TestResolve:
    async def test_blank_query_is_refused_before_any_navigation(self, monkeypatch):
        page = _FakeJobsSearchPage()
        resolver = _resolver(page, monkeypatch)

        with pytest.raises(FilterValidationError, match="must not be blank"):
            await resolver.resolve("   ")

        assert page.navigations == 0

    async def test_no_suggestions_is_no_match(self, monkeypatch):
        """Empty on the direct query *and* on the one shorter-prefix retry.

        "Nowhereville" is one word, so the retry is the first half of it
        ("Nowher") -- confirming the no-match path still exhausts the retry
        before giving up, rather than reporting no-match too early. Also
        proves the retry reuses the page already open: the whole path costs
        exactly one navigation, not two.
        """
        page = _FakeJobsSearchPage()  # nothing is ever suggested
        resolver = _resolver(page, monkeypatch)

        result = await resolver.resolve("Nowhereville")

        assert result == GeoResolution()
        assert result.is_no_match
        assert not result.is_ambiguous
        assert page.navigations == 1

    async def test_a_stalled_dropdown_is_no_match_not_an_error(self, monkeypatch):
        """A listbox that never appears reads as unknown, not as a failure."""
        page = _FakeJobsSearchPage()
        resolver = _resolver(page, monkeypatch)

        result = await resolver.resolve("Atlantis")

        assert result.is_no_match

    async def test_a_single_suggestion_resolves_silently(self, monkeypatch):
        page = _FakeJobsSearchPage(
            suggestions_by_query={"Dubai": ["Dubai, United Arab Emirates"]},
            geo_id_by_label={"Dubai, United Arab Emirates": "104246948"},
        )
        resolver = _resolver(page, monkeypatch)

        result = await resolver.resolve("Dubai")

        assert result.resolved == GeoCandidate(
            "Dubai, United Arab Emirates", "104246948"
        )
        assert result.candidates == ()
        assert not result.is_ambiguous
        assert not result.is_no_match
        assert not result.cached
        assert page.navigations == 1

    async def test_multiple_suggestions_are_all_resolved_as_candidates(
        self, monkeypatch
    ):
        """Ambiguous: every candidate's id is independently confirmed.

        Mutation guard: swap either resolved id for the other's and this
        fails, so picking by position rather than re-verifying each
        selection cannot pass silently. "Georgia" also exercises the exact-
        match disambiguation added alongside this test: the country name
        equals the query outright *and* "Georgia, United States" is
        simultaneously the query's canonical form, so two candidates match
        and the query correctly stays ambiguous rather than picking either.
        The whole exchange -- both selections -- happens on one page: this
        is the case that used to cost three navigations and now costs one.
        """
        labels = ["Georgia", "Georgia, United States"]
        page = _FakeJobsSearchPage(
            suggestions_by_query={"Georgia": labels},
            geo_id_by_label={
                "Georgia": "101452733",
                "Georgia, United States": "104081876",
            },
        )
        resolver = _resolver(page, monkeypatch)

        result = await resolver.resolve("Georgia")

        assert result.resolved is None
        assert result.is_ambiguous
        assert result.candidates == (
            GeoCandidate("Georgia", "101452733"),
            GeoCandidate("Georgia, United States", "104081876"),
        )
        assert page.navigations == 1

    async def test_a_candidate_is_dropped_when_the_reopened_order_shifts(
        self, monkeypatch
    ):
        """Never trusts a re-selected index without re-checking its label.

        If the suggestion order is not stable across an identical query, the
        candidate at that position is skipped rather than mislabeled with
        the wrong id. Only "Georgia" survives re-verification here, and it
        is an exact match for the query, so it comes back ``resolved``
        rather than as a one-item ambiguous list -- the mutation guard
        (swap its id for the dropped candidate's) still fails either way.
        """
        page = _FakeJobsSearchPage(
            suggestions_by_query={"Georgia": ["Georgia", "Georgia, United States"]},
            geo_id_by_label={
                "Georgia": "101452733",
                "Georgia, United States": "104081876",
            },
        )
        # The fake's simple query->labels map cannot express "the same query
        # renders different labels the second time", so this drives the
        # sequence directly against the page double for the one reopen that
        # matters here.
        original_evaluate = page.evaluate
        reopened = {"done": False}

        async def evaluate_with_shifted_reopen(script, arg=None):
            if (
                script == geo_resolver_module._FILL_LOCATION_BOX_JS
                and arg == "Georgia"
                and page._current_geo_id is not None
                and not reopened["done"]
            ):
                reopened["done"] = True
                page._current_labels = ["Georgia", "Something Else"]
                return True
            return await original_evaluate(script, arg)

        resolver = _resolver(page, monkeypatch)
        monkeypatch.setattr(page, "evaluate", evaluate_with_shifted_reopen)

        result = await resolver.resolve("Georgia")

        assert result.resolved == GeoCandidate("Georgia", "101452733")
        assert result.candidates == ()
        assert not result.is_ambiguous
        assert page.navigations == 1

    async def test_a_click_that_never_carries_a_geo_id_is_dropped(self, monkeypatch):
        page = _FakeJobsSearchPage(
            suggestions_by_query={"Dubai": ["Dubai, United Arab Emirates"]},
            geo_id_by_label={},  # LinkedIn never puts a geoId in the url
        )
        resolver = _resolver(page, monkeypatch)

        result = await resolver.resolve("Dubai")

        assert result.is_no_match

    async def test_an_exact_match_resolves_even_with_several_suggestions(
        self, monkeypatch
    ):
        """A country-level exact match wins over its own cities.

        Live behavior (2026-09-17): "Saudi Arabia" rendered five suggestions
        and came back ``ambiguous`` even though one of them was the country
        itself -- the bug this covers. Only "Saudi Arabia" equals the query
        outright; the cities are neither an exact nor a canonical match, so
        they never compete with it.
        """
        labels = ["Saudi Arabia", "Riyadh, Saudi Arabia", "Jeddah, Saudi Arabia"]
        page = _FakeJobsSearchPage(
            suggestions_by_query={"Saudi Arabia": labels},
            geo_id_by_label={
                "Saudi Arabia": "100459316",
                "Riyadh, Saudi Arabia": "104305776",
                "Jeddah, Saudi Arabia": "106204383",
            },
        )
        resolver = _resolver(page, monkeypatch)

        result = await resolver.resolve("Saudi Arabia")

        assert result.resolved == GeoCandidate("Saudi Arabia", "100459316")
        assert result.candidates == ()
        assert not result.is_ambiguous
        assert page.navigations == 1

    async def test_a_canonical_city_country_match_resolves_among_lookalikes(
        self, monkeypatch
    ):
        """A "query, single region" match wins over a same-prefixed decoy.

        Live behavior (2026-09-17): "Dubai" rendered three suggestions and
        came back ``ambiguous``. Two of them start with "Dubai," but only
        "Dubai, United Arab Emirates" has exactly one comma; the emirate-
        level "Dubai, Dubai, United Arab Emirates" has two, and the third
        ("Dubail Union, Dhaka, Bangladesh") does not even share the first
        segment, so neither ties with the real answer.
        """
        labels = [
            "Dubai, United Arab Emirates",
            "Dubai, Dubai, United Arab Emirates",
            "Dubail Union, Dhaka, Bangladesh",
        ]
        page = _FakeJobsSearchPage(
            suggestions_by_query={"Dubai": labels},
            geo_id_by_label={
                "Dubai, United Arab Emirates": "104246948",
                "Dubai, Dubai, United Arab Emirates": "105117694",
                "Dubail Union, Dhaka, Bangladesh": "103537801",
            },
        )
        resolver = _resolver(page, monkeypatch)

        result = await resolver.resolve("Dubai")

        assert result.resolved == GeoCandidate(
            "Dubai, United Arab Emirates", "104246948"
        )
        assert not result.is_ambiguous
        assert page.navigations == 1

    async def test_matching_is_case_diacritic_and_whitespace_insensitive(
        self, monkeypatch
    ):
        labels = ["São Paulo, Brazil", "São Paulo, State of São Paulo, Brazil"]
        page = _FakeJobsSearchPage(
            suggestions_by_query={"SAO   paulo": labels},
            geo_id_by_label={
                "São Paulo, Brazil": "106057199",
                "São Paulo, State of São Paulo, Brazil": "90009659",
            },
        )
        resolver = _resolver(page, monkeypatch)

        result = await resolver.resolve("  SAO   paulo  ")

        assert result.resolved == GeoCandidate("São Paulo, Brazil", "106057199")

    async def test_no_candidate_matching_at_all_stays_ambiguous(self, monkeypatch):
        """Distinct from the "two matches" ambiguous case: here neither
        candidate is an exact or canonical match for the query at all, so
        the ``matches`` list inside ``_find_exact_match`` is empty rather
        than having more than one entry -- a different branch to guard."""
        labels = [
            "Springfield, Illinois, United States",
            "Springfield, Massachusetts, United States",
        ]
        page = _FakeJobsSearchPage(
            suggestions_by_query={"Springfield": labels},
            geo_id_by_label={
                "Springfield, Illinois, United States": "103112676",
                "Springfield, Massachusetts, United States": "102380872",
            },
        )
        resolver = _resolver(page, monkeypatch)

        result = await resolver.resolve("Springfield")

        assert result.is_ambiguous
        assert result.candidates == (
            GeoCandidate("Springfield, Illinois, United States", "103112676"),
            GeoCandidate("Springfield, Massachusetts, United States", "102380872"),
        )
        assert page.navigations == 1

    async def test_a_multi_word_query_retries_with_a_shorter_prefix(self, monkeypatch):
        """The reported bug: a three-word country name renders nothing on
        the first pass. Dropping the last word ("United Arab") is enough
        for LinkedIn's own typeahead to answer, and the result is filtered
        to the suggestion that still contains the full original phrase --
        "United Kingdom" is on the same reopened listbox but is not a match
        for "United Arab Emirates" and must not be surfaced as one. The
        retry reuses the page already open, so this still costs one
        navigation, not two.
        """
        page = _FakeJobsSearchPage(
            suggestions_by_query={
                "United Arab": ["United Arab Emirates", "United Kingdom"]
            },
            geo_id_by_label={"United Arab Emirates": "104246948"},
        )
        resolver = _resolver(page, monkeypatch)

        result = await resolver.resolve("United Arab Emirates")

        assert result.resolved == GeoCandidate("United Arab Emirates", "104246948")
        assert not result.is_ambiguous
        assert page.navigations == 1

    async def test_shorter_prefix_retry_with_nothing_relevant_is_still_no_match(
        self, monkeypatch
    ):
        page = _FakeJobsSearchPage(
            suggestions_by_query={"United Arab": ["United Kingdom", "United States"]},
        )
        resolver = _resolver(page, monkeypatch)

        result = await resolver.resolve("United Arab Emirates")

        assert result.is_no_match
        assert page.navigations == 1

    async def test_a_slow_listbox_is_retried_before_giving_up(self, monkeypatch):
        """Bounded retry: a first timeout re-fills the box and waits again
        rather than immediately reporting no match."""
        page = _FakeJobsSearchPage(
            suggestions_by_query={"Dubai": ["Dubai, United Arab Emirates"]},
            geo_id_by_label={"Dubai, United Arab Emirates": "104246948"},
            stall_once_for={"Dubai"},
        )
        resolver = _resolver(page, monkeypatch)

        result = await resolver.resolve("Dubai")

        assert result.resolved == GeoCandidate(
            "Dubai, United Arab Emirates", "104246948"
        )
        assert page.navigations == 1


class TestNavigationBudget:
    """Dedicated proof of the speed fix: at most one navigation per resolve
    (plus, in the last case, exactly one bounded recovery)."""

    async def test_a_single_match_costs_exactly_one_navigation(self, monkeypatch):
        page = _FakeJobsSearchPage(
            suggestions_by_query={"Dubai": ["Dubai, United Arab Emirates"]},
            geo_id_by_label={"Dubai, United Arab Emirates": "104246948"},
        )
        resolver = _resolver(page, monkeypatch)

        await resolver.resolve("Dubai")

        assert page.navigations == 1

    async def test_several_candidates_still_cost_exactly_one_navigation(
        self, monkeypatch
    ):
        labels = ["Georgia", "Georgia, United States"]
        page = _FakeJobsSearchPage(
            suggestions_by_query={"Georgia": labels},
            geo_id_by_label={
                "Georgia": "101452733",
                "Georgia, United States": "104081876",
            },
        )
        resolver = _resolver(page, monkeypatch)

        await resolver.resolve("Georgia")

        assert page.navigations == 1

    async def test_a_page_lost_mid_resolution_gets_exactly_one_recovery(
        self, monkeypatch
    ):
        """The page reports itself lost right after the first candidate is
        clicked (a client-side redirect mid-resolution); the resolver
        recovers with exactly one extra navigation and still finishes."""
        labels = ["Georgia", "Georgia, United States"]
        page = _FakeJobsSearchPage(
            suggestions_by_query={"Georgia": labels},
            geo_id_by_label={
                "Georgia": "101452733",
                "Georgia, United States": "104081876",
            },
            lose_after_clicks=1,
        )
        resolver = _resolver(page, monkeypatch)

        result = await resolver.resolve("Georgia")

        assert result.is_ambiguous
        assert result.candidates == (
            GeoCandidate("Georgia", "101452733"),
            GeoCandidate("Georgia, United States", "104081876"),
        )
        # The original load plus exactly one recovery -- never more.
        assert page.navigations == 2


class TestResolveWithCache:
    """When :meth:`GeoLocationResolver.resolve` reads and writes a cache."""

    async def test_a_cache_hit_resolves_with_zero_navigation(self, monkeypatch):
        page = _FakeJobsSearchPage()  # would raise AssertionError if driven
        cache = _RecordingCache()
        cache.store["dubai"] = CachedGeoResolution(
            name="Dubai, United Arab Emirates",
            geo_urn_id="104246948",
            resolved_at=0.0,
        )
        resolver = _resolver(page, monkeypatch, cache=cache)

        result = await resolver.resolve("Dubai")

        assert result.resolved == GeoCandidate(
            "Dubai, United Arab Emirates", "104246948"
        )
        assert result.cached is True
        assert page.navigations == 0

    async def test_an_unambiguous_single_suggestion_is_cached(self, monkeypatch):
        page = _FakeJobsSearchPage(
            suggestions_by_query={"Dubai": ["Dubai, United Arab Emirates"]},
            geo_id_by_label={"Dubai, United Arab Emirates": "104246948"},
        )
        cache = _RecordingCache()
        resolver = _resolver(page, monkeypatch, cache=cache)

        result = await resolver.resolve("Dubai")

        assert result.cached is False
        assert cache.puts == [("dubai", "Dubai, United Arab Emirates", "104246948")]

    async def test_an_exact_match_among_several_candidates_is_cached(self, monkeypatch):
        labels = ["Saudi Arabia", "Riyadh, Saudi Arabia"]
        page = _FakeJobsSearchPage(
            suggestions_by_query={"Saudi Arabia": labels},
            geo_id_by_label={
                "Saudi Arabia": "100459316",
                "Riyadh, Saudi Arabia": "104305776",
            },
        )
        cache = _RecordingCache()
        resolver = _resolver(page, monkeypatch, cache=cache)

        result = await resolver.resolve("Saudi Arabia")

        assert result.resolved == GeoCandidate("Saudi Arabia", "100459316")
        assert cache.puts == [("saudi arabia", "Saudi Arabia", "100459316")]

    async def test_an_ambiguous_result_is_never_cached(self, monkeypatch):
        labels = ["Georgia", "Georgia, United States"]
        page = _FakeJobsSearchPage(
            suggestions_by_query={"Georgia": labels},
            geo_id_by_label={
                "Georgia": "101452733",
                "Georgia, United States": "104081876",
            },
        )
        cache = _RecordingCache()
        resolver = _resolver(page, monkeypatch, cache=cache)

        result = await resolver.resolve("Georgia")

        assert result.is_ambiguous
        assert cache.puts == []

    async def test_a_no_match_result_is_never_cached(self, monkeypatch):
        page = _FakeJobsSearchPage()
        cache = _RecordingCache()
        resolver = _resolver(page, monkeypatch, cache=cache)

        result = await resolver.resolve("Nowhereville")

        assert result.is_no_match
        assert cache.puts == []

    async def test_a_repeat_query_hits_the_real_cache_case_insensitively(
        self, monkeypatch, tmp_path
    ):
        """End-to-end against the real ``GeoResolutionCache`` (not the
        recording double): the first call resolves live and stores it, the
        second -- differently cased and padded -- hits with zero navigation.
        """
        page = _FakeJobsSearchPage(
            suggestions_by_query={"Dubai": ["Dubai, United Arab Emirates"]},
            geo_id_by_label={"Dubai, United Arab Emirates": "104246948"},
        )
        cache = GeoResolutionCache(lambda: tmp_path / "geo-resolution-cache.json")
        resolver = _resolver(page, monkeypatch, cache=cache)

        first = await resolver.resolve("Dubai")
        assert first.cached is False
        assert page.navigations == 1

        second = await resolver.resolve("  DUBAI  ")

        assert second.cached is True
        assert second.resolved == GeoCandidate(
            "Dubai, United Arab Emirates", "104246948"
        )
        assert page.navigations == 1  # unchanged -- the hit cost nothing


class TestGeoResolutionCache:
    """``GeoResolutionCache`` itself, against a real file under ``tmp_path``.

    Exercised directly by key (this class does its own normalization in
    production -- see ``_normalize`` and ``GeoLocationResolver.resolve``'s
    use of it), since the cache itself is deliberately key-agnostic.
    """

    def _cache(self, tmp_path, **kwargs) -> GeoResolutionCache:
        path = tmp_path / "geo-resolution-cache.json"
        return GeoResolutionCache(lambda: path, **kwargs)

    def test_a_missing_file_is_a_clean_miss(self, tmp_path):
        cache = self._cache(tmp_path)

        assert cache.get("dubai") is None

    def test_put_then_get_round_trips(self, tmp_path):
        cache = self._cache(tmp_path)
        cache.put("dubai", name="Dubai, United Arab Emirates", geo_urn_id="104246948")

        entry = cache.get("dubai")

        assert entry is not None
        assert entry.name == "Dubai, United Arab Emirates"
        assert entry.geo_urn_id == "104246948"

    def test_the_cache_file_persists_across_instances(self, tmp_path):
        path = tmp_path / "geo-resolution-cache.json"
        GeoResolutionCache(lambda: path).put(
            "dubai", name="Dubai, United Arab Emirates", geo_urn_id="104246948"
        )

        reloaded = GeoResolutionCache(lambda: path).get("dubai")

        assert reloaded is not None
        assert reloaded.name == "Dubai, United Arab Emirates"
        assert reloaded.geo_urn_id == "104246948"
        assert path.exists()

    def test_an_entry_older_than_the_ttl_is_a_miss(self, tmp_path):
        clock = {"now": 1_000.0}
        cache = self._cache(tmp_path, clock=lambda: clock["now"], ttl_seconds=60.0)
        cache.put("dubai", name="Dubai, United Arab Emirates", geo_urn_id="104246948")

        clock["now"] += 30.0
        assert cache.get("dubai") is not None  # still inside the TTL

        clock["now"] += 31.0  # 61s since the put -- past the 60s TTL
        assert cache.get("dubai") is None

    def test_a_refreshed_entry_extends_the_ttl_from_the_refresh(self, tmp_path):
        clock = {"now": 1_000.0}
        cache = self._cache(tmp_path, clock=lambda: clock["now"], ttl_seconds=60.0)
        cache.put("dubai", name="Dubai, United Arab Emirates", geo_urn_id="104246948")

        clock["now"] += 50.0
        cache.put("dubai", name="Dubai, United Arab Emirates", geo_urn_id="104246948")
        clock["now"] += 50.0  # 100s after the first put, 50s after the refresh

        assert cache.get("dubai") is not None

    def test_a_corrupt_file_is_tolerated_and_still_usable_afterwards(self, tmp_path):
        path = tmp_path / "geo-resolution-cache.json"
        path.write_text("not json at all {{{", encoding="utf-8")
        cache = GeoResolutionCache(lambda: path)

        assert cache.get("dubai") is None

        # Corruption does not wedge the cache -- it can still be written to.
        cache.put("dubai", name="Dubai, United Arab Emirates", geo_urn_id="104246948")
        entry = cache.get("dubai")
        assert entry is not None
        assert entry.name == "Dubai, United Arab Emirates"
        assert entry.geo_urn_id == "104246948"

    def test_an_unknown_version_is_treated_as_corrupt(self, tmp_path):
        path = tmp_path / "geo-resolution-cache.json"
        path.write_text(json.dumps({"version": 999, "entries": {}}), encoding="utf-8")
        cache = GeoResolutionCache(lambda: path)

        assert cache.get("dubai") is None

    def test_a_malformed_entry_is_tolerated(self, tmp_path):
        path = tmp_path / "geo-resolution-cache.json"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "entries": {"dubai": {"name": "Dubai", "geo_urn_id": 123}},
                }
            ),
            encoding="utf-8",
        )
        cache = GeoResolutionCache(lambda: path)

        assert cache.get("dubai") is None

    def test_entries_is_not_an_object_is_tolerated(self, tmp_path):
        path = tmp_path / "geo-resolution-cache.json"
        path.write_text(json.dumps({"version": 1, "entries": "nope"}), encoding="utf-8")
        cache = GeoResolutionCache(lambda: path)

        assert cache.get("dubai") is None


class TestNormalize:
    def test_case_insensitive(self):
        assert _normalize("DUBAI") == _normalize("dubai")

    def test_diacritics_are_stripped(self):
        assert _normalize("São Paulo") == "sao paulo"

    def test_whitespace_is_collapsed_and_trimmed(self):
        assert _normalize("  United   Arab  Emirates ") == "united arab emirates"


class TestShorterPrefix:
    def test_multi_word_query_drops_the_last_word(self):
        assert _shorter_prefix("United Arab Emirates") == "United Arab"

    def test_single_word_takes_the_first_half(self):
        assert _shorter_prefix("Nowhereville") == "Nowher"

    def test_a_short_single_word_has_no_shorter_form(self):
        assert _shorter_prefix("Uk") is None
        assert _shorter_prefix("USA") is None


class TestFindExactMatch:
    def test_exact_name_match(self):
        candidates = (GeoCandidate("Saudi Arabia", "1"),)
        assert _find_exact_match("Saudi Arabia", candidates) == candidates[0]

    def test_canonical_city_country_match(self):
        candidates = (GeoCandidate("Dubai, United Arab Emirates", "1"),)
        assert _find_exact_match("Dubai", candidates) == candidates[0]

    def test_three_segment_name_is_not_a_canonical_match(self):
        candidates = (GeoCandidate("Dubai, Dubai, United Arab Emirates", "1"),)
        assert _find_exact_match("Dubai", candidates) is None

    def test_two_simultaneous_matches_are_not_resolved(self):
        candidates = (
            GeoCandidate("Georgia", "1"),
            GeoCandidate("Georgia, United States", "2"),
        )
        assert _find_exact_match("Georgia", candidates) is None

    def test_no_match_returns_none(self):
        candidates = (GeoCandidate("Riyadh, Saudi Arabia", "1"),)
        assert _find_exact_match("Dubai", candidates) is None


class TestResolveGeoLocationFacade:
    async def test_returns_a_flat_candidate_list_for_a_single_match(
        self, mock_page, monkeypatch
    ):
        from linkedin_mcp_server.scraping.person import PersonScraper

        scraper = PersonScraper.__new__(PersonScraper)
        scraper._geo_resolver = AsyncMock()
        scraper._geo_resolver.resolve = AsyncMock(
            return_value=GeoResolution(
                resolved=GeoCandidate("Dubai, United Arab Emirates", "104246948")
            )
        )

        result = await PersonScraper.resolve_geo_location(scraper, "Dubai")

        assert result == {
            "query": "Dubai",
            "candidates": [
                {"name": "Dubai, United Arab Emirates", "geo_urn_id": "104246948"}
            ],
            "ambiguous": False,
            "cached": False,
        }

    async def test_a_cached_resolution_is_reported_as_such(self, mock_page):
        from linkedin_mcp_server.scraping.person import PersonScraper

        scraper = PersonScraper.__new__(PersonScraper)
        scraper._geo_resolver = AsyncMock()
        scraper._geo_resolver.resolve = AsyncMock(
            return_value=GeoResolution(
                resolved=GeoCandidate("Dubai, United Arab Emirates", "104246948"),
                cached=True,
            )
        )

        result = await PersonScraper.resolve_geo_location(scraper, "Dubai")

        assert result["cached"] is True

    async def test_returns_every_candidate_when_ambiguous(self, mock_page):
        from linkedin_mcp_server.scraping.person import PersonScraper

        scraper = PersonScraper.__new__(PersonScraper)
        scraper._geo_resolver = AsyncMock()
        scraper._geo_resolver.resolve = AsyncMock(
            return_value=GeoResolution(
                candidates=(
                    GeoCandidate("Georgia", "101452733"),
                    GeoCandidate("Georgia, United States", "104081876"),
                )
            )
        )

        result = await PersonScraper.resolve_geo_location(scraper, "Georgia")

        assert result["ambiguous"] is True
        assert result["cached"] is False
        assert result["candidates"] == [
            {"name": "Georgia", "geo_urn_id": "101452733"},
            {"name": "Georgia, United States", "geo_urn_id": "104081876"},
        ]

    async def test_returns_an_empty_list_for_no_match(self, mock_page):
        from linkedin_mcp_server.scraping.person import PersonScraper

        scraper = PersonScraper.__new__(PersonScraper)
        scraper._geo_resolver = AsyncMock()
        scraper._geo_resolver.resolve = AsyncMock(return_value=GeoResolution())

        result = await PersonScraper.resolve_geo_location(scraper, "Nowhereville")

        assert result == {
            "query": "Nowhereville",
            "candidates": [],
            "ambiguous": False,
            "cached": False,
        }
