"""Tests for free-text LinkedIn location resolution."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from patchright.async_api import TimeoutError as PlaywrightTimeoutError

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


def _resolver(mock_page, monkeypatch) -> GeoLocationResolver:
    """Wire a resolver whose navigation and rate-limit checks are no-ops.

    Those two are exercised elsewhere (navigation.py's own tests); what this
    file verifies is the selection logic once a page is already open, so
    both are replaced with no-ops rather than driven through the real
    goto/auth-barrier machinery a bare mock page cannot satisfy.
    ``ScrapingSession`` is a frozen dataclass, so the module-level function
    it delegates to is patched instead of an instance attribute.
    """
    session = ScrapingSession(mock_page)
    navigator = PageNavigator(session)
    resolver = GeoLocationResolver(session, navigator)
    monkeypatch.setattr(navigator, "_navigate_to_page", AsyncMock())
    monkeypatch.setattr(session_module, "detect_rate_limit", AsyncMock())
    return resolver


class TestResolve:
    async def test_blank_query_is_refused_before_any_navigation(
        self, mock_page, monkeypatch
    ):
        resolver = _resolver(mock_page, monkeypatch)

        with pytest.raises(FilterValidationError, match="must not be blank"):
            await resolver.resolve("   ")

        mock_page.goto.assert_not_awaited()

    async def test_no_suggestions_is_no_match(self, mock_page, monkeypatch):
        """Empty on the direct query *and* on the one shorter-prefix retry.

        "Nowhereville" is one word, so the retry is the first half of it
        ("Nowher") -- confirming the no-match path still exhausts the retry
        before giving up, rather than reporting no-match too early.
        """
        resolver = _resolver(mock_page, monkeypatch)
        monkeypatch.setattr(geo_resolver_module, "NAV_DELAY", 0.0)
        mock_page.evaluate = AsyncMock(
            side_effect=[
                True,  # fill "Nowhereville"
                [],  # read labels: none
                True,  # fill "Nowher" (shorter-prefix retry)
                [],  # read labels: still none
            ]
        )
        mock_page.wait_for_function = AsyncMock()

        result = await resolver.resolve("Nowhereville")

        assert result == GeoResolution()
        assert result.is_no_match
        assert not result.is_ambiguous

    async def test_a_stalled_dropdown_is_no_match_not_an_error(
        self, mock_page, monkeypatch
    ):
        """A listbox that never appears reads as unknown, not as a failure."""
        resolver = _resolver(mock_page, monkeypatch)
        mock_page.evaluate = AsyncMock(return_value=True)
        mock_page.wait_for_function = AsyncMock(
            side_effect=PlaywrightTimeoutError("no listbox")
        )

        result = await resolver.resolve("Atlantis")

        assert result.is_no_match

    async def test_a_single_suggestion_resolves_silently(self, mock_page, monkeypatch):
        resolver = _resolver(mock_page, monkeypatch)
        mock_page.evaluate = AsyncMock(
            side_effect=[
                True,  # fill
                ["Dubai, United Arab Emirates"],  # read labels
                True,  # click index 0
                "https://www.linkedin.com/jobs/search/?geoId=104246948&keywords=",
            ]
        )
        mock_page.wait_for_function = AsyncMock()

        result = await resolver.resolve("Dubai")

        assert result.resolved == GeoCandidate(
            "Dubai, United Arab Emirates", "104246948"
        )
        assert result.candidates == ()
        assert not result.is_ambiguous
        assert not result.is_no_match

    async def test_multiple_suggestions_are_all_resolved_as_candidates(
        self, mock_page, monkeypatch
    ):
        """Ambiguous: every candidate's id is independently confirmed.

        Mutation guard: swap either resolved id for the other's and this
        fails, so picking by position rather than by re-verifying each
        selection cannot pass silently. "Georgia" also exercises the exact-
        match disambiguation added alongside this test: the country name
        equals the query outright *and* "Georgia, United States" is
        simultaneously the query's canonical form, so two candidates match
        and the query correctly stays ambiguous rather than picking either.
        """
        resolver = _resolver(mock_page, monkeypatch)
        monkeypatch.setattr(geo_resolver_module, "NAV_DELAY", 0.0)
        labels = ["Georgia", "Georgia, United States"]
        mock_page.evaluate = AsyncMock(
            side_effect=[
                True,  # fill (first open)
                labels,  # read labels (first open)
                True,  # click index 0
                "https://www.linkedin.com/jobs/search/?geoId=101452733&keywords=",
                True,  # fill (re-open for index 1)
                labels,  # read labels (re-open, same order)
                True,  # click index 1
                "https://www.linkedin.com/jobs/search/?geoId=104081876&keywords=",
            ]
        )
        mock_page.wait_for_function = AsyncMock()

        result = await resolver.resolve("Georgia")

        assert result.resolved is None
        assert result.is_ambiguous
        assert result.candidates == (
            GeoCandidate("Georgia", "101452733"),
            GeoCandidate("Georgia, United States", "104081876"),
        )

    async def test_a_candidate_is_dropped_when_the_reopened_order_shifts(
        self, mock_page, monkeypatch
    ):
        """Never trusts a re-selected index without re-checking its label.

        If the suggestion order is not stable across an identical query, the
        candidate at that position is skipped rather than mislabeled with
        the wrong id. Only "Georgia" survives re-verification here, and it
        is an exact match for the query, so it comes back ``resolved``
        rather than as a one-item ambiguous list -- the mutation guard
        (swap its id for the dropped candidate's) still fails either way.
        """
        resolver = _resolver(mock_page, monkeypatch)
        monkeypatch.setattr(geo_resolver_module, "NAV_DELAY", 0.0)
        mock_page.evaluate = AsyncMock(
            side_effect=[
                True,
                ["Georgia", "Georgia, United States"],
                True,
                "https://www.linkedin.com/jobs/search/?geoId=101452733&keywords=",
                True,
                # Order changed on re-query: index 1 now reads differently.
                ["Georgia", "Something Else"],
            ]
        )
        mock_page.wait_for_function = AsyncMock()

        result = await resolver.resolve("Georgia")

        assert result.resolved == GeoCandidate("Georgia", "101452733")
        assert result.candidates == ()
        assert not result.is_ambiguous

    async def test_a_click_that_never_carries_a_geo_id_is_dropped(
        self, mock_page, monkeypatch
    ):
        resolver = _resolver(mock_page, monkeypatch)
        mock_page.evaluate = AsyncMock(
            side_effect=[
                True,
                ["Dubai, United Arab Emirates"],
                True,  # click succeeds
            ]
        )
        mock_page.wait_for_function = AsyncMock(
            side_effect=[None, PlaywrightTimeoutError("no geoId ever appeared")]
        )

        result = await resolver.resolve("Dubai")

        assert result.is_no_match

    async def test_an_exact_match_resolves_even_with_several_suggestions(
        self, mock_page, monkeypatch
    ):
        """A country-level exact match wins over its own cities.

        Live behavior (2026-09-17): "Saudi Arabia" rendered five suggestions
        and came back ``ambiguous`` even though one of them was the country
        itself -- the bug this covers. Only "Saudi Arabia" equals the query
        outright; the cities are neither an exact nor a canonical match, so
        they never compete with it.
        """
        resolver = _resolver(mock_page, monkeypatch)
        monkeypatch.setattr(geo_resolver_module, "NAV_DELAY", 0.0)
        labels = ["Saudi Arabia", "Riyadh, Saudi Arabia", "Jeddah, Saudi Arabia"]
        mock_page.evaluate = AsyncMock(
            side_effect=[
                True,  # fill (first open)
                labels,  # read labels
                True,  # click index 0
                "https://www.linkedin.com/jobs/search/?geoId=100459316&keywords=",
                True,  # fill (re-open for index 1)
                labels,  # read labels (re-open, same order)
                True,  # click index 1
                "https://www.linkedin.com/jobs/search/?geoId=104305776&keywords=",
                True,  # fill (re-open for index 2)
                labels,  # read labels (re-open, same order)
                True,  # click index 2
                "https://www.linkedin.com/jobs/search/?geoId=106204383&keywords=",
            ]
        )
        mock_page.wait_for_function = AsyncMock()

        result = await resolver.resolve("Saudi Arabia")

        assert result.resolved == GeoCandidate("Saudi Arabia", "100459316")
        assert result.candidates == ()
        assert not result.is_ambiguous

    async def test_a_canonical_city_country_match_resolves_among_lookalikes(
        self, mock_page, monkeypatch
    ):
        """A "query, single region" match wins over a same-prefixed decoy.

        Live behavior (2026-09-17): "Dubai" rendered three suggestions and
        came back ``ambiguous``. Two of them start with "Dubai," but only
        "Dubai, United Arab Emirates" has exactly one comma; the emirate-
        level "Dubai, Dubai, United Arab Emirates" has two, and the third
        ("Dubail Union, Dhaka, Bangladesh") does not even share the first
        segment, so neither ties with the real answer.
        """
        resolver = _resolver(mock_page, monkeypatch)
        monkeypatch.setattr(geo_resolver_module, "NAV_DELAY", 0.0)
        labels = [
            "Dubai, United Arab Emirates",
            "Dubai, Dubai, United Arab Emirates",
            "Dubail Union, Dhaka, Bangladesh",
        ]
        mock_page.evaluate = AsyncMock(
            side_effect=[
                True,
                labels,
                True,
                "https://www.linkedin.com/jobs/search/?geoId=104246948&keywords=",
                True,
                labels,
                True,
                "https://www.linkedin.com/jobs/search/?geoId=105117694&keywords=",
                True,
                labels,
                True,
                "https://www.linkedin.com/jobs/search/?geoId=103537801&keywords=",
            ]
        )
        mock_page.wait_for_function = AsyncMock()

        result = await resolver.resolve("Dubai")

        assert result.resolved == GeoCandidate(
            "Dubai, United Arab Emirates", "104246948"
        )
        assert not result.is_ambiguous

    async def test_matching_is_case_diacritic_and_whitespace_insensitive(
        self, mock_page, monkeypatch
    ):
        resolver = _resolver(mock_page, monkeypatch)
        monkeypatch.setattr(geo_resolver_module, "NAV_DELAY", 0.0)
        labels = ["São Paulo, Brazil", "São Paulo, State of São Paulo, Brazil"]
        mock_page.evaluate = AsyncMock(
            side_effect=[
                True,
                labels,
                True,
                "https://www.linkedin.com/jobs/search/?geoId=106057199&keywords=",
                True,
                labels,
                True,
                "https://www.linkedin.com/jobs/search/?geoId=90009659&keywords=",
            ]
        )
        mock_page.wait_for_function = AsyncMock()

        result = await resolver.resolve("  SAO   paulo  ")

        assert result.resolved == GeoCandidate("São Paulo, Brazil", "106057199")

    async def test_no_candidate_matching_at_all_stays_ambiguous(
        self, mock_page, monkeypatch
    ):
        """Distinct from the "two matches" ambiguous case: here neither
        candidate is an exact or canonical match for the query at all, so
        the ``matches`` list inside ``_find_exact_match`` is empty rather
        than having more than one entry -- a different branch to guard."""
        resolver = _resolver(mock_page, monkeypatch)
        monkeypatch.setattr(geo_resolver_module, "NAV_DELAY", 0.0)
        labels = [
            "Springfield, Illinois, United States",
            "Springfield, Massachusetts, United States",
        ]
        mock_page.evaluate = AsyncMock(
            side_effect=[
                True,
                labels,
                True,
                "https://www.linkedin.com/jobs/search/?geoId=103112676&keywords=",
                True,
                labels,
                True,
                "https://www.linkedin.com/jobs/search/?geoId=102380872&keywords=",
            ]
        )
        mock_page.wait_for_function = AsyncMock()

        result = await resolver.resolve("Springfield")

        assert result.is_ambiguous
        assert result.candidates == (
            GeoCandidate("Springfield, Illinois, United States", "103112676"),
            GeoCandidate("Springfield, Massachusetts, United States", "102380872"),
        )

    async def test_a_multi_word_query_retries_with_a_shorter_prefix(
        self, mock_page, monkeypatch
    ):
        """The reported bug: a three-word country name renders nothing on
        the first pass. Dropping the last word ("United Arab") is enough
        for LinkedIn's own typeahead to answer, and the result is filtered
        to the suggestion that still contains the full original phrase --
        "United Kingdom" is on the same reopened listbox but is not a match
        for "United Arab Emirates" and must not be surfaced as one.
        """
        resolver = _resolver(mock_page, monkeypatch)
        monkeypatch.setattr(geo_resolver_module, "NAV_DELAY", 0.0)
        mock_page.evaluate = AsyncMock(
            side_effect=[
                True,  # fill "United Arab Emirates"
                [],  # read labels: none
                True,  # fill "United Arab" (shorter-prefix retry)
                ["United Arab Emirates", "United Kingdom"],  # read labels
                True,  # click index 0 (the relevant one)
                "https://www.linkedin.com/jobs/search/?geoId=104246948&keywords=",
            ]
        )
        mock_page.wait_for_function = AsyncMock()

        result = await resolver.resolve("United Arab Emirates")

        assert result.resolved == GeoCandidate("United Arab Emirates", "104246948")
        assert not result.is_ambiguous

    async def test_shorter_prefix_retry_with_nothing_relevant_is_still_no_match(
        self, mock_page, monkeypatch
    ):
        resolver = _resolver(mock_page, monkeypatch)
        monkeypatch.setattr(geo_resolver_module, "NAV_DELAY", 0.0)
        mock_page.evaluate = AsyncMock(
            side_effect=[
                True,  # fill "United Arab Emirates"
                [],  # read labels: none
                True,  # fill "United Arab" (shorter-prefix retry)
                ["United Kingdom", "United States"],  # neither is relevant
            ]
        )
        mock_page.wait_for_function = AsyncMock()

        result = await resolver.resolve("United Arab Emirates")

        assert result.is_no_match

    async def test_a_slow_listbox_is_retried_before_giving_up(
        self, mock_page, monkeypatch
    ):
        """Bounded retry: a first timeout re-fills the box and waits again
        rather than immediately reporting no match."""
        resolver = _resolver(mock_page, monkeypatch)
        monkeypatch.setattr(geo_resolver_module, "NAV_DELAY", 0.0)
        mock_page.evaluate = AsyncMock(
            side_effect=[
                True,  # fill
                True,  # re-fill after the first timed-out wait
                ["Dubai, United Arab Emirates"],  # read labels: succeeds now
                True,  # click index 0
                "https://www.linkedin.com/jobs/search/?geoId=104246948&keywords=",
            ]
        )
        mock_page.wait_for_function = AsyncMock(
            side_effect=[PlaywrightTimeoutError("slow"), None, None]
        )

        result = await resolver.resolve("Dubai")

        assert result.resolved == GeoCandidate(
            "Dubai, United Arab Emirates", "104246948"
        )


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
        }

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

        assert result == {"query": "Nowhereville", "candidates": [], "ambiguous": False}
