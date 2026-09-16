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
        resolver = _resolver(mock_page, monkeypatch)
        mock_page.evaluate = AsyncMock(side_effect=[True, []])
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
        selection cannot pass silently.
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
        the wrong id.
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

        assert result.candidates == (GeoCandidate("Georgia", "101452733"),)

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
