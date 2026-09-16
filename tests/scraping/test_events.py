"""Tests for the event scraping owner (search, details, attendees)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from linkedin_mcp_server.core.exceptions import InvalidReferenceError
from linkedin_mcp_server.scraping.capture import (
    CaptureMode,
    SectionCapture,
)
from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.contracts import ExtractedSection
from linkedin_mcp_server.scraping.events import EventScraper
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession


def _scraper(page) -> EventScraper:
    session = ScrapingSession(page)
    navigator = PageNavigator(session)
    return EventScraper(SectionCapture(session, navigator, PageContentReader(session)))


def extracted(text: str, references=None, error=None) -> ExtractedSection:
    return ExtractedSection(text=text, references=references or [], error=error)


class TestSearchEvents:
    async def test_builds_the_event_search_url_and_scales_scrolls_by_max_pages(
        self, mock_page
    ):
        scraper = _scraper(mock_page)
        with patch.object(
            scraper._capture,
            "capture",
            new_callable=AsyncMock,
            return_value=extracted("RPA Summit 2026"),
        ) as mock_capture:
            result = await scraper.search_events("rpa summit", max_pages=2)

        url, section_name, plan = mock_capture.call_args.args
        assert url == (
            "https://www.linkedin.com/search/results/events/?keywords=rpa+summit"
        )
        assert section_name == "search_results"
        assert plan.mode == CaptureMode.SEARCH_RESULTS
        assert plan.max_scrolls == 10  # 2 pages * 5 scrolls/page
        assert result["sections"]["search_results"] == "RPA Summit 2026"


class TestGetEventDetails:
    async def test_normalizes_a_slugged_event_link(self, mock_page):
        scraper = _scraper(mock_page)
        with patch.object(
            scraper._capture,
            "capture",
            new_callable=AsyncMock,
            return_value=extracted("RPA Summit 2026\nVirtual event"),
        ) as mock_capture:
            result = await scraper.get_event_details(
                "https://www.linkedin.com/events/rpa-summit-2026-1234567890/"
            )

        url, section_name, _plan = mock_capture.call_args.args
        assert url == "https://www.linkedin.com/events/1234567890/"
        assert section_name == "event_details"
        assert result["sections"]["event_details"] == "RPA Summit 2026\nVirtual event"

    async def test_a_non_numeric_event_reference_is_refused_before_navigating(
        self, mock_page
    ):
        scraper = _scraper(mock_page)
        with patch.object(
            scraper._capture, "capture", new_callable=AsyncMock
        ) as mock_capture:
            with pytest.raises(InvalidReferenceError):
                await scraper.get_event_details("not-an-event")
        mock_capture.assert_not_awaited()


class TestGetEventAttendees:
    async def test_uses_company_people_capture_mode_on_the_attendees_subpath(
        self, mock_page
    ):
        scraper = _scraper(mock_page)
        with patch.object(
            scraper._capture,
            "capture",
            new_callable=AsyncMock,
            return_value=extracted("Jane Doe\nAda Lovelace"),
        ) as mock_capture:
            result = await scraper.get_event_attendees("1234567890", max_attendees=50)

        url, section_name, plan = mock_capture.call_args.args
        assert url == "https://www.linkedin.com/events/1234567890/attendees/"
        assert section_name == "attendees"
        assert plan.mode == CaptureMode.COMPANY_PEOPLE
        assert result["sections"]["attendees"] == "Jane Doe\nAda Lovelace"

    async def test_max_attendees_bounds_the_scroll_budget(self, mock_page):
        scraper = _scraper(mock_page)
        with patch.object(
            scraper._capture,
            "capture",
            new_callable=AsyncMock,
            return_value=extracted("text"),
        ) as mock_capture:
            await scraper.get_event_attendees("1234567890", max_attendees=5000)

        plan = mock_capture.call_args.args[2]
        assert plan.max_scrolls == 20  # capped
