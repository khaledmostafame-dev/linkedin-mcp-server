"""Tests for the group scraping owner (search, posts, members)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from linkedin_mcp_server.core.exceptions import InvalidReferenceError
from linkedin_mcp_server.scraping.capture import CaptureMode
from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.contracts import ExtractedSection
from linkedin_mcp_server.scraping.group import GroupScraper
from linkedin_mcp_server.scraping.capture import SectionCapture
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession
import pytest


def _scraper(page) -> GroupScraper:
    session = ScrapingSession(page)
    navigator = PageNavigator(session)
    return GroupScraper(
        session,
        navigator,
        SectionCapture(session, navigator, PageContentReader(session)),
    )


def extracted(text: str, references=None, error=None) -> ExtractedSection:
    return ExtractedSection(text=text, references=references or [], error=error)


class TestSearchGroups:
    async def test_builds_the_group_search_url_and_captures_once(self, mock_page):
        scraper = _scraper(mock_page)
        with patch.object(
            scraper._capture,
            "capture",
            new_callable=AsyncMock,
            return_value=extracted("Automation Anywhere Users Group"),
        ) as mock_capture:
            result = await scraper.search_groups("automation anywhere")

        url, section_name, plan = mock_capture.call_args.args
        assert url == (
            "https://www.linkedin.com/search/results/groups/"
            "?keywords=automation+anywhere"
        )
        assert section_name == "search_results"
        assert plan.mode == CaptureMode.SEARCH_RESULTS
        assert result["sections"]["search_results"] == "Automation Anywhere Users Group"


class TestGetGroupPosts:
    async def test_normalizes_a_pasted_group_link_and_captures_activity(
        self, mock_page
    ):
        scraper = _scraper(mock_page)
        with patch.object(
            scraper._capture,
            "capture",
            new_callable=AsyncMock,
            return_value=extracted("Post 1\nPost 2"),
        ) as mock_capture:
            result = await scraper.get_group_posts(
                "https://www.linkedin.com/groups/12345/", max_posts=20
            )

        url, section_name, plan = mock_capture.call_args.args
        assert url == "https://www.linkedin.com/groups/12345/"
        assert section_name == "posts"
        assert plan.mode == CaptureMode.ACTIVITY
        assert result["sections"]["posts"] == "Post 1\nPost 2"

    async def test_a_non_numeric_group_id_is_refused_before_navigating(self, mock_page):
        scraper = _scraper(mock_page)
        with patch.object(
            scraper._capture, "capture", new_callable=AsyncMock
        ) as mock_capture:
            with pytest.raises(InvalidReferenceError):
                await scraper.get_group_posts("not-a-group-id")
        mock_capture.assert_not_awaited()

    async def test_max_posts_bounds_the_scroll_budget(self, mock_page):
        scraper = _scraper(mock_page)
        with patch.object(
            scraper._capture,
            "capture",
            new_callable=AsyncMock,
            return_value=extracted("text"),
        ) as mock_capture:
            await scraper.get_group_posts("12345", max_posts=100)

        plan = mock_capture.call_args.args[2]
        assert plan.max_scrolls == 10  # capped


class TestGetGroupMembers:
    async def test_unfiltered_uses_company_people_capture_mode(self, mock_page):
        scraper = _scraper(mock_page)
        with patch.object(
            scraper._capture,
            "capture",
            new_callable=AsyncMock,
            return_value=extracted("Jane Doe\nSoftware Engineer"),
        ) as mock_capture:
            result = await scraper.get_group_members("12345", max_members=50)

        url, section_name, plan = mock_capture.call_args.args
        assert url == "https://www.linkedin.com/groups/12345/members/"
        assert section_name == "members"
        assert plan.mode == CaptureMode.COMPANY_PEOPLE
        assert result["sections"]["members"] == "Jane Doe\nSoftware Engineer"

    async def test_keywords_filter_via_search_box_without_renavigating(self, mock_page):
        """Filling the search box must not trigger a second navigation that
        would discard the filter -- the capture is done via the loaded-section
        seam, not by calling capture() (which always navigates)."""
        scraper = _scraper(mock_page)

        search_box = MagicMock()
        search_box.count = AsyncMock(return_value=1)
        search_box.click = AsyncMock()
        search_box.fill = AsyncMock()
        search_box.first = search_box
        mock_page.locator = MagicMock(return_value=search_box)

        with (
            patch.object(
                PageNavigator, "_navigate_to_page", new_callable=AsyncMock
            ) as mock_nav,
            patch.object(
                scraper._capture,
                "_extract_loaded_section",
                new_callable=AsyncMock,
                return_value=extracted("Filtered Jane Doe"),
            ) as mock_extract,
        ):
            result = await scraper.get_group_members(
                "12345", max_members=50, keywords="jane"
            )

        mock_nav.assert_awaited_once()  # exactly one navigation, not two
        search_box.fill.assert_awaited_once_with("jane")
        mock_extract.assert_awaited_once()
        assert result["sections"]["members"] == "Filtered Jane Doe"

    async def test_filtered_extraction_failure_is_reported_as_section_error(
        self, mock_page
    ):
        scraper = _scraper(mock_page)
        with (
            patch.object(
                PageNavigator,
                "_navigate_to_page",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
        ):
            result = await scraper.get_group_members(
                "12345", max_members=50, keywords="jane"
            )

        assert result["sections"] == {}
        assert "members" in result["section_errors"]
