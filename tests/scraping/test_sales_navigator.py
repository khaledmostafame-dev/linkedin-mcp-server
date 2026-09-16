"""Tests for the Sales Navigator scraper: seat gating and bounded pagination.

Seat detection is the load-bearing behavior here (AGENTS.md -> Scraping
Rules: locale-independent detection): every scenario below drives it through
the landed URL, never through page text, and asserts that a redirect stops
the call before any content extraction is attempted.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from linkedin_mcp_server.scraping.contracts import RATE_LIMITED_SECTION_TEXT
from linkedin_mcp_server.scraping.sales_navigator import (
    SALES_NAVIGATOR_UNAVAILABLE_ERROR,
    SalesNavigatorScraper,
)


def _page_result(text: str = "Lead: Ada Lovelace", references: list | None = None):
    return {"text": text, "references": references or []}


def _scraper(landed_urls: list[str], page_results: list | None = None):
    """Build a scraper whose landed URL and extracted page advance per call.

    ``landed_urls`` feeds ``navigator._navigate_to_page`` (one entry per
    navigation); ``page_results`` feeds ``content._extract_root_content``
    (one entry per extraction, defaulting to a single successful page
    repeated for every call if only one is given).
    """
    session = MagicMock()
    page = MagicMock()
    urls = list(landed_urls)
    call_index = {"value": 0}

    async def navigate(_url: str) -> None:
        # Repeats the last url once the scripted list is exhausted, so a
        # test only needs to supply as many landings as it cares about
        # (e.g. just the first, seat-deciding one) even when the scraper's
        # own pagination loop navigates further than that.
        index = min(call_index["value"], len(urls) - 1)
        page.url = urls[index]
        call_index["value"] += 1

    navigator = MagicMock()
    navigator._navigate_to_page = AsyncMock(side_effect=navigate)
    session.page = page
    session.check_rate_limit = AsyncMock()
    session.dismiss_modal = AsyncMock()
    session.scroll_body = AsyncMock()

    content = MagicMock()
    if page_results is None:
        content._extract_root_content = AsyncMock(return_value=_page_result())
    else:
        content._extract_root_content = AsyncMock(side_effect=page_results)

    return SalesNavigatorScraper(session, navigator, content), navigator, content


class TestSeatDetection:
    async def test_redirect_away_from_sales_returns_unavailable_error(self):
        scraper, _navigator, content = _scraper(
            ["https://www.linkedin.com/premium/products/"]
        )

        result = await scraper.search_leads("VP Engineering")

        assert result["sections"] == {}
        assert result["pages_fetched"] == 0
        assert result["stopped_reason"] == "no_seat"
        assert result["url"] == "https://www.linkedin.com/premium/products/"
        assert (
            result["section_errors"]["search_results"]["error_type"]
            == SALES_NAVIGATOR_UNAVAILABLE_ERROR
        )
        content._extract_root_content.assert_not_awaited()

    async def test_a_seat_proceeds_to_extraction(self):
        scraper, _navigator, _content = _scraper(
            ["https://www.linkedin.com/sales/search/people?keywords=VP+Engineering"]
        )

        result = await scraper.search_leads("VP Engineering")

        assert result["sections"]["search_results"] == "Lead: Ada Lovelace"
        assert result["pages_fetched"] == 1
        assert "section_errors" not in result

    async def test_seat_check_also_gates_get_lists(self):
        scraper, _navigator, content = _scraper(["https://www.linkedin.com/feed/"])

        result = await scraper.get_lists("accounts")

        assert result["sections"] == {}
        assert (
            result["section_errors"]["search_results"]["error_type"]
            == SALES_NAVIGATOR_UNAVAILABLE_ERROR
        )
        content._extract_root_content.assert_not_awaited()

    async def test_seat_check_also_gates_get_list(self):
        scraper, _navigator, content = _scraper(["https://www.linkedin.com/feed/"])

        result = await scraper.get_list(
            "https://www.linkedin.com/sales/lists/people/12345"
        )

        assert result["pages_fetched"] == 0
        assert result["stopped_reason"] == "no_seat"
        content._extract_root_content.assert_not_awaited()


class TestBoundedSearchPagination:
    async def test_stops_at_no_more_results(self):
        scraper, _navigator, _content = _scraper(
            ["https://www.linkedin.com/sales/search/people?keywords=ada&page=1"],
            page_results=[
                _page_result("Page one leads"),
                _page_result(""),  # page 2: nothing left
            ],
        )

        result = await scraper.search_leads("ada", max_pages=5)

        assert result["pages_fetched"] == 1
        assert result["stopped_reason"] == "no_more_results"
        assert result["sections"]["search_results"] == "Page one leads"

    async def test_rate_limited_page_is_reported_not_silent(self):
        scraper, _navigator, _content = _scraper(
            ["https://www.linkedin.com/sales/search/companies?keywords=acme"],
            page_results=[_page_result(RATE_LIMITED_SECTION_TEXT)],
        )

        result = await scraper.search_accounts("acme")

        assert result["stopped_reason"] == "rate_limited"
        assert result["section_errors"]["search_results"]["error_type"] == (
            "rate_limit"
        )
        assert result["sections"] == {}

    async def test_max_pages_is_capped_independent_of_caller_input(self):
        """A caller-supplied max_pages above the hard ceiling doesn't reach it."""
        scraper, navigator, content = _scraper(
            ["https://www.linkedin.com/sales/search/people?keywords=ada&page=1"]
        )

        await scraper.search_leads("ada", max_pages=10_000)

        # One page per successful iteration; the hard ceiling is 10, so at
        # most 10 extraction calls happen even though 10_000 was requested,
        # and each of the fixture's single scripted result satisfies every
        # call the same way (`_extract_root_content` was built with the
        # single-value default, which AsyncMock repeats).
        assert content._extract_root_content.await_count <= 10


class TestGetList:
    async def test_invalid_list_url_is_refused_before_any_navigation(self):
        scraper, navigator, content = _scraper([])

        with pytest.raises(ValueError, match="Sales Navigator URL"):
            await scraper.get_list("https://www.linkedin.com/feed/")

        navigator._navigate_to_page.assert_not_awaited()
        content._extract_root_content.assert_not_awaited()

    async def test_relative_list_url_is_joined_against_linkedin(self):
        scraper, navigator, _content = _scraper(
            ["https://www.linkedin.com/sales/lists/people/12345"]
        )

        await scraper.get_list("/sales/lists/people/12345")

        navigator._navigate_to_page.assert_awaited_once_with(
            "https://www.linkedin.com/sales/lists/people/12345"
        )

    async def test_max_items_stops_pagination_once_reached(self):
        # Raw anchor shape (RawReference), not the Reference build_references
        # returns -- _extract_current_page runs the real build_references
        # over this, same as production, so a shape mismatch here would
        # silently zero out every reference rather than fail loudly.
        member_refs = [
            {
                "href": f"https://www.linkedin.com/in/member-{i}/",
                "text": f"Member {i}",
            }
            for i in range(3)
        ]
        scraper, navigator, _content = _scraper(
            ["https://www.linkedin.com/sales/lists/people/1"],
            page_results=[_page_result("Members page 1", member_refs)],
        )

        result = await scraper.get_list(
            "https://www.linkedin.com/sales/lists/people/1", max_items=2
        )

        assert result["pages_fetched"] == 1
        assert result["stopped_reason"] == "max_items"
        assert len(result["references"]["list_members"]) == 2

    async def test_a_page_with_no_member_references_ends_the_list(self):
        scraper, navigator, _content = _scraper(
            ["https://www.linkedin.com/sales/lists/people/1"],
            page_results=[_page_result("Some chrome text, no members", [])],
        )

        result = await scraper.get_list("https://www.linkedin.com/sales/lists/people/1")

        assert result["pages_fetched"] == 0
        assert result["stopped_reason"] == "empty_page"
        assert result["sections"] == {}
