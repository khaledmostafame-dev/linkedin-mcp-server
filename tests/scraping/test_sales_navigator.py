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


#: Comfortably above ``_APP_SHELL_MIN_NODES`` (40) and stable across repeated
#: samples, so the render-wait settles on its second poll by default. Tests
#: that care about the render gate itself override this explicitly.
_RENDERED_NODE_COUNT = 999


def _scraper(
    landed_urls: list[str],
    page_results: list | None = None,
    *,
    node_counts: int | list[int] = _RENDERED_NODE_COUNT,
    upsell_present: bool = False,
):
    """Build a scraper whose landed URL and extracted page advance per call.

    ``landed_urls`` feeds ``navigator._navigate_to_page`` (one entry per
    navigation); ``page_results`` feeds ``content._extract_root_content``
    (one entry per extraction, defaulting to a single successful page
    repeated for every call if only one is given). ``node_counts`` feeds the
    app-shell render wait's ``document.querySelectorAll('*').length`` poll
    (a constant repeats every sample; a list is consumed one value per
    poll, repeating its last entry once exhausted). ``upsell_present`` makes
    the rendered page's premium-link check trip.
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
    session.delay = AsyncMock()

    if isinstance(node_counts, int):
        page.evaluate = AsyncMock(return_value=node_counts)
    else:
        counts = list(node_counts)

        async def evaluate(_script: str) -> int:
            index = min(evaluate.calls, len(counts) - 1)
            evaluate.calls += 1
            return counts[index]

        evaluate.calls = 0
        page.evaluate = AsyncMock(side_effect=evaluate)

    upsell_locator = MagicMock()
    upsell_locator.count = AsyncMock(return_value=1 if upsell_present else 0)
    page.locator = MagicMock(return_value=upsell_locator)

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


class TestAppShellRenderGate:
    """The SPA-render wait (live capture 2026-09-17: a real seat landed on
    `/sales/search/people` with only 15 DOM nodes -- the bare app shell,
    read before its JS bundle rendered anything). None of these scenarios
    may report `empty_page`: that value must mean "a real, rendered page
    had nothing on it", never "we read a page before it could render".
    """

    async def test_app_never_renders_reports_unavailable_not_empty_page(self):
        scraper, _navigator, content = _scraper(
            ["https://www.linkedin.com/sales/search/people?keywords=ada&page=1"],
            node_counts=15,  # stays at bare-shell size for the whole budget
        )

        result = await scraper.search_leads("ada")

        assert result["sections"] == {}
        assert result["pages_fetched"] == 0
        assert result["stopped_reason"] == "app_not_rendered"
        assert (
            result["section_errors"]["search_results"]["error_type"]
            == SALES_NAVIGATOR_UNAVAILABLE_ERROR
        )
        content._extract_root_content.assert_not_awaited()

    async def test_a_rendered_page_proceeds_normally(self):
        """The render gate doesn't get in the way of the ordinary success path."""
        scraper, _navigator, content = _scraper(
            ["https://www.linkedin.com/sales/search/people?keywords=ada&page=1"]
        )

        result = await scraper.search_leads("ada")

        assert result["sections"]["search_results"] == "Lead: Ada Lovelace"
        assert result["pages_fetched"] == 1
        assert "section_errors" not in result
        content._extract_root_content.assert_awaited()

    async def test_upsell_link_after_render_reports_unavailable(self):
        """A rendered page whose only content is an upgrade prompt.

        Locale-independent per AGENTS.md: a premium/upsell href path, never
        page text, is what marks this as "no seat", not "no results".
        """
        scraper, _navigator, content = _scraper(
            ["https://www.linkedin.com/sales/search/people?keywords=ada&page=1"],
            upsell_present=True,
        )

        result = await scraper.search_leads("ada")

        assert result["sections"] == {}
        assert result["stopped_reason"] == "app_not_rendered"
        assert (
            result["section_errors"]["search_results"]["error_type"]
            == SALES_NAVIGATOR_UNAVAILABLE_ERROR
        )
        content._extract_root_content.assert_not_awaited()

    async def test_client_side_redirect_during_render_wait_is_not_empty_page(self):
        """A redirect off `/sales/` that fires only after the initial `goto`.

        `_navigate_and_check_seat` only reads the landed URL right after
        navigation resolves; a seat-check redirect LinkedIn's own JS fires
        only once it finishes checking access happens later, while
        `_wait_for_app_render` is still polling. This proves that later
        redirect is still caught, and still reported as
        `sales_navigator_unavailable`, never `empty_page`.
        """
        session = MagicMock()
        page = MagicMock()
        page.url = "https://www.linkedin.com/sales/search/people?keywords=ada&page=1"
        session.page = page
        session.check_rate_limit = AsyncMock()
        session.dismiss_modal = AsyncMock()
        session.scroll_body = AsyncMock()
        session.delay = AsyncMock()

        async def evaluate(_script: str) -> int:
            evaluate.calls += 1
            if evaluate.calls == 2:
                page.url = "https://www.linkedin.com/premium/products/"
            return 999

        evaluate.calls = 0
        page.evaluate = AsyncMock(side_effect=evaluate)
        upsell_locator = MagicMock()
        upsell_locator.count = AsyncMock(return_value=0)
        page.locator = MagicMock(return_value=upsell_locator)

        navigator = MagicMock()
        navigator._navigate_to_page = AsyncMock()  # url is already set, above
        content = MagicMock()
        content._extract_root_content = AsyncMock(return_value=_page_result())

        scraper = SalesNavigatorScraper(session, navigator, content)

        result = await scraper.search_leads("ada")

        assert result["stopped_reason"] == "app_not_rendered"
        assert (
            result["section_errors"]["search_results"]["error_type"]
            == SALES_NAVIGATOR_UNAVAILABLE_ERROR
        )
        content._extract_root_content.assert_not_awaited()

    async def test_render_gate_also_covers_get_lists(self):
        scraper, _navigator, content = _scraper(
            ["https://www.linkedin.com/sales/lists/people"], node_counts=15
        )

        result = await scraper.get_lists("leads")

        assert result["sections"] == {}
        assert (
            result["section_errors"]["lists"]["error_type"]
            == SALES_NAVIGATOR_UNAVAILABLE_ERROR
        )
        content._extract_root_content.assert_not_awaited()

    async def test_render_gate_also_covers_get_list(self):
        scraper, _navigator, content = _scraper(
            ["https://www.linkedin.com/sales/lists/people/1"], node_counts=15
        )

        result = await scraper.get_list("https://www.linkedin.com/sales/lists/people/1")

        assert result["pages_fetched"] == 0
        assert result["stopped_reason"] == "app_not_rendered"
        assert (
            result["section_errors"]["list_members"]["error_type"]
            == SALES_NAVIGATOR_UNAVAILABLE_ERROR
        )
        content._extract_root_content.assert_not_awaited()
