"""Company page admin analytics: sections, tool refusals, and landing checks.

The browser cases run the real reads against synthetic pages served from a
LinkedIn origin. Redirects are simulated by a page whose script replaces
``location``: slug to numeric id for an admin, back to the public page for a
non-admin. All pages and names are synthetic.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import asyncio

import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import FunctionTool
from patchright.async_api import Page, async_playwright

from linkedin_mcp_server.core.exceptions import InvalidReferenceError
from linkedin_mcp_server.scraping import analytics as analytics_module
from linkedin_mcp_server.scraping.analytics import (
    COMPANY_ANALYTICS_SECTIONS,
    AnalyticsScraper,
    parse_company_analytics_sections,
)
from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.session import ScrapingSession


class TestSections:
    def test_default_is_all_three_in_order(self):
        assert parse_company_analytics_sections(None) == (
            list(COMPANY_ANALYTICS_SECTIONS),
            [],
        )
        assert list(COMPANY_ANALYTICS_SECTIONS) == ["visitors", "followers", "content"]

    def test_selection_is_canonical_with_unknowns_reported(self):
        assert parse_company_analytics_sections("content, Followers,leads") == (
            ["followers", "content"],
            ["leads"],
        )


async def _tool() -> FunctionTool:
    from linkedin_mcp_server.tools.analytics import register_analytics_tools

    mcp = FastMCP("test")
    # Experimental and off by default; these tests cover the tool itself.
    register_analytics_tools(mcp, company_page_tools=True)
    tool = await mcp.get_tool("get_company_page_analytics")
    assert isinstance(tool, FunctionTool)
    return tool


class TestTool:
    async def test_read_only_with_the_three_sections_by_default(self):
        tool = await _tool()
        assert tool.annotations is not None and tool.annotations.readOnlyHint is True
        assert "read" in tool.tags
        assert tool.parameters["properties"]["sections"]["default"] == (
            "visitors,followers,content"
        )

    @pytest.mark.parametrize(
        "company, sections, match",
        [
            ("../feed", None, "company"),
            ("https://www.linkedin.com/in/synthetic-person/", None, "company page"),
            ("synthetic-page", "leads", "Unknown company analytics sections"),
        ],
    )
    async def test_refusals_never_acquire_a_browser(
        self, mock_context, company, sections, match
    ):
        tool = await _tool()
        with patch(
            "linkedin_mcp_server.tools.analytics.get_ready_extractor",
            new=AsyncMock(side_effect=AssertionError("browser acquired")),
        ):
            with pytest.raises(ToolError, match=match):
                await tool.fn(company, mock_context, sections=sections)

    async def test_delegates_the_company_and_sections(self, mock_context):
        tool = await _tool()
        extractor = MagicMock()
        extractor.get_company_page_analytics = AsyncMock(return_value={"url": "u"})
        result = await tool.fn(
            "synthetic-page", mock_context, sections="followers", extractor=extractor
        )
        assert result == {"url": "u"}
        extractor.get_company_page_analytics.assert_awaited_once_with(
            "synthetic-page", "followers"
        )

    async def test_the_scraper_refuses_unknown_sections_before_navigating(self):
        scraper = AnalyticsScraper(MagicMock(), MagicMock(), MagicMock())
        with pytest.raises(InvalidReferenceError):
            await scraper.get_company_page_analytics("synthetic-page", "leads")


# -- browser ------------------------------------------------------------------


class _Session(ScrapingSession):
    async def delay(self, seconds: float) -> None:
        await asyncio.sleep(0)

    async def check_rate_limit(self) -> None:
        return None

    async def dismiss_modal(self) -> bool:
        return False

    async def scroll_body(self, pause_time: float = 1.0, max_scrolls: int = 10) -> None:
        return None


class _Navigator:
    def __init__(self, page: Any):
        self._page = page

    async def _navigate_to_page(self, url: str) -> None:
        await self._page.goto(url)
        await self._page.wait_for_load_state("load")


def _redirect(path: str) -> str:
    return (
        "<html><body><main>x</main>"
        f"<script>location.replace('{path}')</script></body></html>"
    )


def _dashboard(label: str, count: str) -> str:
    filler = "Synthetic page analytics filler. " * 10
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8"></head><body><main>'
        f"<p>{filler}</p><section><div><span>{count}</span></div>"
        f"<div><span>{label}</span></div></section></main></body></html>"
    )


@pytest.fixture
async def dom_page():
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(channel="chromium", headless=True)
            page = await browser.new_page()
        except Exception as exc:  # browser binary missing
            pytest.skip(f"chromium unavailable: {exc}")
        try:
            yield page
        finally:
            await browser.close()


async def _scraper(page: Any, routes: dict[str, str]) -> AnalyticsScraper:
    async def handle(route: Any) -> None:
        path = route.request.url.split("linkedin.com", 1)[1]
        body = routes.get(path, "<html><body><main>public page</main></body></html>")
        await route.fulfill(content_type="text/html; charset=utf-8", body=body)

    await page.route("https://www.linkedin.com/**", handle)
    session = _Session(cast(Page, page))
    return AnalyticsScraper(
        session, cast(Any, _Navigator(page)), PageContentReader(session)
    )


@pytest.mark.browser_dom
@pytest.mark.xdist_group("browser_runtime")
class TestLandingOnTheAdminPages:
    @pytest.fixture(autouse=True)
    def fast_waits(self, monkeypatch):
        monkeypatch.setattr(analytics_module, "_CONTENT_WAIT_MS", 500)

    async def test_an_admin_slug_lands_on_the_numeric_id_and_is_read(self, dom_page):
        base = "/company/synthetic-page/admin/analytics"
        by_id = "/company/12345/admin/analytics"
        scraper = await _scraper(
            dom_page,
            {
                f"{base}/followers/": _redirect(f"{by_id}/followers/"),
                f"{by_id}/followers/": _dashboard("Total followers", "2,500"),
                f"{base}/updates/": _redirect(f"{by_id}/updates/"),
                f"{by_id}/updates/": _dashboard("Impressions", "9,876"),
            },
        )
        result = await scraper.get_company_page_analytics(
            "https://www.linkedin.com/company/synthetic-page/", "followers,content"
        )

        assert list(result["sections"]) == ["followers", "content"]
        assert result["company_id"] == "12345"
        assert result["named_metrics"] == {
            "followers": {"followers": 2500},
            "content": {"impressions": 9876},
        }
        assert "section_errors" not in result

    async def test_a_non_admin_is_not_authorized_per_section(self, dom_page):
        base = "/company/other-page/admin/analytics"
        scraper = await _scraper(
            dom_page,
            {
                f"{base}/visitors/": _redirect("/company/other-page/"),
                f"{base}/followers/": _redirect("/company/other-page/"),
                "/company/other-page/": _dashboard("Followers", "700"),
            },
        )
        result = await scraper.get_company_page_analytics(
            "other-page", "visitors,followers"
        )

        assert result["sections"] == {}
        assert "metrics" not in result
        for name in ("visitors", "followers"):
            error = result["section_errors"][name]
            assert error["error_type"] == "not_authorized"
            assert error["landed_path"] == "/company/other-page/"

    async def test_a_second_section_landing_on_another_page_id_is_refused(
        self, dom_page
    ):
        base = "/company/synthetic-page/admin/analytics"
        scraper = await _scraper(
            dom_page,
            {
                f"{base}/visitors/": _redirect(
                    "/company/12345/admin/analytics/visitors/"
                ),
                "/company/12345/admin/analytics/visitors/": _dashboard(
                    "Page views", "40"
                ),
                f"{base}/followers/": _redirect(
                    "/company/67890/admin/analytics/followers/"
                ),
                "/company/67890/admin/analytics/followers/": _dashboard(
                    "Total followers", "1"
                ),
            },
        )
        result = await scraper.get_company_page_analytics(
            "synthetic-page", "visitors,followers"
        )
        assert list(result["sections"]) == ["visitors"]
        assert result["section_errors"]["followers"]["error_type"] == "not_authorized"


class TestExperimentalSwitch:
    async def test_the_tool_is_not_registered_by_default(self, monkeypatch):
        from linkedin_mcp_server.tools.analytics import register_analytics_tools

        monkeypatch.delenv("ENABLE_COMPANY_PAGE_TOOLS", raising=False)
        mcp = FastMCP("test")
        register_analytics_tools(mcp)

        names = {tool.name for tool in await mcp.list_tools()}
        assert "get_company_page_analytics" not in names
        # The member's own analytics are unaffected by the switch.
        assert {"get_post_analytics", "get_profile_analytics"} <= names

    async def test_the_environment_switch_registers_it(self, monkeypatch):
        from linkedin_mcp_server.tools.analytics import register_analytics_tools

        monkeypatch.setenv("ENABLE_COMPANY_PAGE_TOOLS", "1")
        mcp = FastMCP("test")
        register_analytics_tools(mcp)

        tool = await mcp.get_tool("get_company_page_analytics")
        assert isinstance(tool, FunctionTool)
        assert "EXPERIMENTAL" in (tool.description or "")
