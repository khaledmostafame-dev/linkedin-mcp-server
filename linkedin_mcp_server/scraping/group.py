"""LinkedIn group scraping workflows: search, posts, and member listings."""

from __future__ import annotations

import logging
from typing import Any

from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.core.exceptions import LinkedInScraperException
from linkedin_mcp_server.error_diagnostics import build_issue_diagnostics
from linkedin_mcp_server.scraping.capture import (
    CaptureMode,
    CapturePlan,
    SectionCapture,
)
from linkedin_mcp_server.scraping.contracts import (
    RATE_LIMITED_SECTION_TEXT,
    ExtractedSection,
    rate_limited_section_error,
)
from linkedin_mcp_server.scraping.identifiers import group_page_url, normalize_group_id
from linkedin_mcp_server.scraping.link_metadata import Reference
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.search_urls import build_group_search_url
from linkedin_mcp_server.scraping.session import ScrapingSession

logger = logging.getLogger(__name__)


def _section_result(
    url: str,
    section_name: str,
    text: str,
    references: list[Reference],
    error: dict[str, Any] | None,
) -> dict[str, Any]:
    sections: dict[str, str] = {}
    references_out: dict[str, list[Reference]] = {}
    section_errors: dict[str, dict[str, Any]] = {}
    if text and text != RATE_LIMITED_SECTION_TEXT:
        sections[section_name] = text
        if references:
            references_out[section_name] = references
    elif text == RATE_LIMITED_SECTION_TEXT:
        section_errors[section_name] = rate_limited_section_error()
    elif error:
        section_errors[section_name] = error

    result: dict[str, Any] = {"url": url, "sections": sections}
    if references_out:
        result["references"] = references_out
    if section_errors:
        result["section_errors"] = section_errors
    return result


class GroupScraper:
    """Own every workflow whose subject is a LinkedIn group."""

    def __init__(
        self,
        session: ScrapingSession,
        navigator: PageNavigator,
        capture: SectionCapture,
    ):
        self._session = session
        self._navigator = navigator
        self._capture = capture

    async def search_groups(self, keywords: str) -> dict[str, Any]:
        """Search for LinkedIn groups by keyword.

        Single navigation, no pagination -- modeled on search_people /
        search_companies.
        """
        url = build_group_search_url(keywords)
        extracted = await self._capture.capture(
            url, "search_results", CapturePlan(CaptureMode.SEARCH_RESULTS)
        )
        return _section_result(
            url, "search_results", extracted.text, extracted.references, extracted.error
        )

    async def get_group_posts(
        self, group_id: str, max_posts: int = 20
    ) -> dict[str, Any]:
        """List recent posts from a group's home/activity feed."""
        group_id = normalize_group_id(group_id)
        url = group_page_url(group_id, "/")
        scrolls = max(1, min(10, (max_posts + 4) // 5))
        extracted = await self._capture.capture(
            url, "posts", CapturePlan(CaptureMode.ACTIVITY, scrolls)
        )
        return _section_result(
            url, "posts", extracted.text, extracted.references, extracted.error
        )

    async def get_group_members(
        self,
        group_id: str,
        max_members: int = 50,
        keywords: str | None = None,
    ) -> dict[str, Any]:
        """List members of a group from its /members/ page.

        The full member list is only visible when the authenticated account
        is a member of the group; otherwise LinkedIn typically shows a
        restricted preview or redirects to the group landing page, and the
        returned text reflects whatever the page actually served.

        keywords filters server-side via the page's member-search box -- a
        structural ``main input[type="text"]`` selector, not a URL param
        LinkedIn would otherwise silently ignore.
        """
        group_id = normalize_group_id(group_id)
        url = group_page_url(group_id, "/members/")
        scrolls = max(1, min(20, (max_members + 9) // 10))

        if keywords:
            extracted = await self._extract_filtered_members(url, keywords, scrolls)
        else:
            extracted = await self._capture.capture(
                url, "members", CapturePlan(CaptureMode.COMPANY_PEOPLE, scrolls)
            )
        return _section_result(
            url, "members", extracted.text, extracted.references, extracted.error
        )

    async def _extract_filtered_members(
        self, url: str, keywords: str, scrolls: int
    ) -> ExtractedSection:
        """Filter the group member list via the page's search box, then extract.

        The members page exposes exactly one text input inside <main> (the
        member search box), so the structural selector is
        locale-independent -- placeholder/aria text is not read. Filling it
        triggers a server-side filtered fetch that replaces the listing in
        place (the URL does not change), so extraction is done via the
        capture's post-navigation seam directly rather than through
        ``capture()``, which would re-navigate and discard the filter just
        applied.
        """
        try:
            await self._navigator._navigate_to_page(url)
            await self._session.check_rate_limit()
            try:
                await self._session.page.wait_for_selector(
                    'main input[type="text"]', timeout=5000
                )
            except PlaywrightTimeoutError:
                logger.debug("Member search input did not appear on %s", url)

            search_box = self._session.page.locator('main input[type="text"]').first
            if await search_box.count() > 0:
                await search_box.click()
                await search_box.fill(keywords)
                # Debounced server-side fetch replaces the listing in place.
                await self._session.delay(2.5)
            else:
                logger.warning(
                    "No member search input on %s; returning unfiltered list", url
                )

            return await self._capture._extract_loaded_section(
                url, "members", CapturePlan(CaptureMode.COMPANY_PEOPLE, scrolls)
            )
        except LinkedInScraperException:
            raise
        except Exception as e:
            logger.warning("Failed to extract filtered group members %s: %s", url, e)
            return ExtractedSection(
                text="",
                references=[],
                error=build_issue_diagnostics(
                    e,
                    context="get_group_members",
                    target_url=url,
                    section_name="members",
                ),
            )
