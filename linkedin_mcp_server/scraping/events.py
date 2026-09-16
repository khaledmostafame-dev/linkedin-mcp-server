"""LinkedIn event scraping workflows: search, details, and attendees.

All three are read-only, single-`SectionCapture` workflows -- the same
navigate-scroll-innerText pattern `PostSearch.search_posts` and
`JobScraper.scrape_job` use, with no LinkedIn class names and no
text-based state detection.
"""

from __future__ import annotations

from typing import Any

from linkedin_mcp_server.scraping.capture import (
    CaptureMode,
    CapturePlan,
    SectionCapture,
)
from linkedin_mcp_server.scraping.contracts import (
    RATE_LIMITED_SECTION_TEXT,
    rate_limited_section_error,
)
from linkedin_mcp_server.scraping.identifiers import event_page_url, normalize_event_id
from linkedin_mcp_server.scraping.link_metadata import Reference
from linkedin_mcp_server.scraping.search_urls import build_event_search_url

# Event search is an infinite scroll with no ``&start=`` pagination (same
# shape as PostSearch's content search), so ``max_pages`` caps scroll depth
# rather than fetching discrete pages.
_EVENT_SEARCH_SCROLLS_PER_PAGE = 5
_MIN_SCROLLS = 1
_ATTENDEE_ROWS_PER_SCROLL = 10
_MAX_ATTENDEE_SCROLLS = 20


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


class EventScraper:
    """Own every workflow whose subject is a LinkedIn event."""

    def __init__(self, capture: SectionCapture):
        self._capture = capture

    async def search_events(self, keywords: str, max_pages: int = 3) -> dict[str, Any]:
        """Search for LinkedIn events by keyword.

        Single navigation with bounded scrolling -- there is no discrete
        per-page URL for this surface, so ``max_pages`` is a scroll-depth
        proxy, the same convention ``search_posts`` uses for content search.
        """
        url = build_event_search_url(keywords)
        max_scrolls = max(_MIN_SCROLLS, max_pages) * _EVENT_SEARCH_SCROLLS_PER_PAGE
        extracted = await self._capture.capture(
            url,
            "search_results",
            CapturePlan(CaptureMode.SEARCH_RESULTS, max_scrolls),
        )
        return _section_result(
            url, "search_results", extracted.text, extracted.references, extracted.error
        )

    async def get_event_details(self, event_url: str) -> dict[str, Any]:
        """Get the details page for a single LinkedIn event."""
        event_id = normalize_event_id(event_url)
        url = event_page_url(event_id, "/")
        extracted = await self._capture.capture(url, "event_details", CapturePlan())
        return _section_result(
            url, "event_details", extracted.text, extracted.references, extracted.error
        )

    async def get_event_attendees(
        self, event_url: str, max_attendees: int = 50
    ) -> dict[str, Any]:
        """List attendees of a LinkedIn event.

        The attendee list is only fully visible when the authenticated
        account can see event attendance (public events, or events the
        account is registered for); otherwise the returned text reflects
        whatever LinkedIn actually served.
        """
        event_id = normalize_event_id(event_url)
        url = event_page_url(event_id, "/attendees/")
        scrolls = max(
            _MIN_SCROLLS,
            min(
                _MAX_ATTENDEE_SCROLLS, (max_attendees + 9) // _ATTENDEE_ROWS_PER_SCROLL
            ),
        )
        extracted = await self._capture.capture(
            url, "attendees", CapturePlan(CaptureMode.COMPANY_PEOPLE, scrolls)
        )
        return _section_result(
            url, "attendees", extracted.text, extracted.references, extracted.error
        )
