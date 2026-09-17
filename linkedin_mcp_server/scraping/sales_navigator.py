"""Sales Navigator search and list workflows (read-only, no writes).

Sales Navigator (``/sales/...``) is a separate paid LinkedIn seat. An account
without one is redirected away from ``/sales/`` the moment it tries to load
any Sales Navigator page. Detected by the landed URL, never by page text, per
AGENTS.md's Scraping Rules ("detection must be locale-independent"): every
method here checks the landed URL immediately after navigating and, on a
redirect, returns a ``sales_navigator_unavailable`` section_error instead of
navigating further (no scroll, no content extraction, no second page).

The URL shapes below (``/sales/search/people``, ``/sales/search/companies``,
``/sales/lists/people``, ``/sales/lists/company``, and the flat
``keywords=``/``page=`` query params) are this project's own construction,
not confirmed against a live Sales Navigator seat. LinkedIn's real Sales
Navigator search is widely reported to encode filters inside one structured
``query`` parameter rather than flat query params; that shape is not
reproduced here for lack of a live capture to verify it against. See this
change's PR description for exactly what needs live verification.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus, urljoin, urlparse

import logging

from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.contracts import (
    RATE_LIMITED_SECTION_TEXT,
    rate_limited_section_error,
)
from linkedin_mcp_server.scraping.link_metadata import Reference, build_references
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession
from linkedin_mcp_server.scraping.text import (
    filter_linkedin_noise_lines,
    truncate_linkedin_noise,
)

logger = logging.getLogger(__name__)

SALES_NAVIGATOR_UNAVAILABLE_ERROR = "sales_navigator_unavailable"
SALES_NAVIGATOR_UNAVAILABLE_MESSAGE = (
    "LinkedIn redirected away from Sales Navigator. This account has no "
    "Sales Navigator seat, or the session lost access to one."
)

# ASSUMED, unverified against a live seat -- see module docstring.
_LEADS_SEARCH_URL = "https://www.linkedin.com/sales/search/people"
_ACCOUNTS_SEARCH_URL = "https://www.linkedin.com/sales/search/companies"
_LEADS_LISTS_URL = "https://www.linkedin.com/sales/lists/people"
_ACCOUNTS_LISTS_URL = "https://www.linkedin.com/sales/lists/company"

# Hard ceilings independent of the caller-supplied bound, so a runaway filter
# value or an unexpectedly deep list can't turn a bounded call into an
# unbounded one.
_MAX_SEARCH_PAGES = 10
_MAX_LIST_PAGES = 10

# Live capture 2026-09-17 (`009-extractor-after-goto.json`, `/sales/search/
# people?keywords=...`) landed on a page that stayed under `/sales/` (the
# seat check passed) but held only 15 DOM nodes total: five empty SPA outlet
# mounts (`hue-web-menu-outlet`, `hue-web-modal-outlet`,
# `hue-web-tooltip-outlet`, `hue-web-typeahead-outlet`,
# `artdeco-modal-outlet`) and their ancestors, none of them a result or any
# other real content. That is the bare app shell captured before its JS
# bundle has rendered anything, not a genuinely empty results page -- every
# other captured page in this same session had 700+ DOM nodes once loaded.
# Reading immediately, as this module used to, reports that as `empty_page`,
# indistinguishable from a seat with zero real results. `_wait_for_app_render`
# waits for the node count to grow past bare-shell size and settle before
# any extraction happens; a page that never clears that bar is reported as
# `sales_navigator_unavailable`, never `empty_page`.
_APP_SHELL_MIN_NODES = 40
_APP_SHELL_WAIT_ATTEMPTS = 10
_APP_SHELL_WAIT_INTERVAL_SECONDS = 0.5
# A locale-independent signal that the SPA rendered an upgrade prompt instead
# of results -- LinkedIn's own upsell/premium flows all live under this path
# segment regardless of UI language.
_UPSELL_HREF_SELECTOR = 'a[href*="/premium"]'


def _is_on_sales_navigator(url: str) -> bool:
    """True while the landed URL is still under LinkedIn's ``/sales/`` tree."""
    return "/sales/" in urlparse(url).path


def _seat_unavailable_error() -> dict[str, str]:
    return {
        "error_type": SALES_NAVIGATOR_UNAVAILABLE_ERROR,
        "error_message": SALES_NAVIGATOR_UNAVAILABLE_MESSAGE,
    }


def _build_search_url(
    base_url: str,
    keywords: str,
    filters: dict[str, Any] | None,
    page: int = 1,
) -> str:
    """Build a Sales Navigator search URL for one page of results.

    See the module docstring: flat ``keywords=``/filter-key=value params are
    this project's simplification of Sales Navigator's real (structured,
    single-``query``-parameter) URL grammar, not a confirmed reproduction of
    it.
    """
    params = f"keywords={quote_plus(keywords)}"
    for key, value in (filters or {}).items():
        params += f"&{quote_plus(str(key))}={quote_plus(str(value))}"
    if page > 1:
        params += f"&page={page}"
    return f"{base_url}?{params}"


def _list_kind_url(kind: str) -> str:
    if kind == "leads":
        return _LEADS_LISTS_URL
    if kind == "accounts":
        return _ACCOUNTS_LISTS_URL
    raise ValueError(f"kind must be 'leads' or 'accounts', got {kind!r}")


class _AppNotRendered(Exception):
    """Internal marker: the Sales Navigator SPA never mounted real content.

    Raised by ``_extract_current_page`` and caught by each public method so
    every caller reports the same ``sales_navigator_unavailable`` shape as
    the immediate-redirect case in ``_navigate_and_check_seat``, instead of
    reading (and mis-scoring as empty) a page that was never given the
    chance to render.
    """

    def __init__(self, landed_url: str):
        super().__init__(landed_url)
        self.landed_url = landed_url


class SalesNavigatorScraper:
    """Own every read-only Sales Navigator workflow (leads, accounts, lists).

    No InMail, connect, save, or list-membership writes live here or are
    planned for this owner -- every method below only navigates and reads.
    """

    def __init__(
        self,
        session: ScrapingSession,
        navigator: PageNavigator,
        content: PageContentReader,
    ):
        self._session = session
        self._navigator = navigator
        self._content = content

    async def _navigate_and_check_seat(self, url: str) -> dict[str, Any] | None:
        """Navigate to a Sales Navigator URL; build an early result on redirect.

        Returns ``None`` when the seat check passed (the landed URL is still
        under ``/sales/``), so the caller goes on to extract content. Returns
        a complete result -- empty sections, one ``section_errors`` entry,
        no further navigation attempted -- when LinkedIn redirected the
        account away from Sales Navigator.
        """
        await self._navigator._navigate_to_page(url)
        landed_url = self._session.page.url
        if _is_on_sales_navigator(landed_url):
            return None
        logger.info(
            "Sales Navigator redirected to %s; no seat on this account", landed_url
        )
        return {
            "url": landed_url,
            "sections": {},
            "section_errors": {"search_results": _seat_unavailable_error()},
        }

    async def _wait_for_app_render(self) -> bool:
        """Bounded poll for the SPA shell to mount real content.

        Polls the live DOM node count rather than any specific selector,
        since the only capture available shows the bare shell has none of
        its own (see the module-level comment on ``_APP_SHELL_MIN_NODES`).
        "Settled" means two consecutive samples at or above the shell-size
        floor, the same idea ``post_composer._open_scheduled_list`` uses for
        its own dynamic dialog. Returns False when the count never reaches
        the floor within the attempt budget.
        """
        previous: int | None = None
        for _ in range(_APP_SHELL_WAIT_ATTEMPTS):
            await self._session.delay(_APP_SHELL_WAIT_INTERVAL_SECONDS)
            count = await self._session.page.evaluate(
                "document.querySelectorAll('*').length"
            )
            if not isinstance(count, int):
                continue
            if count >= _APP_SHELL_MIN_NODES and count == previous:
                return True
            previous = count
        return isinstance(previous, int) and previous >= _APP_SHELL_MIN_NODES

    async def _extract_current_page(
        self, section_name: str, *, max_scrolls: int = 5
    ) -> tuple[str, list[Reference]]:
        """Scroll and extract innerText + references from the loaded page.

        Raises ``_AppNotRendered`` -- never returns an ``empty_page``-shaped
        result -- when the app shell never grows into real content, when a
        client-side redirect carries the page off ``/sales/`` only after
        that wait (the immediate seat check in ``_navigate_and_check_seat``
        cannot see a redirect that fires later), or when the rendered page
        shows an upsell/premium link instead of results.
        """
        await self._session.check_rate_limit()
        await self._session.dismiss_modal()

        rendered = await self._wait_for_app_render()
        landed_url = self._session.page.url
        if not _is_on_sales_navigator(landed_url):
            logger.info(
                "Sales Navigator redirected to %s after render; no seat",
                landed_url,
            )
            raise _AppNotRendered(landed_url)
        if not rendered:
            logger.info(
                "Sales Navigator app shell never rendered content on %s",
                landed_url,
            )
            raise _AppNotRendered(landed_url)
        if await self._session.page.locator(_UPSELL_HREF_SELECTOR).count() > 0:
            logger.info(
                "Sales Navigator showed an upsell/premium prompt on %s", landed_url
            )
            raise _AppNotRendered(landed_url)

        await self._session.scroll_body(pause_time=0.5, max_scrolls=max_scrolls)

        raw_result = await self._content._extract_root_content(["main"])
        raw = raw_result["text"]
        if not raw:
            return "", []
        truncated = truncate_linkedin_noise(raw)
        if not truncated and raw.strip():
            logger.warning(
                "Sales Navigator page returned only LinkedIn chrome "
                "(likely rate-limited)"
            )
            return RATE_LIMITED_SECTION_TEXT, []
        cleaned = filter_linkedin_noise_lines(truncated)
        return cleaned, build_references(raw_result["references"], section_name)

    async def _bounded_search(
        self,
        base_url: str,
        keywords: str,
        filters: dict[str, Any] | None,
        max_pages: int,
        section_name: str,
    ) -> dict[str, Any]:
        """Fetch up to ``max_pages`` of a Sales Navigator search, bounded."""
        first_url = _build_search_url(base_url, keywords, filters, page=1)
        early = await self._navigate_and_check_seat(first_url)
        if early is not None:
            early["pages_fetched"] = 0
            early["stopped_reason"] = "no_seat"
            return early

        capped_pages = min(max(max_pages, 1), _MAX_SEARCH_PAGES)
        page_texts: list[str] = []
        all_references: list[Reference] = []
        section_errors: dict[str, dict[str, Any]] = {}
        pages_fetched = 0
        stopped_reason = "max_pages"

        for page_number in range(1, capped_pages + 1):
            if page_number > 1:
                await self._navigator._navigate_to_page(
                    _build_search_url(base_url, keywords, filters, page=page_number)
                )

            try:
                text, refs = await self._extract_current_page(section_name)
            except _AppNotRendered:
                section_errors[section_name] = _seat_unavailable_error()
                stopped_reason = "app_not_rendered"
                break
            if text == RATE_LIMITED_SECTION_TEXT:
                section_errors[section_name] = rate_limited_section_error()
                stopped_reason = "rate_limited"
                break
            if not text:
                stopped_reason = "no_more_results" if pages_fetched else "empty_page"
                break

            page_texts.append(text)
            all_references.extend(refs)
            pages_fetched += 1

        sections: dict[str, str] = {}
        if page_texts:
            sections[section_name] = "\n\n".join(page_texts)

        references: dict[str, list[Reference]] = {}
        if all_references:
            references[section_name] = all_references

        result: dict[str, Any] = {
            "url": first_url,
            "sections": sections,
            "pages_fetched": pages_fetched,
            "stopped_reason": stopped_reason,
        }
        if references:
            result["references"] = references
        if section_errors:
            result["section_errors"] = section_errors
        return result

    async def search_leads(
        self,
        keywords: str,
        filters: dict[str, Any] | None = None,
        max_pages: int = 1,
    ) -> dict[str, Any]:
        """Search Sales Navigator leads (people). Requires a Sales Navigator seat.

        Returns:
            Dict with url, sections (search_results -> raw text),
            pages_fetched, stopped_reason, and optional references /
            section_errors. ``section_errors["search_results"]`` carries
            ``error_type: "sales_navigator_unavailable"`` when the account
            has no Sales Navigator seat.
        """
        return await self._bounded_search(
            _LEADS_SEARCH_URL, keywords, filters, max_pages, "search_results"
        )

    async def search_accounts(
        self,
        keywords: str,
        filters: dict[str, Any] | None = None,
        max_pages: int = 1,
    ) -> dict[str, Any]:
        """Search Sales Navigator accounts (companies). Requires a seat.

        Same result shape as :meth:`search_leads`.
        """
        return await self._bounded_search(
            _ACCOUNTS_SEARCH_URL, keywords, filters, max_pages, "search_results"
        )

    async def get_lists(self, kind: str = "leads") -> dict[str, Any]:
        """List the authenticated user's Sales Navigator lead/account lists.

        Args:
            kind: "leads" or "accounts".

        Returns:
            Dict with url, sections (lists -> raw text), and optional
            references (list URLs, usable with :meth:`get_list`) /
            section_errors.
        """
        url = _list_kind_url(kind)
        early = await self._navigate_and_check_seat(url)
        if early is not None:
            return early

        try:
            text, refs = await self._extract_current_page("lists")
        except _AppNotRendered as exc:
            return {
                "url": exc.landed_url,
                "sections": {},
                "section_errors": {"lists": _seat_unavailable_error()},
            }
        sections: dict[str, str] = {}
        section_errors: dict[str, dict[str, Any]] = {}
        references: dict[str, list[Reference]] = {}
        if text == RATE_LIMITED_SECTION_TEXT:
            section_errors["lists"] = rate_limited_section_error()
        elif text:
            sections["lists"] = text
            if refs:
                references["lists"] = refs

        result: dict[str, Any] = {"url": url, "sections": sections}
        if references:
            result["references"] = references
        if section_errors:
            result["section_errors"] = section_errors
        return result

    async def get_list(self, list_url: str, max_items: int = 100) -> dict[str, Any]:
        """Read one Sales Navigator list's members, bounded by ``max_items``.

        Args:
            list_url: A Sales Navigator list URL, e.g. one returned by
                :meth:`get_lists` in ``references["lists"]``. A relative path
                is joined against ``https://www.linkedin.com``.
            max_items: Stop once this many members have been collected.

        Returns:
            Dict with url, sections (list_members -> raw text),
            pages_fetched, stopped_reason, and optional references (member
            URLs, capped at max_items) / section_errors.
        """
        url = urljoin("https://www.linkedin.com", list_url)
        if "/sales/" not in urlparse(url).path:
            raise ValueError(
                "list_url must be a Sales Navigator URL (path containing "
                f"'/sales/'); got {list_url!r}"
            )

        early = await self._navigate_and_check_seat(url)
        if early is not None:
            early["pages_fetched"] = 0
            early["stopped_reason"] = "no_seat"
            return early

        page_texts: list[str] = []
        all_references: list[Reference] = []
        section_errors: dict[str, dict[str, Any]] = {}
        pages_fetched = 0
        stopped_reason = "end_of_list"
        current_url = url

        for page_number in range(1, _MAX_LIST_PAGES + 1):
            if page_number > 1:
                current_url = f"{url}{'&' if '?' in url else '?'}page={page_number}"
                await self._navigator._navigate_to_page(current_url)

            try:
                text, refs = await self._extract_current_page(
                    "list_members", max_scrolls=10
                )
            except _AppNotRendered:
                section_errors["list_members"] = _seat_unavailable_error()
                stopped_reason = "app_not_rendered"
                break
            if text == RATE_LIMITED_SECTION_TEXT:
                section_errors["list_members"] = rate_limited_section_error()
                stopped_reason = "rate_limited"
                break
            if not text or not refs:
                # Empty text, or text with no member references: either the
                # list is shorter than one page, or LinkedIn's markup for
                # member rows changed. Either way, another navigation would
                # only refetch the same page, not new members.
                stopped_reason = "end_of_list" if pages_fetched else "empty_page"
                break

            page_texts.append(text)
            all_references.extend(refs)
            pages_fetched += 1

            if len(all_references) >= max_items:
                stopped_reason = "max_items"
                break
        else:
            stopped_reason = "max_pages"

        sections: dict[str, str] = {}
        if page_texts:
            sections["list_members"] = "\n\n".join(page_texts)

        references: dict[str, list[Reference]] = {}
        if all_references:
            references["list_members"] = all_references[:max_items]

        result: dict[str, Any] = {
            "url": url,
            "sections": sections,
            "pages_fetched": pages_fetched,
            "stopped_reason": stopped_reason,
        }
        if references:
            result["references"] = references
        if section_errors:
            result["section_errors"] = section_errors
        return result
