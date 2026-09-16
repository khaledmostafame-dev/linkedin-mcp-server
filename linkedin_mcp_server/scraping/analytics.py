"""The signed-in member's own analytics: one post, and the profile dashboards.

Both surfaces are "private to you" pages under ``/analytics/``. LinkedIn serves
them only to their owner and sends anyone else somewhere else, so whether a
page is the requested one is decided by the URL it landed on, never by its
text: a post analytics page that lands off ``/analytics/post-summary/`` is
reported as not authorized, and a dashboard that lands off its own route is
reported as redirected. Nothing is guessed from what the page says.

Every section is one navigation (AGENTS.md Scraping Rules). Sections carry the
page's raw text; ``metrics`` adds the numbers that could be read structurally:
an element whose whole text is a count, paired with the nearest text beside it
as its label. The label is returned as LinkedIn printed it. Mapping a label to
a stable name (``impressions``, ``reactions`` …) needs the words, so it goes
through the explicit per-locale table ``METRIC_LABELS`` and only English is
listed; in any other UI language the raw pairs are still returned and
``named_metrics`` is simply absent.

Dashboard routes follow upstream PR 571, which verified them live on
2026-07-07. The post-summary route is the one LinkedIn's own "View analytics"
link on a member's post points to; for a ``ugcPost`` or ``share`` URN the
activity URN is read from that link on the post page.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.parse import unquote, urlparse

import logging
import re
import unicodedata

from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.core.exceptions import (
    InvalidReferenceError,
    LinkedInScraperException,
)
from linkedin_mcp_server.error_diagnostics import build_issue_diagnostics
from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.contracts import rate_limited_section_error
from linkedin_mcp_server.scraping.identifiers import (
    company_page_url,
    normalize_company_identifier,
    normalize_post_urn,
    post_update_url,
)
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import NAV_DELAY, ScrapingSession
from linkedin_mcp_server.scraping.text import (
    filter_linkedin_noise_lines,
    truncate_linkedin_noise,
)

logger = logging.getLogger(__name__)

_BASE = "https://www.linkedin.com"
POST_ANALYTICS_SECTION = "post_analytics"

# Section -> (route navigated to, routes that count as having landed there).
# LinkedIn has served profile views and search appearances under /me/ as well,
# so both spellings are accepted as the same page.
PROFILE_ANALYTICS_SECTIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "profile_viewers": (
        "/analytics/profile-views/",
        ("/analytics/profile-views/", "/me/profile-views/"),
    ),
    "search_appearances": (
        "/analytics/search-appearances/",
        ("/analytics/search-appearances/", "/me/search-appearances/"),
    ),
    "followers": ("/analytics/creator/audience/", ("/analytics/creator/audience/",)),
    "post_impressions": (
        "/analytics/creator/content/",
        ("/analytics/creator/content/",),
    ),
}

# Company page admin analytics: section -> route under /company/<page>/admin/.
# Only a page's admins are served these; anyone else is sent back to the public
# page, which is what the landing check below reports as not authorized.
COMPANY_ANALYTICS_SECTIONS: dict[str, str] = {
    "visitors": "analytics/visitors/",
    "followers": "analytics/followers/",
    "content": "analytics/updates/",
}

# Label text -> stable metric name, per UI locale. Text is the only signal that
# names a stat card, so this is the explicit per-locale table the Scraping Rules
# require; a locale without an entry gets raw label/value pairs only.
METRIC_LABELS: dict[str, dict[str, str]] = {
    "en": {
        "impressions": "impressions",
        "members reached": "members_reached",
        "reactions": "reactions",
        "comments": "comments",
        "reposts": "reposts",
        "saves": "saves",
        "sends on linkedin": "sends",
        "profile viewers from this post": "profile_viewers_from_post",
        "followers gained from this post": "followers_gained",
        "profile viewers": "profile_viewers",
        "search appearances": "search_appearances",
        "total followers": "followers",
        "new followers": "new_followers",
        "page views": "page_views",
        "unique visitors": "unique_visitors",
        "clicks": "clicks",
    }
}

_POST_SUMMARY_PREFIX = "/analytics/post-summary/"
_ACTIVITY_IN_HREF_RE = re.compile(r"/analytics/post-summary/(urn:li:activity:[0-9]+)")
_BIDI_AND_SPACES_RE = re.compile(
    "[\u200e\u200f\u061c\u202a-\u202e\u2066-\u2069\u00a0\u202f\u2009\u2007\\s]"
)
_GROUPED_RE = re.compile(r"^[0-9]{1,3}(?:[,.\u066C'][0-9]{3})+$")
_CONTENT_WAIT_MS = 10_000
_MAX_METRICS = 60


def parse_count(value: str) -> int | None:
    """An integer count as LinkedIn prints it, in any digit script, or ``None``.

    Bidirectional marks and spaces are dropped, every Unicode decimal digit is
    read by its value (so Arabic-Indic ``\u0661\u066c\u0662\u0663\u0664`` is 1234), and a grouping
    separator is accepted only between groups of exactly three digits. Anything
    else — a decimal, a percentage, a ``1.2K`` whose suffix is a word — is not a
    count this function will guess at.
    """
    if not isinstance(value, str):
        return None
    compact = _BIDI_AND_SPACES_RE.sub("", value)
    if not compact:
        return None
    digits: list[str] = []
    for character in compact:
        decimal = unicodedata.decimal(character, None)
        if decimal is not None:
            digits.append(str(decimal))
        elif character in ",.\u066c'":
            digits.append(character)
        else:
            return None
    text = "".join(digits)
    if text.isdigit():
        return int(text)
    if _GROUPED_RE.match(text):
        return int(re.sub(r"[^0-9]", "", text))
    return None


def parse_profile_analytics_sections(
    sections: str | None,
) -> tuple[list[str], list[str]]:
    """``(requested in canonical order, unknown names)``; empty selects all."""
    if sections is None or not sections.strip():
        return list(PROFILE_ANALYTICS_SECTIONS), []
    names = [name.strip().lower() for name in sections.split(",") if name.strip()]
    unknown = sorted({name for name in names if name not in PROFILE_ANALYTICS_SECTIONS})
    requested = [name for name in PROFILE_ANALYTICS_SECTIONS if name in names]
    return requested, unknown


_COMPANY_ADMIN_PATH_RE = re.compile(r"^/company/([^/]+)/admin/(analytics/[a-z-]+/)")


def parse_company_analytics_sections(
    sections: str | None,
) -> tuple[list[str], list[str]]:
    """``(requested in canonical order, unknown names)``; empty selects all."""
    if sections is None or not sections.strip():
        return list(COMPANY_ANALYTICS_SECTIONS), []
    names = [name.strip().lower() for name in sections.split(",") if name.strip()]
    unknown = sorted({name for name in names if name not in COMPANY_ANALYTICS_SECTIONS})
    requested = [name for name in COMPANY_ANALYTICS_SECTIONS if name in names]
    return requested, unknown


def _starts_with(*prefixes: str) -> Callable[[str], bool]:
    """A landing check that accepts a path under any of *prefixes*."""
    return lambda path: path.startswith(prefixes)


def post_analytics_url(activity_urn: str) -> str:
    return f"{_BASE}{_POST_SUMMARY_PREFIX}{activity_urn}/"


def build_metrics(pairs: Any) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Parse the browser's label/value pairs into metrics and named metrics."""
    metrics: list[dict[str, Any]] = []
    named: dict[str, int] = {}
    if not isinstance(pairs, list):
        return metrics, named
    table = METRIC_LABELS["en"]
    for pair in pairs[:_MAX_METRICS]:
        if not isinstance(pair, dict):
            continue
        label = pair.get("label")
        raw = pair.get("value")
        if not isinstance(label, str) or not isinstance(raw, str):
            continue
        value = parse_count(raw)
        label = " ".join(label.split())[:120]
        if value is None or not label:
            continue
        metrics.append({"label": label, "value": value})
        name = table.get(label.lower())
        if name is not None and name not in named:
            named[name] = value
    return metrics, named


_METRIC_PAIRS_JS = r"""() => {
    const root = document.querySelector('main') || document.body;
    const COUNT = /^[\p{Nd}\s,.\u066B\u066C'\u200E\u200F\u061C\u202A-\u202E\u2066-\u2069\u00A0\u202F]+$/u;
    const DIGIT = /\p{Nd}/u;
    // Visible text beside a count: every text node under the ancestor except
    // the count's own, and except other counts, in document order.
    const besides = (above, element) => {
        const texts = [];
        const nodes = document.createTreeWalker(above, NodeFilter.SHOW_TEXT);
        for (let node = nodes.nextNode(); node; node = nodes.nextNode()) {
            if (element.contains(node)) continue;
            const text = (node.nodeValue || '').replace(/\s+/g, ' ').trim();
            const parent = node.parentElement;
            if (!text || COUNT.test(text) || !parent || !parent.getClientRects().length) {
                continue;
            }
            texts.push(text);
        }
        return texts;
    };
    const pairs = [];
    const seen = new Set();
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
    for (let element = walker.nextNode(); element; element = walker.nextNode()) {
        if (element.children.length > 0) continue;
        const own = (element.innerText || '').trim();
        if (!own || !COUNT.test(own) || !DIGIT.test(own)) continue;
        let label = null;
        for (let above = element.parentElement, level = 0;
             above && above !== root && level < 3;
             above = above.parentElement, level += 1) {
            const rest = besides(above, element);
            if (rest.length === 0) continue;
            if (rest.length > 3) break;
            label = rest[0];
            break;
        }
        if (!label) continue;
        const key = `${label}\u0000${own}`;
        if (seen.has(key)) continue;
        seen.add(key);
        pairs.push({label, value: own});
        if (pairs.length >= 60) break;
    }
    return pairs;
}"""

_POST_ANALYTICS_LINKS_JS = r"""() => Array.from(
    document.querySelectorAll('a[href*="/analytics/post-summary/"]')
).map(anchor => anchor.getAttribute('href') || '')"""


class AnalyticsScraper:
    """Read the signed-in member's own analytics pages."""

    def __init__(
        self,
        session: ScrapingSession,
        navigator: PageNavigator,
        content: PageContentReader,
    ):
        self._session = session
        self._navigator = navigator
        self._content = content

    def _landed_path(self) -> str:
        try:
            parsed = urlparse(self._session.page.url)
        except Exception:
            return ""
        host = (parsed.hostname or "").lower()
        if not (host == "linkedin.com" or host.endswith(".linkedin.com")):
            return ""
        return unquote(parsed.path)

    async def _open(self, url: str) -> None:
        page = self._session.page
        await self._navigator._navigate_to_page(url)
        await self._session.check_rate_limit()
        try:
            await page.wait_for_selector("main")
        except PlaywrightTimeoutError:
            logger.debug("No <main> element found on %s", url)
        await self._session.dismiss_modal()

    async def _read_section(
        self,
        url: str,
        section: str,
        landed: Callable[[str], bool],
        not_landed: dict[str, Any],
        sections: dict[str, str],
        metrics: dict[str, list[dict[str, Any]]],
        named: dict[str, dict[str, int]],
        errors: dict[str, dict[str, Any]],
    ) -> None:
        """One navigation, a landing check, then text and structural numbers."""
        page = self._session.page

        try:
            await self._open(url)
            if not landed(self._landed_path()):
                errors[section] = {
                    **not_landed,
                    "landed_path": self._landed_path() or None,
                }
                return
            try:
                await page.wait_for_function(
                    """() => {
                        const main = document.querySelector('main');
                        return !!main && main.innerText.length > 150;
                    }""",
                    timeout=_CONTENT_WAIT_MS,
                )
            except PlaywrightTimeoutError:
                logger.debug("Analytics content did not appear on %s", url)
            await self._session.scroll_body(pause_time=0.5, max_scrolls=3)
            if not landed(self._landed_path()):
                errors[section] = {
                    **not_landed,
                    "landed_path": self._landed_path() or None,
                }
                return
            raw = (await self._content._extract_root_content(["main"]))["text"]
            truncated = truncate_linkedin_noise(raw) if raw else ""
            if raw and not truncated and raw.strip():
                errors[section] = rate_limited_section_error()
                return
            if not truncated:
                return
            sections[section] = filter_linkedin_noise_lines(truncated)
            section_metrics, section_named = build_metrics(
                await page.evaluate(_METRIC_PAIRS_JS)
            )
            if section_metrics:
                metrics[section] = section_metrics
            if section_named:
                named[section] = section_named
        except LinkedInScraperException:
            raise
        except Exception as error:
            logger.warning("Failed to read analytics %s: %s", url, error)
            errors[section] = build_issue_diagnostics(
                error, context="analytics", target_url=url, section_name=section
            )

    @staticmethod
    def _result(
        url: str,
        sections: dict[str, str],
        metrics: dict[str, list[dict[str, Any]]],
        named: dict[str, dict[str, int]],
        errors: dict[str, dict[str, Any]],
        unknown: list[str] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"url": url, "sections": sections}
        if metrics:
            result["metrics"] = metrics
        if named:
            result["named_metrics"] = named
        if errors:
            result["section_errors"] = errors
        if unknown:
            result["unknown_sections"] = unknown
        return result

    async def get_post_analytics(self, post_url: str) -> dict[str, Any]:
        """Read one of the signed-in member's own posts' analytics page."""
        post_urn = normalize_post_urn(post_url)
        sections: dict[str, str] = {}
        metrics: dict[str, list[dict[str, Any]]] = {}
        named: dict[str, dict[str, int]] = {}
        errors: dict[str, dict[str, Any]] = {}
        not_authorized = {
            "error_type": "not_authorized",
            "error_message": "LinkedIn did not open this post's analytics page. Post "
            "analytics exist only for the signed-in member's own posts; nothing was "
            "inferred from the page it opened instead.",
        }

        activity_urn = post_urn if post_urn.startswith("urn:li:activity:") else None
        if activity_urn is None:
            # Only the post's own "View analytics" link names its activity URN.
            try:
                await self._open(post_update_url(post_urn))
                hrefs = await self._session.page.evaluate(_POST_ANALYTICS_LINKS_JS)
            except LinkedInScraperException:
                raise
            except Exception as error:
                errors[POST_ANALYTICS_SECTION] = build_issue_diagnostics(
                    error, context="get_post_analytics", target_url=post_url
                )
                return self._result(
                    post_update_url(post_urn), sections, metrics, named, errors
                )
            found = {
                match.group(1)
                for href in (hrefs if isinstance(hrefs, list) else [])
                if isinstance(href, str)
                and (match := _ACTIVITY_IN_HREF_RE.search(unquote(href)))
            }
            if len(found) != 1:
                errors[POST_ANALYTICS_SECTION] = {
                    **not_authorized,
                    "error_message": "The post page does not link to exactly one "
                    "analytics page. Post analytics exist only for the signed-in "
                    "member's own posts.",
                }
                return self._result(
                    post_update_url(post_urn), sections, metrics, named, errors
                )
            activity_urn = found.pop()
            await self._session.delay(NAV_DELAY)

        url = post_analytics_url(activity_urn)
        await self._read_section(
            url,
            POST_ANALYTICS_SECTION,
            _starts_with(f"{_POST_SUMMARY_PREFIX}{activity_urn}"),
            not_authorized,
            sections,
            metrics,
            named,
            errors,
        )
        return self._result(url, sections, metrics, named, errors)

    async def get_profile_analytics(
        self, sections: str | None = None
    ) -> dict[str, Any]:
        """Read the member's profile and creator dashboards, one page per section."""
        requested, unknown = parse_profile_analytics_sections(sections)
        url = f"{_BASE}/analytics/"
        texts: dict[str, str] = {}
        metrics: dict[str, list[dict[str, Any]]] = {}
        named: dict[str, dict[str, int]] = {}
        errors: dict[str, dict[str, Any]] = {}
        if not requested:
            if unknown:
                raise InvalidReferenceError(
                    f"Unknown analytics sections: {', '.join(unknown)}. Valid: "
                    f"{', '.join(PROFILE_ANALYTICS_SECTIONS)}."
                )
            return self._result(url, texts, metrics, named, errors)
        for index, name in enumerate(requested):
            if index:
                await self._session.delay(NAV_DELAY)
            route, landing = PROFILE_ANALYTICS_SECTIONS[name]
            await self._read_section(
                f"{_BASE}{route}",
                name,
                _starts_with(*landing),
                {
                    "error_type": "redirected",
                    "error_message": "LinkedIn redirected away from this dashboard "
                    "(for example to a Premium or creator-mode prompt), so it is not "
                    "available to this account. Nothing was read from the page it "
                    "opened instead.",
                },
                texts,
                metrics,
                named,
                errors,
            )
        return self._result(url, texts, metrics, named, errors, unknown)

    async def get_company_page_analytics(
        self, company: str, sections: str | None = None
    ) -> dict[str, Any]:
        """Read a company page's admin analytics, one navigation per section.

        The page may be named by slug or numeric id. LinkedIn may answer a slug
        from the numeric-id route, so a section counts as landed on this page
        when its path names the same slug or a numeric id, and every section
        after the first must name the id the first one landed on.
        """
        identifier = normalize_company_identifier(company)
        requested, unknown = parse_company_analytics_sections(sections)
        if not requested:
            raise InvalidReferenceError(
                f"Unknown company analytics sections: {', '.join(unknown)}. Valid: "
                f"{', '.join(COMPANY_ANALYTICS_SECTIONS)}."
            )
        url = company_page_url(identifier, "/admin/analytics/")
        texts: dict[str, str] = {}
        metrics: dict[str, list[dict[str, Any]]] = {}
        named: dict[str, dict[str, int]] = {}
        errors: dict[str, dict[str, Any]] = {}
        pinned: dict[str, str] = {}
        not_authorized = {
            "error_type": "not_authorized",
            "error_message": "LinkedIn did not open this page's admin analytics. They "
            "exist only for pages the signed-in member administers; nothing was read "
            "from the page it opened instead.",
        }
        for index, name in enumerate(requested):
            if index:
                await self._session.delay(NAV_DELAY)
            route = COMPANY_ANALYTICS_SECTIONS[name]

            def landed(path: str, route: str = route) -> bool:
                match = _COMPANY_ADMIN_PATH_RE.match(path)
                if match is None or match.group(2) != route:
                    return False
                segment = match.group(1)
                if "id" in pinned:
                    return segment == pinned["id"]
                return segment.lower() == identifier.lower() or segment.isdigit()

            await self._read_section(
                company_page_url(identifier, f"/admin/{route}"),
                name,
                landed,
                not_authorized,
                texts,
                metrics,
                named,
                errors,
            )
            if name not in errors and "id" not in pinned:
                match = _COMPANY_ADMIN_PATH_RE.match(self._landed_path())
                if match is not None and match.group(1).isdigit():
                    pinned["id"] = match.group(1)
        result = self._result(url, texts, metrics, named, errors, unknown)
        if "id" in pinned:
            result["company_id"] = pinned["id"]
        return result
