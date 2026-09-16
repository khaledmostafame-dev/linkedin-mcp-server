"""LinkedIn network-graph read workflows, plus the follow/unfollow write.

Read-only listings that surface a member's network graph: shared
connections with another profile, the authenticated user's own connection
list, and the invitation-manager queues. Every listing here is a
navigate-scroll-innerText capture through ``SectionCapture`` -- the same
pattern ``CompanyScraper.get_company_employees`` and
``ConversationReader.get_inbox`` use -- so no LinkedIn class names and no
text-based state detection are involved.

Writes that act on *one member's pending invitation* (accept, ignore,
withdraw) live in ``connection_actions.ConnectionActions`` instead of here,
because they reuse its already live-verified per-profile action-row
signals rather than the unverified invitation-manager list layout. ``follow``
lives here because it targets a page (person or company), not an invitation.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal
from urllib.parse import urljoin

from linkedin_mcp_server.scraping.capture import (
    CaptureMode,
    CapturePlan,
    SectionCapture,
)
from linkedin_mcp_server.scraping.contracts import (
    RATE_LIMITED_SECTION_TEXT,
    rate_limited_section_error,
)
from linkedin_mcp_server.scraping.identifiers import (
    company_page_url,
    normalize_company_identifier,
    normalize_person_identifier,
    person_profile_url,
)
from linkedin_mcp_server.scraping.link_metadata import Reference
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession

logger = logging.getLogger(__name__)

InvitationDirection = Literal["received", "sent"]
ConnectionSort = Literal["recently_added", "first_name", "last_name"]

# LinkedIn's own query tokens for /mynetwork/invite-connect/connections/.
_SORT_TYPE_MAP: dict[str, str] = {
    "recently_added": "RECENTLY_ADDED",
    "first_name": "FIRST_NAME",
    "last_name": "LAST_NAME",
}

_INVITATION_URLS: dict[InvitationDirection, str] = {
    "received": "https://www.linkedin.com/mynetwork/invitation-manager/",
    "sent": "https://www.linkedin.com/mynetwork/invitation-manager/sent/",
}

# One scroll loads roughly this many additional rows on LinkedIn's
# infinite-scroll network listings. Not exact -- LinkedIn's own page size
# varies -- but bounding the scroll budget by it keeps a caller's
# max_results meaningful without adding a second, independent
# scroll-tracking mechanism to SectionCapture.
_ROWS_PER_SCROLL = 10
_MIN_SCROLLS = 1
_MAX_SCROLLS_CAP = 25

# Read structurally (URL pattern, not label text) from the profile top card:
# LinkedIn's "N mutual connections" control is a canned people-search anchor
# carrying the facetConnectionOf URN facet. The member URN it requires is
# not derivable from a public identifier, so the anchor has to be followed
# rather than reconstructed.
_MUTUAL_CONNECTIONS_LINK_JS = r"""
(() => {
  const main = document.querySelector('main');
  if (!main) return null;
  const scope = main.querySelector('section') || main;
  const anchor = scope.querySelector('a[href*="facetConnectionOf"]');
  return anchor ? anchor.getAttribute('href') : null;
})
"""

# The single unambiguous Follow/Following toggle in a top card: a labeled
# button that is not the aria-expanded More-menu opener (same
# presence-based distinction connection.py uses). Exactly one match is
# required; zero or several is refused rather than guessed, matching the
# "ambiguity is treated as no match" discipline connection_actions.py
# already applies to the irreversible incoming-accept click.
_FOLLOW_CANDIDATE_JS = r"""
(() => {
  const main = document.querySelector('main');
  if (!main) return null;
  const scope = main.querySelector('section') || main;
  const buttons = Array.from(scope.querySelectorAll('button[aria-label]'))
    .filter((b) => !b.hasAttribute('aria-expanded'));
  if (buttons.length !== 1) {
    return { ok: false, count: buttons.length };
  }
  const b = buttons[0];
  return {
    ok: true,
    count: 1,
    ariaPressed: b.getAttribute('aria-pressed'),
    ariaExpanded: b.getAttribute('aria-expanded'),
    ariaHasPopup: b.getAttribute('aria-haspopup'),
  };
})
"""

_FOLLOW_CLICK_JS = r"""
(() => {
  const main = document.querySelector('main');
  if (!main) return false;
  const scope = main.querySelector('section') || main;
  const buttons = Array.from(scope.querySelectorAll('button[aria-label]'))
    .filter((b) => !b.hasAttribute('aria-expanded'));
  if (buttons.length !== 1) return false;
  buttons[0].click();
  return true;
})
"""


def _scrolls_for(max_results: int) -> int:
    """Bound the scroll budget for a target row count."""
    return max(
        _MIN_SCROLLS,
        min(_MAX_SCROLLS_CAP, math.ceil(max(1, max_results) / _ROWS_PER_SCROLL)),
    )


def _listing_result(
    url: str,
    section_name: str,
    extracted_text: str,
    references: list[Reference],
    error: dict[str, Any] | None,
    *,
    max_results: int,
) -> dict[str, Any]:
    """Build the shared {url, sections, references?, section_errors?, stopped_reason}."""
    sections: dict[str, str] = {}
    references_out: dict[str, list[Reference]] = {}
    section_errors: dict[str, dict[str, Any]] = {}

    if extracted_text and extracted_text != RATE_LIMITED_SECTION_TEXT:
        sections[section_name] = extracted_text
        if references:
            references_out[section_name] = references
    elif extracted_text == RATE_LIMITED_SECTION_TEXT:
        section_errors[section_name] = rate_limited_section_error()
    elif error:
        section_errors[section_name] = error

    result: dict[str, Any] = {"url": url, "sections": sections}
    if references_out:
        result["references"] = references_out
    if section_errors:
        result["section_errors"] = section_errors
    # A heuristic, not a promise: SectionCapture does not report whether the
    # scroll loop stopped because the page ran out of content or because the
    # budget ran out first. Reaching (or exceeding) max_results is treated as
    # "there may be more"; anything short of it is treated as the end.
    result["stopped_reason"] = (
        "max_results_reached" if len(references) >= max_results else "end_of_results"
    )
    return result


def resolve_follow_target(target_url: str) -> str:
    """Resolve a person or company reference to its canonical page URL."""
    if "/company/" in target_url:
        return company_page_url(normalize_company_identifier(target_url), "/")
    if "/in/" in target_url:
        return person_profile_url(normalize_person_identifier(target_url), "/")
    # A bare slug with no path context: assume a person, the more common
    # case for search_people-driven prospecting workflows.
    return person_profile_url(normalize_person_identifier(target_url), "/")


def follow_preview(target_url: str, *, unfollow: bool) -> dict[str, Any]:
    """The ``confirm=False`` answer for follow/unfollow, browser-free."""
    resolved = resolve_follow_target(target_url)
    requested = "unfollow" if unfollow else "follow"
    return {
        "url": resolved,
        "status": "preview",
        "requested": requested,
        "message": f"Would {requested} {resolved}. Pass confirm=True to proceed.",
    }


class NetworkScraper:
    """Own network-graph read workflows: connections, invitations, follow."""

    def __init__(
        self,
        session: ScrapingSession,
        navigator: PageNavigator,
        capture: SectionCapture,
    ):
        self._session = session
        self._navigator = navigator
        self._capture = capture

    async def get_mutual_connections(
        self, username: str, max_results: int = 50
    ) -> dict[str, Any]:
        """List connections shared between the authenticated user and a profile."""
        username = normalize_person_identifier(username)
        profile_url = person_profile_url(username, "/")
        section_name = "mutual_connections"

        await self._navigator._navigate_to_page(profile_url)
        await self._session.check_rate_limit()
        try:
            await self._session.page.wait_for_selector("main", timeout=10000)
        except Exception:
            logger.debug("No <main> element on %s", profile_url)
        await self._session.dismiss_modal()

        try:
            href = await self._session.page.evaluate(_MUTUAL_CONNECTIONS_LINK_JS)
        except Exception:
            logger.debug(
                "Could not read mutual-connections anchor on %s",
                profile_url,
                exc_info=True,
            )
            href = None

        if not href:
            return {
                "url": profile_url,
                "sections": {},
                "stopped_reason": "no_mutual_connections_link_found",
            }

        search_url = urljoin(profile_url, href)
        plan = CapturePlan(CaptureMode.SEARCH_RESULTS, _scrolls_for(max_results))
        extracted = await self._capture.capture(search_url, section_name, plan)
        return _listing_result(
            search_url,
            section_name,
            extracted.text,
            extracted.references,
            extracted.error,
            max_results=max_results,
        )

    async def list_connections(
        self,
        max_results: int = 100,
        sort: ConnectionSort = "recently_added",
    ) -> dict[str, Any]:
        """List the authenticated user's 1st-degree connections."""
        sort_token = _SORT_TYPE_MAP.get(sort, _SORT_TYPE_MAP["recently_added"])
        url = (
            "https://www.linkedin.com/mynetwork/invite-connect/connections/"
            f"?sortType={sort_token}"
        )
        section_name = "connections"
        plan = CapturePlan(CaptureMode.COMPANY_PEOPLE, _scrolls_for(max_results))
        extracted = await self._capture.capture(url, section_name, plan)
        return _listing_result(
            url,
            section_name,
            extracted.text,
            extracted.references,
            extracted.error,
            max_results=max_results,
        )

    async def get_invitations(
        self,
        direction: InvitationDirection = "received",
        max_results: int = 50,
    ) -> dict[str, Any]:
        """List outgoing or incoming connection-request invitations."""
        url = _INVITATION_URLS.get(direction, _INVITATION_URLS["received"])
        section_name = f"{direction}_invitations"
        plan = CapturePlan(CaptureMode.COMPANY_PEOPLE, _scrolls_for(max_results))
        extracted = await self._capture.capture(url, section_name, plan)
        return _listing_result(
            url,
            section_name,
            extracted.text,
            extracted.references,
            extracted.error,
            max_results=max_results,
        )

    def _resolve_follow_target(self, target_url: str) -> str:
        """Resolve a person or company reference to its canonical page URL."""
        return resolve_follow_target(target_url)

    async def follow(
        self,
        target_url: str,
        *,
        confirm: bool,
        unfollow: bool = False,
    ) -> dict[str, Any]:
        """Follow or unfollow a person or company page.

        LinkedIn's Follow/Following control is a stateful toggle with no
        locale-independent text this server is allowed to read (per
        AGENTS.md, button labels are never inspected). Detection is
        therefore structural and deliberately conservative: the tool acts
        only when exactly one unambiguous, non-menu labeled button exists in
        the top card and that button reports its state through
        ``aria-pressed``. If the page is already in the requested state
        nothing is clicked, so a follow request can never undo an existing
        follow; success is reported only when ``aria-pressed`` afterwards
        shows the requested state. confirm=False returns a preview with no
        browser interaction and no state change.
        """
        if not confirm:
            return follow_preview(target_url, unfollow=unfollow)
        resolved = self._resolve_follow_target(target_url)
        requested = "unfollow" if unfollow else "follow"

        await self._navigator._navigate_to_page(resolved)
        await self._session.check_rate_limit()
        try:
            await self._session.page.wait_for_selector("main", timeout=10000)
        except Exception:
            logger.debug("No <main> element on %s", resolved)
        await self._session.dismiss_modal()

        try:
            before = await self._session.page.evaluate(_FOLLOW_CANDIDATE_JS)
        except Exception:
            logger.debug("Follow candidate probe failed on %s", resolved, exc_info=True)
            before = None

        if not before or not before.get("ok"):
            count = before.get("count", 0) if before else 0
            return {
                "url": resolved,
                "status": "action_unavailable",
                "requested": requested,
                "message": (
                    "Could not find a single unambiguous Follow/Following "
                    f"control in the top card (found {count} candidate "
                    "button(s)). Refusing to guess which one to click."
                ),
            }

        # The toggle flips whatever state the page is in, so clicking without
        # knowing that state could undo an existing follow when a follow was
        # asked for. ``aria-pressed`` is the only structural, locale-independent
        # signal of the current state; without it nothing is clicked.
        pressed = before.get("ariaPressed")
        if pressed not in ("true", "false"):
            return {
                "url": resolved,
                "status": "action_unavailable",
                "requested": requested,
                "message": (
                    "The follow control exposes no aria-pressed state, so whether "
                    "this account already follows the page cannot be read "
                    "structurally. Refusing to click a toggle that could undo an "
                    "existing follow."
                ),
            }
        wanted = "false" if unfollow else "true"
        if pressed == wanted:
            return {
                "url": resolved,
                "status": "already_unfollowed" if unfollow else "already_following",
                "requested": requested,
                "message": (
                    f"Nothing clicked: the follow control already reports the "
                    f"{requested} state."
                ),
            }

        try:
            clicked = bool(await self._session.page.evaluate(_FOLLOW_CLICK_JS))
        except Exception:
            logger.debug("Follow click failed on %s", resolved, exc_info=True)
            clicked = False

        if not clicked:
            return {
                "url": resolved,
                "status": "action_unavailable",
                "requested": requested,
                "message": "The follow control disappeared before it could be clicked.",
            }

        await self._session.delay(1.5)
        try:
            after = await self._session.page.evaluate(_FOLLOW_CANDIDATE_JS)
        except Exception:
            logger.debug(
                "Follow candidate re-probe failed on %s", resolved, exc_info=True
            )
            after = None

        if after and after.get("ok") and after.get("ariaPressed") == wanted:
            return {
                "url": resolved,
                "status": "toggled",
                "requested": requested,
                "message": (
                    f"Clicked the follow control ({requested} requested) and its "
                    "aria-pressed state now reports the requested state."
                ),
            }
        return {
            "url": resolved,
            "status": "uncertain",
            "requested": requested,
            "message": (
                "Clicked the follow control but its aria-pressed state does not "
                "report the requested state yet. The click may not have "
                "registered, or LinkedIn may be asking for confirmation -- "
                "verify manually before relying on this outcome."
            ),
        }
