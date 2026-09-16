"""Save/unsave a LinkedIn post through its overflow menu's save toggle.

LinkedIn's "Save"/"Saved" action lives inside a post's "..." overflow
menu rather than in its top-level action row, so this is a two-step
structural probe: find the single menu-opener button, then — once the
resulting menu is open — the single toggle control inside it. Every step
is fail-closed, matching ``reactions.py``: an ambiguous match (zero or
more than one candidate) stops the whole action before anything is
clicked, and ``confirm=False`` always stops it before the toggle itself
is clicked (see ``save_post``).

The DOM assumptions below (see the three ``_*_JS`` constants) are **not**
verified against a live account — this fork never signs in to LinkedIn
(AGENTS.md Hard safety rules) — and must be confirmed live before this
tool is trusted; see this tool's ``live_verification_needed`` entry.
"""

from __future__ import annotations

from typing import Any

import logging

from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.scraping.identifiers import normalize_post_url
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession

logger = logging.getLogger(__name__)

_MENU_SELECTOR = '[role="menu"]'

# The post's overflow-menu opener, identified the same way
# connection_actions.OPEN_MORE_BUTTON_JS identifies a profile's More
# button: aria-expanded is present on a menu opener and on nothing else
# in a post's <main> (Like/Comment/Repost/Send carry aria-label, not
# aria-expanded). Exact-count guard: clicks only when there is exactly
# one candidate, never "the first one".
_OPEN_POST_OVERFLOW_MENU_JS = r"""
() => {
  const main = document.querySelector('main');
  if (!main) return false;
  const openers = main.querySelectorAll('button[aria-expanded]');
  if (openers.length !== 1) return false;
  openers[0].click();
  return true;
}
"""

# The save/unsave toggle inside the open overflow menu, identified by
# aria-pressed presence (a toggle button) rather than by its label text —
# the one attribute a "Save"/"Saved" menu item is expected to carry that
# "Report", "Copy link", etc. would not. Returns the toggle's current
# state, or null when the match is not exactly one element.
_READ_SAVE_TOGGLE_STATE_JS = r"""
() => {
  const menu = document.querySelector('[role="menu"]');
  if (!menu) return null;
  const toggles = menu.querySelectorAll('[aria-pressed]');
  if (toggles.length !== 1) return null;
  return toggles[0].getAttribute('aria-pressed') === 'true';
}
"""

_CLICK_SAVE_TOGGLE_JS = r"""
() => {
  const menu = document.querySelector('[role="menu"]');
  if (!menu) return false;
  const toggles = menu.querySelectorAll('[aria-pressed]');
  if (toggles.length !== 1) return false;
  toggles[0].click();
  return true;
}
"""


def _save_post_result(
    url: str,
    status: str,
    message: str,
    *,
    saved: bool | None = None,
    retry_safe: bool = True,
) -> dict[str, Any]:
    return {
        "url": url,
        "status": status,
        "message": message,
        "saved": saved,
        "retry_safe": retry_safe,
    }


class PostActions:
    """Save or unsave one LinkedIn post, with explicit confirmation gating."""

    def __init__(self, session: ScrapingSession, navigator: PageNavigator):
        self._session = session
        self._navigator = navigator

    async def save_post(
        self, post_url: str, *, confirm: bool, unsave: bool = False
    ) -> dict[str, Any]:
        """Save or unsave a post via its overflow menu's toggle.

        ``confirm=False`` never changes LinkedIn state — it reports the
        current save state and what would happen, and returns before the
        overflow menu's toggle is ever clicked (the menu is opened either
        way, since that is the only structural probe available for the
        current state; opening a menu is not itself a state change).

        Args:
            post_url: A /feed/update/<urn>/ or /posts/<slug> permalink.
            confirm: Must be True to actually click the toggle.
            unsave: False (default) saves the post; True unsaves it. If
                the post is already in the requested state, nothing is
                clicked either way.

        Returns:
            {url, status, message, saved, retry_safe}. ``saved`` is the
            toggle's last-observed state (None when it could not be
            read). ``retry_safe`` is False once a click has been
            dispatched, since ``state_unconfirmed`` means the outcome is
            unknown and a retry could toggle the post back.
        """
        url = normalize_post_url(post_url)
        page = self._session.page
        await self._navigator._navigate_to_page(url)
        await self._session.check_rate_limit()

        try:
            opened = await page.evaluate(_OPEN_POST_OVERFLOW_MENU_JS)
        except Exception as e:
            logger.warning("Failed to probe the overflow-menu opener on %s: %s", url, e)
            opened = False
        if not opened:
            return _save_post_result(
                url,
                "structural_signal_not_found",
                "Could not identify the post's overflow-menu opener "
                "structurally (expected exactly one button[aria-expanded] "
                "in <main>). Nothing was clicked.",
            )

        try:
            await page.wait_for_selector(_MENU_SELECTOR, timeout=5000)
        except PlaywrightTimeoutError:
            return _save_post_result(
                url,
                "structural_signal_not_found",
                'The overflow menu did not open (no [role="menu"] '
                "appeared). Nothing further was clicked.",
            )

        try:
            currently_saved = await page.evaluate(_READ_SAVE_TOGGLE_STATE_JS)
        except Exception as e:
            logger.warning("Failed to read the save-toggle state on %s: %s", url, e)
            currently_saved = None
        if currently_saved is None:
            return _save_post_result(
                url,
                "structural_signal_not_found",
                "Could not identify a single aria-pressed save toggle "
                "inside the overflow menu. Nothing was clicked.",
            )

        desired_saved = not unsave
        if currently_saved == desired_saved:
            return _save_post_result(
                url,
                "already_in_desired_state",
                f"Post is already {'saved' if desired_saved else 'not saved'}.",
                saved=currently_saved,
            )

        if not confirm:
            action = "unsave" if unsave else "save"
            return _save_post_result(
                url,
                "confirmation_required",
                f"Would {action} this post. Call again with confirm=True to proceed.",
                saved=currently_saved,
            )

        try:
            clicked = await page.evaluate(_CLICK_SAVE_TOGGLE_JS)
        except Exception as e:
            logger.warning("Save-toggle click failed on %s: %s", url, e)
            clicked = False
        if not clicked:
            return _save_post_result(
                url,
                "structural_signal_not_found",
                "The save toggle disappeared before it could be clicked. "
                "Nothing was changed.",
            )

        # In flight from here: a failure below cannot tell whether the
        # click already changed LinkedIn's state (mirrors send_message's
        # own retry_safe contract in scraping/contracts.py).
        await self._session.delay(0.5)
        try:
            new_state = await page.evaluate(_READ_SAVE_TOGGLE_STATE_JS)
        except Exception as e:
            logger.warning(
                "Failed to re-read the save-toggle state after clicking on %s: %s",
                url,
                e,
            )
            new_state = None

        if new_state == desired_saved:
            verb = "saved" if desired_saved else "unsaved"
            return _save_post_result(
                url,
                verb,
                f"Post {verb}.",
                saved=new_state,
                retry_safe=False,
            )
        return _save_post_result(
            url,
            "state_unconfirmed",
            "Clicked the save toggle, but its state afterward does not "
            "match what was expected; the outcome is unknown. Check "
            "get_saved_posts before retrying, since a retry may toggle "
            "it back.",
            saved=new_state,
            retry_safe=False,
        )
