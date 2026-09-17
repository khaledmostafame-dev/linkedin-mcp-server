"""Save/unsave a LinkedIn post through its overflow menu's save item.

LinkedIn's "Save"/"Unsave" action lives inside a post's "..." control
menu rather than in its top-level action row, so this is a two-step
structural probe: find the post's single menu opener, then — once the
panel it controls is open — the single save item inside it. Every step
is fail-closed, matching ``reactions.py``: an ambiguous match (zero or
more than one candidate) stops the whole action before anything is
clicked, and ``confirm=False`` always stops it before the item itself is
clicked (see ``save_post``).

The opener probe matches the structure of a signed-in, English-UI post
page captured on 2026-09-17. The open menu was not part of that capture:
its item shape (``[role="button"]`` items carrying ``data-test-icon``
hooks) is the one ``post_composer.py`` ports from upstream measurements
of the same control menu, and the bookmark icon names follow the
``<glyph>-outline|fill-<size>`` scheme LinkedIn uses elsewhere on the
page. Both remain to be confirmed live with the menu open.
"""

from __future__ import annotations

from typing import Any

import logging

from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.scraping.identifiers import normalize_post_url
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession

logger = logging.getLogger(__name__)

# The post's own control menu, shared with ``post_composer.py`` (whose
# delete_post/edit_post open the same menu). Prepended to every program
# that needs it, so each one re-derives the elements from the live DOM
# instead of holding a handle across evaluate calls. ``postMenu(main)``
# takes the page's <main>, found by the caller.
#
# The opener (measured 2026-09-17). A post page holds dozens of
# ``button[aria-expanded]`` (46 on the captured one): the post's menu
# opener, the Like reaction picker, Repost, the comment editor's emoji
# picker, the comment sort control and one overflow opener per comment.
# The post's own opener is told apart by position and shape only:
#   * it precedes the post's first social-action control in document
#     order. Those controls carry LinkedIn's ``data-finite-scroll-hotkey``
#     hook, the post's own row renders first, and everything belonging to
#     comments (sort control, per-comment menus) renders after it. That
#     first control must itself sit outside every comment: a post whose
#     own action row is missing would otherwise hand the anchor to the
#     first comment's row and let the comment sort control qualify;
#   * its next element sibling is the dropdown panel it controls, which
#     carries ``aria-hidden`` while collapsed. That excludes the reactor
#     facepile's "see more" control, the one other aria-expanded button
#     above the action row, which has no sibling at all;
#   * its panel holds no comment (an element whose ``data-id``/``data-urn``/
#     ``data-entity-urn`` names a comment URN, the hook ``comments.py``
#     reads), collapsed or open.
# Exactly one such button, or nothing.
#
# The menu counts as open only when the opener reports
# aria-expanded="true", its panel is no longer aria-hidden="true", and the
# panel holds at least one actionable item and still no comment.
POST_MENU_JS = r"""
  const MENU_ITEM = '[role="button"], [role="menuitem"], button, a[href]';
  const COMMENT_UNIT = ['data-id', 'data-urn', 'data-entity-urn']
    .map(name => `[${name}*="comment:("]`).join(', ');
  function postMenu(main) {
    if (!main) return null;
    const firstAction = main.querySelector('[data-finite-scroll-hotkey]');
    if (!firstAction || firstAction.closest(COMMENT_UNIT)) return null;
    const openers = Array.from(
      main.querySelectorAll('button[aria-expanded]')
    ).filter(button => {
      const precedes = button.compareDocumentPosition(firstAction)
        & Node.DOCUMENT_POSITION_FOLLOWING;
      const panel = button.nextElementSibling;
      return Boolean(precedes) && panel !== null && (
        panel.hasAttribute('aria-hidden')
        || button.getAttribute('aria-expanded') === 'true'
      );
    });
    if (openers.length !== 1) return null;
    const opener = openers[0];
    const panel = opener.nextElementSibling;
    if (panel.querySelector(COMMENT_UNIT)) return null;
    const open = opener.getAttribute('aria-expanded') === 'true'
      && panel.getAttribute('aria-hidden') !== 'true'
      && panel.querySelector(MENU_ITEM) !== null;
    return { opener, panel, open };
  }
"""

# The save item is the one item holding a ``bookmark-*`` icon hook: an
# outline glyph reads as "not saved", a fill glyph as "saved". Any other
# bookmark glyph, or more than one such item, is refused.
_SAVE_ITEM_JS = r"""
  const pagePostMenu = () => postMenu(document.querySelector('main'));
  function saveItem() {
    const menu = pagePostMenu();
    if (!menu || !menu.open) return null;
    const icons = menu.panel.querySelectorAll(
      'svg[data-test-icon^="bookmark-"], use[href^="#bookmark-"]'
    );
    const items = new Map();
    for (const icon of icons) {
      const item = icon.closest(MENU_ITEM);
      if (!item || !menu.panel.contains(item)) return null;
      const name = icon.getAttribute('data-test-icon')
        || (icon.getAttribute('href') || '').slice(1);
      const saved = /^bookmark-fill(-|$)/.test(name) ? true
        : /^bookmark-outline(-|$)/.test(name) ? false : null;
      if (!items.has(item)) items.set(item, new Set());
      items.get(item).add(saved);
    }
    if (items.size !== 1) return null;
    const [[item, states]] = Array.from(items.entries());
    if (states.size !== 1 || states.has(null)) return null;
    return { item, saved: states.has(true) };
  }
"""


def _post_menu_program(body: str) -> str:
    return "() => {" + POST_MENU_JS + _SAVE_ITEM_JS + body + "}"


# Clicks the post's menu opener; an already-open menu is left as it is.
_OPEN_POST_OVERFLOW_MENU_JS = _post_menu_program(
    r"""
  const menu = pagePostMenu();
  if (!menu) return false;
  if (!menu.open) menu.opener.click();
  return true;
"""
)

_POST_MENU_OPEN_JS = _post_menu_program(
    r"""
  const menu = pagePostMenu();
  return menu !== null && menu.open;
"""
)

# The save item's current state, or null when there is not exactly one
# item with a recognised bookmark glyph.
_READ_SAVE_TOGGLE_STATE_JS = _post_menu_program(
    r"""
  const found = saveItem();
  return found === null ? null : found.saved;
"""
)

_CLICK_SAVE_TOGGLE_JS = _post_menu_program(
    r"""
  const found = saveItem();
  if (found === null) return false;
  found.item.click();
  return true;
"""
)


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

    async def _reread_after_click(self, page: Any, url: str) -> bool | None:
        """The save item's state after its click, or None when unknown.

        Clicking a dropdown item normally closes the dropdown, and the
        state can only be read while it is open, so a closed menu is
        reopened first. Opening the menu changes nothing on LinkedIn.
        """
        try:
            state = await page.evaluate(_READ_SAVE_TOGGLE_STATE_JS)
            if state is not None:
                return state
            if not await page.evaluate(_OPEN_POST_OVERFLOW_MENU_JS):
                return None
            await page.wait_for_function(_POST_MENU_OPEN_JS, timeout=5000)
            return await page.evaluate(_READ_SAVE_TOGGLE_STATE_JS)
        except Exception as e:
            logger.warning(
                "Failed to re-read the save state after clicking on %s: %s", url, e
            )
            return None

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
                "with a collapsed panel beside it, above the post's action "
                "row). Nothing was clicked.",
            )

        try:
            await page.wait_for_function(_POST_MENU_OPEN_JS, timeout=5000)
        except PlaywrightTimeoutError:
            return _save_post_result(
                url,
                "structural_signal_not_found",
                "The overflow menu did not open (its opener never reported "
                'aria-expanded="true" with a populated panel). Nothing '
                "further was clicked.",
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
                "Could not identify a single save item (one bookmark "
                "outline/fill icon hook) inside the overflow menu. Nothing "
                "was clicked.",
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
        new_state = await self._reread_after_click(page, url)

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
