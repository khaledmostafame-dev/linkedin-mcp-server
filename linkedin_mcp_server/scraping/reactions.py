"""Reactor list for a LinkedIn post, read from its reactions dialog.

The reactions dialog is opened by clicking the post's social-counts
control (the clickable summary above the Like/Comment/Repost/Send action
row) rather than by navigating to a distinct URL: a signed-in post page
carries no link to a reactor list, only that button. So this module
drives that click and the scroll-then-read inside the resulting dialog.
Every step is fail-closed: if the control or the dialog cannot be
identified unambiguously through structure alone, nothing further is
clicked and a ``section_errors`` entry is returned instead of guessing
(per the AGENTS.md Scraping Rules — no text/label values are read
anywhere here).

The control probe (``_FIND_SOCIAL_COUNTS_CONTROL_JS``) matches the
structure of a signed-in, English-UI post page captured on 2026-09-17.
The dialog it opens was not part of that capture, so the dialog wait and
the scroll-then-read are still unconfirmed live. Reaction *type*
(Like/Celebrate/Support/…) is deliberately not extracted: no structural,
non-text signal for it was identified, and omitting rather than guessing
is what was asked for.
"""

from __future__ import annotations

from typing import Any

import logging

from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.contracts import rate_limited_section_error
from linkedin_mcp_server.scraping.identifiers import normalize_post_url
from linkedin_mcp_server.scraping.link_metadata import build_references
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession
from linkedin_mcp_server.scraping.text import (
    filter_linkedin_noise_lines,
    truncate_linkedin_noise,
)

logger = logging.getLogger(__name__)

_DIALOG_SELECTOR = '[role="dialog"]'

# Structural probe for the post's social-counts control.
#
# Measured on the live post page (2026-09-17): the reactions summary is a
# ``button`` carrying LinkedIn's own ``data-reaction-details`` hook — an
# attribute name, not text — and it is the only element in <main> that
# carries it. Comments on the same page show their own reaction counts
# without that hook. The action row does not help here: its controls sit
# one per wrapper (span/div), so no container holds the Like/Comment/
# Repost/Send buttons as direct children, which is why the earlier
# "container with >= 3 labelled buttons" probe found nothing.
#
# Exact-count guard: clicks only when there is exactly one hook. A dialog
# already on the page refuses too, because every later step reads the
# first ``[role="dialog"]`` and could not tell it from the one this click
# opens (the captured post page has none before interaction).
_FIND_SOCIAL_COUNTS_CONTROL_JS = r"""
() => {
  const main = document.querySelector('main');
  if (!main) return false;
  if (document.querySelector('[role="dialog"]')) return false;
  const hooks = main.querySelectorAll('[data-reaction-details]');
  if (hooks.length !== 1) return false;
  const control = hooks[0].closest('button, [role="button"]');
  if (!control || !main.contains(control)) return false;
  control.click();
  return true;
}
"""

# Bounded single-step scroll of whichever element inside the dialog is
# actually the scrollable list (not the dialog chrome itself, which
# typically does not scroll). Returns whether the scroll moved anything,
# so the caller can stop as soon as a step is a no-op.
_SCROLL_DIALOG_JS = r"""
() => {
  const dialog = document.querySelector('[role="dialog"]');
  if (!dialog) return false;
  const scrollable = Array.from(dialog.querySelectorAll('*')).find(
    el => el.scrollHeight > el.clientHeight + 4
  ) || dialog;
  const before = scrollable.scrollTop;
  scrollable.scrollTop = scrollable.scrollHeight;
  return scrollable.scrollTop > before;
}
"""


def _structural_signal_error(context: str) -> dict[str, Any]:
    return {
        "error_type": "structural_signal_not_found",
        "error_message": (
            f"Could not identify the {context} structurally; LinkedIn's "
            "markup may not match the assumed shape. Nothing was clicked."
        ),
    }


class ReactionsReader:
    """Read the reactor list from one post's reactions dialog."""

    def __init__(
        self,
        session: ScrapingSession,
        navigator: PageNavigator,
        content: PageContentReader,
    ):
        self._session = session
        self._navigator = navigator
        self._content = content

    async def get_post_reactions(
        self, post_url: str, max_reactors: int = 50
    ) -> dict[str, Any]:
        """Open a post's reactions dialog and read the reactor list.

        Returns:
            ``{url, sections: {reactions: text}, references: {reactions:
            [{kind: "person", ...}, ...]}}`` on success, or ``{url,
            sections: {}, section_errors: {reactions: {...}}}`` when the
            social-counts control or the dialog it should open cannot be
            identified unambiguously.
        """
        url = normalize_post_url(post_url)
        page = self._session.page
        await self._navigator._navigate_to_page(url)
        await self._session.check_rate_limit()

        try:
            clicked = await page.evaluate(_FIND_SOCIAL_COUNTS_CONTROL_JS)
        except Exception as e:
            logger.warning(
                "Failed to probe the social-counts control on %s: %s", url, e
            )
            clicked = False

        if not clicked:
            return {
                "url": url,
                "sections": {},
                "section_errors": {
                    "reactions": _structural_signal_error("social-counts control")
                },
            }

        try:
            await page.wait_for_selector(_DIALOG_SELECTOR, timeout=5000)
        except PlaywrightTimeoutError:
            return {
                "url": url,
                "sections": {},
                "section_errors": {
                    "reactions": _structural_signal_error("reactions dialog")
                },
            }

        # ~10 reactors load per scroll step, capped well under the tool's
        # own ceiling so a runaway list cannot spend the whole timeout.
        max_scrolls = min(max(1, -(-max_reactors // 10)), 15)
        for _ in range(max_scrolls):
            try:
                moved = await page.evaluate(_SCROLL_DIALOG_JS)
            except Exception as e:
                logger.debug("Reactions dialog scroll step failed: %s", e)
                break
            if not moved:
                break
            await self._session.delay(0.5)

        raw_result = await self._content._extract_root_content([_DIALOG_SELECTOR])
        raw = raw_result["text"]
        if not raw:
            return {"url": url, "sections": {}}
        truncated = truncate_linkedin_noise(raw)
        if not truncated and raw.strip():
            logger.warning(
                "Reactions dialog on %s returned only chrome (likely rate-limited)",
                url,
            )
            return {
                "url": url,
                "sections": {},
                "section_errors": {"reactions": rate_limited_section_error()},
            }
        cleaned = filter_linkedin_noise_lines(truncated)

        references = build_references(raw_result["references"], "reactions")
        result: dict[str, Any] = {"url": url, "sections": {"reactions": cleaned}}
        if references:
            result["references"] = {"reactions": references}
        return result
