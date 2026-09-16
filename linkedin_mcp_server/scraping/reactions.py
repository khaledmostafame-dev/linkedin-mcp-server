"""Reactor list for a LinkedIn post, read from its reactions dialog.

The reactions dialog is opened by clicking the post's social-counts
control (the clickable summary above the Like/Comment/Repost/Send action
row) rather than by navigating to a distinct URL, so this module drives
that click and the scroll-then-read inside the resulting dialog. Every
step is fail-closed: if the control or the dialog cannot be identified
unambiguously through structure alone, nothing further is clicked and a
``section_errors`` entry is returned instead of guessing (per the
AGENTS.md Scraping Rules — no text/label values are read anywhere here).

The DOM assumptions below (see ``_FIND_SOCIAL_COUNTS_CONTROL_JS``) are
**not** verified against a live account — this fork never signs in to
LinkedIn (AGENTS.md Hard safety rules) — and must be confirmed live
before this tool is trusted. Reaction *type* (Like/Celebrate/Support/…)
is deliberately not extracted: no structural, non-text signal for it was
identified, and omitting rather than guessing is what was asked for.
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
# The action row (Like/Comment/Repost/Send) is found first, by its own
# structural fingerprint (a container with >= 3 directly-nested
# aria-labelled buttons — attribute *presence*, never the label value).
# The social-counts summary that opens the reactions dialog renders as a
# sibling immediately above that row, and — on every LinkedIn post layout
# this heuristic was designed against on paper — holds exactly one
# clickable element. Both counts are exact-match guards specifically so a
# page that does not match this shape finds nothing and clicks nothing,
# rather than clicking the nearest plausible button.
#
# UNVERIFIED against a live post (no LinkedIn login is available while
# building this) — see the module docstring and this tool's
# ``live_verification_needed`` entry.
_FIND_SOCIAL_COUNTS_CONTROL_JS = r"""
(() => {
  function findActionRow(main) {
    const candidates = main.querySelectorAll('section, article, div, ul, li');
    for (const el of candidates) {
      const labeled = el.querySelectorAll(':scope > button[aria-label]');
      if (labeled.length >= 3) return el;
    }
    return null;
  }
  const main = document.querySelector('main');
  if (!main) return false;
  const actionRow = findActionRow(main);
  if (!actionRow) return false;
  let sibling = actionRow.previousElementSibling;
  for (let hop = 0; sibling && hop < 3; hop += 1) {
    const clickable = sibling.querySelectorAll('button, a[href]');
    if (clickable.length === 1) {
      clickable[0].click();
      return true;
    }
    sibling = sibling.previousElementSibling;
  }
  return false;
})
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
