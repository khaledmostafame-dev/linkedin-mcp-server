# tests/test_post_composer_icons_dom.py
"""Browser-DOM tests for the composer's icon-button selectors.

``_icon_button`` builds a CSS selector using Playwright's ``:has()``
pseudo-class, which only a real browser engine evaluates -- the unit suite
mocks ``page.evaluate``/locators, so a selector-matching regression is
invisible there. These cases run the real selector against synthetic HTML
that mirrors the structural shape measured live 2026-09-17
(``005-extractor-after-goto.json``, ``/sharing/compose``): the schedule
button's ``clock-medium`` icon is an inline ``<svg id="clock-medium">``, not
``data-test-icon`` and not a `<use href="#...">`, and its clickable ancestor
is an ``<a aria-haspopup="dialog">``, not a ``<button>``.

Skipped automatically when chromium is not installed; run locally after
``uv run patchright install chromium --no-shell``.
"""

from __future__ import annotations

from typing import Any

import pytest
from patchright.async_api import async_playwright

from linkedin_mcp_server.scraping.post_composer import (
    _SCHEDULE_BUTTON_SELECTOR,
    _VIEW_ALL_SCHEDULED_SELECTOR,
    _icon_button,
)

pytestmark = [
    pytest.mark.browser_dom,
    pytest.mark.xdist_group("browser_runtime"),
]

# The markup this project's *old* selector variants (measured upstream
# against an older LinkedIn surface) were written for: a `<button>` element
# carrying a `data-test-icon` attribute or wrapping a `<use href="#...">`.
# Live capture 2026-09-17 found neither anywhere on the composer page.
_OLD_STYLE_ONLY = """
<div id="root">
  <button aria-label="Old style, unrelated to schedule">
    <svg data-test-icon="clock-medium"></svg>
  </button>
</div>
"""

# The markup actually measured live: the schedule control is an anchor, and
# its icon is an inline svg identified by its own id.
_CURRENT_LIVE_MARKUP = """
<div id="root">
  <a href="/sharing/compose" aria-haspopup="dialog" aria-expanded="false">
    <span><svg id="clock-medium"></svg></span>
  </a>
  <a href="/mynetwork/discover-hub/">
    <span><svg id="arrow-right-small"></svg></span>
  </a>
</div>
"""


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


async def _set_html(page: Any, html: str) -> None:
    await page.set_content(f"<!DOCTYPE html><html><body>{html}</body></html>")


class TestScheduleButtonSelectorAgainstRealDom:
    async def test_old_style_data_test_icon_button_still_matches(self, dom_page):
        """The pre-existing hooks are kept, not replaced (older surfaces)."""
        await _set_html(dom_page, _OLD_STYLE_ONLY)
        assert await dom_page.locator(_SCHEDULE_BUTTON_SELECTOR).count() == 1

    async def test_current_live_anchor_wrapped_svg_id_icon_is_found(self, dom_page):
        """The selector this fix adds: an <a> wrapping an svg[id=...] icon.

        This is the exact shape live capture measured for the schedule
        control. Before this fix, `_icon_button` only checked
        `data-test-icon`/`use[href]` on a `<button>`, which matches neither
        the tag (`<a>`) nor the icon markup (`svg[id]`) LinkedIn uses now --
        this assertion fails against that old selector.
        """
        await _set_html(dom_page, _CURRENT_LIVE_MARKUP)
        matches = dom_page.locator(_SCHEDULE_BUTTON_SELECTOR)
        assert await matches.count() == 1
        tag = await matches.first.evaluate("el => el.tagName.toLowerCase()")
        assert tag == "a"

    async def test_view_all_scheduled_selector_matches_its_own_icon_only(
        self, dom_page
    ):
        """Only the arrow-right-small pill matches, not the clock icon."""
        await _set_html(dom_page, _CURRENT_LIVE_MARKUP)
        matches = dom_page.locator(_VIEW_ALL_SCHEDULED_SELECTOR)
        assert await matches.count() == 1

    async def test_pre_fix_selector_shape_does_not_match_live_markup(self, dom_page):
        """Mutation check: the old two-variant selector misses live markup.

        Rebuilds what `_icon_button` produced before this fix (no
        `svg[id=...]` variant) and proves *that* selector finds nothing in
        the current live markup -- the failure this whole fix addresses.
        """
        old_style_selector = (
            'button:has(svg[data-test-icon="clock-medium"]), '
            'button:has(use[href="#clock-medium"])'
        )
        await _set_html(dom_page, _CURRENT_LIVE_MARKUP)
        assert await dom_page.locator(old_style_selector).count() == 0
        # And the fixed selector does find it, in the same DOM.
        assert await dom_page.locator(_SCHEDULE_BUTTON_SELECTOR).count() == 1


def test_icon_button_helper_includes_svg_id_variant():
    """Fast non-DOM guard: the generated selector text has the new clause.

    Cheap enough to run without a browser, and a decent early signal if the
    helper's shape regresses -- the DOM tests above are what actually prove
    it matches.
    """
    selector = _icon_button("clock-medium")
    assert 'svg[id="clock-medium"]' in selector
    assert 'svg[data-test-icon="clock-medium"]' in selector
    assert 'use[href="#clock-medium"]' in selector
