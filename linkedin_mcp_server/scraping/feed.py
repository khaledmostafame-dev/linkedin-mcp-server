"""Home-feed scraping with SDUI permalink capture."""

from __future__ import annotations

from typing import Any

import asyncio
import logging

from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.core.exceptions import LinkedInScraperException
from linkedin_mcp_server.error_diagnostics import build_issue_diagnostics
from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.contracts import (
    RATE_LIMITED_SECTION_TEXT,
    ExtractedSection,
)
from linkedin_mcp_server.scraping.feed_payload import (
    POST_SLUG_URL_RE,
    build_feed_references,
    is_feed_payload_response,
)
from linkedin_mcp_server.scraping.link_metadata import build_references
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.response_capture import (
    drain_listener_tasks as _drain_listener_tasks_impl,
)
from linkedin_mcp_server.scraping.session import ScrapingSession
from linkedin_mcp_server.scraping.text import (
    filter_linkedin_noise_lines,
    truncate_linkedin_noise,
)

logger = logging.getLogger(__name__)

NOTIFICATIONS_URL = "https://www.linkedin.com/notifications/"

# LinkedIn drops an unrecognised query parameter rather than erroring: live
# capture 2026-09-17 (`007-extractor-after-goto.json`) navigated with
# `?filterType=MENTIONS` and landed on `/notifications/` with `query_params:
# []` -- the parameter never reached the page, so "mentions" silently
# returned the same content as "all". The same capture's node list shows the
# real mechanism instead: a row of four sibling pills at depth 14, three
# exposing `role="radio"` + `aria-checked` (nodes 165/169/177 -- the first
# checked, matching "All" being the default view) and a fourth rendered as a
# plain `<a href="/notifications/?filter=...">` (node 173) whose href *does*
# carry a query parameter -- but named `filter`, not `filterType`, and the
# capture strips attribute values, so the token(s) LinkedIn assigns to
# "my_posts"/"mentions" were not captured either. Guessing a new token would
# repeat the same class of bug with a different name, so filtering instead
# clicks the pill at the target's position and verifies its `aria-checked`
# flips to "true" -- position, role and state, never text, per AGENTS.md's
# scraping rules. Only the three true `role="radio"` pills are treated as
# candidates (the fourth, plain-anchor pill exposes no checked state to
# verify against, so it is not a safe click target for a checkable filter).
# This position mapping is still a best-effort assumption -- the capture
# that proved the mechanism could not also prove which of the two unchecked
# pills is "my_posts" and which is "mentions" -- so a click that does not
# verifiably flip the state, or fewer than three radio pills at all, returns
# `filter_unavailable` rather than unfiltered content mislabeled as filtered.
_NOTIFICATION_FILTER_PILL_INDEX = {"my_posts": 1, "mentions": 2}
_FILTER_PILL_SELECTOR = '[role="radio"]'
_FILTER_VERIFY_ATTEMPTS = 10
_FILTER_VERIFY_INTERVAL = 0.2


class FeedScraper:
    """Scrape the home feed and the post permalinks its SDUI payloads carry."""

    def __init__(
        self,
        session: ScrapingSession,
        navigator: PageNavigator,
        content: PageContentReader,
    ):
        self._session = session
        self._navigator = navigator
        self._content = content

    @staticmethod
    async def _drain_listener_tasks(pending: list[asyncio.Task[None]]) -> None:
        """Bounded teardown for fire-and-forget response listener tasks.

        The feed scroll loop appends a read task per matching response;
        those tasks must finish (or be cancelled) before we leave the
        extractor or the event loop's "Task exception was never retrieved"
        warnings will surface unrelated errors.

        Delegates to ``response_capture.drain_listener_tasks``, the same
        teardown any posts-listing page's permalink capture uses — see that
        function's docstring for the full incident history behind each line.
        Kept as a method here (rather than switching call sites to import
        the shared function directly) so this class's own tests, which patch
        and call ``_drain_listener_tasks`` by name, keep working unmodified.
        """
        await _drain_listener_tasks_impl(pending)

    async def extract_feed(
        self,
        num_posts: int = 10,
    ) -> ExtractedSection:
        """Scrape the LinkedIn home feed, scrolling until *num_posts* are loaded."""
        try:
            return await self._extract_feed_once(num_posts)
        except LinkedInScraperException:
            raise
        except Exception as e:
            logger.warning("Failed to extract feed: %s", e)
            return ExtractedSection(
                text="",
                references=[],
                error=build_issue_diagnostics(e, context="extract_feed"),
            )

    async def _extract_feed_once(
        self,
        num_posts: int,
    ) -> ExtractedSection:
        """Single attempt: navigate, scroll until post count, extract."""
        url = "https://www.linkedin.com/feed/"
        page = self._session.page

        # Post permalinks live in the SDUI pagination response (field:
        # "postSlugUrl"). The initial /feed/ HTML embeds the same data in
        # an RSC flight payload. Listen for both during the whole scroll
        # loop. ``seen_urls`` doubles as the locale-independent scroll
        # progress signal, replacing the previous "Feed post" innerText
        # marker that broke on non-English UIs.
        captured_urls: list[str] = []
        seen_urls: set[str] = set()
        pending_reads: list[asyncio.Task[None]] = []

        def _handle_response(resp: Any) -> None:
            if not is_feed_payload_response(resp.url):
                return

            async def _read() -> None:
                try:
                    body = await resp.body()
                except Exception:
                    return
                if not body:
                    return
                text = body.decode("utf-8", errors="replace")
                for match in POST_SLUG_URL_RE.finditer(text):
                    post_url = f"https://www.linkedin.com/posts/{match.group('slug')}"
                    if post_url not in seen_urls:
                        seen_urls.add(post_url)
                        captured_urls.append(post_url)

            pending_reads.append(asyncio.create_task(_read()))

        page.on("response", _handle_response)
        try:
            return await self._extract_feed_body(
                url, num_posts, captured_urls, pending_reads
            )
        finally:
            try:
                # The very object that was registered, never a fresh equivalent:
                # Playwright matches a listener by identity, so a re-created
                # closure removes nothing and leaves the read subscribed for the
                # rest of the page's life. The drain below runs either way,
                # because a removal that raised is exactly the case where the
                # reads still need stopping.
                page.remove_listener("response", _handle_response)
            except Exception:
                pass
            await self._drain_listener_tasks(pending_reads)

    async def _extract_feed_body(
        self,
        url: str,
        num_posts: int,
        captured_urls: list[str],
        pending_reads: list[asyncio.Task[None]],
    ) -> ExtractedSection:
        page = self._session.page
        await self._navigator._navigate_to_page(url)
        await self._session.check_rate_limit()

        try:
            await page.wait_for_selector("main")
        except PlaywrightTimeoutError:
            logger.debug("No <main> element found on %s", url)

        await self._session.dismiss_modal()

        try:
            await page.wait_for_function(
                """() => {
                    const main = document.querySelector('main');
                    if (!main) return false;
                    return main.innerText.length > 200;
                }""",
                timeout=10000,
            )
        except PlaywrightTimeoutError:
            logger.debug("Feed content did not appear on %s", url)

        # The feed has its own scroll container — window.scrollTo is a no-op.
        # mouse.wheel over the viewport center triggers the real scroll.
        _MAX_SCROLLS = 12
        _MAX_STALE = 3
        _BATCH_WAIT = 6.0
        _WHEEL_DELTA = 2000
        _IN_LOOP_DRAIN_TIMEOUT = 1.0
        stale_count = 0

        viewport = page.viewport_size or {"width": 1280, "height": 720}
        cx, cy = viewport["width"] // 2, viewport["height"] // 2
        await page.mouse.move(cx, cy)

        for i in range(_MAX_SCROLLS):
            count = len(captured_urls)
            logger.debug("Feed scroll %d: %d permalinks captured", i, count)
            if count >= num_posts:
                break

            await page.mouse.wheel(0, _WHEEL_DELTA)

            new_count = count
            for _ in range(int(_BATCH_WAIT)):
                await self._session.delay(1.0)
                # Drain in-flight response reads so captured_urls reflects
                # everything Playwright already delivered. Without this,
                # the count comparison races: the wheel fires a network
                # response, the listener creates a read task, and the loop
                # sleeps and re-checks before _read() finishes appending —
                # producing false-stale verdicts.
                if pending_reads:
                    done, _still = await asyncio.wait(
                        pending_reads, timeout=_IN_LOOP_DRAIN_TIMEOUT
                    )
                    if done:
                        # Surface unexpected exceptions. _read() catches
                        # expected playwright errors, but a parser bug
                        # would otherwise vanish into the loop. Log them
                        # rather than raising so a single bad response
                        # doesn't abort the whole scroll session.
                        for result in await asyncio.gather(
                            *done, return_exceptions=True
                        ):
                            if isinstance(result, BaseException):
                                logger.warning(
                                    "Unhandled error in feed _read task: %r",
                                    result,
                                )
                    pending_reads[:] = [t for t in pending_reads if not t.done()]
                new_count = len(captured_urls)
                if new_count > count:
                    break

            if new_count > count:
                stale_count = 0
            else:
                stale_count += 1
                logger.debug(
                    "Feed stale scroll %d/%d (still at %d permalinks)",
                    stale_count,
                    _MAX_STALE,
                    new_count,
                )
                if stale_count >= _MAX_STALE:
                    logger.debug("Feed stopped producing new posts")
                    break

        # Give any in-flight response reads a beat to finish recording URLs.
        await self._session.delay(0.2)

        raw_result = await self._content._extract_root_content(["main"])
        raw = raw_result["text"]

        if not raw:
            return ExtractedSection(text="", references=[])
        truncated = truncate_linkedin_noise(raw)
        if not truncated and raw.strip():
            logger.warning(
                "Page %s returned only LinkedIn chrome (likely rate-limited)", url
            )
            return ExtractedSection(text=RATE_LIMITED_SECTION_TEXT, references=[])
        cleaned = filter_linkedin_noise_lines(truncated)
        return ExtractedSection(
            text=cleaned,
            references=build_feed_references(raw_result["references"], captured_urls),
        )

    async def extract_notifications(
        self,
        filter_: str = "all",
        max_scrolls: int = 6,
    ) -> ExtractedSection:
        """Scrape the notifications page, optionally selecting a filter pill.

        See the module-level comment above ``_NOTIFICATION_FILTER_PILL_INDEX``
        for the live-capture evidence behind this mechanism (structural
        click + state verification, not a query parameter).
        """
        try:
            return await self._extract_notifications_once(filter_, max_scrolls)
        except LinkedInScraperException:
            raise
        except Exception as e:
            logger.warning("Failed to extract notifications: %s", e)
            return ExtractedSection(
                text="",
                references=[],
                error=build_issue_diagnostics(e, context="extract_notifications"),
            )

    async def _extract_notifications_once(
        self,
        filter_: str,
        max_scrolls: int,
    ) -> ExtractedSection:
        page = self._session.page
        await self._navigator._navigate_to_page(NOTIFICATIONS_URL)
        await self._session.check_rate_limit()

        try:
            await page.wait_for_selector("main")
        except PlaywrightTimeoutError:
            logger.debug("No <main> element found on %s", NOTIFICATIONS_URL)

        await self._session.dismiss_modal()

        if filter_ != "all":
            applied = await self._select_notification_filter(filter_)
            if not applied:
                return ExtractedSection(
                    text="",
                    references=[],
                    error={
                        "error_type": "filter_unavailable",
                        "error_message": (
                            f"Could not verify that the {filter_!r} notifications "
                            "filter was applied: LinkedIn's filter pills carry no "
                            "text this fork can read, so the pill for this filter "
                            "cannot be identified with certainty, or the click did "
                            "not verifiably change its selected state. Returning "
                            "nothing rather than unfiltered content mislabeled as "
                            "filtered."
                        ),
                    },
                )

        await self._session.scroll_body(pause_time=0.5, max_scrolls=max_scrolls)

        raw_result = await self._content._extract_root_content(["main"])
        raw = raw_result["text"]

        if not raw:
            return ExtractedSection(text="", references=[])
        truncated = truncate_linkedin_noise(raw)
        if not truncated and raw.strip():
            logger.warning(
                "Page %s returned only LinkedIn chrome (likely rate-limited)",
                NOTIFICATIONS_URL,
            )
            return ExtractedSection(text=RATE_LIMITED_SECTION_TEXT, references=[])
        cleaned = filter_linkedin_noise_lines(truncated)
        return ExtractedSection(
            text=cleaned,
            references=build_references(raw_result["references"], "notifications"),
        )

    async def _select_notification_filter(self, filter_: str) -> bool:
        """Click the filter pill at ``filter_``'s position; verify it checked.

        Returns False (never raises) when fewer than three ``role="radio"``
        pills are present, the target index is unknown, or the click does
        not verifiably flip ``aria-checked`` to "true" within the budget.
        """
        index = _NOTIFICATION_FILTER_PILL_INDEX.get(filter_)
        if index is None:
            return False
        radios = self._session.page.locator(_FILTER_PILL_SELECTOR)
        if await radios.count() <= index:
            return False
        target = radios.nth(index)
        try:
            await target.click(timeout=5000)
        except Exception:
            logger.debug("Notification filter pill click failed", exc_info=True)
            return False
        for _ in range(_FILTER_VERIFY_ATTEMPTS):
            try:
                if await target.get_attribute("aria-checked") == "true":
                    return True
            except Exception:
                break
            await self._session.delay(_FILTER_VERIFY_INTERVAL)
        return False
