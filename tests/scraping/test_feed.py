"""Tests for the home-feed scraping owner."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import asyncio
import logging
import time

import anyio
import pytest
from fastmcp import FastMCP

from linkedin_mcp_server.core.exceptions import AuthenticationError
from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.contracts import ExtractedSection
from linkedin_mcp_server.scraping.feed import FeedScraper
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession

from .policy_scenarios import _COMMON_ALLOWED, _page, _root, boundaries
from .support.policy_trace import (
    FakeClock,
    ScriptedPage,
    ScriptedResponse,
    TraceRecorder,
)


def _scraper(page) -> FeedScraper:
    """Wire the feed owner the way the facade does."""
    session = ScrapingSession(page)
    return FeedScraper(session, PageNavigator(session), PageContentReader(session))


class _ListenerPage:
    """A page that remembers which object was subscribed, not which shape.

    Playwright matches a listener by identity, so an equivalent callable is
    not the registered one. A double that compares behaviour would accept the
    replacement and leave the leak invisible.
    """

    def __init__(self, *, removal_error: Exception | None = None):
        self.added: list[Any] = []
        self.removed: list[Any] = []
        self.subscribed: list[Any] = []
        self._removal_error = removal_error

    def on(self, event: str, callback: Any) -> None:
        assert event == "response"
        self.added.append(callback)
        self.subscribed.append(callback)

    def remove_listener(self, event: str, callback: Any) -> None:
        assert event == "response"
        self.removed.append(callback)
        if self._removal_error is not None:
            raise self._removal_error
        # `list.remove` is identity for a function object, so a freshly built
        # equivalent raises here exactly as the browser would ignore it.
        self.subscribed.remove(callback)


class TestFeedListenerLifecycle:
    """Subscription and teardown around the scroll loop.

    The body is stubbed on the instance: what is under test is the frame
    around it, and the scroll loop needs a full browser to reach at all.
    """

    async def test_the_removed_listener_is_the_object_that_was_registered(self):
        page = _ListenerPage()
        scraper = _scraper(page)

        async def body(
            url: str,
            num_posts: int,
            captured_urls: list[str],
            pending_reads: list[asyncio.Task[None]],
        ) -> ExtractedSection:
            assert url == "https://www.linkedin.com/feed/"
            assert num_posts == 3
            return ExtractedSection(text="Feed content", references=[])

        with patch.object(scraper, "_extract_feed_body", body):
            result = await scraper._extract_feed_once(3)

        assert result.text == "Feed content"
        assert len(page.added) == 1
        assert len(page.removed) == 1
        assert page.removed[0] is page.added[0]
        # Nothing is left listening on the page the caller keeps using.
        assert page.subscribed == []

    async def test_the_reads_are_drained_even_when_the_removal_raises(self):
        page = _ListenerPage(removal_error=RuntimeError("listener already gone"))
        scraper = _scraper(page)
        reads: list[asyncio.Task[None]] = []

        async def failing_read() -> None:
            raise ValueError("body decode failed")

        async def body(
            url: str,
            num_posts: int,
            captured_urls: list[str],
            pending_reads: list[asyncio.Task[None]],
        ) -> ExtractedSection:
            task = asyncio.create_task(failing_read())
            await asyncio.wait({task})
            pending_reads.append(task)
            reads.append(task)
            return ExtractedSection(text="Feed content", references=[])

        with patch.object(scraper, "_extract_feed_body", body):
            result = await scraper._extract_feed_once(1)

        # The removal failure is swallowed rather than replacing the result.
        assert result.text == "Feed content"
        assert page.removed
        # And the drain still ran: the read's failure is consumed here instead
        # of resurfacing from the loop long after the feed call returned.
        assert reads[0]._log_traceback is False


class TestExtractFeedFailures:
    """The envelope ``extract_feed`` wraps around one attempt.

    Two lines decide which of ``get_feed``'s two paths a failure takes. A
    ``LinkedInScraperException`` is re-raised so the tool can hand it to
    ``handle_auth_error`` and ask the caller to close the stale browser and
    sign in again; anything else becomes a section error on a call that
    otherwise reports success. Swallowing the first turns a challenged
    session into a success payload carrying an empty feed, which is the
    one shape the recovery path exists to prevent.

    Patched on the instance: what is under test is the frame, and the
    attempt it wraps needs a full browser to reach at all.
    """

    @staticmethod
    def _once_raising(error: Exception):
        async def _extract_feed_once(num_posts: int) -> ExtractedSection:
            raise error

        return _extract_feed_once

    async def test_a_scraper_exception_reaches_the_tool_unwrapped(self):
        scraper = _scraper(_ListenerPage())
        challenged = AuthenticationError("LinkedIn challenged this session")

        with patch.object(
            scraper, "_extract_feed_once", self._once_raising(challenged)
        ):
            with pytest.raises(AuthenticationError) as raised:
                await scraper.extract_feed(num_posts=10)

        assert raised.value is challenged

    async def test_any_other_failure_becomes_a_section_error(self, caplog):
        scraper = _scraper(_ListenerPage())
        broken = RuntimeError("feed payload parser failed")

        with patch.object(scraper, "_extract_feed_once", self._once_raising(broken)):
            with caplog.at_level(logging.WARNING):
                result = await scraper.extract_feed(num_posts=10)

        assert result.text == ""
        assert result.references == []
        assert result.error is not None
        # The context names this workflow, so the issue report the diagnostics
        # write is filed against the feed rather than against the tool above it.
        assert result.error["context"] == "extract_feed"
        assert result.error["error_type"] == "RuntimeError"
        assert result.error["error_message"] == "feed payload parser failed"
        assert "Failed to extract feed: feed payload parser failed" in caplog.text


class TestFeedScrollCeiling:
    """``_MAX_SCROLLS``, on a feed that never stops producing.

    ``stale_count`` resets on every round that yields a permalink, so a
    feed that keeps loading never stale-stops and the ceiling is the only
    thing left to end the loop. Neither canonical fixture reaches it: one
    stale-stops after three wheels, the other satisfies ``num_posts=1`` on
    the first. The pacing is what differs here: exactly one batch lands per
    wheel, which keeps every round productive without ever reaching
    ``num_posts``, so the count the loop returns is the ceiling itself.
    """

    _CEILING = 12
    # The tool's own upper bound. Anything the loop could reach in twelve
    # rounds at one post per scroll stays far below it, which is the point:
    # the ceiling truncates the result the caller asked for.
    _NUM_POSTS = 50

    @staticmethod
    def _one_batch_per_wheel(page: ScriptedPage, index: int):
        """One SDUI payload, delivered only when the wheel fires."""
        slug = f"ceiling-ugcPost-{index}-example"
        response = ScriptedResponse(
            page.recorder,
            "https://www.linkedin.com/feed/",
            f'{{"postSlugUrl":"https://www.linkedin.com/posts/{slug}"}}'.encode(),
        )
        return lambda: page.emit("response", response)

    async def test_a_producing_feed_is_truncated_at_the_twelfth_scroll(self):
        recorder = TraceRecorder(
            "feed-scroll-ceiling",
            _COMMON_ALLOWED | {"response.body.start", "response.body.finish"},
        )
        clock = FakeClock(recorder)
        page = _page(recorder).script("evaluate:root_content", _root("Feed content"))
        page.script(
            "mouse.wheel",
            *[self._one_batch_per_wheel(page, index) for index in range(self._CEILING)],
        )
        scraper = _scraper(page)

        async with boundaries(recorder, clock):
            with recorder.context("extract_feed", "feed"):
                result = await scraper.extract_feed(num_posts=self._NUM_POSTS)

        wheels = [event for event in recorder.events if event["kind"] == "mouse.wheel"]
        # Exactly the literal, in both directions: a thirteenth wheel finds no
        # scripted batch behind it, and an eleventh leaves one unspent, which
        # ``assert_clean`` below reports as well.
        assert len(wheels) == self._CEILING
        # Every round produced, so nothing stopped for staleness here, and the
        # requested count is still nowhere near when the loop gives up.
        assert len(result.references) == self._CEILING
        assert len(result.references) < self._NUM_POSTS
        page.assert_clean()


class TestDrainListenerTasks:
    """Teardown of the feed response reads, on every path out of it.

    The reads are fire-and-forget: ``_extract_feed_once`` unsubscribes the
    response listener before it drains, so once this helper returns nothing
    in the process holds a reference that could still stop them. Every case
    here is therefore about what is left running afterwards.
    """

    @staticmethod
    async def _blocked_read() -> asyncio.Task[None]:
        """A started, cooperative read that never finishes on its own.

        Stands in for ``resp.body()`` on a response whose body never
        arrives, which is what the browser probe for this behaviour drove.
        """
        started = asyncio.Event()

        async def read() -> None:
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(read())
        await started.wait()
        return task

    async def test_an_empty_list_is_a_no_op(self):
        begun = time.monotonic()
        await FeedScraper._drain_listener_tasks([])
        assert time.monotonic() - begun < 0.5

    async def test_an_empty_list_never_suspends(self):
        """The fast path deliberately carries no checkpoint.

        The one at the end of the helper exists to deliver a deadline the
        shield held off, and there is no shield on this path: adding a
        checkpoint here would instead hand the caller's own pending
        cancellation to a teardown that did nothing. A counter that only
        advances when this task yields is what separates the two, and it
        has to be read without awaiting anything in between.
        """
        turns = 0

        async def count_turns() -> None:
            nonlocal turns
            while True:
                turns += 1
                await asyncio.sleep(0)

        counter = asyncio.create_task(count_turns())
        await asyncio.sleep(0)
        before = turns
        try:
            await FeedScraper._drain_listener_tasks([])
            after = turns
        finally:
            counter.cancel()

        assert before > 0, "the counter never started"
        assert after == before

    async def test_reads_that_finish_are_left_alone_and_their_failures_read(self):
        order: list[int] = []

        async def read(index: int) -> None:
            await asyncio.sleep(0.01)
            if index == 1:
                raise ValueError("body decode failed")
            order.append(index)

        reads = [asyncio.create_task(read(index)) for index in range(3)]
        begun = time.monotonic()
        await FeedScraper._drain_listener_tasks(reads)
        elapsed = time.monotonic() - begun

        assert order == [0, 2]
        assert all(task.done() for task in reads)
        assert not any(task.cancelled() for task in reads)
        # Left unretrieved, the failure resurfaces from the loop long after
        # the feed call returned. ``_log_traceback`` is the flag
        # ``Task.__del__`` reads for that, and no public API exposes it.
        assert reads[1]._log_traceback is False
        assert elapsed < 1.0

    async def test_a_stuck_read_is_cancelled_without_failing_the_call(self, caplog):
        """The ordinary slow-response path, with no outer cancellation.

        The read is cancelled here by the helper itself, so reading its
        result has to account for that: a bare ``exception()`` on it would
        re-raise the ``CancelledError`` and turn a successful feed call into
        a cancelled one from inside its own ``finally``.
        """
        read = await self._blocked_read()

        begun = time.monotonic()
        with caplog.at_level(logging.WARNING):
            await FeedScraper._drain_listener_tasks([read])
        elapsed = time.monotonic() - begun

        assert read.cancelled()
        # Two seconds of settling; the cancel is honoured well inside the
        # second that follows, so nothing is reported as left behind.
        assert 1.9 <= elapsed < 3.0, elapsed
        assert "leaking" not in caplog.text

    async def test_a_failed_read_is_still_read_when_the_drain_is_cancelled(self):
        """Cancelling the caller used to skip the consumption step entirely.

        The failure then belongs to nobody: the listener is gone, the feed
        call is unwinding, and the loop reports it whenever the task is
        finally collected.
        """

        async def read() -> None:
            raise ValueError("body decode failed")

        failed = asyncio.create_task(read())
        await asyncio.wait({failed})
        blocked = await self._blocked_read()

        drain = asyncio.create_task(
            FeedScraper._drain_listener_tasks([failed, blocked])
        )
        await asyncio.sleep(0.05)
        drain.cancel()

        done, _pending = await asyncio.wait({drain}, timeout=2.0)
        assert done == {drain}
        assert failed._log_traceback is False

    async def test_cancelling_the_drain_still_cancels_the_reads(self):
        """The caller's cancellation reaches this helper mid-wait.

        A tool timeout lands here, and previously the first
        ``asyncio.wait`` just propagated it: measured against a real
        ``resp.body()``, the read stayed pending afterwards with
        ``cancelling() == 0``, with the listener already unsubscribed.
        """
        read = await self._blocked_read()

        drain = asyncio.create_task(FeedScraper._drain_listener_tasks([read]))
        # Let the drain reach its first wait before cancelling it.
        await asyncio.sleep(0.05)
        drain.cancel()

        done, _pending = await asyncio.wait({drain}, timeout=2.0)

        assert done == {drain}
        # Cancellation is not converted into a successful teardown.
        assert drain.cancelled()
        # The read was asked to stop, and being cooperative it is already
        # finished by the time the helper gives up ownership of it.
        assert read.cancelling() >= 1
        assert read.done()
        assert read.cancelled()

    async def test_a_repeated_cancellation_still_leaves_the_reads_cancelled(self):
        """A second request lands while the helper is already in teardown."""
        read = await self._blocked_read()

        drain = asyncio.create_task(FeedScraper._drain_listener_tasks([read]))
        await asyncio.sleep(0.05)
        drain.cancel()
        await asyncio.sleep(0)
        drain.cancel()

        done, _pending = await asyncio.wait({drain}, timeout=2.0)
        assert done == {drain}
        assert drain.cancelled()
        assert read.cancelling() >= 1

        finished, _still = await asyncio.wait({read}, timeout=2.0)
        assert finished == {read}
        assert read.cancelled()

    async def test_a_second_cancellation_inside_the_cleanup_still_reports(self, caplog):
        """The shield holds off AnyIO's delivery, not a plain ``cancel()``.

        The cooperative case above cannot see this: both reads are settled
        by the time the second request lands, so nothing is left to read or
        report. Here a failure arrived before the cancel and a read outlives
        it, and the second request cuts the bounded wait short between the
        two.
        """

        async def failing() -> None:
            raise ValueError("body decode failed")

        failed = asyncio.create_task(failing())
        await asyncio.wait({failed})

        release = asyncio.Event()
        started = asyncio.Event()

        async def stubborn() -> None:
            started.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass

        read = asyncio.create_task(stubborn())
        await started.wait()
        drain = asyncio.create_task(FeedScraper._drain_listener_tasks([failed, read]))

        try:
            with caplog.at_level(logging.WARNING):
                await asyncio.sleep(0.05)
                drain.cancel()
                # Land the second request inside the bounded wait. The cancel
                # loop and that wait are one stretch with no await between
                # them, so a read that has been asked to stop means the drain
                # is already suspended in it.
                for _ in range(100):
                    await asyncio.sleep(0)
                    if read.cancelling() >= 1:
                        break
                assert read.cancelling() >= 1, "drain never reached its cleanup"
                drain.cancel()

                done, _pending = await asyncio.wait({drain}, timeout=2.0)

            assert done == {drain}
            assert drain.cancelled()
            # The read refused the cancel, so it is genuinely still running
            # and has to be named rather than passed over in silence.
            assert not read.done()
            assert "leaking 1 task(s)" in caplog.text
            # And the failure that landed before any of this is read, not
            # left for the loop to report against an unrelated call.
            assert failed._log_traceback is False
        finally:
            release.set()
            await asyncio.wait({drain, read}, timeout=2.0)

    async def test_a_tool_deadline_does_not_cut_the_bounded_cleanup_short(self):
        """FastMCP runs every tool call inside ``anyio.fail_after``.

        That scope re-delivers its cancellation on every loop iteration
        until the task leaves it, so an unshielded wait in the teardown is
        cancelled again as soon as it starts. A read that unwinds within a
        single iteration cannot show this, because the one iteration it
        needs is granted either way; this one awaits on its cleanup path,
        which is what makes the difference observable.
        """
        started = asyncio.Event()
        unwound = False

        async def read() -> None:
            nonlocal unwound
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                for _ in range(20):
                    await asyncio.sleep(0)
                unwound = True

        task = asyncio.create_task(read())
        await started.wait()

        with pytest.raises(TimeoutError):
            with anyio.fail_after(0.2):
                await FeedScraper._drain_listener_tasks([task])

        # Asserted without awaiting anything first: further loop iterations
        # would let the read unwind on its own, and the assertions would then
        # hold whether or not the cleanup was shielded.
        assert task.cancelling() >= 1
        assert unwound
        assert task.done()

    async def test_a_deadline_falling_due_inside_the_shield_still_fires(self):
        """The shield can swallow the moment a deadline comes due.

        AnyIO skips a shielded scope while delivering, and the restart on
        the way out runs inside this task, where it can only schedule
        delivery for the next turn. The ``fail_after(0.2)`` case above
        never reaches that: its deadline is already past before the first
        wait ends, so the cancel is delivered before the shield is entered.
        Here it first comes due while the shield is open, and a caller that
        does not suspend again would carry the expired deadline to a
        successful return.
        """
        started = asyncio.Event()

        async def read() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                # Outlives the deadline, so the shield is still open when it
                # falls due, and closes only afterwards.
                await asyncio.sleep(0.7)

        task = asyncio.create_task(read())
        await started.wait()

        reached_the_caller = False
        with pytest.raises(TimeoutError):
            with anyio.fail_after(2.3):
                await FeedScraper._drain_listener_tasks([task])
                # Nothing between here and the scope's close suspends, which
                # is exactly get_feed's own report_progress when the client
                # sent no progress token.
                reached_the_caller = True

        assert not reached_the_caller
        assert task.cancelling() >= 1

    async def test_a_read_refusing_cancellation_cannot_outlast_the_ceiling(
        self, caplog
    ):
        """The three-second ceiling the docstring claims, measured.

        Waiting on ``gather`` waits for the requested cancellation to
        *complete*, so a read that swallows ``CancelledError`` held teardown
        open for as long as it liked. The outer deadline here is an
        ``asyncio.wait`` rather than a ``wait_for``, which would cancel the
        drain itself and measure something else.
        """
        release = asyncio.Event()
        started = asyncio.Event()
        cancels = 0

        async def stubborn() -> None:
            nonlocal cancels
            started.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancels += 1

        read = asyncio.create_task(stubborn())
        await started.wait()
        drain = asyncio.create_task(FeedScraper._drain_listener_tasks([read]))

        try:
            with caplog.at_level(logging.WARNING):
                begun = time.monotonic()
                done, _pending = await asyncio.wait({drain}, timeout=4.5)
                elapsed = time.monotonic() - begun

            assert done == {drain}, "drain still running 4.5s into a 3s ceiling"
            assert drain.exception() is None
            # Two seconds of settling plus one after the cancel, and nothing
            # spent waiting on the cancellation itself to be honoured.
            assert 2.9 <= elapsed < 4.0, elapsed
            assert cancels == 1
            assert not read.done()
            assert "leaking 1 task(s)" in caplog.text
        finally:
            release.set()
            await asyncio.wait({drain, read}, timeout=2.0)


async def test_listener_drain_waits_two_seconds_then_cancels_with_one_second_cap(
    monkeypatch,
):
    waits: list[float | None] = []

    async def blocked() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(blocked())

    async def wait(
        pending: Any, *, timeout: float | None = None
    ) -> tuple[set[Any], set[Any]]:
        waits.append(timeout)
        return set(), set(pending)

    monkeypatch.setattr(asyncio, "wait", wait)

    await FeedScraper._drain_listener_tasks([task])

    assert task.cancelled()
    assert waits == [2.0, 1.0]


async def test_listener_drain_logs_an_uncooperative_task(monkeypatch, caplog):
    class PendingTask:
        cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

        def done(self) -> bool:
            return False

    task = PendingTask()

    waits: list[float | None] = []

    async def wait(
        _pending: Any, *, timeout: float | None = None
    ) -> tuple[set[Any], set[Any]]:
        waits.append(timeout)
        return set(), {task}

    monkeypatch.setattr(asyncio, "wait", wait)

    with caplog.at_level(logging.WARNING):
        await FeedScraper._drain_listener_tasks([cast(asyncio.Task[None], task)])

    assert task.cancelled is True
    assert waits == [2.0, 1.0]
    assert "leaking 1 task(s)" in caplog.text


class TestFeedToolDeadline:
    """A tool deadline that comes due inside the cleanup shield.

    ``_drain_listener_tasks`` shields its bounded teardown, and AnyIO does
    not deliver into a shielded scope: on the way out it can only schedule
    delivery for the next turn. ``get_feed`` then calls ``report_progress``,
    which suspends only when the client sent a progress token, so the two
    cases have to be driven separately. Both go over a real client session
    against a real ``anyio.fail_after``; nothing about the timeout is mocked.
    """

    @staticmethod
    def _server_and_reads(mcp_timeout: float, cleanup: float):
        from linkedin_mcp_server.tools.feed import register_feed_tools

        reads: list[asyncio.Task[None]] = []

        async def extract_feed(num_posts: int = 1) -> ExtractedSection:
            started = asyncio.Event()

            async def read() -> None:
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    # Holds the shield open across the deadline.
                    await asyncio.sleep(cleanup)

            task = asyncio.create_task(read())
            reads.append(task)
            await started.wait()
            await FeedScraper._drain_listener_tasks([task])
            return ExtractedSection(text="synthetic feed", references=[])

        mcp = FastMCP("deadline-test")
        register_feed_tools(mcp, tool_timeout=mcp_timeout)
        return mcp, reads, SimpleNamespace(extract_feed=extract_feed)

    async def _call(self, use_session: bool):
        from fastmcp import Client

        from linkedin_mcp_server.tools import feed as feed_tools

        mcp, reads, extractor = self._server_and_reads(2.5, 0.7)
        try:
            with patch.object(
                feed_tools,
                "get_ready_extractor",
                AsyncMock(return_value=extractor),
            ):
                async with Client(mcp) as client:
                    if use_session:
                        # No progress token: report_progress never suspends.
                        return await client.session.call_tool(
                            "get_feed", {"num_posts": 1}
                        )
                    # Client.call_tool installs a progress handler, so the
                    # request carries a token and report_progress awaits.
                    return await client.call_tool("get_feed", {"num_posts": 1})
        finally:
            for task in reads:
                if not task.done():
                    task.cancel()
            if reads:
                await asyncio.wait(reads, timeout=2.0)

    async def test_the_deadline_fires_without_a_progress_token(self):
        result = await self._call(use_session=True)

        assert result.isError, "expired call returned a feed result"
        assert "timed out" in str(result.content)

    async def test_the_deadline_fires_with_a_progress_token(self):
        from fastmcp.exceptions import ToolError

        with pytest.raises(ToolError, match="timed out"):
            await self._call(use_session=False)


class _FakeRadioLocator:
    """A `[role="radio"]` locator double over a shared list of pill states.

    ``flip_on_click`` toggles whether ``click`` actually flips the target's
    ``checked`` flag -- False reproduces a click that "succeeds" (no
    exception) but never changes the pill's accessible state, the case
    ``_select_notification_filter`` must catch rather than trust the click.
    """

    def __init__(
        self,
        radios: list[dict[str, bool]],
        indices: list[int] | None = None,
        *,
        flip_on_click: bool = True,
    ):
        self._radios = radios
        self._indices = indices if indices is not None else list(range(len(radios)))
        self._flip_on_click = flip_on_click

    async def count(self) -> int:
        return len(self._indices)

    def nth(self, index: int) -> "_FakeRadioLocator":
        return _FakeRadioLocator(
            self._radios, [self._indices[index]], flip_on_click=self._flip_on_click
        )

    async def click(self, timeout: int | None = None) -> None:
        if self._flip_on_click:
            idx = self._indices[0]
            for other in self._radios:
                other["checked"] = False
            self._radios[idx]["checked"] = True

    async def get_attribute(self, name: str) -> str:
        assert name == "aria-checked"
        idx = self._indices[0]
        return "true" if self._radios[idx]["checked"] else "false"


class _FakeRadioPage:
    def __init__(self, radios: list[dict[str, bool]], *, flip_on_click: bool = True):
        self._radios = radios
        self._flip_on_click = flip_on_click

    def locator(self, selector: str) -> _FakeRadioLocator:
        from linkedin_mcp_server.scraping.feed import _FILTER_PILL_SELECTOR

        assert selector == _FILTER_PILL_SELECTOR
        return _FakeRadioLocator(self._radios, flip_on_click=self._flip_on_click)


def _filter_scraper(radios: list[dict[str, bool]], *, flip_on_click: bool = True):
    """A FeedScraper wired only with what ``_select_notification_filter`` reads.

    Built via ``__new__`` rather than the real constructor: the method under
    test never touches ``_navigator``/``_content``, and wiring those up (real
    navigation, rate-limit checks) would only obscure what this is asserting
    -- the pill-position mapping and the click-verification loop.
    """
    scraper = FeedScraper.__new__(FeedScraper)
    scraper._session = SimpleNamespace(
        page=_FakeRadioPage(radios, flip_on_click=flip_on_click),
        delay=AsyncMock(),
    )
    return scraper


class TestSelectNotificationFilterPositionAndVerification:
    """The click-by-position + verify-state mechanism, isolated.

    Live capture 2026-09-17 proved LinkedIn drops the old `?filterType=`
    query parameter entirely (see the module-level comment in
    ``scraping/feed.py`` above ``_NOTIFICATION_FILTER_PILL_INDEX``), so
    filtering has to click a structural pill and verify the click actually
    changed the pill's ``aria-checked`` state -- never trust a click that
    "succeeded" with no exception, and never fall back to unfiltered content.
    """

    async def test_mentions_targets_the_third_radio_pill(self):
        radios = [
            {"checked": True},  # "All", checked by default
            {"checked": False},
            {"checked": False},
        ]
        scraper = _filter_scraper(radios)

        applied = await scraper._select_notification_filter("mentions")

        assert applied is True
        assert radios[2]["checked"] is True
        assert radios[0]["checked"] is False  # the old selection was cleared

    async def test_my_posts_targets_the_second_radio_pill(self):
        radios = [{"checked": True}, {"checked": False}, {"checked": False}]
        scraper = _filter_scraper(radios)

        applied = await scraper._select_notification_filter("my_posts")

        assert applied is True
        assert radios[1]["checked"] is True

    async def test_fewer_than_three_radio_pills_is_refused_without_clicking(self):
        # Only the default "All" pill is present -- LinkedIn's markup
        # changed, or this ran before the row finished rendering.
        radios = [{"checked": True}]
        scraper = _filter_scraper(radios)

        applied = await scraper._select_notification_filter("mentions")

        assert applied is False
        assert radios[0]["checked"] is True  # untouched

    async def test_a_click_that_never_flips_aria_checked_is_reported_false(self):
        """The click "succeeds" (no exception) but the pill never checks.

        This is exactly the class of bug the fix exists to catch: a click
        landing on the wrong element, or LinkedIn's own state update lagging
        past the verification budget. Trusting the click alone (no
        exception raised) would silently return unfiltered content
        mislabeled as filtered -- the one outcome AGENTS.md rules out.
        """
        radios = [{"checked": True}, {"checked": False}, {"checked": False}]
        scraper = _filter_scraper(radios, flip_on_click=False)

        applied = await scraper._select_notification_filter("mentions")

        assert applied is False

    async def test_all_is_not_a_valid_click_target(self):
        """ "all" has no pill index -- it is the default view, never clicked."""
        radios = [{"checked": True}, {"checked": False}, {"checked": False}]
        scraper = _filter_scraper(radios)

        applied = await scraper._select_notification_filter("all")

        assert applied is False


def _notification_scraper():
    """A FeedScraper with session/navigator/content mocked for envelope tests."""
    session = MagicMock()
    page = MagicMock()
    session.page = page
    session.check_rate_limit = AsyncMock()
    session.dismiss_modal = AsyncMock()
    session.scroll_body = AsyncMock()
    session.delay = AsyncMock()
    page.wait_for_selector = AsyncMock()

    navigator = MagicMock()
    navigator._navigate_to_page = AsyncMock()

    content = MagicMock()
    content._extract_root_content = AsyncMock(
        return_value={"text": "Someone reacted to your post", "references": []}
    )

    scraper = FeedScraper(session, navigator, content)
    return scraper, session, navigator, content


class TestExtractNotificationsFilterEnvelope:
    """``extract_notifications``'s decision to extract vs. refuse.

    ``_select_notification_filter`` is patched on the instance here: this
    layer's own job is the envelope around it (skip selection for "all",
    refuse to read the page at all when selection could not be verified),
    which the position/verification tests above already cover directly.
    """

    async def test_all_filter_never_touches_pill_selection(self):
        scraper, *_ = _notification_scraper()

        with patch.object(
            scraper, "_select_notification_filter", AsyncMock()
        ) as select:
            result = await scraper.extract_notifications("all")

        select.assert_not_awaited()
        assert result.text == "Someone reacted to your post"
        assert result.error is None

    async def test_a_verified_filter_click_proceeds_to_extraction(self):
        scraper, *_ = _notification_scraper()

        with patch.object(
            scraper, "_select_notification_filter", AsyncMock(return_value=True)
        ):
            result = await scraper.extract_notifications("mentions")

        assert result.text == "Someone reacted to your post"
        assert result.error is None

    async def test_an_unverified_filter_click_never_reads_the_page(self):
        """The hard rule this whole fix exists for.

        When the pill's selected state cannot be verified, the page is never
        even read -- returning its content, unfiltered, labelled "mentions"
        would be exactly the silent-wrong-result bug being fixed.
        """
        scraper, _session, _navigator, content = _notification_scraper()

        with patch.object(
            scraper, "_select_notification_filter", AsyncMock(return_value=False)
        ):
            result = await scraper.extract_notifications("my_posts")

        assert result.text == ""
        assert result.error is not None
        assert result.error["error_type"] == "filter_unavailable"
        content._extract_root_content.assert_not_awaited()
