"""Shared teardown for fire-and-forget network-response listener tasks.

Any scraping workflow that installs a Playwright ``response`` listener to
capture data alongside a page navigation and scroll loop (post permalinks
today; potentially other payload captures later) needs the same bounded
teardown: the response-body reads it starts are fire-and-forget from the
listener callback, and they must finish (or be cancelled) before the caller
leaves the page, or the event loop's "Task exception was never retrieved"
warnings surface unrelated errors later.

This is a straight extraction of ``FeedScraper._drain_listener_tasks`` (see
that method's own history for the incident behind each line — a stuck
``resp.body()``, FastMCP's ``anyio.fail_after`` re-delivering cancellation on
every loop iteration, a shield that can itself swallow a deadline). Behavior
is unchanged; ``FeedScraper`` keeps its own entry point and delegates here so
its existing tests (which patch and call ``_drain_listener_tasks`` directly)
keep working unmodified.
"""

from __future__ import annotations

import asyncio
import logging

import anyio
import anyio.lowlevel

logger = logging.getLogger(__name__)


async def drain_listener_tasks(pending: list[asyncio.Task[None]]) -> None:
    """Bounded teardown for fire-and-forget response listener tasks.

    The caller unsubscribes its response listener before calling this, so
    once it returns nothing in the process holds a reference that could
    still stop the reads it started. Two seconds of cooperative waiting,
    then cancellation, then one more second to observe the cancellation —
    a read that refuses to unwind past that is logged as a leak rather than
    left to block teardown indefinitely.
    """
    if not pending:
        return
    try:
        await asyncio.wait(pending, timeout=2.0)
    finally:
        # Cancel on *every* exit of that wait, the caller's own cancellation
        # included. The response listener is unsubscribed before we get here,
        # so no one else will ever ask these reads to stop; returning through
        # the cancelled path without asking left a real ``resp.body()``
        # running with no cancellation requested at all.
        for task in pending:
            if not task.done():
                task.cancel()
        # FastMCP wraps each tool call in ``anyio.fail_after``, whose scope
        # re-delivers its cancellation on every loop iteration until the task
        # leaves it. Unshielded, the wait below would be cancelled before the
        # reads it watches can act on the cancel above, which is the case the
        # budget exists for. The shield covers a bounded wait only, and the
        # outer cancellation resumes as soon as the scope closes.
        with anyio.CancelScope(shield=True):
            try:
                await asyncio.wait(pending, timeout=1.0)
            finally:
                # A shield only holds off AnyIO's own delivery, so a second
                # plain ``Task.cancel()`` still cuts that wait short. Read
                # and report the reads as they actually stand, or a failure
                # that arrived before the cancel is left for the loop to
                # report and a task still running is left unmentioned.
                leftover = [task for task in pending if not task.done()]
                for task in pending:
                    if task.done() and not task.cancelled():
                        # Consume the failure; unretrieved, it reaches the
                        # loop's handler long after the caller returned.
                        task.exception()
                if leftover:
                    logger.warning(
                        "Response listener tasks did not drain after cancel; "
                        "leaking %d task(s)",
                        len(leftover),
                    )
    # A deadline that first comes due inside the shield has nowhere to land:
    # AnyIO skips a shielded scope while delivering, and the restart on the
    # way out runs in this very task, so it can only schedule delivery for the
    # next turn. This unshielded checkpoint is that next turn, so a
    # cancellation deferred by the shield is not silently lost to a caller
    # whose next step never suspends on its own.
    await anyio.lowlevel.checkpoint()
