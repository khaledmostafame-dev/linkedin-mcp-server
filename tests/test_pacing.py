"""Account pacing: spacing, rolling caps, cooldown, persistence, status."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import mcp.types as mt
import pytest
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import MiddlewareContext

from linkedin_mcp_server import pacing_signals
from linkedin_mcp_server.config.schema import PacingConfig
from linkedin_mcp_server.pacing import (
    DAY,
    HOUR,
    MAX_INLINE_WAIT_SECONDS,
    PacingMiddleware,
    PacingState,
    PacingStateStore,
    ToolKind,
    classify_tool,
)

START = 1_800_000_000.0


class FakeClock:
    """Wall clock that only moves when a test or a pacing wait moves it."""

    def __init__(self, now: float = START) -> None:
        self.now = now
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _policy(**overrides: Any) -> PacingConfig:
    values: dict[str, Any] = {
        "enabled": True,
        "min_interval_seconds": 0.0,
        "jitter_seconds": 0.0,
        "write_min_interval_seconds": 0.0,
        "write_jitter_seconds": 0.0,
        "max_reads_per_hour": 0,
        "max_writes_per_hour": 0,
        "max_writes_per_day": 0,
        "private_write_min_interval_seconds": 0.0,
        "private_write_jitter_seconds": 0.0,
        "max_private_writes_per_hour": 0,
        "max_private_writes_per_day": 0,
        "max_calls_per_minute": 0,
        "cooldown_base_seconds": 0.0,
    }
    values.update(overrides)
    return PacingConfig(**values)


def _middleware(
    tmp_path: Path, clock: FakeClock, policy: PacingConfig, *, seed: int = 7
) -> PacingMiddleware:
    return PacingMiddleware(
        policy=lambda: policy,
        store=PacingStateStore(lambda: tmp_path / "pacing-state.json"),
        clock=clock,
        sleep=clock.sleep,
        rng=random.Random(seed),
    )


READ_ONLY = SimpleNamespace(readOnlyHint=True, destructiveHint=None)
DESTRUCTIVE = SimpleNamespace(readOnlyHint=None, destructiveHint=True)


def _next(fn: Any) -> Any:
    """A plain coroutine function standing in for FastMCP's `call_next`."""
    return fn


def _context(
    name: str = "read_tool",
    *,
    annotations: Any = None,
    tags: set[str] | None = None,
    resolvable: bool = True,
    parameters: dict[str, Any] | None = None,
    arguments: dict[str, Any] | None = None,
) -> MiddlewareContext[mt.CallToolRequestParams]:
    fastmcp_context = MagicMock()
    fastmcp_context.request_context = object()
    fastmcp_context.report_progress = AsyncMock()
    tool = SimpleNamespace(
        annotations=annotations, tags=tags or set(), parameters=parameters or {}
    )
    fastmcp_context.fastmcp.get_tool = AsyncMock(
        return_value=tool if resolvable else None
    )
    return MiddlewareContext(
        message=mt.CallToolRequestParams(name=name, arguments=arguments or {}),
        method="tools/call",
        fastmcp_context=fastmcp_context,
    )


def _schema(*names: str) -> dict[str, Any]:
    """A tool parameter schema declaring boolean parameters *names*."""
    return {
        "type": "object",
        "properties": {name: {"type": "boolean"} for name in names},
        "required": list(names),
    }


def _read() -> MiddlewareContext[mt.CallToolRequestParams]:
    return _context("read_tool", annotations=READ_ONLY)


def _write(confirm: Any = True) -> MiddlewareContext[mt.CallToolRequestParams]:
    """A public write; `confirm=False` makes it a preview."""
    return _context(
        "send_message",
        annotations=DESTRUCTIVE,
        tags={"messaging", "write"},
        parameters=_schema("confirm_send"),
        arguments={"confirm_send": confirm},
    )


def _private(confirm: Any = True) -> MiddlewareContext[mt.CallToolRequestParams]:
    """A private write (tagged "private"); `confirm=False` makes it a preview."""
    return _context(
        "save_post",
        annotations=DESTRUCTIVE,
        tags={"post", "write", "private"},
        parameters=_schema("confirm"),
        arguments={"confirm": confirm},
    )


def _takes(clock: FakeClock, seconds: float) -> Any:
    """A call that spends *seconds* on the clock and returns normally."""

    async def call(_context):
        clock.now += seconds

    return _next(call)


async def _never_launched(_context):
    """A call that fails before any request to LinkedIn, as a failed launch does."""
    raise ToolError("Chromium could not start")


class TestClassification:
    def test_destructive_hint_makes_a_write(self):
        assert classify_tool("x", DESTRUCTIVE, set()) is ToolKind.WRITE

    def test_the_write_tag_makes_a_write_without_annotations(self):
        assert classify_tool("x", None, {"posts", "write"}) is ToolKind.WRITE

    def test_everything_else_is_a_read(self):
        assert classify_tool("x", READ_ONLY, {"scraping"}) is ToolKind.READ
        assert classify_tool("x", None, None) is ToolKind.READ

    def test_close_session_is_unpaced_although_destructive(self):
        assert (
            classify_tool("close_session", DESTRUCTIVE, {"session"})
            is ToolKind.BROWSER_ONLY
        )

    def test_the_local_tag_wins_over_everything(self):
        assert classify_tool("x", DESTRUCTIVE, {"local", "write"}) is ToolKind.LOCAL

    async def test_an_unresolvable_tool_is_paced_as_a_write(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(
            tmp_path, clock, _policy(write_min_interval_seconds=10)
        )
        call_next = AsyncMock()

        await middleware.on_call_tool(_context("mystery", resolvable=False), call_next)
        await middleware.on_call_tool(_context("mystery", resolvable=False), call_next)

        assert clock.slept == [10]


class TestSpacing:
    async def test_the_next_call_waits_the_interval_plus_jitter(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(
            tmp_path, clock, _policy(min_interval_seconds=8, jitter_seconds=7)
        )
        call_next = AsyncMock()

        await middleware.on_call_tool(_read(), call_next)
        assert clock.slept == []
        await middleware.on_call_tool(_read(), call_next)

        assert len(clock.slept) == 1
        assert 8 <= clock.slept[0] <= 15
        assert call_next.await_count == 2

    async def test_the_gap_is_never_a_constant(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(
            tmp_path, clock, _policy(min_interval_seconds=8, jitter_seconds=7)
        )
        call_next = AsyncMock()

        for _ in range(5):
            await middleware.on_call_tool(_read(), call_next)

        assert len({round(wait, 3) for wait in clock.slept}) > 1

    async def test_the_gap_counts_from_the_end_of_the_previous_call(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(min_interval_seconds=8))

        async def slow_call(_context):
            clock.now += 5  # the scrape itself takes five seconds

        await middleware.on_call_tool(_read(), _next(slow_call))
        await middleware.on_call_tool(_read(), AsyncMock())

        assert clock.slept == [8]

    async def test_the_wait_is_reported_as_progress(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(min_interval_seconds=8))
        await middleware.on_call_tool(_read(), AsyncMock())

        context = _read()
        await middleware.on_call_tool(context, AsyncMock())

        assert context.fastmcp_context is not None
        messages = [
            call.kwargs["message"]
            for call in cast(
                AsyncMock, context.fastmcp_context.report_progress
            ).await_args_list
        ]
        assert messages == ["Pacing LinkedIn calls: starting in 8s"]

    async def test_a_short_write_gap_is_served_inline(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(
            tmp_path,
            clock,
            _policy(min_interval_seconds=2, write_min_interval_seconds=20),
        )
        call_next = AsyncMock()

        await middleware.on_call_tool(_write(), call_next)
        await middleware.on_call_tool(_read(), call_next)
        await middleware.on_call_tool(_write(), call_next)

        # The read after a write pays only the general gap; the second write
        # waits out what is left of the write gap.
        assert clock.slept == [2, 18]

    async def test_a_long_wait_fails_fast_with_the_time_it_ends(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(
            tmp_path, clock, _policy(write_min_interval_seconds=90)
        )
        call_next = AsyncMock()
        await middleware.on_call_tool(_write(), call_next)

        with pytest.raises(ToolError) as excinfo:
            await middleware.on_call_tool(_write(), call_next)

        message = str(excinfo.value)
        assert "write_min_interval_seconds" in message
        assert "2027-01-15T08:01:30Z" in message  # START + 90s
        assert clock.slept == []
        assert call_next.await_count == 1
        assert MAX_INLINE_WAIT_SECONDS < 90


class TestRollingCaps:
    async def test_reads_per_hour(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(max_reads_per_hour=3))
        call_next = AsyncMock()
        for _ in range(3):
            await middleware.on_call_tool(_read(), call_next)
            clock.now += 60

        with pytest.raises(ToolError) as excinfo:
            await middleware.on_call_tool(_read(), call_next)

        message = str(excinfo.value)
        assert "max_reads_per_hour (3)" in message
        assert "2027-01-15T09:00:00Z" in message  # the first read, plus an hour
        assert clock.slept == []
        assert call_next.await_count == 3

        clock.now = START + HOUR + 1
        await middleware.on_call_tool(_read(), call_next)
        assert call_next.await_count == 4

    async def test_writes_do_not_spend_the_read_budget(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(max_reads_per_hour=1))
        call_next = AsyncMock()

        await middleware.on_call_tool(_write(), call_next)
        await middleware.on_call_tool(_write(), call_next)
        await middleware.on_call_tool(_read(), call_next)

        assert call_next.await_count == 3

    async def test_writes_per_hour(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(max_writes_per_hour=2))
        call_next = AsyncMock()
        await middleware.on_call_tool(_write(), call_next)
        clock.now += 600
        await middleware.on_call_tool(_write(), call_next)

        with pytest.raises(ToolError, match=r"max_writes_per_hour \(2\)"):
            await middleware.on_call_tool(_write(), call_next)
        # Reads are still allowed while writes are capped.
        await middleware.on_call_tool(_read(), call_next)

        clock.now = START + HOUR + 1
        await middleware.on_call_tool(_write(), call_next)
        assert call_next.await_count == 4

    async def test_writes_per_day(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(max_writes_per_day=2))
        call_next = AsyncMock()
        await middleware.on_call_tool(_write(), call_next)
        clock.now += 2 * HOUR
        await middleware.on_call_tool(_write(), call_next)
        clock.now += 2 * HOUR

        with pytest.raises(ToolError) as excinfo:
            await middleware.on_call_tool(_write(), call_next)
        assert "max_writes_per_day (2)" in str(excinfo.value)
        assert "2027-01-16T08:00:00Z" in str(excinfo.value)  # first write + 24h

        clock.now = START + DAY + 1
        await middleware.on_call_tool(_write(), call_next)
        assert call_next.await_count == 3


class TestCooldown:
    @staticmethod
    def _pushed_back(kind: str = pacing_signals.HTTP_429):
        async def call(_context):
            pacing_signals.report(kind, "https://www.linkedin.com/in/someone/")

        return _next(call)

    async def test_a_429_refuses_every_linkedin_call_until_it_ends(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(cooldown_base_seconds=1800))
        await middleware.on_call_tool(_read(), self._pushed_back())

        call_next = AsyncMock()
        for context in (_read(), _write()):
            with pytest.raises(ToolError) as excinfo:
                await middleware.on_call_tool(context, call_next)
            assert "cooldown" in str(excinfo.value)
            assert "http_429" in str(excinfo.value)
            assert "2027-01-15T08:30:00Z" in str(excinfo.value)
        call_next.assert_not_awaited()

        clock.now = START + 1800 + 1
        await middleware.on_call_tool(_read(), call_next)
        call_next.assert_awaited_once()

    async def test_the_cooldown_doubles_when_linkedin_pushes_back_again(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(cooldown_base_seconds=100))
        await middleware.on_call_tool(_read(), self._pushed_back())
        clock.now = START + 101
        await middleware.on_call_tool(
            _read(), self._pushed_back(pacing_signals.SECURITY_CHALLENGE)
        )

        state = middleware.current_state()
        assert state.strikes == 2
        assert state.cooldown_until == START + 101 + 200
        assert state.last_signals == ["security_challenge"]

    async def test_strikes_are_forgotten_after_a_quiet_day(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(cooldown_base_seconds=100))
        await middleware.on_call_tool(_read(), self._pushed_back())
        clock.now = START + DAY + 1
        await middleware.on_call_tool(_read(), self._pushed_back())

        assert middleware.current_state().cooldown_until == START + DAY + 1 + 100

    async def test_a_call_that_failed_still_starts_the_cooldown(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(cooldown_base_seconds=100))

        async def refused(_context):
            pacing_signals.report(pacing_signals.SECURITY_CHALLENGE)
            raise ToolError("Authentication failed. Run with --login")

        with pytest.raises(ToolError, match="Authentication failed"):
            await middleware.on_call_tool(_read(), _next(refused))

        with pytest.raises(ToolError, match="cooldown"):
            await middleware.on_call_tool(_read(), AsyncMock())

    async def test_a_quiet_call_starts_no_cooldown(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(cooldown_base_seconds=100))
        await middleware.on_call_tool(_read(), AsyncMock())
        await middleware.on_call_tool(_read(), AsyncMock())

        assert middleware.current_state().strikes == 0

    async def test_a_login_task_started_inside_a_call_reports_nowhere(self):
        import asyncio

        async def passes_a_checkpoint() -> None:
            pacing_signals.report(pacing_signals.SECURITY_CHALLENGE)

        with pacing_signals.collecting() as signals:
            await asyncio.create_task(
                passes_a_checkpoint(), context=pacing_signals.detached_context()
            )
            assert signals == set()
            # An ordinary helper task spawned by a scrape still reports.
            await asyncio.create_task(passes_a_checkpoint())

        assert signals == {pacing_signals.SECURITY_CHALLENGE}

    def test_a_report_outside_a_tool_call_goes_nowhere(self):
        pacing_signals.report(pacing_signals.HTTP_429)
        with pacing_signals.collecting() as signals:
            pass
        assert signals == set()


class TestPersistence:
    async def test_a_restart_keeps_counters_gaps_and_cooldown(self, tmp_path):
        clock = FakeClock()
        policy = _policy(
            min_interval_seconds=8, max_reads_per_hour=5, cooldown_base_seconds=0
        )
        first = _middleware(tmp_path, clock, policy)
        await first.on_call_tool(_read(), AsyncMock())

        restarted = _middleware(tmp_path, clock, policy)
        assert len(restarted.current_state().reads) == 1
        await restarted.on_call_tool(_read(), AsyncMock())

        assert clock.slept == [8]

    async def test_a_restart_keeps_the_cooldown(self, tmp_path):
        clock = FakeClock()
        policy = _policy(cooldown_base_seconds=600)
        first = _middleware(tmp_path, clock, policy)
        await first.on_call_tool(_read(), TestCooldown._pushed_back())

        restarted = _middleware(tmp_path, clock, policy)
        with pytest.raises(ToolError, match="cooldown"):
            await restarted.on_call_tool(_read(), AsyncMock())

    async def test_a_corrupt_file_starts_fresh_with_a_warning(self, tmp_path, caplog):
        (tmp_path / "pacing-state.json").write_text("{not json", encoding="utf-8")
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(max_reads_per_hour=1))

        with caplog.at_level(logging.WARNING, logger="linkedin_mcp_server.pacing"):
            await middleware.on_call_tool(_read(), AsyncMock())

        assert "Ignoring corrupt pacing state" in caplog.text
        saved = json.loads((tmp_path / "pacing-state.json").read_text("utf-8"))
        assert saved["reads"] == [START]

    async def test_a_wrongly_typed_field_is_corrupt_too(self, tmp_path, caplog):
        (tmp_path / "pacing-state.json").write_text(
            json.dumps({"version": 1, "reads": ["soon"]}), encoding="utf-8"
        )
        middleware = _middleware(tmp_path, FakeClock(), _policy())

        with caplog.at_level(logging.WARNING, logger="linkedin_mcp_server.pacing"):
            state = middleware.current_state()

        assert state.reads == []
        assert "Ignoring corrupt pacing state" in caplog.text

    async def test_an_unwritable_file_does_not_lift_the_limits(self, tmp_path, caplog):
        blocked = tmp_path / "is-a-directory"
        blocked.mkdir()
        clock = FakeClock()
        middleware = PacingMiddleware(
            policy=lambda: _policy(max_reads_per_hour=1),
            store=PacingStateStore(lambda: blocked),
            clock=clock,
            sleep=clock.sleep,
        )
        call_next = AsyncMock()

        with caplog.at_level(logging.WARNING, logger="linkedin_mcp_server.pacing"):
            await middleware.on_call_tool(_read(), call_next)
            with pytest.raises(ToolError, match="max_reads_per_hour"):
                await middleware.on_call_tool(_read(), call_next)

        assert "Could not" in caplog.text
        call_next.assert_awaited_once()

    def test_a_deadline_from_a_clock_that_jumped_back_is_pulled_in(self):
        policy = _policy(min_interval_seconds=8, jitter_seconds=7)
        state = PacingState(next_call_at=START + 10 * DAY, reads=[START + DAY])

        sane = state.sanitized(START, policy)

        assert sane.next_call_at == START + 15
        assert sane.reads == []


class TestExemptAndDisabled:
    async def test_disabled_pacing_neither_waits_nor_writes(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(
            tmp_path,
            clock,
            _policy(enabled=False, min_interval_seconds=8, max_reads_per_hour=1),
        )
        call_next = AsyncMock()

        for _ in range(3):
            await middleware.on_call_tool(_read(), call_next)

        assert clock.slept == []
        assert call_next.await_count == 3
        assert not (tmp_path / "pacing-state.json").exists()

    @pytest.mark.parametrize(
        "context",
        [
            _context("get_pacing_status", tags={"local", "pacing"}),
            _context("close_session", tags={"session"}, annotations=DESTRUCTIVE),
        ],
        ids=["local", "close_session"],
    )
    async def test_exempt_tools_are_not_paced_or_counted(self, tmp_path, context):
        clock = FakeClock()
        middleware = _middleware(
            tmp_path, clock, _policy(min_interval_seconds=8, max_reads_per_hour=1)
        )
        await middleware.on_call_tool(_read(), AsyncMock())
        call_next = AsyncMock()

        await middleware.on_call_tool(context, call_next)
        await middleware.on_call_tool(context, call_next)

        assert clock.slept == []
        assert call_next.await_count == 2
        assert len(middleware.current_state().reads) == 1


class TestStatus:
    async def test_status_reports_counters_and_next_allowed_times(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(
            tmp_path,
            clock,
            _policy(
                min_interval_seconds=8,
                write_min_interval_seconds=90,
                max_writes_per_day=20,
            ),
        )
        await middleware.on_call_tool(_read(), AsyncMock())
        clock.now += 30
        await middleware.on_call_tool(_write(), AsyncMock())

        status = middleware.status()

        assert status["enabled"] is True
        assert status["counters"] == {
            "calls_last_minute": 2,
            "reads_last_hour": 1,
            "writes_last_hour": 1,
            "writes_last_day": 1,
            "private_writes_last_hour": 0,
            "private_writes_last_day": 0,
        }
        assert status["next_read"]["allowed_now"] is False
        assert status["next_read"]["seconds_until_allowed"] == 8
        assert status["next_write"]["held_by"] == "write_min_interval_seconds"
        assert status["next_write"]["seconds_until_allowed"] == 90
        # The private bucket is untouched by a public write: only the general
        # gap holds it.
        assert status["next_private_write"]["held_by"] == "min_interval_seconds"
        assert status["next_private_write"]["seconds_until_allowed"] == 8
        assert status["cooldown"]["active"] is False
        assert status["config"]["max_writes_per_day"] == 20

    async def test_status_reports_every_setting_of_both_buckets(self, tmp_path):
        policy = PacingConfig()
        middleware = _middleware(tmp_path, FakeClock(), policy)

        config = middleware.status()["config"]

        assert config == {
            "min_interval_seconds": policy.min_interval_seconds,
            "jitter_seconds": policy.jitter_seconds,
            "max_calls_per_minute": policy.max_calls_per_minute,
            "max_reads_per_hour": policy.max_reads_per_hour,
            "write_min_interval_seconds": policy.write_min_interval_seconds,
            "write_jitter_seconds": policy.write_jitter_seconds,
            "max_writes_per_hour": policy.max_writes_per_hour,
            "max_writes_per_day": policy.max_writes_per_day,
            "private_write_min_interval_seconds": (
                policy.private_write_min_interval_seconds
            ),
            "private_write_jitter_seconds": policy.private_write_jitter_seconds,
            "max_private_writes_per_hour": policy.max_private_writes_per_hour,
            "max_private_writes_per_day": policy.max_private_writes_per_day,
            "cooldown_base_seconds": policy.cooldown_base_seconds,
            "max_inline_wait_seconds": MAX_INLINE_WAIT_SECONDS,
        }

    async def test_status_changes_nothing(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(max_reads_per_hour=1))

        middleware.status()
        middleware.status()

        assert not (tmp_path / "pacing-state.json").exists()
        await middleware.on_call_tool(_read(), AsyncMock())


class TestPreviews:
    """A write call with its confirmation false changes nothing: it is a read."""

    async def test_a_preview_is_counted_as_a_read_not_a_write(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(max_writes_per_hour=1))
        call_next = AsyncMock()

        for context in (_write(False), _private(False)):
            await middleware.on_call_tool(context, call_next)
            clock.now += 1
        # The one write the hour allows is still there.
        await middleware.on_call_tool(_write(True), call_next)

        state = middleware.current_state()
        assert state.reads == [START, START + 1]
        assert state.writes == [START + 2]
        assert state.private_writes == []
        assert call_next.await_count == 3

    async def test_a_preview_is_capped_as_a_read(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(max_reads_per_hour=1))
        await middleware.on_call_tool(_write(False), AsyncMock())

        with pytest.raises(ToolError, match=r"max_reads_per_hour \(1\)"):
            await middleware.on_call_tool(_private(False), AsyncMock())

    async def test_a_preview_does_not_start_or_wait_out_the_write_gap(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(
            tmp_path, clock, _policy(write_min_interval_seconds=20)
        )
        call_next = AsyncMock()

        await middleware.on_call_tool(_write(False), call_next)
        assert middleware.current_state().next_write_at == 0.0
        await middleware.on_call_tool(_write(True), call_next)
        # The write after the preview did not wait; the preview after the write
        # does not wait for the write gap either.
        await middleware.on_call_tool(_write(False), call_next)

        assert clock.slept == []
        assert middleware.current_state().next_write_at == START + 20

    async def test_a_confirmed_write_is_still_a_write(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(max_writes_per_hour=1))
        await middleware.on_call_tool(_write(True), AsyncMock())

        with pytest.raises(ToolError, match=r"max_writes_per_hour \(1\)"):
            await middleware.on_call_tool(_write(True), AsyncMock())
        assert middleware.current_state().reads == []

    @pytest.mark.parametrize(
        ("arguments", "kind"),
        [
            ({"confirm_send": False}, "read"),
            ({"confirm_send": True}, "write"),
            # send_message's confirmation is confirm_send; a stray confirm is
            # not a preview.
            ({"confirm": False, "confirm_send": True}, "write"),
            ({"confirm": False}, "write"),
            ({}, "write"),
            ({"confirm_send": "false"}, "write"),
            ({"confirm_send": 0}, "write"),
        ],
    )
    async def test_send_message_is_a_preview_only_on_confirm_send_false(
        self, tmp_path, arguments, kind
    ):
        middleware = _middleware(tmp_path, FakeClock(), _policy())
        context = _context(
            "send_message",
            annotations=DESTRUCTIVE,
            tags={"messaging", "write"},
            parameters=_schema("confirm_send"),
            arguments=arguments,
        )

        await middleware.on_call_tool(context, AsyncMock())

        state = middleware.current_state()
        assert (len(state.reads), len(state.writes)) == (
            (1, 0) if kind == "read" else (0, 1)
        )

    async def test_an_unresolvable_tool_is_a_write_whatever_it_confirms(self, tmp_path):
        middleware = _middleware(tmp_path, FakeClock(), _policy())
        context = _context(
            "mystery",
            resolvable=False,
            arguments={"confirm": False, "confirm_send": False},
        )

        await middleware.on_call_tool(context, AsyncMock())

        state = middleware.current_state()
        assert (state.reads, len(state.writes)) == ([], 1)

    async def test_a_tool_declaring_no_confirmation_is_never_a_preview(self, tmp_path):
        middleware = _middleware(tmp_path, FakeClock(), _policy())
        context = _context(
            "new_write",
            annotations=DESTRUCTIVE,
            parameters=_schema("dry_run"),
            arguments={"confirm": False},
        )

        await middleware.on_call_tool(context, AsyncMock())

        assert len(middleware.current_state().writes) == 1


class TestCallsThatNeverReachedLinkedIn:
    async def test_a_call_failing_before_linkedin_spends_no_budget(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(
            tmp_path,
            clock,
            _policy(max_writes_per_hour=1, max_calls_per_minute=1),
        )

        with pytest.raises(ToolError, match="could not start"):
            await middleware.on_call_tool(_write(True), _next(_never_launched))

        state = middleware.current_state()
        assert state.writes == []
        assert state.recent_calls(clock.now) == []
        # The one write the hour allows is still there.
        call_next = AsyncMock()
        await middleware.on_call_tool(_write(True), call_next)
        call_next.assert_awaited_once()

    async def test_a_failed_launch_starts_no_gap(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(
            tmp_path,
            clock,
            _policy(min_interval_seconds=8, write_min_interval_seconds=20),
        )

        with pytest.raises(ToolError):
            await middleware.on_call_tool(_write(True), _next(_never_launched))

        state = middleware.current_state()
        assert state.next_write_at == 0.0
        assert state.next_call_at == 0.0
        await middleware.on_call_tool(_write(True), AsyncMock())
        assert clock.slept == []

    async def test_a_failed_launch_keeps_the_gap_of_the_last_real_call(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(min_interval_seconds=8))
        await middleware.on_call_tool(_read(), AsyncMock())
        clock.now += 8

        with pytest.raises(ToolError):
            await middleware.on_call_tool(_read(), _next(_never_launched))

        assert middleware.current_state().next_call_at == START + 8
        assert middleware.current_state().reads == [START]

    async def test_a_call_that_reached_linkedin_before_failing_counts(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(
            tmp_path,
            clock,
            _policy(max_writes_per_hour=1, write_min_interval_seconds=20),
        )

        async def navigated_then_failed(_context):
            pacing_signals.report_contact()
            raise ToolError("the send button never appeared")

        with pytest.raises(ToolError, match="send button"):
            await middleware.on_call_tool(_write(True), _next(navigated_then_failed))

        state = middleware.current_state()
        assert state.writes == [START]
        assert state.next_write_at == START + 20
        with pytest.raises(ToolError, match=r"max_writes_per_hour \(1\)"):
            await middleware.on_call_tool(_write(True), AsyncMock())

    async def test_a_cancelled_call_that_never_reached_linkedin_is_not_counted(
        self, tmp_path
    ):
        import asyncio

        middleware = _middleware(tmp_path, FakeClock(), _policy())

        async def cancelled(_context):
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await middleware.on_call_tool(_read(), _next(cancelled))

        assert middleware.current_state().reads == []

    async def test_contact_reported_from_a_helper_task_arrives(self):
        import asyncio

        with pacing_signals.tracking_contact() as contact:
            assert contact.reached is False
            await asyncio.create_task(_report_contact())
        assert contact.reached is True

    async def test_a_detached_task_reports_no_contact(self):
        import asyncio

        with pacing_signals.tracking_contact() as contact:
            await asyncio.create_task(
                _report_contact(), context=pacing_signals.detached_context()
            )
        assert contact.reached is False

    async def test_a_push_back_is_contact(self):
        with pacing_signals.tracking_contact() as contact:
            with pacing_signals.collecting():
                pacing_signals.report(pacing_signals.HTTP_429)
        assert contact.reached is True


async def _report_contact() -> None:
    pacing_signals.report_contact()


class TestWhereContactIsReported:
    """The production sites that mark a call as having reached LinkedIn."""

    @staticmethod
    def _patched(**overrides: Any):
        from contextlib import ExitStack

        browser = MagicMock()
        stack = ExitStack()
        mocks = {
            "ensure_tool_ready_or_raise": AsyncMock(),
            "get_or_create_browser": AsyncMock(return_value=browser),
            "ensure_authenticated": AsyncMock(),
        }
        mocks.update(overrides)
        for name, mock in mocks.items():
            stack.enter_context(patch(f"linkedin_mcp_server.dependencies.{name}", mock))
        return stack

    async def test_a_ready_extractor_is_contact(self):
        from linkedin_mcp_server.dependencies import get_ready_extractor

        with self._patched(), pacing_signals.tracking_contact() as contact:
            await get_ready_extractor(None, tool_name="save_post")

        assert contact.reached is True

    async def test_a_browser_that_never_launched_is_no_contact(self):
        from linkedin_mcp_server.dependencies import get_ready_extractor

        failing = AsyncMock(side_effect=RuntimeError("Chromium could not start"))
        with (
            self._patched(get_or_create_browser=failing),
            pacing_signals.tracking_contact() as contact,
        ):
            with pytest.raises(RuntimeError, match="could not start"):
                await get_ready_extractor(None, tool_name="save_post")

        assert contact.reached is False

    async def test_a_session_refused_before_navigating_is_no_contact(self):
        from linkedin_mcp_server.dependencies import get_ready_extractor
        from linkedin_mcp_server.exceptions import AuthenticationStartedError

        refused = AsyncMock(side_effect=AuthenticationStartedError("sign in first"))
        with (
            self._patched(ensure_tool_ready_or_raise=refused),
            pacing_signals.tracking_contact() as contact,
        ):
            with pytest.raises(ToolError):
                await get_ready_extractor(None, tool_name="save_post")

        assert contact.reached is False

    async def test_a_navigation_is_contact_even_when_it_fails(self):
        from linkedin_mcp_server.core.proxy_errors import goto_reporting_proxy_errors

        page = MagicMock()
        page.goto = AsyncMock(side_effect=TimeoutError("navigation timed out"))
        with (
            patch(
                "linkedin_mcp_server.core.proxy_errors._browser_config",
                return_value=SimpleNamespace(proxy_server=None),
            ),
            pacing_signals.tracking_contact() as contact,
        ):
            with pytest.raises(TimeoutError):
                await goto_reporting_proxy_errors(page, "https://www.linkedin.com/")

        assert contact.reached is True


class TestPrivateBucket:
    async def test_private_writes_spend_neither_public_counter_nor_gap(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(
            tmp_path,
            clock,
            _policy(
                max_writes_per_hour=1,
                max_writes_per_day=1,
                write_min_interval_seconds=20,
            ),
        )
        call_next = AsyncMock()

        for _ in range(3):
            await middleware.on_call_tool(_private(), call_next)
            clock.now += 1
        state = middleware.current_state()
        assert state.writes == []
        assert state.next_write_at == 0.0
        assert len(state.private_writes) == 3

        await middleware.on_call_tool(_write(), call_next)
        assert clock.slept == []
        assert call_next.await_count == 4

    async def test_public_writes_spend_nothing_of_the_private_bucket(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(
            tmp_path,
            clock,
            _policy(
                max_private_writes_per_hour=1, private_write_min_interval_seconds=20
            ),
        )
        call_next = AsyncMock()

        for _ in range(3):
            await middleware.on_call_tool(_write(), call_next)
            clock.now += 1
        await middleware.on_call_tool(_private(), call_next)

        assert clock.slept == []
        assert call_next.await_count == 4

    async def test_the_private_hourly_cap(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(
            tmp_path, clock, _policy(max_private_writes_per_hour=2)
        )
        call_next = AsyncMock()
        for _ in range(2):
            await middleware.on_call_tool(_private(), call_next)
            clock.now += 60

        with pytest.raises(ToolError) as excinfo:
            await middleware.on_call_tool(_private(), call_next)
        assert "max_private_writes_per_hour (2)" in str(excinfo.value)
        assert "next private write" in str(excinfo.value)
        # Public writes are still allowed while private writes are capped.
        await middleware.on_call_tool(_write(), call_next)

        clock.now = START + HOUR + 1
        await middleware.on_call_tool(_private(), call_next)
        assert call_next.await_count == 4

    async def test_the_private_daily_cap(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(max_private_writes_per_day=2))
        call_next = AsyncMock()
        for _ in range(2):
            await middleware.on_call_tool(_private(), call_next)
            clock.now += 2 * HOUR

        with pytest.raises(ToolError, match=r"max_private_writes_per_day \(2\)"):
            await middleware.on_call_tool(_private(), call_next)

        clock.now = START + DAY + 1
        await middleware.on_call_tool(_private(), call_next)
        assert call_next.await_count == 3

    async def test_the_private_gap_is_its_own(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(
            tmp_path,
            clock,
            _policy(
                write_min_interval_seconds=25, private_write_min_interval_seconds=10
            ),
        )
        call_next = AsyncMock()

        await middleware.on_call_tool(_private(), call_next)
        await middleware.on_call_tool(_private(), call_next)

        assert clock.slept == [10]

    async def test_private_writes_still_pay_the_general_gap(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(min_interval_seconds=8))

        await middleware.on_call_tool(_read(), AsyncMock())
        await middleware.on_call_tool(_private(), AsyncMock())

        assert clock.slept == [8]

    async def test_a_cooldown_blocks_private_writes(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(cooldown_base_seconds=1800))
        await middleware.on_call_tool(_read(), TestCooldown._pushed_back())
        call_next = AsyncMock()

        with pytest.raises(ToolError, match="cooldown"):
            await middleware.on_call_tool(_private(), call_next)
        call_next.assert_not_awaited()

    def test_the_private_tag_makes_a_private_write(self):
        assert (
            classify_tool("x", DESTRUCTIVE, {"write", "private"})
            is ToolKind.PRIVATE_WRITE
        )
        assert classify_tool("x", None, {"write", "private"}) is ToolKind.PRIVATE_WRITE
        # Without being a write, the tag changes nothing.
        assert classify_tool("x", READ_ONLY, {"private"}) is ToolKind.READ

    async def test_a_file_from_before_the_private_bucket_still_loads(self, tmp_path):
        (tmp_path / "pacing-state.json").write_text(
            json.dumps({"version": 1, "reads": [START], "writes": [START]}),
            encoding="utf-8",
        )
        middleware = _middleware(tmp_path, FakeClock(START + 1), _policy())

        state = middleware.current_state()

        assert (state.reads, state.writes, state.private_writes) == (
            [START],
            [START],
            [],
        )


class TestJitteredWriteGaps:
    @pytest.mark.parametrize(
        ("context", "settings", "deadline"),
        [
            (
                _write,
                {"write_min_interval_seconds": 35, "write_jitter_seconds": 25},
                "next_write_at",
            ),
            (
                _private,
                {
                    "private_write_min_interval_seconds": 10,
                    "private_write_jitter_seconds": 10,
                },
                "next_private_write_at",
            ),
        ],
        ids=["public", "private"],
    )
    async def test_each_gap_is_drawn_afresh_within_its_bounds(
        self, tmp_path, context, settings, deadline
    ):
        base = next(v for k, v in settings.items() if "interval" in k)
        jitter = next(v for k, v in settings.items() if "jitter" in k)
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(**settings), seed=11)

        gaps = []
        for _ in range(8):
            await middleware.on_call_tool(context(), AsyncMock())
            ended = clock.now
            gaps.append(getattr(middleware.current_state(), deadline) - ended)
            clock.now += base + jitter + 1  # past any gap, so none is waited

        assert all(base <= gap <= base + jitter for gap in gaps), gaps
        assert len({round(gap, 3) for gap in gaps}) > 1, gaps


class TestCallsPerMinute:
    async def test_the_call_over_the_cap_waits_for_the_oldest_to_age_out(
        self, tmp_path
    ):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(max_calls_per_minute=5))

        for context in (_read(), _write(), _private(), _write(False), _read()):
            await middleware.on_call_tool(context, _takes(clock, 10))
        assert clock.slept == []

        sixth = _read()
        await middleware.on_call_tool(sixth, AsyncMock())

        # Calls started at 0, 10, 20, 30 and 40s; the sixth, at 50s, waits for
        # the first to leave the window.
        assert clock.slept == [10]
        assert sixth.fastmcp_context is not None
        progress = cast(AsyncMock, sixth.fastmcp_context.report_progress)
        assert progress.await_args is not None
        assert progress.await_args.kwargs["message"] == (
            "Pacing LinkedIn calls: starting in 10s"
        )

    async def test_a_slot_beyond_the_inline_limit_fails_fast_and_rolls(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(max_calls_per_minute=5))
        call_next = AsyncMock()
        for _ in range(5):
            await middleware.on_call_tool(_read(), call_next)
            clock.now += 1

        with pytest.raises(ToolError) as excinfo:
            await middleware.on_call_tool(_write(), call_next)

        message = str(excinfo.value)
        assert "max_calls_per_minute (5)" in message
        assert "2027-01-15T08:01:00Z" in message  # the first call, plus a minute
        assert clock.slept == []
        assert call_next.await_count == 5

        clock.now = START + 60
        await middleware.on_call_tool(_write(), call_next)
        assert call_next.await_count == 6

    async def test_exempt_tools_do_not_count(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(max_calls_per_minute=2))
        exempt = (
            _context("get_pacing_status", tags={"local", "pacing"}),
            _context("close_session", tags={"session"}, annotations=DESTRUCTIVE),
        )
        for context in exempt * 3:
            await middleware.on_call_tool(context, AsyncMock())
            clock.now += 1

        for _ in range(2):
            await middleware.on_call_tool(_read(), AsyncMock())
            clock.now += 1
        with pytest.raises(ToolError, match=r"max_calls_per_minute \(2\)"):
            await middleware.on_call_tool(_read(), AsyncMock())

    async def test_calls_that_never_reached_linkedin_do_not_count(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(max_calls_per_minute=2))
        for _ in range(3):
            with pytest.raises(ToolError):
                await middleware.on_call_tool(_read(), _next(_never_launched))
            clock.now += 1

        for _ in range(2):
            await middleware.on_call_tool(_read(), AsyncMock())
            clock.now += 1
        with pytest.raises(ToolError, match=r"max_calls_per_minute \(2\)"):
            await middleware.on_call_tool(_read(), AsyncMock())

    @pytest.mark.parametrize(
        ("min_interval", "slept", "held_by"),
        [
            # The cap is stricter: the fourth call waits for the first to age out.
            (15, [15, 15, 30], "max_calls_per_minute"),
            # The gap is stricter: the cap never adds a wait of its own.
            (25, [25, 25, 25], "min_interval_seconds"),
        ],
        ids=["cap-wins", "gap-wins"],
    )
    async def test_the_stricter_of_the_cap_and_the_gap_wins(
        self, tmp_path, min_interval, slept, held_by
    ):
        clock = FakeClock()
        middleware = _middleware(
            tmp_path,
            clock,
            _policy(max_calls_per_minute=3, min_interval_seconds=min_interval),
        )
        for _ in range(3):
            await middleware.on_call_tool(_read(), AsyncMock())

        assert middleware.status()["next_read"]["held_by"] == held_by
        await middleware.on_call_tool(_read(), AsyncMock())

        assert clock.slept == slept

    async def test_status_counts_calls_in_the_last_minute(self, tmp_path):
        clock = FakeClock()
        middleware = _middleware(tmp_path, clock, _policy(max_calls_per_minute=20))
        for context in (_read(), _write(), _private()):
            await middleware.on_call_tool(context, _takes(clock, 25))

        status = middleware.status()

        # Started at 0, 25 and 50s; at 75s the first is out of the window.
        assert status["counters"]["calls_last_minute"] == 2
        assert status["config"]["max_calls_per_minute"] == 20

    async def test_the_counters_survive_a_restart(self, tmp_path):
        clock = FakeClock()
        policy = _policy(max_calls_per_minute=1)
        await _middleware(tmp_path, clock, policy).on_call_tool(_read(), AsyncMock())
        clock.now += 1

        restarted = _middleware(tmp_path, clock, policy)
        with pytest.raises(ToolError, match=r"max_calls_per_minute \(1\)"):
            await restarted.on_call_tool(_private(), AsyncMock())


class TestServerWiring:
    def test_pacing_sits_inside_the_serialization(self):
        from linkedin_mcp_server.pacing import PacingMiddleware as Pacing
        from linkedin_mcp_server.sequential_tool_middleware import (
            SequentialToolExecutionMiddleware,
        )
        from linkedin_mcp_server.server import create_mcp_server

        kinds = [type(m) for m in create_mcp_server().middleware]

        assert kinds.index(SequentialToolExecutionMiddleware) < kinds.index(Pacing)

    async def test_the_status_tool_never_waits_for_the_browser(self, tmp_path):
        from linkedin_mcp_server.server import create_mcp_server

        mcp = create_mcp_server()
        tool = await mcp.get_tool("get_pacing_status")
        assert tool is not None
        assert "local" in tool.tags

        with patch(
            "linkedin_mcp_server.sequential_tool_middleware.get_profile_lease",
            side_effect=AssertionError("the status tool took the browser lease"),
        ):
            result = await mcp.call_tool("get_pacing_status", {})

        assert result.structured_content is not None
        assert result.structured_content["enabled"] is True
        assert "next_write" in result.structured_content

    async def test_serialization_still_holds_for_a_linkedin_tool(self):
        from fastmcp import FastMCP

        from linkedin_mcp_server.sequential_tool_middleware import (
            SequentialToolExecutionMiddleware,
        )

        mcp = FastMCP("test")
        mcp.add_middleware(SequentialToolExecutionMiddleware())

        @mcp.tool(annotations={"readOnlyHint": True})
        async def reads_linkedin() -> dict[str, bool]:
            return {"ok": True}

        with patch(
            "linkedin_mcp_server.sequential_tool_middleware.get_profile_lease",
            side_effect=RuntimeError("lease requested"),
        ):
            with pytest.raises(Exception, match="lease requested"):
                await mcp.call_tool("reads_linkedin", {})
