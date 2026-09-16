"""Account-level pacing for tool calls that reach LinkedIn.

LinkedIn restricts an *account*, not a client. Several MCP clients share one
server and one logged-in browser, so the only place a limit can hold for all of
them is the server, inside the serialization that already makes calls take
turns. Three mechanisms, each answering a different failure:

* **Spacing.** A gap after every LinkedIn call, longer after a write, with random
  jitter so the cadence is never a constant (upstream #860/#877, #732/#734).
  Short waits are served inline with progress; a wait longer than
  :data:`MAX_INLINE_WAIT_SECONDS` fails fast with the time it ends instead.
* **Rolling caps.** Reads per hour, writes per hour and per day. Exceeding one
  fails fast naming the limit and when it frees up. Never a silent sleep: a
  client waiting minutes on a call reads as a hung server and retries.
* **Cooldown.** When a call sees HTTP 429 or a checkpoint/challenge/authwall
  redirect (``pacing_signals``), every LinkedIn call is refused until the
  cooldown ends. It doubles per repeat within a day (upstream #957 backs off
  inside one call; a checkpoint needs longer than a call).

Tool classification is shared with every tool module: a tool is a **write** if
its annotations say ``destructiveHint`` or its tags include ``"write"``; tools
tagged ``"local"`` never touch the browser or LinkedIn and bypass both pacing and
serialization; ``close_session`` drives the browser but not LinkedIn, so it is
serialized and unpaced. Everything else is a **read**. An unresolvable tool is
treated as a write, the safe direction.

State (counters, next allowed times, cooldown) persists as
``pacing-state.json`` in the auth root, so a container restart does not reset
the budget. A missing or corrupt file starts fresh with a warning; the in-memory
state keeps the limits in force even when the file cannot be written.
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import math
import random
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import mcp.types as mt
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools import ToolResult

from linkedin_mcp_server import pacing_signals
from linkedin_mcp_server.common_utils import secure_write_text
from linkedin_mcp_server.config import get_config
from linkedin_mcp_server.config.schema import PacingConfig

logger = logging.getLogger(__name__)

STATE_FILE = "pacing-state.json"
_STATE_VERSION = 1

PACING_STATUS_TOOL = "get_pacing_status"
#: Tag for a tool that never touches the browser or LinkedIn.
LOCAL_TAG = "local"
#: Tag that makes a tool a write even without ``destructiveHint``.
WRITE_TAG = "write"
#: Drive the browser but never LinkedIn: serialized, not paced.
_BROWSER_ONLY_TOOLS = frozenset({"close_session"})

HOUR = 3600.0
DAY = 24 * HOUR

#: Longest wait served inside a call. The wait happens before the tool runs and
#: outside its own timeout, so it spends the *client's* patience; the smallest
#: client timeouts sit near 60s (``config/schema.py``), and this leaves the tool
#: itself the rest. Deliberately not configurable: past this, telling the client
#: when to come back is the better answer at any setting.
MAX_INLINE_WAIT_SECONDS = 30.0
#: Longest cooldown, however many times LinkedIn pushed back in a day.
MAX_COOLDOWN_SECONDS = DAY
#: A strike older than this no longer escalates the next cooldown.
STRIKE_MEMORY_SECONDS = DAY
#: How far in the future a persisted timestamp may sit before it is treated as
#: a clock that jumped, rather than a limit.
_CLOCK_SLACK_SECONDS = 60.0


class ToolKind(enum.Enum):
    """How a tool call is paced."""

    LOCAL = "local"
    BROWSER_ONLY = "browser_only"
    READ = "read"
    WRITE = "write"


def classify_tool(name: str, annotations: Any, tags: Iterable[str] | None) -> ToolKind:
    """Classify a tool from its declared metadata, per the module contract."""
    tag_set = set(tags or ())
    if LOCAL_TAG in tag_set or name == PACING_STATUS_TOOL:
        return ToolKind.LOCAL
    if name in _BROWSER_ONLY_TOOLS:
        return ToolKind.BROWSER_ONLY
    if WRITE_TAG in tag_set or getattr(annotations, "destructiveHint", None) is True:
        return ToolKind.WRITE
    return ToolKind.READ


async def classify_call(
    context: MiddlewareContext[mt.CallToolRequestParams],
) -> ToolKind:
    """Classify the tool a call names, reading its metadata from the server.

    Read from the registered tool rather than a list of names kept here, so a
    tool added later is classified by what it declares. A tool that cannot be
    resolved counts as a write, the direction that cannot cost the account.
    """
    name = getattr(context.message, "name", "") or ""
    tool = None
    fastmcp_context = context.fastmcp_context
    if fastmcp_context is not None:
        try:
            tool = await fastmcp_context.fastmcp.get_tool(name)
        except Exception:  # noqa: BLE001 - unresolvable is not a reason to skip pacing
            logger.debug("Could not read the metadata of tool %s", name, exc_info=True)
    if tool is None:
        if name == PACING_STATUS_TOOL or name in _BROWSER_ONLY_TOOLS:
            return classify_tool(name, None, None)
        return ToolKind.WRITE
    return classify_tool(
        name, getattr(tool, "annotations", None), getattr(tool, "tags", None)
    )


def _iso(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _duration(seconds: float) -> str:
    seconds = max(0, math.ceil(seconds))
    if seconds < 120:
        return f"{seconds}s"
    minutes = math.ceil(seconds / 60)
    if minutes < 120:
        return f"{minutes} min"
    return f"{minutes / 60:.1f} h"


@dataclass
class PacingState:
    """Everything pacing remembers, in wall-clock seconds since the epoch."""

    reads: list[float] = field(default_factory=list)
    writes: list[float] = field(default_factory=list)
    next_call_at: float = 0.0
    next_write_at: float = 0.0
    cooldown_until: float = 0.0
    strikes: int = 0
    last_strike_at: float = 0.0
    last_signals: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "version": _STATE_VERSION,
            "reads": self.reads,
            "writes": self.writes,
            "next_call_at": self.next_call_at,
            "next_write_at": self.next_write_at,
            "cooldown_until": self.cooldown_until,
            "strikes": self.strikes,
            "last_strike_at": self.last_strike_at,
            "last_signals": self.last_signals,
        }

    @classmethod
    def from_json(cls, data: object) -> PacingState:
        """Parse persisted state, raising ``ValueError`` on any malformed field."""
        if not isinstance(data, dict):
            raise ValueError("pacing state is not an object")
        payload = cast(dict[str, Any], data)
        if payload.get("version") != _STATE_VERSION:
            raise ValueError("unknown pacing state version")

        def number(name: str) -> float:
            value = payload.get(name, 0.0)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} is not a number")
            return float(value)

        def numbers(name: str) -> list[float]:
            values = payload.get(name, [])
            if not isinstance(values, list):
                raise ValueError(f"{name} is not a list")
            out = []
            for value in values:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(f"{name} holds a non-number")
                out.append(float(value))
            return out

        strikes = payload.get("strikes", 0)
        if isinstance(strikes, bool) or not isinstance(strikes, int) or strikes < 0:
            raise ValueError("strikes is not a count")
        last_signals = payload.get("last_signals", [])
        if not isinstance(last_signals, list) or not all(
            isinstance(item, str) for item in last_signals
        ):
            raise ValueError("last_signals is not a list of names")
        return cls(
            reads=numbers("reads"),
            writes=numbers("writes"),
            next_call_at=number("next_call_at"),
            next_write_at=number("next_write_at"),
            cooldown_until=number("cooldown_until"),
            strikes=strikes,
            last_strike_at=number("last_strike_at"),
            last_signals=[str(item) for item in last_signals],
        )

    def merged(self, other: PacingState) -> PacingState:
        """The stricter of two states, for combining disk with memory.

        Conservative in every field, so neither a stale file nor a lost write
        can loosen a limit: timestamps are unioned, deadlines take the later.
        """
        if other.last_strike_at > self.last_strike_at:
            last_signals = other.last_signals
        else:
            last_signals = self.last_signals
        return PacingState(
            reads=sorted(set(self.reads) | set(other.reads)),
            writes=sorted(set(self.writes) | set(other.writes)),
            next_call_at=max(self.next_call_at, other.next_call_at),
            next_write_at=max(self.next_write_at, other.next_write_at),
            cooldown_until=max(self.cooldown_until, other.cooldown_until),
            strikes=max(self.strikes, other.strikes),
            last_strike_at=max(self.last_strike_at, other.last_strike_at),
            last_signals=list(last_signals),
        )

    def sanitized(self, now: float, policy: PacingConfig) -> PacingState:
        """Drop what has aged out, and what a clock jump put in the future.

        Only the budget windows are pruned. A deadline further ahead than any
        setting could have produced says the wall clock went backwards, and is
        pulled in to the longest honest value rather than trusted.
        """
        slack = now + _CLOCK_SLACK_SECONDS
        gap = policy.min_interval_seconds + policy.jitter_seconds
        write_gap = policy.write_min_interval_seconds + policy.jitter_seconds
        return PacingState(
            reads=[t for t in self.reads if now - HOUR < t <= slack],
            writes=[t for t in self.writes if now - DAY < t <= slack],
            next_call_at=min(self.next_call_at, now + gap),
            next_write_at=min(self.next_write_at, now + write_gap),
            cooldown_until=min(self.cooldown_until, now + MAX_COOLDOWN_SECONDS),
            strikes=self.strikes,
            last_strike_at=min(self.last_strike_at, slack),
            last_signals=list(self.last_signals),
        )


@dataclass(frozen=True)
class Verdict:
    """When a call of one kind may start, and what holds it back."""

    earliest: float
    reason: str
    refusal: str | None = None

    def wait_from(self, now: float) -> float:
        return max(0.0, self.earliest - now)


def _cap_release(stamps: list[float], cap: int, window: float) -> float:
    """When the count inside *window* drops below *cap* (stamps sorted)."""
    return stamps[len(stamps) - cap] + window


def assess(
    state: PacingState, kind: ToolKind, now: float, policy: PacingConfig
) -> Verdict:
    """Decide when a call of *kind* may start. Pure, so the status tool agrees.

    A refusal (cooldown or cap) is reported as the earliest moment the limit
    frees up; waits for spacing alone carry no refusal and may be served inline.
    """
    if kind in (ToolKind.LOCAL, ToolKind.BROWSER_ONLY) or not policy.enabled:
        return Verdict(now, "not paced")

    if state.cooldown_until > now:
        signals = ", ".join(state.last_signals) or "a LinkedIn challenge"
        return Verdict(
            state.cooldown_until,
            "cooldown",
            refusal=(
                f"LinkedIn pacing cooldown: LinkedIn pushed back ({signals}), so "
                f"calls to LinkedIn are refused until {_iso(state.cooldown_until)} "
                f"(in {_duration(state.cooldown_until - now)}). Retrying earlier "
                "extends nothing but the risk to the account. If LinkedIn showed "
                "a checkpoint, resolve it in a normal browser session first."
            ),
        )

    refusals: list[tuple[float, str]] = []
    if kind is ToolKind.READ:
        reads = sorted(state.reads)
        if policy.max_reads_per_hour and len(reads) >= policy.max_reads_per_hour:
            release = _cap_release(reads, policy.max_reads_per_hour, HOUR)
            refusals.append(
                (release, f"max_reads_per_hour ({policy.max_reads_per_hour})")
            )
    else:
        writes = sorted(state.writes)
        last_hour = [t for t in writes if t > now - HOUR]
        if policy.max_writes_per_hour and len(last_hour) >= policy.max_writes_per_hour:
            release = _cap_release(last_hour, policy.max_writes_per_hour, HOUR)
            refusals.append(
                (release, f"max_writes_per_hour ({policy.max_writes_per_hour})")
            )
        if policy.max_writes_per_day and len(writes) >= policy.max_writes_per_day:
            release = _cap_release(writes, policy.max_writes_per_day, DAY)
            refusals.append(
                (release, f"max_writes_per_day ({policy.max_writes_per_day})")
            )
    if refusals:
        release, limit = max(refusals)
        noun = "read" if kind is ToolKind.READ else "write"
        return Verdict(
            release,
            limit,
            refusal=(
                f"LinkedIn pacing limit reached: {limit}. The next {noun} is "
                f"allowed at {_iso(release)} (in {_duration(release - now)}). "
                "Calls are refused rather than queued; retry after that time."
            ),
        )

    earliest, reason = state.next_call_at, "min_interval_seconds"
    if kind is ToolKind.WRITE and state.next_write_at > earliest:
        earliest, reason = state.next_write_at, "write_min_interval_seconds"
    if earliest <= now:
        return Verdict(now, "ready")
    return Verdict(earliest, reason)


class PacingStateStore:
    """``pacing-state.json``: fail open for the file, never for the limits."""

    def __init__(self, path: Callable[[], Path]) -> None:
        self._path = path

    def path(self) -> Path:
        return self._path()

    def load(self) -> PacingState | None:
        """The persisted state, or ``None`` when missing or unreadable."""
        try:
            path = self._path()
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            logger.warning("Could not read pacing state, starting fresh: %s", exc)
            return None
        try:
            return PacingState.from_json(json.loads(raw))
        except ValueError as exc:
            logger.warning(
                "Ignoring corrupt pacing state at %s, starting fresh: %s", path, exc
            )
            return None

    def save(self, state: PacingState) -> None:
        try:
            secure_write_text(
                self._path(), json.dumps(state.to_json(), sort_keys=True) + "\n"
            )
        except OSError as exc:
            logger.warning(
                "Could not persist pacing state; limits stay in force for this "
                "process only: %s",
                exc,
            )


def _default_state_path() -> Path:
    # Imported here: session_state pulls in the config singleton at import.
    from linkedin_mcp_server.session_state import auth_root_dir

    return auth_root_dir() / STATE_FILE


class PacingMiddleware(Middleware):
    """Space, budget and cool down tool calls that reach LinkedIn.

    Must be installed *inside* ``SequentialToolExecutionMiddleware`` (added after
    it): the decision reads, waits on and writes shared state, and only the
    serialization around it makes that atomic across concurrent clients, in
    this process through its lock and across processes through the profile
    lease. Outside it, two clients could both be admitted into one slot.
    """

    def __init__(
        self,
        *,
        policy: Callable[[], PacingConfig] | None = None,
        store: PacingStateStore | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._policy = policy or (lambda: get_config().pacing)
        self._store = store or PacingStateStore(_default_state_path)
        self._clock = clock
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._memory = PacingState()

    def current_state(self) -> PacingState:
        """Disk merged with memory, pruned to what still counts."""
        now = self._clock()
        disk = self._store.load()
        state = self._memory if disk is None else self._memory.merged(disk)
        return state.sanitized(now, self._policy())

    def _commit(self, state: PacingState) -> None:
        self._memory = state
        self._store.save(state)

    async def _report_progress(
        self, context: MiddlewareContext[mt.CallToolRequestParams], message: str
    ) -> None:
        fastmcp_context = context.fastmcp_context
        if fastmcp_context is None or fastmcp_context.request_context is None:
            return
        try:
            await fastmcp_context.report_progress(
                progress=0, total=100, message=message
            )
        except Exception:  # noqa: BLE001 - progress is a courtesy
            logger.debug("Could not report pacing progress", exc_info=True)

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        policy = self._policy()
        kind = await classify_call(context)
        if not policy.enabled or kind in (ToolKind.LOCAL, ToolKind.BROWSER_ONLY):
            return await call_next(context)

        tool_name = context.message.name
        state = self.current_state()
        verdict = assess(state, kind, self._clock(), policy)
        if verdict.refusal is not None:
            logger.info("Refused %s: %s", tool_name, verdict.reason)
            raise ToolError(verdict.refusal)
        wait = verdict.wait_from(self._clock())
        if wait > MAX_INLINE_WAIT_SECONDS:
            logger.info("Refused %s: %s wait of %.0fs", tool_name, verdict.reason, wait)
            raise ToolError(
                f"LinkedIn pacing: {verdict.reason} has not elapsed. The next "
                f"{kind.value} is allowed at {_iso(verdict.earliest)} (in "
                f"{_duration(wait)}); retry after that time."
            )
        if wait > 0:
            logger.debug("Pacing %s for %.1fs (%s)", tool_name, wait, verdict.reason)
            await self._report_progress(
                context,
                f"Pacing LinkedIn calls: starting in {math.ceil(wait)}s",
            )
            await self._sleep(wait)

        # Counted at the start: a call that fails still reached LinkedIn.
        started = self._clock()
        state = self.current_state()
        if kind is ToolKind.READ:
            state.reads.append(started)
        else:
            state.writes.append(started)
        # Held at the full gap for the duration of the call, so a status read or
        # a peer process mid-call never sees the slot as free.
        state.next_call_at = max(state.next_call_at, started + DAY)
        self._commit(state)

        signals: set[str] = set()
        try:
            with pacing_signals.collecting() as signals:
                return await call_next(context)
        finally:
            self._finish(kind, signals, policy)

    def _finish(self, kind: ToolKind, signals: set[str], policy: PacingConfig) -> None:
        ended = self._clock()
        state = self.current_state()
        # The placeholder set at the start is replaced, not maxed against.
        state.next_call_at = ended + policy.min_interval_seconds + self._jitter(policy)
        if kind is ToolKind.WRITE:
            state.next_write_at = max(
                state.next_write_at,
                ended + policy.write_min_interval_seconds + self._jitter(policy),
            )
        if signals and policy.cooldown_base_seconds > 0:
            if ended - state.last_strike_at > STRIKE_MEMORY_SECONDS:
                state.strikes = 0
            state.strikes += 1
            cooldown = min(
                MAX_COOLDOWN_SECONDS,
                policy.cooldown_base_seconds * 2 ** min(state.strikes - 1, 16),
            )
            state.cooldown_until = max(state.cooldown_until, ended + cooldown)
            state.last_strike_at = ended
            state.last_signals = sorted(signals)
            logger.warning(
                "LinkedIn pushed back (%s); pacing cooldown %s (strike %d)",
                ", ".join(sorted(signals)),
                _duration(cooldown),
                state.strikes,
            )
        self._commit(state)

    def _jitter(self, policy: PacingConfig) -> float:
        if policy.jitter_seconds <= 0:
            return 0.0
        return self._rng.uniform(0.0, policy.jitter_seconds)

    def status(self) -> dict[str, Any]:
        """What ``get_pacing_status`` reports. Reads state; changes nothing."""
        policy = self._policy()
        now = self._clock()
        state = self.current_state()

        def when(kind: ToolKind) -> dict[str, Any]:
            verdict = assess(state, kind, now, policy)
            return {
                "allowed_now": verdict.earliest <= now,
                "allowed_at": _iso(verdict.earliest),
                "seconds_until_allowed": round(verdict.wait_from(now), 1),
                "held_by": None if verdict.earliest <= now else verdict.reason,
            }

        writes_last_hour = sum(1 for t in state.writes if t > now - HOUR)
        return {
            "enabled": policy.enabled,
            "now": _iso(now),
            "counters": {
                "reads_last_hour": len(state.reads),
                "writes_last_hour": writes_last_hour,
                "writes_last_day": len(state.writes),
            },
            "next_read": when(ToolKind.READ),
            "next_write": when(ToolKind.WRITE),
            "cooldown": {
                "active": policy.enabled and state.cooldown_until > now,
                "until": _iso(state.cooldown_until)
                if state.cooldown_until > now
                else None,
                "strikes": state.strikes,
                "last_strike_at": _iso(state.last_strike_at)
                if state.last_strike_at
                else None,
                "last_signals": list(state.last_signals),
            },
            "config": {
                "min_interval_seconds": policy.min_interval_seconds,
                "jitter_seconds": policy.jitter_seconds,
                "write_min_interval_seconds": policy.write_min_interval_seconds,
                "max_reads_per_hour": policy.max_reads_per_hour,
                "max_writes_per_hour": policy.max_writes_per_hour,
                "max_writes_per_day": policy.max_writes_per_day,
                "cooldown_base_seconds": policy.cooldown_base_seconds,
                "max_inline_wait_seconds": MAX_INLINE_WAIT_SECONDS,
            },
        }
