"""Account-level pacing for tool calls that reach LinkedIn.

LinkedIn restricts an *account*, not a client. Several MCP clients share one
server and one logged-in browser, so the only place a limit can hold for all of
them is the server, inside the serialization that already makes calls take
turns. Three mechanisms, each answering a different failure:

* **Spacing.** A gap after every LinkedIn call, longer after a write, each with
  random jitter drawn per call so the cadence is never a constant (upstream
  #860/#877, #732/#734). A per-minute cap across every kind of call works the
  same way. Short waits are served inline with progress; a wait longer than
  :data:`MAX_INLINE_WAIT_SECONDS` fails fast with the time it ends instead.
* **Rolling caps.** Reads per hour; public writes per hour and per day; private
  writes per hour and per day. Exceeding one fails fast naming the limit and
  when it frees up. Never a silent sleep: a client waiting minutes on a call
  reads as a hung server and retries.
* **Cooldown.** When a call sees HTTP 429 or a checkpoint/challenge/authwall
  redirect (``pacing_signals``), every LinkedIn call is refused until the
  cooldown ends. It doubles per repeat within a day (upstream #957 backs off
  inside one call; a checkpoint needs longer than a call).

Tool classification is shared with every tool module: a tool is a **write** if
its annotations say ``destructiveHint`` or its tags include ``"write"``. A write
also tagged ``"private"`` changes only what this account sees (saving a post or
a job, marking a conversation read, archiving one) and spends a **private**
bucket with its own gap and caps, never the public one. A write *call* whose
confirmation argument (``confirm``, or ``confirm_send`` for ``send_message``) is
``false`` is a preview that changes nothing, so it is paced and counted as a
**read**; it stays paced because some previews open a page to report the
current state. Tools tagged ``"local"`` never touch the browser or LinkedIn and
bypass both pacing and serialization; ``close_session`` drives the browser but
not LinkedIn, so it is serialized and unpaced. Everything else is a **read**. An
unresolvable tool is treated as a public write, the safe direction.

A call is counted when it starts. One that raises before it reached LinkedIn
(``pacing_signals.report_contact``: a browser that never launched, a session
refused before any navigation) is taken back out and leaves the gap where it
was. A call that completes always counts.

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
#: Tag for a write only this account can see; it spends the private bucket.
PRIVATE_TAG = "private"
#: The parameter a write tool takes its explicit confirmation in. Exactly one
#: per write tool (``tests/test_write_tool_safety.py``).
CONFIRM_PARAMETERS = ("confirm", "confirm_send")
#: Drive the browser but never LinkedIn: serialized, not paced.
_BROWSER_ONLY_TOOLS = frozenset({"close_session"})

MINUTE = 60.0
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
    PRIVATE_WRITE = "private_write"

    @property
    def noun(self) -> str:
        """The kind as a message names it (``private write``)."""
        return self.value.replace("_", " ")


_WRITES = frozenset({ToolKind.WRITE, ToolKind.PRIVATE_WRITE})
_UNPACED = frozenset({ToolKind.LOCAL, ToolKind.BROWSER_ONLY})


def classify_tool(name: str, annotations: Any, tags: Iterable[str] | None) -> ToolKind:
    """Classify a tool from its declared metadata, per the module contract."""
    tag_set = set(tags or ())
    if LOCAL_TAG in tag_set or name == PACING_STATUS_TOOL:
        return ToolKind.LOCAL
    if name in _BROWSER_ONLY_TOOLS:
        return ToolKind.BROWSER_ONLY
    if WRITE_TAG in tag_set or getattr(annotations, "destructiveHint", None) is True:
        return ToolKind.PRIVATE_WRITE if PRIVATE_TAG in tag_set else ToolKind.WRITE
    return ToolKind.READ


def is_preview(parameters: Any, arguments: Any) -> bool:
    """Whether a write call only previews: its confirmation argument is ``false``.

    The confirmation parameter is whichever of :data:`CONFIRM_PARAMETERS` the
    tool declares. Anything less certain than exactly one declared and a JSON
    ``false`` passed for it (a missing argument, a string, a tool declaring
    neither or both) is not a preview, so the call stays a write.
    """
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
    if not isinstance(properties, dict) or not isinstance(arguments, dict):
        return False
    declared = [name for name in CONFIRM_PARAMETERS if name in properties]
    return len(declared) == 1 and arguments.get(declared[0]) is False


async def classify_call(
    context: MiddlewareContext[mt.CallToolRequestParams],
) -> ToolKind:
    """Classify the call from the tool's registered metadata and its arguments.

    Read from the registered tool rather than a list of names kept here, so a
    tool added later is classified by what it declares. A tool that cannot be
    resolved counts as a public write, the direction that cannot cost the
    account, whatever its arguments say.
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
    kind = classify_tool(
        name, getattr(tool, "annotations", None), getattr(tool, "tags", None)
    )
    if kind in _WRITES and is_preview(
        getattr(tool, "parameters", None), getattr(context.message, "arguments", None)
    ):
        return ToolKind.READ
    return kind


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
    """Everything pacing remembers, in wall-clock seconds since the epoch.

    ``writes`` and ``next_write_at`` are the public bucket. Every counted call
    is in exactly one of the three stamp lists, which is what the per-minute
    cap counts across.
    """

    reads: list[float] = field(default_factory=list)
    writes: list[float] = field(default_factory=list)
    private_writes: list[float] = field(default_factory=list)
    next_call_at: float = 0.0
    next_write_at: float = 0.0
    next_private_write_at: float = 0.0
    cooldown_until: float = 0.0
    strikes: int = 0
    last_strike_at: float = 0.0
    last_signals: list[str] = field(default_factory=list)

    def stamps(self, kind: ToolKind) -> list[float]:
        """The rolling-window list a call of *kind* is counted in."""
        if kind is ToolKind.READ:
            return self.reads
        if kind is ToolKind.PRIVATE_WRITE:
            return self.private_writes
        return self.writes

    def recent_calls(self, now: float) -> list[float]:
        """Every counted call of any kind in the last minute, sorted."""
        return sorted(
            t
            for t in (*self.reads, *self.writes, *self.private_writes)
            if t > now - MINUTE
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "version": _STATE_VERSION,
            "reads": self.reads,
            "writes": self.writes,
            "private_writes": self.private_writes,
            "next_call_at": self.next_call_at,
            "next_write_at": self.next_write_at,
            "next_private_write_at": self.next_private_write_at,
            "cooldown_until": self.cooldown_until,
            "strikes": self.strikes,
            "last_strike_at": self.last_strike_at,
            "last_signals": self.last_signals,
        }

    @classmethod
    def from_json(cls, data: object) -> PacingState:
        """Parse persisted state, raising ``ValueError`` on any malformed field.

        A field missing from the file reads as empty, so a file written before
        the private bucket existed still loads with its counters intact.
        """
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
            private_writes=numbers("private_writes"),
            next_call_at=number("next_call_at"),
            next_write_at=number("next_write_at"),
            next_private_write_at=number("next_private_write_at"),
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
            private_writes=sorted(set(self.private_writes) | set(other.private_writes)),
            next_call_at=max(self.next_call_at, other.next_call_at),
            next_write_at=max(self.next_write_at, other.next_write_at),
            next_private_write_at=max(
                self.next_private_write_at, other.next_private_write_at
            ),
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
        write_gap = policy.write_min_interval_seconds + policy.write_jitter_seconds
        private_gap = (
            policy.private_write_min_interval_seconds
            + policy.private_write_jitter_seconds
        )
        return PacingState(
            reads=[t for t in self.reads if now - HOUR < t <= slack],
            writes=[t for t in self.writes if now - DAY < t <= slack],
            private_writes=[t for t in self.private_writes if now - DAY < t <= slack],
            next_call_at=min(self.next_call_at, now + gap),
            next_write_at=min(self.next_write_at, now + write_gap),
            next_private_write_at=min(self.next_private_write_at, now + private_gap),
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


def _over_cap(
    stamps: list[float], now: float, window: float, cap: int, name: str
) -> tuple[float, str] | None:
    """When a rolling cap frees up and its label, or ``None`` while under it."""
    if not cap:
        return None
    inside = sorted(t for t in stamps if t > now - window)
    if len(inside) < cap:
        return None
    return _cap_release(inside, cap, window), f"{name} ({cap})"


def _holding(reason: str, policy: PacingConfig) -> str:
    """What holds a call back, as a refusal names it."""
    if reason == "max_calls_per_minute":
        return f"max_calls_per_minute ({policy.max_calls_per_minute}) is reached"
    return f"{reason} has not elapsed"


def assess(
    state: PacingState, kind: ToolKind, now: float, policy: PacingConfig
) -> Verdict:
    """Decide when a call of *kind* may start. Pure, so the status tool agrees.

    A refusal (cooldown or cap) is reported as the earliest moment the limit
    frees up; waits for spacing and the per-minute cap carry no refusal and may
    be served inline.
    """
    if kind in _UNPACED or not policy.enabled:
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

    if kind is ToolKind.READ:
        caps = [(state.reads, HOUR, policy.max_reads_per_hour, "max_reads_per_hour")]
    elif kind is ToolKind.PRIVATE_WRITE:
        caps = [
            (
                state.private_writes,
                HOUR,
                policy.max_private_writes_per_hour,
                "max_private_writes_per_hour",
            ),
            (
                state.private_writes,
                DAY,
                policy.max_private_writes_per_day,
                "max_private_writes_per_day",
            ),
        ]
    else:
        caps = [
            (state.writes, HOUR, policy.max_writes_per_hour, "max_writes_per_hour"),
            (state.writes, DAY, policy.max_writes_per_day, "max_writes_per_day"),
        ]
    refusals = [
        refusal
        for stamps, window, cap, name in caps
        if (refusal := _over_cap(stamps, now, window, cap, name)) is not None
    ]
    if refusals:
        release, limit = max(refusals)
        return Verdict(
            release,
            limit,
            refusal=(
                f"LinkedIn pacing limit reached: {limit}. The next {kind.noun} is "
                f"allowed at {_iso(release)} (in {_duration(release - now)}). "
                "Calls are refused rather than queued; retry after that time."
            ),
        )

    holds = [(state.next_call_at, "min_interval_seconds")]
    if kind is ToolKind.WRITE:
        holds.append((state.next_write_at, "write_min_interval_seconds"))
    elif kind is ToolKind.PRIVATE_WRITE:
        holds.append(
            (state.next_private_write_at, "private_write_min_interval_seconds")
        )
    recent = state.recent_calls(now)
    if policy.max_calls_per_minute and len(recent) >= policy.max_calls_per_minute:
        holds.append(
            (
                _cap_release(recent, policy.max_calls_per_minute, MINUTE),
                "max_calls_per_minute",
            )
        )
    # The stricter hold wins; on a tie, the earlier entry names it.
    earliest, reason = holds[0]
    for when, why in holds[1:]:
        if when > earliest:
            earliest, reason = when, why
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
        if not policy.enabled or kind in _UNPACED:
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
                f"LinkedIn pacing: {_holding(verdict.reason, policy)}. The next "
                f"{kind.noun} is allowed at {_iso(verdict.earliest)} (in "
                f"{_duration(wait)}); retry after that time."
            )
        if wait > 0:
            logger.debug("Pacing %s for %.1fs (%s)", tool_name, wait, verdict.reason)
            await self._report_progress(
                context,
                f"Pacing LinkedIn calls: starting in {math.ceil(wait)}s",
            )
            await self._sleep(wait)

        # Counted at the start, so a status read or a peer process mid-call
        # sees the slot as spent; taken back out below if the call fails before
        # it reaches LinkedIn.
        started = self._clock()
        state = self.current_state()
        stamps = state.stamps(kind)
        counted = started not in stamps
        if counted:
            stamps.append(started)
        next_call_before = state.next_call_at
        # Held at the full gap for the duration of the call, so a status read or
        # a peer process mid-call never sees the slot as free.
        state.next_call_at = max(state.next_call_at, started + DAY)
        self._commit(state)

        signals: set[str] = set()
        contact = pacing_signals.Contact()
        completed = False
        try:
            with (
                pacing_signals.collecting() as signals,
                pacing_signals.tracking_contact() as contact,
            ):
                result = await call_next(context)
            completed = True
            return result
        finally:
            if completed or contact.reached or signals:
                self._finish(kind, signals, policy)
            else:
                self._take_back(
                    kind, started if counted else None, next_call_before, tool_name
                )

    def _take_back(
        self,
        kind: ToolKind,
        stamp: float | None,
        next_call_at: float,
        tool_name: str,
    ) -> None:
        """Undo the start of a call that raised before it reached LinkedIn.

        LinkedIn saw nothing, so the budget is returned and the gap still runs
        from the last call it did see; no write gap is started.
        """
        state = self.current_state()
        stamps = state.stamps(kind)
        if stamp is not None and stamp in stamps:
            stamps.remove(stamp)
        state.next_call_at = next_call_at
        self._commit(state)
        logger.info("Not counting %s: it failed before reaching LinkedIn", tool_name)

    def _finish(self, kind: ToolKind, signals: set[str], policy: PacingConfig) -> None:
        ended = self._clock()
        state = self.current_state()
        # The placeholder set at the start is replaced, not maxed against.
        state.next_call_at = (
            ended + policy.min_interval_seconds + self._jitter(policy.jitter_seconds)
        )
        if kind is ToolKind.WRITE:
            state.next_write_at = max(
                state.next_write_at,
                ended
                + policy.write_min_interval_seconds
                + self._jitter(policy.write_jitter_seconds),
            )
        elif kind is ToolKind.PRIVATE_WRITE:
            state.next_private_write_at = max(
                state.next_private_write_at,
                ended
                + policy.private_write_min_interval_seconds
                + self._jitter(policy.private_write_jitter_seconds),
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

    def _jitter(self, upper: float) -> float:
        """A fresh draw from [0, *upper*] for every gap, never a fixed offset."""
        if upper <= 0:
            return 0.0
        return self._rng.uniform(0.0, upper)

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

        def last_hour(stamps: list[float]) -> int:
            return sum(1 for t in stamps if t > now - HOUR)

        return {
            "enabled": policy.enabled,
            "now": _iso(now),
            "counters": {
                "calls_last_minute": len(state.recent_calls(now)),
                "reads_last_hour": len(state.reads),
                "writes_last_hour": last_hour(state.writes),
                "writes_last_day": len(state.writes),
                "private_writes_last_hour": last_hour(state.private_writes),
                "private_writes_last_day": len(state.private_writes),
            },
            "next_read": when(ToolKind.READ),
            "next_write": when(ToolKind.WRITE),
            "next_private_write": when(ToolKind.PRIVATE_WRITE),
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
            },
        }
