"""Where the scraper reports that LinkedIn is pushing back.

Kept apart from ``pacing.py`` and free of imports from the rest of the package,
because the reporters live at the bottom of the stack (``core/auth.py``,
``core/utils.py``, ``scraping/navigation.py``) and the pacing middleware sits at
the top of it. The detectors decide *that* LinkedIn pushed back, from URL
patterns and status codes rather than page text, and this module only carries
the verdict upward to the tool call it happened in.

Recorded rather than inferred from the exception a tool raises, because the
exception does not reliably arrive: an auth barrier goes through
``handle_auth_error``, which replaces it with a relogin error, and a section
scraper may fold a failure into ``section_errors`` and return normally.

It also carries whether a call reached LinkedIn at all (:func:`report_contact`),
so pacing can leave a call that failed before its first request out of the
budget: a browser that never launched has spent nothing LinkedIn could count.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import Context, ContextVar, copy_context
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

#: LinkedIn answered a navigation with HTTP 429.
HTTP_429 = "http_429"
#: LinkedIn redirected to a checkpoint, challenge or authwall route.
SECURITY_CHALLENGE = "security_challenge"

#: Routes LinkedIn sends a session to when it wants proof of a person, or wants
#: the visitor gone. ``/login`` is deliberately absent: an expired session lands
#: there too, and a routine logout is not a reason to stop for half an hour.
_CHALLENGE_ROUTES = (
    "/checkpoint",
    "/challenge",
    "/authwall",
    "/uas/consumer-email-challenge",
)

# One set per tool call, installed by the pacing middleware. A mutable holder
# rather than a value set by the reporter: a reporter running in a task the tool
# spawned sees a *copy* of the context, so a `.set()` there would never reach
# the middleware, while a mutation of the shared set does.
_signals: ContextVar[set[str] | None] = ContextVar(
    "linkedin_pacing_signals", default=None
)


class Contact:
    """Whether the current tool call has reached LinkedIn yet.

    A mutable holder for the same reason as the signal set: a helper task sees a
    copy of the context, and only a mutation of the shared object gets back.
    """

    __slots__ = ("reached",)

    def __init__(self) -> None:
        self.reached = False


_contact: ContextVar[Contact | None] = ContextVar(
    "linkedin_pacing_contact", default=None
)


def is_challenge_url(url: str) -> bool:
    """Whether *url* is one of LinkedIn's checkpoint, challenge or authwall routes.

    Matched on whole path segments, so a profile slug that merely contains the
    word is not a checkpoint.
    """
    try:
        path = urlparse(url).path or "/"
    except ValueError:
        return False
    return any(
        path == route or path.startswith(f"{route}/") for route in _CHALLENGE_ROUTES
    )


def report(kind: str, url: str = "") -> None:
    """Record that LinkedIn pushed back during the current tool call.

    Outside a paced tool call (a login run, a startup check) there is nobody to
    tell, and this does nothing but log.
    """
    try:
        # Path only: a checkpoint URL carries session-bound tokens in its query.
        where = urlparse(url).path if url else ""
    except ValueError:
        where = ""
    signals = _signals.get()
    if signals is None:
        logger.info("LinkedIn pushed back outside a tool call: %s %s", kind, where)
        return
    if kind not in signals:
        logger.warning("LinkedIn pushed back during a tool call: %s %s", kind, where)
    signals.add(kind)
    # LinkedIn answered, so the call reached it, whatever else was reported.
    report_contact()


def report_contact() -> None:
    """Record that the current tool call may now reach LinkedIn.

    Called before each navigation in the shared helpers and when a tool is
    handed a ready browser, since from then on it may act on a page it did not
    navigate to itself. Pacing treats a call that *raised* without this ever
    being called as one LinkedIn never saw, and leaves it out of the budget.
    Reporting too early only costs budget; a path that reaches LinkedIn without
    reporting would under-count, so when in doubt, report.
    """
    contact = _contact.get()
    if contact is not None:
        contact.reached = True


def report_if_challenge(url: str) -> None:
    """Report *url* when it is a checkpoint, challenge or authwall route."""
    if is_challenge_url(url):
        report(SECURITY_CHALLENGE, url)


def detached_context() -> Context:
    """The current context without the collector, for a background task.

    A task copies the context it was created in, collector included, so a login
    browser started from inside a tool call would report the checkpoint a person
    passes through while signing in, and put the account into a cooldown for
    completing a normal 2FA step. Pass this as ``create_task(..., context=)``
    for work that outlives the call or belongs to someone at a keyboard.
    """
    context = copy_context()
    context.run(_signals.set, None)
    context.run(_contact.set, None)
    return context


@contextmanager
def collecting() -> Iterator[set[str]]:
    """Collect every report made inside the block into the yielded set."""
    signals: set[str] = set()
    token = _signals.set(signals)
    try:
        yield signals
    finally:
        _signals.reset(token)


@contextmanager
def tracking_contact() -> Iterator[Contact]:
    """Track, in the yielded holder, whether the block reached LinkedIn."""
    contact = Contact()
    token = _contact.set(contact)
    try:
        yield contact
    finally:
        _contact.reset(token)
