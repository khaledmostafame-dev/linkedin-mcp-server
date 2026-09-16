"""Browser-free parsing and validation of a LinkedIn post request.

Everything here runs before a browser session is acquired, so an invalid
request is refused without spending a navigation, and a ``confirm=False``
preview never touches LinkedIn at all.

Mentions use an inline syntax that names the identity explicitly::

    @[Display Name](https://www.linkedin.com/in/slug/)
    @[Company Name](https://www.linkedin.com/company/slug/)
    @[Display Name](urn:li:fsd_profile:ACoAA...)
    @[Company Name](urn:li:organization:12345)

The display name is only what gets typed into LinkedIn's typeahead. Which
option is selected is decided by the URL or URN, never by the name, so a
namesake can never be mentioned in place of the intended member.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
import re
from typing import Any, Literal
from urllib.parse import unquote, urlparse

LINKEDIN_POST_CHARACTER_LIMIT = 3000
# LinkedIn documents scheduling "up to three months in advance". Ninety days
# is the conservative reading of that window; the lower bound keeps a request
# from racing the clock while the composer is being driven.
SCHEDULE_MIN_LEAD = timedelta(minutes=10)
SCHEDULE_MAX_AHEAD = timedelta(days=90)

Visibility = Literal["anyone", "connections"]
VISIBILITIES: tuple[str, ...] = ("anyone", "connections")

MentionKind = Literal["person", "company"]

_MENTION_RE = re.compile(r"@\[([^\[\]\r\n]{1,100})\]\(([^()\s]{1,512})\)")
_LINKEDIN_HOST_RE = re.compile(r"^(?:[a-z0-9-]+\.)*linkedin\.com$")
_SLUG_RE = re.compile(r"^[^/?#\s]{1,200}$")
_URN_RE = re.compile(r"^urn:li:([a-z_]+):([A-Za-z0-9_-]{1,100})$")
_PERSON_URN_KINDS = {"fsd_profile": "profile", "member": "member", "person": "member"}
_COMPANY_URN_KINDS = {"company", "fsd_company", "organization"}


class PostValidationError(ValueError):
    """A post request that cannot be sent as given."""


@dataclass(frozen=True)
class MentionTarget:
    """The identity a mention must resolve to."""

    kind: MentionKind
    # ``person:/in/slug/``, ``company:/company/slug/``, ``profile:ACoAA..``,
    # ``member:123`` or ``company:urn:123``. Two targets are the same identity
    # exactly when their keys are equal.
    key: str
    reference: str


@dataclass(frozen=True)
class TextSegment:
    text: str


@dataclass(frozen=True)
class MentionSegment:
    name: str
    target: MentionTarget


Segment = TextSegment | MentionSegment


@dataclass(frozen=True)
class PostAttachment:
    """A local file the composer uploads, already validated by type and size."""

    kind: Literal["image", "document"]
    path: str
    filename: str
    content_type: str
    size_bytes: int
    source: str


@dataclass(frozen=True)
class PostRequest:
    """A validated post, ready for the composer."""

    segments: tuple[Segment, ...]
    visibility: Visibility
    schedule_at: datetime | None = None
    images: tuple[PostAttachment, ...] = ()
    document: PostAttachment | None = None
    document_title: str | None = None
    post_as: MentionTarget | None = None

    @property
    def rendered_text(self) -> str:
        return render_segments(self.segments)

    @property
    def mentions(self) -> tuple[MentionSegment, ...]:
        return tuple(s for s in self.segments if isinstance(s, MentionSegment))


def utf16_length(text: str) -> int:
    """Length as a browser counts it, which is what a JS counter measures."""
    return len(text.encode("utf-16-le")) // 2


def normalize_text(text: str) -> str:
    """Normalize line endings and reject control characters other than LF."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for character in text:
        code = ord(character)
        if (code < 32 and character != "\n") or code == 127:
            raise PostValidationError(
                "Post text must not contain control characters other than line breaks."
            )
    return text


def identity_key_from_url(value: str) -> MentionTarget | None:
    """Return the identity a LinkedIn member or company URL names."""
    if not isinstance(value, str) or re.search(r"[\\\x00-\x1f\x7f]", value):
        return None
    try:
        parsed = urlparse(value.strip())
    except ValueError:
        return None
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme and parsed.scheme != "https":
        return None
    if parsed.scheme and (
        not _LINKEDIN_HOST_RE.match(host)
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        return None
    if not parsed.scheme and parsed.netloc:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2 or parts[0] not in {"in", "company"}:
        return None
    slug = unquote(parts[1]).strip()
    if not _SLUG_RE.match(slug):
        return None
    slug = slug.casefold()
    if parts[0] == "in":
        return MentionTarget("person", f"person:/in/{slug}/", value)
    if slug.isdigit():
        # /company/12345/ names the same page as urn:li:organization:12345.
        return MentionTarget("company", f"company:urn:{slug}", value)
    return MentionTarget("company", f"company:/company/{slug}/", value)


def identity_key_from_urn(value: str) -> MentionTarget | None:
    """Return the identity a LinkedIn member or company URN names."""
    match = _URN_RE.match(value.strip()) if isinstance(value, str) else None
    if match is None:
        return None
    namespace, identifier = match.groups()
    if namespace in _PERSON_URN_KINDS:
        return MentionTarget(
            "person", f"{_PERSON_URN_KINDS[namespace]}:{identifier}", value
        )
    if namespace in _COMPANY_URN_KINDS and identifier.isdigit():
        return MentionTarget("company", f"company:urn:{identifier}", value)
    return None


def parse_mention_target(reference: str) -> MentionTarget:
    target = (
        identity_key_from_urn(reference)
        if reference.startswith("urn:")
        else identity_key_from_url(reference)
    )
    if target is None or not reference.startswith(("urn:", "https://")):
        raise PostValidationError(
            f"Mention target {reference!r} is not a LinkedIn member or company "
            "URL (https://www.linkedin.com/in/<slug>/, "
            "https://www.linkedin.com/company/<slug>/) or URN "
            "(urn:li:fsd_profile:<id>, urn:li:organization:<id>)."
        )
    return target


def parse_post_text(text: str) -> tuple[Segment, ...]:
    """Split post text into plain text and explicitly targeted mentions."""
    text = normalize_text(text)
    segments: list[Segment] = []
    position = 0
    for match in _MENTION_RE.finditer(text):
        if match.start() > position:
            segments.append(TextSegment(text[position : match.start()]))
        name = " ".join(match.group(1).split())
        if not name:
            raise PostValidationError("A mention needs a non-blank display name.")
        segments.append(MentionSegment(name, parse_mention_target(match.group(2))))
        position = match.end()
    if position < len(text):
        segments.append(TextSegment(text[position:]))
    leftover = "".join(s.text for s in segments if isinstance(s, TextSegment))
    if "@[" in leftover and re.search(r"@\[[^\]]*\]\(", leftover):
        # Looks like a mention but did not parse: typing it verbatim would
        # post a plain "@Name" instead of a mention, which is the one outcome
        # the syntax exists to prevent.
        raise PostValidationError(
            "Text contains a malformed mention. Use @[Display Name](URL or URN)."
        )
    return tuple(segments)


def render_segments(segments: tuple[Segment, ...] | list[Segment]) -> str:
    return "".join(
        segment.text if isinstance(segment, TextSegment) else segment.name
        for segment in segments
    )


def parse_schedule_at(value: str, *, now: datetime | None = None) -> datetime:
    """Parse an ISO 8601 instant with an explicit offset into UTC."""
    now = now or datetime.now(UTC)
    try:
        parsed = datetime.fromisoformat(value.strip())
    except (TypeError, ValueError) as error:
        raise PostValidationError(
            f"schedule_at {value!r} is not an ISO 8601 date-time "
            "(e.g. 2026-09-20T09:00:00+04:00)."
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PostValidationError(
            "schedule_at needs an explicit UTC offset or Z; a naive time would be "
            "read in whatever timezone the server happens to run in."
        )
    instant = parsed.astimezone(UTC).replace(second=0, microsecond=0)
    if parsed.second or parsed.microsecond:
        raise PostValidationError(
            "schedule_at must be a whole minute; LinkedIn schedules to the minute."
        )
    if instant < now + SCHEDULE_MIN_LEAD:
        raise PostValidationError(
            f"schedule_at must be at least {int(SCHEDULE_MIN_LEAD.total_seconds() // 60)} "
            "minutes in the future."
        )
    if instant > now + SCHEDULE_MAX_AHEAD:
        raise PostValidationError(
            f"schedule_at is beyond LinkedIn's scheduling window "
            f"({SCHEDULE_MAX_AHEAD.days} days ahead)."
        )
    return instant


def build_post_request(
    text: str,
    *,
    visibility: str = "anyone",
    schedule_at: str | None = None,
    images: tuple[PostAttachment, ...] = (),
    document: PostAttachment | None = None,
    document_title: str | None = None,
    now: datetime | None = None,
    attachments_pending: bool = False,
    post_as: str | None = None,
) -> PostRequest:
    """Validate everything about a post that can be checked without a browser."""
    if visibility not in VISIBILITIES:
        raise PostValidationError(
            f"visibility must be one of {', '.join(VISIBILITIES)}."
        )
    actor = parse_post_as(post_as) if post_as is not None else None
    if actor is not None and visibility != "anyone":
        raise PostValidationError(
            'A company page post is public; visibility must be "anyone".'
        )
    segments = parse_post_text(text)
    rendered = render_segments(segments)
    has_attachments = attachments_pending or bool(images) or document is not None
    if not rendered.strip() and not has_attachments:
        raise PostValidationError("A post needs text or an attachment.")
    if utf16_length(rendered) > LINKEDIN_POST_CHARACTER_LIMIT:
        raise PostValidationError(
            f"Post text is {utf16_length(rendered)} characters; LinkedIn allows "
            f"{LINKEDIN_POST_CHARACTER_LIMIT}."
        )
    if images and document is not None:
        raise PostValidationError(
            "LinkedIn posts carry either images or one document, not both."
        )
    if (
        document is not None
        and not attachments_pending
        and not (document_title or "").strip()
    ):
        raise PostValidationError("A document post needs a non-blank title.")
    instant = parse_schedule_at(schedule_at, now=now) if schedule_at else None
    return PostRequest(
        segments=segments,
        visibility=visibility,  # ty: ignore[invalid-argument-type]
        schedule_at=instant,
        images=images,
        document=document,
        document_title=" ".join((document_title or "").split()) or None,
        post_as=actor,
    )


def parse_post_as(value: str) -> MentionTarget:
    """A company page to post as: its URL, numeric id or organization URN."""
    reference = value.strip() if isinstance(value, str) else ""
    if reference.isdigit():
        return MentionTarget("company", f"company:urn:{reference}", reference)
    try:
        target = parse_mention_target(reference)
    except PostValidationError:
        target = None
    if target is None or target.kind != "company":
        raise PostValidationError(
            "post_as must name a company page: https://www.linkedin.com/company/"
            "<slug>/, its numeric id, or urn:li:organization:<id>."
        )
    return target


@dataclass(frozen=True)
class PostEdit:
    """A validated edit: new text, a new schedule, or both."""

    segments: tuple[Segment, ...] | None
    schedule_at: datetime | None = None

    @property
    def rendered_text(self) -> str | None:
        return render_segments(self.segments) if self.segments is not None else None

    @property
    def mentions(self) -> tuple[MentionSegment, ...]:
        return tuple(s for s in self.segments or () if isinstance(s, MentionSegment))


def build_post_edit(
    text: str | None,
    *,
    schedule_at: str | None = None,
    allow_schedule: bool = True,
    now: datetime | None = None,
) -> PostEdit:
    """Validate an edit to an existing post without a browser."""
    if text is None and schedule_at is None:
        raise PostValidationError("Pass text, schedule_at, or both.")
    if schedule_at is not None and not allow_schedule:
        raise PostValidationError("A published post cannot be rescheduled.")
    segments = None
    if text is not None:
        segments = parse_post_text(text)
        rendered = render_segments(segments)
        if not rendered.strip():
            raise PostValidationError("Edited text must not be blank.")
        if utf16_length(rendered) > LINKEDIN_POST_CHARACTER_LIMIT:
            raise PostValidationError(
                f"Post text is {utf16_length(rendered)} characters; LinkedIn "
                f"allows {LINKEDIN_POST_CHARACTER_LIMIT}."
            )
    instant = parse_schedule_at(schedule_at, now=now) if schedule_at else None
    return PostEdit(segments=segments, schedule_at=instant)


_POST_URN_RE = re.compile(r"^urn:li:(activity|share|ugcPost):(\d{5,25})$")
_ACTIVITY_SLUG_RE = re.compile(r"-activity-(\d{5,25})-")


def parse_post_url(value: str) -> str:
    """Normalize a LinkedIn post permalink to its /feed/update/<urn>/ form."""
    try:
        parsed = urlparse(value.strip()) if isinstance(value, str) else None
    except ValueError:
        parsed = None
    host = (parsed.hostname or "").lower().rstrip(".") if parsed else ""
    if (
        parsed is None
        or parsed.scheme != "https"
        or not _LINKEDIN_HOST_RE.match(host)
        or parsed.username
        or parsed.password
    ):
        raise PostValidationError(
            "post_url must be an https://www.linkedin.com post permalink."
        )
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    urn = None
    if len(parts) >= 3 and parts[:2] == ["feed", "update"]:
        urn = parts[2]
    elif len(parts) >= 2 and parts[0] == "posts":
        match = _ACTIVITY_SLUG_RE.search(parts[1] + "-")
        urn = f"urn:li:activity:{match.group(1)}" if match else None
    if urn is None or not _POST_URN_RE.match(urn):
        raise PostValidationError(
            "post_url must be a post permalink such as "
            "https://www.linkedin.com/feed/update/urn:li:activity:<id>/."
        )
    return f"https://www.linkedin.com/feed/update/{urn}/"


def post_preview(request: PostRequest) -> dict[str, Any]:
    """Describe what a confirmed call would do, without contacting LinkedIn."""
    rendered = request.rendered_text
    action = "publish" if request.schedule_at is None else "schedule"
    preview: dict[str, Any] = {
        "url": "https://www.linkedin.com/feed/",
        "status": "preview",
        "message": (
            f"Nothing was sent to LinkedIn. Call again with confirm=true to {action}. "
            "Mentions were parsed, not yet resolved against LinkedIn's typeahead."
        ),
        "text": rendered,
        "characters": utf16_length(rendered),
        "character_limit": LINKEDIN_POST_CHARACTER_LIMIT,
        "visibility": request.visibility,
        "mentions": [
            {
                "name": mention.name,
                "kind": mention.target.kind,
                "target": mention.target.reference,
                "resolved_against_linkedin": False,
            }
            for mention in request.mentions
        ],
        "media": [attachment_summary(image) for image in request.images],
        "document": (
            {**attachment_summary(request.document), "title": request.document_title}
            if request.document is not None
            else None
        ),
        "schedule": schedule_summary(request.schedule_at),
        "post_as": (
            {
                "kind": "company",
                "target": request.post_as.reference,
                "resolved_against_linkedin": False,
            }
            if request.post_as is not None
            else None
        ),
    }
    return preview


def attachment_summary(attachment: PostAttachment) -> dict[str, Any]:
    return {
        "kind": attachment.kind,
        "filename": attachment.filename,
        "content_type": attachment.content_type,
        "size_bytes": attachment.size_bytes,
        "source": attachment.source,
    }


def schedule_summary(
    instant: datetime | None,
    *,
    browser_timezone: str | None = None,
    browser_local: str | None = None,
) -> dict[str, Any] | None:
    if instant is None:
        return None
    return {
        "utc": instant.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "browser_timezone": browser_timezone,
        "browser_local": browser_local,
    }


# --- Schedule dialog value formatting ---------------------------------------
#
# Ported from upstream PRs 690 and 696 (schedule_post / edit_scheduled_post).
# LinkedIn's schedule dialog prefills its date and time inputs in the profile's
# own locale, so the value typed back is formatted by copying that prefill
# rather than by assuming an English layout.


def resolve_date_order(
    current_value: str, placeholder: str, today: date | None
) -> tuple[str, str, str] | None:
    """Determine the date input's slot order, e.g. ``("m", "d", "y")``.

    Independent measurements, tried in turn; None when none decides.

    1. A small number above 12 cannot be a month, so it names the day slot.
    2. The prefill renders LinkedIn's own "today", at most one calendar day
       from ``today`` in any timezone. Two distinct readings of the same
       numbers are never both inside a three-day window, so a reading inside
       it proves the order. Withheld when ``today`` is None.
    3. The placeholder's short tokens. Across the Latin-script locales
       LinkedIn ships, the month token starts with ``m`` (en ``mm``, de
       ``MM``, fr ``mm``, es ``mes``, it ``mese``, pt ``mês``, nl ``maand``)
       while the day letter varies. A per-locale table, documented as such;
       other scripts fall through to None and the caller refuses.
    """
    numbers = re.findall(r"\d+", current_value)
    if len(numbers) == 3:
        values = [int(n) for n in numbers]
        year_pos = max(range(3), key=lambda i: (len(numbers[i]) >= 4, values[i]))
        rest = [i for i in range(3) if i != year_pos]
        a, b = values[rest[0]], values[rest[1]]
        year = values[year_pos]

        def order_of(day_first: bool) -> tuple[str, str, str]:
            slots = ["", "", ""]
            slots[year_pos] = "y"
            slots[rest[0]] = "d" if day_first else "m"
            slots[rest[1]] = "m" if day_first else "d"
            return (slots[0], slots[1], slots[2])

        if a > 12 >= b:
            return order_of(day_first=True)
        if b > 12 >= a:
            return order_of(day_first=False)
        if today is not None:
            window = {today + timedelta(days=k) for k in (-1, 0, 1)}

            def in_window(month: int, day: int) -> bool:
                try:
                    return date(year, month, day) in window
                except ValueError:
                    return False

            month_first = in_window(a, b)
            day_first = in_window(b, a)
            if month_first != day_first:
                return order_of(day_first)

    tokens = re.findall(r"[^\W\d_]+", placeholder or "")
    if len(tokens) == 3:
        parts = [
            "y"
            if len(t) > 2 and not t.lower().startswith("m")
            else ("m" if t.lower().startswith("m") else "d")
            for t in tokens
        ]
        if sorted(parts) == ["d", "m", "y"]:
            return (parts[0], parts[1], parts[2])
    return None


def format_schedule_date(
    year: int,
    month: int,
    day: int,
    current_value: str,
    placeholder: str = "",
    today: date | None = None,
) -> tuple[str | None, tuple[str, str, str] | None]:
    """Format a date the way the schedule dialog's own prefill formats one.

    Returns ``(value, order)``, or ``(None, None)`` when the order cannot be
    proven and day and month differ: a reversed date passes every later check
    LinkedIn offers, because both readings are valid dates.
    """
    separators = re.findall(r"\D+", current_value.strip())
    separator = separators[0] if separators else "/"
    order = resolve_date_order(current_value, placeholder, today)
    if order is None:
        if day != month:
            return None, None
        order = ("m", "d", "y")
    mapping = {"y": str(year), "m": str(month), "d": str(day)}
    return separator.join(mapping[slot] for slot in order), order


def date_matches(
    value: str, order: tuple[str, str, str], y: int, m: int, d: int
) -> bool:
    """Position-by-position check; an unordered one would pass a reversal."""
    expected = [{"y": y, "m": m, "d": d}[slot] for slot in order]
    return [int(n) for n in re.findall(r"\d+", value)] == expected


def format_schedule_time(hour: int, minute: int, current_value: str) -> str:
    """Format a time the way the timepicker's own prefill formats one.

    A meridiem marker in the prefill means the locale renders 12-hour time;
    the en markers are the only ones tabled here and 24-hour ``HH:MM`` is the
    fallback. A non-English 12-hour locale may reject that; the caller reads
    the input back and refuses rather than scheduling at a wrong time.
    """
    if re.search(r"\b[AP]\.?M\.?\b", current_value, re.IGNORECASE):
        marker = "AM" if hour < 12 else "PM"
        return f"{hour % 12 or 12}:{minute:02d} {marker}"
    return f"{hour:02d}:{minute:02d}"


def time_matches(value: str, hour: int, minute: int) -> bool:
    numbers = [int(n) for n in re.findall(r"\d+", value)]
    if len(numbers) != 2:
        return False
    shown_hour, shown_minute = numbers
    if re.search(r"\b[AP]\.?M\.?\b", value, re.IGNORECASE):
        pm = re.search(r"\bP\.?M\.?\b", value, re.IGNORECASE) is not None
        shown_hour = (shown_hour % 12) + (12 if pm else 0)
    return (shown_hour, shown_minute) == (hour, minute)
