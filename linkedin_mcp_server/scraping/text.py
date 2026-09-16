"""Locale-guarded cleanup of LinkedIn innerText captures."""

from __future__ import annotations

from dataclasses import dataclass

import re

# Digit scripts LinkedIn's Arabic UI renders counts and pagination state in,
# mapped to their ASCII equivalents. `str.maketrans` needs a dict of
# ``ord(char) -> replacement``, hence the ``ord()`` keys rather than the
# characters themselves.
#
# - Arabic-Indic (٠-٩, U+0660-0669): the digits LinkedIn's own Arabic locale
#   renders numbers in (confirmed by the accepted-limitation test this table
#   replaces, ``tests/test_job_pagination_dom.py``'s prior
#   ``test_non_ascii_numerals_degrade_to_no_count``).
# - Extended Arabic-Indic / Persian-Urdu (۰-۹, U+06F0-06F9): a different
#   digit shape some Arabic-script locales (Persian, Urdu, and some Gulf
#   keyboards) use instead of U+0660. Included defensively; not verified
#   against a live LinkedIn session in either of those locales.
# - Arabic thousands separator ٬ (U+066C) and decimal separator ٫ (U+066B):
#   render in follower/connection counts formatted with grouping, e.g.
#   "١٬٢٣٤" for "1,234". Rewritten to their ASCII equivalents so a later
#   ``int(text.replace(",", ""))``-style call sees ordinary punctuation.
_DIGIT_TRANSLATION = str.maketrans(
    {
        **{0x0660 + i: str(i) for i in range(10)},
        **{0x06F0 + i: str(i) for i in range(10)},
        0x066C: ",",
        0x066B: ".",
    }
)

# Bidi control characters LinkedIn's RTL rendering wraps around numbers and
# mixed-direction text: LRM (U+200E), RLM (U+200F), Arabic Letter Mark
# (U+061C). Invisible in a UI, but they land in `innerText`/`textContent`
# and break an exact-match or `isdigit()` check that does not expect them.
_BIDI_MARKS_RE = re.compile("[‎‏؜]")


def normalize_localized_digits(text: str) -> str:
    """Rewrite non-Latin decimal digits and their separators to ASCII.

    Strips bidi control marks and rewrites Arabic-Indic and Extended
    Arabic-Indic digits, plus the Arabic thousands/decimal separators, to
    ASCII. Latin digits and punctuation pass through unchanged, so this is
    safe to call unconditionally before parsing a count, a page number or a
    date out of scraped text — see ``_DIGIT_TRANSLATION`` for exactly which
    code points it rewrites. Does not cover other numeral scripts (Devanagari,
    CJK, ...); extend the table above if one turns up against a real session.
    """
    return _BIDI_MARKS_RE.sub("", text).translate(_DIGIT_TRANSLATION)


@dataclass(frozen=True)
class DetailCaptureTextTable:
    """Visible-text policy for hydrating and expanding profile details."""

    readiness_blocking_prefixes: tuple[str, ...]
    expansion_button_pattern: re.Pattern[str]

    def readiness_expression(self) -> str:
        """Build the historical readiness predicate without changing its bytes."""
        conditions = "\n                            && ".join(
            f"!text.startsWith({prefix!r})"
            for prefix in self.readiness_blocking_prefixes
        )
        return (
            "() => {\n"
            "                        const main = document.querySelector('main');\n"
            "                        if (!main) return false;\n"
            "                        const text = main.innerText.trimStart();\n"
            f"                        return {conditions};\n"
            "                    }"
        )


_DETAIL_CAPTURE_TEXT: dict[str, DetailCaptureTextTable] = {
    "en-US": DetailCaptureTextTable(
        readiness_blocking_prefixes=(
            "Load more",
            "More profiles for you",
            "Explore premium profiles",
        ),
        expansion_button_pattern=re.compile(
            r"^Show (more|all)\b",
            re.IGNORECASE,
        ),
    ),
}

# BrowserManager forces the browser context to en-US (core/browser.py), so the
# capture owner receives this exact entry. Unsupported locales are deliberately
# not inferred from language prefixes or detected from page text.
DETAIL_CAPTURE_EN_US = _DETAIL_CAPTURE_TEXT["en-US"]

# Patterns that mark the start of LinkedIn page chrome (sidebar/footer).
# Everything from the earliest match onwards is stripped.
_NOISE_MARKERS: list[re.Pattern[str]] = [
    # Footer nav links: "About" immediately followed by "Accessibility" or "Talent Solutions"
    re.compile(r"^About\n+(?:Accessibility|Talent Solutions)", re.MULTILINE),
    # Sidebar profile recommendations
    re.compile(r"^More profiles for you$", re.MULTILINE),
    # Sidebar premium upsell
    re.compile(r"^Explore premium profiles$", re.MULTILINE),
    # InMail upsell in contact info overlay
    re.compile(r"^Get up to .+ replies when you message with InMail$", re.MULTILINE),
    # Footer nav clusters in profile/posts pages
    re.compile(
        r"^(?:Careers|Privacy & Terms|Questions\?|Select language)\n+"
        r"(?:Privacy & Terms|Questions\?|Select language|Advertising|Ad Choices|"
        r"[A-Za-z]+ \([A-Za-z]+\))",
        re.MULTILINE,
    ),
]

_NOISE_LINES: list[re.Pattern[str]] = [
    re.compile(r"^(?:Play|Pause|Playback speed|Turn fullscreen on|Fullscreen)$"),
    re.compile(r"^(?:Show captions|Close modal window|Media player modal window)$"),
    re.compile(r"^(?:Loaded:.*|Remaining time.*|Stream Type.*)$"),
]


def strip_linkedin_noise(text: str) -> str:
    """Remove LinkedIn page chrome (footer, sidebar recommendations) from innerText.

    Finds the earliest occurrence of any known noise marker and truncates there.
    """
    cleaned = truncate_linkedin_noise(text)
    return filter_linkedin_noise_lines(cleaned)


def filter_linkedin_noise_lines(text: str) -> str:
    """Remove known media/control noise lines from already-truncated content."""
    filtered_lines = [
        line
        for line in text.splitlines()
        if not any(pattern.match(line.strip()) for pattern in _NOISE_LINES)
    ]
    return "\n".join(filtered_lines).strip()


def truncate_linkedin_noise(text: str) -> str:
    """Trim known LinkedIn chrome blocks before any per-line noise filtering."""
    earliest = len(text)
    for pattern in _NOISE_MARKERS:
        match = pattern.search(text)
        if match and match.start() < earliest:
            earliest = match.start()

    return text[:earliest].strip()


# Messaging-page chrome around an opened conversation thread. innerText on
# /messaging/thread/ pages carries no URL or attribute signal separating the
# inbox sidebar from the thread, so the boundaries are matched on visible
# strings — guarded by an explicit per-locale table (CLAUDE.md → Scraping
# Rules). BrowserManager forces the context locale to en-US (core/browser.py),
# so the "en" entry is the operative one; a locale without a table entry
# passes through unstripped.
@dataclass(frozen=True)
class _MessagingChromeTable:
    # Sidebar pagination control; the last line of the inbox sidebar. Pins
    # the thread header so quoted UI text inside messages can't move the
    # start boundary.
    sidebar_end: str
    # Screen-reader label on the options dropdown; appears once per sidebar
    # entry and once in the opened thread's header. The thread's own line is
    # the first occurrence after ``sidebar_end``.
    thread_header_prefix: str
    # First control of the trailing message-composer block.
    composer_start: str
    # Standalone controls of the composer block, matched exactly. At least
    # one must follow a ``composer_start`` candidate to confirm it is the
    # real composer rather than a message quoting the label. Controls whose
    # text embeds the participant name (the Attach lines) are deliberately
    # excluded: they would need prefix matching, and any prefix match lets
    # quoted control text with a suffix confirm a false boundary.
    composer_companions: tuple[str, ...]


# How far below a composer-label candidate a companion control may sit and
# still count as the same block. The observed block spans 6 lines; the slack
# covers extra controls LinkedIn injects (e.g. "Press Enter to Send").
_COMPOSER_COMPANION_WINDOW = 8

_MESSAGING_CHROME_STRINGS: dict[str, _MessagingChromeTable] = {
    "en": _MessagingChromeTable(
        sidebar_end="Load more conversations",
        thread_header_prefix="Open the options list in your conversation with",
        composer_start="Maximize compose field",
        composer_companions=(
            "Open GIF Keyboard",
            "Open Emoji Keyboard",
            "Open send options",
        ),
    ),
}


def strip_conversation_chrome(text: str, locale: str = "en") -> str:
    """Trim messaging chrome around an opened conversation thread.

    A conversation page's innerText embeds the thread between three chrome
    blocks: the messaging header, the inbox sidebar (which previews *other*
    conversations), and the trailing message composer. Drops everything
    through the thread-header line and everything from the composer onward.
    Each boundary independently falls back to keeping the text when its
    marker is absent (unknown locale, layout change), so a failed match
    leaks chrome rather than dropping messages.
    """
    table = _MESSAGING_CHROME_STRINGS.get(locale)
    if table is None:
        return text

    lines = text.splitlines()

    # End boundary: the last composer-label line, accepted only when an
    # exact companion control follows within the next few lines. The real
    # composer block is contiguous (label + controls observed within 6
    # lines), so a nearby companion confirms chrome, while a message that
    # quotes the label — or control text with any suffix — falls through to
    # the missing-marker fallback. A verbatim multi-line reproduction of the
    # block inside a message remains indistinguishable from the block itself;
    # that ambiguity is inherent to text-only stripping.
    end = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() != table.composer_start:
            continue
        if any(
            lines[j].strip() in table.composer_companions
            for j in range(i + 1, min(i + 1 + _COMPOSER_COMPANION_WINDOW, len(lines)))
        ):
            end = i
        break

    # Start boundary: the sidebar's pagination line, when present, pins the
    # real thread header as the first options line after it; quoted UI text
    # inside messages can no longer pull the boundary into the thread. The
    # sidebar omits the pagination control when there are few conversations —
    # then fall back to the last options line before the composer.
    start = 0
    sidebar_end = next(
        (i for i in range(end) if lines[i].strip() == table.sidebar_end), None
    )
    if sidebar_end is not None:
        header = next(
            (
                i
                for i in range(sidebar_end + 1, end)
                if lines[i].strip().startswith(table.thread_header_prefix)
            ),
            None,
        )
        start = (header + 1) if header is not None else sidebar_end + 1
    else:
        for i in range(end - 1, -1, -1):
            if lines[i].strip().startswith(table.thread_header_prefix):
                start = i + 1
                break

    return "\n".join(lines[start:end]).strip()


# Job search's advertised result count. LinkedIn prints it as a standalone
# line near the top of the results rail ("28 results", "1,000+ results"), with
# no URL or attribute carrying the number, so it is read from the same
# innerText `search_jobs` already captures rather than a second navigation.
# Guarded by an explicit per-locale table like the other text-only signals in
# this module. Only the first three non-empty lines are checked, matching
# where both the classic layout (count under the heading) and the redesigned
# layout (count first) place it; a number appearing later belongs to a job
# card, not the header.
@dataclass(frozen=True)
class JobSearchTextTable:
    """Visible-text policy for reading the job-search result count."""

    result_count_pattern: re.Pattern[str]

    def result_count(self, text: str) -> tuple[int, bool] | None:
        """The advertised `(count, exact)` from the first lines, or ``None``.

        ``exact`` is false when LinkedIn printed a trailing ``+`` (a lower
        bound, e.g. "1,000+ results"). Returns ``None`` when no candidate line
        matches — an absent count is not reported as zero.
        """
        for line in [line.strip() for line in text.splitlines() if line.strip()][:3]:
            match = self.result_count_pattern.fullmatch(line)
            if match:
                digits = match.group("count").replace(",", "")
                return int(digits), match.group("plus") is None
        return None


_JOB_SEARCH_TEXT: dict[str, JobSearchTextTable] = {
    "en-US": JobSearchTextTable(
        result_count_pattern=re.compile(
            r"(?P<count>[0-9][0-9,]*)(?P<plus>\+)? results?", re.IGNORECASE
        ),
    ),
}

# BrowserManager forces the browser context to en-US (core/browser.py), so the
# search workflow receives this exact entry. An unsupported locale reports no
# count rather than guessing at a translated word for "results".
JOB_SEARCH_EN_US = _JOB_SEARCH_TEXT["en-US"]


# The job-posting Save control's two states. LinkedIn exposes no URL,
# attribute or icon distinguishing "not yet saved" from "already saved" that
# has been found so far (a bookmark-glyph or `aria-pressed` toggle would be
# preferred per CLAUDE.md -> Scraping Rules, but the button was not observed
# to carry one) — the button's own text is the only signal, so detection is
# guarded by this explicit per-locale table and fails closed on an unknown
# locale rather than guessing. Every table is tried at once (the same shape
# as `core/utils.py`'s `_MODAL_DISMISS_ARIA_LABELS`), not gated on
# `navigator.language`: a session's actual UI locale is what LinkedIn chose
# for it, and there is no guarantee it matches the browser's reported
# language, so this reads whichever locale's label is actually on the page
# rather than trusting the report and refusing everyone else.
@dataclass(frozen=True)
class JobSaveTextTable:
    """Visible-text policy for reading and toggling the job Save control."""

    saved: str
    unsaved: str


_JOB_SAVE_TEXT: dict[str, JobSaveTextTable] = {
    "en": JobSaveTextTable(saved="Saved", unsaved="Save"),
    # Best-effort transcription, not verified against a live LinkedIn Arabic
    # session — see docs/i18n-audit.md's caveat on every table in this file.
    "ar": JobSaveTextTable(saved="تم الحفظ", unsaved="حفظ"),
}

# Kept for callers that still want the single historical entry (and for
# backward compatibility with existing tests); `save_job` itself reads
# `JOB_SAVE_TABLES` so every listed locale's label is recognized.
JOB_SAVE_EN_US = _JOB_SAVE_TEXT["en"]
JOB_SAVE_TABLES: tuple[JobSaveTextTable, ...] = tuple(_JOB_SAVE_TEXT.values())


# The per-thread options menu (opened from the header of an open conversation)
# used to mark a thread read/unread and archive/unarchive it. Preferred
# locale-independent signals — item order, an icon/`data-test-icon`, or an
# `aria-pressed`/`aria-checked` toggle state on the item itself — were looked
# for and not found: LinkedIn renders each item as a plain `role="menuitem"`
# with no icon markup and no ARIA toggle state, and item order was not
# confirmed stable (archive/unarchive and mark-read/unread are offered
# together with no documented fixed position). So the item's own label is the
# only signal available today, and it names the action offered rather than
# the current state (a thread already read offers "Mark as unread", not the
# reverse) — callers read whichever of a pair is present to tell the two
# apart, guarded by this explicit per-locale table (CLAUDE.md -> Scraping
# Rules) and failing closed when none of the tabled labels are found. If a
# structural signal is confirmed later, prefer it and keep this table only as
# a fallback.
#
# ``menu_opener_prefix`` reuses the exact aria-label prefix
# `_MessagingChromeTable.thread_header_prefix` already relies on for chrome
# stripping ("Open the options list in your conversation with") — that
# English string is a rendered, tested signal in this codebase, not a fresh
# guess; its Arabic counterpart below is not.
#
# Every table is tried at once (matching `core/utils.py`'s
# `_MODAL_DISMISS_ARIA_LABELS` shape) rather than gated on a single detected
# locale, so a session in any listed locale is recognized without first
# proving which one it is. The four toggle labels and the Arabic opener
# prefix are best-effort transcriptions, not confirmed against a live
# LinkedIn menu in either language; see docs/i18n-audit.md's caveat on every
# table in this file.
@dataclass(frozen=True)
class ConversationOptionsTextTable:
    """Visible-text policy for the per-thread options menu."""

    menu_opener_prefix: str
    mark_read: str
    mark_unread: str
    archive: str
    unarchive: str


_CONVERSATION_OPTIONS_STRINGS: dict[str, ConversationOptionsTextTable] = {
    "en": ConversationOptionsTextTable(
        menu_opener_prefix="Open the options list in your conversation with",
        mark_read="Mark as read",
        mark_unread="Mark as unread",
        archive="Archive",
        unarchive="Unarchive",
    ),
    "ar": ConversationOptionsTextTable(
        menu_opener_prefix="افتح قائمة الخيارات في محادثتك مع",
        mark_read="وضع علامة كمقروءة",
        mark_unread="وضع علامة كغير مقروءة",
        archive="أرشفة",
        unarchive="إلغاء الأرشفة",
    ),
}

# Kept for callers/tests that want the single historical entry; the DOM
# lookups themselves read `CONVERSATION_OPTIONS_TABLES` so every listed
# locale's labels are recognized.
CONVERSATION_OPTIONS_EN = _CONVERSATION_OPTIONS_STRINGS["en"]
CONVERSATION_OPTIONS_TABLES: tuple[ConversationOptionsTextTable, ...] = tuple(
    _CONVERSATION_OPTIONS_STRINGS.values()
)


# Sidebar recommendation headings on a person page, and the control that opens
# the full list behind one. Neither carries a URL, an attribute or a structural
# count separating it from any other heading or anchor in the same container,
# so both are matched on visible strings — guarded by an explicit per-locale
# table (CLAUDE.md → Scraping Rules) exactly like the messaging chrome above.
# This is the only place the strings are written down; `person.py` builds its
# extraction program from this table rather than repeating them.
@dataclass(frozen=True)
class SidebarChromeTable:
    # Headings of the recommendation sections worth collecting, matched whole
    # against a normalized `h1`/`h2`/`h3`. A heading outside the table is left
    # alone rather than guessed at.
    section_headings: tuple[str, ...]
    # Prefixes of the anchor that expands a section to its full list, matched
    # against lowercased anchor text. LinkedIn labels that control either way
    # depending on the surface, so both spellings are listed.
    show_all_prefixes: tuple[str, ...]


_SIDEBAR_CHROME_STRINGS: dict[str, SidebarChromeTable] = {
    "en": SidebarChromeTable(
        section_headings=(
            "More profiles for you",
            "Explore premium profiles",
            "People you may know",
        ),
        show_all_prefixes=("show all", "see all"),
    ),
}

# BrowserManager forces the context locale to en-US (core/browser.py), so this
# is the entry a running server reads, and the dictionary above is what makes
# that dependency visible instead of implicit. A locale with no entry would
# collect nothing here, which is why the sidebar is the one workflow whose
# coverage has to be stated per locale rather than assumed.
SIDEBAR_CHROME_EN = _SIDEBAR_CHROME_STRINGS["en"]
