"""Read a post's comment thread and write comments through LinkedIn's browser UI.

Everything here addresses a comment by its URN, never by position or by text.
The DOM carries it as an attribute value on the comment's own container
(``data-id="urn:li:comment:(activity:<post>,<comment>)"``, stickerdaniel/linkedin-mcp-server
issue 828), and nesting between containers is what makes one comment a reply to
another. Identity, parentage and authorship are therefore structural reads, and
the Scraping Rules' locale independence holds for every decision below:

- **Comment identity**: the outermost element whose ``data-id``, ``data-urn`` or
  ``data-entity-urn`` parses as a comment URN. Two disjoint containers claiming
  one id are ambiguous, and every write refuses them.
- **Load-more controls**: a button inside a comment list but outside every comment
  in it, with no toggle, menu or submit semantics, in a scope that holds no toggle
  (``aria-pressed``), menu or editor of its own. The last rule is the safety one:
  it stops the scope from widening into the post's or a comment's action bar,
  where a click could react to something.
- **Reply action**: the comment's own action bar is anchored on its single own
  toggle (the reaction button, ``aria-pressed``). The reply action is a plain
  button after it in that bar, and a click only counts once a *new* editor
  appears inside that comment (or, for a reply, inside its parent thread, which
  is where LinkedIn opens replies to replies).
- **Submission**: the one ``type=submit`` button of the editor's form, or else the
  one button near the editor that typing turned from disabled to enabled.
- **Confirmation**: a comment id absent before submission appears under the
  expected parent and its own text contains exactly what was typed.

Known limits, documented rather than guessed around: the on-page sort menu is not
driven (its options differ only by text), so ``sort="recent"`` orders by comment
id, which LinkedIn issues in time order; @mentions are refused because typing
``@`` opens a picker this module does not drive; only the default reaction (Like)
is supported, since the others live in a hover menu whose items are text-only.
Deleting a comment needs the overflow menu for the same reason and is not offered.

Prior art: upstream PR 683 (get_post_comments, bounded thread expansion) and PR
571 (comment reading; lazy SDUI batches load on wheel scroll, verified live
2026-07-07). Both matched pagination by English label, which this port replaces
with the structural rules above.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from urllib.parse import quote, unquote, urlparse

import logging
import random
import re
import secrets

import anyio
from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.error_diagnostics import build_issue_diagnostics
from linkedin_mcp_server.core.exceptions import (
    InvalidReferenceError,
    LinkedInScraperException,
)
from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.contracts import rate_limited_section_error
from linkedin_mcp_server.scraping.identifiers import (
    normalize_post_urn,
    post_update_url,
)
from linkedin_mcp_server.scraping.link_metadata import Reference
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession
from linkedin_mcp_server.scraping.text import (
    filter_linkedin_noise_lines,
    truncate_linkedin_noise,
)

logger = logging.getLogger(__name__)

# LinkedIn's own limit for a comment or a reply.
COMMENT_MAX_LENGTH = 1250
MAX_COMMENTS_LIMIT = 100
CommentSort = Literal["relevant", "recent"]

_EXPANSION_CLICK_BUDGET = 20
_EXPANSION_STALE_ROUNDS = 2
_EXPANSION_SETTLE_SECONDS = 1.2
_WHEEL_DELTA = 1800
_CONTENT_WAIT_MS = 8_000
_CLICK_TIMEOUT_MS = 3_000
_REPLY_BOX_WAIT_SECONDS = 3.0
_SUBMIT_READY_WAIT_SECONDS = 2.0
_CONFIRMATION_WAIT_SECONDS = 15.0
_REACTION_WAIT_SECONDS = 5.0
_POLL_SECONDS = 0.25
_MAX_REPLY_CANDIDATES = 2
_CLEANUP_TIMEOUT_SECONDS = 1.0

# Namespaces a comment URN may name its thread by, as the page writes them.
_COMMENT_URN_RE = re.compile(
    r"^urn:li:comment:\((?:urn:li:)?(activity|ugcPost|share):([0-9]+),([0-9]+)\)$"
)
_FSD_COMMENT_URN_RE = re.compile(
    r"^urn:li:fsd_comment:\(([0-9]+),urn:li:(activity|ugcPost|share):([0-9]+)\)$"
)
_AUTHOR_PATH_RE = re.compile(r"^/(?:in|company)/[^/?#\s]+/$")
_NUMERIC_ID_RE = re.compile(r"^[0-9]{1,20}$")
_POST_ROUTE_URN_RE = re.compile(
    r"^/feed/update/(urn:li:(?:activity|ugcPost|share):[0-9]+)/?$"
)

# LinkedIn ids carry their creation time in the high bits: the id shifted right
# by 22 is milliseconds since the Unix epoch. Anything outside this window is
# not such an id, and no timestamp is reported for it.
_ID_EPOCH_FLOOR = datetime(2003, 1, 1, tzinfo=timezone.utc)

_random = random.Random()


@dataclass(frozen=True, slots=True)
class CommentUrn:
    """A comment, as ``(thread namespace, thread id, comment id)``."""

    thread_kind: str
    thread_id: str
    comment_id: str

    @property
    def urn(self) -> str:
        return f"urn:li:comment:({self.thread_kind}:{self.thread_id},{self.comment_id})"

    @property
    def thread_urn(self) -> str:
        return f"urn:li:{self.thread_kind}:{self.thread_id}"


class CommentReference(Reference, total=False):
    """One comment in ``references["comments"]``.

    ``url`` is the author's profile or company path, ``text`` the author's name,
    ``context`` the lines LinkedIn prints under the name (headline, degree),
    ``value`` the comment URN. ``parent`` is the URN of the comment a reply sits
    under, ``permalink`` a link that opens the post scrolled to this comment,
    ``excerpt`` the start of the comment's own text and ``posted_at`` the UTC
    time decoded from the comment id.
    """

    parent: str
    permalink: str
    excerpt: str
    posted_at: str


def parse_comment_urn(value: str) -> CommentUrn | None:
    """Parse ``urn:li:comment:(…)`` or ``urn:li:fsd_comment:(…)``, or ``None``.

    One layer of percent-encoding is accepted, because a permalink carries the
    URN encoded in its query and a caller may paste it from there.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if "%" in text:
        try:
            text = unquote(text, errors="strict")
        except UnicodeDecodeError:
            return None
    if match := _COMMENT_URN_RE.match(text):
        return CommentUrn(match.group(1), match.group(2), match.group(3))
    if match := _FSD_COMMENT_URN_RE.match(text):
        return CommentUrn(match.group(2), match.group(3), match.group(1))
    return None


def normalize_comment_urn(value: str) -> CommentUrn:
    """Parse a comment URN or raise the correction the caller can act on."""
    parsed = parse_comment_urn(value)
    if parsed is None:
        raise InvalidReferenceError(
            "comment_urn is not a LinkedIn comment URN. Pass the `value` of an "
            'entry in get_post_comments\' references["comments"], for example '
            '"urn:li:comment:(activity:7000000000000000000,7000000000000000001)".'
        )
    return parsed


def comment_permalink(comment: CommentUrn, parent: CommentUrn | None = None) -> str:
    """Relative link that opens the post scrolled to *comment*."""
    base = f"/feed/update/{comment.thread_urn}/"
    if parent is None:
        return f"{base}?commentUrn={quote(comment.urn, safe='')}"
    return (
        f"{base}?commentUrn={quote(parent.urn, safe='')}"
        f"&replyUrn={quote(comment.urn, safe='')}"
    )


def posted_at_from_id(comment_id: str) -> str | None:
    """The creation time a LinkedIn id encodes, or ``None`` when implausible."""
    if not _NUMERIC_ID_RE.match(comment_id):
        return None
    milliseconds = int(comment_id) >> 22
    moment = datetime.fromtimestamp(0, tz=timezone.utc) + timedelta(
        milliseconds=milliseconds
    )
    if moment < _ID_EPOCH_FLOOR or moment > datetime.now(timezone.utc) + timedelta(
        days=1
    ):
        return None
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_comment_text(text: str) -> tuple[str | None, str | None]:
    """``(normalized text, None)`` or ``(None, reason)`` for text to post.

    Line breaks are the one control character supported: CRLF and CR become LF
    and each is typed as Shift+Enter, which inserts a break in LinkedIn's editor
    without submitting. Every other C0 control and DEL is refused, as are lone
    surrogates (they cannot be typed) and ``@``, which opens LinkedIn's mention
    picker; @mentions are out of scope for these tools.
    """
    if not isinstance(text, str):
        return None, "Text must be a string."
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return None, "Text must contain non-whitespace characters."
    if any(
        (ord(character) < 32 and character != "\n") or ord(character) == 127
        for character in normalized
    ):
        return None, "Text must not contain control characters other than line breaks."
    if any(0xD800 <= ord(character) <= 0xDFFF for character in normalized):
        return None, "Text must not contain unpaired surrogate code points."
    if "@" in normalized:
        return (
            None,
            "Text must not contain '@': it opens LinkedIn's mention picker, and "
            "@mentions are not supported by this tool.",
        )
    if len(normalized) > COMMENT_MAX_LENGTH:
        return (
            None,
            f"Text is {len(normalized)} characters; LinkedIn allows at most "
            f"{COMMENT_MAX_LENGTH}.",
        )
    return normalized, None


def comment_action_result(
    url: str,
    status: str,
    message: str,
    *,
    post_urn: str | None = None,
    comment_urn: str | None = None,
    posted_comment_urn: str | None = None,
    text: str | None = None,
    target: dict[str, Any] | None = None,
    posted: bool = False,
    retry_safe: bool = True,
) -> dict[str, Any]:
    """Structured answer for every comment write tool.

    ``posted`` is true only after the new comment was observed under the expected
    parent carrying the typed text. ``retry_safe`` is false from the moment a
    submission may have been dispatched, because a retry can then post twice.
    """
    result: dict[str, Any] = {"url": url, "status": status, "message": message}
    if post_urn is not None:
        result["post_urn"] = post_urn
    if comment_urn is not None:
        result["comment_urn"] = comment_urn
    if posted_comment_urn is not None:
        result["posted_comment_urn"] = posted_comment_urn
    if text is not None:
        result["text"] = text
    if target is not None:
        result["target"] = target
    result["posted"] = posted
    result["retry_safe"] = retry_safe
    return result


# Logged when a write is cancelled (FastMCP's deadline) after its submit click
# may have been dispatched. A cancelled scope discards any result, so this line
# is the only record that a comment may already be on the post.
COMMENT_INTERRUPTED_WARNING = (
    "Comment submission was interrupted while in flight. The outcome is unknown; "
    "read the post with get_post_comments before retrying, as a retry may post "
    "the comment twice."
)


def prepare_comment_write(
    post_url: str,
    comment_urn: str | None,
    text: str,
    *,
    confirm: bool,
) -> dict[str, Any] | None:
    """Answer a comment write that needs no browser, or ``None`` if it does.

    Invalid text and every ``confirm=False`` call are answered here, before a
    session is acquired, so a preview can never change anything on LinkedIn.

    Raises:
        InvalidReferenceError: for a post URL or comment URN that names nothing.
    """
    post_urn = normalize_post_urn(post_url)
    url = post_update_url(post_urn)
    target = normalize_comment_urn(comment_urn) if comment_urn is not None else None
    common: dict[str, Any] = {
        "post_urn": post_urn,
        "comment_urn": target.urn if target is not None else None,
    }
    normalized, reason = normalize_comment_text(text)
    if normalized is None:
        return comment_action_result(
            url, "invalid_text", reason or "Invalid text.", **common
        )
    if confirm:
        return None
    what = f"reply to comment {target.urn}" if target is not None else "comment"
    return comment_action_result(
        url,
        "preview",
        f"Nothing was posted. Set confirm=true to post this {what} on {post_urn} "
        f"({len(normalized)} characters). @mentions are not supported.",
        text=normalized,
        **common,
    )


def prepare_comment_reaction(
    post_url: str,
    comment_urn: str,
    reaction: str,
    *,
    confirm: bool,
) -> dict[str, Any] | None:
    """Answer a reaction that needs no browser, or ``None`` if it does."""
    post_urn = normalize_post_urn(post_url)
    url = post_update_url(post_urn)
    target = normalize_comment_urn(comment_urn)
    common: dict[str, Any] = {"post_urn": post_urn, "comment_urn": target.urn}
    if reaction != "like":
        return comment_action_result(
            url,
            "unsupported_reaction",
            'Only reaction="like" is supported: the other reactions sit in a hover '
            "menu whose items are identified only by text.",
            **common,
        )
    if confirm:
        return None
    return comment_action_result(
        url,
        "preview",
        f"Nothing was changed. Set confirm=true to like comment {target.urn}.",
        **common,
    )


# ---------------------------------------------------------------------------
# Browser programs. Each is self-contained: it re-reads the page from scratch,
# so nothing a previous call saw can vouch for what the page holds now.
# ---------------------------------------------------------------------------

_DOM_PRELUDE = r"""
    const visible = element => {
        const visibility = element && getComputedStyle(element).visibility;
        return !!(
            element &&
            visibility !== 'hidden' &&
            visibility !== 'collapse' &&
            (element.offsetWidth || element.offsetHeight || element.getClientRects().length)
        );
    };
    const normalizeText = value => (value || '').replace(/\s+/g, ' ').trim();
    const COMMENT_ATTRIBUTES = ['data-id', 'data-urn', 'data-entity-urn'];
    const COMMENT_SELECTOR = COMMENT_ATTRIBUTES
        .map(name => `[${name}*="comment:("]`)
        .join(', ');
    const EDITOR_SELECTOR = '[contenteditable]:not([contenteditable="false"])';
    const DIALOG_SELECTOR = 'dialog[open], [role="dialog"], [role="alertdialog"]';
    const RISKY_SELECTOR =
        '[aria-pressed], [aria-checked], [aria-haspopup], ' +
        EDITOR_SELECTOR + ', form, dialog, [role="dialog"]';
    const parseCommentUrn = value => {
        if (typeof value !== 'string') return null;
        const text = value.trim();
        let match = /^urn:li:comment:\((?:urn:li:)?(activity|ugcPost|share):([0-9]+),([0-9]+)\)$/
            .exec(text);
        if (match) return {kind: match[1], threadId: match[2], id: match[3]};
        match = /^urn:li:fsd_comment:\(([0-9]+),urn:li:(activity|ugcPost|share):([0-9]+)\)$/
            .exec(text);
        if (match) return {kind: match[2], threadId: match[3], id: match[1]};
        return null;
    };
    const urnOf = element => {
        for (const name of COMMENT_ATTRIBUTES) {
            if (!element.hasAttribute || !element.hasAttribute(name)) continue;
            const parsed = parseCommentUrn(element.getAttribute(name));
            if (parsed) return parsed;
        }
        return null;
    };
    const scanRoot = () => document.querySelector('main') || document.body;
    const profilePath = href => {
        if (typeof href !== 'string' || /[\\\x00-\x1f\x7f]/.test(href)) return null;
        try {
            const url = new URL(href, window.location.href);
            const host = url.hostname.toLowerCase().replace(/\.$/, '');
            if (url.protocol !== 'https:' || !/(^|\.)linkedin\.com$/.test(host)) {
                return null;
            }
            const match = /^\/(in|company)\/([^/?#]+)/.exec(url.pathname);
            return match ? `/${match[1]}/${match[2]}/` : null;
        } catch {
            return null;
        }
    };
    const readUnits = () => {
        const tagged = [];
        for (const element of scanRoot().querySelectorAll(COMMENT_SELECTOR)) {
            const urn = urnOf(element);
            if (urn) tagged.push({element, urn});
        }
        const units = tagged.filter(({element, urn}) => {
            for (let above = element.parentElement; above; above = above.parentElement) {
                const outer = urnOf(above);
                if (outer && outer.id === urn.id) return false;
            }
            return true;
        });
        const byElement = new Map(units.map(unit => [unit.element, unit]));
        const enclosingUnit = node => {
            let element = node && node.nodeType === Node.ELEMENT_NODE
                ? node
                : node && node.parentElement;
            for (; element; element = element.parentElement) {
                const unit = byElement.get(element);
                if (unit) return unit;
            }
            return null;
        };
        for (const unit of units) {
            unit.parent = enclosingUnit(unit.element.parentElement);
        }
        const withId = id => units.filter(unit => unit.urn.id === id);
        return {units, enclosingUnit, withId};
    };
    const ownText = (state, unit, skipPath) => {
        const parts = [];
        const walker = document.createTreeWalker(unit.element, NodeFilter.SHOW_TEXT);
        for (let node = walker.nextNode(); node; node = walker.nextNode()) {
            const parent = node.parentElement;
            if (!parent || state.enclosingUnit(parent) !== unit) continue;
            if (parent.closest(
                'button, script, style, [aria-hidden="true"], ' + EDITOR_SELECTOR
            )) {
                continue;
            }
            if (skipPath) {
                const anchor = parent.closest('a[href]');
                if (anchor && profilePath(anchor.getAttribute('href')) === skipPath) {
                    continue;
                }
            }
            parts.push(node.nodeValue);
        }
        return normalizeText(parts.join(' '));
    };
    const authorOf = (state, unit) => {
        let path = null;
        let lines = [];
        for (const anchor of unit.element.querySelectorAll('a[href]')) {
            if (state.enclosingUnit(anchor) !== unit) continue;
            if (anchor.closest(EDITOR_SELECTOR)) continue;
            const candidate = profilePath(anchor.getAttribute('href'));
            if (!candidate) continue;
            if (path === null) path = candidate;
            if (candidate !== path) continue;
            const text = (anchor.innerText || anchor.textContent || '')
                .split('\n')
                .map(normalizeText)
                .filter(Boolean);
            if (text.length > lines.length) lines = text;
        }
        return path ? {path, lines: lines.slice(0, 6)} : null;
    };
    const isDisabled = button =>
        button.disabled ||
        (button.getAttribute('aria-disabled') || '').toLowerCase() === 'true';
    const visibleEditors = () =>
        Array.from(document.querySelectorAll(EDITOR_SELECTOR)).filter(editor =>
            visible(editor) &&
            !(editor.parentElement && editor.parentElement.closest(EDITOR_SELECTOR))
        );
    const visibleDialogs = () =>
        Array.from(document.querySelectorAll(DIALOG_SELECTOR)).filter(visible).length;
    const describe = (state, unit) => {
        const author = authorOf(state, unit);
        return {
            kind: unit.urn.kind,
            threadId: unit.urn.threadId,
            id: unit.urn.id,
            parentKind: unit.parent ? unit.parent.urn.kind : null,
            parentThreadId: unit.parent ? unit.parent.urn.threadId : null,
            parentId: unit.parent ? unit.parent.urn.id : null,
            authorPath: author ? author.path : null,
            authorLines: author ? author.lines : [],
            text: ownText(state, unit, author ? author.path : null).slice(0, 2000),
        };
    };
    const MARKS = [
        'data-linkedin-mcp-loader',
        'data-linkedin-mcp-reply-candidate',
        'data-linkedin-mcp-preexisting',
        'data-linkedin-mcp-comment-editor',
        'data-linkedin-mcp-disabled-before',
        'data-linkedin-mcp-submit',
        'data-linkedin-mcp-react',
    ];
    const clearMarks = token => {
        const selector = MARKS.map(name => `[${name}]`).join(', ');
        for (const element of document.querySelectorAll(selector)) {
            for (const name of MARKS) {
                const value = element.getAttribute(name);
                if (value !== null && (value === token || value.startsWith(token + '-'))) {
                    element.removeAttribute(name);
                }
            }
        }
    };
    const locate = (state, arg) => {
        const matches = state.withId(arg.id);
        if (matches.length === 0) return {status: 'missing'};
        if (matches.length > 1) return {status: 'ambiguous'};
        const unit = matches[0];
        if (unit.urn.kind === arg.kind && unit.urn.threadId !== arg.threadId) {
            return {status: 'contradiction'};
        }
        return {status: 'found', unit};
    };
    const pinnedEditor = (state, arg) => {
        const editors = Array.from(
            document.querySelectorAll(`[data-linkedin-mcp-comment-editor="${arg.token}"]`)
        );
        if (editors.length !== 1) return {status: 'editor_lost'};
        const editor = editors[0];
        if (!editor.isConnected || !visible(editor)) return {status: 'editor_lost'};
        const owner = state.enclosingUnit(editor);
        if (arg.id === null) {
            if (owner !== null || editor.closest(DIALOG_SELECTOR)) {
                return {status: 'editor_moved'};
            }
            return {status: 'ok', editor, unit: null};
        }
        const located = locate(state, arg);
        if (located.status !== 'found') return {status: located.status};
        const unit = located.unit;
        if (owner !== unit && !(unit.parent && owner === unit.parent)) {
            return {status: 'editor_moved'};
        }
        return {status: 'ok', editor, unit};
    };
"""

_READ_COMMENTS_JS = (
    "() => {"
    + _DOM_PRELUDE
    + r"""
        const state = readUnits();
        const counts = new Map();
        for (const unit of state.units) {
            counts.set(unit.urn.id, (counts.get(unit.urn.id) || 0) + 1);
        }
        return state.units.map(unit => ({
            ...describe(state, unit),
            duplicate: counts.get(unit.urn.id) > 1,
        }));
    }"""
)

_EXPANSION_SCAN_JS = (
    "(arg) => {"
    + _DOM_PRELUDE
    + r"""
        clearMarks(arg.token);
        const state = readUnits();
        const dead = window.__linkedinMcpDeadLoaders ||
            (window.__linkedinMcpDeadLoaders = new WeakSet());
        const groups = new Map();
        for (const unit of state.units) {
            const key = unit.parent || null;
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key).push(unit);
        }
        if (!groups.has(null)) groups.set(null, []);
        const eligible = (button, owner) =>
            visible(button) &&
            !isDisabled(button) &&
            state.enclosingUnit(button) === owner &&
            !button.matches(
                '[aria-pressed], [aria-checked], [aria-haspopup], [aria-expanded], ' +
                '[type="submit"]'
            ) &&
            !button.closest('a[href], form, ' + DIALOG_SELECTOR + ', ' + EDITOR_SELECTOR) &&
            !button.querySelector('a[href], ' + EDITOR_SELECTOR) &&
            !dead.has(button);
        const loaders = [];
        const seen = new Set();
        for (const [owner, children] of groups) {
            if (owner && !arg.includeReplies) continue;
            if (children.length === 0) continue;
            let scope = children[0].element.parentElement;
            while (scope && !children.every(child => scope.contains(child.element))) {
                scope = scope.parentElement;
            }
            const limit = owner ? owner.element : scanRoot();
            for (let level = 0; scope && level < 3; level += 1, scope = scope.parentElement) {
                if (scope === limit || !limit.contains(scope)) break;
                const unsafe = Array.from(scope.querySelectorAll(RISKY_SELECTOR))
                    .some(element => state.enclosingUnit(element) === owner);
                if (unsafe) break;
                for (const button of scope.querySelectorAll('button')) {
                    if (seen.has(button) || !eligible(button, owner)) continue;
                    seen.add(button);
                    loaders.push(button);
                }
            }
        }
        if (loaders.length > 0) {
            loaders[0].setAttribute('data-linkedin-mcp-loader', arg.token);
        }
        return {
            count: state.units.length,
            targetCount: arg.targetId ? state.withId(arg.targetId).length : 0,
            loaders: loaders.length,
        };
    }"""
)

_EXPANSION_DEAD_JS = r"""(token) => {
    const dead = window.__linkedinMcpDeadLoaders ||
        (window.__linkedinMcpDeadLoaders = new WeakSet());
    for (const button of document.querySelectorAll(
        `[data-linkedin-mcp-loader="${token}"]`
    )) {
        dead.add(button);
    }
}"""

_LOCATE_COMMENT_JS = (
    "(arg) => {"
    + _DOM_PRELUDE
    + r"""
        const state = readUnits();
        const located = locate(state, arg);
        if (located.status !== 'found') return {status: located.status};
        return {status: 'found', ...describe(state, located.unit)};
    }"""
)

_REPLY_PREPARE_JS = (
    "(arg) => {"
    + _DOM_PRELUDE
    + r"""
        clearMarks(arg.token);
        const state = readUnits();
        const located = locate(state, arg);
        if (located.status !== 'found') return {status: located.status};
        const unit = located.unit;
        const editors = visibleEditors();
        const open = editors.filter(editor => state.enclosingUnit(editor) === unit);
        if (open.length > 1) return {status: 'ambiguous_editor'};
        if (open.length === 1) {
            if (normalizeText(open[0].innerText || open[0].textContent)) {
                return {status: 'occupied'};
            }
            open[0].setAttribute('data-linkedin-mcp-comment-editor', arg.token);
            return {status: 'ready', prefill: ''};
        }
        for (const editor of editors) {
            editor.setAttribute('data-linkedin-mcp-preexisting', arg.token);
        }
        const toggles = Array.from(unit.element.querySelectorAll('button[aria-pressed]'))
            .filter(button => visible(button) && state.enclosingUnit(button) === unit);
        if (toggles.length !== 1) return {status: 'unresolved'};
        const toggle = toggles[0];
        let candidates = [];
        let bar = toggle.parentElement;
        for (let level = 0; bar && level < 3 && bar !== unit.element; level += 1) {
            const reachesHeader = Array.from(bar.querySelectorAll('a[href]'))
                .some(anchor => profilePath(anchor.getAttribute('href')));
            const reachesReplies = state.units.some(
                other => other !== unit && bar.contains(other.element)
            );
            if (reachesHeader || reachesReplies || bar.querySelector(EDITOR_SELECTOR)) break;
            candidates = Array.from(bar.querySelectorAll('button')).filter(button =>
                button !== toggle &&
                state.enclosingUnit(button) === unit &&
                visible(button) &&
                !isDisabled(button) &&
                !button.matches(
                    '[aria-pressed], [aria-checked], [aria-haspopup], [aria-expanded], ' +
                    '[type="submit"]'
                ) &&
                !button.closest('a[href], form, ' + DIALOG_SELECTOR) &&
                (toggle.compareDocumentPosition(button) & Node.DOCUMENT_POSITION_FOLLOWING)
            );
            if (candidates.length) break;
            bar = bar.parentElement;
        }
        const media = button => (button.querySelector('img, svg, picture') ? 1 : 0);
        candidates.sort((first, second) => media(first) - media(second));
        candidates = candidates.slice(0, arg.maxCandidates);
        candidates.forEach((button, index) =>
            button.setAttribute('data-linkedin-mcp-reply-candidate', `${arg.token}-${index}`)
        );
        return {
            status: candidates.length ? 'candidates' : 'unresolved',
            candidates: candidates.length,
            dialogs: visibleDialogs(),
        };
    }"""
)

_REPLY_EDITOR_JS = (
    "(arg) => {"
    + _DOM_PRELUDE
    + r"""
        const state = readUnits();
        const located = locate(state, arg);
        if (located.status !== 'found') return {status: located.status};
        const unit = located.unit;
        if (visibleDialogs() > arg.dialogs) return {status: 'dialog'};
        const fresh = visibleEditors().filter(editor =>
            editor.getAttribute('data-linkedin-mcp-preexisting') !== arg.token
        );
        const owned = fresh.filter(editor => {
            const owner = state.enclosingUnit(editor);
            return owner === unit || (unit.parent !== null && owner === unit.parent);
        });
        if (owned.length !== fresh.length) return {status: 'foreign_editor'};
        if (owned.length === 0) return {status: 'waiting'};
        if (owned.length > 1) return {status: 'ambiguous_editor'};
        owned[0].setAttribute('data-linkedin-mcp-comment-editor', arg.token);
        return {
            status: 'opened',
            prefill: normalizeText(owned[0].innerText || owned[0].textContent),
        };
    }"""
)

_POST_COMPOSER_JS = (
    "(arg) => {"
    + _DOM_PRELUDE
    + r"""
        clearMarks(arg.token);
        const state = readUnits();
        const root = scanRoot();
        const editors = visibleEditors().filter(editor =>
            root.contains(editor) &&
            state.enclosingUnit(editor) === null &&
            !editor.closest(DIALOG_SELECTOR)
        );
        if (editors.length === 0) return {status: 'unavailable'};
        if (editors.length > 1) return {status: 'ambiguous_editor'};
        if (normalizeText(editors[0].innerText || editors[0].textContent)) {
            return {status: 'occupied'};
        }
        editors[0].setAttribute('data-linkedin-mcp-comment-editor', arg.token);
        return {status: 'ready', prefill: ''};
    }"""
)

_EDITOR_FOCUS_JS = (
    "(arg) => {"
    + _DOM_PRELUDE
    + r"""
        const state = readUnits();
        const pinned = pinnedEditor(state, arg);
        if (pinned.status !== 'ok') return {status: pinned.status};
        const editor = pinned.editor;
        if (normalizeText(editor.innerText || editor.textContent) !== arg.prefill) {
            return {status: 'occupied'};
        }
        editor.focus();
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(editor);
        range.collapse(false);
        selection.removeAllRanges();
        selection.addRange(range);
        // The neighbourhood a submit button can sit in: up to four levels above
        // the editor, never reaching a comment container or the page root.
        const editorOwner = state.enclosingUnit(editor);
        let scope = editor.parentElement;
        for (let level = 0; scope && level < 4; level += 1) {
            const next = scope.parentElement;
            if (!next || next.matches('main, body')) break;
            const nextOwner = state.enclosingUnit(next);
            if (nextOwner !== editorOwner || (nextOwner && nextOwner.element === next)) {
                break;
            }
            scope = next;
        }
        for (const button of (scope || editor).querySelectorAll('button')) {
            if (isDisabled(button)) {
                button.setAttribute('data-linkedin-mcp-disabled-before', arg.token);
            }
        }
        const active = document.activeElement;
        return {
            status: active === editor || editor.contains(active) ? 'ready' : 'unfocused',
        };
    }"""
)

_SUBMIT_RESOLVE_JS = (
    "(arg) => {"
    + _DOM_PRELUDE
    + r"""
        for (const element of document.querySelectorAll(
            `[data-linkedin-mcp-submit="${arg.token}"]`
        )) {
            element.removeAttribute('data-linkedin-mcp-submit');
        }
        const state = readUnits();
        const pinned = pinnedEditor(state, arg);
        if (pinned.status !== 'ok') return {status: pinned.status};
        const editor = pinned.editor;
        const typed = normalizeText(editor.innerText || editor.textContent);
        if (typed !== normalizeText(`${arg.prefill} ${arg.expected}`)) {
            return {status: 'mismatch'};
        }
        const form = editor.closest('form');
        let submits = form
            ? Array.from(form.querySelectorAll('button[type="submit"]')).filter(visible)
            : [];
        if (submits.length !== 1) {
            submits = Array.from(document.querySelectorAll(
                `[data-linkedin-mcp-disabled-before="${arg.token}"]`
            )).filter(button => visible(button) && !isDisabled(button));
        }
        if (submits.length === 0) return {status: 'unavailable'};
        if (submits.length > 1) return {status: 'ambiguous'};
        const button = submits[0];
        if (button.matches('[aria-pressed], [aria-haspopup], [aria-expanded]')) {
            return {status: 'unavailable'};
        }
        if (isDisabled(button)) return {status: 'disabled'};
        button.setAttribute('data-linkedin-mcp-submit', arg.token);
        return {status: 'ready'};
    }"""
)

_COMMENT_IDS_JS = (
    "() => {" + _DOM_PRELUDE + "return readUnits().units.map(unit => unit.urn.id); }"
)

_CONFIRM_POSTED_JS = (
    "(arg) => {"
    + _DOM_PRELUDE
    + r"""
        const state = readUnits();
        const baseline = new Set(arg.baseline);
        const expected = normalizeText(arg.expected);
        const matching = state.units.filter(unit =>
            !baseline.has(unit.urn.id) &&
            arg.parents.includes(unit.parent ? unit.parent.urn.id : null) &&
            ownText(state, unit, null).includes(expected)
        );
        if (matching.length === 0) return {status: 'waiting'};
        if (matching.length > 1) return {status: 'ambiguous'};
        return {status: 'confirmed', ...describe(state, matching[0])};
    }"""
)

_EDITOR_CLEANUP_JS = (
    "(arg) => {"
    + _DOM_PRELUDE
    + r"""
        const state = readUnits();
        const pinned = pinnedEditor(state, arg);
        if (pinned.status !== 'ok') return false;
        const editor = pinned.editor;
        const typed = normalizeText(editor.innerText || editor.textContent);
        const prefill = normalizeText(arg.prefill);
        const rest = normalizeText(typed.slice(prefill.length));
        if (!typed.startsWith(prefill) || !normalizeText(arg.expected).startsWith(rest)) {
            return false;
        }
        editor.focus();
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(editor);
        selection.removeAllRanges();
        selection.addRange(range);
        return document.execCommand('delete');
    }"""
)

_REACTION_JS = (
    "(arg) => {"
    + _DOM_PRELUDE
    + r"""
        if (arg.mark) clearMarks(arg.token);
        const state = readUnits();
        const located = locate(state, arg);
        if (located.status !== 'found') return {status: located.status};
        const unit = located.unit;
        const toggles = Array.from(unit.element.querySelectorAll('button[aria-pressed]'))
            .filter(button => visible(button) && state.enclosingUnit(button) === unit);
        if (toggles.length !== 1) return {status: 'unresolved'};
        const toggle = toggles[0];
        if (arg.mark) {
            toggle.setAttribute('data-linkedin-mcp-react', arg.token);
        } else if (toggle.getAttribute('data-linkedin-mcp-react') !== arg.token) {
            return {status: 'replaced', pressed: toggle.getAttribute('aria-pressed')};
        }
        return {
            status: 'found',
            pressed: (toggle.getAttribute('aria-pressed') || '').toLowerCase(),
            disabled: isDisabled(toggle),
        };
    }"""
)

_CLEANUP_JS = (
    "(token) => {"
    + _DOM_PRELUDE
    + r"""
        clearMarks(token);
        delete window.__linkedinMcpDeadLoaders;
    }"""
)


def _token() -> str:
    return f"lmcp{secrets.token_hex(6)}"


def _clean_lines(lines: Any) -> list[str]:
    cleaned: list[str] = []
    if not isinstance(lines, list):
        return cleaned
    for line in lines:
        if not isinstance(line, str):
            continue
        line = " ".join(line.split())
        if line and (not cleaned or cleaned[-1] != line):
            cleaned.append(line)
    return cleaned


def _unit_urn(item: dict[str, Any]) -> CommentUrn | None:
    kind = item.get("kind")
    thread_id = item.get("threadId")
    comment_id = item.get("id")
    if (
        kind not in {"activity", "ugcPost", "share"}
        or not isinstance(thread_id, str)
        or not isinstance(comment_id, str)
        or not _NUMERIC_ID_RE.match(thread_id)
        or not _NUMERIC_ID_RE.match(comment_id)
    ):
        return None
    return CommentUrn(kind, thread_id, comment_id)


def _describe_target(item: dict[str, Any]) -> dict[str, Any]:
    """The part of a located comment a caller can check before writing."""
    lines = _clean_lines(item.get("authorLines"))
    target: dict[str, Any] = {}
    author_path = item.get("authorPath")
    if isinstance(author_path, str) and _AUTHOR_PATH_RE.match(author_path):
        target["author_url"] = author_path
    if lines:
        target["author_name"] = lines[0]
    text = item.get("text")
    if isinstance(text, str) and text:
        target["excerpt"] = text[:300]
    return target


def build_comment_references(
    items: list[Any],
    *,
    include_replies: bool,
    sort: CommentSort,
    max_comments: int,
) -> list[CommentReference]:
    """Turn the browser's comment read into ``references["comments"]``."""
    parsed: list[tuple[CommentUrn, CommentUrn | None, dict[str, Any]]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        comment = _unit_urn(item)
        if comment is None or comment.comment_id in seen:
            continue
        parent = None
        if item.get("parentId") is not None:
            parent = _unit_urn(
                {
                    "kind": item.get("parentKind"),
                    "threadId": item.get("parentThreadId"),
                    "id": item.get("parentId"),
                }
            )
            if parent is None:
                continue
        if parent is not None and not include_replies:
            continue
        seen.add(comment.comment_id)
        parsed.append((comment, parent, item))

    if sort == "recent":
        # Top-level comments newest first, each followed by its replies newest
        # first. Replies whose parent is not in the read stay at the end.
        def key(entry: tuple[CommentUrn, CommentUrn | None, dict[str, Any]]) -> int:
            return int(entry[0].comment_id)

        tops = sorted((e for e in parsed if e[1] is None), key=key, reverse=True)
        top_ids = {entry[0].comment_id for entry in tops}
        ordered: list[tuple[CommentUrn, CommentUrn | None, dict[str, Any]]] = []
        for top in tops:
            ordered.append(top)
            ordered.extend(
                sorted(
                    (
                        e
                        for e in parsed
                        if e[1] is not None and e[1].comment_id == top[0].comment_id
                    ),
                    key=key,
                    reverse=True,
                )
            )
        ordered.extend(
            sorted(
                (
                    e
                    for e in parsed
                    if e[1] is not None and e[1].comment_id not in top_ids
                ),
                key=key,
                reverse=True,
            )
        )
        parsed = ordered

    references: list[CommentReference] = []
    for comment, parent, item in parsed[:max_comments]:
        permalink = comment_permalink(comment, parent)
        author_path = item.get("authorPath")
        reference: CommentReference = {
            "kind": "comment",
            "url": author_path
            if isinstance(author_path, str) and _AUTHOR_PATH_RE.match(author_path)
            else permalink,
            "value": comment.urn,
            "permalink": permalink,
        }
        lines = _clean_lines(item.get("authorLines"))
        if lines:
            reference["text"] = lines[0][:120]
            if len(lines) > 1:
                reference["context"] = " · ".join(lines[1:4])[:240]
        if parent is not None:
            reference["parent"] = parent.urn
        text = item.get("text")
        if isinstance(text, str) and text:
            reference["excerpt"] = text[:300]
        posted_at = posted_at_from_id(comment.comment_id)
        if posted_at is not None:
            reference["posted_at"] = posted_at
        references.append(reference)
    return references


class CommentScraper:
    """Read and write comments on one post through the browser UI."""

    def __init__(
        self,
        session: ScrapingSession,
        navigator: PageNavigator,
        content: PageContentReader,
    ):
        self._session = session
        self._navigator = navigator
        self._content = content

    @property
    def _page(self) -> Any:
        return self._session.page

    # -- shared page lifecycle ------------------------------------------------

    async def _open_post(self, url: str) -> None:
        page = self._page
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
                    return !!main && (
                        main.querySelector('[contenteditable]:not([contenteditable="false"])') ||
                        main.querySelector('[data-id*="comment:("], [data-urn*="comment:("]') ||
                        main.innerText.length > 400
                    );
                }""",
                timeout=_CONTENT_WAIT_MS,
            )
        except PlaywrightTimeoutError:
            logger.debug("Post content did not settle on %s", url)

    def _landed_on_post(self, post_urn: str) -> bool:
        """Whether the page is still a post route that does not name another post."""
        try:
            parsed = urlparse(self._page.url)
        except Exception:
            return False
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not (
            host == "linkedin.com" or host.endswith(".linkedin.com")
        ):
            return False
        if parsed.path.startswith("/posts/"):
            return True
        match = _POST_ROUTE_URN_RE.match(parsed.path)
        if match is None:
            return False
        landed = match.group(1)
        landed_kind = landed.split(":")[2]
        wanted_kind = post_urn.split(":")[2]
        return landed == post_urn or landed_kind != wanted_kind

    async def _cleanup(self, token: str) -> None:
        with anyio.move_on_after(_CLEANUP_TIMEOUT_SECONDS, shield=True) as scope:
            try:
                await self._page.evaluate(_CLEANUP_JS, token)
            except Exception:
                logger.debug("Could not clear comment markers", exc_info=True)
        if scope.cancel_called:
            logger.warning("Timed out clearing comment markers")

    async def _expand(
        self,
        *,
        max_comments: int,
        include_replies: bool,
        token: str,
        target_id: str | None = None,
    ) -> int:
        """Load more comments with a bounded number of structural clicks.

        Returns how many comments carry a URN once expansion stops.
        """
        page = self._page
        viewport = page.viewport_size or {"width": 1280, "height": 720}
        try:
            await page.mouse.move(viewport["width"] // 2, viewport["height"] // 2)
        except Exception:
            logger.debug("Could not move the mouse over the post", exc_info=True)

        clicks = 0
        stale = 0
        count = 0
        for _round in range(_EXPANSION_CLICK_BUDGET + _EXPANSION_STALE_ROUNDS + 1):
            scan = await page.evaluate(
                _EXPANSION_SCAN_JS,
                {
                    "includeReplies": include_replies,
                    "token": token,
                    "targetId": target_id,
                },
            )
            if not isinstance(scan, dict):
                break
            count = int(scan.get("count") or 0)
            if target_id is not None and int(scan.get("targetCount") or 0) > 0:
                break
            if target_id is None and count >= max_comments:
                break
            if clicks >= _EXPANSION_CLICK_BUDGET or stale >= _EXPANSION_STALE_ROUNDS:
                break

            clicked = False
            if int(scan.get("loaders") or 0) > 0:
                try:
                    await page.locator(f'[data-linkedin-mcp-loader="{token}"]').click(
                        timeout=_CLICK_TIMEOUT_MS
                    )
                    clicked = True
                    clicks += 1
                except Exception:
                    logger.debug("Comment loader click failed", exc_info=True)
            if not clicked:
                try:
                    await page.mouse.wheel(0, _WHEEL_DELTA)
                except Exception:
                    logger.debug("Wheel scroll over the post failed", exc_info=True)
            await self._session.delay(
                _EXPANSION_SETTLE_SECONDS + _random.uniform(0.0, 0.6)
            )

            # Its own token suffix, so this read cannot move the mark off the
            # button just clicked before that button is judged below.
            after = await page.evaluate(
                _EXPANSION_SCAN_JS,
                {
                    "includeReplies": include_replies,
                    "token": f"{token}-after",
                    "targetId": target_id,
                },
            )
            new_count = int(after.get("count") or 0) if isinstance(after, dict) else 0
            if new_count > count:
                stale = 0
            else:
                stale += 1
                if clicked:
                    await page.evaluate(_EXPANSION_DEAD_JS, token)
            count = new_count
        return count

    # -- read -------------------------------------------------------------------

    async def get_post_comments(
        self,
        post_url: str,
        max_comments: int = 50,
        include_replies: bool = True,
        sort: str = "relevant",
    ) -> dict[str, Any]:
        """Read a post's comments with structured ids for each one."""
        if sort not in ("relevant", "recent"):
            raise InvalidReferenceError('sort must be "relevant" or "recent".')
        if not 1 <= max_comments <= MAX_COMMENTS_LIMIT:
            raise InvalidReferenceError(
                f"max_comments must be between 1 and {MAX_COMMENTS_LIMIT}."
            )
        post_urn = normalize_post_urn(post_url)
        url = post_update_url(post_urn)
        token = _token()
        sections: dict[str, str] = {}
        references: dict[str, list[CommentReference]] = {}
        section_errors: dict[str, dict[str, Any]] = {}
        try:
            await self._open_post(url)
            await self._expand(
                max_comments=max_comments,
                include_replies=include_replies,
                token=token,
            )
            items = await self._page.evaluate(_READ_COMMENTS_JS)
            raw_result = await self._content._extract_root_content(["main"])
            raw = raw_result["text"]
            truncated = truncate_linkedin_noise(raw) if raw else ""
            if raw and not truncated and raw.strip():
                section_errors["comments"] = rate_limited_section_error()
            elif truncated:
                sections["comments"] = filter_linkedin_noise_lines(truncated)
            refs = build_comment_references(
                items if isinstance(items, list) else [],
                include_replies=include_replies,
                sort="recent" if sort == "recent" else "relevant",
                max_comments=max_comments,
            )
            if refs:
                references["comments"] = refs
        except LinkedInScraperException:
            raise
        except Exception as error:
            logger.warning("Failed to read comments on %s: %s", url, error)
            section_errors["comments"] = build_issue_diagnostics(
                error, context="get_post_comments", target_url=url
            )
        finally:
            await self._cleanup(token)

        result: dict[str, Any] = {"url": url, "sections": sections}
        if references:
            result["references"] = references
        if section_errors:
            result["section_errors"] = section_errors
        return result

    # -- writes -----------------------------------------------------------------

    async def comment_on_post(
        self, post_url: str, text: str, *, confirm: bool
    ) -> dict[str, Any]:
        """Post a top-level comment, confirmed by the new comment appearing."""
        return await self._write(post_url, None, text, confirm=confirm)

    async def reply_to_comment(
        self, post_url: str, comment_urn: str, text: str, *, confirm: bool
    ) -> dict[str, Any]:
        """Reply to one comment, located and verified by its URN."""
        return await self._write(post_url, comment_urn, text, confirm=confirm)

    async def _write(
        self,
        post_url: str,
        comment_urn: str | None,
        text: str,
        *,
        confirm: bool,
    ) -> dict[str, Any]:
        answer = prepare_comment_write(post_url, comment_urn, text, confirm=confirm)
        if answer is not None:
            return answer
        post_urn = normalize_post_urn(post_url)
        target = normalize_comment_urn(comment_urn) if comment_urn is not None else None
        normalized, _reason = normalize_comment_text(text)
        assert normalized is not None  # prepare_comment_write refused anything else
        action = "reply" if target is not None else "comment"
        common: dict[str, Any] = {
            "post_urn": post_urn,
            "comment_urn": target.urn if target is not None else None,
        }
        progress = _WriteProgress(token=_token())
        try:
            return await self._write_confirmed(
                post_update_url(post_urn),
                post_urn,
                target,
                normalized,
                action,
                common,
                progress,
            )
        except Exception:
            if not progress.submitted:
                # Nothing can have been posted, so the error itself is the answer
                # and the caller may retry on it.
                raise
            logger.debug("Comment %s failed after submission", action, exc_info=True)
            return comment_action_result(
                self._page.url,
                "unconfirmed",
                f"The {action} may already have been submitted and LinkedIn did not "
                "confirm it. Read the post with get_post_comments before retrying; "
                "retrying may post it twice.",
                retry_safe=False,
                **common,
            )
        except BaseException:
            # Cancellation only. FastMCP discards a cancelled scope's result, so
            # after a possible submission the log line is all that is left.
            if progress.submitted:
                logger.warning(COMMENT_INTERRUPTED_WARNING)
            raise
        finally:
            if progress.typed and not progress.submitted:
                await self._clear_typed(progress)
            await self._cleanup(progress.token)

    async def _write_confirmed(
        self,
        url: str,
        post_urn: str,
        target: CommentUrn | None,
        text: str,
        action: str,
        common: dict[str, Any],
        progress: _WriteProgress,
    ) -> dict[str, Any]:
        page = self._page
        token = progress.token
        await self._open_post(url)
        if not self._landed_on_post(post_urn):
            return comment_action_result(
                page.url,
                "post_unavailable",
                "LinkedIn did not open the requested post. Nothing was typed.",
                **common,
            )

        target_arg: dict[str, Any] = {"id": None, "token": token}
        described: dict[str, Any] | None = None
        if target is not None:
            target_arg = {
                "id": target.comment_id,
                "kind": target.thread_kind,
                "threadId": target.thread_id,
                "token": token,
            }
            await self._expand(
                max_comments=MAX_COMMENTS_LIMIT,
                include_replies=True,
                token=token,
                target_id=target.comment_id,
            )
            located = await page.evaluate(_LOCATE_COMMENT_JS, target_arg)
            status = located.get("status") if isinstance(located, dict) else None
            if status != "found" or not isinstance(located, dict):
                return comment_action_result(
                    page.url,
                    "comment_ambiguous"
                    if status in {"ambiguous", "contradiction"}
                    else "comment_not_found",
                    "The comment could not be located as exactly one comment on "
                    "this post. Nothing was typed.",
                    **common,
                )
            described = located

        prefill = await self._open_editor(target_arg, token)
        if isinstance(prefill, dict):
            return comment_action_result(
                page.url,
                prefill["status"],
                prefill["message"],
                target=_describe_target(described) if described else None,
                **common,
            )
        progress.target_arg = target_arg
        progress.prefill = prefill
        progress.text = text

        focused = await page.evaluate(
            _EDITOR_FOCUS_JS, {**target_arg, "prefill": prefill}
        )
        focus_status = focused.get("status") if isinstance(focused, dict) else None
        if focus_status != "ready":
            return comment_action_result(
                page.url,
                "composer_occupied"
                if focus_status == "occupied"
                else "editor_unavailable",
                "The comment box already holds a draft; it was left untouched."
                if focus_status == "occupied"
                else "The comment box could not be verified and focused. Nothing was "
                "typed.",
                **common,
            )

        progress.typed = True
        await self._type(
            page.locator(f'[data-linkedin-mcp-comment-editor="{token}"]'), text
        )
        submit = await self._resolve_submit(
            {**target_arg, "prefill": prefill, "expected": text}
        )
        if submit != "ready":
            return comment_action_result(
                page.url,
                "typing_mismatch" if submit == "mismatch" else "submit_unavailable",
                "The typed text did not match what was requested; nothing was posted."
                if submit == "mismatch"
                else "No single submit button belongs to the verified comment box; "
                "nothing was posted.",
                **common,
            )
        if not self._landed_on_post(post_urn):
            return comment_action_result(
                page.url,
                "post_unavailable",
                "The page left the post before submission. Nothing was posted.",
                **common,
            )
        baseline = await page.evaluate(_COMMENT_IDS_JS)

        # A click can dispatch before the round trip reports an error, so from
        # here on every failure is an unknown outcome rather than a refusal.
        progress.submitted = True
        await page.locator(f'[data-linkedin-mcp-submit="{token}"]').click(
            timeout=_CLICK_TIMEOUT_MS
        )

        parents: list[str | None] = [None]
        if target is not None:
            parents = [target.comment_id]
            parent_id = described.get("parentId") if described else None
            if isinstance(parent_id, str):
                parents.append(parent_id)
        confirmed = await self._wait_for_posted(
            baseline if isinstance(baseline, list) else [], parents, text
        )
        posted = _unit_urn(confirmed) if confirmed is not None else None
        if confirmed is None or posted is None:
            return comment_action_result(
                page.url,
                "unconfirmed",
                f"The {action} was submitted but did not appear under the expected "
                "comment in time. Read the post with get_post_comments before "
                "retrying; retrying may post it twice.",
                retry_safe=False,
                **common,
            )
        parent = (
            _unit_urn(
                {
                    "kind": confirmed.get("parentKind"),
                    "threadId": confirmed.get("parentThreadId"),
                    "id": confirmed.get("parentId"),
                }
            )
            if confirmed.get("parentId") is not None
            else None
        )
        return comment_action_result(
            f"https://www.linkedin.com{comment_permalink(posted, parent)}",
            "posted",
            f"The {action} was submitted and appeared on the post.",
            posted_comment_urn=posted.urn,
            text=text,
            target=_describe_target(described) if described else None,
            posted=True,
            retry_safe=False,
            **common,
        )

    async def _open_editor(
        self, target_arg: dict[str, Any], token: str
    ) -> str | dict[str, str]:
        """Pin the editor the text goes into; returns its prefill or a refusal."""
        page = self._page
        if target_arg["id"] is None:
            state = await page.evaluate(_POST_COMPOSER_JS, {"token": token})
            status = state.get("status") if isinstance(state, dict) else None
            if status == "ready":
                return ""
            if status == "occupied":
                return {
                    "status": "composer_occupied",
                    "message": "The post's comment box already holds a draft. It was "
                    "left untouched.",
                }
            return {
                "status": "comment_box_unavailable",
                "message": "The post did not expose exactly one comment box.",
            }

        prepared = await page.evaluate(
            _REPLY_PREPARE_JS, {**target_arg, "maxCandidates": _MAX_REPLY_CANDIDATES}
        )
        status = prepared.get("status") if isinstance(prepared, dict) else None
        if status == "ready":
            return str(prepared.get("prefill") or "")
        if status == "occupied":
            return {
                "status": "composer_occupied",
                "message": "The reply box under this comment already holds a draft. It "
                "was left untouched.",
            }
        if status != "candidates" or not isinstance(prepared, dict):
            return {
                "status": "reply_box_unavailable",
                "message": "The comment's reply action could not be identified. "
                "Nothing was typed.",
            }
        dialogs = int(prepared.get("dialogs") or 0)
        for index in range(int(prepared.get("candidates") or 0)):
            candidate = page.locator(
                f'[data-linkedin-mcp-reply-candidate="{token}-{index}"]'
            )
            try:
                await candidate.click(timeout=_CLICK_TIMEOUT_MS)
            except Exception:
                logger.debug("Reply candidate click failed", exc_info=True)
                continue
            outcome = await self._wait_for_reply_editor(
                {**target_arg, "dialogs": dialogs}
            )
            if outcome.get("status") == "opened":
                return str(outcome.get("prefill") or "")
            if outcome.get("status") == "dialog":
                try:
                    await page.keyboard.press("Escape")
                except Exception:
                    logger.debug("Could not close a dialog", exc_info=True)
                continue
            if outcome.get("status") != "waiting":
                break
        return {
            "status": "reply_box_unavailable",
            "message": "No reply box opened under exactly this comment. Nothing was "
            "typed.",
        }

    async def _wait_for_reply_editor(self, arg: dict[str, Any]) -> dict[str, Any]:
        deadline = self._session.monotonic() + _REPLY_BOX_WAIT_SECONDS
        state: dict[str, Any] = {"status": "waiting"}
        while True:
            read = await self._page.evaluate(_REPLY_EDITOR_JS, arg)
            state = read if isinstance(read, dict) else {"status": "invalid"}
            if state.get("status") != "waiting":
                return state
            if self._session.monotonic() >= deadline:
                return state
            await self._session.delay(_POLL_SECONDS)

    async def _type(self, editor: Any, text: str) -> None:
        """Type with small randomised per-key delays; line breaks as Shift+Enter."""
        for index, line in enumerate(text.split("\n")):
            if index:
                await editor.press("Shift+Enter", delay=_random.randint(20, 60))
            for chunk in re.findall(r"\S+|\s+", line):
                await editor.press_sequentially(chunk, delay=_random.randint(18, 55))
                await self._session.delay(_random.uniform(0.02, 0.12))

    async def _resolve_submit(self, arg: dict[str, Any]) -> str:
        deadline = self._session.monotonic() + _SUBMIT_READY_WAIT_SECONDS
        while True:
            read = await self._page.evaluate(_SUBMIT_RESOLVE_JS, arg)
            status = read.get("status") if isinstance(read, dict) else "invalid"
            if status != "disabled" or self._session.monotonic() >= deadline:
                return str(status)
            await self._session.delay(_POLL_SECONDS)

    async def _clear_typed(self, progress: _WriteProgress) -> None:
        """Best-effort removal of text this call typed and never submitted.

        Only text that is still exactly a prefix of what was typed, behind the
        box's own prefill, is removed; anything else belongs to someone else.
        Shielded and bounded, because it also runs while a cancellation unwinds.
        """
        if progress.target_arg is None:
            return
        with anyio.move_on_after(_CLEANUP_TIMEOUT_SECONDS, shield=True) as scope:
            try:
                await self._page.evaluate(
                    _EDITOR_CLEANUP_JS,
                    {
                        **progress.target_arg,
                        "prefill": progress.prefill,
                        "expected": progress.text,
                    },
                )
            except Exception:
                logger.debug("Could not remove typed comment text", exc_info=True)
        if scope.cancel_called:
            logger.warning("Timed out removing typed comment text")

    async def _wait_for_posted(
        self, baseline: list[Any], parents: list[str | None], text: str
    ) -> dict[str, Any] | None:
        deadline = self._session.monotonic() + _CONFIRMATION_WAIT_SECONDS
        arg = {"baseline": baseline, "parents": parents, "expected": text}
        while True:
            read = await self._page.evaluate(_CONFIRM_POSTED_JS, arg)
            status = read.get("status") if isinstance(read, dict) else "invalid"
            if status == "confirmed" and isinstance(read, dict):
                return read
            if status == "ambiguous" or self._session.monotonic() >= deadline:
                return None
            await self._session.delay(_POLL_SECONDS)

    # -- reactions ----------------------------------------------------------------

    async def react_to_comment(
        self,
        post_url: str,
        comment_urn: str,
        reaction: str = "like",
        *,
        confirm: bool,
    ) -> dict[str, Any]:
        """Like one comment through its own reaction toggle."""
        answer = prepare_comment_reaction(
            post_url, comment_urn, reaction, confirm=confirm
        )
        if answer is not None:
            return answer
        post_urn = normalize_post_urn(post_url)
        url = post_update_url(post_urn)
        target = normalize_comment_urn(comment_urn)
        common: dict[str, Any] = {"post_urn": post_urn, "comment_urn": target.urn}
        token = _token()
        arg = {
            "id": target.comment_id,
            "kind": target.thread_kind,
            "threadId": target.thread_id,
            "token": token,
        }
        page = self._page
        try:
            await self._open_post(url)
            if not self._landed_on_post(post_urn):
                return comment_action_result(
                    page.url,
                    "post_unavailable",
                    "LinkedIn did not open the requested post.",
                    **common,
                )
            await self._expand(
                max_comments=MAX_COMMENTS_LIMIT,
                include_replies=True,
                token=token,
                target_id=target.comment_id,
            )
            before = await page.evaluate(_REACTION_JS, {**arg, "mark": True})
            status = before.get("status") if isinstance(before, dict) else None
            if status in {"missing", "ambiguous", "contradiction"}:
                return comment_action_result(
                    page.url,
                    "comment_not_found" if status == "missing" else "comment_ambiguous",
                    "The comment could not be located as exactly one comment on this "
                    "post. Nothing was changed.",
                    **common,
                )
            if (
                status != "found"
                or not isinstance(before, dict)
                or before.get("disabled")
            ):
                return comment_action_result(
                    page.url,
                    "reaction_unavailable",
                    "The comment did not expose exactly one usable reaction toggle.",
                    **common,
                )
            if before.get("pressed") == "true":
                return comment_action_result(
                    page.url,
                    "already_reacted",
                    "The comment already carries your reaction. Nothing was changed.",
                    **common,
                )
            if before.get("pressed") != "false":
                return comment_action_result(
                    page.url,
                    "reaction_unavailable",
                    "The reaction toggle did not report a readable state.",
                    **common,
                )
            try:
                await page.locator(f'[data-linkedin-mcp-react="{token}"]').click(
                    timeout=_CLICK_TIMEOUT_MS
                )
            except Exception:
                logger.debug("Reaction click failed", exc_info=True)
                return comment_action_result(
                    page.url,
                    "unconfirmed",
                    "The reaction click did not complete. Read the post before "
                    "retrying.",
                    retry_safe=False,
                    **common,
                )
            deadline = self._session.monotonic() + _REACTION_WAIT_SECONDS
            while True:
                after = await page.evaluate(_REACTION_JS, {**arg, "mark": False})
                if (
                    isinstance(after, dict)
                    and after.get("status") == "found"
                    and after.get("pressed") == "true"
                ):
                    return comment_action_result(
                        page.url,
                        "reacted",
                        "The comment's reaction toggle now reports pressed.",
                        posted=True,
                        retry_safe=False,
                        **common,
                    )
                if self._session.monotonic() >= deadline:
                    return comment_action_result(
                        page.url,
                        "unconfirmed",
                        "The reaction was clicked but the toggle did not report "
                        "pressed in time. Read the post before retrying.",
                        retry_safe=False,
                        **common,
                    )
                await self._session.delay(_POLL_SECONDS)
        finally:
            await self._cleanup(token)


@dataclass(slots=True)
class _WriteProgress:
    """How far one write got, so every exit knows what it may have left behind."""

    token: str
    target_arg: dict[str, Any] | None = None
    prefill: str = ""
    text: str = ""
    typed: bool = False
    submitted: bool = False
