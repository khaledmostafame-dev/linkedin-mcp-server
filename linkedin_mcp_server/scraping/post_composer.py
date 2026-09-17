"""Drive LinkedIn's share composer: publish, schedule, list and delete posts.

The composer renders inside an open shadow root (LinkedIn's SDUI interop
outlet). Patchright locators pierce open shadow roots, and ``locator.evaluate``
hands the resolved element to the page script, so element-scoped JavaScript
works there. A bare ``page.evaluate`` with ``querySelectorAll`` does not; the
few page-level programs below walk shadow roots explicitly. (Measured upstream
in stickerdaniel/linkedin-mcp-server PRs 690, 692, 696 and 835, whose selectors
this module ports.)

Every identification is structural and locale-independent: roles, attribute
presence, LinkedIn's own ``data-test-icon`` hook names, semantic element ids,
URL and URN patterns. Text is compared only against text this module typed.

Ordering matters and was measured upstream: the schedule round-trip rebuilds
the editor and drops typed text, so every dialog round-trip (visibility,
attachments, schedule) happens before any text is typed. And a composer that
opens holding a restored draft is refused, never cleared: that draft belongs to
whoever wrote it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import logging
import random
import re
import secrets
import time
from typing import Any

from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.post_content import (
    MentionSegment,
    MentionTarget,
    PostAttachment,
    PollSpec,
    PostEdit,
    PostRequest,
    TextSegment,
    date_matches,
    format_schedule_date,
    format_schedule_time,
    identity_key_from_url,
    identity_key_from_urn,
    schedule_summary,
    time_matches,
)
from linkedin_mcp_server.scraping.session import ScrapingSession

logger = logging.getLogger(__name__)

SHARE_URL = "https://www.linkedin.com/feed/?shareActive=true"
FEED_URL = "https://www.linkedin.com/feed/"

_EDITOR_SELECTOR = 'div[role="textbox"][contenteditable="true"]'
_DIALOG_SELECTOR = 'dialog[open], [role="dialog"]'
_DISMISS_SELECTOR = "button[data-test-modal-close-btn]"
_UNLABELLED_BUTTON_SELECTOR = "button:not([aria-label])"
_FILE_INPUT_SELECTOR = 'input[type="file"]'
_PROGRESS_SELECTOR = '[role="progressbar"]'
_LISTBOX_SELECTOR = '[role="listbox"]'
_OPTION_SELECTOR = '[role="listbox"] [role="option"]'
_SCHEDULE_DATE_SELECTOR = "input#share-post__scheduled-date"
_SCHEDULE_TIME_SELECTOR = "input#share-post__scheduled-time"
_TEXT_INPUT_SELECTOR = 'input[type="text"], input:not([type])'


def _icon_button(*icons: str, role: str = "button") -> str:
    """An element carrying one of LinkedIn's icon test hooks (names, not text).

    Live capture 2026-09-17 (`005-extractor-after-goto.json`, `/sharing/
    compose`) found neither of the two variants below on the current
    composer: no element anywhere on the page carries `data-test-icon`, and
    there are no `<use>` elements at all. LinkedIn now renders each icon as
    an inline `<svg id="icon-name">` -- the same capture lists `clock-medium`
    and `arrow-right-small` by id among 90 such svgs -- so that is added as a
    third variant rather than replacing the other two, in case an older
    LinkedIn surface still uses them.
    """
    parts: list[str] = []
    for icon in icons:
        parts.append(f'{role}:has(svg[data-test-icon="{icon}"])')
        parts.append(f'{role}:has(use[href="#{icon}"])')
        parts.append(f'{role}:has(svg[id="{icon}"])')
    return ", ".join(parts)


# Measured upstream (PRs 690, 692, 696); the `data-test-icon`/`use[href]`
# hooks they measured are gone on the current live composer (see
# `_icon_button`'s docstring). The same capture also found the schedule
# button's `clock-medium` icon wrapped not in a `<button>` but in an
# `<a href="/sharing/compose" aria-haspopup="dialog" aria-expanded="false">`
# (capture node 93/95) -- a real link, not a button element -- so both
# selectors below now match either wrapper tag. `arrow-right-small` (the
# "view all scheduled" arrow) is confirmed present on the page by id, but
# only outside the schedule mini-dialog (an unrelated "see all" link at
# node 1069/1071); that sub-dialog only exists after clicking the schedule
# button, so its own markup is unverified -- a capture of the DOM
# immediately after that click is the next evidence needed if this selector
# still misses live.
_SCHEDULE_BUTTON_SELECTOR = ", ".join(
    [
        _icon_button("clock-medium", role="button"),
        _icon_button("clock-medium", role="a"),
    ]
)
_VIEW_ALL_SCHEDULED_SELECTOR = ", ".join(
    [
        _icon_button("arrow-right-small", role="button"),
        _icon_button("arrow-right-small", role="a"),
    ]
)
_MENU_DELETE_SELECTOR = _icon_button("trash-medium", role='[role="button"]')
_MENU_EDIT_SELECTOR = _icon_button("edit-medium", role='[role="button"]')
# Post settings (upstream PR 835): the author row is ``#ACTOR`` and the
# "Posting as" list is a radio group. Ids and roles, not labels.
_SETTINGS_HEADER_SELECTOR = "#share-to-linkedin-modal__header"
_ACTOR_ROW_SELECTOR = "#ACTOR"
_ACTOR_RADIO_SELECTOR = '[role="radiogroup"] [role="radio"]'
# A post's own control menu: an overflow-icon button. Its menu items are
# div[role=button] (upstream PR 696) or menuitems carrying icon test hooks.
_OVERFLOW_BUTTON_SELECTOR = (
    'button:has(svg[data-test-icon*="overflow"]), button:has(use[href*="overflow"])'
)
_OWNER_DELETE_ITEM_SELECTOR = ", ".join(
    [
        _icon_button("trash-medium", role='[role="button"]'),
        _icon_button("trash-medium", role='[role="menuitem"]'),
    ]
)
_OWNER_EDIT_ITEM_SELECTOR = ", ".join(
    [
        _icon_button("edit-medium", role='[role="button"]'),
        _icon_button("edit-medium", role='[role="menuitem"]'),
    ]
)
# Not yet measured against a live account; each is a table of candidate icon
# hook names so a live check only has to extend a tuple. A miss fails closed.
_IMAGE_BUTTON_SELECTOR = _icon_button("image-medium", "photo-medium")
_MORE_BUTTON_SELECTOR = _icon_button(
    "add-medium", "plus-medium", "overflow-web-ios-medium", "overflow-medium"
)
_DOCUMENT_BUTTON_SELECTOR = ", ".join(
    [
        _icon_button("document-medium", "file-medium", "doc-medium"),
        _icon_button(
            "document-medium", "file-medium", "doc-medium", role='[role="button"]'
        ),
    ]
)
# Poll: unmeasured candidate icon hooks, looked for in the composer and then
# in its "more" menu. A miss fails closed.
_POLL_BUTTON_SELECTOR = ", ".join(
    [
        _icon_button("poll-medium", "poll-small", "survey-medium"),
        _icon_button(
            "poll-medium", "poll-small", "survey-medium", role='[role="button"]'
        ),
    ]
)
_ADD_OPTION_BUTTON_SELECTOR = _icon_button(
    "add-small", "add-medium", "plus-small", "plus-medium"
)
_POLL_FIELD_SELECTOR = 'textarea, input[type="text"], input:not([type])'
# Duration <select> option values, when LinkedIn uses enum values. Otherwise
# the four options are taken in their fixed order (1 day, 3 days, 1 week,
# 2 weeks): a structural assumption, listed for live verification.
_POLL_DURATION_VALUES: dict[int, tuple[str, ...]] = {
    1: ("ONE_DAY", "1", "P1D"),
    3: ("THREE_DAYS", "3", "P3D"),
    7: ("ONE_WEEK", "SEVEN_DAYS", "7", "P7D", "P1W"),
    14: ("TWO_WEEKS", "FOURTEEN_DAYS", "14", "P14D", "P2W"),
}
_POLL_DURATION_ORDER: tuple[int, ...] = (1, 3, 7, 14)
_SELECT_STATE_JS = """(select) => ({
    values: Array.from(select.options).map(option => option.value),
    selectedIndex: select.selectedIndex,
})"""

# Visibility option ids: LinkedIn's share settings use upper-case enum ids for
# its rows (``#ACTOR`` was measured upstream). Enum names are not translated.
_VISIBILITY_OPTION_SELECTORS: dict[str, str] = {
    "anyone": '[id="ANYONE"], input[type="radio"][value="ANYONE"]',
    "connections": (
        '[id="CONNECTIONS_ONLY"], input[type="radio"][value="CONNECTIONS_ONLY"]'
    ),
}

_OPEN_TIMEOUT_MS = 15_000
_STEP_TIMEOUT_MS = 8_000
_TYPEAHEAD_TIMEOUT_SECONDS = 8.0
_UPLOAD_TIMEOUT_SECONDS = 90.0
_PUBLISH_TIMEOUT_MS = 20_000
_POST_LINK_TIMEOUT_SECONDS = 8.0

# Identity evidence carried by elements: every href, and every attribute value
# that contains a LinkedIn URN. Returned raw; Python normalizes and decides.
_EVIDENCE_JS = r"""
    const URN = /urn:li:[a-z_]+:[A-Za-z0-9_-]+/g;
    const visible = element => {
        const style = element && getComputedStyle(element);
        return !!(
            element && style.visibility !== 'hidden' && style.display !== 'none' &&
            (element.offsetWidth || element.offsetHeight || element.getClientRects().length)
        );
    };
    const evidence = element => {
        const hrefs = [];
        const urns = [];
        for (const node of [element, ...element.querySelectorAll('*')]) {
            for (const attribute of Array.from(node.attributes || [])) {
                if (attribute.name.startsWith('data-linkedin-mcp')) continue;
                if (attribute.name === 'href') hrefs.push(attribute.value);
                for (const match of attribute.value.match(URN) || []) urns.push(match);
            }
        }
        return {hrefs, urns};
    };
"""

_OPTION_EVIDENCE_JS = (
    "(options) => {"
    + _EVIDENCE_JS
    + """
        return options.map(option => ({
            visible: visible(option),
            ...evidence(option),
        }));
    }"""
)

_ELEMENT_EVIDENCE_JS = "(element) => {" + _EVIDENCE_JS + "return evidence(element); }"

_OPTION_MARK_JS = (
    "(options, arg) => {"
    + _EVIDENCE_JS
    + """
        const option = options[arg.index];
        if (!option || !visible(option)) return null;
        option.setAttribute('data-linkedin-mcp-mention', arg.token);
        return evidence(option);
    }"""
)

# Entities inside the editor: nodes that carry a link, a URN, or that the
# editor made non-editable. Outermost only, in document order.
_EDITOR_STATE_JS = (
    "(editor) => {"
    + _EVIDENCE_JS
    + """
        const candidates = Array.from(editor.querySelectorAll(
            'a[href], [contenteditable="false"], [data-entity-urn], [data-urn]'
        ));
        const outermost = candidates.filter(node =>
            !candidates.some(other => other !== node && other.contains(node))
        );
        const root = editor.getRootNode();
        return {
            text: editor.innerText || editor.textContent || '',
            focused: root.activeElement === editor || editor.contains(root.activeElement),
            entities: outermost.map(node => ({
                text: (node.innerText || node.textContent || '').trim(),
                ...evidence(node),
            })),
        };
    }"""
)

_RADIO_EVIDENCE_JS = (
    "(radios) => {"
    + _EVIDENCE_JS
    + """
        return radios.map(radio => ({
            visible: visible(radio),
            checked: (radio.getAttribute('aria-checked') || '').toLowerCase() === 'true',
            ...evidence(radio),
        }));
    }"""
)

_CHECKED_JS = """(element) => {
    const radio = element.matches('input') ? element : element.querySelector('input[type="radio"]');
    return (element.getAttribute('aria-checked') || '').toLowerCase() === 'true' ||
        (radio !== null && radio.checked === true);
}"""

_MAXLENGTH_JS = "(input) => input.maxLength"

# The browser converts the instant itself, so the wall time typed into the
# schedule dialog is exactly what that browser's timezone calls it.
_BROWSER_LOCAL_TIME_JS = """(epochMs) => {
    const d = new Date(epochMs);
    return {
        timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone || null,
        year: d.getFullYear(), month: d.getMonth() + 1, day: d.getDate(),
        hour: d.getHours(), minute: d.getMinutes(),
        todayYear: new Date().getFullYear(), todayMonth: new Date().getMonth() + 1,
        todayDay: new Date().getDate(),
    };
}"""

# Page-level walk into open shadow roots.
_SHADOW_WALK_JS = r"""
    const roots = [];
    const collect = root => {
        roots.push(root);
        for (const element of root.querySelectorAll('*')) {
            if (element.shadowRoot) collect(element.shadowRoot);
        }
    };
    collect(document);
    const visible = element => {
        const style = element && getComputedStyle(element);
        return !!(
            element && style.visibility !== 'hidden' && style.display !== 'none' &&
            (element.offsetWidth || element.offsetHeight || element.getClientRects().length)
        );
    };
"""

# The post a /feed/update/<urn>/ page shows: the first visible profile or
# company link in <main> is its actor (header precedes body and comments), and
# the first overflow button is its own control menu. Returned raw.
_POST_OWNERSHIP_JS = (
    "(urn) => {"
    + _SHADOW_WALK_JS
    + """
        const main = roots.map(root => root.querySelector('main')).find(Boolean);
        if (!main) return null;
        const actor = Array.from(main.querySelectorAll('a[href*="/in/"], a[href*="/company/"]'))
            .find(visible);
        const controls = Array.from(main.querySelectorAll('button')).filter(button =>
            visible(button) &&
            Array.from(button.querySelectorAll('svg, use')).some(node =>
                (node.getAttribute('data-test-icon') || node.getAttribute('href') || '')
                    .includes('overflow')));
        const text = main.innerText || '';
        const domUrn = Array.from(main.querySelectorAll('*')).some(node =>
            Array.from(node.attributes || []).some(a => a.value.includes(urn))
        );
        return {
            actorHref: actor ? actor.getAttribute('href') : null,
            controlCount: controls.length,
            domUrn,
            text: text.slice(0, 2000),
        };
    }"""
)

_POST_LINKS_JS = (
    "() => {"
    + _SHADOW_WALK_JS
    + r"""
        const PATTERN = /\/feed\/update\/(urn:li:(?:activity|share|ugcPost):\d+)/;
        const all = [];
        const announced = [];
        for (const root of roots) {
            for (const anchor of root.querySelectorAll('a[href*="/feed/update/"]')) {
                const match = PATTERN.exec(anchor.getAttribute('href') || '');
                if (!match) continue;
                all.push(match[1]);
                if (anchor.closest('[role="alert"], [aria-live]')) announced.push(match[1]);
            }
        }
        return {all, announced};
    }"""
)

# One entry per visible overflow-menu button in a visible dialog; the entry's
# text is the largest ancestor still holding a single entry (upstream PR 696).
_SCHEDULED_ENTRIES_BODY_JS = (
    _SHADOW_WALK_JS
    + r"""
    const entries = [];
    for (const root of roots) {
        for (const button of root.querySelectorAll('button')) {
            const icon = Array.from(button.querySelectorAll('svg, use')).some(node =>
                (node.getAttribute('data-test-icon') || node.getAttribute('href') || '')
                    .includes('overflow'));
            if (!icon || !visible(button)) continue;
            if (!button.closest('dialog[open], [role="dialog"]')) continue;
            let node = button;
            let text = '';
            for (let depth = 0; depth < 8 && node; depth++) {
                node = node.parentElement;
                if (!node) break;
                const value = node.innerText || '';
                if (value.length >= 2500) break;
                if (value.trim().length > 0) text = value;
            }
            entries.push({button, text});
        }
    }
"""
)

_SCHEDULED_ENTRIES_JS = (
    "() => {" + _SCHEDULED_ENTRIES_BODY_JS + "return entries.map(e => e.text); }"
)

_SCHEDULED_ENTRY_MARK_JS = (
    "(arg) => {"
    + _SCHEDULED_ENTRIES_BODY_JS
    + """
        const entry = entries[arg.index];
        if (!entry || entry.text !== arg.text) return false;
        entry.button.setAttribute('data-linkedin-mcp-entry', arg.token);
        return true;
    }"""
)

_VISIBLE_DIALOG_TEXT_JS = (
    "() => {"
    + _SHADOW_WALK_JS
    + """
        const texts = [];
        for (const root of roots) {
            for (const dialog of root.querySelectorAll('dialog[open], [role="dialog"]')) {
                if (visible(dialog)) texts.push(dialog.innerText || '');
            }
        }
        return texts;
    }"""
)

_SCHEDULED_ID_RE = re.compile(r"^sched-[0-9a-f]{16}$")


class _Abort(Exception):
    """A step refused; carries the result status and message."""

    def __init__(self, status: str, message: str, **extra: Any):
        super().__init__(message)
        self.status = status
        self.message = message
        self.extra = extra


@dataclass
class _Typed:
    """What the editor should hold, built from what was actually inserted."""

    expected: list[str]
    mentions: list[dict[str, Any]]


def collapse_whitespace(text: str) -> str:
    return " ".join(text.replace(" ", " ").split())


def scheduled_identifier(text: str) -> str:
    digest = hashlib.sha256(collapse_whitespace(text).encode("utf-8")).hexdigest()
    return f"sched-{digest[:16]}"


def evidence_keys(evidence: dict[str, Any]) -> set[str]:
    """Normalize raw hrefs and URNs into identity keys."""
    keys: set[str] = set()
    for href in evidence.get("hrefs") or []:
        target = identity_key_from_url(href) if isinstance(href, str) else None
        if target is not None:
            keys.add(target.key)
    for urn in evidence.get("urns") or []:
        target = identity_key_from_urn(urn) if isinstance(urn, str) else None
        if target is not None:
            keys.add(target.key)
    return keys


def _namespace(key: str) -> str:
    if key.startswith("company:urn:"):
        return "company:urn"
    return key.split(":", 1)[0] if not key.startswith("company:/") else "company:/"


def evidence_matches(keys: set[str], target: MentionTarget) -> bool:
    """True when the evidence names the target and nothing that contradicts it.

    Keys in different namespaces (a vanity path and a profile URN) can describe
    one member, so only a second key in the target's own namespace counts as a
    contradiction.
    """
    if target.key not in keys:
        return False
    namespace = _namespace(target.key)
    return all(key == target.key or _namespace(key) != namespace for key in keys)


def post_result(
    url: str,
    status: str,
    message: str,
    *,
    retry_safe: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "url": url,
        "status": status,
        "message": message,
        "retry_safe": retry_safe,
    }
    result.update(extra)
    return result


class PostComposer:
    """Publish, schedule and manage posts through LinkedIn's share composer."""

    def __init__(self, session: ScrapingSession, navigator: PageNavigator):
        self._session = session
        self._navigator = navigator
        self._page = session.page
        self._random = random.Random()

    # --- composer surface -------------------------------------------------

    def _editor(self) -> Any:
        return self._page.locator(_EDITOR_SELECTOR).first

    def _composer_dialog(self) -> Any:
        return self._page.locator(_DIALOG_SELECTOR).filter(
            has=self._page.locator(_EDITOR_SELECTOR)
        )

    async def _open_composer(self) -> Any | None:
        """Open the share composer by URL; return its one editor or None."""
        await self._navigator._navigate_to_page(SHARE_URL)
        await self._session.check_rate_limit()
        editor = self._editor()
        try:
            await editor.wait_for(state="visible", timeout=_OPEN_TIMEOUT_MS)
        except PlaywrightTimeoutError:
            return None
        if await self._page.locator(f"{_EDITOR_SELECTOR}:visible").count() != 1:
            return None
        return editor

    async def _editor_state(self, editor: Any) -> dict[str, Any]:
        state = await editor.evaluate(_EDITOR_STATE_JS)
        if not isinstance(state, dict):
            raise _Abort("composer_changed", "The composer editor could not be read.")
        return state

    async def _dismiss_composer(self, *, clear: bool) -> None:
        """Best effort: empty what this call typed, then close without a prompt.

        An empty composer closes from its dismiss button without a discard
        prompt (measured upstream). Where something else keeps it open, a
        navigation tears the modal down instead of answering a prompt whose
        buttons have only label text to tell them apart.
        """
        try:
            editor = self._editor()
            if clear and await editor.is_visible():
                await editor.click(timeout=2_000)
                await self._page.keyboard.press("ControlOrMeta+a")
                await self._page.keyboard.press("Delete")
            for _ in range(3):
                dismiss = self._page.locator(f"{_DISMISS_SELECTOR}:visible")
                if await dismiss.count() == 0:
                    break
                await dismiss.last.click(timeout=3_000)
                await asyncio.sleep(0.3)
            if await self._page.locator(f"{_EDITOR_SELECTOR}:visible").count():
                await self._page.goto(FEED_URL, wait_until="domcontentloaded")
        except Exception:
            logger.debug("Composer dismissal failed", exc_info=True)

    async def _pause(self, low: float = 0.15, high: float = 0.45) -> None:
        await asyncio.sleep(self._random.uniform(low, high))

    # --- visibility -------------------------------------------------------

    async def _apply_visibility(self, visibility: str) -> None:
        dialog = self._composer_dialog()
        settings_button = dialog.locator(_UNLABELLED_BUTTON_SELECTOR).first
        try:
            await settings_button.wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
            await settings_button.click(timeout=_STEP_TIMEOUT_MS)
            option = (
                self._page.locator(_DIALOG_SELECTOR)
                .locator(_VISIBILITY_OPTION_SELECTORS[visibility])
                .first
            )
            await option.wait_for(state="attached", timeout=_STEP_TIMEOUT_MS)
        except Exception as error:
            raise _Abort(
                "visibility_unavailable",
                "LinkedIn's post settings did not expose the requested visibility "
                "option; nothing was posted.",
            ) from error
        if not await option.evaluate(_CHECKED_JS):
            await option.click(timeout=_STEP_TIMEOUT_MS)
            await self._pause()
            if not await option.evaluate(_CHECKED_JS):
                raise _Abort(
                    "visibility_unavailable",
                    f"The {visibility!r} visibility option did not become selected.",
                )
        settings = self._page.locator(_DIALOG_SELECTOR).filter(
            has=self._page.locator(_VISIBILITY_OPTION_SELECTORS[visibility])
        )
        done = settings.locator("button:not([disabled]):visible").last
        try:
            await done.click(timeout=_STEP_TIMEOUT_MS)
            await option.wait_for(state="hidden", timeout=_STEP_TIMEOUT_MS)
            await self._editor().wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
        except Exception as error:
            raise _Abort(
                "visibility_unavailable",
                "The post settings did not return to the composer.",
            ) from error

    # --- author (post as a company page) -----------------------------------

    async def _apply_actor(self, target: MentionTarget) -> dict[str, Any]:
        """Switch the composer's author to the page the target identifies.

        Ported from upstream PR 835, which matched the author by its visible
        name. Names are neither unique nor locale-proof, so here the one radio
        option whose link or URN names the target is selected, it must read
        back as checked, and the composer's author control must not contradict
        it afterwards. Anything less fails closed.
        """
        dialog = self._composer_dialog()
        author_button = dialog.locator(_UNLABELLED_BUTTON_SELECTOR).first
        try:
            await author_button.wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
            current = evidence_keys(await author_button.evaluate(_ELEMENT_EVIDENCE_JS))
            if evidence_matches(current, target):
                return {"target": target.reference, "verified_by": "author_control"}
            await author_button.click(timeout=_STEP_TIMEOUT_MS)
            settings = self._page.locator(_DIALOG_SELECTOR).filter(
                has=self._page.locator(_SETTINGS_HEADER_SELECTOR)
            )
            actor_row = settings.locator(_ACTOR_ROW_SELECTOR).first
            await actor_row.wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
            await actor_row.click(timeout=_STEP_TIMEOUT_MS)
            radios = settings.locator(_ACTOR_RADIO_SELECTOR)
            await radios.first.wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
        except Exception as error:
            raise _Abort(
                "actor_unavailable",
                "LinkedIn did not offer an author switch in the composer; this "
                "account may not administer any page. Nothing was posted.",
            ) from error
        options = await radios.evaluate_all(_RADIO_EVIDENCE_JS)
        options = options if isinstance(options, list) else []
        matched = [
            index
            for index, item in enumerate(options)
            if isinstance(item, dict)
            and item.get("visible")
            and evidence_matches(evidence_keys(item), target)
        ]
        if len(matched) != 1:
            carried = any(
                evidence_keys(item) for item in options if isinstance(item, dict)
            )
            raise _Abort(
                "actor_unresolved",
                f"Could not verify an author option for {target.reference}: "
                + (
                    "no option carried that page's identity (try its numeric id "
                    "or URL form)."
                    if carried and not matched
                    else "the options carried no page link or URN to verify."
                    if not matched
                    else "more than one option matched."
                )
                + " Nothing was posted.",
            )
        option = radios.nth(matched[0])
        await option.click(timeout=_STEP_TIMEOUT_MS)
        await self._pause()
        after = await radios.evaluate_all(_RADIO_EVIDENCE_JS)
        checked = [
            item
            for item in (after if isinstance(after, list) else [])
            if isinstance(item, dict) and item.get("checked")
        ]
        if len(checked) != 1 or not evidence_matches(evidence_keys(checked[0]), target):
            raise _Abort(
                "actor_unresolved",
                f"The author option for {target.reference} did not become the one "
                "selected. Nothing was posted.",
            )
        try:
            # Save (the "Posting as" list), then Done (post settings): each is
            # the dialog's last enabled button (upstream PR 835).
            for _ in range(2):
                await settings.locator(
                    'button:not([disabled]):not([aria-disabled="true"]):visible'
                ).last.click(timeout=_STEP_TIMEOUT_MS)
                await self._pause(0.3, 0.6)
                if (
                    await self._page.locator(
                        f"{_SETTINGS_HEADER_SELECTOR}:visible"
                    ).count()
                    == 0
                ):
                    break
            await self._editor().wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
        except Exception as error:
            raise _Abort(
                "actor_unavailable",
                "The post settings did not return to the composer.",
            ) from error
        shown = evidence_keys(await author_button.evaluate(_ELEMENT_EVIDENCE_JS))
        if shown and not evidence_matches(shown, target):
            raise _Abort(
                "actor_mismatch",
                "The composer's author control names a different author. Nothing "
                "was posted.",
            )
        return {
            "target": target.reference,
            "verified_by": "author_control" if shown else "selected_option",
        }

    # --- attachments --------------------------------------------------------

    async def _set_files(self, button: Any, paths: list[str], accept_hint: str) -> None:
        """Hand files to the chooser the button opens, or to its file input."""
        try:
            async with self._page.expect_file_chooser(
                timeout=_STEP_TIMEOUT_MS
            ) as chooser_info:
                await button.click(timeout=_STEP_TIMEOUT_MS)
            chooser = await chooser_info.value
            if len(paths) > 1 and not chooser.is_multiple():
                raise _Abort(
                    "media_unavailable",
                    "LinkedIn's file chooser accepts one file at a time.",
                )
            await chooser.set_files(paths)
            return
        except _Abort:
            raise
        except PlaywrightTimeoutError:
            logger.debug("No file chooser opened; looking for a file input")
        inputs = self._page.locator(f'{_FILE_INPUT_SELECTOR}[accept*="{accept_hint}"]')
        if await inputs.count() != 1:
            raise _Abort(
                "media_unavailable",
                "LinkedIn did not expose one file input for the attachment.",
            )
        await inputs.first.set_input_files(paths)

    async def _wait_for_upload(
        self, dialog: Any, *, require_enabled: bool = True
    ) -> None:
        """Wait until no progress bar remains and the dialog's action is enabled."""
        deadline = time.monotonic() + _UPLOAD_TIMEOUT_SECONDS
        stable = 0
        while time.monotonic() < deadline:
            busy = await self._page.locator(f"{_PROGRESS_SELECTOR}:visible").count()
            enabled = not require_enabled
            if require_enabled:
                try:
                    action = dialog.locator("button:visible").last
                    enabled = await action.is_enabled(timeout=1_000)
                except Exception:
                    enabled = False
            stable = stable + 1 if (not busy and enabled) else 0
            if stable >= 2:
                return
            await asyncio.sleep(0.5)
        raise _Abort(
            "media_processing_timeout",
            "LinkedIn did not finish processing the attachment in time; nothing "
            "was posted.",
        )

    async def _attach_images(self, images: tuple[PostAttachment, ...]) -> None:
        composer = self._composer_dialog()
        images_before = await composer.locator("img").count()
        button = composer.locator(_IMAGE_BUTTON_SELECTOR).first
        if not await button.is_visible():
            raise _Abort(
                "media_unavailable", "The composer did not expose an add-media action."
            )
        await self._set_files(button, [image.path for image in images], "image")
        editor_dialog = (
            self._page.locator(_DIALOG_SELECTOR)
            .filter(has_not=self._page.locator(_EDITOR_SELECTOR))
            .filter(visible=True)
            .last
        )
        try:
            await editor_dialog.wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
        except PlaywrightTimeoutError:
            editor_dialog = None
        if editor_dialog is not None:
            await self._wait_for_upload(editor_dialog)
            await editor_dialog.locator("button:visible").last.click(
                timeout=_STEP_TIMEOUT_MS
            )
        await self._editor().wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
        await self._wait_for_upload(self._composer_dialog())
        if await self._composer_dialog().locator("img").count() <= images_before:
            raise _Abort(
                "media_unavailable",
                "The composer shows no attached image after the upload.",
            )

    async def _attach_document(self, document: PostAttachment, title: str) -> None:
        composer = self._composer_dialog()
        button = composer.locator(_DOCUMENT_BUTTON_SELECTOR).first
        if not await button.is_visible():
            more = composer.locator(_MORE_BUTTON_SELECTOR).first
            if await more.is_visible():
                await more.click(timeout=_STEP_TIMEOUT_MS)
                await self._pause()
            button = self._page.locator(f"{_DOCUMENT_BUTTON_SELECTOR}").first
            if not await button.is_visible():
                raise _Abort(
                    "document_unavailable",
                    "The composer did not expose an add-document action.",
                )
        await self._set_files(button, [document.path], "pdf")
        dialog = (
            self._page.locator(_DIALOG_SELECTOR)
            .filter(has=self._page.locator(_TEXT_INPUT_SELECTOR))
            .filter(has_not=self._page.locator(_EDITOR_SELECTOR))
            .filter(visible=True)
            .last
        )
        try:
            await dialog.wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
        except PlaywrightTimeoutError as error:
            raise _Abort(
                "document_unavailable", "The document dialog did not open."
            ) from error
        title_inputs = dialog.locator(f":is({_TEXT_INPUT_SELECTOR}):visible")
        if await title_inputs.count() != 1:
            raise _Abort(
                "document_unavailable",
                "The document dialog did not expose one title field.",
            )
        title_input = title_inputs.first
        max_length = await title_input.evaluate(_MAXLENGTH_JS)
        if isinstance(max_length, int) and 0 < max_length < len(title):
            raise _Abort(
                "document_title_too_long",
                f"LinkedIn limits the document title to {max_length} characters.",
            )
        await self._wait_for_upload(dialog, require_enabled=False)
        await title_input.fill(title)
        if (await title_input.input_value()) != title:
            raise _Abort(
                "document_unavailable", "The document title field did not accept it."
            )
        await self._wait_for_upload(dialog)
        await dialog.locator("button:visible").last.click(timeout=_STEP_TIMEOUT_MS)
        await self._editor().wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
        composer_text = await self._composer_dialog().first.inner_text()
        if collapse_whitespace(title) not in collapse_whitespace(composer_text):
            raise _Abort(
                "document_unavailable",
                "The composer does not show the attached document.",
            )

    # --- poll ---------------------------------------------------------------

    async def _open_poll_form(self) -> Any:
        composer = self._composer_dialog()
        button = composer.locator(_POLL_BUTTON_SELECTOR).first
        if not await button.is_visible():
            more = composer.locator(_MORE_BUTTON_SELECTOR).first
            if await more.is_visible():
                await more.click(timeout=_STEP_TIMEOUT_MS)
                await self._pause()
            button = self._page.locator(_POLL_BUTTON_SELECTOR).first
            if not await button.is_visible():
                raise _Abort(
                    "poll_unavailable",
                    "The composer did not expose a poll action; nothing was posted.",
                )
        await button.click(timeout=_STEP_TIMEOUT_MS)
        dialog = (
            self._page.locator(_DIALOG_SELECTOR)
            .filter(has=self._page.locator(_POLL_FIELD_SELECTOR))
            .filter(has=self._page.locator("select"))
            .filter(has_not=self._page.locator(_EDITOR_SELECTOR))
            .filter(visible=True)
            .last
        )
        try:
            await dialog.wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
        except PlaywrightTimeoutError as error:
            raise _Abort("poll_unavailable", "The poll form did not open.") from error
        return dialog

    async def _fill_field(self, field: Any, value: str, name: str) -> None:
        max_length = await field.evaluate(_MAXLENGTH_JS)
        if isinstance(max_length, int) and 0 < max_length < len(value):
            raise _Abort(
                "poll_rejected",
                f"LinkedIn limits the poll {name} to {max_length} characters.",
            )
        await field.fill(value)
        if await field.input_value() != value:
            raise _Abort("poll_rejected", f"The poll {name} field did not accept it.")

    async def _select_duration(self, dialog: Any, days: int) -> None:
        selects = dialog.locator("select:visible")
        if await selects.count() != 1:
            raise _Abort("poll_unavailable", "The poll form has no single duration.")
        select = selects.first
        state = await select.evaluate(_SELECT_STATE_JS)
        values = [str(v) for v in (state or {}).get("values") or []]
        known = [i for i, v in enumerate(values) if v in _POLL_DURATION_VALUES[days]]
        if len(known) == 1:
            index = known[0]
        elif len(values) == len(_POLL_DURATION_ORDER):
            index = _POLL_DURATION_ORDER.index(days)
        else:
            raise _Abort(
                "poll_unavailable",
                "The poll duration choices could not be identified.",
            )
        await select.select_option(index=index)
        after = await select.evaluate(_SELECT_STATE_JS)
        if (after or {}).get("selectedIndex") != index:
            raise _Abort("poll_rejected", "The poll duration did not stay selected.")

    async def _attach_poll(self, poll: PollSpec) -> None:
        """Fill LinkedIn's poll form and verify the composer shows it.

        Field order is structural: the question field comes first and one
        field per option follows; more option fields appear through the add
        control. Each value must read back exactly, and afterwards the
        composer must show the question and every option.
        """
        dialog = await self._open_poll_form()
        fields = dialog.locator(f":is({_POLL_FIELD_SELECTOR}):visible")
        count = await fields.count()
        if count < 1 + 2:
            raise _Abort("poll_unavailable", "The poll form has too few fields.")
        while count < 1 + len(poll.options):
            add = dialog.locator(_ADD_OPTION_BUTTON_SELECTOR).filter(visible=True)
            if await add.count() != 1:
                raise _Abort(
                    "poll_unavailable", "The poll form did not offer one add control."
                )
            await add.first.click(timeout=_STEP_TIMEOUT_MS)
            await self._pause()
            grown = await fields.count()
            if grown != count + 1:
                raise _Abort("poll_unavailable", "Adding a poll option failed.")
            count = grown
        if count != 1 + len(poll.options):
            raise _Abort(
                "poll_unavailable",
                "The poll form holds more option fields than requested.",
            )
        await self._fill_field(fields.nth(0), poll.question, "question")
        for index, option in enumerate(poll.options, start=1):
            await self._fill_field(fields.nth(index), option, f"option {index}")
        await self._select_duration(dialog, poll.duration_days)
        done = dialog.locator(
            'button:not([disabled]):not([aria-disabled="true"]):visible'
        ).last
        try:
            await done.click(timeout=_STEP_TIMEOUT_MS)
            await dialog.wait_for(state="hidden", timeout=_STEP_TIMEOUT_MS)
            await self._editor().wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
        except Exception as error:
            raise _Abort(
                "poll_rejected", "LinkedIn did not accept the poll form."
            ) from error
        shown = collapse_whitespace(await self._composer_dialog().first.inner_text())
        missing = [
            value
            for value in (poll.question, *poll.options)
            if collapse_whitespace(value) not in shown
        ]
        if missing:
            raise _Abort(
                "poll_mismatch",
                "The composer's poll preview does not show the requested question "
                "and options; nothing was posted.",
            )

    # --- schedule -----------------------------------------------------------

    async def _apply_schedule(
        self, instant: datetime, *, edit_mode: bool = False
    ) -> dict[str, Any]:
        """Drive the schedule dialog to the instant, in the browser's timezone.

        In edit mode the dialog prefills the post's *current* schedule, not
        today, so the today-window proof of the date order is withheld
        (upstream PR 695), and its confirm action is the last *enabled*
        button because Next stays disabled until something changed.
        """
        local = await self._page.evaluate(
            _BROWSER_LOCAL_TIME_JS, int(instant.timestamp() * 1000)
        )
        if not isinstance(local, dict) or not all(
            isinstance(local.get(key), int)
            for key in ("year", "month", "day", "hour", "minute")
        ):
            raise _Abort("schedule_unavailable", "The browser clock could not be read.")
        summary = schedule_summary(
            instant,
            browser_timezone=local.get("timeZone"),
            browser_local=(
                f"{local['year']:04d}-{local['month']:02d}-{local['day']:02d}T"
                f"{local['hour']:02d}:{local['minute']:02d}"
            ),
        )
        composer = self._composer_dialog()
        try:
            await composer.locator(_SCHEDULE_BUTTON_SELECTOR).first.click(
                timeout=_STEP_TIMEOUT_MS
            )
            date_input = self._page.locator(_SCHEDULE_DATE_SELECTOR).first
            time_input = self._page.locator(_SCHEDULE_TIME_SELECTOR).first
            await date_input.wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
        except Exception as error:
            raise _Abort(
                "schedule_unavailable",
                "LinkedIn did not open its schedule dialog.",
            ) from error
        try:
            today = date(local["todayYear"], local["todayMonth"], local["todayDay"])
        except (KeyError, TypeError, ValueError):
            today = None
        if edit_mode:
            today = None
        date_value, order = format_schedule_date(
            local["year"],
            local["month"],
            local["day"],
            await date_input.input_value(),
            await date_input.get_attribute("placeholder") or "",
            today,
        )
        if date_value is None or order is None:
            raise _Abort(
                "schedule_rejected",
                "The schedule dialog's date order could not be proven, and a "
                "reversed day and month would schedule the wrong day.",
            )
        time_value = format_schedule_time(
            local["hour"], local["minute"], await time_input.input_value()
        )
        await date_input.fill(date_value)
        await time_input.fill(time_value)
        await self._page.keyboard.press("Tab")
        await asyncio.sleep(0.5)
        shown_date = await date_input.input_value()
        shown_time = await time_input.input_value()
        if not date_matches(
            shown_date, order, local["year"], local["month"], local["day"]
        ):
            raise _Abort(
                "schedule_rejected",
                f"The date picker did not accept {date_value!r}; it shows "
                f"{shown_date!r}.",
            )
        if not time_matches(shown_time, local["hour"], local["minute"]):
            raise _Abort(
                "schedule_rejected",
                f"The time picker did not accept {time_value!r}; it shows "
                f"{shown_time!r}.",
            )
        schedule_dialog = self._page.locator(_DIALOG_SELECTOR).filter(
            has=self._page.locator(_SCHEDULE_DATE_SELECTOR)
        )
        try:
            confirm_selector = (
                'button:not([disabled]):not([aria-disabled="true"]):visible'
                if edit_mode
                else "button:visible"
            )
            await schedule_dialog.locator(confirm_selector).last.click(
                timeout=_STEP_TIMEOUT_MS
            )
            await date_input.wait_for(state="hidden", timeout=_STEP_TIMEOUT_MS)
            await self._editor().wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
        except Exception as error:
            raise _Abort(
                "schedule_rejected",
                "LinkedIn did not accept the scheduled date and time.",
                schedule=summary,
            ) from error
        return summary or {}

    # --- typing -------------------------------------------------------------

    async def _type_plain(self, text: str) -> None:
        """Type word by word with small randomised per-key and per-word delays."""
        keyboard = self._page.keyboard
        lines = text.split("\n")
        for index, line in enumerate(lines):
            for word in re.findall(r"\S+\s*|\s+", line):
                await keyboard.type(word, delay=self._random.randint(12, 45))
                await asyncio.sleep(self._random.uniform(0.01, 0.08))
            if index < len(lines) - 1:
                # An open typeahead (a #hashtag or @ being typed) would take
                # Enter as a selection. A space closes it without selecting.
                listboxes = self._page.locator(f"{_LISTBOX_SELECTOR}:visible")
                if await listboxes.count():
                    await keyboard.type(" ")
                    await asyncio.sleep(0.3)
                    if await listboxes.count():
                        raise _Abort(
                            "composer_changed",
                            "A typeahead stayed open before a line break.",
                        )
                await keyboard.press("Enter")

    async def _insert_mention(
        self, editor: Any, mention: MentionSegment, typed: _Typed
    ) -> None:
        before = await self._editor_state(editor)
        await self._type_plain("@" + mention.name)
        options = self._page.locator(_OPTION_SELECTOR)
        deadline = time.monotonic() + _TYPEAHEAD_TIMEOUT_SECONDS
        previous: list[Any] | None = None
        evidence: list[Any] = []
        matched: list[int] = []
        while time.monotonic() < deadline:
            await asyncio.sleep(0.4)
            raw = await options.evaluate_all(_OPTION_EVIDENCE_JS)
            evidence = raw if isinstance(raw, list) else []
            visible = [
                (index, item)
                for index, item in enumerate(evidence)
                if isinstance(item, dict) and item.get("visible")
            ]
            matched = [
                index
                for index, item in visible
                if evidence_matches(evidence_keys(item), mention.target)
            ]
            if visible and evidence == previous:
                break
            previous = evidence
        if len(matched) != 1:
            carried = any(
                evidence_keys(item) for item in evidence if isinstance(item, dict)
            )
            if not matched:
                detail = (
                    "no suggestion carried that identity"
                    if carried
                    else "the suggestions carried no profile URL or URN to verify"
                )
                raise _Abort(
                    "mention_unresolved",
                    f"Could not mention {mention.name!r} "
                    f"({mention.target.reference}): {detail}. Nothing was posted.",
                )
            raise _Abort(
                "mention_ambiguous",
                f"More than one suggestion matched {mention.target.reference}.",
            )
        token = secrets.token_hex(8)
        marked = await options.evaluate_all(
            _OPTION_MARK_JS, {"index": matched[0], "token": token}
        )
        if not isinstance(marked, dict) or not evidence_matches(
            evidence_keys(marked), mention.target
        ):
            raise _Abort(
                "mention_unresolved",
                f"The suggestion for {mention.name!r} changed before selection.",
            )
        await self._page.locator(f'[data-linkedin-mcp-mention="{token}"]').click(
            timeout=_STEP_TIMEOUT_MS
        )
        await asyncio.sleep(0.4)
        after = await self._editor_state(editor)
        entities_before = len(before.get("entities") or [])
        entities = after.get("entities") or []
        if len(entities) != entities_before + 1:
            raise _Abort(
                "mention_not_inserted",
                f"LinkedIn did not insert a mention entity for {mention.name!r}.",
            )
        entity = entities[-1]
        keys = evidence_keys(entity)
        if keys and not evidence_matches(keys, mention.target):
            raise _Abort(
                "mention_mismatch",
                f"The inserted mention does not identify {mention.target.reference}.",
            )
        inserted = collapse_whitespace(str(entity.get("text") or ""))
        text_after = collapse_whitespace(str(after.get("text") or ""))
        if not inserted or text_after.endswith("@" + collapse_whitespace(mention.name)):
            raise _Abort(
                "mention_not_inserted",
                f"The typed @{mention.name} was not converted into a mention.",
            )
        typed.expected.append(inserted)
        typed.mentions.append(
            {
                "name": mention.name,
                "inserted_text": inserted,
                "target": mention.target.reference,
                "verified_by": "evidence" if keys else "suggestion",
            }
        )
        if not after.get("focused"):
            await editor.focus()
            await self._page.keyboard.press("ControlOrMeta+End")

    async def _type_segments(
        self, editor: Any, request: PostRequest | PostEdit
    ) -> _Typed:
        typed = _Typed(expected=[], mentions=[])
        await editor.click(timeout=_STEP_TIMEOUT_MS)
        for segment in request.segments or ():
            if isinstance(segment, TextSegment):
                await self._type_plain(segment.text)
                typed.expected.append(segment.text)
            else:
                await self._insert_mention(editor, segment, typed)
        await asyncio.sleep(0.8)
        state = await self._editor_state(editor)
        shown = collapse_whitespace(str(state.get("text") or ""))
        expected = collapse_whitespace("".join(typed.expected))
        if shown != expected:
            raise _Abort(
                "composer_text_mismatch",
                "The composer does not hold exactly the requested text; nothing "
                "was posted.",
            )
        entities = [
            entity
            for entity in state.get("entities") or []
            if not str(entity.get("text") or "").startswith("#")
        ]
        if len(entities) != len(request.mentions):
            raise _Abort(
                "composer_text_mismatch",
                f"The composer holds {len(entities)} mention entities; "
                f"{len(request.mentions)} were requested.",
            )
        return typed

    # --- publishing ---------------------------------------------------------

    async def _post_links(self) -> dict[str, list[str]]:
        try:
            links = await self._page.evaluate(_POST_LINKS_JS)
        except Exception:
            logger.debug("Could not read post links", exc_info=True)
            return {"all": [], "announced": []}
        return links if isinstance(links, dict) else {"all": [], "announced": []}

    async def _capture_new_post(self, before: dict[str, list[str]]) -> str | None:
        """The one new post link LinkedIn announces after publishing, if any.

        Only links inside an announcement region (``role="alert"`` or
        ``aria-live``, where the "view post" toast lives) count. The feed under
        the composer keeps hydrating, so a link that is merely new on the page
        can be someone else's post, and naming the wrong URL is worse than
        naming none.
        """
        known = set(before.get("all") or [])
        deadline = time.monotonic() + _POST_LINK_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            links = await self._post_links()
            announced = {
                urn for urn in links.get("announced") or [] if urn not in known
            }
            if len(announced) == 1:
                return announced.pop()
            if len(announced) > 1:
                return None
            await asyncio.sleep(0.5)
        return None

    async def create_post(self, request: PostRequest) -> dict[str, Any]:
        """Publish or schedule a validated post through the share composer."""
        editor = await self._open_composer()
        if editor is None:
            return post_result(
                self._page.url,
                "composer_unavailable",
                "LinkedIn did not open exactly one share composer editor.",
            )
        state = await self._editor_state(editor)
        if collapse_whitespace(str(state.get("text") or "")):
            return post_result(
                self._page.url,
                "composer_occupied",
                "LinkedIn restored a draft into the composer. It was left untouched; "
                "publish or discard it on LinkedIn, then retry.",
            )

        typed_anything = False
        submitted = False
        schedule: dict[str, Any] | None = None
        typed = _Typed(expected=[], mentions=[])
        links_before: dict[str, list[str]] = {"all": [], "announced": []}
        actor: dict[str, Any] | None = None
        try:
            if request.post_as is not None:
                # A page post is public; LinkedIn offers no member visibility
                # choice for it, so the visibility step is skipped.
                actor = await self._apply_actor(request.post_as)
            else:
                await self._apply_visibility(request.visibility)
            if request.document is not None:
                await self._attach_document(
                    request.document, request.document_title or ""
                )
            elif request.images:
                await self._attach_images(request.images)
            elif request.poll is not None:
                await self._attach_poll(request.poll)
            if request.schedule_at is not None:
                schedule = await self._apply_schedule(request.schedule_at)
            typed_anything = True
            typed = await self._type_segments(editor, request)

            primary = (
                self._composer_dialog()
                .locator(f"{_UNLABELLED_BUTTON_SELECTOR}:visible")
                .last
            )
            if not await primary.is_enabled():
                raise _Abort(
                    "post_unavailable",
                    "The composer's primary action stayed disabled.",
                )
            if await self._page.locator(f"{_EDITOR_SELECTOR}:visible").count() != 1:
                raise _Abort("composer_changed", "The composer changed before posting.")
            links_before = await self._post_links()
            submitted = True
            await primary.click(timeout=_STEP_TIMEOUT_MS)
            try:
                await editor.wait_for(state="hidden", timeout=_PUBLISH_TIMEOUT_MS)
            except PlaywrightTimeoutError:
                return post_result(
                    self._page.url,
                    "post_unconfirmed",
                    "The post was submitted but the composer did not close. Check "
                    "your LinkedIn activity before retrying; retrying may post twice.",
                    retry_safe=False,
                    mentions=typed.mentions,
                    schedule=schedule,
                )
        except _Abort as abort:
            if submitted:
                return post_result(
                    self._page.url,
                    "post_unconfirmed",
                    abort.message,
                    retry_safe=False,
                )
            await self._dismiss_composer(clear=typed_anything)
            return post_result(
                self._page.url, abort.status, abort.message, **abort.extra
            )
        except Exception:
            if not submitted:
                await self._dismiss_composer(clear=typed_anything)
                raise
            logger.debug("Post failed after submission", exc_info=True)
            return post_result(
                self._page.url,
                "post_unconfirmed",
                "The post may already have been submitted when an error occurred. "
                "Check your LinkedIn activity before retrying; retrying may post "
                "twice.",
                retry_safe=False,
            )

        if request.schedule_at is not None:
            listing = await self.get_scheduled_posts()
            snippet = collapse_whitespace(
                "".join(typed.expected)
                or (request.poll.question if request.poll is not None else "")
            )[:60]
            entries = [
                entry
                for entry in listing.get("scheduled_posts") or []
                if snippet and snippet in collapse_whitespace(entry.get("text", ""))
            ]
            return post_result(
                FEED_URL,
                "scheduled",
                "Post scheduled."
                if len(entries) == 1
                else "LinkedIn accepted the schedule, but the scheduled list did not "
                "show exactly one matching entry. Check it with get_scheduled_posts "
                "before retrying.",
                retry_safe=False,
                schedule=schedule,
                mentions=typed.mentions,
                post_as=actor,
                scheduled_entry=entries[0] if len(entries) == 1 else None,
                scheduled_posts=listing.get("scheduled_posts"),
            )

        urn = await self._capture_new_post(links_before)
        return post_result(
            self._page.url,
            "published",
            "Post published."
            if urn
            else "The composer closed after posting, but the new post's link was "
            "not captured. Check your activity before retrying.",
            retry_safe=False,
            post_urn=urn,
            post_url=f"https://www.linkedin.com/feed/update/{urn}/" if urn else None,
            mentions=typed.mentions,
            post_as=actor,
        )

    async def create_poll(self, request: PostRequest) -> dict[str, Any]:
        """Publish or schedule a validated poll through the share composer."""
        if request.poll is None:
            return post_result(
                FEED_URL, "invalid_request", "create_poll needs a validated poll."
            )
        return await self.create_post(request)

    # --- scheduled posts ----------------------------------------------------

    async def _open_scheduled_list(self) -> dict[str, Any] | None:
        """Open the scheduled posts modal; return a refusal result on failure."""
        editor = await self._open_composer()
        if editor is None:
            return post_result(
                self._page.url,
                "composer_unavailable",
                "LinkedIn did not open exactly one share composer editor.",
            )
        state = await self._editor_state(editor)
        if collapse_whitespace(str(state.get("text") or "")):
            # The schedule round-trip drops editor content (measured upstream),
            # so opening it would destroy someone's restored draft.
            return post_result(
                self._page.url,
                "composer_occupied",
                "LinkedIn restored a draft into the composer; opening the schedule "
                "view would discard it, so nothing was opened.",
            )
        try:
            await (
                self._composer_dialog()
                .locator(_SCHEDULE_BUTTON_SELECTOR)
                .first.click(timeout=_STEP_TIMEOUT_MS)
            )
            date_input = self._page.locator(_SCHEDULE_DATE_SELECTOR).first
            await date_input.wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
            schedule_dialog = self._page.locator(_DIALOG_SELECTOR).filter(
                has=self._page.locator(_SCHEDULE_DATE_SELECTOR)
            )
            await schedule_dialog.locator(_VIEW_ALL_SCHEDULED_SELECTOR).first.click(
                timeout=_STEP_TIMEOUT_MS
            )
            await date_input.wait_for(state="hidden", timeout=_STEP_TIMEOUT_MS)
        except Exception:
            logger.debug("Scheduled posts view did not open", exc_info=True)
            await self._dismiss_composer(clear=False)
            return post_result(
                self._page.url,
                "list_unavailable",
                "LinkedIn did not open the scheduled posts view. This step is "
                "unverified against a live account past the schedule button "
                "itself: the mini date/time-picker dialog it opens, and its "
                "'view all scheduled posts' arrow, only exist after that "
                "click, so no capture of their markup exists yet. A DOM "
                "capture taken right after clicking the schedule control "
                "would confirm or fix this selector.",
            )
        # The modal renders its heading before its entries; settled means two
        # identical non-empty samples (upstream PR 692).
        previous: list[str] | None = None
        for _ in range(16):
            await asyncio.sleep(0.5)
            sample = await self._page.evaluate(_VISIBLE_DIALOG_TEXT_JS)
            if isinstance(sample, list) and any(t.strip() for t in sample):
                if sample == previous:
                    break
                previous = sample
        return None

    async def _read_scheduled_entries(self) -> list[dict[str, Any]]:
        raw = await self._page.evaluate(_SCHEDULED_ENTRIES_JS)
        texts = (
            [text for text in raw if isinstance(text, str)]
            if isinstance(raw, list)
            else []
        )
        return [
            {"index": index, "identifier": scheduled_identifier(text), "text": text}
            for index, text in enumerate(texts)
        ]

    async def get_scheduled_posts(self) -> dict[str, Any]:
        """List the authenticated user's scheduled posts."""
        failure = await self._open_scheduled_list()
        if failure is not None:
            return failure
        url = self._page.url
        try:
            texts = await self._page.evaluate(_VISIBLE_DIALOG_TEXT_JS)
            raw_text = "\n\n".join(t for t in texts if isinstance(t, str)).strip()
            entries = await self._read_scheduled_entries()
        finally:
            await self._dismiss_composer(clear=False)
        if not raw_text:
            return post_result(
                url,
                "list_read_failed",
                "The scheduled posts view opened but could not be read. Retry "
                "rather than assuming nothing is scheduled.",
            )
        return {
            "url": url,
            "sections": {"scheduled_posts": raw_text},
            "scheduled_posts": [
                {"identifier": entry["identifier"], "text": entry["text"]}
                for entry in entries
            ],
        }

    async def _abandon_edit(self) -> None:
        """Leave an edit composer unsaved by navigating away.

        A dirty composer's dismiss raises a discard prompt whose buttons only
        differ by label text; an SPA route change tears the modal down and the
        post keeps its stored state (verified live upstream, PR 695).
        """
        try:
            await self._page.goto(FEED_URL, wait_until="domcontentloaded")
        except Exception:
            logger.debug("Leaving the edit composer failed", exc_info=True)

    async def _replace_editor_text(self, editor: Any, edit: PostEdit) -> _Typed:
        await editor.click(timeout=_STEP_TIMEOUT_MS)
        await self._page.keyboard.press("ControlOrMeta+a")
        await self._page.keyboard.press("Delete")
        await asyncio.sleep(0.3)
        cleared = await self._editor_state(editor)
        if collapse_whitespace(str(cleared.get("text") or "")) or cleared.get(
            "entities"
        ):
            raise _Abort("composer_changed", "The existing text could not be cleared.")
        return await self._type_segments(editor, edit)

    async def _save_edit(self, editor: Any) -> None:
        """Click the edit composer's save action and wait for it to close.

        In edit mode LinkedIn appends a labelled Back button, so the primary is
        the last visible button without an aria-label (upstream PR 695).
        """
        primary = (
            self._composer_dialog()
            .locator(f"{_UNLABELLED_BUTTON_SELECTOR}:visible")
            .last
        )
        if not await primary.is_enabled():
            raise _Abort("edit_unavailable", "The save action stayed disabled.")
        await primary.click(timeout=_STEP_TIMEOUT_MS)
        await editor.wait_for(state="hidden", timeout=_PUBLISH_TIMEOUT_MS)

    async def edit_scheduled_post(
        self, identifier: str, edit: PostEdit, *, confirm: bool
    ) -> dict[str, Any]:
        """Change a scheduled post's text, schedule, or both (upstream PR 695)."""
        if not isinstance(identifier, str) or not _SCHEDULED_ID_RE.match(identifier):
            return post_result(
                FEED_URL,
                "invalid_identifier",
                "identifier must be one returned by get_scheduled_posts "
                "(sched-<16 hex digits>).",
            )
        failure = await self._open_scheduled_list()
        if failure is not None:
            return failure
        entries = await self._read_scheduled_entries()
        matches = [e for e in entries if e["identifier"] == identifier]
        if len(matches) != 1:
            await self._dismiss_composer(clear=False)
            return post_result(
                self._page.url,
                "entry_not_found" if not matches else "entry_ambiguous",
                "No scheduled post has that identifier; re-read get_scheduled_posts."
                if not matches
                else "More than one scheduled post has that identifier.",
            )
        entry = matches[0]
        changes = {
            "text": edit.rendered_text,
            "schedule": schedule_summary(edit.schedule_at),
        }
        if not confirm:
            await self._dismiss_composer(clear=False)
            return post_result(
                self._page.url,
                "preview",
                "Nothing was changed. Call again with confirm=true to apply the edit.",
                identifier=identifier,
                entry=entry["text"],
                changes=changes,
            )
        token = secrets.token_hex(8)
        marked = await self._page.evaluate(
            _SCHEDULED_ENTRY_MARK_JS,
            {"index": entry["index"], "text": entry["text"], "token": token},
        )
        if marked is not True:
            await self._dismiss_composer(clear=False)
            return post_result(
                self._page.url,
                "entry_moved",
                "The scheduled list changed while the entry was being resolved; "
                "nothing was changed.",
            )
        submitted = False
        schedule: dict[str, Any] | None = None
        typed = _Typed(expected=[], mentions=[])
        try:
            try:
                await self._page.locator(f'[data-linkedin-mcp-entry="{token}"]').click(
                    timeout=_STEP_TIMEOUT_MS
                )
                await (
                    self._page.locator(_MENU_EDIT_SELECTOR)
                    .filter(visible=True)
                    .first.click(timeout=_STEP_TIMEOUT_MS)
                )
                editor = self._editor()
                await editor.wait_for(state="visible", timeout=_OPEN_TIMEOUT_MS)
            except Exception as error:
                raise _Abort(
                    "edit_unavailable",
                    "The scheduled post's menu did not open its edit composer.",
                ) from error
            current = collapse_whitespace(
                str((await self._editor_state(editor)).get("text") or "")
            )
            if not current or current not in collapse_whitespace(entry["text"]):
                raise _Abort(
                    "edit_mismatch",
                    "The edit composer opened a post that does not match the "
                    "identified entry; nothing was changed.",
                )
            if edit.schedule_at is not None:
                schedule = await self._apply_schedule(edit.schedule_at, edit_mode=True)
            if edit.segments is not None:
                typed = await self._replace_editor_text(editor, edit)
            submitted = True
            await self._save_edit(editor)
        except _Abort as abort:
            await self._abandon_edit()
            if submitted:
                return post_result(
                    self._page.url,
                    "edit_unconfirmed",
                    abort.message,
                    retry_safe=False,
                )
            return post_result(self._page.url, abort.status, abort.message)
        except Exception:
            await self._abandon_edit()
            if not submitted:
                raise
            return post_result(
                self._page.url,
                "edit_unconfirmed",
                "The edit may have been saved when an error occurred. Re-read "
                "get_scheduled_posts before retrying.",
                retry_safe=False,
            )
        await self._dismiss_composer(clear=False)
        listing = await self.get_scheduled_posts()
        entries_after = listing.get("scheduled_posts") or []
        snippet = collapse_whitespace("".join(typed.expected))[:60]
        updated = [
            item
            for item in entries_after
            if snippet and snippet in collapse_whitespace(item.get("text", ""))
        ]
        return post_result(
            FEED_URL,
            "edited",
            "Scheduled post updated."
            if not snippet or len(updated) == 1
            else "LinkedIn closed the editor, but the scheduled list does not show "
            "exactly one entry with the new text. Re-read get_scheduled_posts.",
            retry_safe=False,
            changes=changes,
            schedule=schedule,
            mentions=typed.mentions,
            scheduled_entry=updated[0] if len(updated) == 1 else None,
            scheduled_posts=entries_after,
        )

    # --- own published posts -----------------------------------------------

    async def _own_member_path(self) -> str | None:
        await self._navigator._navigate_to_page("https://www.linkedin.com/in/me/")
        target = identity_key_from_url(self._page.url)
        if target is None or target.kind != "person" or target.key.endswith("/me/"):
            return None
        return target.key

    async def _open_own_post_menu(self, post_url: str) -> dict[str, Any]:
        """Load an own post and open its control menu; raise _Abort otherwise.

        Authorship needs two structural signals, both required: the post's
        actor link is the logged-in member's own profile (resolved through the
        /in/me/ redirect), and its control menu offers the owner-only edit and
        delete actions (icon test hooks, not labels).
        """
        own = await self._own_member_path()
        if own is None:
            raise _Abort(
                "author_unverified", "The logged-in member's profile was not resolved."
            )
        urn = post_url.rstrip("/").rsplit("/", 1)[-1]
        await self._navigator._navigate_to_page(post_url)
        await self._session.check_rate_limit()
        if urn not in self._page.url:
            raise _Abort("post_unavailable", "LinkedIn did not open that post's page.")
        try:
            await self._page.wait_for_selector("main", timeout=_OPEN_TIMEOUT_MS)
        except PlaywrightTimeoutError:
            logger.debug("Post page main did not appear")
        ownership = await self._page.evaluate(_POST_OWNERSHIP_JS, urn)
        if not isinstance(ownership, dict):
            raise _Abort("post_unavailable", "The post page could not be read.")
        actor = identity_key_from_url(str(ownership.get("actorHref") or ""))
        if actor is None or actor.key != own:
            raise _Abort(
                "not_own_post",
                "The post's author is not the logged-in member; nothing was changed.",
            )
        if not ownership.get("controlCount"):
            raise _Abort(
                "post_unavailable", "The post page did not show that post's controls."
            )
        try:
            await (
                self._page.locator(f"main {_OVERFLOW_BUTTON_SELECTOR}")
                .filter(visible=True)
                .first.click(timeout=_STEP_TIMEOUT_MS)
            )
            await (
                self._page.locator(_OWNER_DELETE_ITEM_SELECTOR)
                .filter(visible=True)
                .first.wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
            )
            await (
                self._page.locator(_OWNER_EDIT_ITEM_SELECTOR)
                .filter(visible=True)
                .first.wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
            )
        except Exception as error:
            await self._close_menu()
            raise _Abort(
                "not_own_post",
                "The post's control menu does not offer owner actions; nothing "
                "was changed.",
            ) from error
        return {**ownership, "ownKey": own}

    async def _close_menu(self) -> None:
        try:
            await self._page.keyboard.press("Escape")
        except Exception:
            logger.debug("Closing the post menu failed", exc_info=True)

    async def delete_post(self, post_url: str, *, confirm: bool) -> dict[str, Any]:
        """Delete one of the logged-in member's own published posts."""
        try:
            ownership = await self._open_own_post_menu(post_url)
        except _Abort as abort:
            return post_result(self._page.url, abort.status, abort.message)
        preview_text = str(ownership.get("text") or "")[:500]
        if not confirm:
            await self._close_menu()
            return post_result(
                post_url,
                "preview",
                "Authorship verified; nothing was deleted. Call again with "
                "confirm=true to delete this post. LinkedIn cannot recover it.",
                post_text=preview_text,
            )
        try:
            await (
                self._page.locator(_OWNER_DELETE_ITEM_SELECTOR)
                .filter(visible=True)
                .first.click(timeout=_STEP_TIMEOUT_MS)
            )
            confirm_dialog = (
                self._page.locator(f"{_DIALOG_SELECTOR}, [role='alertdialog']")
                .filter(visible=True)
                .last
            )
            await confirm_dialog.wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
            await confirm_dialog.locator("button:visible").last.click(
                timeout=_STEP_TIMEOUT_MS
            )
        except Exception:
            logger.debug("Post delete confirmation failed", exc_info=True)
            return post_result(
                post_url,
                "delete_unconfirmed",
                "LinkedIn's delete confirmation could not be completed. Check the "
                "post before retrying.",
                retry_safe=False,
            )
        await asyncio.sleep(1.5)
        await self._navigator._navigate_to_page(post_url)
        urn = post_url.rstrip("/").rsplit("/", 1)[-1]
        after = await self._page.evaluate(_POST_OWNERSHIP_JS, urn)
        after_actor = (
            identity_key_from_url(str(after.get("actorHref") or ""))
            if isinstance(after, dict)
            else None
        )
        # Gone means the page no longer shows this member's post with its
        # controls. Anything ambiguous reads as "still there": the answer
        # then asks for a check instead of claiming a deletion.
        still_there = not isinstance(after, dict) or bool(
            after.get("controlCount")
            and after_actor is not None
            and after_actor.key == ownership.get("ownKey")
        )
        if still_there:
            return post_result(
                post_url,
                "delete_unconfirmed",
                "The post still loads with its controls after deleting. Check it "
                "before retrying.",
                retry_safe=False,
            )
        return post_result(
            post_url,
            "deleted",
            "Post deleted.",
            retry_safe=False,
            post_text=preview_text,
        )

    async def edit_post(
        self, post_url: str, edit: PostEdit, *, confirm: bool
    ) -> dict[str, Any]:
        """Replace the text of one of the logged-in member's own posts."""
        try:
            ownership = await self._open_own_post_menu(post_url)
        except _Abort as abort:
            return post_result(self._page.url, abort.status, abort.message)
        if not confirm:
            await self._close_menu()
            return post_result(
                post_url,
                "preview",
                "Authorship verified; nothing was changed. Call again with "
                "confirm=true to replace the text.",
                post_text=str(ownership.get("text") or "")[:500],
                new_text=edit.rendered_text,
            )
        submitted = False
        typed = _Typed(expected=[], mentions=[])
        try:
            try:
                await (
                    self._page.locator(_OWNER_EDIT_ITEM_SELECTOR)
                    .filter(visible=True)
                    .first.click(timeout=_STEP_TIMEOUT_MS)
                )
                editor = self._editor()
                await editor.wait_for(state="visible", timeout=_OPEN_TIMEOUT_MS)
            except Exception as error:
                raise _Abort(
                    "edit_unavailable", "LinkedIn did not open the post's editor."
                ) from error
            if await self._page.locator(f"{_EDITOR_SELECTOR}:visible").count() != 1:
                raise _Abort("edit_unavailable", "More than one editor is open.")
            typed = await self._replace_editor_text(editor, edit)
            submitted = True
            await self._save_edit(editor)
        except _Abort as abort:
            await self._abandon_edit()
            if submitted:
                return post_result(
                    post_url, "edit_unconfirmed", abort.message, retry_safe=False
                )
            return post_result(post_url, abort.status, abort.message)
        except Exception:
            await self._abandon_edit()
            if not submitted:
                raise
            return post_result(
                post_url,
                "edit_unconfirmed",
                "The edit may have been saved when an error occurred. Check the "
                "post before retrying.",
                retry_safe=False,
            )
        await self._navigator._navigate_to_page(post_url)
        urn = post_url.rstrip("/").rsplit("/", 1)[-1]
        after = await self._page.evaluate(_POST_OWNERSHIP_JS, urn)
        shown = collapse_whitespace(str((after or {}).get("text") or ""))
        expected = collapse_whitespace("".join(typed.expected))
        return post_result(
            post_url,
            "edited" if expected[:200] in shown else "edit_unconfirmed",
            "Post updated."
            if expected[:200] in shown
            else "The editor closed, but the post page does not show the new text "
            "yet. Check the post before retrying.",
            retry_safe=False,
            mentions=typed.mentions,
        )

    async def delete_scheduled_post(
        self, identifier: str, *, confirm: bool
    ) -> dict[str, Any]:
        """Delete one scheduled post named by its get_scheduled_posts identifier."""
        if not isinstance(identifier, str) or not _SCHEDULED_ID_RE.match(identifier):
            return post_result(
                FEED_URL,
                "invalid_identifier",
                "identifier must be one returned by get_scheduled_posts "
                "(sched-<16 hex digits>).",
            )
        failure = await self._open_scheduled_list()
        if failure is not None:
            return failure
        try:
            entries = await self._read_scheduled_entries()
            matches = [e for e in entries if e["identifier"] == identifier]
            if len(matches) != 1:
                return post_result(
                    self._page.url,
                    "entry_not_found" if not matches else "entry_ambiguous",
                    "No scheduled post has that identifier; it changes whenever "
                    "the entry's text does. Re-read get_scheduled_posts."
                    if not matches
                    else "More than one scheduled post has that identifier.",
                )
            entry = matches[0]
            if not confirm:
                return post_result(
                    self._page.url,
                    "preview",
                    "Nothing was deleted. Call again with confirm=true to delete "
                    "this scheduled post; LinkedIn cannot recover it.",
                    entry=entry["text"],
                    identifier=identifier,
                )
            token = secrets.token_hex(8)
            marked = await self._page.evaluate(
                _SCHEDULED_ENTRY_MARK_JS,
                {"index": entry["index"], "text": entry["text"], "token": token},
            )
            if marked is not True:
                return post_result(
                    self._page.url,
                    "entry_moved",
                    "The scheduled list changed while the entry was being resolved; "
                    "nothing was deleted.",
                )
            try:
                await self._page.locator(f'[data-linkedin-mcp-entry="{token}"]').click(
                    timeout=_STEP_TIMEOUT_MS
                )
                await (
                    self._page.locator(f"{_MENU_DELETE_SELECTOR}")
                    .filter(visible=True)
                    .first.click(timeout=_STEP_TIMEOUT_MS)
                )
            except Exception:
                logger.debug("Scheduled entry menu failed", exc_info=True)
                return post_result(
                    self._page.url,
                    "menu_unavailable",
                    "The scheduled post's menu did not expose a delete action; "
                    "nothing was deleted.",
                )
            confirm_dialog = (
                self._page.locator(f"{_DIALOG_SELECTOR}, [role='alertdialog']")
                .filter(visible=True)
                .filter(
                    has_not=self._page.locator(
                        'svg[data-test-icon*="overflow"], use[href*="overflow"]'
                    )
                )
                .last
            )
            try:
                await confirm_dialog.wait_for(state="visible", timeout=_STEP_TIMEOUT_MS)
                await confirm_dialog.locator("button:visible").last.click(
                    timeout=_STEP_TIMEOUT_MS
                )
            except Exception:
                logger.debug("Delete confirmation failed", exc_info=True)
                return post_result(
                    self._page.url,
                    "delete_unconfirmed",
                    "LinkedIn's delete confirmation could not be completed. Re-read "
                    "get_scheduled_posts before retrying.",
                    retry_safe=False,
                )
            await asyncio.sleep(1.5)
            remaining = await self._read_scheduled_entries()
            if any(e["identifier"] == identifier for e in remaining):
                return post_result(
                    self._page.url,
                    "delete_unconfirmed",
                    "The entry is still listed after confirming the deletion. "
                    "Re-read get_scheduled_posts before retrying.",
                    retry_safe=False,
                )
            return post_result(
                self._page.url,
                "deleted",
                "Scheduled post deleted.",
                retry_safe=False,
                identifier=identifier,
                entry=entry["text"],
            )
        finally:
            await self._dismiss_composer(clear=False)
