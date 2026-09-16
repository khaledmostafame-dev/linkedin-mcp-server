"""Messaging inbox, thread and conversation-search workflows."""

from __future__ import annotations

from typing import Any, Literal
from urllib.parse import quote_plus, urlparse

import logging
import re

from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.core.exceptions import LinkedInScraperException
from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.identifiers import (
    messaging_thread_url,
    normalize_person_identifier,
    normalize_thread_id,
    person_profile_url,
)
from linkedin_mcp_server.scraping.link_metadata import (
    Reference,
    build_references,
    dedupe_references,
)
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.profile_page import ProfilePageReader
from linkedin_mcp_server.scraping.session import ScrapingSession
from linkedin_mcp_server.scraping.text import (
    CONVERSATION_OPTIONS_EN,
    strip_conversation_chrome,
    strip_linkedin_noise,
)

logger = logging.getLogger(__name__)

# Best-effort prefix strip for the en-US "Select conversation with " verb.
# Browser locale is forced to en-US (see BrowserManager) so this normally
# succeeds; the regex falls through silently for any other locale, in
# which case the full aria-label flows into the ref's text field rather
# than a stripped name.
_SELECT_CONVERSATION_PREFIX_RE = re.compile(
    r"^Select conversation with\s+", re.IGNORECASE
)


def strip_select_conversation_prefix(aria_label: str) -> str:
    """Drop the en-US selection verb from one conversation row's aria-label."""
    return _SELECT_CONVERSATION_PREFIX_RE.sub("", aria_label).strip()


# The opener for the per-thread options menu carries no aria-label or class
# name of its own; the only tested signal is the visually-hidden label text
# in front of it (`ConversationOptionsTextTable.menu_opener_prefix`, the same
# string `strip_conversation_chrome` already relies on). This walks up from
# whichever leaf text node starts with that prefix to the nearest clickable
# ancestor within a few levels — the same bounded-ancestor-walk shape
# `connection_actions.py` uses to find a profile's own More button — and
# clicks it. Returns false rather than guessing when no such ancestor exists.
_OPEN_CONVERSATION_OPTIONS_JS = r"""(prefix) => {
    const visible = element => {
        const visibility = element && getComputedStyle(element).visibility;
        return !!(
            element &&
            visibility !== 'hidden' &&
            visibility !== 'collapse' &&
            (element.offsetWidth || element.offsetHeight || element.getClientRects().length)
        );
    };
    const normalize = value => (value || '').replace(/\s+/g, ' ').trim();
    const main = document.querySelector('main');
    if (!main) return false;
    const labels = Array.from(main.querySelectorAll('*')).filter(
        element =>
            element.children.length === 0 &&
            normalize(element.textContent).startsWith(prefix)
    );
    for (const label of labels) {
        let ancestor = label;
        for (let depth = 0; depth < 6 && ancestor; depth++) {
            if (
                ancestor.matches('button, [role="button"]') &&
                visible(ancestor) &&
                !ancestor.hasAttribute('disabled')
            ) {
                ancestor.click();
                return true;
            }
            ancestor = ancestor.parentElement;
        }
    }
    return false;
}"""

# Menu items inside the opened `[role="menu"]`. `role="menuitem"` covers the
# accessible case; a plain `button` inside the menu is kept as a fallback for
# a menu LinkedIn renders without the ARIA role.
_READ_CONVERSATION_MENU_ITEMS_JS = r"""() => {
    const normalize = value => (value || '').replace(/\s+/g, ' ').trim();
    return Array.from(
        document.querySelectorAll('[role="menu"] [role="menuitem"], [role="menu"] button')
    ).map(element => normalize(element.innerText || element.textContent))
        .filter(Boolean);
}"""

_CLICK_CONVERSATION_MENU_ITEM_JS = r"""(label) => {
    const normalize = value => (value || '').replace(/\s+/g, ' ').trim();
    const items = Array.from(
        document.querySelectorAll('[role="menu"] [role="menuitem"], [role="menu"] button')
    ).filter(element => normalize(element.innerText || element.textContent) === label);
    if (items.length !== 1) return false;
    items[0].click();
    return true;
}"""


class ConversationReader:
    """Own every workflow whose subject is a LinkedIn messaging thread.

    The sidebar is the reason this reads the page directly rather than through
    `SectionCapture`: LinkedIn renders conversation rows with no anchor href,
    no thread-id attribute and no embedded URN, so a thread id can only be had
    by clicking a row and reading the SPA URL the click lands on. That click
    may mark the row read, which is the closest thing to a write anywhere in
    this module and why every caller filters by participant name *before* a
    row is ever clicked.
    """

    def __init__(
        self,
        session: ScrapingSession,
        navigator: PageNavigator,
        content: PageContentReader,
        profile_page: ProfilePageReader,
    ):
        self._session = session
        self._navigator = navigator
        self._content = content
        self._profile_page = profile_page

    @staticmethod
    def _single_section_result(
        url: str,
        section_name: str,
        text: str,
        references: list[Reference] | None = None,
    ) -> dict[str, Any]:
        """Build a standard single-section scraping response."""
        result: dict[str, Any] = {"url": url, "sections": {}}
        if text:
            result["sections"][section_name] = text
            if references:
                result["references"] = {section_name: references}
        return result

    async def _wait_for_main_text(
        self,
        *,
        minimum_length: int = 100,
        timeout: int = 10000,
        log_context: str,
    ) -> None:
        """Wait for main content to populate enough text to scrape."""
        try:
            await self._session.page.wait_for_function(
                """({ minimumLength }) => {
                    const main = document.querySelector('main');
                    if (!main) return false;
                    return main.innerText.length > minimumLength;
                }""",
                arg={"minimumLength": minimum_length},
                timeout=timeout,
            )
        except PlaywrightTimeoutError:
            logger.debug("%s content did not appear", log_context)

    async def _scroll_main_scrollable_region(
        self,
        *,
        position: Literal["top", "bottom"],
        attempts: int,
        pause_time: float = 0.5,
    ) -> None:
        """Scroll the largest scrollable region inside main when one exists."""
        for _ in range(attempts):
            await self._session.page.evaluate(
                """({ position }) => {
                    const main = document.querySelector('main');
                    if (!main) return false;

                    const isScrollable = element => {
                        const style = window.getComputedStyle(element);
                        return (
                            (style.overflowY === 'auto' || style.overflowY === 'scroll') &&
                            element.scrollHeight > element.clientHeight + 20
                        );
                    };

                    const candidates = [main, ...main.querySelectorAll('*')].filter(isScrollable);
                    const target = candidates.sort(
                        (left, right) => right.scrollHeight - left.scrollHeight
                    )[0] || main;
                    target.scrollTop = position === 'top' ? 0 : target.scrollHeight;
                    return true;
                }""",
                {"position": position},
            )
            await self._session.delay(pause_time)

    async def _extract_conversation_thread_refs(
        self, limit: int | None, context: str, *, name_filter: str | None = None
    ) -> list[Reference]:
        """Click each visible conversation item and capture the thread URL.

        Works for both the inbox sidebar and the URL-driven search-results
        sidebar (`/messaging/?searchTerm=…`), which share the same DOM shape:
        each conversation row is an ``<li>`` containing a ``<label>`` with an
        ``aria-label`` attribute carrying the participant name.

        LinkedIn renders the sidebar with no ``<a href>`` tags, no
        ``data-thread-id`` attributes, and no embedded URNs — clicking each
        row and reading the SPA URL is the only reliable extraction path.
        Pass ``limit=None`` to capture every visible row.

        When ``name_filter`` is provided, every row's aria-label is still read
        but only rows whose cleaned participant name equals it (case-insensitive)
        are clicked; non-matching rows are skipped without clicking. Clicking a
        row may mark it as read, so the filter keeps the read-marking side effect
        scoped to the requested participant when resolving by username.
        """
        # The conversation list mounts after main text settles, so wait
        # explicitly for at least one label rather than relying on
        # _wait_for_main_text alone (which only checks chrome text). LinkedIn
        # routinely takes several seconds to hydrate the messaging sidebar
        # after a navigation; an empty sidebar (zero matches) returns on
        # timeout.
        #
        # Selector is structural (`main li label[aria-label]`) rather than
        # text-prefix-based (`aria-label^="Select conversation"`) so it
        # survives any LinkedIn locale — the verb in the aria-label is
        # locale-dependent, the attribute's presence inside a list-item label
        # is not.
        #
        # Wait on `state="attached"` instead of the default `visible`:
        # Ember-managed labels are reliably attached but Playwright's
        # visibility heuristic doesn't always consider them visible.
        try:
            await self._session.page.wait_for_selector(
                "main li label[aria-label]",
                state="attached",
                timeout=10000,
            )
        except PlaywrightTimeoutError:
            logger.debug(
                "conversation labels did not appear within 10s (context=%s)",
                context,
            )
            return []

        # The Ember click handler lives on an inner div; the <li> and <label>
        # don't trigger SPA navigation.  No role/aria attributes exist on the
        # clickable element, so class-name selectors are unavoidable here.
        # The aria-label value flows through unmodified — Python strips any
        # known locale prefix to derive a clean participant name for refs.
        conversations: list[dict[str, str]] = await self._session.page.evaluate(
            """async ({ limit, nameFilter }) => {
                const labels = Array.from(document.querySelectorAll(
                    'main li label[aria-label]'
                ));
                const cap = (limit == null)
                    ? labels.length
                    : Math.min(labels.length, limit);
                // Normalize the optional participant filter the same way the
                // Python prefix-strip does (en-US "Select conversation with"
                // verb, collapsed whitespace) so the JS-side comparison
                // matches. Only the matching row is clicked — clicking marks a
                // row read, so unrelated threads must not be clicked.
                const wanted = (nameFilter || '')
                    .replace(/\\s+/g, ' ').trim().toLowerCase();
                const results = [];
                for (let i = 0; i < cap; i++) {
                    const label = labels[i];
                    const ariaLabel = label.getAttribute('aria-label') || '';
                    const rowName = ariaLabel
                        .replace(/^Select conversation with\\s+/i, '')
                        .replace(/\\s+/g, ' ').trim().toLowerCase();
                    if (wanted && rowName !== wanted) continue;
                    const clickTarget = label.closest('li')
                        ?.querySelector('div[class*="listitem__link"]');
                    if (!clickTarget) continue;
                    const before = location.href;
                    clickTarget.click();
                    // Poll for the SPA URL to settle on the thread route. The
                    // Ember click handler can take a moment to bind after the
                    // label mounts, and a fixed sleep races the initial click.
                    let after = before;
                    for (let waits = 0; waits < 12; waits++) {
                        await new Promise(r => setTimeout(r, 100));
                        after = location.href;
                        if (after !== before
                            && /\\/messaging\\/thread\\//.test(after)) break;
                    }
                    const match = after.match(
                        /\\/messaging\\/thread\\/([^/?#]+)/
                    );
                    if (match) {
                        results.push({ ariaLabel, threadId: match[1] });
                    }
                }
                return results;
            }""",
            {"limit": limit, "nameFilter": name_filter},
        )
        refs: list[Reference] = []
        for conv in conversations:
            ref: Reference = {
                "kind": "conversation",
                "url": f"/messaging/thread/{conv['threadId']}/",
                "context": context,
            }
            name = strip_select_conversation_prefix(conv.get("ariaLabel", ""))
            if name:
                ref["text"] = name
            refs.append(ref)
        return refs

    async def _resolve_conversation_thread_urls(self, display_name: str) -> list[str]:
        """Return all thread URLs whose participant name matches display_name.

        Enumerates the plain messaging inbox (`/messaging/`) plus click-to-capture
        because LinkedIn renders the messaging sidebar with no anchor hrefs, no
        data-thread attributes, and no embedded URNs — clicking each row and
        reading the resulting SPA URL is the only available extraction path.
        The inbox is used rather than `?searchTerm=` because LinkedIn's
        messaging search frequently returns "We didn't find anything" for a
        participant whose thread is plainly present in the inbox (issue #434).
        ``name_filter`` is passed to the enumerator so only the matching row is
        clicked — clicking a row may mark it read, so unrelated threads stay
        untouched.

        Matches by case-insensitive equality on the cleaned participant name
        derived from the row's aria-label, which tolerates duplicate threads
        with the same participant. Browser locale is forced to en-US so the
        verb prefix strips reliably; in any other locale the comparison fails
        cleanly with "Could not find a conversation" rather than returning
        a wrong-thread match. If the inbox scan finds nothing (a thread buried
        below the scrolled rows), it falls back to the `?searchTerm=` search as
        a last resort.

        For a participant with multiple threads, the returned set — and thus
        ``index`` selection in the caller — covers the threads visible in the
        scanned inbox; the search fallback only runs when the inbox scan is
        empty. Open a buried duplicate thread directly via ``thread_id``
        (enumerate IDs with ``search_conversations``).
        """
        target_name = display_name.strip().lower()

        def _match(refs: list[Reference]) -> list[str]:
            # name_filter already gated the clicks; this enforces the same
            # exact-equality match Python-side and tolerates duplicate threads.
            return [
                f"https://www.linkedin.com{ref['url']}"
                for ref in refs
                if (ref.get("text") or "").strip().lower() == target_name
            ]

        # Primary path: enumerate the plain inbox. Reliable for the recent
        # threads that the verify-after-send workflow needs (issue #434).
        await self._navigator._navigate_to_page("https://www.linkedin.com/messaging/")
        await self._session.check_rate_limit()
        await self._wait_for_main_text(log_context="Messaging inbox")
        await self._session.dismiss_modal()
        await self._scroll_main_scrollable_region(
            position="bottom", attempts=2, pause_time=0.5
        )
        urls = _match(
            await self._extract_conversation_thread_refs(
                limit=None, context="inbox", name_filter=display_name
            )
        )
        if urls:
            return urls

        # Fallback: LinkedIn's messaging search. Unreliable (often returns
        # "We didn't find anything" even for present threads, see #434), so it
        # runs only when the inbox scan came up empty — e.g. a thread buried
        # below the scrolled inbox window.
        await self._navigator._navigate_to_page(
            f"https://www.linkedin.com/messaging/?searchTerm={quote_plus(display_name)}"
        )
        await self._session.check_rate_limit()
        await self._session.dismiss_modal()
        await self._wait_for_main_text(log_context="Messaging search results")
        return _match(
            await self._extract_conversation_thread_refs(
                limit=None, context="search", name_filter=display_name
            )
        )

    async def _open_conversation_by_username(
        self, linkedin_username: str, index: int = 0
    ) -> None:
        """Open the ``index``-th conversation thread for the named participant.

        ``index`` is 0-based and orders threads as the search-results sidebar
        renders them (LinkedIn surfaces newest activity first).
        """
        if index < 0:
            raise LinkedInScraperException(f"index must be non-negative (got {index}).")

        linkedin_username = normalize_person_identifier(linkedin_username)
        profile_url = person_profile_url(linkedin_username, "/")
        await self._navigator._navigate_to_page(profile_url)
        await self._session.check_rate_limit()

        try:
            await self._session.page.wait_for_selector("main")
        except PlaywrightTimeoutError:
            logger.debug("Profile page did not load for %s", linkedin_username)

        await self._session.dismiss_modal()
        display_name = await self._profile_page._read_profile_display_name()
        if not display_name:
            raise LinkedInScraperException(
                f"Could not resolve a display name for {linkedin_username}."
            )

        try:
            thread_urls = await self._resolve_conversation_thread_urls(display_name)
            if not thread_urls:
                raise LinkedInScraperException(
                    f"Could not find a conversation for {linkedin_username}."
                )
            if index >= len(thread_urls):
                raise LinkedInScraperException(
                    f"index {index} out of range: only {len(thread_urls)} "
                    f"thread(s) exist for {linkedin_username}."
                )

            await self._navigator._navigate_to_page(thread_urls[index])
        except PlaywrightTimeoutError as exc:
            raise LinkedInScraperException(
                "Messaging search results did not load in time."
            ) from exc

    async def get_inbox(self, limit: int = 20) -> dict[str, Any]:
        """List recent conversations from the messaging inbox."""
        url = "https://www.linkedin.com/messaging/"
        await self._navigator._navigate_to_page(url)
        await self._session.check_rate_limit()
        await self._wait_for_main_text(log_context="Messaging inbox")
        await self._session.dismiss_modal()

        scrolls = max(1, limit // 10)
        await self._scroll_main_scrollable_region(
            position="bottom", attempts=scrolls, pause_time=0.5
        )

        raw_result = await self._content._extract_root_content(["main"])
        raw = raw_result["text"]
        cleaned = strip_linkedin_noise(raw) if raw else ""
        references: list[Reference] = (
            build_references(raw_result["references"], "inbox") if cleaned else []
        )

        # LinkedIn's conversation sidebar uses JS click handlers instead of
        # <a> tags, so anchor extraction cannot capture thread IDs.  Click each
        # conversation item and read the resulting SPA URL to build references.
        conversation_refs = await self._extract_conversation_thread_refs(
            limit=limit, context="inbox"
        )
        if conversation_refs:
            references = dedupe_references(conversation_refs + references)

        return self._single_section_result(
            url,
            "inbox",
            cleaned,
            references=references,
        )

    async def get_conversation(
        self,
        linkedin_username: str | None = None,
        thread_id: str | None = None,
        index: int = 0,
    ) -> dict[str, Any]:
        """Read a specific messaging conversation by thread ID or username.

        ``index`` (0-based) selects which thread to open when a participant has
        multiple conversation threads — e.g. an organic 1-on-1 plus a separate
        InMail. Ignored when ``thread_id`` is provided. Use
        ``search_conversations`` to enumerate thread IDs first if disambiguation
        by index is impractical.

        Side effect when looked up by username: resolution enumerates the
        messaging inbox and click-visits only the row(s) matching the
        participant's display name to capture the thread ID (no anchor hrefs or
        thread-id attributes exist in the sidebar). Each visit selects the row
        in the LinkedIn UI and may mark it as read. Pass ``thread_id`` directly
        to skip this enumeration.
        """
        if not linkedin_username and not thread_id:
            raise LinkedInScraperException(
                "Provide at least one of linkedin_username or thread_id"
            )

        if thread_id:
            thread_id = normalize_thread_id(thread_id)
            await self._navigator._navigate_to_page(
                messaging_thread_url(thread_id, "/")
            )
        else:
            await self._open_conversation_by_username(
                linkedin_username or "", index=index
            )

        await self._session.check_rate_limit()
        await self._wait_for_main_text(log_context="Conversation")
        await self._session.dismiss_modal()
        await self._scroll_main_scrollable_region(
            position="top", attempts=3, pause_time=0.5
        )

        raw_result = await self._content._extract_root_content(["main"])
        raw = raw_result["text"]
        # Conversation chrome first: a sidebar preview containing a generic
        # noise marker would otherwise truncate the page before the thread
        # markers are ever seen.
        cleaned = strip_conversation_chrome(raw) if raw else ""
        cleaned = strip_linkedin_noise(cleaned) if cleaned else ""
        references = (
            build_references(raw_result["references"], "conversation")
            if cleaned
            else []
        )
        return self._single_section_result(
            self._session.page.url,
            "conversation",
            cleaned,
            references=references,
        )

    async def search_conversations(
        self, keywords: str, limit: int = 20
    ) -> dict[str, Any]:
        """Search messages by keyword.

        Uses LinkedIn's ``?searchTerm=`` URL parameter to drive the search
        rather than typing into the searchbox — the URL form is reliable
        regardless of how soon the messaging SPA mounts its searchbox role,
        and (critically) preserves the search filter across click-to-capture
        navigations so per-thread refs can be enumerated.

        ``limit`` caps how many search-result rows the click-to-capture loop
        visits. Each visit selects the row in LinkedIn's UI (and may mark it
        as read), so a low cap is preferable for noisy queries.
        """
        search_url = (
            f"https://www.linkedin.com/messaging/?searchTerm={quote_plus(keywords)}"
        )
        await self._navigator._navigate_to_page(search_url)
        await self._session.check_rate_limit()
        await self._session.dismiss_modal()
        await self._wait_for_main_text(log_context="Messaging search")

        raw_result = await self._content._extract_root_content(["main"])
        raw = raw_result["text"]
        cleaned = strip_linkedin_noise(raw) if raw else ""
        references: list[Reference] = (
            build_references(raw_result["references"], "search_results")
            if cleaned
            else []
        )

        # Same click-to-capture path as get_inbox: LinkedIn's search sidebar
        # has no anchor hrefs or thread-id attributes, so the only way to
        # surface per-result thread IDs is to click each row and read the SPA
        # URL. URL-driven search keeps the filter active across clicks.
        conversation_refs = await self._extract_conversation_thread_refs(
            limit=limit, context="search_results"
        )
        if conversation_refs:
            references = dedupe_references(conversation_refs + references)

        return self._single_section_result(
            self._session.page.url,
            "search_results",
            cleaned,
            references=references,
        )

    async def _navigate_to_thread_for_options(
        self, conversation_url_or_thread_id: str
    ) -> tuple[str, str] | dict[str, Any]:
        """Open a thread by id and confirm it is the one that loaded.

        Returns ``(thread_id, url)`` on success, or a result dict callers can
        return directly on failure — mirroring the route-pinning discipline
        ``reply_to_conversation`` uses, since acting on the options menu of
        whatever page LinkedIn substituted would be worse than failing.
        """
        thread_id = normalize_thread_id(conversation_url_or_thread_id)
        url = messaging_thread_url(thread_id, "/")
        await self._navigator._navigate_to_page(url)
        await self._session.check_rate_limit()

        landed = urlparse(self._session.page.url)
        if landed.netloc != "www.linkedin.com" or not re.fullmatch(
            r"/messaging/thread/[A-Za-z0-9_=-]+", landed.path.rstrip("/")
        ):
            return {
                "url": self._session.page.url,
                "thread_id": thread_id,
                "status": "conversation_unavailable",
                "changed": False,
                "message": (
                    "LinkedIn did not open the requested conversation thread. "
                    "It may not exist, or this account may not have access "
                    "to it."
                ),
            }

        try:
            await self._session.page.wait_for_selector("main")
        except PlaywrightTimeoutError:
            logger.debug("Thread page did not load for %s", thread_id)
        await self._session.dismiss_modal()
        return thread_id, self._session.page.url

    async def _toggle_conversation_option(
        self,
        conversation_url_or_thread_id: str,
        *,
        confirm: bool,
        action_label: str,
        opposite_label: str,
        action_description: str,
    ) -> dict[str, Any]:
        """Open the thread's options menu and act on one toggle-labelled item.

        The menu item names the action it offers, not the current state (a
        thread already read offers "Mark as unread"), so whichever of
        ``action_label``/``opposite_label`` is present tells the two apart.
        Neither present, or the menu itself unreachable, fails closed as
        ``action_unavailable`` rather than guessing.
        """
        navigated = await self._navigate_to_thread_for_options(
            conversation_url_or_thread_id
        )
        if isinstance(navigated, dict):
            return navigated
        thread_id, url = navigated

        opened = False
        try:
            opened = await self._session.page.evaluate(
                _OPEN_CONVERSATION_OPTIONS_JS,
                CONVERSATION_OPTIONS_EN.menu_opener_prefix,
            )
        except Exception:
            logger.debug("Could not open the conversation options menu", exc_info=True)
        if opened:
            try:
                await self._session.page.wait_for_selector(
                    "[role='menu']", state="visible", timeout=3000
                )
            except PlaywrightTimeoutError:
                logger.debug("Conversation options menu did not appear")
                opened = False
        if not opened:
            return {
                "url": url,
                "thread_id": thread_id,
                "status": "action_unavailable",
                "changed": False,
                "message": (
                    "LinkedIn did not expose an options menu for this conversation."
                ),
            }

        try:
            items = await self._session.page.evaluate(_READ_CONVERSATION_MENU_ITEMS_JS)
            if not isinstance(items, list):
                items = []

            if action_label in items:
                if not confirm:
                    return {
                        "url": url,
                        "thread_id": thread_id,
                        "status": "preview",
                        "changed": False,
                        "message": f"Set confirm=true to {action_description}.",
                    }
                clicked = await self._session.page.evaluate(
                    _CLICK_CONVERSATION_MENU_ITEM_JS, action_label
                )
                if not clicked:
                    return {
                        "url": url,
                        "thread_id": thread_id,
                        "status": "action_unavailable",
                        "changed": False,
                        "message": (
                            "Could not click the LinkedIn conversation menu item."
                        ),
                    }
                return {
                    "url": url,
                    "thread_id": thread_id,
                    "status": "ok",
                    "changed": True,
                }

            if opposite_label in items:
                # LinkedIn is already offering the reverse action, which means
                # the requested state already holds. Nothing to click.
                return {
                    "url": url,
                    "thread_id": thread_id,
                    "status": "ok",
                    "changed": False,
                }

            return {
                "url": url,
                "thread_id": thread_id,
                "status": "action_unavailable",
                "changed": False,
                "message": (
                    "Could not find this option in LinkedIn's conversation menu."
                ),
            }
        finally:
            try:
                await self._session.page.keyboard.press("Escape")
            except Exception:
                logger.debug(
                    "Could not close the conversation options menu",
                    exc_info=True,
                )

    async def mark_conversation_read(
        self,
        conversation_url_or_thread_id: str,
        *,
        read: bool = True,
        confirm: bool,
    ) -> dict[str, Any]:
        """Mark a conversation thread read or unread via its options menu.

        Idempotent: a thread already in the requested state is reported
        without clicking anything, so a retry is always safe to make.
        """
        table = CONVERSATION_OPTIONS_EN
        action_label = table.mark_read if read else table.mark_unread
        opposite_label = table.mark_unread if read else table.mark_read
        action_description = (
            "mark this conversation as read"
            if read
            else "mark this conversation as unread"
        )
        result = await self._toggle_conversation_option(
            conversation_url_or_thread_id,
            confirm=confirm,
            action_label=action_label,
            opposite_label=opposite_label,
            action_description=action_description,
        )
        if result["status"] == "ok":
            result["read"] = read
        return result

    async def archive_conversation(
        self,
        conversation_url_or_thread_id: str,
        *,
        confirm: bool,
        unarchive: bool = False,
    ) -> dict[str, Any]:
        """Archive or unarchive a conversation thread via its options menu.

        Idempotent: a thread already in the requested state is reported
        without clicking anything, so a retry is always safe to make.
        """
        table = CONVERSATION_OPTIONS_EN
        action_label = table.unarchive if unarchive else table.archive
        opposite_label = table.archive if unarchive else table.unarchive
        action_description = (
            "unarchive this conversation" if unarchive else "archive this conversation"
        )
        result = await self._toggle_conversation_option(
            conversation_url_or_thread_id,
            confirm=confirm,
            action_label=action_label,
            opposite_label=opposite_label,
            action_description=action_description,
        )
        if result["status"] == "ok":
            result["archived"] = not unarchive
        return result
