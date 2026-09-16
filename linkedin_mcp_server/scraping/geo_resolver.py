"""Free-text location resolution for LinkedIn's numeric geo URN facet.

People search's location facet (``geoUrn``, see ``search_urls.py``) only
filters on a numeric LinkedIn geo id; there is no stable public endpoint to
map a place name to one (the REST typeahead is gone and the people-search
box is an opaque server-driven-UI action). This resolves a name the way a
person does: drive the jobs-search page's own location typeahead -- a
stable on-page combobox -- and read the numeric ``geoId`` LinkedIn itself
puts in the resulting url once a suggestion is selected. That id doubles as
the people-search ``geoUrn`` (both facets share LinkedIn's one geo
taxonomy). Approach ported from stickerdaniel/linkedin-mcp-server#708 and
#710; the selector is inherited from their live measurement of the
jobs-search box, not reverified against a live session in this fork --
flag for live verification before relying on it in production.

Every step here goes through ``page.evaluate``/``page.wait_for_function``
rather than individual locator calls (``fill``, ``click``,
``wait_for_url``): those two are the only browser primitives this needs,
they are what the rest of this codebase already uses for structural DOM
reads (see ``_SIDEBAR_PROFILES_JS`` in ``person.py``), and setting the
input's value through React's own native-setter/dispatchEvent path is more
reliable against a controlled input than ``locator.fill()``, which many
such typeaheads never see as a real keystroke.

Never guesses among several suggestions. A query naming exactly one place
resolves silently; a query LinkedIn's own typeahead considers ambiguous
(more than one suggestion) comes back as an explicit candidate list, each
entry's id independently confirmed by actually selecting that specific
suggestion -- never inferred from its position in the list or by matching
its label text against the query, which is the one thing this module
refuses to do on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

import logging
import re

from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.scraping.contracts import FilterValidationError
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import NAV_DELAY, ScrapingSession

logger = logging.getLogger(__name__)

# The jobs-search typeahead renders suggestions from a network round trip;
# bounded so a stalled page reads as "no match" rather than hanging the tool
# call.
_TYPEAHEAD_TIMEOUT_MS = 5000
_JOBS_SEARCH_URL = "https://www.linkedin.com/jobs/search/?keywords="
_GEO_ID_RE = re.compile(r"[?&]geoId=(\d+)")

# Every candidate beyond the first costs its own fresh navigation (see
# ``_resolve_candidate``, which re-navigates to re-select by index once the
# first click has already carried the page to a ``geoId=`` url). This caps
# the cost of a broad query like "san" rather than resolving every
# suggestion LinkedIn is willing to render.
MAX_CANDIDATES = 6

# Sets the location box's value through its native setter and dispatches a
# real ``input`` event, so a React-controlled field notices the change the
# same way it would notice a keystroke -- setting ``.value`` directly (what
# ``locator.fill()`` does under the hood on some element types) is not
# guaranteed to trigger a controlled input's own state update.
_FILL_LOCATION_BOX_JS = """(query) => {
    const box = document.querySelector('input[id*="jobs-search-box-location"]');
    if (!box) return false;
    const setter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype, 'value'
    ).set;
    setter.call(box, query);
    box.dispatchEvent(new Event('input', { bubbles: true }));
    return true;
}"""

# The combobox pattern names its listbox in aria-controls (or the older
# aria-owns); scoping to it is what keeps this from matching some other
# role=option element on the jobs page (the keyword typeahead, a filter
# menu). Falls back to the whole document only when neither attribute is
# present, which a locale or markup change could produce.
_LISTBOX_HAS_OPTIONS_JS = """() => {
    const box = document.querySelector('input[id*="jobs-search-box-location"]');
    if (!box) return false;
    const listboxId = box.getAttribute('aria-controls') || box.getAttribute('aria-owns');
    const scope = listboxId ? document.getElementById(listboxId) : document;
    return !!scope && scope.querySelectorAll('[role="option"]').length > 0;
}"""

_READ_SUGGESTION_LABELS_JS = """() => {
    const box = document.querySelector('input[id*="jobs-search-box-location"]');
    if (!box) return [];
    const listboxId = box.getAttribute('aria-controls') || box.getAttribute('aria-owns');
    const scope = listboxId ? document.getElementById(listboxId) : document;
    if (!scope) return [];
    return Array.from(scope.querySelectorAll('[role="option"]'))
        .map((el) => (el.innerText || el.textContent || '').trim())
        .filter(Boolean);
}"""

_CLICK_SUGGESTION_AT_INDEX_JS = """(index) => {
    const box = document.querySelector('input[id*="jobs-search-box-location"]');
    if (!box) return false;
    const listboxId = box.getAttribute('aria-controls') || box.getAttribute('aria-owns');
    const scope = listboxId ? document.getElementById(listboxId) : document;
    if (!scope) return false;
    const option = scope.querySelectorAll('[role="option"]')[index];
    if (!option) return false;
    option.click();
    return true;
}"""

_URL_CARRIES_GEO_ID_JS = "() => /[?&]geoId=\\d+/.test(location.href)"
_CURRENT_URL_JS = "() => location.href"


@dataclass(frozen=True)
class GeoCandidate:
    """One place LinkedIn's own typeahead offered for a free-text query."""

    name: str
    geo_urn_id: str


@dataclass(frozen=True)
class GeoResolution:
    """The outcome of one resolution attempt.

    Exactly one of ``resolved`` or ``candidates`` is meaningful: ``resolved``
    is set when the query named exactly one place (no guessing was needed);
    ``candidates`` lists every place LinkedIn's own typeahead considered a
    match when there was more than one, each with its own independently
    confirmed id.
    """

    resolved: GeoCandidate | None = None
    candidates: tuple[GeoCandidate, ...] = ()

    @property
    def is_ambiguous(self) -> bool:
        return self.resolved is None and bool(self.candidates)

    @property
    def is_no_match(self) -> bool:
        return self.resolved is None and not self.candidates


class GeoLocationResolver:
    """Resolve a free-text place name to LinkedIn's numeric geo URN id."""

    def __init__(self, session: ScrapingSession, navigator: PageNavigator):
        self._session = session
        self._navigator = navigator

    async def resolve(self, query: str) -> GeoResolution:
        """Resolve ``query`` against LinkedIn's own location typeahead.

        Raises ``FilterValidationError`` for a blank query before any
        navigation -- there is nothing for the typeahead to be driven with.
        """
        query = query.strip()
        if not query:
            raise FilterValidationError(
                "location query must not be blank; pass a place name "
                "(e.g. 'Dubai', 'Saudi Arabia') or a numeric geo URN id."
            )

        labels = await self._open_and_list(query)
        if not labels:
            return GeoResolution()

        first = await self._select_by_index(0, labels[0])
        if len(labels) == 1:
            return GeoResolution(resolved=first) if first else GeoResolution()

        candidates: list[GeoCandidate] = [first] if first else []
        for index, label in enumerate(labels[1:MAX_CANDIDATES], start=1):
            await self._session.delay(NAV_DELAY)
            reopened = await self._open_and_list(query)
            if index >= len(reopened) or reopened[index] != label:
                logger.debug(
                    "Suggestion order changed on re-query for %r at index %d",
                    query,
                    index,
                )
                continue
            candidate = await self._select_by_index(index, label)
            if candidate is not None:
                candidates.append(candidate)
        return GeoResolution(candidates=tuple(candidates))

    async def _open_and_list(self, query: str) -> list[str]:
        """Navigate to a fresh jobs-search box, type ``query``, list options.

        A fresh navigation every time (rather than reusing one page across
        calls) because selecting a suggestion carries the page to a
        ``geoId=`` url, which leaves no location box behind to type into
        again.
        """
        await self._navigator._navigate_to_page(_JOBS_SEARCH_URL)
        page = self._session.page
        await self._session.check_rate_limit()
        opened = await page.evaluate(_FILL_LOCATION_BOX_JS, query)
        if not opened:
            logger.debug("Location box did not appear for query %r", query)
            return []
        try:
            await page.wait_for_function(
                _LISTBOX_HAS_OPTIONS_JS, timeout=_TYPEAHEAD_TIMEOUT_MS
            )
        except PlaywrightTimeoutError:
            # A dropdown that never opened is a stalled page as often as an
            # unknown name; this is "no match" either way, not an error.
            return []
        labels = await page.evaluate(_READ_SUGGESTION_LABELS_JS)
        return [label for label in labels if label]

    async def _select_by_index(self, index: int, label: str) -> GeoCandidate | None:
        """Click the option at ``index`` on the page already listing it, and
        read the ``geoId`` LinkedIn puts in the url as a result."""
        page = self._session.page
        clicked = await page.evaluate(_CLICK_SUGGESTION_AT_INDEX_JS, index)
        if not clicked:
            return None
        try:
            await page.wait_for_function(
                _URL_CARRIES_GEO_ID_JS, timeout=_TYPEAHEAD_TIMEOUT_MS
            )
        except PlaywrightTimeoutError:
            return None
        current_url = await page.evaluate(_CURRENT_URL_JS)
        match = _GEO_ID_RE.search(current_url or "")
        if match is None:
            return None
        return GeoCandidate(name=label, geo_urn_id=match.group(1))
