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
#710; the selector was inherited from their live measurement of the
jobs-search box and reverified against a live session in this fork on
2026-09-17 (see ``_open_and_list``'s docstring for what that confirmed).

Every step here goes through ``page.evaluate``/``page.wait_for_function``
rather than individual locator calls (``fill``, ``click``,
``wait_for_url``): those two are the only browser primitives this needs,
they are what the rest of this codebase already uses for structural DOM
reads (see ``_SIDEBAR_PROFILES_JS`` in ``person.py``), and setting the
input's value through React's own native-setter/dispatchEvent path is more
reliable against a controlled input than ``locator.fill()``, which many
such typeaheads never see as a real keystroke.

Never guesses among several suggestions by *position*. A query naming
exactly one place resolves silently; a query LinkedIn's own typeahead
considers ambiguous (more than one suggestion) comes back as an explicit
candidate list, each entry's id independently confirmed by actually
selecting that specific suggestion -- never inferred from its position in
the list.

One narrow exception to "always return the full candidate list when there
is more than one suggestion": once every candidate's id has been
independently confirmed the same way as above, a single candidate whose
*name* -- not position -- either equals the query outright or is the
query's canonical "query, single region" form (see ``_find_exact_match``)
is returned as ``resolved`` rather than folded into an ambiguous list of
one obviously-intended entry plus decoys. This is still not guessing: it
never picks among competing plausible candidates (two exact/canonical
matches, e.g. a country and a same-named US state, stay ambiguous), and
the id it returns was already confirmed by selecting that exact
suggestion, not inferred from the label.

A query that a fresh typeahead renders no suggestions for is retried once
with a shorter prefix of the query (dropping the last word, or -- for a
single unsplittable word -- taking its first half) before being reported
as no match; live observation (2026-09-17) showed the full phrase "United
Arab Emirates" reliably rendering zero suggestions through the
programmatic value-set this module uses (see ``_FILL_LOCATION_BOX_JS``)
while the same typeahead resolves "Dubai" and "Saudi Arabia" without
trouble, so the failure is query-shape-dependent, not account- or
session-specific. The retry's suggestions are filtered to those whose
label still contains the *original* query text (case/diacritic/whitespace
-insensitive) before being treated as answers, so a broadened prefix like
"United" can never surface an unrelated country ("United Kingdom",
"United States") as if it were a match for the original phrase.
"""

from __future__ import annotations

from dataclasses import dataclass

import logging
import re
import unicodedata

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

# The listbox populating is itself a network round trip layered on top of the
# page navigation above; a single 5s wait occasionally loses that race on a
# slow response. Bounded at two attempts (never unbounded polling) -- between
# them the query is re-filled in case the first ``input`` dispatch landed
# before the combobox had finished wiring itself up, which would otherwise
# leave the retry waiting on an event that already fired.
_LISTBOX_POLL_ATTEMPTS = 2

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


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(value: str) -> str:
    """Fold a name for comparison: casefold, strip diacritics, collapse space.

    NFKD decomposition splits a base letter from its combining diacritic
    (``e`` + combining-acute), and the diacritic's Unicode category is
    always ``Mn`` (nonspacing mark), so filtering that category leaves the
    plain letters -- e.g. turns "São Paulo" into "sao paulo" -- without a
    hand-maintained transliteration table.
    """
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return _WHITESPACE_RE.sub(" ", without_marks).strip().casefold()


def _find_exact_match(
    query: str, candidates: tuple[GeoCandidate, ...]
) -> GeoCandidate | None:
    """Find the one candidate a person would call an unambiguous answer.

    Two shapes count, both compared case/diacritic/whitespace-insensitively:

    * the candidate's whole name equals the query outright (e.g. querying
      "Saudi Arabia" against a candidate literally named "Saudi Arabia");
    * the candidate is the query's canonical "<query>, <single region>"
      form -- exactly one comma, and the text before it equals the query
      (e.g. querying "Dubai" against "Dubai, United Arab Emirates").

    Deliberately not a prefix match: "Dubai, Dubai, United Arab Emirates"
    (an emirate-level entry LinkedIn's own typeahead offers alongside the
    city) starts with "Dubai," too but has two commas, not one, so it is
    correctly excluded rather than tying with the real answer. A query
    matching more than one candidate this way (e.g. "Georgia" the country
    equals one candidate outright while "Georgia, United States" is
    simultaneously that query's canonical form) is genuinely ambiguous and
    returns ``None`` rather than picking either -- this function only ever
    resolves a query that has exactly one candidate satisfying either shape.
    """
    normalized_query = _normalize(query)
    matches = []
    for candidate in candidates:
        normalized_name = _normalize(candidate.name)
        if normalized_name == normalized_query:
            matches.append(candidate)
            continue
        segments = candidate.name.split(",")
        if len(segments) == 2 and _normalize(segments[0]) == normalized_query:
            matches.append(candidate)
    return matches[0] if len(matches) == 1 else None


def _shorter_prefix(query: str) -> str | None:
    """A shorter form of ``query`` worth trying once the full text yields no
    suggestions -- dropping the last word for a multi-word phrase (LinkedIn's
    own typeahead matched "Dubai" and "Saudi Arabia" live but not the
    three-word "United Arab Emirates" through this module's programmatic
    value-set), or the first half of a single word too short to split.

    Returns ``None`` when there is no meaningfully shorter form left to try
    (a single word of 3 characters or fewer), so the caller can stop rather
    than retry with something no shorter than the original.
    """
    words = query.split()
    if len(words) > 1:
        shortened = " ".join(words[:-1])
        return shortened or None
    word = words[0] if words else query
    if len(word) <= 3:
        return None
    return word[: max(3, len(word) // 2)]


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

        # ``effective_query`` is what the *page* currently has open -- the
        # original query normally, or the shorter-prefix retry's text once
        # that path is taken. ``indices`` are positions into ``labels`` (the
        # listbox exactly as LinkedIn rendered it), so clicking always hits
        # the option actually at that position even after ``labels`` has
        # been filtered down to the ones relevant to the original query.
        effective_query = query
        labels = await self._open_and_list(effective_query)
        indices = list(range(len(labels)))

        if not indices:
            prefix = _shorter_prefix(query)
            if prefix:
                logger.debug(
                    "No suggestions for %r; retrying once with shorter prefix %r",
                    query,
                    prefix,
                )
                await self._session.delay(NAV_DELAY)
                retried = await self._open_and_list(prefix)
                normalized_query = _normalize(query)
                relevant = [
                    i
                    for i, label in enumerate(retried)
                    if normalized_query in _normalize(label)
                ]
                if relevant:
                    effective_query = prefix
                    labels = retried
                    indices = relevant

        if not indices:
            # Exhausted the direct query and the one shorter-prefix retry
            # with nothing relevant either time -- an honest no-match, not a
            # guess and not a silently swallowed failure.
            return GeoResolution()

        indices = indices[:MAX_CANDIDATES]
        first_index = indices[0]
        first = await self._select_by_index(first_index, labels[first_index])
        if len(indices) == 1:
            return GeoResolution(resolved=first) if first else GeoResolution()

        candidates: list[GeoCandidate] = [first] if first else []
        for index in indices[1:]:
            label = labels[index]
            await self._session.delay(NAV_DELAY)
            reopened = await self._open_and_list(effective_query)
            if index >= len(reopened) or reopened[index] != label:
                logger.debug(
                    "Suggestion order changed on re-query for %r at index %d",
                    effective_query,
                    index,
                )
                continue
            candidate = await self._select_by_index(index, label)
            if candidate is not None:
                candidates.append(candidate)

        exact_match = _find_exact_match(query, tuple(candidates))
        if exact_match is not None:
            return GeoResolution(resolved=exact_match)
        return GeoResolution(candidates=tuple(candidates))

    async def _open_and_list(self, query: str) -> list[str]:
        """Navigate to a fresh jobs-search box, type ``query``, list options.

        A fresh navigation every time (rather than reusing one page across
        calls) because selecting a suggestion carries the page to a
        ``geoId=`` url, which leaves no location box behind to type into
        again. The location box itself is confirmed live (2026-09-17): its
        id is ``jobs-search-box-location-id-<ember-suffix>``, matching the
        ``input[id*="jobs-search-box-location"]`` selector below, and it
        exposes no ``aria-controls``/``aria-owns`` to its listbox -- so the
        document-wide fallback in ``_LISTBOX_HAS_OPTIONS_JS`` and friends is
        the live path, not a defensive one that never actually runs.
        """
        await self._navigator._navigate_to_page(_JOBS_SEARCH_URL)
        page = self._session.page
        await self._session.check_rate_limit()
        opened = await page.evaluate(_FILL_LOCATION_BOX_JS, query)
        if not opened:
            logger.debug("Location box did not appear for query %r", query)
            return []
        for attempt in range(1, _LISTBOX_POLL_ATTEMPTS + 1):
            try:
                await page.wait_for_function(
                    _LISTBOX_HAS_OPTIONS_JS, timeout=_TYPEAHEAD_TIMEOUT_MS
                )
                break
            except PlaywrightTimeoutError:
                if attempt >= _LISTBOX_POLL_ATTEMPTS:
                    # A dropdown that never opened is a stalled page as often
                    # as an unknown name; this is "no match" either way, not
                    # an error.
                    return []
                logger.debug(
                    "Listbox not populated yet for %r (attempt %d/%d);"
                    " re-filling and retrying",
                    query,
                    attempt,
                    _LISTBOX_POLL_ATTEMPTS,
                )
                await page.evaluate(_FILL_LOCATION_BOX_JS, query)
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
