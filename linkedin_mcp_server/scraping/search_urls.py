"""URL grammar and filter validation for LinkedIn's four search surfaces."""

from __future__ import annotations

from urllib.parse import quote_plus

import json
import re

from linkedin_mcp_server.scraping.contracts import FilterValidationError

# Normalization maps for job search filters. Job search encodes recency as
# ``f_TPR=r<seconds>``; content search uses named tokens, hence the separate
# ``CONTENT_DATE_POSTED_MAP`` below.
JOB_DATE_POSTED_MAP = {
    "past_hour": "r3600",
    "past_24_hours": "r86400",
    "past_week": "r604800",
    "past_month": "r2592000",
}

EXPERIENCE_LEVEL_MAP = {
    "internship": "1",
    "entry": "2",
    "associate": "3",
    "mid_senior": "4",
    "director": "5",
    "executive": "6",
}

JOB_TYPE_MAP = {
    "full_time": "F",
    "part_time": "P",
    "contract": "C",
    "temporary": "T",
    "volunteer": "V",
    "internship": "I",
    "other": "O",
}

WORK_TYPE_MAP = {"on_site": "1", "remote": "2", "hybrid": "3"}

SORT_BY_MAP = {"date": "DD", "relevance": "R"}

# Content (post) search uses literal ``datePosted`` tokens inside a JSON-list
# facet, e.g. ``datePosted=["past-week"]`` — unlike job search, which uses
# ``f_TPR=r<seconds>`` codes. The three hyphenated values are LinkedIn's
# complete set, verified live: the filter dropdown offers exactly Past 24
# hours / week / month, and anything else is ignored while still being echoed
# back in the url, so a near-miss spelling returns unfiltered results that
# look filtered. The underscore keys are this server's own spelling, carried
# over so ``date_posted`` reads the same here as in ``search_jobs``
# (``JOB_DATE_POSTED_MAP``); ``past_hour`` has no content-search equivalent.
CONTENT_DATE_POSTED_MAP = {
    "past-24h": "past-24h",
    "past_24_hours": "past-24h",
    "past-week": "past-week",
    "past_week": "past-week",
    "past-month": "past-month",
    "past_month": "past-month",
}

# Content-search ``sortBy`` facet tokens. Ported from
# stickerdaniel/linkedin-mcp-server#880, which verified live that LinkedIn's
# Posts tab encodes sort the same way as ``datePosted`` -- a one-element JSON
# list carrying a literal token (``sortBy=["date_posted"]`` /
# ``["relevance"]``) -- not the ``sortBy=DD|R`` codes job search uses. "latest"
# is this server's own spelling (matching the tool's documented values);
# "date"/"date_posted" are accepted too so the naming lines up with
# ``search_jobs``'s ``sort_by``.
CONTENT_SORT_BY_MAP = {
    "relevance": "relevance",
    "latest": "date_posted",
    "date": "date_posted",
    "date_posted": "date_posted",
}

# Valid tokens for the people-search ``network`` facet.
# LinkedIn accepts "F" (1st-degree), "S" (2nd-degree), "O" (3rd-degree and beyond).
NETWORK_TOKENS = ("F", "S", "O")

# People-search facets whose value is a numeric LinkedIn URN id, validated and
# encoded identically to the pre-existing ``current_company`` facet. Not
# verified live in this fork (unlike the facets above, which each carry a
# "verified live" note) -- these are the widely documented parameter names
# for LinkedIn's People search "All filters" panel. Flag for live
# verification before relying on them in production.
_URN_LIST_FACETS = {
    "current_company": "currentCompany",
    "past_company": "pastCompany",
    "school": "school",
    "industry": "industry",
}

# ISO 639-1 two-letter codes, lowercase. LinkedIn's people-search
# ``profileLanguage`` facet takes language codes rather than numeric ids
# (unlike every other facet below it) -- not verified live in this fork.
_PROFILE_LANGUAGE_RE = re.compile(r"^[a-z]{2}$")


def _normalize_csv(value: str, mapping: dict[str, str]) -> str:
    """Normalize a comma-separated filter value using the provided mapping."""
    parts = [v.strip() for v in value.split(",")]
    return ",".join(mapping.get(p, p) for p in parts)


def _encode_list_facet(values: list[str]) -> str:
    """Encode a list of string values for a LinkedIn search list facet.

    LinkedIn's people- and content-search URLs use JSON-list encoded facets of
    the form ``["A","B"]``. This helper URL-encodes the rendered JSON so the
    final URL contains e.g. ``%5B%22F%22%5D`` for ``["F"]``.
    """
    return quote_plus(json.dumps(values, separators=(",", ":")))


def build_job_search_url(
    keywords: str,
    location: str | None = None,
    date_posted: str | None = None,
    job_type: str | None = None,
    experience_level: str | None = None,
    work_type: str | None = None,
    easy_apply: bool = False,
    sort_by: str | None = None,
) -> str:
    """Build a LinkedIn job search URL with optional filters.

    Human-readable names are normalized to LinkedIn URL codes.
    Comma-separated values are normalized individually.
    Unknown values pass through unchanged.
    """
    params = f"keywords={quote_plus(keywords)}"
    if location:
        params += f"&location={quote_plus(location)}"

    if date_posted:
        mapped = JOB_DATE_POSTED_MAP.get(date_posted.strip(), date_posted)
        params += f"&f_TPR={quote_plus(mapped)}"
    if job_type:
        params += f"&f_JT={_normalize_csv(job_type, JOB_TYPE_MAP)}"
    if experience_level:
        params += f"&f_E={_normalize_csv(experience_level, EXPERIENCE_LEVEL_MAP)}"
    if work_type:
        params += f"&f_WT={_normalize_csv(work_type, WORK_TYPE_MAP)}"
    if easy_apply:
        params += "&f_EA=true"
    if sort_by:
        mapped = SORT_BY_MAP.get(sort_by.strip(), sort_by)
        params += f"&sortBy={quote_plus(mapped)}"

    return f"https://www.linkedin.com/jobs/search/?{params}"


def _validate_urn_list(name: str, values: list[str]) -> None:
    """Refuse a URN-id facet list carrying anything but ASCII digits.

    Shared by every numeric-id people-search facet (``current_company`` and
    the prospecting facets below it), so a caller sees the same actionable
    message regardless of which one they got wrong.
    """
    invalid = [v for v in values if not re.fullmatch(r"[0-9]+", v)]
    if invalid:
        raise FilterValidationError(
            f"{name} values must be numeric LinkedIn URN ids (e.g. '1115' "
            f"for the SAP company URN); got {invalid!r}. Plain-text names "
            f"are silently ignored by LinkedIn. Look up a company/school URN "
            f'via get_company_profile -> references["about"] (or the '
            f"equivalent school reference); there is no in-app lookup for "
            f"industry codes -- read them off LinkedIn's own People search "
            f"'Industry' filter panel."
        )


def build_people_search_url(
    keywords: str,
    location: str | None = None,
    network: list[str] | None = None,
    current_company: list[str] | None = None,
    past_company: list[str] | None = None,
    school: list[str] | None = None,
    industry: list[str] | None = None,
    title: str | None = None,
    profile_language: list[str] | None = None,
) -> str:
    """Build a LinkedIn people search URL, refusing filters LinkedIn ignores.

    Every refusal happens before a URL exists, so a workflow calling this can
    never navigate on a filter LinkedIn would swallow. An unknown ``network``
    token, a plain-text company/school/industry id, or a free-text
    ``location`` are each accepted by the URL and then dropped, which answers
    with the unfiltered result set while the request still reads as filtered.

    ``current_company``, ``past_company``, ``school`` and ``industry`` all
    take numeric LinkedIn URN ids (see ``_validate_urn_list``); ``location``
    takes a single numeric geo URN id (LinkedIn's ``geoUrn`` facet -- the
    free-text ``location`` param it also accepts is silently ignored, ported
    from stickerdaniel/linkedin-mcp-server#722); ``profile_language`` takes
    ISO 639-1 two-letter codes; ``title`` is free text matched against the
    member's current/past title, sent as-is like ``keywords``.
    """
    if network is not None:
        invalid = [t for t in network if t not in NETWORK_TOKENS]
        if invalid:
            raise FilterValidationError(
                "Invalid network token(s) "
                f"{invalid!r}; expected any of {list(NETWORK_TOKENS)!r}"
            )

    if location and not re.fullmatch(r"[0-9]+", location):
        raise FilterValidationError(
            f"location must be a numeric LinkedIn geo URN id (e.g. "
            f"'103644278' for the United States); got {location!r}. "
            f"LinkedIn's people-search geo facet only filters on the URN id; "
            f"plain-text place names are silently ignored and return the "
            f"unfiltered result set. Omit the filter and put the place name "
            f"in `keywords` if you do not have the URN."
        )

    urn_list_facets = {
        "current_company": current_company,
        "past_company": past_company,
        "school": school,
        "industry": industry,
    }
    for name, values in urn_list_facets.items():
        if values:
            _validate_urn_list(name, values)

    if profile_language:
        invalid = [
            code
            for code in profile_language
            if not _PROFILE_LANGUAGE_RE.fullmatch(code)
        ]
        if invalid:
            raise FilterValidationError(
                f"profile_language values must be lowercase ISO 639-1 "
                f"two-letter codes (e.g. 'en', 'ar'); got {invalid!r}."
            )

    params = f"keywords={quote_plus(keywords)}"
    if location:
        params += f"&geoUrn={_encode_list_facet([location])}"
    if network:
        params += f"&network={_encode_list_facet(network)}"
    for name, values in urn_list_facets.items():
        if values:
            params += f"&{_URN_LIST_FACETS[name]}={_encode_list_facet(values)}"
    if title and title.strip():
        params += f"&title={quote_plus(title)}"
    if profile_language:
        params += f"&profileLanguage={_encode_list_facet(profile_language)}"

    return f"https://www.linkedin.com/search/results/people/?{params}"


def build_company_search_url(keywords: str) -> str:
    """Build a LinkedIn company search URL.

    One parameter today, and a function anyway: every search URL this server
    navigates to is then written in one place, so the next caller has nothing
    to assemble by hand and no second spelling of the host and route to keep
    in step with the other three.
    """
    return (
        "https://www.linkedin.com/search/results/companies/"
        f"?keywords={quote_plus(keywords)}"
    )


def build_group_search_url(keywords: str) -> str:
    """Build a LinkedIn group search URL.

    Modeled on ``build_company_search_url``: one parameter, single
    navigation, no pagination -- matching the existing ``search_people`` /
    ``search_companies`` pattern for a keyword-only surface.
    """
    return (
        "https://www.linkedin.com/search/results/groups/"
        f"?keywords={quote_plus(keywords)}"
    )


def build_content_search_url(
    keywords: str,
    date_posted: str | None = None,
    sort_by: str | None = None,
) -> str:
    """Build a LinkedIn content (post) search URL.

    Reproduces the ``FACETED_SEARCH`` URL LinkedIn produces from the
    Posts results tab, e.g. for "Buscamos Unity" in the past week:
    ``/search/results/content/?keywords=Buscamos+Unity&origin=FACETED_SEARCH&datePosted=%5B%22past-week%22%5D``

    The ``datePosted`` and ``sortBy`` facets are each a one-element JSON list
    carrying a literal LinkedIn token, URL-encoded — unlike job search, which
    uses ``f_TPR=r<seconds>`` / ``sortBy=DD|R``. Values are mapped through
    ``CONTENT_DATE_POSTED_MAP`` / ``CONTENT_SORT_BY_MAP`` (ported from
    stickerdaniel/linkedin-mcp-server#880) so the server's own spelling
    reaches LinkedIn in the form it recognizes. An unmapped value is refused
    here rather than sent, because LinkedIn ignores one instead of rejecting
    it and answers an unfiltered search that reads as a filtered one.
    """
    if (
        date_posted is not None
        and date_posted.strip()
        and date_posted.strip() not in CONTENT_DATE_POSTED_MAP
    ):
        raise FilterValidationError(
            f"Invalid date_posted {date_posted!r}; expected one of "
            f"{list(CONTENT_DATE_POSTED_MAP)!r}."
        )

    if (
        sort_by is not None
        and sort_by.strip()
        and sort_by.strip() not in CONTENT_SORT_BY_MAP
    ):
        raise FilterValidationError(
            f"Invalid sort_by {sort_by!r}; expected one of "
            f"{list(CONTENT_SORT_BY_MAP)!r}."
        )

    params = f"keywords={quote_plus(keywords)}&origin=FACETED_SEARCH"
    if date_posted and date_posted.strip():
        token = CONTENT_DATE_POSTED_MAP.get(date_posted.strip(), date_posted.strip())
        params += f"&datePosted={_encode_list_facet([token])}"
    if sort_by and sort_by.strip():
        token = CONTENT_SORT_BY_MAP.get(sort_by.strip(), sort_by.strip())
        params += f"&sortBy={_encode_list_facet([token])}"
    return f"https://www.linkedin.com/search/results/content/?{params}"
