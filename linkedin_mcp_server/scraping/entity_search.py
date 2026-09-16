"""Shared ``&page=N`` pagination walk for LinkedIn people/company search.

Ported from stickerdaniel/linkedin-mcp-server#733 and #708 (people/company
search pagination), adapted to this fork's ``SectionCapture``/``CapturePlan``
capture seam and to the ``pages_fetched``/``stopped_reason`` result contract
used across this fork's own paginated tools (mirrors ``search_jobs``'s
``&start=`` walk in ``jobs.py``, and the stop-reason reporting idea from
stickerdaniel/linkedin-mcp-server#876).

People and company search page with ``&page=N`` (one-based, roughly ten
result cards each). There is no page-count element to read on these surfaces
(unlike job search's "Page X of Y"), so completeness is inferred the way
``search_jobs``'s own early-stop already works: a page that adds no entity
url not already seen ends the walk.
"""

from __future__ import annotations

from typing import Any

import asyncio
import logging
import time

from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.core.exceptions import LinkedInScraperException
from linkedin_mcp_server.error_diagnostics import build_issue_diagnostics
from linkedin_mcp_server.scraping.capture import (
    CaptureMode,
    CapturePlan,
    SectionCapture,
)
from linkedin_mcp_server.scraping.contracts import (
    RATE_LIMITED_SECTION_TEXT,
    rate_limited_section_error,
)
from linkedin_mcp_server.scraping.job_policy import SEARCH_TIMEOUT_FRACTION
from linkedin_mcp_server.scraping.link_metadata import Reference, dedupe_references
from linkedin_mcp_server.scraping.session import NAV_DELAY

logger = logging.getLogger(__name__)

# The tool-side ceiling every paginated entity-search tool validates
# ``max_pages`` against (``Field(ge=1, le=10)``).
MAX_ENTITY_SEARCH_PAGES = 10

# Per-page reference cap already applied to ``search_results`` by
# ``link_metadata.build_references``; the walk-wide cap multiplies it by the
# page ceiling so a full-depth walk keeps a reference for every distinct
# entity seen across every page requested rather than truncating to one
# page's worth.
_PER_PAGE_REFERENCE_CAP = 15
ENTITY_SEARCH_REFERENCE_CAP = _PER_PAGE_REFERENCE_CAP * MAX_ENTITY_SEARCH_PAGES

StoppedReason = str  # "max_pages" | "no_more_results" | "limit" | "error"


async def paginated_entity_search(
    capture: SectionCapture,
    base_url: str,
    *,
    entity_kind: str,
    max_pages: int,
    context: str,
    tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Walk ``&page=N`` over a people/company search, merging pages.

    Args:
        capture: Shared section capture for the navigation + extraction.
        base_url: The page-1 URL (no ``&page=``), already filter-validated.
        entity_kind: The ``Reference["kind"]`` that counts as one result
            ("person" or "company"); a page contributing no new url of this
            kind ends the walk.
        max_pages: How many pages to walk at most (1-10).
        context: Passed to ``build_issue_diagnostics`` for a failing page.
        tool_timeout: The registered MCP tool timeout, used to derive a
            wall-clock budget so a slow multi-page walk stops itself with
            time left to assemble a partial answer rather than being
            cancelled with nothing to return (same idea as ``search_jobs``'s
            ``SEARCH_TIMEOUT_FRACTION`` budget).

    Returns:
        {url, sections: {search_results: text}, pages_fetched: int,
        stopped_reason: "max_pages"|"no_more_results"|"limit"|"error",
        truncated: bool, references?, section_errors?}
    """
    page_texts: list[str] = []
    page_references: list[Reference] = []
    section_errors: dict[str, dict[str, Any]] = {}
    seen_entities: set[str] = set()
    pages_fetched = 0
    stopped_reason: StoppedReason = "max_pages"

    started = time.monotonic()
    budget = tool_timeout * SEARCH_TIMEOUT_FRACTION

    for offset in range(max_pages):
        page_num = offset + 1

        if offset > 0:
            if time.monotonic() - started > budget:
                logger.debug(
                    "Stopping %s after %d pages: time budget %.1fs spent",
                    context,
                    pages_fetched,
                    budget,
                )
                stopped_reason = "limit"
                break
            await asyncio.sleep(NAV_DELAY)

        url = base_url if page_num == 1 else f"{base_url}&page={page_num}"

        try:
            extracted = await capture.capture(
                url,
                section_name="search_results",
                plan=CapturePlan(CaptureMode.SEARCH_RESULTS),
            )
        except LinkedInScraperException:
            raise
        except Exception as e:
            logger.warning("Error on %s page %d: %s", context, page_num, e)
            section_errors["search_results"] = build_issue_diagnostics(
                e,
                context=context,
                target_url=url,
                section_name="search_results",
            )
            stopped_reason = "error"
            break

        if extracted.text == RATE_LIMITED_SECTION_TEXT:
            section_errors["search_results"] = rate_limited_section_error()
            stopped_reason = "error"
            break

        if not extracted.text:
            if extracted.error:
                section_errors["search_results"] = extracted.error
                stopped_reason = "error"
            else:
                stopped_reason = "no_more_results"
            break

        pages_fetched += 1
        new_entities = [
            ref["url"]
            for ref in extracted.references
            if ref.get("kind") == entity_kind and ref["url"] not in seen_entities
        ]
        page_texts.append(extracted.text)
        if extracted.references:
            page_references.extend(extracted.references)

        if not new_entities:
            logger.debug(
                "No new %s results on %s page %d, stopping",
                entity_kind,
                context,
                page_num,
            )
            stopped_reason = "no_more_results"
            break

        seen_entities.update(new_entities)

    references = (
        dedupe_references(page_references, cap=ENTITY_SEARCH_REFERENCE_CAP)
        if page_references
        else []
    )
    result: dict[str, Any] = {
        "url": base_url,
        "sections": {"search_results": "\n---\n".join(page_texts)}
        if page_texts
        else {},
        "pages_fetched": pages_fetched,
        "stopped_reason": stopped_reason,
        # A caller who asked for more pages than came back has to be able to
        # tell "LinkedIn had no more" from "the walk was cut short" without
        # comparing pages_fetched to the max_pages it happened to pass.
        "truncated": stopped_reason in ("max_pages", "limit"),
    }
    if references:
        result["references"] = {"search_results": references}
    if section_errors:
        result["section_errors"] = section_errors
    return result
