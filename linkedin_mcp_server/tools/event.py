"""
LinkedIn event scraping tools.

Read-only search, details, and attendee listing for LinkedIn events.
"""

import logging
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from pydantic import Field

from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.core.exceptions import AuthenticationError
from linkedin_mcp_server.dependencies import get_ready_extractor, handle_auth_error
from linkedin_mcp_server.error_handler import raise_tool_error

logger = logging.getLogger(__name__)


def register_event_tools(
    mcp: FastMCP, *, tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS
) -> None:
    """Register all event-related tools with the MCP server."""

    @mcp.tool(
        timeout=tool_timeout,
        title="Search Events",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"event", "search", "read"},
        exclude_args=["extractor"],
    )
    async def search_events(
        keywords: str,
        ctx: Context,
        max_pages: Annotated[int, Field(ge=1, le=10)] = 3,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Search for LinkedIn events by keyword.

        Single navigation with bounded scrolling -- there is no discrete
        per-page URL for this surface, so max_pages caps scroll depth
        rather than fetching distinct pages (the same convention
        search_posts uses for content search).

        Args:
            keywords: Search keywords (e.g., "automation anywhere conference")
            ctx: FastMCP context for progress reporting
            max_pages: Scroll-depth budget in nominal "pages" (1-10, default 3)

        Returns:
            Dict with url, sections (search_results -> raw text), and
            optional references (event URLs classified as kind="event").
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="search_events"
            )
            logger.info(
                "Searching events: keywords='%s', max_pages=%d", keywords, max_pages
            )
            await ctx.report_progress(progress=0, total=100, message="Searching events")
            result = await extractor.search_events(keywords, max_pages=max_pages)
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "search_events")
        except Exception as e:
            raise_tool_error(e, "search_events")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Event Details",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"event", "scraping", "read"},
        exclude_args=["extractor"],
    )
    async def get_event_details(
        event_url: str,
        ctx: Context,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Get the details page for a single LinkedIn event.

        event_url accepts either a full event URL (e.g.
        "https://www.linkedin.com/events/12345/") or a bare numeric event
        id -- LinkedIn serves an event under both a bare id and a slugged
        path, the same shape as a job posting.

        Args:
            event_url: LinkedIn event URL or numeric event id
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with url, sections (event_details -> raw text), and
            optional references. The LLM should parse the raw text for the
            event title, date/time, host, format (virtual/in-person), and
            description.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_event_details"
            )
            logger.info("Fetching event details: %s", event_url)
            await ctx.report_progress(
                progress=0, total=100, message="Loading event details"
            )
            result = await extractor.get_event_details(event_url)
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_event_details")
        except Exception as e:
            raise_tool_error(e, "get_event_details")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Event Attendees",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"event", "scraping", "read"},
        exclude_args=["extractor"],
    )
    async def get_event_attendees(
        event_url: str,
        ctx: Context,
        max_attendees: Annotated[int, Field(ge=1, le=500)] = 50,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        List attendees of a LinkedIn event.

        The attendee list is only fully visible when the authenticated
        account can see event attendance (public events, or events the
        account is registered for); an unexpectedly short attendees
        section usually means the account cannot see the full list.

        event_url accepts either a full event URL or a bare numeric event id.

        Args:
            event_url: LinkedIn event URL or numeric event id
            ctx: FastMCP context for progress reporting
            max_attendees: Approximate cap on how many attendees to load via
                bounded scrolling (1-500, default 50)

        Returns:
            Dict with url, sections (attendees -> raw text), and optional
            references (/in/ profile paths for listed attendees). Warm-intro
            use case: cross-reference attendees against get_mutual_connections
            before an event to plan introductions.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_event_attendees"
            )
            logger.info(
                "Fetching event attendees: %s (max_attendees=%d)",
                event_url,
                max_attendees,
            )
            await ctx.report_progress(
                progress=0, total=100, message="Loading event attendees"
            )
            result = await extractor.get_event_attendees(
                event_url, max_attendees=max_attendees
            )
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_event_attendees")
        except Exception as e:
            raise_tool_error(e, "get_event_attendees")  # NoReturn
