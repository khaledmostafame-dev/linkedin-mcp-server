"""
LinkedIn Sales Navigator tools: leads, accounts and list search (read-only).

Requires the authenticated account to hold a Sales Navigator seat. An
account without one is redirected away from Sales Navigator by LinkedIn
itself; every tool here surfaces that as a `sales_navigator_unavailable`
section_error rather than an exception, so a client can tell "no seat" apart
from "no results" or an ordinary scraping failure.

No InMail, connect, save, or list-membership writes are exposed here.
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


def register_sales_navigator_tools(
    mcp: FastMCP, *, tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS
) -> None:
    """Register all Sales Navigator tools with the MCP server."""

    @mcp.tool(
        timeout=tool_timeout,
        title="Search Sales Navigator Leads",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"sales-navigator", "search"},
        exclude_args=["extractor"],
    )
    async def sales_nav_search_leads(
        keywords: str,
        ctx: Context,
        filters: dict[str, Any] | None = None,
        max_pages: Annotated[int, Field(ge=1, le=10)] = 1,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Search Sales Navigator leads (people) by keyword and optional filters.

        Requires the authenticated account to hold a Sales Navigator seat.

        Args:
            keywords: Search keywords (e.g., "VP Engineering fintech")
            ctx: FastMCP context for progress reporting
            filters: Optional filter key/value pairs appended to the search
                URL. The exact filter names Sales Navigator accepts have not
                been verified live; pass LinkedIn's own URL parameter names.
            max_pages: Maximum number of result pages to load (1-10, default 1)

        Returns:
            Dict with url, sections (search_results -> raw text),
            pages_fetched, stopped_reason, and optional references /
            section_errors. section_errors["search_results"] carries
            error_type "sales_navigator_unavailable" when the account has
            no Sales Navigator seat.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="sales_nav_search_leads"
            )
            logger.info(
                "Searching Sales Navigator leads: keywords='%s', max_pages=%d",
                keywords,
                max_pages,
            )

            await ctx.report_progress(
                progress=0, total=100, message="Starting Sales Navigator lead search"
            )

            result = await extractor.sales_nav_search_leads(
                keywords, filters=filters, max_pages=max_pages
            )

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "sales_nav_search_leads")
        except Exception as e:
            raise_tool_error(e, "sales_nav_search_leads")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Search Sales Navigator Accounts",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"sales-navigator", "search"},
        exclude_args=["extractor"],
    )
    async def sales_nav_search_accounts(
        keywords: str,
        ctx: Context,
        filters: dict[str, Any] | None = None,
        max_pages: Annotated[int, Field(ge=1, le=10)] = 1,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Search Sales Navigator accounts (companies) by keyword and optional filters.

        Requires the authenticated account to hold a Sales Navigator seat.

        Args:
            keywords: Search keywords (e.g., "fintech Series B")
            ctx: FastMCP context for progress reporting
            filters: Optional filter key/value pairs appended to the search
                URL. The exact filter names Sales Navigator accepts have not
                been verified live; pass LinkedIn's own URL parameter names.
            max_pages: Maximum number of result pages to load (1-10, default 1)

        Returns:
            Same shape as sales_nav_search_leads.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="sales_nav_search_accounts"
            )
            logger.info(
                "Searching Sales Navigator accounts: keywords='%s', max_pages=%d",
                keywords,
                max_pages,
            )

            await ctx.report_progress(
                progress=0,
                total=100,
                message="Starting Sales Navigator account search",
            )

            result = await extractor.sales_nav_search_accounts(
                keywords, filters=filters, max_pages=max_pages
            )

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "sales_nav_search_accounts")
        except Exception as e:
            raise_tool_error(e, "sales_nav_search_accounts")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Sales Navigator Lists",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"sales-navigator", "lists"},
        exclude_args=["extractor"],
    )
    async def sales_nav_get_lists(
        ctx: Context,
        kind: str = "leads",
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        List the authenticated user's Sales Navigator lead or account lists.

        Requires the authenticated account to hold a Sales Navigator seat.

        Args:
            ctx: FastMCP context for progress reporting
            kind: "leads" or "accounts" (default "leads")

        Returns:
            Dict with url, sections (lists -> raw text), and optional
            references (list URLs, usable with sales_nav_get_list) /
            section_errors.
        """
        try:
            if kind not in ("leads", "accounts"):
                raise ValueError(f"kind must be 'leads' or 'accounts', got {kind!r}")
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="sales_nav_get_lists"
            )
            logger.info("Fetching Sales Navigator %s lists", kind)

            await ctx.report_progress(
                progress=0, total=100, message="Loading Sales Navigator lists"
            )

            result = await extractor.sales_nav_get_lists(kind)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "sales_nav_get_lists")
        except Exception as e:
            raise_tool_error(e, "sales_nav_get_lists")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Sales Navigator List",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"sales-navigator", "lists"},
        exclude_args=["extractor"],
    )
    async def sales_nav_get_list(
        list_url: str,
        ctx: Context,
        max_items: Annotated[int, Field(ge=1, le=500)] = 100,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Read one Sales Navigator list's members, bounded by max_items.

        Requires the authenticated account to hold a Sales Navigator seat.

        Args:
            list_url: A Sales Navigator list URL, e.g. one returned by
                sales_nav_get_lists in references["lists"]. A relative path
                is joined against linkedin.com.
            ctx: FastMCP context for progress reporting
            max_items: Stop once this many members have been collected
                (1-500, default 100)

        Returns:
            Dict with url, sections (list_members -> raw text),
            pages_fetched, stopped_reason, and optional references (member
            URLs, capped at max_items) / section_errors.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="sales_nav_get_list"
            )
            logger.info(
                "Fetching Sales Navigator list %s (max_items=%d)",
                list_url,
                max_items,
            )

            await ctx.report_progress(
                progress=0, total=100, message="Loading Sales Navigator list"
            )

            result = await extractor.sales_nav_get_list(list_url, max_items)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "sales_nav_get_list")
        except Exception as e:
            raise_tool_error(e, "sales_nav_get_list")  # NoReturn
