"""
LinkedIn group scraping tools.

Uses innerText extraction for resilient group search, post, and member
listing capture -- read-only.
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


def register_group_tools(
    mcp: FastMCP, *, tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS
) -> None:
    """Register all group-related tools with the MCP server."""

    @mcp.tool(
        timeout=tool_timeout,
        title="Search Groups",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"group", "search", "read"},
        exclude_args=["extractor"],
    )
    async def search_groups(
        keywords: str,
        ctx: Context,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Search for LinkedIn groups by keyword.

        Single navigation, no pagination (mirrors search_people /
        search_companies).

        Args:
            keywords: Search keywords (e.g., "automation anywhere", "RPA UAE")
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with url, sections (search_results -> raw text), and
            optional references (group URLs classified as kind="group").
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="search_groups"
            )
            logger.info("Searching groups: keywords='%s'", keywords)
            await ctx.report_progress(progress=0, total=100, message="Searching groups")
            result = await extractor.search_groups(keywords)
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "search_groups")
        except Exception as e:
            raise_tool_error(e, "search_groups")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Group Posts",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"group", "scraping", "read"},
        exclude_args=["extractor"],
    )
    async def get_group_posts(
        group_id: str,
        ctx: Context,
        max_posts: Annotated[int, Field(ge=1, le=100)] = 20,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        List recent posts from a LinkedIn group's home feed.

        group_id is the numeric id from the group URL -- the path segment
        after /groups/ (e.g. "12345" for linkedin.com/groups/12345/). A
        full group URL is accepted too.

        Args:
            group_id: Numeric LinkedIn group id, or a group URL
            ctx: FastMCP context for progress reporting
            max_posts: Approximate cap on how many posts to load via bounded
                scrolling (1-100, default 20)

        Returns:
            Dict with url, sections (posts -> raw text), and optional
            references (post authors and attachments).
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_group_posts"
            )
            logger.info("Fetching group posts: %s (max_posts=%d)", group_id, max_posts)
            await ctx.report_progress(
                progress=0, total=100, message="Loading group posts"
            )
            result = await extractor.get_group_posts(group_id, max_posts=max_posts)
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_group_posts")
        except Exception as e:
            raise_tool_error(e, "get_group_posts")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Group Members",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"group", "scraping", "read"},
        exclude_args=["extractor"],
    )
    async def get_group_members(
        group_id: str,
        ctx: Context,
        max_members: Annotated[int, Field(ge=1, le=500)] = 50,
        keywords: str | None = None,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        List members of a LinkedIn group from its /members/ page.

        The full member list is only visible when the authenticated account
        is a member of the group. For groups the account has not joined,
        LinkedIn typically shows a restricted preview or redirects to the
        group landing page; the returned text reflects whatever the page
        actually served, so an unexpectedly short members section usually
        means the account is not in that group.

        group_id is the numeric id from the group URL -- the path segment
        after /groups/ (e.g. "12345" for linkedin.com/groups/12345/). A
        full group URL is accepted too.

        Args:
            group_id: Numeric LinkedIn group id, or a group URL
            ctx: FastMCP context for progress reporting
            max_members: Approximate cap on how many members to load via
                bounded scrolling (1-500, default 50)
            keywords: Optional server-side member filter (name, headline
                term) applied via the page's member search box

        Returns:
            Dict with url, sections (members -> raw text), and optional
            references (/in/ profile paths for listed members).
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_group_members"
            )
            logger.info(
                "Scraping group members: %s (max_members=%d, keywords=%s)",
                group_id,
                max_members,
                keywords,
            )
            await ctx.report_progress(
                progress=0, total=100, message="Loading group members"
            )
            result = await extractor.get_group_members(
                group_id, max_members=max_members, keywords=keywords
            )
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_group_members")
        except Exception as e:
            raise_tool_error(e, "get_group_members")  # NoReturn
