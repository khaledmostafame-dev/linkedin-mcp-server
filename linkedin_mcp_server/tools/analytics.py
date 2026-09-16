"""
LinkedIn analytics tools for the signed-in member's own content and profile.

Both tools read "private to you" pages. A page LinkedIn will not show this
account is reported as a section error decided by the URL it redirected to,
never inferred from the page's text.
"""

import logging
from typing import Any

from fastmcp import Context, FastMCP

from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.core.exceptions import AuthenticationError
from linkedin_mcp_server.dependencies import get_ready_extractor, handle_auth_error
from linkedin_mcp_server.error_handler import raise_tool_error
from linkedin_mcp_server.scraping.analytics import parse_profile_analytics_sections
from linkedin_mcp_server.scraping.identifiers import normalize_post_urn
from linkedin_mcp_server.core.exceptions import InvalidReferenceError

logger = logging.getLogger(__name__)


def register_analytics_tools(
    mcp: FastMCP, *, tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS
) -> None:
    """Register the member's own analytics tools with the MCP server."""

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Post Analytics",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"analytics", "read"},
        exclude_args=["extractor"],
    )
    async def get_post_analytics(
        post_url: str,
        ctx: Context,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Read the analytics page of one of your own LinkedIn posts.

        Covers what LinkedIn shows there: impressions, members reached,
        reactions, comments, reposts, saves, profile viewers and followers
        gained from the post, and audience demographics (job title, location,
        industry, seniority, company size) when LinkedIn has enough data.
        Only the signed-in member's own posts have analytics; for anyone
        else's post the result carries section_errors["post_analytics"] with
        error_type "not_authorized" and no text.

        Args:
            post_url: The post, as /feed/update/urn:li:activity:<id>/, a
                /posts/<slug> permalink, or urn:li:activity|ugcPost|share:<id>.
                A ugcPost or share URN costs one extra navigation to read the
                post's own analytics link.
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with url, sections["post_analytics"] (raw page text),
            metrics["post_analytics"] (list of {label, value}: each count read
            from the page with the label printed beside it, digits in any
            script), named_metrics["post_analytics"] ({impressions,
            members_reached, reactions, comments, reposts, saves, …} — English
            UI only; otherwise parse the labels), and section_errors when the
            page was not available.
        """
        try:
            normalize_post_urn(post_url)
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_post_analytics"
            )
            logger.info("Reading post analytics for %s", post_url)
            await ctx.report_progress(
                progress=0, total=100, message="Loading post analytics"
            )
            result = await extractor.get_post_analytics(post_url)
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_post_analytics")
        except Exception as e:
            raise_tool_error(e, "get_post_analytics")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Profile Analytics",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"analytics", "read"},
        exclude_args=["extractor"],
    )
    async def get_profile_analytics(
        ctx: Context,
        sections: str | None = None,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Read your own LinkedIn profile and creator analytics dashboards.

        Each section is its own page navigation, so request only what you need.

        Args:
            ctx: FastMCP context for progress reporting
            sections: Comma-separated sections; default is all four.
                - profile_viewers: who viewed your profile (/analytics/profile-views/)
                - search_appearances: search appearance counts and sources
                - followers: total followers, growth and follower demographics
                  (/analytics/creator/audience/)
                - post_impressions: impressions and engagement summary across
                  your posts (/analytics/creator/content/)

        Returns:
            Dict with url, sections (name -> raw text), metrics and
            named_metrics per section (as in get_post_analytics),
            unknown_sections for names that are not sections, and
            section_errors: error_type "redirected" when LinkedIn sent this
            account elsewhere (for example a Premium prompt).
        """
        try:
            requested, unknown = parse_profile_analytics_sections(sections)
            if not requested and unknown:
                raise InvalidReferenceError(
                    f"Unknown analytics sections: {', '.join(unknown)}. Valid: "
                    "profile_viewers, search_appearances, followers, post_impressions."
                )
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_profile_analytics"
            )
            logger.info("Reading profile analytics (sections=%s)", sections)
            await ctx.report_progress(
                progress=0, total=100, message="Loading profile analytics"
            )
            result = await extractor.get_profile_analytics(sections)
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_profile_analytics")
        except Exception as e:
            raise_tool_error(e, "get_profile_analytics")  # NoReturn
