"""
LinkedIn feed scraping tool.

Fetches posts from the authenticated user's LinkedIn home feed using
innerText extraction. Scrolls until the requested number of post
permalinks have been observed in SDUI pagination responses — a
locale-independent progress signal, since the feed DOM exposes no
stable per-post container selector.
"""

import logging
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from pydantic import Field

from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.core.exceptions import AuthenticationError
from linkedin_mcp_server.dependencies import get_ready_extractor, handle_auth_error
from linkedin_mcp_server.error_handler import raise_tool_error
from linkedin_mcp_server.scraping.contracts import RATE_LIMITED_SECTION_TEXT
from linkedin_mcp_server.scraping.contracts import rate_limited_section_error
from linkedin_mcp_server.scraping.identifiers import hashtag_feed_url, normalize_hashtag
from linkedin_mcp_server.scraping.link_metadata import Reference

logger = logging.getLogger(__name__)

# LinkedIn's notifications page renders "All" / "My posts" / "Mentions" as
# client-side tabs. The only observed mechanism for reaching a specific tab
# by URL is a `filterType` query parameter carrying one of these tokens;
# unverified against a live account, since this fork never signs in to
# LinkedIn (see AGENTS.md Hard safety rules). "all" needs no parameter and
# is the well-precedented path (see get_feed / get_inbox); the other two
# should be treated as best-effort until confirmed live.
_NOTIFICATION_FILTER_QUERY = {"my_posts": "MY_POSTS", "mentions": "MENTIONS"}


def _notifications_url(filter_: str) -> str:
    base = "https://www.linkedin.com/notifications/"
    token = _NOTIFICATION_FILTER_QUERY.get(filter_)
    return f"{base}?filterType={token}" if token else base


def register_feed_tools(
    mcp: FastMCP, *, tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS
) -> None:
    """Register feed-related tools with the MCP server."""

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Feed",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"feed", "scraping"},
        exclude_args=["extractor"],
    )
    async def get_feed(
        ctx: Context,
        num_posts: Annotated[int, Field(ge=1, le=50)] = 10,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Get posts from the authenticated user's LinkedIn feed.

        Args:
            ctx: FastMCP context for progress reporting
            num_posts: Number of posts to fetch (1-50, default 10).
                       Posts are loaded in batches of ~5 as the page scrolls,
                       so the actual count may slightly exceed the target.

        Returns:
            Dict with url, sections (name -> raw text), and optional keys:
            - references["feed"]: list of {kind: "feed_post", url, ...}
              entries. URLs are relative paths and may carry either
              ``/feed/update/<urn>/`` (DOM-anchor-derived) or
              ``/posts/<slug>`` (SDUI-derived) shape — both are valid
              LinkedIn permalinks.
            - section_errors: present when the feed is rate-limited or
              extraction fails.

            Truncated posts are not auto-expanded; full text for any post
            is reachable via its permalink in references["feed"]. The LLM
            should parse sections["feed"] for post bodies.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_feed"
            )
            logger.info("Scraping feed (num_posts=%d)", num_posts)

            await ctx.report_progress(
                progress=0, total=100, message="Starting feed scrape"
            )

            extracted = await extractor.extract_feed(num_posts=num_posts)

            url = "https://www.linkedin.com/feed/"
            sections: dict[str, str] = {}
            references: dict[str, list[Reference]] = {}
            section_errors: dict[str, dict[str, Any]] = {}
            if extracted.text and extracted.text != RATE_LIMITED_SECTION_TEXT:
                sections["feed"] = extracted.text
                if extracted.references:
                    references["feed"] = extracted.references
            elif extracted.text == RATE_LIMITED_SECTION_TEXT:
                section_errors["feed"] = rate_limited_section_error()
            elif extracted.error:
                section_errors["feed"] = extracted.error

            await ctx.report_progress(progress=100, total=100, message="Complete")

            result: dict[str, Any] = {"url": url, "sections": sections}
            if references:
                result["references"] = references
            if section_errors:
                result["section_errors"] = section_errors
            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_feed")
        except Exception as e:
            raise_tool_error(e, "get_feed")

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Notifications",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"feed", "notifications", "scraping", "read"},
        exclude_args=["extractor"],
    )
    async def get_notifications(
        ctx: Context,
        max_items: Annotated[int, Field(ge=1, le=50)] = 30,
        filter: Annotated[str, Field(pattern="^(all|my_posts|mentions)$")] = "all",
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        List recent notifications from the LinkedIn notifications page.

        Covers every notification type the page shows as raw text — replies
        to your comments, reactions, mentions, connection requests/accepts,
        endorsements, job alerts, and company news — plus references to the
        actors and the posts/profiles/companies each notification refers to.

        Args:
            ctx: FastMCP context for progress reporting
            max_items: Maximum number of notifications to load (1-50,
                default 30). Items load in batches as the page scrolls, so
                the actual count may slightly exceed the target.
            filter: One of "all" (default), "my_posts" (activity on your own
                posts), or "mentions". The non-default values are sent as
                LinkedIn's `filterType` query parameter (`MY_POSTS` /
                `MENTIONS`); this has not been confirmed against a live
                account (this fork never signs in to LinkedIn) and may need
                adjusting if LinkedIn's actual parameter differs — treat
                anything other than "all" as best-effort until verified.

        Returns:
            Dict with url, sections (notifications -> raw text), and
            optional references["notifications"] and section_errors.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_notifications"
            )
            logger.info(
                "Fetching notifications (max_items=%d, filter=%s)",
                max_items,
                filter,
            )

            await ctx.report_progress(
                progress=0, total=100, message="Loading notifications"
            )

            url = _notifications_url(filter)
            # Notification cards load in batches similar to search results;
            # one scroll per ~5 items mirrors search_posts' pacing.
            scrolls = max(1, -(-max_items // 5))
            extracted = await extractor.extract_page(
                url, section_name="notifications", max_scrolls=scrolls
            )

            sections: dict[str, str] = {}
            references: dict[str, list[Reference]] = {}
            section_errors: dict[str, dict[str, Any]] = {}
            if extracted.text and extracted.text != RATE_LIMITED_SECTION_TEXT:
                sections["notifications"] = extracted.text
                if extracted.references:
                    references["notifications"] = extracted.references
            elif extracted.text == RATE_LIMITED_SECTION_TEXT:
                section_errors["notifications"] = rate_limited_section_error()
            elif extracted.error:
                section_errors["notifications"] = extracted.error

            await ctx.report_progress(progress=100, total=100, message="Complete")

            result: dict[str, Any] = {"url": url, "sections": sections}
            if references:
                result["references"] = references
            if section_errors:
                result["section_errors"] = section_errors
            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_notifications")
        except Exception as e:
            raise_tool_error(e, "get_notifications")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Hashtag Feed",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"feed", "hashtag", "scraping", "read"},
        exclude_args=["extractor"],
    )
    async def get_hashtag_feed(
        hashtag: str,
        ctx: Context,
        max_posts: Annotated[int, Field(ge=1, le=50)] = 20,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Get recent posts from a LinkedIn hashtag feed.

        Args:
            hashtag: The hashtag, with or without a leading '#' (e.g.
                "womenintech" or "#womenintech"). A full /feed/hashtag/ URL
                is accepted too and is reduced to the tag.
            ctx: FastMCP context for progress reporting
            max_posts: Maximum number of posts to load (1-50, default 20).
                Posts load in batches as the page scrolls, so the actual
                count may slightly exceed the target.

        Returns:
            Dict with url, sections (feed_hashtag -> raw text), and optional
            references["feed_hashtag"] and section_errors. Post permalinks
            carry kind "feed_post" (same /feed/update/<urn>/ or /posts/<slug>
            shapes as get_feed / get_company_posts); other reference kinds
            (mentioned people/companies) may also appear, matching
            get_company_posts' behavior. The hashtag-feed permalink capture
            is unverified against a live account — see AGENTS.md.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_hashtag_feed"
            )
            tag = normalize_hashtag(hashtag)
            url = hashtag_feed_url(tag, "/")
            logger.info("Scraping hashtag feed: %s (max_posts=%d)", tag, max_posts)

            await ctx.report_progress(
                progress=0, total=100, message="Starting hashtag feed scrape"
            )

            # ~5 posts load per scroll batch, mirroring get_notifications'
            # and get_saved_posts' pacing.
            scrolls = min(max(1, -(-max_posts // 5)), 20)
            extracted = await extractor.extract_page(
                url, section_name="feed_hashtag", max_scrolls=scrolls
            )

            sections: dict[str, str] = {}
            references: dict[str, list[Reference]] = {}
            section_errors: dict[str, dict[str, Any]] = {}
            if extracted.text and extracted.text != RATE_LIMITED_SECTION_TEXT:
                sections["feed_hashtag"] = extracted.text
                if extracted.references:
                    references["feed_hashtag"] = extracted.references
            elif extracted.text == RATE_LIMITED_SECTION_TEXT:
                section_errors["feed_hashtag"] = rate_limited_section_error()
            elif extracted.error:
                section_errors["feed_hashtag"] = extracted.error

            await ctx.report_progress(progress=100, total=100, message="Complete")

            result: dict[str, Any] = {"url": url, "sections": sections}
            if references:
                result["references"] = references
            if section_errors:
                result["section_errors"] = section_errors
            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_hashtag_feed")
        except Exception as e:
            raise_tool_error(e, "get_hashtag_feed")  # NoReturn
