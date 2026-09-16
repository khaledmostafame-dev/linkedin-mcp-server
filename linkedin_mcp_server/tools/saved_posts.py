"""
LinkedIn saved-posts tool.

Lists the authenticated user's saved posts from ``/my-items/saved-posts/``
using innerText extraction — the read counterpart of ``get_saved_jobs`` for
posts saved via LinkedIn's "Save" action (issue #600).
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
from linkedin_mcp_server.scraping.link_metadata import Reference

logger = logging.getLogger(__name__)

SAVED_POSTS_URL = "https://www.linkedin.com/my-items/saved-posts/"


def register_saved_posts_tools(
    mcp: FastMCP, *, tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS
) -> None:
    """Register saved-posts tools with the MCP server."""

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Saved Posts",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"post", "saved", "scraping", "read"},
        exclude_args=["extractor"],
    )
    async def get_saved_posts(
        ctx: Context,
        max_posts: Annotated[int, Field(ge=1, le=50)] = 20,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        List the authenticated user's saved posts from
        ``/my-items/saved-posts/`` (posts saved via LinkedIn's "Save"
        action on a post's overflow menu).

        Args:
            ctx: FastMCP context for progress reporting
            max_posts: Maximum number of saved posts to load (1-50,
                default 20). The list lazy-loads on scroll (no ``?start=``
                pagination like saved jobs), so this sizes the scroll budget
                rather than a page count; the actual count may slightly
                exceed the target.

        Returns:
            Dict with url, sections (saved_posts -> raw text), and optional
            references["saved_posts"] and section_errors. The LLM should
            parse the raw text to extract each saved post's author, body,
            and date; use save_post(post_url, confirm=False) or a fresh
            get_saved_posts call to check current save state.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_saved_posts"
            )
            logger.info("Fetching saved posts (max_posts=%d)", max_posts)

            await ctx.report_progress(
                progress=0, total=100, message="Loading saved posts"
            )

            # ~5 posts load per scroll batch, mirroring get_notifications'
            # and search_posts' pacing; capped well under the tool ceiling
            # so a runaway lazy-load list cannot spend the whole timeout.
            scrolls = min(max(1, -(-max_posts // 5)), 20)
            extracted = await extractor.extract_page(
                SAVED_POSTS_URL, section_name="saved_posts", max_scrolls=scrolls
            )

            sections: dict[str, str] = {}
            references: dict[str, list[Reference]] = {}
            section_errors: dict[str, dict[str, Any]] = {}
            if extracted.text and extracted.text != RATE_LIMITED_SECTION_TEXT:
                sections["saved_posts"] = extracted.text
                if extracted.references:
                    references["saved_posts"] = extracted.references
            elif extracted.text == RATE_LIMITED_SECTION_TEXT:
                section_errors["saved_posts"] = rate_limited_section_error()
            elif extracted.error:
                section_errors["saved_posts"] = extracted.error

            await ctx.report_progress(progress=100, total=100, message="Complete")

            result: dict[str, Any] = {"url": SAVED_POSTS_URL, "sections": sections}
            if references:
                result["references"] = references
            if section_errors:
                result["section_errors"] = section_errors
            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_saved_posts")
        except Exception as e:
            raise_tool_error(e, "get_saved_posts")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Save Post",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"post", "saved", "actions", "write"},
        exclude_args=["extractor"],
    )
    async def save_post(
        post_url: str,
        confirm: bool,
        ctx: Context,
        unsave: bool = False,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Save or unsave a LinkedIn post via its overflow menu's Save
        toggle.

        The overflow-menu opener and the save toggle inside it are found
        structurally (an aria-expanded menu opener, then the single
        aria-pressed control inside the resulting menu — see
        scraping/post_actions.py), never by matching visible text. This
        DOM probe is unverified against a live account — see AGENTS.md.
        If either cannot be identified unambiguously, nothing is clicked
        and status is "structural_signal_not_found".

        confirm=False never changes LinkedIn state: it opens the overflow
        menu (the only way to read the current save state) but returns
        before the toggle itself is clicked, reporting what would happen.

        Args:
            post_url: A /feed/update/<urn>/ or /posts/<slug> permalink,
                e.g. from references["feed"], references["posts"], or
                references["saved_posts"].
            confirm: Must be True to actually save/unsave.
            ctx: FastMCP context for progress reporting
            unsave: False (default) saves the post; True unsaves it. If
                the post is already in the requested state, nothing is
                clicked either way.

        Returns:
            Dict with url, status, message, saved, retry_safe. ``saved``
            is the toggle's last-observed state (None when it could not
            be read). ``retry_safe`` is False once a click has been
            dispatched and its outcome could not be confirmed — status
            "state_unconfirmed" — since a retry there may toggle the post
            back rather than complete the original request.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="save_post"
            )
            logger.info(
                "%s post %s (confirm=%s)",
                "Unsaving" if unsave else "Saving",
                post_url,
                confirm,
            )

            await ctx.report_progress(
                progress=0, total=100, message="Opening the post's overflow menu"
            )

            result = await extractor.save_post(post_url, confirm=confirm, unsave=unsave)

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "save_post")
        except Exception as e:
            raise_tool_error(e, "save_post")  # NoReturn
