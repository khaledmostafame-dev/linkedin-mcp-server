"""
LinkedIn post-reactions tool.

Reads the reactor list from a post's reactions dialog. The dialog is
opened via a structural probe rather than a URL (see
``scraping/reactions.py``); when that probe cannot identify the dialog's
opening control or the dialog itself unambiguously, the tool reports a
``section_errors`` entry rather than guessing.
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


def register_reaction_tools(
    mcp: FastMCP, *, tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS
) -> None:
    """Register post-reaction tools with the MCP server."""

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Post Reactions",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"post", "reactions", "scraping", "read"},
        exclude_args=["extractor"],
    )
    async def get_post_reactions(
        post_url: str,
        ctx: Context,
        max_reactors: Annotated[int, Field(ge=1, le=50)] = 50,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        List who reacted to a LinkedIn post, by opening its reactions
        dialog.

        The dialog is opened by clicking the post's social-counts summary
        (a structural probe, not a text match — see
        scraping/reactions.py), and the reactor list is read with bounded
        scrolling inside it. Reaction type (Like/Celebrate/Support/...) is
        not returned: no non-text structural signal for it was found, and
        this tool omits rather than guesses one. This DOM probe is
        unverified against a live account — see AGENTS.md.

        Args:
            post_url: A /feed/update/<urn>/ or /posts/<slug> permalink,
                e.g. from references["feed"] or references["posts"].
            ctx: FastMCP context for progress reporting
            max_reactors: Maximum number of reactors to load (1-50,
                default 50). Reactors load in batches as the dialog
                scrolls, so the actual count may slightly exceed the
                target.

        Returns:
            Dict with url, sections (reactions -> raw text), and optional
            references["reactions"] (kind "person", one per reactor
            profile) and section_errors. A section_errors["reactions"]
            entry with error_type "structural_signal_not_found" means the
            social-counts control or the dialog it should open could not
            be identified unambiguously; nothing was clicked beyond the
            probe itself in that case.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_post_reactions"
            )
            logger.info(
                "Fetching post reactions: %s (max_reactors=%d)",
                post_url,
                max_reactors,
            )

            await ctx.report_progress(
                progress=0, total=100, message="Opening reactions dialog"
            )

            result = await extractor.get_post_reactions(
                post_url, max_reactors=max_reactors
            )

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_post_reactions")
        except Exception as e:
            raise_tool_error(e, "get_post_reactions")  # NoReturn
