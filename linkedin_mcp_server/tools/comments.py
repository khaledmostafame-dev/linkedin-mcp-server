"""
LinkedIn post comment tools.

Reads a post's comment thread with a structured id for every comment, and writes
comments, replies and likes through the browser UI. Every write takes a required
``confirm``: with ``confirm=False`` nothing is sent to LinkedIn at all, and the
tool answers ``status: "preview"`` with what it would post.
"""

import logging
from typing import Annotated, Any, Literal

from fastmcp import Context, FastMCP
from pydantic import Field

from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.core.exceptions import AuthenticationError
from linkedin_mcp_server.dependencies import get_ready_extractor, handle_auth_error
from linkedin_mcp_server.error_handler import raise_tool_error
from linkedin_mcp_server.scraping.comments import (
    COMMENT_INTERRUPTED_WARNING,
    MAX_COMMENTS_LIMIT,
    prepare_comment_reaction,
    prepare_comment_write,
)

logger = logging.getLogger(__name__)

_READ_ANNOTATIONS = {"readOnlyHint": True, "openWorldHint": True}
_WRITE_ANNOTATIONS = {"destructiveHint": True, "openWorldHint": True}


async def _report_completion(ctx: Context, result: dict[str, Any]) -> None:
    """Final progress note that never hides a possibly posted comment."""
    try:
        await ctx.report_progress(progress=100, total=100, message="Complete")
    except BaseException:
        if result.get("retry_safe") is False:
            logger.warning(COMMENT_INTERRUPTED_WARNING)
        raise


def register_comment_tools(
    mcp: FastMCP, *, tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS
) -> None:
    """Register comment read and write tools with the MCP server."""

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Post Comments",
        annotations=_READ_ANNOTATIONS,
        tags={"comments", "read"},
        exclude_args=["extractor"],
    )
    async def get_post_comments(
        post_url: str,
        ctx: Context,
        max_comments: Annotated[int, Field(ge=1, le=MAX_COMMENTS_LIMIT)] = 50,
        include_replies: bool = True,
        sort: Literal["relevant", "recent"] = "relevant",
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Read the comments on one LinkedIn post, with a structured id for each.

        Opens the post, loads more comments and reply threads with a bounded
        number of clicks, and returns the thread text plus one reference per
        comment. Use the reference `value` (the comment URN) with
        reply_to_comment or react_to_comment.

        Args:
            post_url: The post, as /feed/update/urn:li:activity:<id>/,
                urn:li:ugcPost:<id> or urn:li:share:<id> (bare or in that path),
                or a /posts/<slug> permalink. references["feed"] from get_feed
                work as-is.
            ctx: FastMCP context for progress reporting
            max_comments: Maximum number of comments (replies included) to
                return, 1-100, default 50.
            include_replies: Also expand and return replies (default true).
            sort: "relevant" keeps LinkedIn's on-page order; "recent" orders
                newest first by comment id (LinkedIn issues ids in time order),
                each top-level comment followed by its replies. The on-page
                sort menu is not driven.

        Returns:
            Dict with url, sections["comments"] (raw thread text) and
            references["comments"]: a list of {kind: "comment", url (author
            /in/ or /company/ path), text (author name), context (lines under
            the name, e.g. headline), value (comment URN), parent (parent
            comment URN, replies only), permalink, excerpt, posted_at (UTC,
            decoded from the comment id)}. section_errors is present when the
            page looked rate-limited or extraction failed.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_post_comments"
            )
            logger.info(
                "Reading comments on %s (max_comments=%d, replies=%s, sort=%s)",
                post_url,
                max_comments,
                include_replies,
                sort,
            )
            await ctx.report_progress(
                progress=0, total=100, message="Loading post comments"
            )
            result = await extractor.get_post_comments(
                post_url,
                max_comments=max_comments,
                include_replies=include_replies,
                sort=sort,
            )
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_post_comments")
        except Exception as e:
            raise_tool_error(e, "get_post_comments")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Reply To Comment",
        annotations=_WRITE_ANNOTATIONS,
        tags={"comments", "write"},
        exclude_args=["extractor"],
    )
    async def reply_to_comment(
        post_url: str,
        comment_urn: str,
        text: str,
        confirm: bool,
        ctx: Context,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Reply to one comment on a LinkedIn post, as the signed-in member.

        The comment is located by its URN and must resolve to exactly one
        comment on the post. The reply box is used only after it is verified
        to have opened under that comment; the text is typed with small
        per-key delays, submitted, and the reply is confirmed by a new comment
        with that text appearing under the target. Any contradiction fails
        closed before submission. This is a write operation when confirm is
        true.

        Args:
            post_url: The post the comment is on (same forms as get_post_comments).
            comment_urn: The comment's URN, e.g. the `value` of an entry in
                get_post_comments' references["comments"].
            text: Reply text, at most 1,250 characters. Line breaks are
                supported; other control characters are rejected. '@' is
                rejected because @mentions are not supported.
            confirm: false returns status "preview" without touching LinkedIn;
                true posts the reply.
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with url, status, message, post_urn, comment_urn, posted and
            retry_safe; on success also posted_comment_urn, text and target
            (author and excerpt of the comment replied to). status is one of
            preview, posted, unconfirmed, invalid_text, post_unavailable,
            comment_not_found, comment_ambiguous, reply_box_unavailable,
            composer_occupied, editor_unavailable, typing_mismatch,
            submit_unavailable. When retry_safe is false the reply may already
            be posted: read the post before calling again.
        """
        try:
            answer = prepare_comment_write(post_url, comment_urn, text, confirm=confirm)
            if answer is not None:
                return answer
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="reply_to_comment"
            )
            logger.info("Replying to comment %s on %s", comment_urn, post_url)
            await ctx.report_progress(progress=0, total=100, message="Posting reply")
            result = await extractor.reply_to_comment(
                post_url, comment_urn, text, confirm=confirm
            )
            await _report_completion(ctx, result)
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "reply_to_comment")
        except Exception as e:
            raise_tool_error(e, "reply_to_comment")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Comment On Post",
        annotations=_WRITE_ANNOTATIONS,
        tags={"comments", "write"},
        exclude_args=["extractor"],
    )
    async def comment_on_post(
        post_url: str,
        text: str,
        confirm: bool,
        ctx: Context,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Post a top-level comment on a LinkedIn post, as the signed-in member.

        Uses the post's own comment box, which must be the only one outside
        existing comments and must be empty. The comment is confirmed by a new
        top-level comment carrying the text appearing on the post. This is a
        write operation when confirm is true.

        Args:
            post_url: The post (same forms as get_post_comments).
            text: Comment text, at most 1,250 characters. Line breaks are
                supported; other control characters and '@' are rejected.
            confirm: false returns status "preview" without touching LinkedIn;
                true posts the comment.
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with url, status, message, post_urn, posted and retry_safe;
            on success also posted_comment_urn and text. status is one of
            preview, posted, unconfirmed, invalid_text, post_unavailable,
            comment_box_unavailable, composer_occupied, editor_unavailable,
            typing_mismatch, submit_unavailable. When retry_safe is false the
            comment may already be posted: read the post before calling again.
        """
        try:
            answer = prepare_comment_write(post_url, None, text, confirm=confirm)
            if answer is not None:
                return answer
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="comment_on_post"
            )
            logger.info("Commenting on %s", post_url)
            await ctx.report_progress(progress=0, total=100, message="Posting comment")
            result = await extractor.comment_on_post(post_url, text, confirm=confirm)
            await _report_completion(ctx, result)
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "comment_on_post")
        except Exception as e:
            raise_tool_error(e, "comment_on_post")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="React To Comment",
        annotations=_WRITE_ANNOTATIONS,
        tags={"comments", "write"},
        exclude_args=["extractor"],
    )
    async def react_to_comment(
        post_url: str,
        comment_urn: str,
        confirm: bool,
        ctx: Context,
        reaction: Literal["like"] = "like",
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Like one comment on a LinkedIn post, as the signed-in member.

        The comment is located by its URN and must expose exactly one reaction
        toggle of its own. A comment you already reacted to is left unchanged
        (status "already_reacted"), so calling twice never removes a reaction.
        Only "like" is supported. This is a write operation when confirm is
        true.

        Args:
            post_url: The post the comment is on (same forms as get_post_comments).
            comment_urn: The comment's URN from get_post_comments.
            confirm: false returns status "preview" without touching LinkedIn;
                true reacts.
            ctx: FastMCP context for progress reporting
            reaction: Only "like".

        Returns:
            Dict with url, status, message, post_urn, comment_urn, posted and
            retry_safe. status is one of preview, reacted, already_reacted,
            unconfirmed, post_unavailable, comment_not_found, comment_ambiguous,
            reaction_unavailable, unsupported_reaction.
        """
        try:
            answer = prepare_comment_reaction(
                post_url, comment_urn, reaction, confirm=confirm
            )
            if answer is not None:
                return answer
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="react_to_comment"
            )
            logger.info("Reacting to comment %s on %s", comment_urn, post_url)
            await ctx.report_progress(progress=0, total=100, message="Reacting")
            result = await extractor.react_to_comment(
                post_url, comment_urn, reaction, confirm=confirm
            )
            await _report_completion(ctx, result)
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "react_to_comment")
        except Exception as e:
            raise_tool_error(e, "react_to_comment")  # NoReturn
