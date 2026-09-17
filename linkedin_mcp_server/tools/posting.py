"""
LinkedIn posting tools: publish, schedule, list and delete posts.

These drive LinkedIn's own share composer in the logged-in browser, which is
what makes real @mentions of arbitrary members and companies, document
(PDF/PPTX carousel) posts and LinkedIn-native scheduling possible.

Every write takes a required ``confirm``. With ``confirm=False`` nothing is
sent to LinkedIn: ``create_post`` validates and returns a preview without
acquiring a browser at all.
"""

import logging
from typing import Annotated, Any, Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, Field

from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.core.exceptions import AuthenticationError
from linkedin_mcp_server.dependencies import get_ready_extractor, handle_auth_error
from linkedin_mcp_server.error_handler import raise_tool_error
from linkedin_mcp_server.post_media import media_staging, stage_post_attachments
from linkedin_mcp_server.scraping.post_content import (
    PostValidationError,
    build_poll,
    build_post_edit,
    build_post_request,
    parse_post_url,
    post_preview,
)
from linkedin_mcp_server.tools.company_pages import (
    company_page_tools_enabled,
    refuse_post_as_when_disabled,
)

logger = logging.getLogger(__name__)

POST_INTERRUPTED_WARNING = (
    "A create_post call was interrupted after the composer's post action may "
    "have been clicked. Check the account's activity before retrying; retrying "
    "may publish the post twice."
)


class MediaInput(BaseModel):
    """One attachment, given as exactly one of url, base64 or path."""

    model_config = ConfigDict(extra="forbid")

    url: str | None = Field(
        default=None,
        description="https URL the server downloads (public hosts only, size-capped).",
    )
    base64: str | None = Field(
        default=None,
        description="File content as base64 (needs filename; capped at 20 MB).",
    )
    filename: str | None = Field(
        default=None,
        description="File name, required with base64 (e.g. deck.pdf).",
    )
    path: str | None = Field(
        default=None,
        description="File name inside the server's uploads directory "
        "(~/.linkedin-mcp/uploads). Paths outside it are refused.",
    )


class DocumentInput(MediaInput):
    """A PDF, PowerPoint or Word document shown as a swipeable carousel."""

    title: str = Field(description="Document title LinkedIn shows on the post.")


def register_posting_tools(
    mcp: FastMCP,
    *,
    tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
    company_page_tools: bool | None = None,
) -> None:
    """Register post publishing and scheduling tools with the MCP server.

    *company_page_tools* decides whether ``post_as`` is accepted; ``None``
    reads ``ENABLE_COMPANY_PAGE_TOOLS`` (off by default, experimental).
    """
    post_as_enabled = company_page_tools_enabled(company_page_tools)

    @mcp.tool(
        timeout=tool_timeout,
        title="Create Post",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"posting", "write"},
        exclude_args=["extractor"],
    )
    async def create_post(
        text: str,
        confirm: bool,
        ctx: Context,
        visibility: Literal["anyone", "connections"] = "anyone",
        schedule_at: str | None = None,
        media: Annotated[list[MediaInput] | None, Field(max_length=20)] = None,
        document: DocumentInput | None = None,
        post_as: str | None = None,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Publish or schedule a LinkedIn post from the authenticated account.

        Mentions: write @[Display Name](https://www.linkedin.com/in/slug/) or
        @[Company](https://www.linkedin.com/company/slug/) (a URN such as
        urn:li:fsd_profile:... or urn:li:organization:... also works). The
        name is typed into LinkedIn's typeahead, and only the suggestion whose
        profile or company identity matches the URL/URN is selected. If none
        matches, nothing is posted. Hashtags stay plain text.

        Attachments: up to 20 images in media, or one document (PDF, PPTX,
        PPT, DOCX, DOC) with a title, never both. Each is given as url,
        base64 + filename, or path inside the server's uploads directory.
        Video is not supported.

        This is a write operation when confirm is True. With confirm False
        nothing is sent to LinkedIn and a preview is returned: the final text
        with mentions as parsed, its character count against LinkedIn's 3,000
        limit, attachments, schedule and visibility.

        Args:
            text: Post body with optional inline mentions. Line breaks allowed.
            confirm: False returns a preview; True publishes or schedules.
            ctx: FastMCP context for progress reporting
            visibility: "anyone" (public) or "connections".
            schedule_at: Optional ISO 8601 date-time with an explicit offset,
                e.g. 2026-09-20T09:00:00+04:00. Must be at least 10 minutes
                ahead and within 90 days. Uses LinkedIn's native scheduler;
                the result reports the instant in UTC and in the browser's
                timezone.
            media: Optional list of images.
            document: Optional document with a required title.
            post_as: EXPERIMENTAL - not fully tested, known not working
                (live check 2026-09-17: admin analytics routes returned
                not_authorized), disabled by default; refused unless the
                server runs with ENABLE_COMPANY_PAGE_TOOLS=true. Optional
                company page to post as, which this account must administer:
                https://www.linkedin.com/company/<slug>/, its numeric id, or
                urn:li:organization:<id>. Only the author option carrying that
                page's identity is selected, and it must read back as
                selected; otherwise nothing is posted. Page posts are public,
                so visibility must stay "anyone".

        Returns:
            Dict with url, status ("preview", "published", "scheduled", or a
            refusal such as "mention_unresolved"), message and retry_safe.
            Published posts carry post_url/post_urn when LinkedIn exposed
            them; scheduled posts carry schedule and the scheduled list.
            retry_safe is False once the post action may have been clicked:
            retrying then can publish twice.
        """
        try:
            refuse_post_as_when_disabled(post_as, enabled=post_as_enabled)
            # Everything that can be refused without LinkedIn is refused
            # before any download, and the whole request before a browser.
            build_post_request(
                text,
                visibility=visibility,
                schedule_at=schedule_at,
                attachments_pending=bool(media or document),
                post_as=post_as,
            )
            with media_staging() as staging:
                images, staged_document, title = await stage_post_attachments(
                    [item.model_dump(exclude_none=True) for item in media or []],
                    document.model_dump(exclude_none=True) if document else None,
                    staging=staging,
                )
                request = build_post_request(
                    text,
                    visibility=visibility,
                    schedule_at=schedule_at,
                    images=images,
                    document=staged_document,
                    document_title=title,
                    post_as=post_as,
                )
                if not confirm:
                    return post_preview(request)

                extractor = extractor or await get_ready_extractor(
                    ctx, tool_name="create_post"
                )
                logger.info(
                    "Creating post: chars=%d mentions=%d images=%d document=%s "
                    "scheduled=%s visibility=%s",
                    len(request.rendered_text),
                    len(request.mentions),
                    len(request.images),
                    request.document is not None,
                    request.schedule_at is not None,
                    request.visibility,
                )
                await ctx.report_progress(
                    progress=0, total=100, message="Opening the share composer"
                )
                result = await extractor.create_post(request)
                try:
                    await ctx.report_progress(
                        progress=100, total=100, message="Complete"
                    )
                except BaseException:
                    if result.get("retry_safe") is False:
                        logger.warning(POST_INTERRUPTED_WARNING)
                    raise
                return result
        except PostValidationError as e:
            raise ToolError(str(e)) from e
        except ToolError:
            raise
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "create_post")
        except Exception as e:
            raise_tool_error(e, "create_post")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Scheduled Posts",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"posting", "read"},
        exclude_args=["extractor"],
    )
    async def get_scheduled_posts(
        ctx: Context,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        List the authenticated account's scheduled LinkedIn posts.

        Opens the share composer's scheduled posts view, reads it, and closes
        the composer again without typing anything. A composer holding a
        restored draft is left untouched and the call is refused instead.

        Args:
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with url, sections (scheduled_posts -> raw text) and
            scheduled_posts: [{identifier, text}]. The identifier is what
            delete_scheduled_post takes; it is derived from the entry's text,
            so re-read the list when an entry changes.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_scheduled_posts"
            )
            await ctx.report_progress(
                progress=0, total=100, message="Opening scheduled posts"
            )
            result = await extractor.get_scheduled_posts()
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_scheduled_posts")
        except Exception as e:
            raise_tool_error(e, "get_scheduled_posts")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Delete Scheduled Post",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"posting", "write"},
        exclude_args=["extractor"],
    )
    async def delete_scheduled_post(
        identifier: str,
        confirm: bool,
        ctx: Context,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Delete one scheduled LinkedIn post.

        The identifier is re-resolved against the live scheduled list; zero or
        several matches refuse. With confirm False the entry is resolved and
        shown but nothing is deleted. LinkedIn cannot recover a deleted post.

        Args:
            identifier: An identifier returned by get_scheduled_posts.
            confirm: False previews the entry; True deletes it.
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with url, status ("preview", "deleted", or a refusal such as
            "entry_not_found"), message, retry_safe and the entry text.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="delete_scheduled_post"
            )
            await ctx.report_progress(
                progress=0, total=100, message="Opening scheduled posts"
            )
            result = await extractor.delete_scheduled_post(identifier, confirm=confirm)
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "delete_scheduled_post")
        except Exception as e:
            raise_tool_error(e, "delete_scheduled_post")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Edit Scheduled Post",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"posting", "write"},
        exclude_args=["extractor"],
    )
    async def edit_scheduled_post(
        identifier: str,
        confirm: bool,
        ctx: Context,
        text: str | None = None,
        schedule_at: str | None = None,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Change a scheduled LinkedIn post's text, its scheduled time, or both.

        The identifier is re-resolved against the live scheduled list, and the
        edit composer must open the matching post before anything is changed.
        New text replaces the old entirely and may use the same
        @[Name](URL or URN) mention syntax as create_post. With confirm False
        the entry is resolved and the planned change shown; nothing is edited.

        Args:
            identifier: An identifier returned by get_scheduled_posts.
            confirm: False previews; True saves the edit.
            ctx: FastMCP context for progress reporting
            text: Optional replacement text.
            schedule_at: Optional new ISO 8601 date-time with an explicit offset.

        Returns:
            Dict with url, status ("preview", "edited", or a refusal such as
            "entry_not_found" or "edit_mismatch"), message, retry_safe, changes
            and the refreshed scheduled list after a save. An edited entry gets
            a new identifier; use the returned list.
        """
        try:
            edit = build_post_edit(text, schedule_at=schedule_at)
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="edit_scheduled_post"
            )
            await ctx.report_progress(
                progress=0, total=100, message="Opening scheduled posts"
            )
            result = await extractor.edit_scheduled_post(
                identifier, edit, confirm=confirm
            )
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except PostValidationError as e:
            raise ToolError(str(e)) from e
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "edit_scheduled_post")
        except Exception as e:
            raise_tool_error(e, "edit_scheduled_post")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Delete Post",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"posting", "write"},
        exclude_args=["extractor"],
    )
    async def delete_post(
        post_url: str,
        confirm: bool,
        ctx: Context,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Delete one of the authenticated member's own published LinkedIn posts.

        Authorship is verified before anything happens: the post's author link
        must be the logged-in member's own profile, and the post's control menu
        must offer the owner-only edit and delete actions. Any other post is
        refused, including posts published as a company page. With confirm
        False authorship is verified and nothing is deleted. LinkedIn cannot
        recover a deleted post.

        Args:
            post_url: The post's permalink (/feed/update/urn:li:activity:<id>/
                or a /posts/...-activity-<id>-... link).
            confirm: False verifies and previews; True deletes.
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with url, status ("preview", "deleted", or a refusal such as
            "not_own_post"), message, retry_safe and the post's text.
        """
        try:
            normalized = parse_post_url(post_url)
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="delete_post"
            )
            await ctx.report_progress(progress=0, total=100, message="Opening post")
            result = await extractor.delete_post(normalized, confirm=confirm)
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except PostValidationError as e:
            raise ToolError(str(e)) from e
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "delete_post")
        except Exception as e:
            raise_tool_error(e, "delete_post")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Edit Post",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"posting", "write"},
        exclude_args=["extractor"],
    )
    async def edit_post(
        post_url: str,
        text: str,
        confirm: bool,
        ctx: Context,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Replace the text of one of the authenticated member's own published posts.

        Authorship is verified exactly as delete_post does, before anything is
        changed. The new text replaces the old entirely and may use the
        @[Name](URL or URN) mention syntax; attachments are left as they are.
        LinkedIn re-evaluates a post's distribution when it is edited, so
        editing a post that is performing well can cost it reach. With confirm
        False authorship is verified and nothing is edited.

        Args:
            post_url: The post's permalink.
            text: Replacement text.
            confirm: False verifies and previews; True saves the edit.
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with url, status ("preview", "edited", or a refusal such as
            "not_own_post"), message and retry_safe.
        """
        try:
            normalized = parse_post_url(post_url)
            edit = build_post_edit(text, allow_schedule=False)
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="edit_post"
            )
            await ctx.report_progress(progress=0, total=100, message="Opening post")
            result = await extractor.edit_post(normalized, edit, confirm=confirm)
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except PostValidationError as e:
            raise ToolError(str(e)) from e
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "edit_post")
        except Exception as e:
            raise_tool_error(e, "edit_post")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Create Poll",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"posting", "write"},
        exclude_args=["extractor"],
    )
    async def create_poll(
        question: str,
        options: Annotated[list[str], Field(min_length=2, max_length=4)],
        duration_days: Literal[1, 3, 7, 14],
        confirm: bool,
        ctx: Context,
        text: str = "",
        post_as: str | None = None,
        schedule_at: str | None = None,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Publish or schedule a LinkedIn poll from the authenticated account.

        LinkedIn's limits are checked before any browser work: a question of
        at most 140 characters, 2 to 4 distinct options of at most 30
        characters each, and a duration of 1, 3, 7 or 14 days. The poll form
        is filled field by field, every value must read back exactly, and the
        composer's poll preview must show the same question and options
        before anything is posted. text is the optional post body and may use
        the @[Name](URL or URN) mention syntax. post_as and schedule_at work
        as in create_post.

        This is a write operation when confirm is True. With confirm False
        nothing is sent to LinkedIn and a preview is returned.

        Args:
            question: The poll question.
            options: 2 to 4 answer options.
            duration_days: How long the poll runs: 1, 3, 7 or 14.
            confirm: False returns a preview; True publishes or schedules.
            ctx: FastMCP context for progress reporting
            text: Optional post text shown above the poll.
            post_as: EXPERIMENTAL - not fully tested, known not working,
                disabled by default (needs ENABLE_COMPANY_PAGE_TOOLS=true).
                Optional company page to post as (URL, numeric id or URN).
            schedule_at: Optional ISO 8601 date-time with an explicit offset.

        Returns:
            Dict with url, status ("preview", "published", "scheduled", or a
            refusal such as "poll_unavailable"), message and retry_safe, which
            is False once the post action may have been clicked.
        """
        try:
            refuse_post_as_when_disabled(post_as, enabled=post_as_enabled)
            poll = build_poll(question, options, duration_days)
            request = build_post_request(
                text,
                schedule_at=schedule_at,
                post_as=post_as,
                poll=poll,
            )
            if not confirm:
                return post_preview(request)
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="create_poll"
            )
            await ctx.report_progress(
                progress=0, total=100, message="Opening the share composer"
            )
            result = await extractor.create_poll(request)
            try:
                await ctx.report_progress(progress=100, total=100, message="Complete")
            except BaseException:
                if result.get("retry_safe") is False:
                    logger.warning(POST_INTERRUPTED_WARNING)
                raise
            return result
        except PostValidationError as e:
            raise ToolError(str(e)) from e
        except ToolError:
            raise
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "create_poll")
        except Exception as e:
            raise_tool_error(e, "create_poll")  # NoReturn
