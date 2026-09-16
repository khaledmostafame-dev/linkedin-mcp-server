"""
LinkedIn network-graph tools.

Read-only listings (shared connections, connection list, invitation
queues) plus the write actions that act on one member's invitation
(accept/ignore/withdraw) or a page's follow toggle. Every write tool here
takes a required ``confirm`` bool: ``confirm=False`` returns a preview with
no browser interaction and no LinkedIn state change; only ``confirm=True``
performs the action.
"""

import logging
from typing import Annotated, Any, Literal

from fastmcp import Context, FastMCP
from pydantic import Field

from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.core.exceptions import AuthenticationError
from linkedin_mcp_server.dependencies import get_ready_extractor, handle_auth_error
from linkedin_mcp_server.error_handler import raise_tool_error

logger = logging.getLogger(__name__)


def register_network_tools(
    mcp: FastMCP, *, tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS
) -> None:
    """Register all network-graph tools with the MCP server."""

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Mutual Connections",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"network", "scraping", "read"},
        exclude_args=["extractor"],
    )
    async def get_mutual_connections(
        linkedin_username: str,
        ctx: Context,
        max_results: Annotated[int, Field(ge=1, le=200)] = 50,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        List LinkedIn connections shared between you and another profile.

        Reached the way a person would from the profile page itself: the
        top card's "N mutual connections" control links to LinkedIn's
        shared-connections search. If that control is not present (e.g. no
        mutual connections, or the account cannot see them), the result
        carries an empty sections dict and stopped_reason
        "no_mutual_connections_link_found" rather than an error.

        Args:
            linkedin_username: LinkedIn username of the other profile (e.g. "stickerdaniel"). A full profile URL is accepted too.
            ctx: FastMCP context for progress reporting
            max_results: Approximate cap on how many shared connections to
                load via bounded scrolling (1-200, default 50). LinkedIn's
                own page size varies, so this bounds effort more than it
                guarantees an exact count.

        Returns:
            Dict with url, sections (mutual_connections -> raw text),
            references (profile URLs for each shared connection), and
            stopped_reason ("max_results_reached", "end_of_results", or
            "no_mutual_connections_link_found"). Warm-intro use case: parse
            the raw text / references for names to ask a shared connection
            for an introduction before cold outreach.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_mutual_connections"
            )
            logger.info(
                "Fetching mutual connections for %s (max_results=%d)",
                linkedin_username,
                max_results,
            )
            await ctx.report_progress(
                progress=0, total=100, message="Loading mutual connections"
            )
            result = await extractor.get_mutual_connections(
                linkedin_username, max_results=max_results
            )
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_mutual_connections")
        except Exception as e:
            raise_tool_error(e, "get_mutual_connections")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="List Connections",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"network", "scraping", "read"},
        exclude_args=["extractor"],
    )
    async def list_connections(
        ctx: Context,
        max_results: Annotated[int, Field(ge=1, le=500)] = 100,
        sort: Literal["recently_added", "first_name", "last_name"] = "recently_added",
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        List the authenticated user's 1st-degree LinkedIn connections.

        Scrapes /mynetwork/invite-connect/connections/ with LinkedIn's own
        sort ordering, so "recently_added" is genuinely recency-sorted
        (unlike search_people's relevance ranking).

        Args:
            ctx: FastMCP context for progress reporting
            max_results: Approximate cap on how many connections to load via
                bounded scrolling (1-500, default 100).
            sort: "recently_added" (default), "first_name", or "last_name".

        Returns:
            Dict with url, sections (connections -> raw text), references
            (profile URLs per connection), and stopped_reason.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="list_connections"
            )
            logger.info(
                "Listing connections (max_results=%d, sort=%s)", max_results, sort
            )
            await ctx.report_progress(
                progress=0, total=100, message="Loading connections"
            )
            result = await extractor.list_connections(
                max_results=max_results, sort=sort
            )
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "list_connections")
        except Exception as e:
            raise_tool_error(e, "list_connections")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Invitations",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"network", "scraping", "read"},
        exclude_args=["extractor"],
    )
    async def get_invitations(
        ctx: Context,
        direction: Literal["received", "sent"] = "received",
        max_results: Annotated[int, Field(ge=1, le=200)] = 50,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        List outgoing or incoming LinkedIn connection-request invitations.

        direction="sent" lists requests you sent that are still awaiting
        acceptance (/mynetwork/invitation-manager/sent/); direction="received"
        (default) lists incoming requests awaiting your action
        (/mynetwork/invitation-manager/). References resolve to the
        inviter/invitee profile URLs so withdraw_invitation /
        respond_to_invitation / get_person_profile can chain naturally.

        Args:
            ctx: FastMCP context for progress reporting
            direction: "received" (default) or "sent"
            max_results: Approximate cap on how many invitations to load via
                bounded scrolling (1-200, default 50).

        Returns:
            Dict with url, sections ({direction}_invitations -> raw text),
            references, and stopped_reason.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_invitations"
            )
            logger.info(
                "Fetching %s invitations (max_results=%d)", direction, max_results
            )
            await ctx.report_progress(
                progress=0, total=100, message=f"Loading {direction} invitations"
            )
            result = await extractor.get_invitations(
                direction=direction, max_results=max_results
            )
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_invitations")
        except Exception as e:
            raise_tool_error(e, "get_invitations")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Withdraw Invitation",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"network", "actions", "write"},
        exclude_args=["extractor"],
    )
    async def withdraw_invitation(
        linkedin_username: str,
        confirm: bool,
        ctx: Context,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Withdraw a previously-sent, still-pending LinkedIn connection request.

        Identifies the target by profile URL (navigates to the profile
        itself), never by position in a list. Acts only when the profile's
        own action row shows a pending outgoing request; otherwise no click
        happens and the observed state is reported instead.

        Args:
            linkedin_username: LinkedIn username of the invitation recipient. A full profile URL is accepted too.
            confirm: Must be True to withdraw. False returns a preview with
                no browser interaction and no state change.
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with url, status ("preview", "withdrawn", "not_pending",
            "unavailable", or "send_failed"), message, and profile (raw
            profile text) when available.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="withdraw_invitation"
            )
            logger.info(
                "Withdrawing invitation to %s (confirm=%s)",
                linkedin_username,
                confirm,
            )
            await ctx.report_progress(
                progress=0, total=100, message="Processing withdraw request"
            )
            result = await extractor.withdraw_invitation(
                linkedin_username, confirm=confirm
            )
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "withdraw_invitation")
        except Exception as e:
            raise_tool_error(e, "withdraw_invitation")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Respond To Invitation",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"network", "actions", "write"},
        exclude_args=["extractor"],
    )
    async def respond_to_invitation(
        linkedin_username: str,
        action: Literal["accept", "ignore"],
        confirm: bool,
        ctx: Context,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Accept or ignore an incoming LinkedIn connection request.

        Identifies the target by profile URL (navigates to the profile
        itself), never by position in a list. Acts only when the profile's
        own action row shows an incoming request; otherwise no click
        happens and the observed state is reported instead. The Ignore path
        mirrors the same live-verified action-row fingerprint
        connect_with_person already uses for Accept, but only the Accept
        click itself has been exercised against a real incoming-request
        profile -- treat an "ignored" result as needing a spot check until
        that changes.

        Args:
            linkedin_username: LinkedIn username of the invitation sender. A full profile URL is accepted too.
            action: "accept" or "ignore"
            confirm: Must be True to act. False returns a preview with no
                browser interaction and no state change.
            ctx: FastMCP context for progress reporting

        Returns:
            Dict with url, status ("preview", "accepted", "ignored",
            "not_incoming", "unavailable", or "send_failed"), message, and
            profile (raw profile text) when available.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="respond_to_invitation"
            )
            logger.info(
                "Responding to invitation from %s: action=%s (confirm=%s)",
                linkedin_username,
                action,
                confirm,
            )
            await ctx.report_progress(
                progress=0, total=100, message=f"Processing {action}"
            )
            result = await extractor.respond_to_invitation(
                linkedin_username, action=action, confirm=confirm
            )
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "respond_to_invitation")
        except Exception as e:
            raise_tool_error(e, "respond_to_invitation")  # NoReturn

    @mcp.tool(
        timeout=tool_timeout,
        title="Follow",
        annotations={"destructiveHint": True, "openWorldHint": True},
        tags={"network", "actions", "write"},
        exclude_args=["extractor"],
    )
    async def follow(
        target_url: str,
        confirm: bool,
        ctx: Context,
        unfollow: bool = False,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Follow or unfollow a LinkedIn person or company page.

        Detection of the Follow/Following control is structural and
        deliberately conservative: it acts only when the top card exposes
        exactly one unambiguous, non-menu labeled button. Because LinkedIn's
        Follow/Following labels are locale-dependent text this server does
        not read, success is reported as "toggled" (an accessible-attribute
        change was observed after the click) rather than as a claim about
        which direction the toggle landed in -- verify with
        get_person_profile/get_company_profile if that distinction matters.

        Args:
            target_url: A LinkedIn person profile or company page URL/slug
                (e.g. "https://www.linkedin.com/company/anthropic/" or
                "stickerdaniel"). A bare slug is treated as a person.
            confirm: Must be True to act. False returns a preview with no
                browser interaction and no state change.
            ctx: FastMCP context for progress reporting
            unfollow: Set True to request unfollowing instead of following.
                The click is the same toggle either way; this only affects
                the returned message and the ``requested`` field.

        Returns:
            Dict with url, status ("preview", "toggled", "uncertain", or
            "action_unavailable"), requested ("follow"/"unfollow"), and
            message.
        """
        try:
            extractor = extractor or await get_ready_extractor(ctx, tool_name="follow")
            logger.info(
                "Follow request for %s (unfollow=%s, confirm=%s)",
                target_url,
                unfollow,
                confirm,
            )
            await ctx.report_progress(
                progress=0, total=100, message="Processing follow"
            )
            result = await extractor.follow(
                target_url, confirm=confirm, unfollow=unfollow
            )
            await ctx.report_progress(progress=100, total=100, message="Complete")
            return result
        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "follow")
        except Exception as e:
            raise_tool_error(e, "follow")  # NoReturn
