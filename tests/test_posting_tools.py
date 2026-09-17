"""Tool-layer contract for posting: previews stay browser-free, writes confirm."""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Coroutine, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import FunctionTool

from linkedin_mcp_server.scraping.post_content import PostRequest
from linkedin_mcp_server.tools.posting import (
    DocumentInput,
    MediaInput,
    register_posting_tools,
)

PDF = b"%PDF-1.7\n" + b"0" * 32
PERSON_URL = "https://www.linkedin.com/in/sample-person/"


def _in_two_days() -> str:
    instant = datetime.now(UTC).replace(second=0, microsecond=0) + timedelta(days=2)
    return instant.isoformat()


async def _tool(
    name: str, *, company_page_tools: bool | None = None
) -> Callable[..., Coroutine[Any, Any, dict[str, Any]]]:
    mcp = FastMCP("test")
    register_posting_tools(mcp, company_page_tools=company_page_tools)
    tool = await mcp.get_tool(name)
    assert tool is not None
    return cast(FunctionTool, tool).fn


async def test_write_and_read_annotations_follow_the_shared_contract():
    mcp = FastMCP("test")
    register_posting_tools(mcp)
    tools = {tool.name: tool for tool in await mcp.list_tools()}

    for name in (
        "create_post",
        "delete_scheduled_post",
        "edit_scheduled_post",
        "delete_post",
        "edit_post",
        "create_poll",
    ):
        annotations = tools[name].annotations
        assert annotations is not None
        assert annotations.destructiveHint is True
        assert annotations.openWorldHint is True
        assert "write" in tools[name].tags
        assert "confirm" in tools[name].parameters["required"]
    read = tools["get_scheduled_posts"]
    assert read.annotations is not None
    assert read.annotations.readOnlyHint is True
    assert "read" in read.tags


async def test_preview_never_acquires_a_browser(mock_context):
    create_post = await _tool("create_post")
    with patch(
        "linkedin_mcp_server.tools.posting.get_ready_extractor", new=AsyncMock()
    ) as ready:
        result = await create_post(
            f"Thanks @[Sample Person]({PERSON_URL}) #automation",
            False,
            mock_context,
            schedule_at=_in_two_days(),
            document=DocumentInput(
                base64=base64.b64encode(PDF).decode(),
                filename="deck.pdf",
                title="Synthetic deck",
            ),
        )
    ready.assert_not_awaited()
    assert result["status"] == "preview"
    assert result["text"] == "Thanks Sample Person #automation"
    assert result["document"]["content_type"] == "application/pdf"
    assert result["document"]["title"] == "Synthetic deck"
    assert result["mentions"][0]["target"] == PERSON_URL
    assert result["schedule"]["utc"].endswith("Z")


async def test_schedule_beyond_the_window_is_refused(mock_context):
    create_post = await _tool("create_post")
    with pytest.raises(ToolError, match="window"):
        await create_post(
            "Hello", False, mock_context, schedule_at="2099-01-01T09:00:00+04:00"
        )


async def test_invalid_text_is_refused_before_any_download(mock_context):
    create_post = await _tool("create_post")
    with (
        patch(
            "linkedin_mcp_server.tools.posting.stage_post_attachments", new=AsyncMock()
        ) as stage,
        pytest.raises(ToolError, match="Mention target"),
    ):
        await create_post(
            "@[Name](https://example.com/in/x/)",
            True,
            mock_context,
            media=[MediaInput(url="https://cdn.example/x.png")],
        )
    stage.assert_not_awaited()


async def test_confirmed_post_hands_a_validated_request_to_the_extractor(
    mock_context,
):
    create_post = await _tool("create_post")
    extractor = MagicMock()
    extractor.create_post = AsyncMock(
        return_value={"url": "u", "status": "published", "retry_safe": False}
    )

    result = await create_post(
        "Hello", True, mock_context, visibility="connections", extractor=extractor
    )

    assert result["status"] == "published"
    call = extractor.create_post.await_args
    assert call is not None
    request = call.args[0]
    assert isinstance(request, PostRequest)
    assert request.visibility == "connections"
    assert request.rendered_text == "Hello"


async def test_scheduled_post_tools_delegate(mock_context):
    extractor = MagicMock()
    extractor.get_scheduled_posts = AsyncMock(return_value={"url": "u"})
    extractor.delete_scheduled_post = AsyncMock(return_value={"status": "preview"})

    get_scheduled_posts = await _tool("get_scheduled_posts")
    delete_scheduled_post = await _tool("delete_scheduled_post")
    await get_scheduled_posts(mock_context, extractor=extractor)
    await delete_scheduled_post(
        "sched-0123456789abcdef", False, mock_context, extractor=extractor
    )

    extractor.get_scheduled_posts.assert_awaited_once_with()
    extractor.delete_scheduled_post.assert_awaited_once_with(
        "sched-0123456789abcdef", confirm=False
    )


async def test_post_as_preview_names_the_requested_page(mock_context):
    create_post = await _tool("create_post", company_page_tools=True)

    result = await create_post(
        "Hello", False, mock_context, post_as="https://www.linkedin.com/company/12345/"
    )

    assert result["post_as"]["kind"] == "company"
    assert result["post_as"]["resolved_against_linkedin"] is False


async def test_own_post_tools_normalize_the_permalink(mock_context):
    extractor = MagicMock()
    extractor.delete_post = AsyncMock(return_value={"status": "preview"})
    extractor.edit_post = AsyncMock(return_value={"status": "preview"})
    slug_url = "https://www.linkedin.com/posts/sample_topic-activity-1234567890-AbCd"
    permalink = "https://www.linkedin.com/feed/update/urn:li:activity:1234567890/"

    await (await _tool("delete_post"))(
        slug_url, False, mock_context, extractor=extractor
    )
    await (await _tool("edit_post"))(
        slug_url, "New text", False, mock_context, extractor=extractor
    )

    extractor.delete_post.assert_awaited_once_with(permalink, confirm=False)
    call = extractor.edit_post.await_args
    assert call is not None
    assert call.args[0] == permalink
    assert call.args[1].rendered_text == "New text"


async def test_invalid_edits_are_refused_without_a_browser(mock_context):
    with patch(
        "linkedin_mcp_server.tools.posting.get_ready_extractor", new=AsyncMock()
    ) as ready:
        with pytest.raises(ToolError, match="post_url"):
            await (await _tool("delete_post"))(
                "https://www.linkedin.com/in/sample-person/", True, mock_context
            )
        with pytest.raises(ToolError, match="Pass text"):
            await (await _tool("edit_scheduled_post"))(
                "sched-0123456789abcdef", True, mock_context
            )
    ready.assert_not_awaited()


async def test_edit_post_warns_about_distribution():
    mcp = FastMCP("test")
    register_posting_tools(mcp)
    tool = await mcp.get_tool("edit_post")
    assert tool is not None
    assert "re-evaluates a post's distribution" in (tool.description or "")


async def test_poll_preview_is_browser_free(mock_context):
    create_poll = await _tool("create_poll")
    with patch(
        "linkedin_mcp_server.tools.posting.get_ready_extractor", new=AsyncMock()
    ) as ready:
        result = await create_poll(
            "Best format?", ["Carousel", "Video"], 7, False, mock_context
        )
    ready.assert_not_awaited()
    assert result["status"] == "preview"
    assert result["poll"]["options"] == ["Carousel", "Video"]


async def test_poll_limits_are_refused_before_a_browser(mock_context):
    create_poll = await _tool("create_poll")
    with patch(
        "linkedin_mcp_server.tools.posting.get_ready_extractor", new=AsyncMock()
    ) as ready:
        with pytest.raises(ToolError, match="140"):
            await create_poll("q" * 141, ["Yes", "No"], 7, True, mock_context)
    ready.assert_not_awaited()


async def test_confirmed_poll_goes_to_the_poll_delegate(mock_context):
    extractor = MagicMock()
    extractor.create_poll = AsyncMock(
        return_value={"status": "published", "retry_safe": False}
    )
    create_poll = await _tool("create_poll", company_page_tools=True)

    await create_poll(
        "Best format?",
        ["Carousel", "Video"],
        3,
        True,
        mock_context,
        post_as="12345",
        extractor=extractor,
    )

    call = extractor.create_poll.await_args
    assert call is not None
    request = call.args[0]
    assert request.poll.duration_days == 3
    assert request.post_as.key == "company:urn:12345"


async def test_poll_schema_carries_the_limits():
    mcp = FastMCP("test")
    register_posting_tools(mcp)
    tool = await mcp.get_tool("create_poll")
    assert tool is not None
    properties = tool.parameters["properties"]
    assert properties["duration_days"]["enum"] == [1, 3, 7, 14]
    assert properties["options"]["minItems"] == 2
    assert properties["options"]["maxItems"] == 4


_POST_AS_DISABLED = "posting as a company page is disabled"


@pytest.mark.parametrize("confirm", [False, True])
async def test_post_as_is_refused_while_company_page_tools_are_off(
    mock_context, confirm
):
    extractor = MagicMock()
    extractor.create_post = AsyncMock(return_value={"status": "published"})
    extractor.create_poll = AsyncMock(return_value={"status": "published"})
    create_post = await _tool("create_post", company_page_tools=False)
    create_poll = await _tool("create_poll", company_page_tools=False)

    with patch(
        "linkedin_mcp_server.tools.posting.get_ready_extractor", new=AsyncMock()
    ) as ready:
        with pytest.raises(ToolError, match=_POST_AS_DISABLED):
            await create_post(
                "Hello", confirm, mock_context, post_as="12345", extractor=extractor
            )
        with pytest.raises(ToolError, match="ENABLE_COMPANY_PAGE_TOOLS=true"):
            await create_poll(
                "Best format?",
                ["Carousel", "Video"],
                3,
                confirm,
                mock_context,
                post_as="https://www.linkedin.com/company/12345/",
                extractor=extractor,
            )

    ready.assert_not_awaited()
    extractor.create_post.assert_not_awaited()
    extractor.create_poll.assert_not_awaited()


async def test_the_company_page_switch_defaults_to_off(mock_context, monkeypatch):
    monkeypatch.delenv("ENABLE_COMPANY_PAGE_TOOLS", raising=False)
    create_post = await _tool("create_post")

    with pytest.raises(ToolError, match=_POST_AS_DISABLED):
        await create_post("Hello", False, mock_context, post_as="12345")
    # Without post_as the default server previews exactly as before.
    result = await create_post("Hello", False, mock_context)
    assert result["status"] == "preview"


async def test_the_company_page_switch_is_read_from_the_environment(
    mock_context, monkeypatch
):
    monkeypatch.setenv("ENABLE_COMPANY_PAGE_TOOLS", "true")
    create_post = await _tool("create_post")

    result = await create_post("Hello", False, mock_context, post_as="12345")

    assert result["post_as"]["kind"] == "company"
