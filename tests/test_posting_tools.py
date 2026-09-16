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


async def _tool(name: str) -> Callable[..., Coroutine[Any, Any, dict[str, Any]]]:
    mcp = FastMCP("test")
    register_posting_tools(mcp)
    tool = await mcp.get_tool(name)
    assert tool is not None
    return cast(FunctionTool, tool).fn


async def test_write_and_read_annotations_follow_the_shared_contract():
    mcp = FastMCP("test")
    register_posting_tools(mcp)
    tools = {tool.name: tool for tool in await mcp.list_tools()}

    for name in ("create_post", "delete_scheduled_post"):
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
