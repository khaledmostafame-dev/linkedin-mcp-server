"""Unit tests for comment references, input validation and the comment tools.

The browser programs are covered by ``test_comments_dom.py``; everything here
runs without a browser. All identifiers are synthetic.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import FunctionTool

from linkedin_mcp_server.core.exceptions import InvalidReferenceError
from linkedin_mcp_server.scraping.comments import (
    COMMENT_MAX_LENGTH,
    CommentUrn,
    build_comment_references,
    comment_permalink,
    normalize_comment_text,
    parse_comment_urn,
    posted_at_from_id,
    prepare_comment_reaction,
    prepare_comment_write,
)
from linkedin_mcp_server.scraping.identifiers import (
    normalize_post_urn,
    post_update_url,
)

POST = "urn:li:activity:7300000000000000000"
COMMENT = "urn:li:comment:(activity:7300000000000000000,7300000000000000101)"


class TestPostUrn:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (POST, POST),
            (
                "urn:li:ugcPost:7300000000000000001",
                "urn:li:ugcPost:7300000000000000001",
            ),
            ("urn:li:share:7300000000000000002", "urn:li:share:7300000000000000002"),
            ("urn%3Ali%3Aactivity%3A7300000000000000000", POST),
            (f"https://www.linkedin.com/feed/update/{POST}/", POST),
            (f"/feed/update/{POST}/?commentUrn=x&trk=y", POST),
            (f"linkedin.com/feed/update/{POST}", POST),
            (
                "https://www.linkedin.com/posts/synthetic-user_some-title-activity-7300000000000000000-AbCd",
                POST,
            ),
            (
                "/posts/synthetic-user_title-ugcPost-7300000000000000001-Zz/",
                "urn:li:ugcPost:7300000000000000001",
            ),
        ],
    )
    def test_accepted_forms_normalize_to_the_urn(self, value, expected):
        assert normalize_post_urn(value) == expected

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "urn:li:activity:abc",
            "urn:li:comment:(activity:1,2)",
            "https://www.linkedin.com/in/synthetic-user/",
            "https://example.com/feed/update/urn:li:activity:1/",
            f"https://www.linkedin.com/feed/update/{POST}/../../in/x/",
            "https://www.linkedin.com/posts/synthetic-user_no-id-here",
            "https://www.linkedin.com:444/feed/update/urn:li:activity:1/",
        ],
    )
    def test_anything_else_is_refused(self, value):
        with pytest.raises(InvalidReferenceError):
            normalize_post_urn(value)

    def test_update_url_is_canonical(self):
        assert post_update_url(POST) == f"https://www.linkedin.com/feed/update/{POST}/"


class TestCommentUrn:
    @pytest.mark.parametrize(
        "value",
        [
            COMMENT,
            "urn:li:comment:(urn:li:activity:7300000000000000000,7300000000000000101)",
            "urn:li:fsd_comment:(7300000000000000101,urn:li:activity:7300000000000000000)",
            "urn%3Ali%3Acomment%3A%28activity%3A7300000000000000000%2C7300000000000000101%29",
        ],
    )
    def test_every_observed_spelling_parses_to_one_comment(self, value):
        parsed = parse_comment_urn(value)
        assert parsed == CommentUrn(
            "activity", "7300000000000000000", "7300000000000000101"
        )
        assert parsed is not None and parsed.urn == COMMENT

    @pytest.mark.parametrize(
        "value",
        ["", "urn:li:comment:(activity:1)", "urn:li:comment:(post:1,2)", POST, "%ZZ"],
    )
    def test_malformed_urns_do_not_parse(self, value):
        assert parse_comment_urn(value) is None

    def test_permalinks_encode_the_comment_and_reply(self):
        top = CommentUrn("activity", "1", "2")
        reply = CommentUrn("activity", "1", "3")
        assert comment_permalink(top) == (
            "/feed/update/urn:li:activity:1/?commentUrn="
            "urn%3Ali%3Acomment%3A%28activity%3A1%2C2%29"
        )
        assert comment_permalink(reply, top).endswith(
            "&replyUrn=urn%3Ali%3Acomment%3A%28activity%3A1%2C3%29"
        )

    def test_posted_at_decodes_the_id_timestamp_and_rejects_implausible_ids(self):
        # 1_700_000_000_000 ms (2023-11-14T22:13:20Z) shifted into the id.
        assert posted_at_from_id(str(1_700_000_000_000 << 22)) == "2023-11-14T22:13:20Z"
        assert posted_at_from_id("12345") is None
        assert posted_at_from_id(str(9_999_999_999_999 << 22)) is None


class TestCommentText:
    def test_line_breaks_are_normalized_and_kept(self):
        assert normalize_comment_text("  one\r\ntwo\rthree ") == (
            "one\ntwo\nthree",
            None,
        )

    @pytest.mark.parametrize(
        "text",
        ["", "   \n ", "tab\there", "bell\x07", "del\x7f", "hi @someone", "x\ud800"],
    )
    def test_unsafe_text_is_refused(self, text):
        normalized, reason = normalize_comment_text(text)
        assert normalized is None and reason

    def test_length_limit_is_exactly_linkedins(self):
        assert normalize_comment_text("a" * COMMENT_MAX_LENGTH)[0] is not None
        assert normalize_comment_text("a" * (COMMENT_MAX_LENGTH + 1))[0] is None


def _item(comment_id: str, parent_id: str | None = None, **extra: Any) -> dict:
    item = {
        "kind": "activity",
        "threadId": "1",
        "id": comment_id,
        "parentKind": "activity" if parent_id else None,
        "parentThreadId": "1" if parent_id else None,
        "parentId": parent_id,
        "authorPath": f"/in/synthetic-{comment_id}/",
        "authorLines": [f"Name {comment_id}", f"Name {comment_id}", "Headline"],
        "text": f"text {comment_id}",
    }
    item.update(extra)
    return item


class TestCommentReferences:
    def test_shape_of_one_reference(self):
        (ref,) = build_comment_references(
            [_item("10")], include_replies=True, sort="relevant", max_comments=5
        )
        assert ref == {
            "kind": "comment",
            "url": "/in/synthetic-10/",
            "value": "urn:li:comment:(activity:1,10)",
            "permalink": comment_permalink(CommentUrn("activity", "1", "10")),
            "text": "Name 10",
            "context": "Headline",
            "excerpt": "text 10",
        }

    def test_recent_orders_threads_newest_first_and_caps(self):
        items = [_item("10"), _item("11", "10"), _item("30"), _item("12", "10")]
        refs = build_comment_references(
            items, include_replies=True, sort="recent", max_comments=3
        )
        assert [ref["value"][-4:] for ref in refs] == [",30)", ",10)", ",12)"]

    def test_replies_are_dropped_on_request_and_bad_items_skipped(self):
        items = [_item("10"), _item("11", "10"), {"id": "x"}, _item("10")]
        refs = build_comment_references(
            items, include_replies=False, sort="relevant", max_comments=10
        )
        assert [ref["value"] for ref in refs] == ["urn:li:comment:(activity:1,10)"]

    def test_a_missing_author_falls_back_to_the_permalink(self):
        (ref,) = build_comment_references(
            [_item("10", authorPath="javascript:alert(1)", authorLines=[])],
            include_replies=True,
            sort="relevant",
            max_comments=1,
        )
        assert ref["url"] == ref["permalink"]
        assert "text" not in ref


class TestPrepare:
    def test_preview_never_needs_a_browser(self):
        result = prepare_comment_write(POST, COMMENT, "Thanks!\nMore", confirm=False)
        assert result is not None
        assert result["status"] == "preview"
        assert result["text"] == "Thanks!\nMore"
        assert result["comment_urn"] == COMMENT
        assert result["posted"] is False and result["retry_safe"] is True

    def test_confirmed_valid_input_needs_the_browser(self):
        assert prepare_comment_write(POST, None, "Hi", confirm=True) is None
        assert prepare_comment_reaction(POST, COMMENT, "like", confirm=True) is None

    def test_invalid_text_is_answered_even_when_confirmed(self):
        result = prepare_comment_write(POST, None, "a@b", confirm=True)
        assert result is not None and result["status"] == "invalid_text"

    def test_bad_references_raise_the_correction(self):
        with pytest.raises(InvalidReferenceError):
            prepare_comment_write(POST, "not-a-urn", "Hi", confirm=False)
        with pytest.raises(InvalidReferenceError):
            prepare_comment_write(
                "https://www.linkedin.com/in/x/", None, "Hi", confirm=False
            )

    def test_only_like_is_supported(self):
        result = prepare_comment_reaction(POST, COMMENT, "love", confirm=True)
        assert result is not None and result["status"] == "unsupported_reaction"


async def _tool(name: str) -> FunctionTool:
    from linkedin_mcp_server.tools.comments import register_comment_tools

    mcp = FastMCP("test")
    register_comment_tools(mcp)
    tool = await mcp.get_tool(name)
    assert isinstance(tool, FunctionTool)
    return tool


class TestCommentTools:
    async def test_annotations_tags_and_required_confirm(self):
        read = await _tool("get_post_comments")
        assert read.annotations is not None and read.annotations.readOnlyHint is True
        assert {"read", "comments"} <= read.tags
        for name in ("reply_to_comment", "comment_on_post", "react_to_comment"):
            tool = await _tool(name)
            assert tool.annotations is not None
            assert tool.annotations.destructiveHint is True
            assert tool.annotations.openWorldHint is True
            assert "write" in tool.tags
            assert "confirm" in tool.parameters["required"]

    async def test_preview_does_not_acquire_a_browser(self, mock_context):
        tool = await _tool("reply_to_comment")
        with patch(
            "linkedin_mcp_server.tools.comments.get_ready_extractor",
            new=AsyncMock(side_effect=AssertionError("browser acquired")),
        ):
            result = await tool.fn(POST, COMMENT, "Thanks", False, mock_context)
        assert result["status"] == "preview"

    async def test_confirmed_reply_delegates_to_the_extractor(self, mock_context):
        tool = await _tool("reply_to_comment")
        extractor = MagicMock()
        extractor.reply_to_comment = AsyncMock(
            return_value={"status": "posted", "retry_safe": False}
        )
        result = await tool.fn(
            POST, COMMENT, "Thanks", True, mock_context, extractor=extractor
        )
        assert result["status"] == "posted"
        extractor.reply_to_comment.assert_awaited_once_with(
            POST, COMMENT, "Thanks", confirm=True
        )

    async def test_get_post_comments_forwards_its_arguments(self, mock_context):
        tool = await _tool("get_post_comments")
        extractor = MagicMock()
        extractor.get_post_comments = AsyncMock(
            return_value={"url": "u", "sections": {}}
        )
        await tool.fn(
            POST,
            mock_context,
            max_comments=7,
            include_replies=False,
            sort="recent",
            extractor=extractor,
        )
        extractor.get_post_comments.assert_awaited_once_with(
            POST, max_comments=7, include_replies=False, sort="recent"
        )

    async def test_an_invalid_urn_surfaces_as_a_tool_error(self, mock_context):
        tool = await _tool("react_to_comment")
        with pytest.raises(ToolError, match="comment URN"):
            await tool.fn(POST, "nope", True, mock_context, extractor=MagicMock())

    async def test_schema_rejects_out_of_range_max_comments(self):
        from fastmcp.exceptions import ValidationError

        from linkedin_mcp_server.tools.comments import register_comment_tools

        mcp = FastMCP("test")
        register_comment_tools(mcp)
        with pytest.raises(ValidationError, match="max_comments"):
            await mcp.call_tool(
                "get_post_comments", {"post_url": POST, "max_comments": 0}
            )
