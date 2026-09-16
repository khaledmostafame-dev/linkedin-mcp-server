"""Unit tests for the analytics helpers and tools. Synthetic data only."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import FunctionTool

from linkedin_mcp_server.scraping.analytics import (
    PROFILE_ANALYTICS_SECTIONS,
    build_metrics,
    parse_count,
    parse_profile_analytics_sections,
    post_analytics_url,
)

ARABIC_1234 = "".join(map(chr, [0x661, 0x66C, 0x662, 0x663, 0x664]))
RLM, NNBSP = chr(0x200F), chr(0x202F)


class TestParseCount:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("7", 7),
            ("1,234", 1234),
            ("1.234.567", 1234567),
            ("4 567", 4567),
            (ARABIC_1234, 1234),
            (f"{RLM}12{NNBSP}345", 12345),
        ],
    )
    def test_counts_in_any_digit_script_and_grouping(self, raw, expected):
        assert parse_count(raw) == expected

    @pytest.mark.parametrize("raw", ["", "1.2K", "12.5", "12%", "1,23", "abc", "-3"])
    def test_anything_that_is_not_a_plain_count_is_refused(self, raw):
        assert parse_count(raw) is None


class TestMetrics:
    def test_pairs_become_metrics_and_english_labels_get_names(self):
        metrics, named = build_metrics(
            [
                {"label": "Impressions", "value": "1,234"},
                {"label": "Members  reached", "value": "900"},
                {"label": ARABIC_1234, "value": "x"},
                {"label": "Unmapped label", "value": "5"},
                "junk",
            ]
        )
        assert metrics == [
            {"label": "Impressions", "value": 1234},
            {"label": "Members reached", "value": 900},
            {"label": "Unmapped label", "value": 5},
        ]
        assert named == {"impressions": 1234, "members_reached": 900}

    def test_other_locales_keep_raw_pairs_without_names(self):
        metrics, named = build_metrics([{"label": "مرات الظهور", "value": ARABIC_1234}])
        assert metrics == [{"label": "مرات الظهور", "value": 1234}]
        assert named == {}


class TestSections:
    def test_default_is_every_dashboard_in_order(self):
        assert parse_profile_analytics_sections(None) == (
            list(PROFILE_ANALYTICS_SECTIONS),
            [],
        )

    def test_selection_is_canonical_and_unknown_names_are_reported(self):
        assert parse_profile_analytics_sections(
            " Followers ,bogus,profile_viewers"
        ) == (
            ["profile_viewers", "followers"],
            ["bogus"],
        )

    def test_post_summary_route(self):
        assert post_analytics_url("urn:li:activity:1") == (
            "https://www.linkedin.com/analytics/post-summary/urn:li:activity:1/"
        )


async def _tool(name: str) -> FunctionTool:
    from linkedin_mcp_server.tools.analytics import register_analytics_tools

    mcp = FastMCP("test")
    register_analytics_tools(mcp)
    tool = await mcp.get_tool(name)
    assert isinstance(tool, FunctionTool)
    return tool


class TestAnalyticsTools:
    async def test_both_tools_are_read_only(self):
        for name in ("get_post_analytics", "get_profile_analytics"):
            tool = await _tool(name)
            assert tool.annotations is not None
            assert tool.annotations.readOnlyHint is True
            assert "read" in tool.tags

    async def test_a_non_post_is_refused_before_a_browser(self, mock_context):
        tool = await _tool("get_post_analytics")
        with patch(
            "linkedin_mcp_server.tools.analytics.get_ready_extractor",
            new=AsyncMock(side_effect=AssertionError("browser acquired")),
        ):
            with pytest.raises(ToolError, match="not a LinkedIn post"):
                await tool.fn("https://www.linkedin.com/in/x/", mock_context)

    async def test_unknown_sections_only_are_refused_before_a_browser(
        self, mock_context
    ):
        tool = await _tool("get_profile_analytics")
        with patch(
            "linkedin_mcp_server.tools.analytics.get_ready_extractor",
            new=AsyncMock(side_effect=AssertionError("browser acquired")),
        ):
            with pytest.raises(ToolError, match="Unknown analytics sections"):
                await tool.fn(mock_context, sections="bogus")

    async def test_delegation(self, mock_context):
        extractor = MagicMock()
        extractor.get_post_analytics = AsyncMock(return_value={"url": "u"})
        extractor.get_profile_analytics = AsyncMock(return_value={"url": "v"})
        post = await _tool("get_post_analytics")
        profile = await _tool("get_profile_analytics")
        urn = "urn:li:activity:7300000000000000000"
        assert (await post.fn(urn, mock_context, extractor=extractor)) == {"url": "u"}
        assert (
            await profile.fn(mock_context, sections="followers", extractor=extractor)
        ) == {"url": "v"}
        extractor.get_post_analytics.assert_awaited_once_with(urn)
        extractor.get_profile_analytics.assert_awaited_once_with("followers")
