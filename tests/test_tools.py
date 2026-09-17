import asyncio
import logging
from typing import Any, Callable, Coroutine, cast
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from fastmcp import FastMCP
from fastmcp.tools import FunctionTool

from linkedin_mcp_server.callbacks import MCPContextProgressCallback
from linkedin_mcp_server.scraping.contracts import (
    RATE_LIMITED_SECTION_TEXT,
    SEND_INTERRUPTED_WARNING,
)
from linkedin_mcp_server.scraping.contracts import ExtractedSection


async def get_tool_fn(
    mcp: FastMCP, name: str
) -> Callable[..., Coroutine[Any, Any, dict[str, Any]]]:
    """Extract tool function from FastMCP by name using public API."""
    tool = await mcp.get_tool(name)
    if tool is None:
        raise ValueError(f"Tool '{name}' not found")
    return cast(FunctionTool, tool).fn


def _make_mock_extractor(scrape_result: dict) -> MagicMock:
    """Create a mock LinkedInExtractor that returns the given result."""
    mock = MagicMock()
    mock.scrape_person = AsyncMock(return_value=scrape_result)
    mock.connect_with_person = AsyncMock(return_value=scrape_result)
    mock.scrape_company = AsyncMock(return_value=scrape_result)
    mock.scrape_job = AsyncMock(return_value=scrape_result)
    mock.search_jobs = AsyncMock(return_value=scrape_result)
    mock.get_saved_jobs = AsyncMock(return_value=scrape_result)
    mock.get_job_alerts = AsyncMock(return_value=scrape_result)
    mock.search_people = AsyncMock(return_value=scrape_result)
    mock.resolve_geo_location = AsyncMock(return_value=scrape_result)
    mock.get_sidebar_profiles = AsyncMock(return_value=scrape_result)
    mock.get_inbox = AsyncMock(return_value=scrape_result)
    mock.get_conversation = AsyncMock(return_value=scrape_result)
    mock.search_conversations = AsyncMock(return_value=scrape_result)
    mock.send_message = AsyncMock(return_value=scrape_result)
    mock.reply_to_conversation = AsyncMock(return_value=scrape_result)
    mock.mark_conversation_read = AsyncMock(return_value=scrape_result)
    mock.archive_conversation = AsyncMock(return_value=scrape_result)
    mock.save_job = AsyncMock(return_value=scrape_result)
    mock.get_my_profile = AsyncMock(return_value=scrape_result)
    mock.search_companies = AsyncMock(return_value=scrape_result)
    mock.search_posts = AsyncMock(return_value=scrape_result)
    mock.get_company_employees = AsyncMock(return_value=scrape_result)
    mock.get_mutual_connections = AsyncMock(return_value=scrape_result)
    mock.list_connections = AsyncMock(return_value=scrape_result)
    mock.get_invitations = AsyncMock(return_value=scrape_result)
    mock.withdraw_invitation = AsyncMock(return_value=scrape_result)
    mock.respond_to_invitation = AsyncMock(return_value=scrape_result)
    mock.follow = AsyncMock(return_value=scrape_result)
    mock.search_groups = AsyncMock(return_value=scrape_result)
    mock.get_group_posts = AsyncMock(return_value=scrape_result)
    mock.get_group_members = AsyncMock(return_value=scrape_result)
    mock.search_events = AsyncMock(return_value=scrape_result)
    mock.get_event_details = AsyncMock(return_value=scrape_result)
    mock.get_event_attendees = AsyncMock(return_value=scrape_result)
    mock.extract_page = AsyncMock(
        return_value=ExtractedSection(text="some text", references=[])
    )
    mock.extract_feed = AsyncMock(return_value=ExtractedSection(text="", references=[]))
    return mock


class TestPersonTool:
    async def test_get_person_profile_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/in/test-user/",
            "sections": {"main_profile": "John Doe\nSoftware Engineer"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_person_profile")
        result = await tool_fn("test-user", mock_context, extractor=mock_extractor)
        assert result["url"] == "https://www.linkedin.com/in/test-user/"
        assert "main_profile" in result["sections"]
        assert "pages_visited" not in result
        assert "sections_requested" not in result

    async def test_get_person_profile_with_sections(self, mock_context):
        """Verify sections parameter is passed through."""
        expected = {
            "url": "https://www.linkedin.com/in/test-user/",
            "sections": {
                "main_profile": "John Doe",
                "experience": "Work history",
                "contact_info": "Email: test@test.com",
            },
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_person_profile")
        result = await tool_fn(
            "test-user",
            mock_context,
            sections="experience,contact_info",
            extractor=mock_extractor,
        )
        assert "main_profile" in result["sections"]
        assert "experience" in result["sections"]
        assert "contact_info" in result["sections"]
        # Verify scrape_person was called exactly once with a set[str]
        mock_extractor.scrape_person.assert_awaited_once()
        call_args = mock_extractor.scrape_person.call_args
        assert isinstance(call_args[0][1], set)
        assert "experience" in call_args[0][1]
        assert "contact_info" in call_args[0][1]

    async def test_get_person_profile_passes_callbacks(self, mock_context):
        """Verify tool wires MCPContextProgressCallback to the extractor."""
        expected = {
            "url": "https://www.linkedin.com/in/test-user/",
            "sections": {"main_profile": "John Doe"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_person_profile")
        await tool_fn("test-user", mock_context, extractor=mock_extractor)

        call_kwargs = mock_extractor.scrape_person.call_args.kwargs
        assert "callbacks" in call_kwargs
        assert isinstance(call_kwargs["callbacks"], MCPContextProgressCallback)

    async def test_get_person_profile_passes_max_scrolls(self, mock_context):
        """Verify max_scrolls parameter is forwarded to scrape_person."""
        expected = {
            "url": "https://www.linkedin.com/in/test-user/",
            "sections": {"main_profile": "John Doe"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_person_profile")
        await tool_fn(
            "test-user",
            mock_context,
            max_scrolls=15,
            extractor=mock_extractor,
        )

        call_kwargs = mock_extractor.scrape_person.call_args.kwargs
        assert call_kwargs["max_scrolls"] == 15

    async def test_get_person_profile_rejects_invalid_max_scrolls(self, mock_context):
        """Verify max_scrolls=0 is rejected by Field(ge=1) validation."""
        # FastMCP wraps the pydantic error raised by Field() constraints in
        # its own ValidationError, which does not subclass pydantic's.
        from fastmcp.exceptions import ValidationError

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        with pytest.raises(ValidationError, match="max_scrolls"):
            await mcp.call_tool(
                "get_person_profile",
                {"linkedin_username": "test-user", "max_scrolls": 0},
            )

    async def test_get_person_profile_unknown_section(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/in/test-user/",
            "sections": {"main_profile": "John Doe"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_person_profile")
        result = await tool_fn(
            "test-user",
            mock_context,
            sections="bogus_section",
            extractor=mock_extractor,
        )
        assert result["unknown_sections"] == ["bogus_section"]

    async def test_get_person_profile_error(self, mock_context):
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.exceptions import SessionExpiredError

        mock_extractor = MagicMock()
        mock_extractor.scrape_person = AsyncMock(side_effect=SessionExpiredError())

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_person_profile")
        with pytest.raises(ToolError, match="Session expired"):
            await tool_fn("test-user", mock_context, extractor=mock_extractor)

    async def test_get_person_profile_auth_error(self, monkeypatch):
        """Auth failures in the DI layer trigger auto-relogin and report the login browser."""
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.core.exceptions import AuthenticationError
        from linkedin_mcp_server.exceptions import AuthenticationStartedError

        mock_browser = MagicMock()
        mock_browser.page = MagicMock()
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.ensure_tool_ready_or_raise",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.get_or_create_browser",
            AsyncMock(return_value=mock_browser),
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.ensure_authenticated",
            AsyncMock(side_effect=AuthenticationError("Session expired or invalid.")),
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.get_runtime_policy",
            lambda: "managed",
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.close_browser",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.invalidate_auth_and_trigger_relogin",
            AsyncMock(
                side_effect=AuthenticationStartedError(
                    "Session expired. A login browser window has been opened."
                )
            ),
        )

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        with pytest.raises(ToolError, match="Session expired"):
            await mcp.call_tool("get_person_profile", {"linkedin_username": "test"})

    async def test_search_people(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/search/results/people/?keywords=AI+engineer&location=New+York",
            "sections": {"search_results": "Jane Doe\nAI Engineer at Acme\nNew York"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "search_people")
        result = await tool_fn(
            "AI engineer", mock_context, location="New York", extractor=mock_extractor
        )
        assert "search_results" in result["sections"]
        assert "pages_visited" not in result
        mock_extractor.search_people.assert_awaited_once_with(
            "AI engineer",
            "New York",
            network=None,
            current_company=None,
            past_company=None,
            school=None,
            industry=None,
            title=None,
            profile_language=None,
            max_pages=1,
            tool_timeout=ANY,
        )

    async def test_search_people_with_network_and_company_filters(self, mock_context):
        expected = {
            "url": (
                "https://www.linkedin.com/search/results/people/"
                "?keywords=engineer&network=%5B%22F%22%5D"
                "&currentCompany=%5B%221115%22%5D"
            ),
            "sections": {
                "search_results": "Jennifer Bonuso\nPresident Americas at SAP"
            },
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "search_people")
        result = await tool_fn(
            "engineer",
            mock_context,
            network=["F"],
            current_company="1115",
            extractor=mock_extractor,
        )
        assert "search_results" in result["sections"]
        mock_extractor.search_people.assert_awaited_once_with(
            "engineer",
            None,
            network=["F"],
            current_company="1115",
            past_company=None,
            school=None,
            industry=None,
            title=None,
            profile_language=None,
            max_pages=1,
            tool_timeout=ANY,
        )

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("F", ["F"]),
            ('["F"]', ["F"]),
            ('["F", "S"]', ["F", "S"]),
            ("F,S", ["F", "S"]),
            (" F , S ", ["F", "S"]),
            ("", []),
            (["F"], ["F"]),
            (None, None),
        ],
    )
    def test_coerce_str_list_repairs_stringified_arrays(self, raw, expected):
        """A client that flattens the array must still produce a list.

        Lists and None pass through untouched, so a well-behaved client is
        unaffected.
        """
        from linkedin_mcp_server.tools.person import _coerce_str_list

        assert _coerce_str_list(raw) == expected

    async def test_search_people_accepts_stringified_network(self, monkeypatch):
        """Regression for #739.

        The published schema for ``network`` is ``anyOf: [array, null]``, but
        clients that collapse that union send a bare string. Going through
        ``call_tool`` exercises the pydantic validation the direct-``fn`` tests
        skip, which is where the original failure lived.
        """
        import linkedin_mcp_server.tools.person as person_module
        from linkedin_mcp_server.tools.person import register_person_tools

        expected = {
            "url": (
                "https://www.linkedin.com/search/results/people/"
                "?keywords=engineer&network=%5B%22F%22%5D"
            ),
            "sections": {"search_results": "Jane Doe"},
        }
        mock_extractor = _make_mock_extractor(expected)

        async def _fake_get_ready_extractor(ctx, tool_name):
            return mock_extractor

        monkeypatch.setattr(
            person_module, "get_ready_extractor", _fake_get_ready_extractor
        )

        mcp = FastMCP("test")
        register_person_tools(mcp)

        await mcp.call_tool("search_people", {"keywords": "engineer", "network": "F"})

        mock_extractor.search_people.assert_awaited_once_with(
            "engineer",
            None,
            network=["F"],
            current_company=None,
            past_company=None,
            school=None,
            industry=None,
            title=None,
            profile_language=None,
            max_pages=1,
            tool_timeout=ANY,
        )

    async def test_search_people_validation_error_surfaced_as_tool_error(
        self, mock_context
    ):
        """A FilterValidationError raised by the extractor should surface to
        the MCP client as a ToolError carrying the same message, rather than
        being collapsed to the generic "Error calling tool" mask."""
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.scraping.contracts import FilterValidationError
        from linkedin_mcp_server.tools.person import register_person_tools

        mock_extractor = MagicMock()
        mock_extractor.search_people = AsyncMock(
            side_effect=FilterValidationError("must be a numeric URN")
        )

        mcp = FastMCP("test")
        register_person_tools(mcp)
        tool_fn = await get_tool_fn(mcp, "search_people")

        with pytest.raises(ToolError, match="must be a numeric URN"):
            await tool_fn(
                "engineer",
                mock_context,
                current_company="SAP",
                extractor=mock_extractor,
            )

    async def test_resolve_geo_location_success(self, mock_context):
        expected = {
            "query": "Dubai",
            "candidates": [
                {"name": "Dubai, United Arab Emirates", "geo_urn_id": "104246948"}
            ],
            "ambiguous": False,
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "resolve_geo_location")
        result = await tool_fn("Dubai", mock_context, extractor=mock_extractor)

        assert result == expected
        mock_extractor.resolve_geo_location.assert_awaited_once_with("Dubai")

    async def test_resolve_geo_location_validation_error_surfaced_as_tool_error(
        self, mock_context
    ):
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.scraping.contracts import FilterValidationError
        from linkedin_mcp_server.tools.person import register_person_tools

        mock_extractor = MagicMock()
        mock_extractor.resolve_geo_location = AsyncMock(
            side_effect=FilterValidationError("location query must not be blank")
        )

        mcp = FastMCP("test")
        register_person_tools(mcp)
        tool_fn = await get_tool_fn(mcp, "resolve_geo_location")

        with pytest.raises(ToolError, match="must not be blank"):
            await tool_fn("   ", mock_context, extractor=mock_extractor)

    async def test_connect_with_person(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/in/test-user/",
            "status": "connected",
            "message": "Connection request sent.",
            "note_sent": True,
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "connect_with_person")
        result = await tool_fn(
            "test-user",
            mock_context,
            confirm=True,
            note="Let us connect.",
            extractor=mock_extractor,
        )

        assert result["status"] == "connected"
        assert result["note_sent"] is True
        mock_extractor.connect_with_person.assert_awaited_once_with(
            "test-user",
            note="Let us connect.",
        )

    async def test_connect_with_person_no_note(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/in/test-user/",
            "status": "connected",
            "message": "Connection request sent.",
            "note_sent": False,
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "connect_with_person")
        result = await tool_fn(
            "test-user",
            mock_context,
            confirm=True,
            extractor=mock_extractor,
        )

        assert result["status"] == "connected"
        mock_extractor.connect_with_person.assert_awaited_once_with(
            "test-user",
            note=None,
        )

    async def test_connect_with_person_custom_note_limit_reached(self, mock_context):
        """The custom_note_limit_reached status returns LinkedIn's message."""
        expected = {
            "url": "https://www.linkedin.com/in/test-user/",
            "status": "custom_note_limit_reached",
            "message": "Wysyłaj nieograniczoną liczbę spersonalizowanych zaproszeń dzięki Premium",
            "note_sent": False,
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "connect_with_person")
        result = await tool_fn(
            "test-user",
            mock_context,
            confirm=True,
            note="Hello!",
            extractor=mock_extractor,
        )

        assert result["status"] == "custom_note_limit_reached"
        assert (
            result["message"]
            == "Wysyłaj nieograniczoną liczbę spersonalizowanych zaproszeń dzięki Premium"
        )
        assert result["note_sent"] is False
        mock_extractor.connect_with_person.assert_awaited_once_with(
            "test-user",
            note="Hello!",
        )

    async def test_connect_with_person_auth_error(self, monkeypatch):
        """Auth failures in the DI layer trigger auto-relogin and report the login browser."""
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.core.exceptions import AuthenticationError
        from linkedin_mcp_server.exceptions import AuthenticationStartedError

        mock_browser = MagicMock()
        mock_browser.page = MagicMock()
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.ensure_tool_ready_or_raise",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.get_or_create_browser",
            AsyncMock(return_value=mock_browser),
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.ensure_authenticated",
            AsyncMock(side_effect=AuthenticationError("Session expired or invalid.")),
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.get_runtime_policy",
            lambda: "managed",
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.close_browser",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.invalidate_auth_and_trigger_relogin",
            AsyncMock(
                side_effect=AuthenticationStartedError(
                    "Session expired. A login browser window has been opened."
                )
            ),
        )

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        with pytest.raises(ToolError, match="Session expired"):
            await mcp.call_tool(
                "connect_with_person",
                {"linkedin_username": "test", "confirm": True},
            )

    async def test_connect_with_person_preview_never_acquires_a_browser(
        self, mock_context
    ):
        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)
        tool_fn = await get_tool_fn(mcp, "connect_with_person")
        refuse = AsyncMock(side_effect=AssertionError("preview acquired a browser"))

        with patch("linkedin_mcp_server.tools.person.get_ready_extractor", refuse):
            result = await tool_fn(
                "https://www.linkedin.com/in/test-user/",
                mock_context,
                confirm=False,
                note="Hello",
            )

        refuse.assert_not_awaited()
        assert result["status"] == "preview"
        assert result["url"] == "https://www.linkedin.com/in/test-user/"
        assert result["note_sent"] is False


class TestCompanyTools:
    async def test_get_company_profile(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/company/testcorp/",
            "sections": {"about": "TestCorp\nWe build things"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_profile")
        result = await tool_fn("testcorp", mock_context, extractor=mock_extractor)
        assert "about" in result["sections"]
        assert "pages_visited" not in result

    async def test_get_company_posts_normalizes_a_pasted_link(self, mock_context):
        """get_company_posts builds its URL in the tool, not in the extractor.

        That makes it the one wiring point the extractor tests cannot reach, and
        the only place a pasted company link would still become
        /company/https://de.linkedin.com/company/testcorp/posts/.
        """
        mock_extractor = _make_mock_extractor({})

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_posts")
        result = await tool_fn(
            "https://de.linkedin.com/company/testcorp/",
            mock_context,
            extractor=mock_extractor,
        )
        assert result["url"] == "https://www.linkedin.com/company/testcorp/posts/"
        assert (
            mock_extractor.extract_page.call_args.args[0]
            == "https://www.linkedin.com/company/testcorp/posts/"
        )

    async def test_get_company_posts_refuses_a_traversal_value(self, mock_context):
        mock_extractor = _make_mock_extractor({})

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_posts")
        with pytest.raises(Exception):
            await tool_fn("../../feed", mock_context, extractor=mock_extractor)
        mock_extractor.extract_page.assert_not_called()

    async def test_get_company_profile_passes_callbacks(self, mock_context):
        """Verify tool wires MCPContextProgressCallback to the extractor."""
        expected = {
            "url": "https://www.linkedin.com/company/testcorp/",
            "sections": {"about": "TestCorp\nWe build things"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_profile")
        await tool_fn("testcorp", mock_context, extractor=mock_extractor)

        call_kwargs = mock_extractor.scrape_company.call_args.kwargs
        assert "callbacks" in call_kwargs
        assert isinstance(call_kwargs["callbacks"], MCPContextProgressCallback)

    async def test_get_company_profile_unknown_section(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/company/testcorp/",
            "sections": {"about": "TestCorp\nWe build things"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_profile")
        result = await tool_fn(
            "testcorp", mock_context, sections="bogus", extractor=mock_extractor
        )
        assert result["unknown_sections"] == ["bogus"]

    async def test_get_company_posts(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(text="Post 1\nPost 2", references=[])
        )

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_posts")
        result = await tool_fn("testcorp", mock_context, extractor=mock_extractor)
        assert "posts" in result["sections"]
        assert result["sections"]["posts"] == "Post 1\nPost 2"
        assert "pages_visited" not in result
        assert "sections_requested" not in result

    async def test_get_company_posts_omits_rate_limited_sentinel(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(text=RATE_LIMITED_SECTION_TEXT, references=[])
        )

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_posts")
        result = await tool_fn("testcorp", mock_context, extractor=mock_extractor)
        assert result["sections"] == {}
        assert result["section_errors"]["posts"]["error_type"] == "rate_limit"

    async def test_get_company_posts_returns_section_errors(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(
                text="",
                references=[],
                error={"issue_template_path": "/tmp/company-posts-issue.md"},
            )
        )

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_posts")
        result = await tool_fn("testcorp", mock_context, extractor=mock_extractor)
        assert result["sections"] == {}
        assert result["section_errors"]["posts"]["issue_template_path"] == (
            "/tmp/company-posts-issue.md"
        )

    async def test_get_company_posts_omits_orphaned_references(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(
                text="",
                references=[
                    {
                        "kind": "company",
                        "url": "/company/testcorp/",
                        "text": "TestCorp",
                    }
                ],
            )
        )

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_posts")
        result = await tool_fn("testcorp", mock_context, extractor=mock_extractor)
        assert result["sections"] == {}
        assert "references" not in result


class TestJobTools:
    async def test_get_job_details(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/jobs/view/12345/",
            "sections": {"job_posting": "Software Engineer\nGreat opportunity"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.job import register_job_tools

        mcp = FastMCP("test")
        register_job_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_job_details")
        result = await tool_fn("12345", mock_context, extractor=mock_extractor)
        assert "job_posting" in result["sections"]
        assert "pages_visited" not in result

    async def test_search_jobs(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/jobs/search/?keywords=python",
            "sections": {"search_results": "Job 1\nJob 2"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.job import register_job_tools

        mcp = FastMCP("test")
        register_job_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "search_jobs")
        result = await tool_fn(
            "python", mock_context, location="Remote", extractor=mock_extractor
        )
        assert "search_results" in result["sections"]
        assert "pages_visited" not in result

    async def test_search_jobs_is_bounded_by_the_registered_timeout(self, mock_context):
        """The loop stops itself by the same figure FastMCP cancels on.

        Registered with a non-default timeout, because dropping the argument
        leaves the extractor budgeting against its own 180s default while
        FastMCP still cancels at 60s. The search would then be killed mid-page
        and every page already gathered thrown away, which is the loss the
        bound exists to prevent.

        What arrives is what is left of it: FastMCP starts its clock before
        this tool runs, and acquiring the browser is spent from the same 60s.
        A cold start of three seconds passed on as a full sixty is the same
        defect one layer down.
        """
        mock_extractor = _make_mock_extractor(
            {
                "url": "https://www.linkedin.com/jobs/search/?keywords=python",
                "sections": {"search_results": "Job 1"},
            }
        )

        from linkedin_mcp_server.tools.job import register_job_tools

        mcp = FastMCP("test")
        register_job_tools(mcp, tool_timeout=60.0)

        tool_fn = await get_tool_fn(mcp, "search_jobs")
        await tool_fn("python", mock_context, extractor=mock_extractor)

        passed = mock_extractor.search_jobs.await_args.kwargs["tool_timeout"]
        assert passed <= 60.0
        assert passed == pytest.approx(60.0, abs=1.0)

    async def test_the_search_budget_pays_for_the_browser_it_waited_on(
        self, mock_context
    ):
        """FastMCP starts its clock before this tool runs.

        Acquiring the browser is spent from the same figure, so passing it on
        whole leaves the extractor planning against time it no longer has. A
        cold start is where this bites: warm, the wait is nothing and the
        budget is the full timeout either way, which is why asserting the
        figure alone cannot see the defect.
        """
        mock_extractor = _make_mock_extractor(
            {
                "url": "https://www.linkedin.com/jobs/search/?keywords=python",
                "sections": {"search_results": "Job 1"},
            }
        )

        async def slow_start(*args, **kwargs):
            await asyncio.sleep(0.3)
            return mock_extractor

        from linkedin_mcp_server.tools.job import register_job_tools

        mcp = FastMCP("test")
        register_job_tools(mcp, tool_timeout=60.0)

        tool_fn = await get_tool_fn(mcp, "search_jobs")
        with patch(
            "linkedin_mcp_server.tools.job.get_ready_extractor",
            side_effect=slow_start,
        ):
            await tool_fn("python", mock_context)

        passed = mock_extractor.search_jobs.await_args.kwargs["tool_timeout"]
        assert passed < 59.9

    async def test_get_saved_jobs(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/my-items/saved-jobs/",
            "sections": {"saved_jobs": "Saved Job 1\nSaved Job 2"},
            "job_ids": ["111", "222"],
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.job import register_job_tools

        mcp = FastMCP("test")
        register_job_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_saved_jobs")
        result = await tool_fn(mock_context, max_pages=2, extractor=mock_extractor)
        assert "saved_jobs" in result["sections"]
        assert result["job_ids"] == ["111", "222"]
        mock_extractor.get_saved_jobs.assert_awaited_once_with(max_pages=2)

    async def test_get_job_alerts(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/my-items/job-alerts/",
            "sections": {"job_alerts": "Python Developer alert\nData Scientist alert"},
            "references": {
                "job_alerts": [
                    {
                        "kind": "job_alert",
                        "url": "https://www.linkedin.com/jobs/search/?keywords=python",
                    }
                ]
            },
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.job import register_job_tools

        mcp = FastMCP("test")
        register_job_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_job_alerts")
        result = await tool_fn(mock_context, extractor=mock_extractor)
        assert "job_alerts" in result["sections"]
        assert result["references"]["job_alerts"][0]["kind"] == "job_alert"
        mock_extractor.get_job_alerts.assert_awaited_once_with()

    async def test_get_job_alerts_error(self, mock_context):
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.exceptions import SessionExpiredError

        mock_extractor = MagicMock()
        mock_extractor.get_job_alerts = AsyncMock(side_effect=SessionExpiredError())

        from linkedin_mcp_server.tools.job import register_job_tools

        mcp = FastMCP("test")
        register_job_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_job_alerts")
        with pytest.raises(ToolError, match="Session expired"):
            await tool_fn(mock_context, extractor=mock_extractor)

    async def test_save_job(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/jobs/view/12345/",
            "job_id": "12345",
            "saved": True,
            "changed": True,
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.job import register_job_tools

        mcp = FastMCP("test")
        register_job_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "save_job")
        result = await tool_fn("12345", True, mock_context, extractor=mock_extractor)
        assert result == expected
        mock_extractor.save_job.assert_awaited_once_with(
            "12345", confirm=True, unsave=False
        )

    async def test_unsave_job(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/jobs/view/12345/",
            "job_id": "12345",
            "saved": False,
            "changed": True,
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.job import register_job_tools

        mcp = FastMCP("test")
        register_job_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "save_job")
        result = await tool_fn(
            "12345", True, mock_context, unsave=True, extractor=mock_extractor
        )
        assert result["saved"] is False
        mock_extractor.save_job.assert_awaited_once_with(
            "12345", confirm=True, unsave=True
        )

    async def test_save_job_error(self, mock_context):
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.exceptions import SessionExpiredError

        mock_extractor = MagicMock()
        mock_extractor.save_job = AsyncMock(side_effect=SessionExpiredError())

        from linkedin_mcp_server.tools.job import register_job_tools

        mcp = FastMCP("test")
        register_job_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "save_job")
        with pytest.raises(ToolError, match="Session expired"):
            await tool_fn("12345", True, mock_context, extractor=mock_extractor)


class TestGetSidebarProfilesTool:
    async def test_get_sidebar_profiles_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/in/test-user/",
            "sidebar_profiles": {
                "more_profiles_for_you": ["/in/alice/", "/in/bob/"],
            },
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_sidebar_profiles")
        result = await tool_fn("test-user", mock_context, extractor=mock_extractor)

        assert result["url"] == "https://www.linkedin.com/in/test-user/"
        assert "more_profiles_for_you" in result["sidebar_profiles"]
        mock_extractor.get_sidebar_profiles.assert_awaited_once_with("test-user")

    async def test_get_sidebar_profiles_empty_result(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/in/test-user/",
            "sidebar_profiles": {},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_sidebar_profiles")
        result = await tool_fn("test-user", mock_context, extractor=mock_extractor)

        assert result["sidebar_profiles"] == {}

    async def test_get_sidebar_profiles_error(self, mock_context):
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.exceptions import SessionExpiredError

        mock_extractor = MagicMock()
        mock_extractor.get_sidebar_profiles = AsyncMock(
            side_effect=SessionExpiredError()
        )

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_sidebar_profiles")
        with pytest.raises(ToolError, match="Session expired"):
            await tool_fn("test-user", mock_context, extractor=mock_extractor)


class TestMessagingTools:
    async def test_get_inbox_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/messaging/",
            "sections": {"inbox": "Conversation 1\nConversation 2"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_inbox")
        result = await tool_fn(mock_context, extractor=mock_extractor)

        assert result["sections"]["inbox"] == "Conversation 1\nConversation 2"
        mock_extractor.get_inbox.assert_awaited_once_with(limit=20)

    async def test_get_conversation_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/messaging/thread/abc123/",
            "sections": {"conversation": "Hello!\nHi there!"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_conversation")
        result = await tool_fn(
            mock_context, linkedin_username="testuser", extractor=mock_extractor
        )

        assert result["sections"]["conversation"] == "Hello!\nHi there!"
        mock_extractor.get_conversation.assert_awaited_once_with(
            linkedin_username="testuser", thread_id=None, index=0
        )

    async def test_search_conversations_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/messaging/",
            "sections": {"search_results": "Result 1\nResult 2"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "search_conversations")
        result = await tool_fn("hello", mock_context, extractor=mock_extractor)

        assert result["sections"]["search_results"] == "Result 1\nResult 2"
        mock_extractor.search_conversations.assert_awaited_once_with("hello", limit=20)

    async def test_send_message_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/messaging/thread/abc123/",
            "status": "sent",
            "message": "Message sent.",
            "recipient_selected": True,
            "sent": True,
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "send_message")
        result = await tool_fn(
            "testuser",
            "Hello!",
            True,
            mock_context,
            extractor=mock_extractor,
        )

        assert result["status"] == "sent"
        assert result["sent"] is True
        mock_extractor.send_message.assert_awaited_once_with(
            "testuser", "Hello!", confirm_send=True, profile_urn=None
        )

    async def test_send_message_description_explains_connection_handoff(self):
        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool = await mcp.get_tool("send_message")
        assert tool is not None
        assert tool.description is not None
        description = " ".join(tool.description.split())
        assert (
            "If LinkedIn does not expose a normal Message action, use "
            "connect_with_person first, then retry send_message only after the "
            "connection request is accepted."
        ) in description

    async def test_send_message_schema_explains_single_line_controls(self):
        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool = await mcp.get_tool("send_message")
        assert tool is not None
        message_schema = tool.parameters["properties"]["message"]
        assert " ".join(message_schema["description"].split()) == (
            "Single-line message text to send. C0 control characters and DEL are "
            "rejected, including CR, LF, and tab."
        )

    @pytest.mark.parametrize("message", ["", "   \t\n"], ids=["empty", "whitespace"])
    async def test_send_message_refuses_blank_before_a_session(
        self, mock_context, message
    ):
        """A blank message is answered without acquiring a browser session.

        The extractor keeps the same guard, but it only runs once a session
        exists. Reaching it means `get_ready_extractor` has already had the
        chance to spend a login attempt and answer with an authentication
        error, which is not the refusal the caller can act on.
        """
        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "send_message")
        with patch(
            "linkedin_mcp_server.tools.messaging.get_ready_extractor",
            new_callable=AsyncMock,
        ) as ready:
            result = await tool_fn("testuser", message, True, mock_context)

        ready.assert_not_awaited()
        assert result["status"] == "invalid_message"
        assert result["sent"] is False
        # Nothing was submitted, so calling again cannot deliver twice.
        assert result["retry_safe"] is True
        assert result["url"] == "https://www.linkedin.com/in/testuser/"

    @pytest.mark.parametrize(
        "message",
        [f"First{chr(codepoint)}Second" for codepoint in (*range(32), 127)],
        ids=[f"U+{codepoint:04X}" for codepoint in (*range(32), 127)],
    )
    async def test_send_message_refuses_controls_before_a_session(
        self, mock_context, message
    ):
        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "send_message")
        with patch(
            "linkedin_mcp_server.tools.messaging.get_ready_extractor",
            new_callable=AsyncMock,
        ) as ready:
            result = await tool_fn("testuser", message, True, mock_context)

        ready.assert_not_awaited()
        assert result["status"] == "invalid_message"
        assert result["message"] == (
            "Message must not contain control characters or line breaks."
        )
        assert result["retry_safe"] is True

    @pytest.mark.parametrize(
        "username",
        ["", "me", "https://www.linkedin.com/company/microsoft/"],
        ids=["empty", "self-alias", "not-a-person"],
    )
    async def test_blank_message_with_an_unusable_recipient_is_mapped(
        self, mock_context, username
    ):
        """A recipient the refusal cannot name still reaches the error mapping.

        Building the refusal normalizes the recipient, so a username that
        cannot become a profile URL raises `InvalidReferenceError` from inside
        the guard. Raised past `raise_tool_error` it would arrive at the caller
        as a generic masked error (`mask_error_details=True` in `server.py`),
        which drops the one sentence that says what to correct.
        """
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.core.exceptions import InvalidReferenceError
        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "send_message")
        with (
            patch(
                "linkedin_mcp_server.tools.messaging.get_ready_extractor",
                new_callable=AsyncMock,
            ) as ready,
            pytest.raises(ToolError) as excinfo,
        ):
            await tool_fn(username, "", True, mock_context)

        ready.assert_not_awaited()
        # A ToolError is what `mask_error_details` lets through, so the
        # correction the message names reaches the caller intact.
        cause = excinfo.value.__cause__
        assert isinstance(cause, InvalidReferenceError)
        assert str(excinfo.value) == str(cause) != ""

    @pytest.mark.parametrize(
        ("result", "warns"),
        [
            ({"status": "sent", "sent": True, "retry_safe": False}, True),
            ({"status": "send_unconfirmed", "sent": False, "retry_safe": False}, True),
            ({"status": "composer_occupied", "sent": False, "retry_safe": True}, False),
        ],
        ids=["sent", "unconfirmed", "refused"],
    )
    async def test_cancelled_completion_notification_warns(
        self, mock_context, caplog, result, warns
    ):
        """The last await can discard an answer that says a message went out.

        `ctx.report_progress` is the final await inside FastMCP's
        `anyio.fail_after()`, so a deadline landing there raises
        `CancelledError` past `except Exception` and throws away the result
        the send already produced. Nothing can hand it back afterwards, and
        the log line is then the only record.

        Silent where the result says a retry is safe: nothing was submitted,
        so there is no duplicate delivery to warn about. That is `retry_safe`
        and not the status, which is why a confirmed send is parametrized
        here alongside an unconfirmed one.
        """
        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mock_extractor = _make_mock_extractor({})
        # Only shapes the extractor can actually return. A confirmed send is
        # the one that most needs the warning and the one an implementation
        # keyed on `status == "send_unconfirmed"` would silently drop.
        mock_extractor.send_message = AsyncMock(
            return_value={
                "url": "https://www.linkedin.com/messaging/compose/",
                **result,
            }
        )
        # Only the completion notification is cancelled; the one before the
        # send has to pass or the send never runs.
        mock_context.report_progress = AsyncMock(
            side_effect=[None, asyncio.CancelledError()]
        )

        mcp = FastMCP("test")
        register_messaging_tools(mcp)
        tool_fn = await get_tool_fn(mcp, "send_message")

        with (
            patch(
                "linkedin_mcp_server.tools.messaging.get_ready_extractor",
                new_callable=AsyncMock,
                return_value=mock_extractor,
            ),
            caplog.at_level(
                logging.WARNING, logger="linkedin_mcp_server.tools.messaging"
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await tool_fn("testuser", "Hello!", True, mock_context)

        mock_extractor.send_message.assert_awaited_once()
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert (SEND_INTERRUPTED_WARNING in warnings) is warns, warnings

    async def test_send_message_with_profile_urn(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/messaging/thread/abc123/",
            "status": "sent",
            "message": "Message sent.",
            "recipient_selected": True,
            "sent": True,
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "send_message")
        result = await tool_fn(
            "testuser",
            "Hello!",
            True,
            mock_context,
            profile_urn="ACoAAB1IelEB",
            extractor=mock_extractor,
        )

        assert result["status"] == "sent"
        mock_extractor.send_message.assert_awaited_once_with(
            "testuser", "Hello!", confirm_send=True, profile_urn="ACoAAB1IelEB"
        )

    async def test_send_message_error(self, mock_context):
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.exceptions import SessionExpiredError

        mock_extractor = MagicMock()
        mock_extractor.send_message = AsyncMock(side_effect=SessionExpiredError())

        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "send_message")
        with pytest.raises(ToolError, match="Session expired"):
            await tool_fn(
                "testuser",
                "Hello!",
                True,
                mock_context,
                extractor=mock_extractor,
            )

    async def test_reply_to_conversation_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/messaging/thread/2-abc/",
            "status": "sent",
            "message": "Reply submitted and confirmed in the conversation UI.",
            "recipient_selected": True,
            "sent": True,
            "retry_safe": False,
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "reply_to_conversation")
        result = await tool_fn(
            "2-abc", "Hello!", True, mock_context, extractor=mock_extractor
        )

        assert result["status"] == "sent"
        mock_extractor.reply_to_conversation.assert_awaited_once_with(
            "2-abc", "Hello!", confirm=True
        )

    async def test_reply_to_conversation_dry_run_refuses_before_the_extractor(
        self, mock_context
    ):
        """A blank message is refused browser-free, before an extractor exists."""
        mock_extractor = MagicMock()
        mock_extractor.reply_to_conversation = AsyncMock()

        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "reply_to_conversation")
        result = await tool_fn(
            "2-abc", "   ", True, mock_context, extractor=mock_extractor
        )

        assert result["status"] == "invalid_message"
        mock_extractor.reply_to_conversation.assert_not_awaited()

    async def test_reply_to_conversation_error(self, mock_context):
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.exceptions import SessionExpiredError

        mock_extractor = MagicMock()
        mock_extractor.reply_to_conversation = AsyncMock(
            side_effect=SessionExpiredError()
        )

        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "reply_to_conversation")
        with pytest.raises(ToolError, match="Session expired"):
            await tool_fn(
                "2-abc", "Hello!", True, mock_context, extractor=mock_extractor
            )

    async def test_mark_conversation_read_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/messaging/thread/2-abc/",
            "thread_id": "2-abc",
            "status": "ok",
            "changed": True,
            "read": True,
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "mark_conversation_read")
        result = await tool_fn("2-abc", True, mock_context, extractor=mock_extractor)

        assert result["status"] == "ok"
        mock_extractor.mark_conversation_read.assert_awaited_once_with(
            "2-abc", read=True, confirm=True
        )

    async def test_mark_conversation_read_unread_direction(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/messaging/thread/2-abc/",
            "thread_id": "2-abc",
            "status": "preview",
            "changed": False,
            "message": "Set confirm=true to mark this conversation as unread.",
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "mark_conversation_read")
        result = await tool_fn(
            "2-abc", False, mock_context, read=False, extractor=mock_extractor
        )

        assert result["status"] == "preview"
        mock_extractor.mark_conversation_read.assert_awaited_once_with(
            "2-abc", read=False, confirm=False
        )

    async def test_archive_conversation_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/messaging/thread/2-abc/",
            "thread_id": "2-abc",
            "status": "ok",
            "changed": True,
            "archived": True,
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "archive_conversation")
        result = await tool_fn("2-abc", True, mock_context, extractor=mock_extractor)

        assert result["status"] == "ok"
        mock_extractor.archive_conversation.assert_awaited_once_with(
            "2-abc", confirm=True, unarchive=False
        )

    async def test_unarchive_conversation_direction(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/messaging/thread/2-abc/",
            "thread_id": "2-abc",
            "status": "ok",
            "changed": True,
            "archived": False,
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "archive_conversation")
        result = await tool_fn(
            "2-abc", True, mock_context, unarchive=True, extractor=mock_extractor
        )

        assert result["archived"] is False
        mock_extractor.archive_conversation.assert_awaited_once_with(
            "2-abc", confirm=True, unarchive=True
        )

    async def test_archive_conversation_error(self, mock_context):
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.exceptions import SessionExpiredError

        mock_extractor = MagicMock()
        mock_extractor.archive_conversation = AsyncMock(
            side_effect=SessionExpiredError()
        )

        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "archive_conversation")
        with pytest.raises(ToolError, match="Session expired"):
            await tool_fn("2-abc", True, mock_context, extractor=mock_extractor)


class TestGetMyProfileTool:
    async def test_get_my_profile_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/in/johndoe/",
            "sections": {"main_profile": "John Doe\nSoftware Engineer"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_my_profile")
        result = await tool_fn(mock_context, extractor=mock_extractor)
        assert result["url"] == "https://www.linkedin.com/in/johndoe/"
        assert "main_profile" in result["sections"]
        mock_extractor.get_my_profile.assert_awaited_once()

    async def test_get_my_profile_with_sections(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/in/johndoe/",
            "sections": {"main_profile": "John Doe", "experience": "Work history"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_my_profile")
        result = await tool_fn(
            mock_context, sections="experience", extractor=mock_extractor
        )
        assert "main_profile" in result["sections"]
        assert "experience" in result["sections"]
        call_kwargs = mock_extractor.get_my_profile.call_args.kwargs
        assert "experience" in call_kwargs["sections"]

    async def test_get_my_profile_passes_callbacks(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/in/johndoe/",
            "sections": {"main_profile": "John Doe"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_my_profile")
        await tool_fn(mock_context, extractor=mock_extractor)

        call_kwargs = mock_extractor.get_my_profile.call_args.kwargs
        assert "callbacks" in call_kwargs
        assert isinstance(call_kwargs["callbacks"], MCPContextProgressCallback)

    async def test_get_my_profile_unknown_section(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/in/johndoe/",
            "sections": {"main_profile": "John Doe"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_my_profile")
        result = await tool_fn(
            mock_context, sections="bogus_section", extractor=mock_extractor
        )
        assert result["unknown_sections"] == ["bogus_section"]

    async def test_get_my_profile_error(self, mock_context):
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.exceptions import SessionExpiredError

        mock_extractor = MagicMock()
        mock_extractor.get_my_profile = AsyncMock(side_effect=SessionExpiredError())

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_my_profile")
        with pytest.raises(ToolError, match="Session expired"):
            await tool_fn(mock_context, extractor=mock_extractor)


class TestSearchCompaniesTool:
    async def test_search_companies_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/search/results/companies/?keywords=fintech",
            "sections": {"search_results": "Stripe\nFintech company\nSan Francisco"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "search_companies")
        result = await tool_fn("fintech", mock_context, extractor=mock_extractor)
        assert "search_results" in result["sections"]
        mock_extractor.search_companies.assert_awaited_once_with(
            "fintech", max_pages=1, tool_timeout=ANY
        )

    async def test_search_companies_error(self, mock_context):
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.exceptions import SessionExpiredError

        mock_extractor = MagicMock()
        mock_extractor.search_companies = AsyncMock(side_effect=SessionExpiredError())

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "search_companies")
        with pytest.raises(ToolError, match="Session expired"):
            await tool_fn("fintech", mock_context, extractor=mock_extractor)


class TestGetCompanyEmployeesTool:
    async def test_get_company_employees_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/company/anthropic/people/",
            "sections": {"employees": "Jane Doe\nResearch Engineer\nSan Francisco"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_employees")
        result = await tool_fn("anthropic", mock_context, extractor=mock_extractor)
        assert "employees" in result["sections"]
        mock_extractor.get_company_employees.assert_awaited_once_with(
            "anthropic", keywords=None
        )

    async def test_get_company_employees_with_keywords(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/company/anthropic/people/?keywords=engineer",
            "sections": {"employees": "Jane Doe\nResearch Engineer"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_employees")
        result = await tool_fn(
            "anthropic", mock_context, keywords="engineer", extractor=mock_extractor
        )
        assert "employees" in result["sections"]
        mock_extractor.get_company_employees.assert_awaited_once_with(
            "anthropic", keywords="engineer"
        )

    async def test_get_company_employees_error(self, mock_context):
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.exceptions import SessionExpiredError

        mock_extractor = MagicMock()
        mock_extractor.get_company_employees = AsyncMock(
            side_effect=SessionExpiredError()
        )

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_employees")
        with pytest.raises(ToolError, match="Session expired"):
            await tool_fn("anthropic", mock_context, extractor=mock_extractor)


class TestFeedTools:
    async def test_get_feed_success(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.extract_feed = AsyncMock(
            return_value=ExtractedSection(text="Post 1\nPost 2", references=[])
        )

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_feed")
        result = await tool_fn(mock_context, extractor=mock_extractor)
        assert result["url"] == "https://www.linkedin.com/feed/"
        assert "feed" in result["sections"]
        assert result["sections"]["feed"] == "Post 1\nPost 2"
        assert "posts" not in result

    async def test_get_feed_surfaces_references(self, mock_context):
        """References from the extractor flow through to the tool result."""
        mock_extractor = MagicMock()
        mock_extractor.extract_feed = AsyncMock(
            return_value=ExtractedSection(
                text="Some feed text",
                references=[
                    {
                        "kind": "feed_post",
                        "url": "/posts/alice_hello-ugcPost-1-xx",
                        "context": "feed",
                    },
                    {
                        "kind": "feed_post",
                        "url": "/feed/update/urn:li:activity:1234567890/",
                    },
                ],
            )
        )

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_feed")
        result = await tool_fn(mock_context, extractor=mock_extractor)
        assert "posts" not in result
        assert "feed" in result["references"]
        urls = [r["url"] for r in result["references"]["feed"]]
        assert "/posts/alice_hello-ugcPost-1-xx" in urls
        assert "/feed/update/urn:li:activity:1234567890/" in urls

    async def test_get_feed_rate_limited_surfaces_section_error(self, mock_context):
        """Rate-limit sentinel becomes a typed section_errors entry."""
        mock_extractor = MagicMock()
        mock_extractor.extract_feed = AsyncMock(
            return_value=ExtractedSection(text=RATE_LIMITED_SECTION_TEXT, references=[])
        )

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_feed")
        result = await tool_fn(mock_context, extractor=mock_extractor)
        assert "feed" not in result["sections"]
        assert result["section_errors"]["feed"]["error_type"] == "rate_limit"
        assert (
            result["section_errors"]["feed"]["error_message"]
            == RATE_LIMITED_SECTION_TEXT
        )

    async def test_get_feed_returns_section_errors(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.extract_feed = AsyncMock(
            return_value=ExtractedSection(
                text="",
                references=[],
                error={"issue_template_path": "/tmp/feed-issue.md"},
            )
        )

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_feed")
        result = await tool_fn(mock_context, extractor=mock_extractor)
        assert result["sections"] == {}
        assert "feed" in result["section_errors"]

    async def test_get_feed_rejects_zero_num_posts(self, mock_context):
        """Verify num_posts=0 is rejected by Field(ge=1) validation."""
        # FastMCP wraps the pydantic error raised by Field() constraints in
        # its own ValidationError, which does not subclass pydantic's.
        from fastmcp.exceptions import ValidationError

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        with pytest.raises(ValidationError, match="num_posts"):
            await mcp.call_tool("get_feed", {"num_posts": 0})

    async def test_get_feed_rejects_excessive_num_posts(self, mock_context):
        """Verify num_posts=51 is rejected by Field(le=50) validation."""
        # FastMCP wraps the pydantic error raised by Field() constraints in
        # its own ValidationError, which does not subclass pydantic's.
        from fastmcp.exceptions import ValidationError

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        with pytest.raises(ValidationError, match="num_posts"):
            await mcp.call_tool("get_feed", {"num_posts": 51})


class TestNotificationsTools:
    async def test_get_notifications_success(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(text="New reply\nNew reaction", references=[])
        )

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_notifications")
        result = await tool_fn(mock_context, extractor=mock_extractor)
        assert result["url"] == "https://www.linkedin.com/notifications/"
        assert result["sections"]["notifications"] == "New reply\nNew reaction"
        mock_extractor.extract_page.assert_awaited_once()
        await_args = mock_extractor.extract_page.await_args
        assert await_args is not None
        assert await_args.args[0] == "https://www.linkedin.com/notifications/"
        assert await_args.kwargs["section_name"] == "notifications"

    async def test_get_notifications_applies_filter_query(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(text="Activity on your post", references=[])
        )

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_notifications")
        result = await tool_fn(
            mock_context, filter="my_posts", extractor=mock_extractor
        )
        assert result["url"] == (
            "https://www.linkedin.com/notifications/?filterType=MY_POSTS"
        )

    async def test_get_notifications_surfaces_references(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(
                text="Ada Lovelace replied to your comment",
                references=[
                    {
                        "kind": "person",
                        "url": "/in/ada-lovelace/",
                        "text": "Ada Lovelace",
                    }
                ],
            )
        )

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_notifications")
        result = await tool_fn(mock_context, extractor=mock_extractor)
        assert result["references"]["notifications"][0]["url"] == "/in/ada-lovelace/"

    async def test_get_notifications_rate_limited_surfaces_section_error(
        self, mock_context
    ):
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(text=RATE_LIMITED_SECTION_TEXT, references=[])
        )

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_notifications")
        result = await tool_fn(mock_context, extractor=mock_extractor)
        assert result["sections"] == {}
        assert result["section_errors"]["notifications"]["error_type"] == "rate_limit"

    async def test_get_notifications_rejects_invalid_filter(self, mock_context):
        from fastmcp.exceptions import ValidationError

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        with pytest.raises(ValidationError, match="filter"):
            await mcp.call_tool("get_notifications", {"filter": "unread"})

    async def test_get_notifications_rejects_excessive_max_items(self, mock_context):
        from fastmcp.exceptions import ValidationError

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        with pytest.raises(ValidationError, match="max_items"):
            await mcp.call_tool("get_notifications", {"max_items": 51})


class TestHashtagFeedTools:
    async def test_get_hashtag_feed_success(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(text="Post 1\nPost 2", references=[])
        )

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_hashtag_feed")
        result = await tool_fn("womenintech", mock_context, extractor=mock_extractor)
        assert result["url"] == "https://www.linkedin.com/feed/hashtag/womenintech/"
        assert result["sections"]["feed_hashtag"] == "Post 1\nPost 2"
        await_args = mock_extractor.extract_page.await_args
        assert await_args is not None
        assert await_args.kwargs["section_name"] == "feed_hashtag"
        # max_posts defaults to 20 -> ceil(20/5) = 4 scrolls.
        assert await_args.kwargs["max_scrolls"] == 4

    async def test_get_hashtag_feed_normalizes_a_leading_hash_and_pasted_link(
        self, mock_context
    ):
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(text="text", references=[])
        )

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_hashtag_feed")
        result = await tool_fn("#WomenInTech", mock_context, extractor=mock_extractor)
        assert result["url"] == "https://www.linkedin.com/feed/hashtag/WomenInTech/"

    async def test_get_hashtag_feed_refuses_an_unrelated_link(self, mock_context):
        """A value that names a different page is refused before any
        navigation — see the same refusal path get_company_posts uses."""
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(text="", references=[])
        )

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_hashtag_feed")
        with pytest.raises(Exception):
            await tool_fn(
                "https://www.linkedin.com/company/microsoft/",
                mock_context,
                extractor=mock_extractor,
            )
        mock_extractor.extract_page.assert_not_called()

    async def test_get_hashtag_feed_surfaces_references(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(
                text="Some hashtag post",
                references=[
                    {
                        "kind": "feed_post",
                        "url": "/posts/alice_hello-ugcPost-1-xx",
                        "context": "feed_hashtag",
                    }
                ],
            )
        )

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_hashtag_feed")
        result = await tool_fn("womenintech", mock_context, extractor=mock_extractor)
        assert (
            result["references"]["feed_hashtag"][0]["url"]
            == "/posts/alice_hello-ugcPost-1-xx"
        )

    async def test_get_hashtag_feed_rate_limited_surfaces_section_error(
        self, mock_context
    ):
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(text=RATE_LIMITED_SECTION_TEXT, references=[])
        )

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_hashtag_feed")
        result = await tool_fn("womenintech", mock_context, extractor=mock_extractor)
        assert result["sections"] == {}
        assert result["section_errors"]["feed_hashtag"]["error_type"] == "rate_limit"

    async def test_get_hashtag_feed_rejects_excessive_max_posts(self, mock_context):
        from fastmcp.exceptions import ValidationError

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        with pytest.raises(ValidationError, match="max_posts"):
            await mcp.call_tool(
                "get_hashtag_feed", {"hashtag": "womenintech", "max_posts": 51}
            )


class TestSavedPostsTools:
    async def test_get_saved_posts_success(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(
                text="Saved post 1\nSaved post 2", references=[]
            )
        )

        from linkedin_mcp_server.tools.saved_posts import register_saved_posts_tools

        mcp = FastMCP("test")
        register_saved_posts_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_saved_posts")
        result = await tool_fn(mock_context, extractor=mock_extractor)
        assert result["url"] == "https://www.linkedin.com/my-items/saved-posts/"
        assert result["sections"]["saved_posts"] == "Saved post 1\nSaved post 2"
        mock_extractor.extract_page.assert_awaited_once()
        await_args = mock_extractor.extract_page.await_args
        assert await_args is not None
        assert await_args.kwargs["section_name"] == "saved_posts"
        # max_posts defaults to 20 -> ceil(20/5) = 4 scrolls.
        assert await_args.kwargs["max_scrolls"] == 4

    async def test_get_saved_posts_scroll_budget_scales_with_max_posts(
        self, mock_context
    ):
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(text="text", references=[])
        )

        from linkedin_mcp_server.tools.saved_posts import register_saved_posts_tools

        mcp = FastMCP("test")
        register_saved_posts_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_saved_posts")
        await tool_fn(mock_context, max_posts=1, extractor=mock_extractor)
        await_args = mock_extractor.extract_page.await_args
        assert await_args is not None
        assert await_args.kwargs["max_scrolls"] == 1

    async def test_get_saved_posts_rate_limited_surfaces_section_error(
        self, mock_context
    ):
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(text=RATE_LIMITED_SECTION_TEXT, references=[])
        )

        from linkedin_mcp_server.tools.saved_posts import register_saved_posts_tools

        mcp = FastMCP("test")
        register_saved_posts_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_saved_posts")
        result = await tool_fn(mock_context, extractor=mock_extractor)
        assert result["sections"] == {}
        assert result["section_errors"]["saved_posts"]["error_type"] == "rate_limit"

    async def test_get_saved_posts_returns_section_errors(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(
                text="",
                references=[],
                error={"issue_template_path": "/tmp/saved-posts-issue.md"},
            )
        )

        from linkedin_mcp_server.tools.saved_posts import register_saved_posts_tools

        mcp = FastMCP("test")
        register_saved_posts_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_saved_posts")
        result = await tool_fn(mock_context, extractor=mock_extractor)
        assert result["sections"] == {}
        assert result["section_errors"]["saved_posts"]["issue_template_path"] == (
            "/tmp/saved-posts-issue.md"
        )

    async def test_get_saved_posts_surfaces_references(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(
                text="Some saved post",
                references=[
                    {
                        "kind": "feed_post",
                        "url": "/posts/alice_hello-ugcPost-1-xx",
                        "context": "saved post",
                    }
                ],
            )
        )

        from linkedin_mcp_server.tools.saved_posts import register_saved_posts_tools

        mcp = FastMCP("test")
        register_saved_posts_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_saved_posts")
        result = await tool_fn(mock_context, extractor=mock_extractor)
        assert (
            result["references"]["saved_posts"][0]["url"]
            == "/posts/alice_hello-ugcPost-1-xx"
        )

    async def test_get_saved_posts_rejects_zero_max_posts(self, mock_context):
        from fastmcp.exceptions import ValidationError

        from linkedin_mcp_server.tools.saved_posts import register_saved_posts_tools

        mcp = FastMCP("test")
        register_saved_posts_tools(mcp)

        with pytest.raises(ValidationError, match="max_posts"):
            await mcp.call_tool("get_saved_posts", {"max_posts": 0})

    async def test_get_saved_posts_rejects_excessive_max_posts(self, mock_context):
        from fastmcp.exceptions import ValidationError

        from linkedin_mcp_server.tools.saved_posts import register_saved_posts_tools

        mcp = FastMCP("test")
        register_saved_posts_tools(mcp)

        with pytest.raises(ValidationError, match="max_posts"):
            await mcp.call_tool("get_saved_posts", {"max_posts": 51})

    async def test_save_post_success(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.save_post = AsyncMock(
            return_value={
                "url": "https://www.linkedin.com/posts/alice_x-ugcPost-1-xx/",
                "status": "saved",
                "message": "Post saved.",
                "saved": True,
                "retry_safe": False,
            }
        )

        from linkedin_mcp_server.tools.saved_posts import register_saved_posts_tools

        mcp = FastMCP("test")
        register_saved_posts_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "save_post")
        result = await tool_fn(
            "https://www.linkedin.com/posts/alice_x-ugcPost-1-xx/",
            True,
            mock_context,
            extractor=mock_extractor,
        )
        assert result["status"] == "saved"
        mock_extractor.save_post.assert_awaited_once_with(
            "https://www.linkedin.com/posts/alice_x-ugcPost-1-xx/",
            confirm=True,
            unsave=False,
        )

    async def test_save_post_forwards_unsave(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.save_post = AsyncMock(
            return_value={
                "url": "https://www.linkedin.com/posts/alice_x-ugcPost-1-xx/",
                "status": "unsaved",
                "message": "Post unsaved.",
                "saved": False,
                "retry_safe": False,
            }
        )

        from linkedin_mcp_server.tools.saved_posts import register_saved_posts_tools

        mcp = FastMCP("test")
        register_saved_posts_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "save_post")
        await tool_fn(
            "https://www.linkedin.com/posts/alice_x-ugcPost-1-xx/",
            True,
            mock_context,
            unsave=True,
            extractor=mock_extractor,
        )
        mock_extractor.save_post.assert_awaited_once_with(
            "https://www.linkedin.com/posts/alice_x-ugcPost-1-xx/",
            confirm=True,
            unsave=True,
        )

    async def test_save_post_requires_confirm_argument(self, mock_context):
        from fastmcp.exceptions import ValidationError

        from linkedin_mcp_server.tools.saved_posts import register_saved_posts_tools

        mcp = FastMCP("test")
        register_saved_posts_tools(mcp)

        with pytest.raises(ValidationError):
            await mcp.call_tool(
                "save_post",
                {"post_url": "https://www.linkedin.com/posts/alice_x-ugcPost-1-xx/"},
            )


class TestPostReactionsTools:
    async def test_get_post_reactions_success(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.get_post_reactions = AsyncMock(
            return_value={
                "url": "https://www.linkedin.com/posts/alice_x-ugcPost-1-xx/",
                "sections": {"reactions": "Bob Smith"},
                "references": {
                    "reactions": [
                        {
                            "kind": "person",
                            "url": "/in/bob-smith/",
                            "context": "reactor",
                        }
                    ]
                },
            }
        )

        from linkedin_mcp_server.tools.reactions import register_reaction_tools

        mcp = FastMCP("test")
        register_reaction_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_post_reactions")
        result = await tool_fn(
            "https://www.linkedin.com/posts/alice_x-ugcPost-1-xx/",
            mock_context,
            extractor=mock_extractor,
        )
        assert result["sections"]["reactions"] == "Bob Smith"
        assert result["references"]["reactions"][0]["url"] == "/in/bob-smith/"
        mock_extractor.get_post_reactions.assert_awaited_once_with(
            "https://www.linkedin.com/posts/alice_x-ugcPost-1-xx/", max_reactors=50
        )

    async def test_get_post_reactions_surfaces_structural_section_error(
        self, mock_context
    ):
        mock_extractor = MagicMock()
        mock_extractor.get_post_reactions = AsyncMock(
            return_value={
                "url": "https://www.linkedin.com/posts/alice_x-ugcPost-1-xx/",
                "sections": {},
                "section_errors": {
                    "reactions": {
                        "error_type": "structural_signal_not_found",
                        "error_message": "Could not identify the reactions "
                        "dialog structurally.",
                    }
                },
            }
        )

        from linkedin_mcp_server.tools.reactions import register_reaction_tools

        mcp = FastMCP("test")
        register_reaction_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_post_reactions")
        result = await tool_fn(
            "https://www.linkedin.com/posts/alice_x-ugcPost-1-xx/",
            mock_context,
            extractor=mock_extractor,
        )
        assert result["sections"] == {}
        assert (
            result["section_errors"]["reactions"]["error_type"]
            == "structural_signal_not_found"
        )

    async def test_get_post_reactions_rejects_excessive_max_reactors(
        self, mock_context
    ):
        from fastmcp.exceptions import ValidationError

        from linkedin_mcp_server.tools.reactions import register_reaction_tools

        mcp = FastMCP("test")
        register_reaction_tools(mcp)

        with pytest.raises(ValidationError, match="max_reactors"):
            await mcp.call_tool(
                "get_post_reactions",
                {
                    "post_url": "https://www.linkedin.com/posts/alice_x-ugcPost-1-xx/",
                    "max_reactors": 51,
                },
            )


class TestPostTools:
    async def test_search_posts_success(self, mock_context):
        expected = {
            "url": (
                "https://www.linkedin.com/search/results/content/"
                "?keywords=Buscamos+Unity&origin=FACETED_SEARCH"
            ),
            "sections": {"search_results": "Acme is hiring a Unity dev!"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.post import register_post_tools

        mcp = FastMCP("test")
        register_post_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "search_posts")
        result = await tool_fn(
            "Buscamos Unity",
            mock_context,
            date_posted="past-week",
            extractor=mock_extractor,
        )
        assert "search_results" in result["sections"]
        mock_extractor.search_posts.assert_awaited_once_with(
            "Buscamos Unity",
            date_posted="past-week",
            max_pages=3,
            sort_by=None,
        )

    async def test_search_posts_validation_error_surfaced_as_tool_error(
        self, mock_context
    ):
        """A FilterValidationError from the extractor surfaces to the client as
        a ToolError carrying the same message, not the generic mask."""
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.scraping.contracts import FilterValidationError
        from linkedin_mcp_server.tools.post import register_post_tools

        mock_extractor = MagicMock()
        mock_extractor.search_posts = AsyncMock(
            side_effect=FilterValidationError("Invalid date_posted 'last-year'")
        )

        mcp = FastMCP("test")
        register_post_tools(mcp)
        tool_fn = await get_tool_fn(mcp, "search_posts")

        with pytest.raises(ToolError, match="Invalid date_posted"):
            await tool_fn(
                "python",
                mock_context,
                date_posted="last-year",
                extractor=mock_extractor,
            )

    async def test_search_posts_rejects_zero_max_pages(self, mock_context):
        """Verify max_pages=0 is rejected by Field(ge=1) validation."""
        # FastMCP wraps the pydantic error raised by Field() constraints in
        # its own ValidationError, which does not subclass pydantic's.
        from fastmcp.exceptions import ValidationError

        from linkedin_mcp_server.tools.post import register_post_tools

        mcp = FastMCP("test")
        register_post_tools(mcp)

        with pytest.raises(ValidationError, match="max_pages"):
            await mcp.call_tool("search_posts", {"keywords": "python", "max_pages": 0})


class TestNetworkTools:
    async def test_get_mutual_connections_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/search/results/people/?connectionOf=%22X%22",
            "sections": {"mutual_connections": "Jane Doe"},
            "stopped_reason": "end_of_results",
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.network import register_network_tools

        mcp = FastMCP("test")
        register_network_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_mutual_connections")
        result = await tool_fn("stickerdaniel", mock_context, extractor=mock_extractor)

        assert result["sections"]["mutual_connections"] == "Jane Doe"
        mock_extractor.get_mutual_connections.assert_awaited_once_with(
            "stickerdaniel", max_results=50
        )

    async def test_list_connections_forwards_sort_and_max_results(self, mock_context):
        expected = {"url": "...", "sections": {"connections": "Alice\nBob"}}
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.network import register_network_tools

        mcp = FastMCP("test")
        register_network_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "list_connections")
        result = await tool_fn(
            mock_context, max_results=25, sort="first_name", extractor=mock_extractor
        )

        assert result["sections"]["connections"] == "Alice\nBob"
        mock_extractor.list_connections.assert_awaited_once_with(
            max_results=25, sort="first_name"
        )

    async def test_get_invitations_defaults_to_received(self, mock_context):
        expected = {"url": "...", "sections": {"received_invitations": "Jane"}}
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.network import register_network_tools

        mcp = FastMCP("test")
        register_network_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_invitations")
        result = await tool_fn(mock_context, extractor=mock_extractor)

        assert result["sections"]["received_invitations"] == "Jane"
        mock_extractor.get_invitations.assert_awaited_once_with(
            direction="received", max_results=50
        )

    async def test_get_invitations_sent_direction(self, mock_context):
        expected = {"url": "...", "sections": {"sent_invitations": "Pending"}}
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.network import register_network_tools

        mcp = FastMCP("test")
        register_network_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_invitations")
        await tool_fn(mock_context, direction="sent", extractor=mock_extractor)

        mock_extractor.get_invitations.assert_awaited_once_with(
            direction="sent", max_results=50
        )

    async def test_withdraw_invitation_forwards_confirm(self, mock_context):
        expected = {"url": "...", "status": "withdrawn", "message": "..."}
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.network import register_network_tools

        mcp = FastMCP("test")
        register_network_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "withdraw_invitation")
        result = await tool_fn(
            "stickerdaniel", True, mock_context, extractor=mock_extractor
        )

        assert result["status"] == "withdrawn"
        mock_extractor.withdraw_invitation.assert_awaited_once_with(
            "stickerdaniel", confirm=True
        )

    async def test_withdraw_invitation_preview_never_touches_the_extractor(
        self, mock_context
    ):
        mock_extractor = _make_mock_extractor({})

        from linkedin_mcp_server.tools.network import register_network_tools

        mcp = FastMCP("test")
        register_network_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "withdraw_invitation")
        result = await tool_fn(
            "stickerdaniel", False, mock_context, extractor=mock_extractor
        )

        assert result["status"] == "preview"
        assert result["url"] == "https://www.linkedin.com/in/stickerdaniel/"
        mock_extractor.withdraw_invitation.assert_not_awaited()

    async def test_respond_to_invitation_forwards_action_and_confirm(
        self, mock_context
    ):
        expected = {"url": "...", "status": "accepted", "message": "..."}
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.network import register_network_tools

        mcp = FastMCP("test")
        register_network_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "respond_to_invitation")
        result = await tool_fn(
            "stickerdaniel", "accept", True, mock_context, extractor=mock_extractor
        )

        assert result["status"] == "accepted"
        mock_extractor.respond_to_invitation.assert_awaited_once_with(
            "stickerdaniel", action="accept", confirm=True
        )

    async def test_follow_forwards_unfollow_and_confirm(self, mock_context):
        expected = {"url": "...", "status": "toggled", "requested": "unfollow"}
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.network import register_network_tools

        mcp = FastMCP("test")
        register_network_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "follow")
        result = await tool_fn(
            "https://www.linkedin.com/company/anthropic/",
            True,
            mock_context,
            unfollow=True,
            extractor=mock_extractor,
        )

        assert result["status"] == "toggled"
        mock_extractor.follow.assert_awaited_once_with(
            "https://www.linkedin.com/company/anthropic/",
            confirm=True,
            unfollow=True,
        )

    async def test_network_tool_error_is_mapped(self, mock_context):
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.exceptions import SessionExpiredError

        mock_extractor = MagicMock()
        mock_extractor.get_mutual_connections = AsyncMock(
            side_effect=SessionExpiredError()
        )

        from linkedin_mcp_server.tools.network import register_network_tools

        mcp = FastMCP("test")
        register_network_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_mutual_connections")
        with pytest.raises(ToolError, match="Session expired"):
            await tool_fn("stickerdaniel", mock_context, extractor=mock_extractor)


class TestGroupTools:
    async def test_search_groups_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/search/results/groups/?keywords=rpa",
            "sections": {"search_results": "RPA Group"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.group import register_group_tools

        mcp = FastMCP("test")
        register_group_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "search_groups")
        result = await tool_fn("rpa", mock_context, extractor=mock_extractor)

        assert result["sections"]["search_results"] == "RPA Group"
        mock_extractor.search_groups.assert_awaited_once_with("rpa")

    async def test_get_group_posts_forwards_max_posts(self, mock_context):
        expected = {"url": "...", "sections": {"posts": "Post 1"}}
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.group import register_group_tools

        mcp = FastMCP("test")
        register_group_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_group_posts")
        result = await tool_fn(
            "12345", mock_context, max_posts=10, extractor=mock_extractor
        )

        assert result["sections"]["posts"] == "Post 1"
        mock_extractor.get_group_posts.assert_awaited_once_with("12345", max_posts=10)

    async def test_get_group_members_forwards_keywords(self, mock_context):
        expected = {"url": "...", "sections": {"members": "Jane Doe"}}
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.group import register_group_tools

        mcp = FastMCP("test")
        register_group_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_group_members")
        result = await tool_fn(
            "12345",
            mock_context,
            max_members=100,
            keywords="jane",
            extractor=mock_extractor,
        )

        assert result["sections"]["members"] == "Jane Doe"
        mock_extractor.get_group_members.assert_awaited_once_with(
            "12345", max_members=100, keywords="jane"
        )

    async def test_get_group_members_error(self, mock_context):
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.exceptions import SessionExpiredError

        mock_extractor = MagicMock()
        mock_extractor.get_group_members = AsyncMock(side_effect=SessionExpiredError())

        from linkedin_mcp_server.tools.group import register_group_tools

        mcp = FastMCP("test")
        register_group_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_group_members")
        with pytest.raises(ToolError, match="Session expired"):
            await tool_fn("12345", mock_context, extractor=mock_extractor)


class TestEventTools:
    async def test_search_events_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/search/results/events/?keywords=rpa",
            "sections": {"search_results": "RPA Summit 2026"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.event import register_event_tools

        mcp = FastMCP("test")
        register_event_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "search_events")
        result = await tool_fn("rpa", mock_context, extractor=mock_extractor)

        assert result["sections"]["search_results"] == "RPA Summit 2026"
        mock_extractor.search_events.assert_awaited_once_with("rpa", max_pages=3)

    async def test_get_event_details_forwards_event_url(self, mock_context):
        expected = {"url": "...", "sections": {"event_details": "RPA Summit"}}
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.event import register_event_tools

        mcp = FastMCP("test")
        register_event_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_event_details")
        result = await tool_fn("1234567890", mock_context, extractor=mock_extractor)

        assert result["sections"]["event_details"] == "RPA Summit"
        mock_extractor.get_event_details.assert_awaited_once_with("1234567890")

    async def test_get_event_attendees_forwards_max_attendees(self, mock_context):
        expected = {"url": "...", "sections": {"attendees": "Jane Doe"}}
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.event import register_event_tools

        mcp = FastMCP("test")
        register_event_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_event_attendees")
        result = await tool_fn(
            "1234567890", mock_context, max_attendees=100, extractor=mock_extractor
        )

        assert result["sections"]["attendees"] == "Jane Doe"
        mock_extractor.get_event_attendees.assert_awaited_once_with(
            "1234567890", max_attendees=100
        )

    async def test_get_event_attendees_error(self, mock_context):
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.exceptions import SessionExpiredError

        mock_extractor = MagicMock()
        mock_extractor.get_event_attendees = AsyncMock(
            side_effect=SessionExpiredError()
        )

        from linkedin_mcp_server.tools.event import register_event_tools

        mcp = FastMCP("test")
        register_event_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_event_attendees")
        with pytest.raises(ToolError, match="Session expired"):
            await tool_fn("1234567890", mock_context, extractor=mock_extractor)


class TestToolTimeouts:
    async def test_all_tools_have_global_timeout(self):
        from linkedin_mcp_server.server import create_mcp_server

        custom_timeout = 7.5
        mcp = create_mcp_server(tool_timeout=custom_timeout)

        tool_names = (
            "get_person_profile",
            "connect_with_person",
            "get_sidebar_profiles",
            "search_people",
            "get_company_profile",
            "get_company_posts",
            "get_job_details",
            "search_jobs",
            "get_saved_jobs",
            "get_inbox",
            "get_conversation",
            "search_conversations",
            "send_message",
            "get_feed",
            "search_posts",
            "get_post_comments",
            "reply_to_comment",
            "comment_on_post",
            "react_to_comment",
            "get_mutual_connections",
            "list_connections",
            "get_invitations",
            "withdraw_invitation",
            "respond_to_invitation",
            "follow",
            "search_groups",
            "get_group_posts",
            "get_group_members",
            "search_events",
            "get_event_details",
            "get_event_attendees",
            "get_post_analytics",
            "get_profile_analytics",
            "close_session",
            "get_pacing_status",
        )

        for name in tool_names:
            tool = await mcp.get_tool(name)
            assert tool is not None
            assert tool.timeout == custom_timeout

    async def test_all_tools_have_default_timeout(self):
        from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
        from linkedin_mcp_server.server import create_mcp_server

        mcp = create_mcp_server()

        tool_names = (
            "get_person_profile",
            "get_my_profile",
            "connect_with_person",
            "get_sidebar_profiles",
            "search_people",
            "get_company_profile",
            "get_company_posts",
            "search_companies",
            "get_company_employees",
            "get_job_details",
            "search_jobs",
            "get_saved_jobs",
            "get_inbox",
            "get_conversation",
            "search_conversations",
            "send_message",
            "get_feed",
            "search_posts",
            "get_post_comments",
            "reply_to_comment",
            "comment_on_post",
            "react_to_comment",
            "get_mutual_connections",
            "list_connections",
            "get_invitations",
            "withdraw_invitation",
            "respond_to_invitation",
            "follow",
            "search_groups",
            "get_group_posts",
            "get_group_members",
            "search_events",
            "get_event_details",
            "get_event_attendees",
            "get_post_analytics",
            "get_profile_analytics",
            "close_session",
            "get_pacing_status",
        )

        for name in tool_names:
            tool = await mcp.get_tool(name)
            assert tool is not None
            assert tool.timeout == DEFAULT_TOOL_TIMEOUT_SECONDS
