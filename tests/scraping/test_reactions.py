"""Tests for the post-reactions dialog reader.

Every case here proves the fail-closed contract: an ambiguous structural
match (or a dialog that never appears) reports a section_errors entry and
touches nothing further, matching the reasoning in
linkedin_mcp_server/scraping/reactions.py. page.evaluate is a mock, so the
real JS never executes and each step's outcome is supplied directly;
tests/test_post_actions_dom.py runs the control probe itself.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.reactions import (
    _FIND_SOCIAL_COUNTS_CONTROL_JS,
    ReactionsReader,
)
from linkedin_mcp_server.scraping.session import ScrapingSession

POST_URL = "https://www.linkedin.com/posts/alice_hello-ugcPost-1-xx/"


def _reader(page) -> ReactionsReader:
    session = ScrapingSession(page)
    return ReactionsReader(session, PageNavigator(session), PageContentReader(session))


async def test_ambiguous_social_counts_control_reports_section_error(mock_page):
    mock_page.evaluate = AsyncMock(return_value=False)
    reader = _reader(mock_page)

    with (
        patch(
            "linkedin_mcp_server.scraping.session.detect_rate_limit",
            new_callable=AsyncMock,
        ),
    ):
        result = await reader.get_post_reactions(POST_URL)

    assert result["sections"] == {}
    assert (
        result["section_errors"]["reactions"]["error_type"]
        == "structural_signal_not_found"
    )
    mock_page.wait_for_selector.assert_not_called()


async def test_dialog_never_appearing_reports_section_error(mock_page):
    mock_page.evaluate = AsyncMock(return_value=True)
    mock_page.wait_for_selector = AsyncMock(
        side_effect=PlaywrightTimeoutError("no dialog")
    )
    reader = _reader(mock_page)

    with patch(
        "linkedin_mcp_server.scraping.session.detect_rate_limit",
        new_callable=AsyncMock,
    ):
        result = await reader.get_post_reactions(POST_URL)

    assert result["sections"] == {}
    assert (
        result["section_errors"]["reactions"]["error_type"]
        == "structural_signal_not_found"
    )


async def test_successful_flow_returns_reactor_references(mock_page):
    mock_page.wait_for_selector = AsyncMock()

    async def evaluate(script, *args, **kwargs):
        if script == _FIND_SOCIAL_COUNTS_CONTROL_JS:
            return True
        if "scrollable.scrollTop" in script:
            return False  # stop scrolling immediately
        # The generic root-content extraction (MAX_HEADING_CONTAINERS marks
        # it, same as tests/scraping/test_content.py).
        return {
            "source": "root",
            "text": "Bob Smith\nCarol Jones",
            "references": [
                {"href": "https://www.linkedin.com/in/bob-smith/", "text": "Bob Smith"},
            ],
            "images": [],
        }

    mock_page.evaluate = AsyncMock(side_effect=evaluate)
    reader = _reader(mock_page)

    with patch(
        "linkedin_mcp_server.scraping.session.detect_rate_limit",
        new_callable=AsyncMock,
    ):
        result = await reader.get_post_reactions(POST_URL, max_reactors=10)

    assert "section_errors" not in result
    assert result["sections"]["reactions"] == "Bob Smith\nCarol Jones"
    assert result["references"]["reactions"] == [
        {
            "kind": "person",
            "url": "/in/bob-smith/",
            "text": "Bob Smith",
            "context": "reactor",
        }
    ]
