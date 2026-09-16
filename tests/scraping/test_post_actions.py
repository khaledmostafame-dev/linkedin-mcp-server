"""Tests for the post save/unsave toggle owner.

Every ambiguous-match case proves nothing was clicked (page.evaluate call
count and mock_page.wait_for_selector are the load-bearing assertions, not
just the returned status), matching the fail-closed contract in
linkedin_mcp_server/scraping/post_actions.py. page.evaluate is a mock, so
the real JS never executes and each step's outcome is supplied directly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.post_actions import PostActions
from linkedin_mcp_server.scraping.session import ScrapingSession

POST_URL = "https://www.linkedin.com/posts/alice_hello-ugcPost-1-xx/"


def _actions(page) -> PostActions:
    session = ScrapingSession(page)
    return PostActions(session, PageNavigator(session))


async def _with_rate_limit_patch(coro):
    with patch(
        "linkedin_mcp_server.scraping.session.detect_rate_limit",
        new_callable=AsyncMock,
    ):
        return await coro


async def test_ambiguous_menu_opener_fails_closed(mock_page):
    mock_page.evaluate = AsyncMock(return_value=False)
    actions = _actions(mock_page)

    result = await _with_rate_limit_patch(actions.save_post(POST_URL, confirm=True))

    assert result["status"] == "structural_signal_not_found"
    assert result["saved"] is None
    assert result["retry_safe"] is True
    mock_page.wait_for_selector.assert_not_called()


async def test_menu_never_opening_fails_closed(mock_page):
    mock_page.evaluate = AsyncMock(return_value=True)
    mock_page.wait_for_selector = AsyncMock(
        side_effect=PlaywrightTimeoutError("no menu")
    )
    actions = _actions(mock_page)

    result = await _with_rate_limit_patch(actions.save_post(POST_URL, confirm=True))

    assert result["status"] == "structural_signal_not_found"


async def test_ambiguous_toggle_fails_closed(mock_page):
    mock_page.wait_for_selector = AsyncMock()
    calls: list[str] = []

    async def evaluate(script, *args, **kwargs):
        calls.append(script)
        if "openers" in script:
            return True
        if "toggles[0].getAttribute" in script:
            return None  # not exactly one aria-pressed candidate
        raise AssertionError(f"unexpected evaluate call: {script[:60]}")

    mock_page.evaluate = AsyncMock(side_effect=evaluate)
    actions = _actions(mock_page)

    result = await _with_rate_limit_patch(actions.save_post(POST_URL, confirm=True))

    assert result["status"] == "structural_signal_not_found"
    # The click program was never reached.
    assert not any("toggles[0].click" in call for call in calls)


async def test_already_in_desired_state_clicks_nothing(mock_page):
    mock_page.wait_for_selector = AsyncMock()

    async def evaluate(script, *args, **kwargs):
        if "openers" in script:
            return True
        if "toggles[0].getAttribute" in script:
            return True  # already saved
        raise AssertionError(f"unexpected evaluate call: {script[:60]}")

    mock_page.evaluate = AsyncMock(side_effect=evaluate)
    actions = _actions(mock_page)

    result = await _with_rate_limit_patch(
        actions.save_post(POST_URL, confirm=True, unsave=False)
    )

    assert result["status"] == "already_in_desired_state"
    assert result["saved"] is True
    assert result["retry_safe"] is True


async def test_confirm_false_previews_without_clicking(mock_page):
    mock_page.wait_for_selector = AsyncMock()

    async def evaluate(script, *args, **kwargs):
        if "openers" in script:
            return True
        if "toggles[0].getAttribute" in script:
            return False  # not yet saved
        raise AssertionError(f"unexpected evaluate call: {script[:60]}")

    mock_page.evaluate = AsyncMock(side_effect=evaluate)
    actions = _actions(mock_page)

    result = await _with_rate_limit_patch(
        actions.save_post(POST_URL, confirm=False, unsave=False)
    )

    assert result["status"] == "confirmation_required"
    assert result["saved"] is False
    assert result["retry_safe"] is True


async def test_confirm_true_clicks_and_confirms_new_state(mock_page):
    mock_page.wait_for_selector = AsyncMock()
    read_count = 0

    async def evaluate(script, *args, **kwargs):
        nonlocal read_count
        if "openers" in script:
            return True
        if "toggles[0].click" in script:
            return True
        if "toggles[0].getAttribute" in script:
            read_count += 1
            return False if read_count == 1 else True
        raise AssertionError(f"unexpected evaluate call: {script[:60]}")

    mock_page.evaluate = AsyncMock(side_effect=evaluate)
    actions = _actions(mock_page)

    result = await _with_rate_limit_patch(
        actions.save_post(POST_URL, confirm=True, unsave=False)
    )

    assert result["status"] == "saved"
    assert result["saved"] is True
    assert result["retry_safe"] is False


async def test_state_unconfirmed_after_click_is_not_retry_safe(mock_page):
    mock_page.wait_for_selector = AsyncMock()

    async def evaluate(script, *args, **kwargs):
        if "openers" in script:
            return True
        if "toggles[0].click" in script:
            return True
        if "toggles[0].getAttribute" in script:
            return False  # never flips to the desired True
        raise AssertionError(f"unexpected evaluate call: {script[:60]}")

    mock_page.evaluate = AsyncMock(side_effect=evaluate)
    actions = _actions(mock_page)

    result = await _with_rate_limit_patch(
        actions.save_post(POST_URL, confirm=True, unsave=False)
    )

    assert result["status"] == "state_unconfirmed"
    assert result["retry_safe"] is False
