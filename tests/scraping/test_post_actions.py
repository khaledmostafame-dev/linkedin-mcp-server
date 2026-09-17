"""Tests for the post save/unsave toggle owner.

Every ambiguous-match case proves nothing was clicked (the programs that
were evaluated and mock_page.wait_for_function are the load-bearing
assertions, not just the returned status), matching the fail-closed
contract in linkedin_mcp_server/scraping/post_actions.py. page.evaluate is
a mock, so the real JS never executes and each step's outcome is supplied
directly; tests/test_post_actions_dom.py runs the programs themselves.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.post_actions import (
    _CLICK_SAVE_TOGGLE_JS,
    _OPEN_POST_OVERFLOW_MENU_JS,
    _READ_SAVE_TOGGLE_STATE_JS,
    PostActions,
)
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


_PROGRAMS = {
    _OPEN_POST_OVERFLOW_MENU_JS,
    _READ_SAVE_TOGGLE_STATE_JS,
    _CLICK_SAVE_TOGGLE_JS,
}


def _evaluated(page) -> list[str]:
    """This module's programs in call order (navigation evaluates others)."""
    calls = [call.args[0] for call in page.evaluate.await_args_list]
    return [script for script in calls if script in _PROGRAMS]


async def test_ambiguous_menu_opener_fails_closed(mock_page):
    mock_page.evaluate = AsyncMock(return_value=False)
    actions = _actions(mock_page)

    result = await _with_rate_limit_patch(actions.save_post(POST_URL, confirm=True))

    assert result["status"] == "structural_signal_not_found"
    assert result["saved"] is None
    assert result["retry_safe"] is True
    mock_page.wait_for_function.assert_not_called()


async def test_menu_never_opening_fails_closed(mock_page):
    mock_page.evaluate = AsyncMock(return_value=True)
    mock_page.wait_for_function = AsyncMock(
        side_effect=PlaywrightTimeoutError("no menu")
    )
    actions = _actions(mock_page)

    result = await _with_rate_limit_patch(actions.save_post(POST_URL, confirm=True))

    assert result["status"] == "structural_signal_not_found"
    # Only the opener program ran; nothing read or clicked the save item.
    assert _evaluated(mock_page) == [_OPEN_POST_OVERFLOW_MENU_JS]


async def test_ambiguous_toggle_fails_closed(mock_page):
    async def evaluate(script, *args, **kwargs):
        if script == _OPEN_POST_OVERFLOW_MENU_JS:
            return True
        if script == _READ_SAVE_TOGGLE_STATE_JS:
            return None  # not exactly one bookmark item
        raise AssertionError(f"unexpected evaluate call: {script[:60]}")

    mock_page.evaluate = AsyncMock(side_effect=evaluate)
    actions = _actions(mock_page)

    result = await _with_rate_limit_patch(actions.save_post(POST_URL, confirm=True))

    assert result["status"] == "structural_signal_not_found"
    # The click program was never reached.
    assert _CLICK_SAVE_TOGGLE_JS not in _evaluated(mock_page)


async def test_already_in_desired_state_clicks_nothing(mock_page):
    async def evaluate(script, *args, **kwargs):
        if script == _OPEN_POST_OVERFLOW_MENU_JS:
            return True
        if script == _READ_SAVE_TOGGLE_STATE_JS:
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
    async def evaluate(script, *args, **kwargs):
        if script == _OPEN_POST_OVERFLOW_MENU_JS:
            return True
        if script == _READ_SAVE_TOGGLE_STATE_JS:
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
    read_count = 0

    async def evaluate(script, *args, **kwargs):
        nonlocal read_count
        if script == _OPEN_POST_OVERFLOW_MENU_JS:
            return True
        if script == _CLICK_SAVE_TOGGLE_JS:
            return True
        if script == _READ_SAVE_TOGGLE_STATE_JS:
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


async def test_menu_closed_by_the_click_is_reopened_to_confirm(mock_page):
    reads = iter([False, None, True])  # before, menu closed, after reopening

    async def evaluate(script, *args, **kwargs):
        if script == _OPEN_POST_OVERFLOW_MENU_JS:
            return True
        if script == _CLICK_SAVE_TOGGLE_JS:
            return True
        if script == _READ_SAVE_TOGGLE_STATE_JS:
            return next(reads)
        raise AssertionError(f"unexpected evaluate call: {script[:60]}")

    mock_page.evaluate = AsyncMock(side_effect=evaluate)
    actions = _actions(mock_page)

    result = await _with_rate_limit_patch(
        actions.save_post(POST_URL, confirm=True, unsave=False)
    )

    assert (result["status"], result["saved"]) == ("saved", True)
    assert _evaluated(mock_page) == [
        _OPEN_POST_OVERFLOW_MENU_JS,
        _READ_SAVE_TOGGLE_STATE_JS,
        _CLICK_SAVE_TOGGLE_JS,
        _READ_SAVE_TOGGLE_STATE_JS,
        _OPEN_POST_OVERFLOW_MENU_JS,
        _READ_SAVE_TOGGLE_STATE_JS,
    ]
    assert mock_page.wait_for_function.await_count == 2


async def test_state_unconfirmed_after_click_is_not_retry_safe(mock_page):
    async def evaluate(script, *args, **kwargs):
        if script == _OPEN_POST_OVERFLOW_MENU_JS:
            return True
        if script == _CLICK_SAVE_TOGGLE_JS:
            return True
        if script == _READ_SAVE_TOGGLE_STATE_JS:
            return False  # never flips to the desired True
        raise AssertionError(f"unexpected evaluate call: {script[:60]}")

    mock_page.evaluate = AsyncMock(side_effect=evaluate)
    actions = _actions(mock_page)

    result = await _with_rate_limit_patch(
        actions.save_post(POST_URL, confirm=True, unsave=False)
    )

    assert result["status"] == "state_unconfirmed"
    assert result["retry_safe"] is False


async def test_menu_that_cannot_be_reopened_leaves_the_state_unconfirmed(mock_page):
    reads = iter([False, None])  # before; after the click the menu is gone

    async def evaluate(script, *args, **kwargs):
        if script == _OPEN_POST_OVERFLOW_MENU_JS:
            return True
        if script == _CLICK_SAVE_TOGGLE_JS:
            return True
        if script == _READ_SAVE_TOGGLE_STATE_JS:
            return next(reads)
        raise AssertionError(f"unexpected evaluate call: {script[:60]}")

    mock_page.evaluate = AsyncMock(side_effect=evaluate)
    mock_page.wait_for_function = AsyncMock(
        side_effect=[None, PlaywrightTimeoutError("menu stayed closed")]
    )
    actions = _actions(mock_page)

    result = await _with_rate_limit_patch(
        actions.save_post(POST_URL, confirm=True, unsave=False)
    )

    assert (result["status"], result["saved"], result["retry_safe"]) == (
        "state_unconfirmed",
        None,
        False,
    )
