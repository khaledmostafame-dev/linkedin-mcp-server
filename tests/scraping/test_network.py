"""Tests for the network-graph scraping owner (connections, invitations, follow)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from linkedin_mcp_server.scraping.capture import SectionCapture
from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.contracts import ExtractedSection
from linkedin_mcp_server.scraping.link_metadata import Reference
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.network import NetworkScraper, _scrolls_for
from linkedin_mcp_server.scraping.session import ScrapingSession


def _scraper(page) -> NetworkScraper:
    """Wire the network owner the way the facade does."""
    session = ScrapingSession(page)
    navigator = PageNavigator(session)
    return NetworkScraper(
        session,
        navigator,
        SectionCapture(session, navigator, PageContentReader(session)),
    )


def extracted(
    text: str,
    references: list[Reference] | None = None,
    error: dict | None = None,
) -> ExtractedSection:
    return ExtractedSection(text=text, references=references or [], error=error)


def _refs(n: int) -> list[Reference]:
    return [{"kind": "person", "url": f"/in/user{i}/"} for i in range(n)]


class TestScrollsFor:
    """Pure bounding logic: a mutation here is caught without any browser."""

    def test_small_result_count_uses_the_minimum(self):
        assert _scrolls_for(1) == 1
        assert _scrolls_for(10) == 1

    def test_scales_with_rows_per_scroll(self):
        assert _scrolls_for(11) == 2
        assert _scrolls_for(50) == 5

    def test_is_capped(self):
        assert _scrolls_for(10_000) == 25


class TestGetMutualConnections:
    async def test_no_mutual_connections_link_returns_empty_without_capturing(
        self, mock_page
    ):
        scraper = _scraper(mock_page)
        mock_page.evaluate = AsyncMock(return_value=None)

        with patch.object(
            scraper._capture, "capture", new_callable=AsyncMock
        ) as mock_capture:
            result = await scraper.get_mutual_connections("testuser")

        assert result["sections"] == {}
        assert result["stopped_reason"] == "no_mutual_connections_link_found"
        mock_capture.assert_not_awaited()

    async def test_found_link_is_resolved_and_captured(self, mock_page):
        scraper = _scraper(mock_page)
        mock_page.evaluate = AsyncMock(
            return_value="/search/results/people/?facetConnectionOf=%22ACoAAA%22"
        )

        with patch.object(
            scraper._capture,
            "capture",
            new_callable=AsyncMock,
            return_value=extracted("Jane Doe\nMutual connection", _refs(3)),
        ) as mock_capture:
            result = await scraper.get_mutual_connections("testuser", max_results=50)

        captured_url = mock_capture.call_args.args[0]
        assert captured_url == (
            "https://www.linkedin.com/search/results/people/"
            "?facetConnectionOf=%22ACoAAA%22"
        )
        assert result["sections"]["mutual_connections"] == "Jane Doe\nMutual connection"
        assert result["references"]["mutual_connections"] == _refs(3)
        assert result["stopped_reason"] == "end_of_results"

    async def test_reaching_max_results_is_reported_as_such(self, mock_page):
        scraper = _scraper(mock_page)
        mock_page.evaluate = AsyncMock(
            return_value="/search/results/people/?facetConnectionOf=%22X%22"
        )

        with patch.object(
            scraper._capture,
            "capture",
            new_callable=AsyncMock,
            return_value=extracted("text", _refs(5)),
        ):
            result = await scraper.get_mutual_connections("testuser", max_results=5)

        assert result["stopped_reason"] == "max_results_reached"


class TestListConnections:
    async def test_default_sort_builds_recently_added_url(self, mock_page):
        scraper = _scraper(mock_page)
        with patch.object(
            scraper._capture,
            "capture",
            new_callable=AsyncMock,
            return_value=extracted("Alice\nBob", _refs(2)),
        ) as mock_capture:
            result = await scraper.list_connections()

        url = mock_capture.call_args.args[0]
        assert url == (
            "https://www.linkedin.com/mynetwork/invite-connect/connections/"
            "?sortType=RECENTLY_ADDED"
        )
        assert result["sections"]["connections"] == "Alice\nBob"

    async def test_sort_token_maps_to_linkedins_own_query_values(self, mock_page):
        scraper = _scraper(mock_page)
        with patch.object(
            scraper._capture,
            "capture",
            new_callable=AsyncMock,
            return_value=extracted("text"),
        ) as mock_capture:
            await scraper.list_connections(sort="first_name")

        url = mock_capture.call_args.args[0]
        assert "sortType=FIRST_NAME" in url

    async def test_unrecognized_sort_falls_back_to_recently_added(self, mock_page):
        scraper = _scraper(mock_page)
        with patch.object(
            scraper._capture,
            "capture",
            new_callable=AsyncMock,
            return_value=extracted("text"),
        ) as mock_capture:
            await scraper.list_connections(sort="bogus")  # ty: ignore[invalid-argument-type]

        url = mock_capture.call_args.args[0]
        assert "sortType=RECENTLY_ADDED" in url


class TestGetInvitations:
    async def test_received_is_the_default_direction(self, mock_page):
        scraper = _scraper(mock_page)
        with patch.object(
            scraper._capture,
            "capture",
            new_callable=AsyncMock,
            return_value=extracted("Jane wants to connect"),
        ) as mock_capture:
            result = await scraper.get_invitations()

        url, section_name = (
            mock_capture.call_args.args[0],
            mock_capture.call_args.args[1],
        )
        assert url == "https://www.linkedin.com/mynetwork/invitation-manager/"
        assert section_name == "received_invitations"
        assert result["sections"]["received_invitations"] == "Jane wants to connect"

    async def test_sent_direction_uses_the_sent_subpath(self, mock_page):
        scraper = _scraper(mock_page)
        with patch.object(
            scraper._capture,
            "capture",
            new_callable=AsyncMock,
            return_value=extracted("Pending invite"),
        ) as mock_capture:
            result = await scraper.get_invitations(direction="sent")

        url = mock_capture.call_args.args[0]
        assert url == "https://www.linkedin.com/mynetwork/invitation-manager/sent/"
        assert "sent_invitations" in result["sections"]


class TestFollow:
    async def test_confirm_false_is_a_preview_with_no_navigation(self, mock_page):
        scraper = _scraper(mock_page)
        with patch.object(
            PageNavigator, "_navigate_to_page", new_callable=AsyncMock
        ) as mock_nav:
            result = await scraper.follow(
                "https://www.linkedin.com/company/anthropic/", confirm=False
            )

        assert result["status"] == "preview"
        assert result["requested"] == "follow"
        mock_nav.assert_not_awaited()

    async def test_unfollow_preview_says_unfollow(self, mock_page):
        scraper = _scraper(mock_page)
        result = await scraper.follow(
            "https://www.linkedin.com/company/anthropic/",
            confirm=False,
            unfollow=True,
        )
        assert result["requested"] == "unfollow"

    async def test_ambiguous_candidates_are_refused_not_guessed(self, mock_page):
        """Two labeled buttons in the top card (e.g. Follow + Message): the
        tool must not guess which one to click."""
        scraper = _scraper(mock_page)
        mock_page.evaluate = AsyncMock(return_value={"ok": False, "count": 2})

        with (
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
        ):
            result = await scraper.follow(
                "https://www.linkedin.com/company/anthropic/", confirm=True
            )

        assert result["status"] == "action_unavailable"

    async def test_unambiguous_toggle_click_reports_toggled_on_attribute_change(
        self, mock_page
    ):
        scraper = _scraper(mock_page)
        before = {
            "ok": True,
            "count": 1,
            "ariaPressed": "false",
            "ariaExpanded": None,
            "ariaHasPopup": None,
        }
        after = {
            "ok": True,
            "count": 1,
            "ariaPressed": "true",
            "ariaExpanded": None,
            "ariaHasPopup": None,
        }
        mock_page.evaluate = AsyncMock(side_effect=[before, True, after])

        with (
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch(
                "linkedin_mcp_server.scraping.session.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await scraper.follow(
                "https://www.linkedin.com/company/anthropic/", confirm=True
            )

        assert result["status"] == "toggled"
        assert result["requested"] == "follow"

    async def test_no_state_change_reports_uncertain(self, mock_page):
        scraper = _scraper(mock_page)
        same = {
            "ok": True,
            "count": 1,
            "ariaPressed": "false",
            "ariaExpanded": None,
            "ariaHasPopup": None,
        }
        mock_page.evaluate = AsyncMock(side_effect=[same, True, dict(same)])

        with (
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch(
                "linkedin_mcp_server.scraping.session.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await scraper.follow(
                "https://www.linkedin.com/company/anthropic/", confirm=True
            )

        assert result["status"] == "uncertain"

    async def test_an_attribute_change_away_from_the_request_is_not_toggled(
        self, mock_page
    ):
        """An unfollow whose control still reports pressed is not a success."""
        scraper = _scraper(mock_page)
        before = {"ok": True, "count": 1, "ariaPressed": "true"}
        after = {"ok": True, "count": 1, "ariaPressed": "true", "ariaExpanded": "x"}
        mock_page.evaluate = AsyncMock(side_effect=[before, True, after])

        with (
            patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock),
            patch(
                "linkedin_mcp_server.scraping.session.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await scraper.follow(
                "https://www.linkedin.com/company/anthropic/",
                confirm=True,
                unfollow=True,
            )

        assert result["status"] == "uncertain"

    async def test_missing_pressed_state_is_refused_without_clicking(self, mock_page):
        """Without aria-pressed the current state is unknown, so the toggle
        could undo an existing follow: nothing is clicked."""
        scraper = _scraper(mock_page)
        before = {
            "ok": True,
            "count": 1,
            "ariaPressed": None,
            "ariaExpanded": None,
            "ariaHasPopup": None,
        }
        mock_page.evaluate = AsyncMock(side_effect=[before, True])

        with patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock):
            result = await scraper.follow(
                "https://www.linkedin.com/company/anthropic/", confirm=True
            )

        assert result["status"] == "action_unavailable"
        assert mock_page.evaluate.await_count == 1

    @pytest.mark.parametrize(
        ("pressed", "unfollow", "status"),
        [("true", False, "already_following"), ("false", True, "already_unfollowed")],
    )
    async def test_requested_state_already_holding_clicks_nothing(
        self, mock_page, pressed, unfollow, status
    ):
        scraper = _scraper(mock_page)
        before = {"ok": True, "count": 1, "ariaPressed": pressed}
        mock_page.evaluate = AsyncMock(side_effect=[before, True])

        with patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock):
            result = await scraper.follow(
                "https://www.linkedin.com/company/anthropic/",
                confirm=True,
                unfollow=unfollow,
            )

        assert result["status"] == status
        assert mock_page.evaluate.await_count == 1

    async def test_click_that_does_not_land_is_reported_as_unavailable(self, mock_page):
        scraper = _scraper(mock_page)
        before = {
            "ok": True,
            "count": 1,
            "ariaPressed": "false",
            "ariaExpanded": None,
            "ariaHasPopup": None,
        }
        mock_page.evaluate = AsyncMock(side_effect=[before, False])

        with patch.object(PageNavigator, "_navigate_to_page", new_callable=AsyncMock):
            result = await scraper.follow(
                "https://www.linkedin.com/company/anthropic/", confirm=True
            )

        assert result["status"] == "action_unavailable"

    async def test_company_url_resolves_to_company_page(self, mock_page):
        scraper = _scraper(mock_page)
        assert (
            scraper._resolve_follow_target(
                "https://www.linkedin.com/company/anthropic/"
            )
            == "https://www.linkedin.com/company/anthropic/"
        )

    async def test_person_url_resolves_to_profile_page(self, mock_page):
        scraper = _scraper(mock_page)
        assert (
            scraper._resolve_follow_target("https://www.linkedin.com/in/stickerdaniel/")
            == "https://www.linkedin.com/in/stickerdaniel/"
        )

    async def test_bare_slug_defaults_to_a_person(self, mock_page):
        scraper = _scraper(mock_page)
        assert (
            scraper._resolve_follow_target("stickerdaniel")
            == "https://www.linkedin.com/in/stickerdaniel/"
        )
