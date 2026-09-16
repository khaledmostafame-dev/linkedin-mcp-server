"""Tests for the shared people/company search pagination walk."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from linkedin_mcp_server.scraping import entity_search as entity_search_module
from linkedin_mcp_server.scraping.capture import CaptureMode, SectionCapture
from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.contracts import ExtractedSection
from linkedin_mcp_server.scraping.entity_search import paginated_entity_search
from linkedin_mcp_server.scraping.link_metadata import Reference
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession

BASE_URL = "https://www.linkedin.com/search/results/people/?keywords=engineer"


def _capture(page) -> SectionCapture:
    session = ScrapingSession(page)
    navigator = PageNavigator(session)
    return SectionCapture(session, navigator, PageContentReader(session))


def extracted(
    text: str,
    references: list[Reference] | None = None,
    error: dict | None = None,
) -> ExtractedSection:
    return ExtractedSection(text=text, references=references or [], error=error)


def _person_ref(index: int) -> Reference:
    return {"kind": "person", "url": f"/in/user{index}/", "text": f"User {index}"}


class TestPaginatedEntitySearch:
    async def test_single_page_reports_no_more_results_when_nothing_new(
        self, mock_page
    ):
        capture = _capture(mock_page)
        with patch.object(
            capture,
            "capture",
            new_callable=AsyncMock,
            return_value=extracted("Page one", [_person_ref(0)]),
        ) as mock_capture:
            result = await paginated_entity_search(
                capture,
                BASE_URL,
                entity_kind="person",
                max_pages=1,
                context="search_people",
            )

        assert mock_capture.await_count == 1
        assert mock_capture.call_args.args[0] == BASE_URL
        assert mock_capture.call_args.kwargs["plan"].mode is CaptureMode.SEARCH_RESULTS
        assert result["pages_fetched"] == 1
        # A single scripted page has no follow-up page to compare against, so
        # every reference on it counts as new and the walk only learns it is
        # done once max_pages is spent -- distinguished below from a second
        # page that genuinely repeats the first.
        assert result["stopped_reason"] == "max_pages"
        assert result["truncated"] is True

    async def test_stops_when_a_later_page_repeats_the_first(self, mock_page):
        capture = _capture(mock_page)
        pages = [
            extracted("Page one", [_person_ref(0), _person_ref(1)]),
            extracted("Page two (repeat)", [_person_ref(0)]),
        ]
        with patch.object(
            capture, "capture", new_callable=AsyncMock, side_effect=pages
        ) as mock_capture:
            result = await paginated_entity_search(
                capture,
                BASE_URL,
                entity_kind="person",
                max_pages=5,
                context="search_people",
            )

        assert mock_capture.await_count == 2
        assert mock_capture.await_args_list[1].args[0] == f"{BASE_URL}&page=2"
        assert result["pages_fetched"] == 2
        assert result["stopped_reason"] == "no_more_results"
        assert result["truncated"] is False
        assert (
            result["sections"]["search_results"] == "Page one\n---\nPage two (repeat)"
        )

    async def test_max_pages_reached_with_fresh_results_is_truncated(self, mock_page):
        capture = _capture(mock_page)
        pages = [extracted("Page", [_person_ref(i)]) for i in range(3)]
        with patch.object(
            capture, "capture", new_callable=AsyncMock, side_effect=pages
        ) as mock_capture:
            result = await paginated_entity_search(
                capture,
                BASE_URL,
                entity_kind="person",
                max_pages=3,
                context="search_people",
            )

        assert mock_capture.await_count == 3
        assert result["pages_fetched"] == 3
        assert result["stopped_reason"] == "max_pages"
        assert result["truncated"] is True

    async def test_zero_max_pages_fetches_nothing(self, mock_page):
        capture = _capture(mock_page)
        with patch.object(capture, "capture", new_callable=AsyncMock) as mock_capture:
            result = await paginated_entity_search(
                capture,
                BASE_URL,
                entity_kind="person",
                max_pages=0,
                context="search_people",
            )

        mock_capture.assert_not_awaited()
        assert result == {
            "url": BASE_URL,
            "sections": {},
            "pages_fetched": 0,
            "stopped_reason": "max_pages",
            "truncated": True,
        }

    async def test_an_empty_page_reports_no_more_results(self, mock_page):
        capture = _capture(mock_page)
        with patch.object(
            capture,
            "capture",
            new_callable=AsyncMock,
            return_value=extracted(""),
        ):
            result = await paginated_entity_search(
                capture,
                BASE_URL,
                entity_kind="person",
                max_pages=3,
                context="search_people",
            )

        assert result["pages_fetched"] == 0
        assert result["stopped_reason"] == "no_more_results"
        assert result["truncated"] is False
        assert "section_errors" not in result

    async def test_a_capture_exception_reports_error_and_keeps_earlier_pages(
        self, mock_page
    ):
        capture = _capture(mock_page)
        pages = [
            extracted("Page one", [_person_ref(0)]),
            RuntimeError("synthetic capture failure"),
        ]
        with patch.object(
            capture, "capture", new_callable=AsyncMock, side_effect=pages
        ) as mock_capture:
            result = await paginated_entity_search(
                capture,
                BASE_URL,
                entity_kind="person",
                max_pages=3,
                context="search_people",
            )

        assert mock_capture.await_count == 2
        assert result["pages_fetched"] == 1
        assert result["stopped_reason"] == "error"
        assert result["truncated"] is False
        assert result["sections"]["search_results"] == "Page one"
        assert result["section_errors"]["search_results"]["error_message"] == (
            "synthetic capture failure"
        )

    async def test_the_wall_clock_budget_stops_the_walk_with_reason_limit(
        self, mock_page
    ):
        """A slow walk stops itself with time left to return what it has.

        Mirrors ``search_jobs``'s own ``SEARCH_TIMEOUT_FRACTION`` budget:
        the second page is never fetched once the elapsed time already
        exceeds ``tool_timeout * SEARCH_TIMEOUT_FRACTION``.
        """
        capture = _capture(mock_page)
        with (
            patch.object(
                capture,
                "capture",
                new_callable=AsyncMock,
                return_value=extracted("Page one", [_person_ref(0)]),
            ) as mock_capture,
            patch.object(
                entity_search_module.time, "monotonic", side_effect=[0.0, 9.0]
            ),
        ):
            result = await paginated_entity_search(
                capture,
                BASE_URL,
                entity_kind="person",
                max_pages=5,
                context="search_people",
                tool_timeout=10.0,  # budget = 10.0 * 0.8 = 8.0s, spent by 9.0s
            )

        assert mock_capture.await_count == 1
        assert result["pages_fetched"] == 1
        assert result["stopped_reason"] == "limit"
        assert result["truncated"] is True
