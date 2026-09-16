"""Public scraping facade and collaborator composition root."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from patchright.async_api import Page

from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.scraping.capture import SectionCapture
from linkedin_mcp_server.scraping.company import CompanyScraper
from linkedin_mcp_server.scraping.connection_actions import ConnectionActions
from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.contracts import (
    ExtractedSection as ExtractedSection,
    FilterValidationError as FilterValidationError,
    rate_limited_section_error as rate_limited_section_error,
)
from linkedin_mcp_server.scraping.conversations import ConversationReader
from linkedin_mcp_server.scraping.events import EventScraper
from linkedin_mcp_server.scraping.feed import FeedScraper
from linkedin_mcp_server.scraping.group import GroupScraper
from linkedin_mcp_server.scraping.job_pages import JobPageReader
from linkedin_mcp_server.scraping.jobs import JobScraper
from linkedin_mcp_server.scraping.message_sender import MessageSender
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.network import NetworkScraper
from linkedin_mcp_server.scraping.person import PersonScraper
from linkedin_mcp_server.scraping.posts import PostSearch
from linkedin_mcp_server.scraping.profile_page import ProfilePageReader
from linkedin_mcp_server.scraping.session import ScrapingSession
from linkedin_mcp_server.scraping.text import (
    strip_conversation_chrome as strip_conversation_chrome,
    strip_linkedin_noise as strip_linkedin_noise,
)


if TYPE_CHECKING:
    from linkedin_mcp_server.callbacks import ProgressCallback


class LinkedInExtractor:
    """Compose scraping owners and expose the stable tool-facing API."""

    def __init__(self, page: Page):
        session = ScrapingSession(page)
        navigator = PageNavigator(session)
        content = PageContentReader(session)
        capture = SectionCapture(session, navigator, content)
        message_sender = MessageSender(session, navigator)
        profile_page = ProfilePageReader(
            session,
            lambda: message_sender._read_profile_message_target(),
        )
        person = PersonScraper(session, navigator, capture, profile_page)

        self._content = content
        self._capture = capture
        self._feed = FeedScraper(session, navigator, content)
        self._message_sender = message_sender
        self._person = person
        self._company = CompanyScraper(session, capture)
        self._connection = ConnectionActions(
            session,
            navigator,
            lambda username: self.scrape_person(username, {"main_profile"}),
        )
        job_pages = JobPageReader(session, navigator, content)
        self._jobs = JobScraper(navigator, capture, job_pages)
        self._posts = PostSearch(capture)
        self._conversations = ConversationReader(
            session, navigator, content, profile_page
        )
        self._network = NetworkScraper(session, navigator, capture)
        self._group = GroupScraper(session, navigator, capture)
        self._event = EventScraper(capture)

    async def get_page_text(self) -> str:
        """Extract innerText from the main content area of the current page."""
        return await self._content.get_page_text()

    async def click_button_by_text(
        self, text: str, *, scope: str = "main", timeout: int = 5000
    ) -> bool:
        """Click the first button or link whose visible text exactly matches."""
        return await self._content.click_button_by_text(
            text, scope=scope, timeout=timeout
        )

    async def extract_feed(self, num_posts: int = 10) -> ExtractedSection:
        """Scrape the LinkedIn home feed, scrolling until enough posts load."""
        return await self._feed.extract_feed(num_posts)

    async def extract_page(
        self,
        url: str,
        section_name: str,
        max_scrolls: int | None = None,
    ) -> ExtractedSection:
        """Navigate, scroll to load lazy content, and extract innerText."""
        return await self._capture.extract_page(url, section_name, max_scrolls)

    async def scrape_person(
        self,
        username: str,
        requested: set[str],
        callbacks: ProgressCallback | None = None,
        max_scrolls: int | None = None,
        *,
        main_profile_already_loaded: bool = False,
        allow_self_alias: bool = False,
    ) -> dict[str, Any]:
        """Scrape a person profile with configurable sections."""
        return await self._person.scrape_person(
            username,
            requested,
            callbacks,
            max_scrolls,
            main_profile_already_loaded=main_profile_already_loaded,
            allow_self_alias=allow_self_alias,
        )

    async def get_my_profile(
        self,
        sections: set[str] | None = None,
        callbacks: ProgressCallback | None = None,
        max_scrolls: int | None = None,
    ) -> dict[str, Any]:
        """Scrape the authenticated user's own LinkedIn profile."""
        return await self._person.get_my_profile(sections, callbacks, max_scrolls)

    async def connect_with_person(
        self,
        username: str,
        *,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Send a LinkedIn connection request or accept an incoming one."""
        return await self._connection.connect_with_person(username, note=note)

    async def get_sidebar_profiles(self, username: str) -> dict[str, Any]:
        """Extract profile links from sidebar sections on a profile page."""
        return await self._person.get_sidebar_profiles(username)

    async def scrape_company(
        self,
        company_name: str,
        requested: set[str],
        callbacks: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Scrape a company profile with configurable sections."""
        return await self._company.scrape_company(company_name, requested, callbacks)

    async def get_company_employees(
        self,
        company_name: str,
        keywords: str | None = None,
    ) -> dict[str, Any]:
        """List employees at a company from the people page."""
        return await self._company.get_company_employees(company_name, keywords)

    async def scrape_job(self, job_id: str) -> dict[str, Any]:
        """Scrape a single job posting."""
        return await self._jobs.scrape_job(job_id)

    async def search_jobs(
        self,
        keywords: str,
        location: str | None = None,
        max_pages: int = 3,
        date_posted: str | None = None,
        job_type: str | None = None,
        experience_level: str | None = None,
        work_type: str | None = None,
        easy_apply: bool = False,
        sort_by: str | None = None,
        tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Search for jobs with pagination and job ID extraction."""
        return await self._jobs.search_jobs(
            keywords,
            location,
            max_pages,
            date_posted,
            job_type,
            experience_level,
            work_type,
            easy_apply,
            sort_by,
            tool_timeout,
        )

    async def get_saved_jobs(self, max_pages: int = 3) -> dict[str, Any]:
        """List the authenticated user's saved job postings."""
        return await self._jobs.get_saved_jobs(max_pages)

    async def search_people(
        self,
        keywords: str,
        location: str | None = None,
        network: list[str] | None = None,
        current_company: str | None = None,
    ) -> dict[str, Any]:
        """Search for people and extract the results page."""
        return await self._person.search_people(
            keywords,
            location=location,
            network=network,
            current_company=current_company,
        )

    async def search_companies(self, keywords: str) -> dict[str, Any]:
        """Search for companies and extract the results page."""
        return await self._company.search_companies(keywords)

    async def search_posts(
        self,
        keywords: str,
        date_posted: str | None = None,
        max_pages: int = 3,
    ) -> dict[str, Any]:
        """Search LinkedIn posts and extract the results page."""
        return await self._posts.search_posts(
            keywords,
            date_posted=date_posted,
            max_pages=max_pages,
        )

    async def get_inbox(self, limit: int = 20) -> dict[str, Any]:
        """List recent conversations from the messaging inbox."""
        return await self._conversations.get_inbox(limit)

    async def get_conversation(
        self,
        linkedin_username: str | None = None,
        thread_id: str | None = None,
        index: int = 0,
    ) -> dict[str, Any]:
        """Read a specific messaging conversation by thread ID or username."""
        return await self._conversations.get_conversation(
            linkedin_username, thread_id, index
        )

    async def search_conversations(
        self, keywords: str, limit: int = 20
    ) -> dict[str, Any]:
        """Search messages by keyword."""
        return await self._conversations.search_conversations(keywords, limit)

    async def send_message(
        self,
        linkedin_username: str,
        message: str,
        *,
        confirm_send: bool,
        profile_urn: str | None = None,
    ) -> dict[str, Any]:
        """Compose and send a new message with explicit confirmation gating."""
        return await self._message_sender.send_message(
            linkedin_username,
            message,
            confirm_send=confirm_send,
            profile_urn=profile_urn,
        )

    async def get_mutual_connections(
        self, linkedin_username: str, max_results: int = 50
    ) -> dict[str, Any]:
        """List connections shared between the authenticated user and a profile."""
        return await self._network.get_mutual_connections(
            linkedin_username, max_results
        )

    async def list_connections(
        self,
        max_results: int = 100,
        sort: Literal["recently_added", "first_name", "last_name"] = "recently_added",
    ) -> dict[str, Any]:
        """List the authenticated user's 1st-degree connections."""
        return await self._network.list_connections(max_results, sort)

    async def get_invitations(
        self,
        direction: Literal["received", "sent"] = "received",
        max_results: int = 50,
    ) -> dict[str, Any]:
        """List outgoing or incoming connection-request invitations."""
        return await self._network.get_invitations(direction, max_results)

    async def withdraw_invitation(
        self, linkedin_username: str, *, confirm: bool
    ) -> dict[str, Any]:
        """Withdraw a previously-sent, still-pending connection request."""
        return await self._connection.withdraw_invitation(
            linkedin_username, confirm=confirm
        )

    async def respond_to_invitation(
        self,
        linkedin_username: str,
        *,
        action: Literal["accept", "ignore"],
        confirm: bool,
    ) -> dict[str, Any]:
        """Accept or ignore an incoming connection request."""
        return await self._connection.respond_to_invitation(
            linkedin_username,
            action=action,
            confirm=confirm,
        )

    async def follow(
        self,
        target_url: str,
        *,
        confirm: bool,
        unfollow: bool = False,
    ) -> dict[str, Any]:
        """Follow or unfollow a person or company page."""
        return await self._network.follow(
            target_url, confirm=confirm, unfollow=unfollow
        )

    async def search_groups(self, keywords: str) -> dict[str, Any]:
        """Search for LinkedIn groups by keyword."""
        return await self._group.search_groups(keywords)

    async def get_group_posts(
        self, group_id: str, max_posts: int = 20
    ) -> dict[str, Any]:
        """List recent posts from a group's home feed."""
        return await self._group.get_group_posts(group_id, max_posts)

    async def get_group_members(
        self,
        group_id: str,
        max_members: int = 50,
        keywords: str | None = None,
    ) -> dict[str, Any]:
        """List members of a LinkedIn group from its /members/ page."""
        return await self._group.get_group_members(group_id, max_members, keywords)

    async def search_events(self, keywords: str, max_pages: int = 3) -> dict[str, Any]:
        """Search for LinkedIn events by keyword."""
        return await self._event.search_events(keywords, max_pages)

    async def get_event_details(self, event_url: str) -> dict[str, Any]:
        """Get the details page for a single LinkedIn event."""
        return await self._event.get_event_details(event_url)

    async def get_event_attendees(
        self, event_url: str, max_attendees: int = 50
    ) -> dict[str, Any]:
        """List attendees of a LinkedIn event."""
        return await self._event.get_event_attendees(event_url, max_attendees)
