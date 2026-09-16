"""Fail-closed behaviour of the share composer workflow.

The browser is mocked: these pin the decisions the composer makes from what
the page reports (identity evidence, editor state, submission outcome), not
LinkedIn's markup, which only a logged-in live check can confirm.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.scraping import post_composer as composer_module
from linkedin_mcp_server.scraping.post_composer import (
    PostComposer,
    _Abort,
    _OPTION_EVIDENCE_JS,
    _OPTION_MARK_JS,
    _Typed,
    evidence_keys,
    evidence_matches,
    scheduled_identifier,
)
from linkedin_mcp_server.scraping.post_content import (
    MentionSegment,
    build_post_request,
    parse_mention_target,
)

TARGET_URL = "https://www.linkedin.com/in/sample-person/"
NOW = datetime(2026, 9, 17, 8, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(composer_module.asyncio, "sleep", AsyncMock())


def _composer() -> tuple[PostComposer, MagicMock]:
    page = MagicMock()
    page.url = "https://www.linkedin.com/feed/?shareActive=true"
    page.keyboard.type = AsyncMock()
    page.keyboard.press = AsyncMock()
    session = MagicMock(page=page)
    return PostComposer(session, MagicMock()), page


def _stub(composer: PostComposer, name: str, mock: MagicMock) -> MagicMock:
    setattr(composer, name, mock)
    return mock


def _mock(composer: PostComposer, name: str) -> MagicMock:
    return cast(MagicMock, getattr(composer, name))


def _option(href: str, *, visible: bool = True) -> dict[str, Any]:
    return {"visible": visible, "hrefs": [href], "urns": []}


class TestIdentityEvidence:
    def test_matching_vanity_path_is_accepted(self):
        target = parse_mention_target(TARGET_URL)
        keys = evidence_keys({"hrefs": ["/in/sample-person/"], "urns": []})
        assert evidence_matches(keys, target)

    def test_a_namesake_with_another_slug_is_not_a_match(self):
        target = parse_mention_target(TARGET_URL)
        keys = evidence_keys({"hrefs": ["/in/sample-person-2/"], "urns": []})
        assert not evidence_matches(keys, target)

    def test_contradicting_evidence_in_the_same_namespace_is_refused(self):
        target = parse_mention_target(TARGET_URL)
        keys = evidence_keys(
            {"hrefs": ["/in/sample-person/", "/in/someone-else/"], "urns": []}
        )
        assert not evidence_matches(keys, target)

    def test_urn_and_vanity_for_one_member_do_not_contradict(self):
        target = parse_mention_target(TARGET_URL)
        keys = evidence_keys(
            {
                "hrefs": ["https://www.linkedin.com/in/sample-person/"],
                "urns": ["urn:li:fsd_profile:ACoAAsynthetic"],
            }
        )
        assert evidence_matches(keys, target)

    def test_scheduled_identifier_ignores_whitespace_layout(self):
        assert scheduled_identifier("a  b\nc") == scheduled_identifier("a b c")
        assert scheduled_identifier("a b c") != scheduled_identifier("a b d")


class TestMentionSelection:
    def _setup(
        self, options: list[dict[str, Any]], states: list[dict[str, Any]]
    ) -> tuple[PostComposer, MagicMock, MagicMock, MagicMock]:
        composer, page = _composer()
        option_locator = self.option_locator = MagicMock()

        async def evaluate_all(program: str, arg: Any = None) -> Any:
            if program == _OPTION_EVIDENCE_JS:
                return options
            if program == _OPTION_MARK_JS:
                return options[arg["index"]]
            raise AssertionError(program)

        option_locator.evaluate_all = AsyncMock(side_effect=evaluate_all)
        marked = MagicMock()
        marked.click = AsyncMock()

        def locator(selector: str) -> Any:
            if selector == composer_module._OPTION_SELECTOR:
                return option_locator
            if selector.startswith("[data-linkedin-mcp-mention="):
                return marked
            raise AssertionError(selector)

        page.locator.side_effect = locator
        editor = MagicMock()
        editor.evaluate = AsyncMock(side_effect=states)
        editor.focus = AsyncMock()
        return composer, page, marked, editor

    async def test_only_the_option_carrying_the_target_identity_is_clicked(self):
        options = [
            _option("/in/sample-person-2/"),  # a namesake listed first
            _option("/in/sample-person/"),
        ]
        states = [
            {"text": "Hi ", "entities": [], "focused": True},
            {
                "text": "Hi Sample Person",
                "entities": [
                    {
                        "text": "Sample Person",
                        "hrefs": ["/in/sample-person/"],
                        "urns": [],
                    }
                ],
                "focused": True,
            },
        ]
        composer, page, marked, editor = self._setup(options, states)
        mention = MentionSegment("Sample Person", parse_mention_target(TARGET_URL))
        typed = _Typed(expected=[], mentions=[])

        await composer._insert_mention(editor, mention, typed)

        marked.click.assert_awaited_once()
        mark_call = page.locator.call_args_list[-1]
        assert mark_call.args[0].startswith("[data-linkedin-mcp-mention=")
        mark_args = [
            call.args[1]
            for call in self.option_locator.evaluate_all.await_args_list
            if call.args[0] == _OPTION_MARK_JS
        ]
        assert len(mark_args) == 1
        mark_arg = mark_args[0]
        assert mark_arg["index"] == 1
        assert typed.expected == ["Sample Person"]

    async def test_suggestions_without_identity_fail_closed_without_selecting(self):
        options = [{"visible": True, "hrefs": [], "urns": []}]
        states = [{"text": "", "entities": [], "focused": True}]
        composer, _page, marked, editor = self._setup(options, states)
        mention = MentionSegment("Sample Person", parse_mention_target(TARGET_URL))

        with pytest.raises(_Abort) as raised:
            await composer._insert_mention(editor, mention, _Typed([], []))

        assert raised.value.status == "mention_unresolved"
        assert "carried no profile URL or URN" in raised.value.message
        marked.click.assert_not_awaited()

    async def test_an_inserted_entity_naming_someone_else_is_refused(self):
        options = [_option("/in/sample-person/")]
        states = [
            {"text": "", "entities": [], "focused": True},
            {
                "text": "Other Person",
                "entities": [
                    {"text": "Other Person", "hrefs": ["/in/other-person/"], "urns": []}
                ],
                "focused": True,
            },
        ]
        composer, _page, _marked, editor = self._setup(options, states)
        mention = MentionSegment("Sample Person", parse_mention_target(TARGET_URL))

        with pytest.raises(_Abort) as raised:
            await composer._insert_mention(editor, mention, _Typed([], []))

        assert raised.value.status == "mention_mismatch"

    async def test_typed_text_left_unconverted_is_refused(self):
        options = [_option("/in/sample-person/")]
        states = [
            {"text": "", "entities": [], "focused": True},
            {"text": "@Sample Person", "entities": [], "focused": True},
        ]
        composer, _page, _marked, editor = self._setup(options, states)
        mention = MentionSegment("Sample Person", parse_mention_target(TARGET_URL))

        with pytest.raises(_Abort) as raised:
            await composer._insert_mention(editor, mention, _Typed([], []))

        assert raised.value.status == "mention_not_inserted"


class TestCreatePostOutcomes:
    def _composer_with_steps(self) -> tuple[PostComposer, MagicMock, MagicMock]:
        composer, page = _composer()
        editor = MagicMock()
        editor.wait_for = AsyncMock()
        _stub(composer, "_open_composer", AsyncMock(return_value=editor))
        _stub(composer, "_editor_state", AsyncMock(return_value={"text": ""}))
        _stub(composer, "_apply_visibility", AsyncMock())
        _stub(composer, "_type_segments", AsyncMock(return_value=_Typed([], [])))
        _stub(composer, "_dismiss_composer", AsyncMock())
        _stub(
            composer,
            "_post_links",
            AsyncMock(return_value={"all": [], "announced": []}),
        )
        return composer, page, editor

    def _primary(self, composer: PostComposer, page: MagicMock) -> MagicMock:
        primary = MagicMock()
        primary.is_enabled = AsyncMock(return_value=True)
        primary.click = AsyncMock()
        dialog = MagicMock()
        dialog.locator.return_value.last = primary
        _stub(composer, "_composer_dialog", MagicMock(return_value=dialog))
        visible_editors = MagicMock()
        visible_editors.count = AsyncMock(return_value=1)
        page.locator.return_value = visible_editors
        return primary

    async def test_a_restored_draft_is_left_untouched(self):
        composer, page, _editor = self._composer_with_steps()
        _stub(
            composer,
            "_editor_state",
            AsyncMock(return_value={"text": "someone's draft"}),
        )

        result = await composer.create_post(build_post_request("Hello", now=NOW))

        assert result["status"] == "composer_occupied"
        assert result["retry_safe"] is True
        _mock(composer, "_apply_visibility").assert_not_awaited()
        _mock(composer, "_dismiss_composer").assert_not_awaited()
        page.keyboard.type.assert_not_awaited()

    async def test_a_refused_step_never_clicks_post_and_clears_typed_text(self):
        composer, page, _editor = self._composer_with_steps()
        primary = self._primary(composer, page)
        _stub(
            composer,
            "_type_segments",
            AsyncMock(side_effect=_Abort("mention_unresolved", "no match")),
        )

        result = await composer.create_post(build_post_request("Hello", now=NOW))

        assert result["status"] == "mention_unresolved"
        assert result["retry_safe"] is True
        primary.click.assert_not_awaited()
        _mock(composer, "_dismiss_composer").assert_awaited_once_with(clear=True)

    async def test_a_visibility_refusal_dismisses_without_clearing(self):
        composer, page, _editor = self._composer_with_steps()
        primary = self._primary(composer, page)
        _stub(
            composer,
            "_apply_visibility",
            AsyncMock(side_effect=_Abort("visibility_unavailable", "no option")),
        )

        result = await composer.create_post(build_post_request("Hello", now=NOW))

        assert result["status"] == "visibility_unavailable"
        _mock(composer, "_type_segments").assert_not_awaited()
        primary.click.assert_not_awaited()
        _mock(composer, "_dismiss_composer").assert_awaited_once_with(clear=False)

    async def test_composer_staying_open_after_the_click_is_not_retry_safe(self):
        composer, page, editor = self._composer_with_steps()
        primary = self._primary(composer, page)
        editor.wait_for = AsyncMock(side_effect=PlaywrightTimeoutError("still open"))

        result = await composer.create_post(build_post_request("Hello", now=NOW))

        primary.click.assert_awaited_once()
        assert result["status"] == "post_unconfirmed"
        assert result["retry_safe"] is False

    async def test_published_post_reports_the_captured_urn(self):
        composer, page, _editor = self._composer_with_steps()
        self._primary(composer, page)
        _stub(
            composer,
            "_capture_new_post",
            AsyncMock(return_value="urn:li:activity:1234567890"),
        )

        result = await composer.create_post(build_post_request("Hello", now=NOW))

        assert result["status"] == "published"
        assert result["post_url"] == (
            "https://www.linkedin.com/feed/update/urn:li:activity:1234567890/"
        )
        assert result["retry_safe"] is False

    async def test_disabled_post_action_refuses_before_submission(self):
        composer, page, _editor = self._composer_with_steps()
        primary = self._primary(composer, page)
        primary.is_enabled = AsyncMock(return_value=False)

        result = await composer.create_post(build_post_request("Hello", now=NOW))

        assert result["status"] == "post_unavailable"
        assert result["retry_safe"] is True
        primary.click.assert_not_awaited()


class TestScheduledPosts:
    async def test_invalid_identifier_is_refused_before_opening_anything(self):
        composer, _page = _composer()
        _stub(composer, "_open_composer", AsyncMock())

        result = await composer.delete_scheduled_post("first one", confirm=True)

        assert result["status"] == "invalid_identifier"
        _mock(composer, "_open_composer").assert_not_awaited()

    async def test_preview_resolves_the_entry_and_deletes_nothing(self):
        composer, page = _composer()
        _stub(composer, "_open_scheduled_list", AsyncMock(return_value=None))
        _stub(composer, "_dismiss_composer", AsyncMock())
        text = "Scheduled synthetic post body"
        page.evaluate = AsyncMock(return_value=[text])

        result = await composer.delete_scheduled_post(
            scheduled_identifier(text), confirm=False
        )

        assert result["status"] == "preview"
        assert result["entry"] == text
        assert page.evaluate.await_count == 1  # read only, never marked
        _mock(composer, "_dismiss_composer").assert_awaited_once()

    async def test_duplicate_entries_are_ambiguous(self):
        composer, page = _composer()
        _stub(composer, "_open_scheduled_list", AsyncMock(return_value=None))
        _stub(composer, "_dismiss_composer", AsyncMock())
        text = "Same body"
        page.evaluate = AsyncMock(return_value=[text, text])

        result = await composer.delete_scheduled_post(
            scheduled_identifier(text), confirm=True
        )

        assert result["status"] == "entry_ambiguous"


class TestPostLinkCapture:
    async def test_a_link_merely_new_on_the_feed_is_not_claimed(self, monkeypatch):
        composer, _page = _composer()
        monkeypatch.setattr(composer_module, "_POST_LINK_TIMEOUT_SECONDS", 0.05)
        _stub(
            composer,
            "_post_links",
            AsyncMock(return_value={"all": ["urn:li:activity:2"], "announced": []}),
        )

        assert await composer._capture_new_post({"all": [], "announced": []}) is None

    async def test_the_announced_link_is_claimed(self):
        composer, _page = _composer()
        _stub(
            composer,
            "_post_links",
            AsyncMock(
                return_value={
                    "all": ["urn:li:activity:1", "urn:li:activity:2"],
                    "announced": ["urn:li:activity:2"],
                }
            ),
        )

        urn = await composer._capture_new_post(
            {"all": ["urn:li:activity:1"], "announced": []}
        )

        assert urn == "urn:li:activity:2"
