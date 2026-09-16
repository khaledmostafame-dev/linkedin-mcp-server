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
    build_poll,
    build_post_edit,
    build_post_request,
    parse_mention_target,
    parse_post_as,
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


class TestPostAs:
    def _setup(
        self, before: list[dict[str, Any]], after: list[dict[str, Any]]
    ) -> tuple[PostComposer, MagicMock, MagicMock]:
        composer, page = _composer()
        author_button = MagicMock()
        author_button.wait_for = AsyncMock()
        author_button.click = AsyncMock()
        author_button.evaluate = AsyncMock(return_value={"hrefs": [], "urns": []})
        dialog = MagicMock()
        dialog.locator.return_value.first = author_button
        _stub(composer, "_composer_dialog", MagicMock(return_value=dialog))
        radios = MagicMock()
        radios.first.wait_for = AsyncMock()
        radios.evaluate_all = AsyncMock(side_effect=[before, after])
        option = MagicMock()
        option.click = AsyncMock()
        radios.nth.return_value = option
        other = MagicMock()
        other.first.wait_for = AsyncMock()
        other.first.click = AsyncMock()
        other.last.click = AsyncMock()
        settings = MagicMock()
        settings.locator.side_effect = lambda selector: (
            radios if selector == composer_module._ACTOR_RADIO_SELECTOR else other
        )
        page.locator.return_value.filter.return_value = settings
        page.locator.return_value.count = AsyncMock(return_value=0)
        _stub(
            composer, "_editor", MagicMock(return_value=MagicMock(wait_for=AsyncMock()))
        )
        return composer, radios, option

    def _radio(self, urn: str, checked: bool = False) -> dict[str, Any]:
        return {"visible": True, "checked": checked, "hrefs": [], "urns": [urn]}

    async def test_only_the_page_carrying_the_identity_is_selected(self):
        before = [
            self._radio("urn:li:fsd_profile:ACoAAself", checked=True),
            self._radio("urn:li:organization:999"),
            self._radio("urn:li:organization:12345"),
        ]
        after = [
            self._radio("urn:li:fsd_profile:ACoAAself"),
            self._radio("urn:li:organization:999"),
            self._radio("urn:li:organization:12345", checked=True),
        ]
        composer, radios, option = self._setup(before, after)

        actor = await composer._apply_actor(parse_post_as("12345"))

        radios.nth.assert_called_once_with(2)
        option.click.assert_awaited_once()
        assert actor["verified_by"] == "selected_option"

    async def test_options_without_identity_fail_closed(self):
        before = [{"visible": True, "checked": False, "hrefs": [], "urns": []}]
        composer, _radios, option = self._setup(before, [])

        with pytest.raises(_Abort) as raised:
            await composer._apply_actor(parse_post_as("12345"))

        assert raised.value.status == "actor_unresolved"
        option.click.assert_not_awaited()

    async def test_a_selection_that_does_not_stick_is_refused(self):
        before = [self._radio("urn:li:organization:12345")]
        after = [self._radio("urn:li:organization:12345", checked=False)]
        composer, _radios, _option = self._setup(before, after)

        with pytest.raises(_Abort) as raised:
            await composer._apply_actor(parse_post_as("12345"))

        assert raised.value.status == "actor_unresolved"

    async def test_create_post_as_a_page_skips_member_visibility(self):
        composer, page = _composer()
        editor = MagicMock(wait_for=AsyncMock())
        _stub(composer, "_open_composer", AsyncMock(return_value=editor))
        _stub(composer, "_editor_state", AsyncMock(return_value={"text": ""}))
        actor = _stub(
            composer,
            "_apply_actor",
            AsyncMock(side_effect=_Abort("actor_unresolved", "no match")),
        )
        visibility = _stub(composer, "_apply_visibility", AsyncMock())
        _stub(composer, "_dismiss_composer", AsyncMock())

        result = await composer.create_post(
            build_post_request("Hello", post_as="12345", now=NOW)
        )

        assert result["status"] == "actor_unresolved"
        actor.assert_awaited_once()
        visibility.assert_not_awaited()
        page.keyboard.type.assert_not_awaited()


POST_URL = "https://www.linkedin.com/feed/update/urn:li:activity:1234567890/"


class TestOwnPosts:
    def _setup(self, actor_href: str | None) -> tuple[PostComposer, MagicMock]:
        composer, page = _composer()

        async def navigate(url: str) -> None:
            page.url = (
                "https://www.linkedin.com/in/sample-person/"
                if url.endswith("/in/me/")
                else url
            )

        setattr(
            composer._navigator, "_navigate_to_page", AsyncMock(side_effect=navigate)
        )
        setattr(composer._session, "check_rate_limit", AsyncMock())
        page.wait_for_selector = AsyncMock()
        page.evaluate = AsyncMock(
            return_value={
                "actorHref": actor_href,
                "controlCount": 2,
                "domUrn": True,
                "text": "Synthetic post",
            }
        )
        menu_item = MagicMock()
        menu_item.click = AsyncMock()
        menu_item.wait_for = AsyncMock()
        page.locator.return_value.filter.return_value.first = menu_item
        self.menu_item = menu_item
        return composer, page

    async def test_another_members_post_is_refused_before_opening_its_menu(self):
        composer, _page = self._setup("/in/someone-else/")

        result = await composer.delete_post(POST_URL, confirm=True)

        assert result["status"] == "not_own_post"
        self.menu_item.click.assert_not_awaited()

    async def test_missing_owner_actions_are_refused(self):
        composer, page = self._setup("/in/sample-person/")
        self.menu_item.wait_for = AsyncMock(side_effect=PlaywrightTimeoutError("no"))

        result = await composer.delete_post(POST_URL, confirm=True)

        assert result["status"] == "not_own_post"
        page.keyboard.press.assert_awaited_with("Escape")

    async def test_preview_verifies_authorship_and_deletes_nothing(self):
        composer, page = self._setup("/in/Sample-Person/")

        result = await composer.delete_post(POST_URL, confirm=False)

        assert result["status"] == "preview"
        assert result["post_text"] == "Synthetic post"
        # One click opens the control menu; the delete item is never clicked.
        assert self.menu_item.click.await_count == 1
        page.keyboard.press.assert_awaited_with("Escape")

    async def test_edit_preview_never_opens_the_editor(self):
        composer, _page = self._setup("/in/sample-person/")

        result = await composer.edit_post(
            POST_URL, build_post_edit("New text", allow_schedule=False), confirm=False
        )

        assert result["status"] == "preview"
        assert result["new_text"] == "New text"
        assert self.menu_item.click.await_count == 1


class TestEditScheduledPost:
    async def test_preview_resolves_and_changes_nothing(self):
        composer, page = _composer()
        _stub(composer, "_open_scheduled_list", AsyncMock(return_value=None))
        dismiss = _stub(composer, "_dismiss_composer", AsyncMock())
        text = "Scheduled synthetic post body"
        page.evaluate = AsyncMock(return_value=[text])

        result = await composer.edit_scheduled_post(
            scheduled_identifier(text), build_post_edit("New body"), confirm=False
        )

        assert result["status"] == "preview"
        assert result["changes"]["text"] == "New body"
        assert page.evaluate.await_count == 1
        dismiss.assert_awaited_once()

    async def test_an_editor_holding_another_post_is_abandoned(self):
        composer, page = _composer()
        _stub(composer, "_open_scheduled_list", AsyncMock(return_value=None))
        _stub(composer, "_dismiss_composer", AsyncMock())
        text = "Scheduled synthetic post body"
        page.evaluate = AsyncMock(side_effect=[[text], True])
        clicker = MagicMock(click=AsyncMock())
        page.locator.return_value = clicker
        clicker.filter.return_value.first = clicker
        editor = MagicMock(wait_for=AsyncMock())
        _stub(composer, "_editor", MagicMock(return_value=editor))
        _stub(
            composer,
            "_editor_state",
            AsyncMock(return_value={"text": "A different post", "entities": []}),
        )
        abandon = _stub(composer, "_abandon_edit", AsyncMock())
        replace = _stub(composer, "_replace_editor_text", AsyncMock())

        result = await composer.edit_scheduled_post(
            scheduled_identifier(text), build_post_edit("New body"), confirm=True
        )

        assert result["status"] == "edit_mismatch"
        assert result["retry_safe"] is True
        replace.assert_not_awaited()
        abandon.assert_awaited_once()


class TestPolls:
    def _form(
        self, composer: PostComposer, *, readback: str | None = None, shown: str
    ) -> tuple[list[MagicMock], MagicMock]:
        values: list[MagicMock] = []

        def field(index: int) -> MagicMock:
            item = MagicMock()
            item.evaluate = AsyncMock(return_value=-1)
            stored: dict[str, str] = {}

            async def fill(value: str) -> None:
                stored["value"] = value

            async def input_value() -> str:
                if readback is not None and index == 1:
                    return readback
                return stored.get("value", "")

            item.fill = AsyncMock(side_effect=fill)
            item.input_value = AsyncMock(side_effect=input_value)
            values.append(item)
            return item

        fields = MagicMock()
        fields.count = AsyncMock(return_value=3)
        made = [field(i) for i in range(3)]
        fields.nth.side_effect = lambda i: made[i]
        select = MagicMock()
        select.select_option = AsyncMock()
        select.evaluate = AsyncMock(
            side_effect=[
                {"values": ["a", "b", "c", "d"], "selectedIndex": 0},
                {"values": ["a", "b", "c", "d"], "selectedIndex": 2},
            ]
        )
        selects = MagicMock(count=AsyncMock(return_value=1), first=select)
        done = MagicMock(click=AsyncMock())
        dialog = MagicMock(wait_for=AsyncMock())

        def locator(selector: str) -> MagicMock:
            if selector == "select:visible":
                return selects
            if selector.startswith("button:not([disabled])"):
                return MagicMock(last=done)
            return fields

        dialog.locator.side_effect = locator
        _stub(composer, "_open_poll_form", AsyncMock(return_value=dialog))
        _stub(
            composer, "_editor", MagicMock(return_value=MagicMock(wait_for=AsyncMock()))
        )
        composer_dialog = MagicMock()
        composer_dialog.first.inner_text = AsyncMock(return_value=shown)
        _stub(composer, "_composer_dialog", MagicMock(return_value=composer_dialog))
        self.select = select
        self.done = done
        return values, done

    async def test_fields_are_filled_in_order_and_the_preview_verified(self):
        composer, _page = _composer()
        poll = build_poll("Best format?", ["Carousel", "Video"], 7)
        values, done = self._form(
            composer, shown="Best format? Carousel Video 1 week left"
        )

        await composer._attach_poll(poll)

        assert [v.fill.await_args.args[0] for v in values] == [
            "Best format?",
            "Carousel",
            "Video",
        ]
        self.select.select_option.assert_awaited_once_with(index=2)
        done.click.assert_awaited_once()

    async def test_a_field_that_does_not_read_back_is_refused(self):
        composer, _page = _composer()
        poll = build_poll("Best format?", ["Carousel", "Video"], 7)
        _values, done = self._form(composer, readback="Carous", shown="")

        with pytest.raises(_Abort) as raised:
            await composer._attach_poll(poll)

        assert raised.value.status == "poll_rejected"
        done.click.assert_not_awaited()

    async def test_a_preview_missing_an_option_is_refused(self):
        composer, _page = _composer()
        poll = build_poll("Best format?", ["Carousel", "Video"], 7)
        self._form(composer, shown="Best format? Carousel")

        with pytest.raises(_Abort) as raised:
            await composer._attach_poll(poll)

        assert raised.value.status == "poll_mismatch"

    async def test_create_poll_refuses_a_request_without_a_poll(self):
        composer, _page = _composer()
        create = _stub(composer, "create_post", AsyncMock())

        result = await composer.create_poll(build_post_request("Hello", now=NOW))

        assert result["status"] == "invalid_request"
        create.assert_not_awaited()

    async def test_a_poll_step_failure_never_clicks_post(self):
        composer, page = _composer()
        editor = MagicMock(wait_for=AsyncMock())
        _stub(composer, "_open_composer", AsyncMock(return_value=editor))
        _stub(composer, "_editor_state", AsyncMock(return_value={"text": ""}))
        _stub(composer, "_apply_visibility", AsyncMock())
        _stub(
            composer,
            "_attach_poll",
            AsyncMock(side_effect=_Abort("poll_unavailable", "no poll action")),
        )
        typed = _stub(composer, "_type_segments", AsyncMock())
        dismiss = _stub(composer, "_dismiss_composer", AsyncMock())
        request = build_post_request(
            "", poll=build_poll("Q?", ["Yes", "No"], 1), now=NOW
        )

        result = await composer.create_poll(request)

        assert result["status"] == "poll_unavailable"
        assert result["retry_safe"] is True
        typed.assert_not_awaited()
        dismiss.assert_awaited_once_with(clear=False)
        page.keyboard.type.assert_not_awaited()
