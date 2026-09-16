"""Browser-free post request parsing: mentions, limits, schedule, formatting."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from linkedin_mcp_server.scraping.post_content import (
    LINKEDIN_POST_CHARACTER_LIMIT,
    MentionSegment,
    PostAttachment,
    PostValidationError,
    TextSegment,
    build_post_edit,
    build_post_request,
    date_matches,
    format_schedule_date,
    format_schedule_time,
    identity_key_from_url,
    identity_key_from_urn,
    parse_post_as,
    parse_post_text,
    parse_post_url,
    parse_schedule_at,
    post_preview,
    resolve_date_order,
    time_matches,
    utf16_length,
)

NOW = datetime(2026, 9, 17, 8, 0, tzinfo=UTC)


def _document() -> PostAttachment:
    return PostAttachment(
        kind="document",
        path="/tmp/x.pdf",
        filename="x.pdf",
        content_type="application/pdf",
        size_bytes=10,
        source="document (base64)",
    )


class TestMentions:
    def test_person_company_and_urn_mentions_split_the_text(self):
        segments = parse_post_text(
            "Thanks @[Sample Person](https://www.linkedin.com/in/sample-person/) "
            "and @[Example Co](https://www.linkedin.com/company/example-co/) "
            "and @[Other](urn:li:fsd_profile:ACoAAsynthetic1) #automation"
        )

        mentions = [s for s in segments if isinstance(s, MentionSegment)]
        assert [m.name for m in mentions] == ["Sample Person", "Example Co", "Other"]
        assert [m.target.key for m in mentions] == [
            "person:/in/sample-person/",
            "company:/company/example-co/",
            "profile:ACoAAsynthetic1",
        ]
        assert isinstance(segments[-1], TextSegment)
        assert segments[-1].text == " #automation"

    def test_rendered_text_uses_display_names(self):
        request = build_post_request(
            "Hi @[Sample Person](https://www.linkedin.com/in/sample-person/)!",
            now=NOW,
        )
        assert request.rendered_text == "Hi Sample Person!"

    @pytest.mark.parametrize(
        "reference",
        [
            "http://www.linkedin.com/in/sample/",
            "https://evil.example/in/sample/",
            "https://www.linkedin.com.evil.example/in/sample/",
            "https://www.linkedin.com/feed/",
            "urn:li:organization:not-a-number",
            "urn:li:activity:123",
            "/in/sample/",
        ],
    )
    def test_unverifiable_targets_are_refused(self, reference):
        with pytest.raises(PostValidationError, match="Mention target"):
            parse_post_text(f"@[Name]({reference})")

    def test_malformed_mention_is_refused_instead_of_typed_as_plain_text(self):
        with pytest.raises(PostValidationError, match="malformed mention"):
            parse_post_text("Hello @[Name](https://www.linkedin.com/in/a b/)")

    def test_same_member_by_vanity_is_case_insensitive(self):
        upper = identity_key_from_url("https://www.linkedin.com/in/Sample-Person")
        relative = identity_key_from_url("/in/sample-person/overlay/about/")
        assert upper is not None and relative is not None
        assert upper.key == relative.key

    def test_company_urn_namespaces_collapse_to_one_identity(self):
        targets = [
            identity_key_from_urn(f"urn:li:{kind}:1234")
            for kind in ("company", "fsd_company", "organization")
        ]
        keys = {target.key for target in targets if target is not None}
        assert len(targets) == 3 and None not in targets
        assert keys == {"company:urn:1234"}


class TestLimitsAndShape:
    def test_character_limit_counts_utf16_units(self):
        # One emoji is two UTF-16 units: a browser counter sees 3,000 here.
        text = "a" * (LINKEDIN_POST_CHARACTER_LIMIT - 2) + "\U0001f600"
        assert utf16_length(text) == LINKEDIN_POST_CHARACTER_LIMIT
        build_post_request(text, now=NOW)
        with pytest.raises(PostValidationError, match="3000"):
            build_post_request(text + "b", now=NOW)

    def test_control_characters_other_than_line_breaks_are_refused(self):
        build_post_request("line one\r\nline two", now=NOW)
        with pytest.raises(PostValidationError, match="control characters"):
            build_post_request("tab\there", now=NOW)

    def test_blank_text_needs_an_attachment(self):
        with pytest.raises(PostValidationError, match="text or an attachment"):
            build_post_request("   ", now=NOW)
        build_post_request("", attachments_pending=True, now=NOW)

    def test_document_needs_a_title(self):
        with pytest.raises(PostValidationError, match="title"):
            build_post_request("x", document=_document(), document_title=" ", now=NOW)

    def test_visibility_is_validated(self):
        with pytest.raises(PostValidationError, match="visibility"):
            build_post_request("x", visibility="public", now=NOW)

    def test_preview_reports_counts_mentions_and_schedule(self):
        request = build_post_request(
            "Hi @[Sample Person](https://www.linkedin.com/in/sample-person/)",
            schedule_at="2026-09-20T09:00:00+04:00",
            document=_document(),
            document_title="Deck",
            now=NOW,
        )
        preview = post_preview(request)

        assert preview["status"] == "preview"
        assert preview["characters"] == len("Hi Sample Person")
        assert preview["mentions"][0]["resolved_against_linkedin"] is False
        assert preview["document"]["title"] == "Deck"
        assert preview["schedule"]["utc"] == "2026-09-20T05:00:00Z"


class TestSchedule:
    def test_offset_is_converted_to_utc(self):
        instant = parse_schedule_at("2026-09-18T12:30:00+04:00", now=NOW)
        assert instant == datetime(2026, 9, 18, 8, 30, tzinfo=UTC)

    def test_naive_times_are_refused(self):
        with pytest.raises(PostValidationError, match="offset"):
            parse_schedule_at("2026-09-18T12:30:00", now=NOW)

    def test_past_and_too_near_times_are_refused(self):
        with pytest.raises(PostValidationError, match="future"):
            parse_schedule_at((NOW + timedelta(minutes=5)).isoformat(), now=NOW)

    def test_times_beyond_the_window_are_refused(self):
        with pytest.raises(PostValidationError, match="window"):
            parse_schedule_at((NOW + timedelta(days=91)).isoformat(), now=NOW)

    def test_seconds_are_refused(self):
        with pytest.raises(PostValidationError, match="whole minute"):
            parse_schedule_at("2026-09-18T12:30:15Z", now=NOW)


class TestScheduleDialogFormatting:
    def test_day_above_twelve_names_the_day_slot(self):
        assert resolve_date_order("17/9/2026", "", None) == ("d", "m", "y")
        assert resolve_date_order("9/17/2026", "", None) == ("m", "d", "y")

    def test_today_window_proves_an_ambiguous_order(self):
        # 3/4/2026 read month-first is March 4, which is today.
        assert resolve_date_order("3/4/2026", "", date(2026, 3, 4)) == (
            "m",
            "d",
            "y",
        )
        assert resolve_date_order("3/4/2026", "", date(2026, 4, 3)) == (
            "d",
            "m",
            "y",
        )

    def test_unprovable_order_refuses_a_date_that_would_reverse(self):
        value, order = format_schedule_date(2026, 10, 5, "3/4/2026", "", None)
        assert (value, order) == (None, None)

    def test_formatting_copies_the_prefill_separator_and_order(self):
        value, order = format_schedule_date(2026, 10, 5, "17.9.2026", "", None)
        assert value == "5.10.2026"
        assert order == ("d", "m", "y")
        assert date_matches("05.10.2026", order, 2026, 10, 5)
        assert not date_matches("10.05.2026", order, 2026, 10, 5)

    def test_time_follows_the_prefill_clock(self):
        assert format_schedule_time(14, 5, "2:00 PM") == "2:05 PM"
        assert format_schedule_time(0, 30, "2:00 PM") == "12:30 AM"
        assert format_schedule_time(14, 5, "14:00") == "14:05"
        assert time_matches("2:05 PM", 14, 5)
        assert not time_matches("2:05 AM", 14, 5)
        assert time_matches("14:05", 14, 5)


class TestPostAsAndEdits:
    @pytest.mark.parametrize(
        "value",
        [
            "https://www.linkedin.com/company/12345/",
            "12345",
            "urn:li:organization:12345",
            "urn:li:fsd_company:12345",
        ],
    )
    def test_numeric_page_forms_are_one_identity(self, value):
        assert parse_post_as(value).key == "company:urn:12345"

    def test_vanity_page_url_is_accepted(self):
        target = parse_post_as("https://www.linkedin.com/company/example-co/")
        assert target.key == "company:/company/example-co/"

    @pytest.mark.parametrize(
        "value", ["https://www.linkedin.com/in/sample-person/", "Example Co", ""]
    )
    def test_a_member_or_a_bare_name_is_not_a_page(self, value):
        with pytest.raises(PostValidationError, match="company page"):
            parse_post_as(value)

    def test_page_posts_must_be_public_and_preview_names_the_actor(self):
        with pytest.raises(PostValidationError, match="public"):
            build_post_request("x", visibility="connections", post_as="12345", now=NOW)
        preview = post_preview(build_post_request("x", post_as="12345", now=NOW))
        assert preview["post_as"] == {
            "kind": "company",
            "target": "12345",
            "resolved_against_linkedin": False,
        }

    @pytest.mark.parametrize(
        "value",
        [
            "https://www.linkedin.com/feed/update/urn:li:activity:1234567890/",
            "https://www.linkedin.com/posts/sample-person_topic-activity-1234567890-AbCd",
        ],
    )
    def test_post_urls_normalize_to_the_update_permalink(self, value):
        assert parse_post_url(value) == (
            "https://www.linkedin.com/feed/update/urn:li:activity:1234567890/"
        )

    @pytest.mark.parametrize(
        "value",
        [
            "http://www.linkedin.com/feed/update/urn:li:activity:1234567890/",
            "https://evil.example/feed/update/urn:li:activity:1234567890/",
            "https://www.linkedin.com/in/sample-person/",
            "https://www.linkedin.com/feed/update/urn:li:fsd_profile:ACoAA/",
        ],
    )
    def test_other_urls_are_not_posts(self, value):
        with pytest.raises(PostValidationError, match="post_url"):
            parse_post_url(value)

    def test_an_edit_needs_a_change_and_published_posts_keep_their_time(self):
        with pytest.raises(PostValidationError, match="Pass text"):
            build_post_edit(None)
        with pytest.raises(PostValidationError, match="rescheduled"):
            build_post_edit(
                "x", schedule_at="2026-09-18T12:30:00Z", allow_schedule=False
            )
        edit = build_post_edit(
            "Hi @[Sample Person](https://www.linkedin.com/in/sample-person/)"
        )
        assert edit.rendered_text == "Hi Sample Person"
        assert len(edit.mentions) == 1
