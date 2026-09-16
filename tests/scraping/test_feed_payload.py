"""Tests for feed permalink recognition across DOM anchors and SDUI payloads."""

from types import SimpleNamespace

from linkedin_mcp_server.scraping.feed_payload import (
    append_captured_post_permalinks,
    build_feed_references,
    is_post_listing_page,
    is_post_listing_response,
)
from linkedin_mcp_server.scraping.link_metadata import Reference


class TestBuildFeedReferences:
    """Tests for build_feed_references SDUI-capture / DOM-anchor merging."""

    def test_sdui_urls_become_relative_feed_post_references(self):
        captured = [
            "https://www.linkedin.com/posts/alice_some-slug-ugcPost-1-xx",
            "https://www.linkedin.com/posts/bob_other-post-share-2-yy",
        ]
        refs = build_feed_references([], captured)
        assert refs == [
            {
                "kind": "feed_post",
                "url": "/posts/alice_some-slug-ugcPost-1-xx",
                "context": "feed",
            },
            {
                "kind": "feed_post",
                "url": "/posts/bob_other-post-share-2-yy",
                "context": "feed",
            },
        ]

    def test_duplicate_sdui_urls_are_deduped(self):
        captured = [
            "https://www.linkedin.com/posts/alice_x-ugcPost-1-xx",
            "https://www.linkedin.com/posts/alice_x-ugcPost-1-xx",
        ]
        refs = build_feed_references([], captured)
        assert len(refs) == 1
        assert refs[0]["url"] == "/posts/alice_x-ugcPost-1-xx"

    def test_dom_anchor_feed_update_passes_through(self):
        # DOM anchors that classify_link recognises as feed_post survive
        # the merge alongside SDUI captures.
        raw_anchors = [
            {
                "href": "https://www.linkedin.com/feed/update/urn:li:activity:1234567890/",
                "text": "View post",
            }
        ]
        refs = build_feed_references(raw_anchors, [])
        assert any(
            r["url"] == "/feed/update/urn:li:activity:1234567890/"
            and r["kind"] == "feed_post"
            for r in refs
        )

    def test_non_posts_paths_in_sdui_capture_are_skipped(self):
        # Defensive: only /posts/<slug> shapes count for SDUI append.
        captured = [
            "https://www.linkedin.com/in/someuser/",
            "https://www.linkedin.com/posts/alice_x-ugcPost-1-xx",
        ]
        refs = build_feed_references([], captured)
        assert [r["url"] for r in refs] == ["/posts/alice_x-ugcPost-1-xx"]

    def test_cap_matches_num_posts_ceiling(self):
        captured = [
            f"https://www.linkedin.com/posts/p{i}-ugcPost-{i}-xx" for i in range(60)
        ]
        refs = build_feed_references([], captured)
        # Cap is 50, mirroring _REFERENCE_CAPS["feed"] / num_posts <= 50.
        assert len(refs) == 50

    def test_non_feed_post_dom_anchors_are_filtered(self):
        # Sidebar profile / company / external anchors must not crowd
        # out SDUI permalinks — references["feed"] is feed_post-only.
        raw_anchors = [
            {
                "href": "https://www.linkedin.com/in/sidebar-user/",
                "text": "Sidebar User",
            },
            {
                "href": "https://www.linkedin.com/company/some-corp/",
                "text": "Some Corp",
            },
            {
                "href": "https://example.com/external/",
                "text": "External Link",
            },
        ]
        refs = build_feed_references(raw_anchors, [])
        assert refs == []

    def test_feed_post_dom_anchors_coexist_with_sdui_captures(self):
        # The two sources fold into the same feed_post kind without
        # collapsing across URL shapes pointing at the same post.
        raw_anchors = [
            {
                "href": "https://www.linkedin.com/feed/update/urn:li:activity:111/",
                "text": "View post",
            }
        ]
        captured = ["https://www.linkedin.com/posts/alice_x-ugcPost-1-xx"]
        refs = build_feed_references(raw_anchors, captured)
        urls = [r["url"] for r in refs]
        kinds = {r["kind"] for r in refs}
        assert urls == [
            "/feed/update/urn:li:activity:111/",
            "/posts/alice_x-ugcPost-1-xx",
        ]
        assert kinds == {"feed_post"}


class TestIsPostListingPage:
    """Tests for is_post_listing_page URL matching (issue #788)."""

    def test_recent_activity_path_matches(self):
        assert is_post_listing_page(
            "https://www.linkedin.com/in/billgates/recent-activity/all/"
        )

    def test_company_posts_path_matches(self):
        assert is_post_listing_page("https://www.linkedin.com/company/microsoft/posts/")

    def test_company_posts_path_with_query_string_matches(self):
        assert is_post_listing_page(
            "https://www.linkedin.com/company/microsoft/posts/?viewAsMember=true"
        )

    def test_plain_profile_path_does_not_match(self):
        assert not is_post_listing_page("https://www.linkedin.com/in/billgates/")

    def test_company_about_path_does_not_match(self):
        assert not is_post_listing_page(
            "https://www.linkedin.com/company/microsoft/about/"
        )


class TestIsPostListingResponse:
    """Tests for is_post_listing_response content-type filtering."""

    @staticmethod
    def _response(content_type: str) -> SimpleNamespace:
        return SimpleNamespace(headers={"content-type": content_type})

    def test_json_response_is_capturable(self):
        assert is_post_listing_response(self._response("application/json"))

    def test_html_response_is_capturable(self):
        assert is_post_listing_response(self._response("text/html; charset=utf-8"))

    def test_missing_content_type_is_capturable(self):
        assert is_post_listing_response(SimpleNamespace(headers={}))

    def test_image_response_is_skipped(self):
        assert not is_post_listing_response(self._response("image/png"))

    def test_video_response_is_skipped(self):
        assert not is_post_listing_response(self._response("video/mp4"))

    def test_css_response_is_skipped(self):
        assert not is_post_listing_response(self._response("text/css"))

    def test_header_lookup_failure_defaults_to_capturable(self):
        class ExplodingHeaders:
            def get(self, *args, **kwargs):
                raise RuntimeError("boom")

        assert is_post_listing_response(SimpleNamespace(headers=ExplodingHeaders()))


class TestAppendCapturedPostPermalinks:
    """Tests for append_captured_post_permalinks (shared feed/posts merge)."""

    def test_appends_new_permalink_with_given_context(self):
        refs = append_captured_post_permalinks(
            [],
            ["https://www.linkedin.com/posts/idsa_slug-activity-1-xx"],
            context="posts",
        )
        assert refs == [
            {
                "kind": "feed_post",
                "url": "/posts/idsa_slug-activity-1-xx",
                "context": "posts",
            }
        ]

    def test_skips_url_already_present(self):
        existing: list[Reference] = [
            {"kind": "company", "url": "/posts/idsa_slug-activity-1-xx"},
        ]
        refs = append_captured_post_permalinks(
            existing,
            ["https://www.linkedin.com/posts/idsa_slug-activity-1-xx"],
            context="posts",
        )
        assert refs == existing

    def test_skips_non_posts_paths(self):
        refs = append_captured_post_permalinks(
            [], ["https://www.linkedin.com/company/idsa/"], context="posts"
        )
        assert refs == []

    def test_does_not_apply_a_cap_itself(self):
        # Capping is the caller's job (see build_feed_references and
        # SectionCapture._extract_loaded_section), so a burst of captures
        # larger than any known section cap must all survive the merge.
        captured = [
            f"https://www.linkedin.com/posts/p{i}-ugcPost-{i}-xx" for i in range(60)
        ]
        refs = append_captured_post_permalinks([], captured, context="posts")
        assert len(refs) == 60
