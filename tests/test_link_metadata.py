"""Tests for compact LinkedIn reference extraction helpers."""

from urllib.parse import quote

from linkedin_mcp_server.scraping.fields import COMPANY_SECTIONS, PERSON_SECTIONS
from linkedin_mcp_server.scraping.link_metadata import (
    _REFERENCE_CAPS,
    RawImage,
    RawReference,
    build_image_references,
    build_references,
    classify_link,
    dedupe_references,
    normalize_url,
)


class TestBuildReferences:
    def test_canonicalizes_and_types_linkedin_urls(self):
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/in/williamhgates?miniProfileUrn=123",
                    "text": "Bill Gates",
                    "heading": "Featured",
                },
                {
                    "href": "https://www.linkedin.com/company/gates-foundation/posts/",
                    "text": "Gates Foundation",
                    "heading": "Experience",
                },
                {
                    "href": "https://www.linkedin.com/pulse/phone-call-saves-lives-bill-gates-yspvc?trackingId=123",
                    "text": "A phone call that saves lives",
                },
            ],
            "main_profile",
        )

        assert references == [
            {
                "kind": "person",
                "url": "/in/williamhgates/",
                "text": "Bill Gates",
                "context": "featured",
            },
            {
                "kind": "company",
                "url": "/company/gates-foundation/",
                "text": "Gates Foundation",
                "context": "experience",
            },
            {
                "kind": "article",
                "url": "/pulse/phone-call-saves-lives-bill-gates-yspvc/",
                "text": "A phone call that saves lives",
                "context": "top card",
            },
        ]

    def test_preserves_person_slug_named_details(self):
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/in/details/",
                    "text": "Details Person",
                }
            ],
            "main_profile",
        )

        assert references == [
            {
                "kind": "person",
                "url": "/in/details/",
                "text": "Details Person",
                "context": "top card",
            }
        ]

    def test_drops_person_details_subpage(self):
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/in/williamhgates/details/experience/",
                    "text": "Bill Gates",
                }
            ],
            "main_profile",
        )

        assert references == []

    def test_unwraps_redirect_and_drops_junk(self):
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/redir/redirect/?url=https%3A%2F%2Fgatesnot.es%2Ftgn&urlhash=abc",
                    "text": "Gates Notes",
                },
                {
                    "href": "blob:https://www.linkedin.com/123",
                    "text": "Video",
                },
                {
                    "href": "#caret-small",
                    "text": "",
                },
                {
                    "href": "https://www.linkedin.com/help/linkedin/",
                    "text": "Questions?",
                },
            ],
            "posts",
        )

        assert references == [
            {
                "kind": "external",
                "url": "https://gatesnot.es/tgn",
                "text": "Gates Notes",
                "context": "post attachment",
            }
        ]

    def test_drops_non_http_external_schemes(self):
        references = build_references(
            [
                {
                    "href": "data:text/html,<p>hello</p>",
                    "text": "Inline payload",
                },
                {
                    "href": "ftp://example.com/report.csv",
                    "text": "FTP report",
                },
                {
                    "href": "https://example.com/report.csv",
                    "text": "HTTPS report",
                },
            ],
            "posts",
        )

        assert references == [
            {
                "kind": "external",
                "url": "https://example.com/report.csv",
                "text": "HTTPS report",
                "context": "post attachment",
            }
        ]

    def test_dedupes_external_tracking_variants(self):
        references = build_references(
            [
                {
                    "href": "https://example.com/report?utm_source=linkedin",
                    "text": "Report",
                },
                {
                    "href": "https://example.com/report?utm_source=share",
                    "text": "Detailed annual report",
                },
            ],
            "posts",
        )

        assert references == [
            {
                "kind": "external",
                "url": "https://example.com/report",
                "text": "Detailed annual report",
                "context": "post attachment",
            }
        ]

    def test_prefers_cleaner_duplicate_label(self):
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/newsletters/gates-notes-123/",
                    "text": "View my newsletter",
                    "aria_label": "Gates Notes",
                },
                {
                    "href": "https://www.linkedin.com/newsletters/gates-notes-123/",
                    "text": "Gates Notes Gates Notes",
                },
            ],
            "posts",
        )

        assert references == [
            {
                "kind": "newsletter",
                "url": "/newsletters/gates-notes-123/",
                "text": "Gates Notes",
                "context": "post attachment",
            }
        ]

    def test_normalize_url_unwraps_nested_redirects_within_cap(self):
        target = "https://example.com/report"
        nested = "https://www.linkedin.com/redir/redirect/?url=" + quote(
            "https://www.linkedin.com/redir/redirect/?url=" + quote(target, safe=""),
            safe="",
        )

        assert normalize_url(nested) == target

    def test_normalize_url_drops_redirect_chain_beyond_cap(self):
        target = "https://example.com/report"
        href = target
        for _ in range(7):
            href = "https://www.linkedin.com/redir/redirect/?url=" + quote(
                href, safe=""
            )

        assert normalize_url(href) is None

    def test_prefers_shorter_clean_label_over_merged_visible_text(self):
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/pulse/test-post?trackingId=123",
                    "text": "Gates Notes Gates Notes A phone call that saves lives Bill Gates",
                    "aria_label": "Open article: A phone call that saves lives by Bill Gates • 3 min read",
                }
            ],
            "posts",
        )

        assert references == [
            {
                "kind": "article",
                "url": "/pulse/test-post/",
                "text": "A phone call that saves lives",
                "context": "post attachment",
            }
        ]

    def test_rejects_single_character_labels(self):
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/in/williamhgates/",
                    "text": "1",
                    "aria_label": "Bill Gates",
                }
            ],
            "main_profile",
        )

        assert references == [
            {
                "kind": "person",
                "url": "/in/williamhgates/",
                "text": "Bill Gates",
                "context": "top card",
            }
        ]

    def test_keeps_cyrillic_only_names(self):
        """Regression: a non-Latin (Cyrillic) name must survive label
        cleaning — the alphanumeric guard uses a Unicode word check, not
        [A-Za-z0-9], so search results from RU/BY/other locales keep
        their person references instead of being dropped."""
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/in/margo-yunanova/",
                    "text": "Маргарита Юнанова",
                }
            ],
            "search_results",
        )

        assert references == [
            {
                "kind": "person",
                "url": "/in/margo-yunanova/",
                "text": "Маргарита Юнанова",
                "context": "search result",
            }
        ]

    def test_rejects_punctuation_only_labels_across_scripts(self):
        """The Unicode word guard must still reject pure punctuation and
        symbols (no letter or digit in any script)."""
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/in/williamhgates/",
                    "text": "—·—",
                    "aria_label": "Bill Gates",
                }
            ],
            "main_profile",
        )

        assert references == [
            {
                "kind": "person",
                "url": "/in/williamhgates/",
                "text": "Bill Gates",
                "context": "top card",
            }
        ]

    def test_rejects_invisible_hangul_filler_labels(self):
        """Hangul fillers carry the Unicode word property but render as
        nothing, so a label made only of them must not shadow the
        aria-label fallback with invisible text."""
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/in/williamhgates/",
                    "text": "\u115f\u1160\u3164\uffa0",
                    "aria_label": "Bill Gates",
                }
            ],
            "main_profile",
        )

        assert references == [
            {
                "kind": "person",
                "url": "/in/williamhgates/",
                "text": "Bill Gates",
                "context": "top card",
            }
        ]

    def test_preserves_words_starting_with_view(self):
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/company/viewpoint-economics/",
                    "text": "Viewpoint Economics",
                }
            ],
            "about",
        )

        assert references == [
            {
                "kind": "company",
                "url": "/company/viewpoint-economics/",
                "text": "Viewpoint Economics",
                "context": "top card",
            }
        ]

    def test_prefers_company_post_context_for_feed_posts(self):
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/feed/update/urn:li:activity:123/",
                    "text": "Original company post",
                    "in_article": True,
                }
            ],
            "posts",
        )

        assert references == [
            {
                "kind": "feed_post",
                "url": "/feed/update/urn:li:activity:123/",
                "text": "Original company post",
                "context": "company post",
            }
        ]

    def test_drops_social_proof_company_labels(self):
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/company/gates-foundation/",
                    "text": "Falguni & 8 other connections follow this page",
                },
                {
                    "href": "https://www.linkedin.com/company/gates-foundation/",
                    "text": "Gates Foundation",
                },
            ],
            "about",
        )

        assert references == [
            {
                "kind": "company",
                "url": "/company/gates-foundation/",
                "text": "Gates Foundation",
                "context": "top card",
            }
        ]

    def test_drops_nav_and_footer_anchors(self):
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/in/williamhgates/",
                    "text": "Bill Gates",
                    "in_nav": True,
                },
                {
                    "href": "https://www.linkedin.com/company/gates-foundation/",
                    "text": "Gates Foundation",
                    "in_footer": True,
                },
            ],
            "main_profile",
        )

        assert references == []

    def test_caps_results_per_section(self):
        raw: list[RawReference] = [
            {
                "href": f"https://www.linkedin.com/company/test-{idx}/",
                "text": f"Company {idx}",
            }
            for idx in range(20)
        ]

        references = build_references(raw, "about")

        assert len(references) == 12
        assert references[0]["url"] == "/company/test-0/"
        assert references[-1]["url"] == "/company/test-11/"

    def test_search_results_cap_can_be_disabled(self):
        raw: list[RawReference] = [
            {
                "href": f"https://www.linkedin.com/jobs/view/{idx}/",
                "text": f"Job {idx}",
            }
            for idx in range(20)
        ]

        capped = build_references(raw, "search_results")
        uncapped = build_references(raw, "search_results", apply_cap=False)

        assert len(capped) == 15
        assert capped[-1]["url"] == "/jobs/view/14/"
        assert len(uncapped) == 20
        assert uncapped[-1]["url"] == "/jobs/view/19/"

    def test_caps_jobs_section_more_tightly(self):
        raw: list[RawReference] = [
            {
                "href": f"https://www.linkedin.com/jobs/view/{idx}/",
                "text": f"Job {idx}",
            }
            for idx in range(20)
        ]

        references = build_references(raw, "jobs")

        assert len(references) == 8
        assert references[0]["url"] == "/jobs/view/0/"
        assert references[-1]["url"] == "/jobs/view/7/"

    def test_uses_default_cap_for_unknown_section(self):
        raw: list[RawReference] = [
            {
                "href": f"https://www.linkedin.com/company/test-{idx}/",
                "text": f"Company {idx}",
            }
            for idx in range(20)
        ]

        references = build_references(raw, "unknown_section")

        assert len(references) == 12

    def test_prefers_richer_duplicate_text(self):
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/jobs/view/12345/",
                    "text": "Job",
                },
                {
                    "href": "https://www.linkedin.com/jobs/view/12345/",
                    "text": "Senior Software Engineer",
                },
            ],
            "search_results",
        )

        assert references == [
            {
                "kind": "job",
                "url": "/jobs/view/12345/",
                "text": "Senior Software Engineer",
                "context": "job result",
            }
        ]

    def test_uses_search_result_contexts(self):
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/jobs/view/12345/",
                    "text": "Senior Engineer",
                },
                {
                    "href": "https://www.linkedin.com/in/stickerdaniel/",
                    "text": "Daniel Sticker",
                },
            ],
            "search_results",
        )

        assert references == [
            {
                "kind": "job",
                "url": "/jobs/view/12345/",
                "text": "Senior Engineer",
                "context": "job result",
            },
            {
                "kind": "person",
                "url": "/in/stickerdaniel/",
                "text": "Daniel Sticker",
                "context": "search result",
            },
        ]

    def test_uses_job_posting_context_for_job_pages(self):
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/company/acme/",
                    "text": "Acme",
                }
            ],
            "job_posting",
        )

        assert references == [
            {
                "kind": "company",
                "url": "/company/acme/",
                "text": "Acme",
                "context": "job posting",
            }
        ]

    def test_uses_person_detail_section_contexts(self):
        """certifications, skills and projects joined PERSON_SECTIONS after
        this table was written, so their references carried no context while
        every sibling detail section named itself."""
        for section in ("certifications", "skills", "projects"):
            references = build_references(
                [
                    {
                        "href": "https://www.linkedin.com/company/aws/",
                        "text": "Amazon Web Services",
                    }
                ],
                section,
            )

            assert references == [
                {
                    "kind": "company",
                    "url": "/company/aws/",
                    "text": "Amazon Web Services",
                    "context": section,
                }
            ]

    def test_uses_employees_context_for_company_people_pages(self):
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/in/stickerdaniel/",
                    "text": "Daniel Sticker",
                }
            ],
            "employees",
        )

        assert references == [
            {
                "kind": "person",
                "url": "/in/stickerdaniel/",
                "text": "Daniel Sticker",
                "context": "employees",
            }
        ]

    def test_names_the_context_for_jobs_saved_jobs_and_feed(self):
        """The structural guard below only proves a context key exists, so it
        survives a wrong label. These are the values themselves."""
        raw: list[RawReference] = [
            {
                "href": "https://www.linkedin.com/jobs/view/123/",
                "text": "Senior Engineer",
            }
        ]

        contexts = {
            section: build_references(raw, section)[0]["context"]
            for section in ("jobs", "saved_jobs", "feed")
        }

        assert contexts == {
            "jobs": "jobs",
            "saved_jobs": "saved jobs",
            "feed": "feed",
        }

    def test_every_scraped_section_gives_its_references_a_context(self):
        """Nothing tied the context table to the section tables, which is how
        seven sections have now reached main without an entry. A context-less
        reference also scores below every duplicate that has one, so it loses
        cross-page dedupe ties it should win.

        Section names are declared in three separate places, and `saved_jobs`
        is declared in none of them -- it exists only as a literal in the
        extractor -- so it is named here explicitly."""
        raw: list[RawReference] = [
            {
                "href": "https://www.linkedin.com/company/aws/",
                "text": "Amazon Web Services",
            }
        ]

        sections = (
            set(PERSON_SECTIONS)
            | set(COMPANY_SECTIONS)
            | set(_REFERENCE_CAPS)
            | {"saved_jobs"}
        )

        missing = sorted(
            section
            for section in sections
            if "context" not in build_references(raw, section)[0]
        )

        assert missing == []

    def test_does_not_treat_lookalike_domains_as_linkedin(self):
        references = build_references(
            [
                {
                    "href": "https://www.notlinkedin.com/company/fake/about/",
                    "text": "Fake Company",
                }
            ],
            "about",
        )

        assert references == [
            {
                "kind": "external",
                "url": "https://www.notlinkedin.com/company/fake/about/",
                "text": "Fake Company",
                "context": "top card",
            }
        ]

    def test_keeps_company_about_routes(self):
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/company/legalzoom/about/",
                    "text": "LegalZoom",
                }
            ],
            "about",
        )

        assert references == [
            {
                "kind": "company",
                "url": "/company/legalzoom/",
                "text": "LegalZoom",
                "context": "top card",
            }
        ]

    def test_cross_page_dedupe_keeps_better_reference(self):
        references = dedupe_references(
            [
                {
                    "kind": "job",
                    "url": "/jobs/view/123/",
                    "text": "Job",
                },
                {
                    "kind": "job",
                    "url": "/jobs/view/123/",
                    "text": "Senior Software Engineer",
                    "context": "job result",
                },
            ]
        )

        assert references == [
            {
                "kind": "job",
                "url": "/jobs/view/123/",
                "text": "Senior Software Engineer",
                "context": "job result",
            }
        ]

    def test_company_urn_single_id_anchor(self):
        """Anthropic-style: single id in the currentCompany list."""
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/search/results/people/"
                    "?currentCompany=%5B%2274126343%22%5D"
                    "&origin=COMPANY_PAGE_CANNED_SEARCH",
                    "text": "501-1K employees",
                }
            ],
            "about",
        )

        assert references == [
            {
                "kind": "company_urn",
                "url": "/search/results/people/?currentCompany=%5B%2274126343%22%5D",
                "value": "74126343",
                "context": "top card",
            }
        ]

    def test_company_urn_multi_id_anchor_uses_first_id(self):
        """SAP-style: parent + subsidiaries; the first id is the parent company."""
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/search/results/people/"
                    "?currentCompany=%5B%221115%22%2C%222573558%22%2C%222818%22%5D"
                    "&origin=COMPANY_PAGE_CANNED_SEARCH",
                    "text": "143,150 associated members",
                }
            ],
            "about",
        )

        assert references == [
            {
                "kind": "company_urn",
                "url": "/search/results/people/?currentCompany=%5B%221115%22%5D",
                "value": "1115",
                "context": "top card",
            }
        ]

    def test_company_urn_suppresses_anchor_text(self):
        """Anchor text like '10K+ employees' is not user-meaningful for a URN
        reference; callers should key off ``value``."""
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/search/results/people/"
                    "?currentCompany=%5B%221115%22%5D",
                    "text": "10K+ employees",
                }
            ],
            "about",
        )

        assert len(references) == 1
        assert references[0]["kind"] == "company_urn"
        assert references[0]["value"] == "1115"
        assert "text" not in references[0]

    def test_company_urn_accepts_unquoted_json_integers(self):
        """Defensive: LinkedIn currently serialises ids as quoted strings,
        but plain JSON integers are also valid and should still classify."""
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/search/results/people/"
                    "?currentCompany=%5B1115%5D",
                    "text": "10K+ employees",
                }
            ],
            "about",
        )

        assert len(references) == 1
        assert references[0]["kind"] == "company_urn"
        assert references[0]["value"] == "1115"

    def test_company_urn_lowercase_percent_escapes(self):
        """``parse_qs`` decodes percent-escapes regardless of case, so
        lowercase variants must still classify and extract the same id."""
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/search/results/people/"
                    "?currentCompany=%5b%221115%22%5d",
                    "text": "10K+ employees",
                }
            ],
            "about",
        )

        assert len(references) == 1
        assert references[0]["kind"] == "company_urn"
        assert references[0]["value"] == "1115"

    def test_plain_people_search_still_dropped(self):
        """A people-search href without ``currentCompany`` is page chrome
        and stays excluded — preserves existing behaviour."""
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/search/results/people/"
                    "?keywords=engineer",
                    "text": "engineer",
                }
            ],
            "about",
        )

        assert references == []


class TestClassifyLink:
    def test_a_slugged_job_url_keeps_its_id(self):
        """LinkedIn serves a job under a bare id and under a slugged path.

        Both 301 to the same page, so the slugged form is just as real, and
        anchoring the id to the front of the segment dropped it: the link
        vanished from ``references`` entirely.
        """
        assert classify_link(
            "https://www.linkedin.com/jobs/view/senior-ai-engineer-at-acme-1967281839/"
        ) == ("job", "/jobs/view/1967281839/")

    def test_a_title_opening_with_a_number_is_not_the_job_id(self):
        """The quiet half of the same bug, and the worse one.

        A title starting with a year matched the front anchor, so the link
        was kept and pointed at a different job. A dropped reference is
        visibly missing; this one looks like a result.
        """
        assert classify_link(
            "https://www.linkedin.com/jobs/view/2026-software-engineer-at-acme-4252026496/"
        ) == ("job", "/jobs/view/4252026496/")

    def test_unicode_digits_are_not_a_job_id(self):
        """Python's ``\\d`` matches more than JavaScript's does.

        Arabic-Indic digits pass ``\\d`` here and fail it in the two
        JavaScript copies of this pattern, and ``normalize_job_id`` accepts
        only ``[0-9]``. Classifying such a link produces a reference whose
        very next use raises, so the digits are ASCII on purpose.
        """
        assert (
            classify_link(
                "https://www.linkedin.com/jobs/view/\u0645\u0647\u0646\u062f\u0633-"
                "\u0664\u0662\u0665\u0662\u0660\u0662\u0666\u0664\u0669\u0666/"
            )
            is None
        )

    def test_a_bare_job_url_is_unchanged(self):
        assert classify_link("https://www.linkedin.com/jobs/view/1967281839/") == (
            "job",
            "/jobs/view/1967281839/",
        )

    def test_messaging_thread_url(self):
        result = classify_link(
            "https://www.linkedin.com/messaging/thread/2-NjAwMDAyMDEtZWVh/"
        )
        assert result == (
            "conversation",
            "/messaging/thread/2-NjAwMDAyMDEtZWVh/",
        )

    def test_messaging_thread_url_with_query(self):
        result = classify_link(
            "https://www.linkedin.com/messaging/thread/2-abc123/?focusedMsgUrn=xyz"
        )
        assert result == ("conversation", "/messaging/thread/2-abc123/")

    def test_inbox_references_include_threads(self):
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/messaging/thread/2-abc123/",
                    "text": "Tony Chan",
                },
                {
                    "href": "https://www.linkedin.com/messaging/thread/2-def456/",
                    "text": "Paul Jasper",
                },
            ],
            "inbox",
        )
        assert len(references) == 2
        assert references[0]["kind"] == "conversation"
        assert references[0]["url"] == "/messaging/thread/2-abc123/"
        assert references[0]["text"] == "Tony Chan"
        assert references[0]["context"] == "inbox"
        assert references[1]["kind"] == "conversation"
        assert references[1]["url"] == "/messaging/thread/2-def456/"
        assert references[1]["text"] == "Paul Jasper"

    def test_inbox_conversation_without_text_still_captured(self):
        """Conversation references are kept even without a usable text label."""
        references = build_references(
            [
                {
                    "href": "https://www.linkedin.com/messaging/thread/2-xyz/",
                    "text": "",
                },
            ],
            "inbox",
        )
        assert len(references) == 1
        assert references[0]["kind"] == "conversation"
        assert references[0]["url"] == "/messaging/thread/2-xyz/"


# The subject, from a large variant.
_SUBJECT = (
    "https://media.licdn.com/dms/image/v2/C4E03AQHaoqb8h-ev4w"
    "/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0"
)
# Everyone else on the page - post authors, mutual connections, suggestions.
_OTHER_MEMBER = (
    "https://media.licdn.com/dms/image/v2/D5603AQExSlVQUDCavQ"
    "/profile-displayphoto-scale_100_100/B56Zy_XQAyHQAc-/0/1772737086600"
)
_COMPANY_LOGO = (
    "https://media.licdn.com/dms/image/v2/D4D0BAQEE_my7WpYL7g"
    "/company-logo_100_100/B4DZpFeaQeGgAQ-/0/1762102192036/geofoundationai_logo"
)
_COVER = (
    "https://media.licdn.com/dms/image/v2/D4D16AQE71FRdhqtCAA"
    "/profile-displaybackgroundimage-shrink_200_800/B4DZpPiJ_JIMAU-/0/176227094"
)
_POST_IMAGE = (
    "https://media.licdn.com/dms/image/v2/D5622AQH1rhjR5nc6nw"
    "/feedshare-shrink_480/B56Z7XNJhOJoAg-/0/1781727009179"
)
# A company page renders its own logo large and every other company small,
# exactly as a profile does with member photos.
_COMPANY_SUBJECT_LOGO = (
    "https://media.licdn.com/dms/image/v2/D4D0BAQGZ3dq_qonY0w"
    "/company-logo_200_200/B4DZpFeaQeGgAQ-/0/1762102192036/nimbus_logo"
)
_ARTICLE_IMAGE = (
    "https://media.licdn.com/dms/image/v2/D4E10AQFvKaHallTUrw"
    "/articleshare-shrink_800/B4EZy0cQDzJcAQ-/0/1772553831073"
)
_STATIC_ICON = "https://static.licdn.com/aero-v1/sc/h/icon.svg"

_PHOTO = "profile-displayphoto-shrink"
_LOGO = "company-logo"


def _cdn(kind: str, size: int, n: int = 0) -> str:
    """A CDN URL for one kind at one size variant."""
    return f"https://media.licdn.com/dms/image/v2/X{n}/{kind}_{size}_{size}/y/0"


class TestBuildImageReferences:
    """Tests for build_image_references — the page subject's own photo/logo.

    ``innerText`` extraction can't see an ``<img>``, so the subject's photo
    is the one thing on a top card that never reaches the caller otherwise
    (issue #663). Fixture URLs mirror LinkedIn's CDN path grammar, the only
    stable discriminator between a subject and everyone else on the page.
    """

    def test_returns_the_subject_photo(self):
        [ref] = build_image_references([RawImage(src=_SUBJECT, alt="")], "main_profile")
        assert ref == {"kind": "image", "url": _SUBJECT, "context": "profile photo"}

    def test_picks_the_subject_out_of_a_whole_page(self):
        page: list[RawImage] = [
            {"src": _COVER, "alt": "Cover photo"},
            {"src": _OTHER_MEMBER, "alt": "View Austin Cruz's profile"},
            {"src": _COMPANY_LOGO, "alt": ""},
            {"src": _SUBJECT, "alt": ""},
            {"src": _POST_IMAGE, "alt": "View image"},
            {"src": _STATIC_ICON, "alt": ""},
        ]
        assert [r["url"] for r in build_image_references(page, "main_profile")] == [
            _SUBJECT
        ]

    def test_rejects_everything_that_is_not_the_subject(self):
        for src in (_OTHER_MEMBER, _COMPANY_LOGO, _COVER, _POST_IMAGE, _STATIC_ICON):
            assert (
                build_image_references([RawImage(src=src, alt="x")], "main_profile")
                == []
            )

    def test_returns_the_company_logo_on_a_company_page(self):
        [ref] = build_image_references(
            [RawImage(src=_COMPANY_SUBJECT_LOGO, alt="Nimbus Structure GmbH logo")],
            "about",
        )
        assert ref["url"] == _COMPANY_SUBJECT_LOGO
        assert ref["context"] == "company logo"
        assert ref["text"] == "Nimbus Structure GmbH logo"

    def test_rejects_anything_off_the_media_cdn(self):
        for src in (
            "",
            "   ",
            "/relative.png",
            "data:image/gif;base64,R0lGOD",
            "https://example.com/a.jpg",
        ):
            assert build_image_references([RawImage(src=src)], "main_profile") == []

    def test_the_subject_is_whoever_is_largest_not_whoever_clears_a_number(self):
        small: list[RawImage] = [
            {"src": _cdn(_PHOTO, 100, 1)},
            {"src": _cdn(_PHOTO, 128, 2)},
        ]
        assert [r["url"] for r in build_image_references(small, "main_profile")] == [
            _cdn(_PHOTO, 128, 2)
        ]

        big: list[RawImage] = [
            {"src": _cdn(_PHOTO, 200, 1)},
            {"src": _cdn(_PHOTO, 800, 2)},
        ]
        assert [r["url"] for r in build_image_references(big, "main_profile")] == [
            _cdn(_PHOTO, 800, 2)
        ]

    def test_returns_nothing_when_no_candidate_stands_out(self):
        page: list[RawImage] = [{"src": _cdn(_PHOTO, 100, i)} for i in range(8)]
        assert build_image_references(page, "search_results") == []

    def test_a_tie_for_largest_yields_no_subject(self):
        tied: list[RawImage] = [
            {"src": _cdn(_PHOTO, 800, 1)},
            {"src": _cdn(_PHOTO, 800, 2)},
            {"src": _cdn(_PHOTO, 100, 3)},
        ]
        assert build_image_references(tied, "main_profile") == []

    def test_the_same_photo_signed_twice_is_one_candidate(self):
        page: list[RawImage] = [
            {"src": f"{_SUBJECT}?e=1&v=beta&t=first-signature"},
            {"src": f"{_SUBJECT}?e=2&v=beta&t=second-signature"},
            {"src": _OTHER_MEMBER},
        ]
        [ref] = build_image_references(page, "main_profile")
        assert ref["url"].startswith(_SUBJECT)

    def test_a_photo_and_its_own_thumbnail_are_one_candidate(self):
        page: list[RawImage] = [
            {"src": _cdn(_PHOTO, 100, 1)},
            {"src": _cdn(_PHOTO, 800, 1)},
        ]
        assert [r["url"] for r in build_image_references(page, "main_profile")] == [
            _cdn(_PHOTO, 800, 1)
        ]

    def test_one_subject_per_kind_at_most(self):
        page: list[RawImage] = [
            {"src": _SUBJECT},
            {"src": _OTHER_MEMBER},
            {"src": _COMPANY_SUBJECT_LOGO},
            {"src": _COMPANY_LOGO},
        ]
        refs = build_image_references(page, "main_profile")
        assert [r["context"] for r in refs] == ["profile photo", "company logo"]

    def test_a_lone_thumbnail_is_not_a_subject(self):
        assert build_image_references([{"src": _cdn(_LOGO, 100)}], "main_profile") == []
        assert build_image_references([{"src": _cdn(_PHOTO, 100)}], "about") == []
        assert build_image_references([{"src": _cdn(_LOGO, 200)}], "about")

    def test_each_kind_is_judged_separately(self):
        member_page: list[RawImage] = [
            {"src": _cdn(_PHOTO, 800, 1)},
            {"src": _cdn(_PHOTO, 100, 2)},
            {"src": _cdn(_LOGO, 100, 3)},
        ]
        assert [
            r["context"] for r in build_image_references(member_page, "main_profile")
        ] == ["profile photo"]

        company_page: list[RawImage] = [
            {"src": _cdn(_PHOTO, 100, 1)},
            {"src": _cdn(_LOGO, 200, 2)},
            {"src": _cdn(_LOGO, 100, 3)},
        ]
        assert [
            r["context"] for r in build_image_references(company_page, "about")
        ] == ["company logo"]

    def test_does_not_depend_on_rendered_size(self):
        assert build_image_references(
            [{"src": _SUBJECT, "width": 0, "height": 0}], "main_profile"
        )

    def test_keeps_alt_when_present_and_omits_it_when_blank(self):
        [with_alt] = build_image_references(
            [{"src": _SUBJECT, "alt": "Konstantin Gerner"}], "main_profile"
        )
        assert with_alt["text"] == "Konstantin Gerner"
        [blank] = build_image_references(
            [RawImage(src=_SUBJECT, alt="   ")], "main_profile"
        )
        assert "text" not in blank

    def test_the_same_image_rendered_twice_is_one_reference(self):
        page: list[RawImage] = [
            {"src": _cdn(_PHOTO, 800)},
            {"src": _cdn(_PHOTO, 800)},
            {"src": _cdn(_PHOTO, 100, 2)},
        ]
        assert [r["url"] for r in build_image_references(page, "main_profile")] == [
            _cdn(_PHOTO, 800)
        ]

    def test_a_crowd_at_one_size_is_not_a_pile_of_subjects(self):
        page: list[RawImage] = [{"src": _cdn(_PHOTO, 800, i)} for i in range(10)]
        page.append({"src": _cdn(_PHOTO, 100, 99)})
        assert build_image_references(page, "main_profile") == []

    def test_context_names_the_kind_not_the_section(self):
        [photo] = build_image_references([RawImage(src=_SUBJECT)], "experience")
        assert photo["context"] == "profile photo"

        [logo] = build_image_references([RawImage(src=_COMPANY_SUBJECT_LOGO)], "posts")
        assert logo["context"] == "company logo"
