"""Feed permalink recognition across DOM anchors and SDUI payloads."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import re

from linkedin_mcp_server.scraping.link_metadata import (
    Reference,
    build_references,
    dedupe_references,
)

_FEED_RSC_MARKER = "sduiid=com.linkedin.sdui.pagers.feed.mainFeed"
# Matches a LinkedIn post permalink in either plain or JSON-escaped form
# (the initial /feed/ HTML embeds the RSC flight data with \u002f for slashes,
# while paginated responses use plain slashes). Captures the slug portion so
# we can rebuild a canonical URL regardless of the source encoding.
POST_SLUG_URL_RE = re.compile(
    r"linkedin\.com(?:\\u002[fF]|/)posts(?:\\u002[fF]|/)"
    r"(?P<slug>[A-Za-z0-9_-]+?-(?:ugcPost|activity|share)-\d+-[A-Za-z0-9_-]+)"
)
_FEED_DOCUMENT_URLS = {
    "https://www.linkedin.com/feed",
    "https://www.linkedin.com/feed/",
}

# Content types that never carry the JSON/RSC payloads POST_SLUG_URL_RE
# matches against. Company/person posts pages don't expose a stable SDUI
# marker analogous to _FEED_RSC_MARKER (unverified whether one exists), so
# is_post_listing_response casts a wider net than is_feed_payload_response
# and relies on this prefix list plus the regex itself to stay cheap and
# correct rather than on a guessed marker string.
_NON_PAYLOAD_CONTENT_TYPE_PREFIXES = (
    "image/",
    "video/",
    "audio/",
    "font/",
    "text/css",
)


def is_feed_payload_response(url: str) -> bool:
    """True if the response URL is one that carries `postSlugUrl` fields."""
    if _FEED_RSC_MARKER in url:
        return True
    return url.split("?", 1)[0] in _FEED_DOCUMENT_URLS


def is_post_listing_page(url: str) -> bool:
    """True for pages whose posts render without a DOM permalink anchor.

    Company posts pages (``/company/<slug>/posts/``), a person's activity
    feed (``/recent-activity/...``), and a hashtag feed
    (``/feed/hashtag/<tag>/``) all lazy-load posts the same way the main
    feed does, and LinkedIn does not render a real ``<a href>`` for the
    individual post on any of them — only the main feed has a dedicated
    DOM-anchor path (``feed_post`` via ``/feed/update/<urn>/``). Matched on
    the parsed path since the url can carry a query string
    (``?viewAsMember=true``) that a raw suffix check would miss.

    The hashtag feed is included on the strength of sharing the same
    feed-rendering surface as the other three, not on a confirmed live
    network capture (this fork never signs in to LinkedIn) — flag for
    live verification if it turns out to carry no such payload.
    """
    path = urlparse(url).path
    if "/recent-activity/" in path or "/feed/hashtag/" in path:
        return True
    return "/company/" in path and path.rstrip("/").endswith("/posts")


def is_post_listing_response(resp: Any) -> bool:
    """True if a response on a posts-listing page is worth scanning for permalinks.

    Company/person posts pages carry permalinks through the initial HTML
    document and through paginated GraphQL responses rather than one fixed
    SDUI marker like the home feed (see ``is_feed_payload_response``), so
    every response is scanned unless its content-type rules it out —
    obvious binary media LinkedIn never embeds a permalink payload in.
    """
    try:
        content_type = resp.headers.get("content-type", "")
    except Exception:
        return True
    content_type = content_type.split(";", 1)[0].strip().lower()
    return not content_type.startswith(_NON_PAYLOAD_CONTENT_TYPE_PREFIXES)


def append_captured_post_permalinks(
    refs: list[Reference],
    captured_urls: list[str],
    *,
    context: str,
) -> list[Reference]:
    """Append ``/posts/<slug>`` permalinks captured from network responses.

    Shared by the feed and by any posts-listing page (company posts, a
    person's activity feed): both render some individual posts with no DOM
    ``<a href>`` permalink, and both capture the real one from a network
    response instead (see ``is_post_listing_page``). Skips any capture
    already present as an exact URL match. The two shapes that can point at
    the same underlying post (a DOM-derived reference vs. a captured
    ``/posts/<slug>`` permalink) will *not* collapse — ``dedupe_references``
    matches strings, not URNs. Both are valid LinkedIn permalinks; URN-based
    equivalence is left to the consumer. Does not apply a cap itself — the
    caller deduplicates and caps after merging, so DOM-derived and captured
    permalinks compete fairly for the section's available slots.
    """
    existing = {r["url"] for r in refs}
    merged = list(refs)
    for sdui_url in captured_urls:
        # AGENTS.md mandates relative paths for LinkedIn references.
        # The capture carries fully-qualified URLs like
        # https://www.linkedin.com/posts/<slug>; strip the host so the
        # relative-path convention holds. ``classify_link`` does not
        # currently route ``/posts/<slug>`` paths to any kind, so we
        # bypass it for this fallback append.
        parsed = urlparse(sdui_url)
        if not parsed.path.startswith("/posts/"):
            continue
        relative = parsed.path
        if relative in existing:
            continue
        merged.append({"kind": "feed_post", "url": relative, "context": context})
        existing.add(relative)
    return merged


def build_feed_references(
    raw_references: list[Any],
    captured_urls: list[str],
) -> list[Reference]:
    """Compose feed references from DOM anchors + SDUI captures.

    The feed page renders many anchors that are not post permalinks:
    sidebar widgets, profile cards, employer logos, etc. Mixing them
    into ``references["feed"]`` blurs the contract and competes with
    SDUI permalinks for the per-section cap. We keep only the
    ``feed_post`` slice from the DOM:

    - DOM anchors → ``feed_post`` entries with ``/feed/update/<urn>/``
      URLs (whatever ``classify_link`` recognises).
    - SDUI captures → ``feed_post`` entries with ``/posts/<slug>`` URLs
      for permalinks that the DOM does not surface as an anchor.
    """
    refs = [
        ref
        for ref in build_references(raw_references, "feed")
        if ref["kind"] == "feed_post"
    ]
    refs = append_captured_post_permalinks(refs, captured_urls, context="feed")
    # Cap kept in sync with _REFERENCE_CAPS["feed"] in link_metadata.py;
    # changing one without the other will drop or duplicate entries
    # silently. Matches get_feed's num_posts ceiling (Field(ge=1, le=50)).
    return dedupe_references(refs, cap=50)
