# tests/test_comments_dom.py
"""Browser-DOM tests for the comment read and write programs.

The unit suite never executes the JavaScript in ``scraping/comments.py``, so
these cases run the real workflows against synthetic markup in headless
chromium, served from a LinkedIn origin because the programs resolve author
links and check the post route against ``window.location``.

The fixture is a claim about the algorithm, not about LinkedIn's markup: it
carries the one attribute issue 828 observed live (``data-id`` holding the
comment URN) and otherwise only generic structure. Every flow runs under three
label sets (English, German, opaque tokens with no verb in them); a decision
that changes between them read a word, which the Scraping Rules forbid.

All names and texts are synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import asyncio

import pytest
from patchright.async_api import Page, async_playwright

from linkedin_mcp_server.scraping import comments as comments_module
from linkedin_mcp_server.scraping.comments import CommentScraper
from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.session import ScrapingSession

pytestmark = [
    pytest.mark.browser_dom,
    pytest.mark.xdist_group("browser_runtime"),
]

POST_URN = "urn:li:activity:7300000000000000000"
POST_URL = f"https://www.linkedin.com/feed/update/{POST_URN}/"
THREAD = "activity:7300000000000000000"
FIRST = "7300000000000000101"
SECOND = "7300000000000000102"
NESTED = "7300000000000000201"


@dataclass(frozen=True, slots=True)
class Labels:
    locale: str
    like: str
    reply: str
    comment: str
    repost: str
    post: str
    load: str
    replies: str


LOCALES = [
    Labels(
        "en",
        "Like",
        "Reply",
        "Comment",
        "Repost",
        "Post",
        "Load more comments",
        "Load previous replies",
    ),
    Labels(
        "de",
        "Gefällt mir",
        "Antworten",
        "Kommentieren",
        "Reposten",
        "Posten",
        "Weitere Kommentare laden",
        "Frühere Antworten laden",
    ),
    Labels("opaque", "t1", "t2", "t3", "t4", "t5", "t6", "t7"),
]


class _Session(ScrapingSession):
    """The real page with the pacing and account checks stubbed out."""

    async def delay(self, seconds: float) -> None:
        await asyncio.sleep(min(seconds, 0.02))

    async def check_rate_limit(self) -> None:
        return None

    async def dismiss_modal(self) -> bool:
        return False


class _Navigator:
    def __init__(self, page: Any):
        self._page = page

    async def _navigate_to_page(self, url: str) -> None:
        await self._page.goto(url)


def _article(
    comment_id: str,
    labels: Labels,
    *,
    slug: str,
    name: str,
    text: str,
    replies: str = "",
    replies_loader: bool = False,
) -> str:
    loader = (
        f'<div class="more-replies"><button type="button" class="load-replies">{labels.replies}</button></div>'
        if replies_loader
        else ""
    )
    return f"""
<article data-id="urn:li:comment:({THREAD},{comment_id})">
  <div class="head">
    <a href="https://www.linkedin.com/in/{slug}/"><span>{name}</span><br><span>Synthetic headline for {name}</span></a>
  </div>
  <div class="body"><span>{text}</span></div>
  <div class="actions">
    <button type="button" aria-pressed="false" class="like">{labels.like}</button>
    <button type="button" class="reply">{labels.reply}</button>
  </div>
  <div class="replies">{loader}{replies}</div>
</article>"""


_SCRIPT = r"""
<script>
let nextId = 7300000000000000900n;
const labels = JSON.parse(document.getElementById('labels').textContent);
const mode = document.body.dataset.mode;
const make = (id, slug, name, text) => {
  const wrap = document.createElement('div');
  wrap.innerHTML = window.articleTemplate
    .replaceAll('__ID__', id).replaceAll('__SLUG__', slug)
    .replaceAll('__NAME__', name).replaceAll('__TEXT__', '');
  const article = wrap.firstElementChild;
  article.querySelector('.body span').textContent = text;
  return article;
};
const replyBox = () => {
  const form = document.createElement('form');
  form.className = 'reply-box';
  form.innerHTML = '<div contenteditable="true" role="textbox"></div>' +
    `<button type="submit" disabled>${labels.post}</button>`;
  return form;
};
document.addEventListener('click', event => {
  const button = event.target.closest && event.target.closest('button');
  if (button) {
    // A DOM attribute, not a window property: Patchright evaluates in an
    // isolated world that cannot see this script's globals.
    const clicks = JSON.parse(document.body.getAttribute('data-test-clicks') || '[]');
    clicks.push(button.id || button.className || button.textContent);
    document.body.setAttribute('data-test-clicks', JSON.stringify(clicks));
  }
}, true);
document.addEventListener('input', event => {
  const form = event.target.closest && event.target.closest('form');
  if (!form) return;
  const editor = form.querySelector('[contenteditable]');
  form.querySelector('button[type="submit"]').disabled = !editor.innerText.trim();
});
document.addEventListener('click', event => {
  const like = event.target.closest('button.like, #post-like');
  if (like) {
    like.setAttribute('aria-pressed', like.getAttribute('aria-pressed') === 'true' ? 'false' : 'true');
    return;
  }
  const reply = event.target.closest('button.reply');
  if (reply) {
    let article = reply.closest('article');
    if (mode === 'foreign') article = document.querySelector('article');
    const outer = article.parentElement.closest('article');
    const host = (outer || article).querySelector(':scope > .replies');
    if (!host.querySelector(':scope > form')) host.appendChild(replyBox());
    return;
  }
  if (event.target.closest('#load-more')) {
    const list = document.getElementById('list');
    const holder = document.getElementById('load-more').parentElement;
    list.insertBefore(make(String(nextId++), 'synthetic-loaded-a', 'Loaded One', 'Loaded comment one'), holder);
    list.insertBefore(make(String(nextId++), 'synthetic-loaded-b', 'Loaded Two', 'Loaded comment two'), holder);
    holder.remove();
    return;
  }
  const replies = event.target.closest('button.load-replies');
  if (replies) {
    const host = replies.closest('.replies');
    host.appendChild(make('7300000000000000301', 'synthetic-late', 'Late Reply', 'An older reply'));
    replies.parentElement.remove();
  }
});
document.addEventListener('submit', event => {
  event.preventDefault();
  const form = event.target;
  const editor = form.querySelector('[contenteditable]');
  const text = editor.innerText;
  if (mode === 'silent') return;
  setTimeout(() => {
    const article = make(String(nextId++), 'synthetic-me', 'Synthetic Me', text);
    if (form.id === 'post-box') {
      const list = document.getElementById('list');
      list.insertBefore(article, list.firstChild);
    } else {
      form.parentElement.appendChild(article);
    }
    editor.innerHTML = '';
  }, 150);
});
</script>
"""


def _document(
    labels: Labels,
    *,
    mode: str = "normal",
    load_more: bool = False,
    duplicate: bool = False,
    draft: str = "",
    replies_loader: bool = False,
    flat: bool = False,
) -> str:
    nested = _article(
        NESTED,
        labels,
        slug="synthetic-carol",
        name="Carol Example",
        text="A nested reply",
    )
    first = _article(
        FIRST,
        labels,
        slug="synthetic-alice",
        name="Alice Example",
        text="First top-level comment",
        replies=nested,
        replies_loader=replies_loader,
    )
    second = _article(
        SECOND, labels, slug="synthetic-bob", name="Bob Example", text="Second comment"
    )
    extra = second if duplicate else ""
    loader = (
        f'<div><button type="button" id="load-more">{labels.load}</button></div>'
        if load_more
        else ""
    )
    template = _article(
        "__ID__", labels, slug="__SLUG__", name="__NAME__", text="__TEXT__"
    ).replace("`", "")
    import json

    comment_list = f'<div id="list">{first}{second}{extra}{loader}</div>'
    return f"""<!DOCTYPE html>
<html lang="{labels.locale}"><head><meta charset="utf-8"></head>
<body data-mode="{mode}">
<script type="application/json" id="labels">{json.dumps({"post": labels.post})}</script>
<script>window.articleTemplate = {json.dumps(template.strip())};</script>
<main>
  <section id="post">
    <a href="https://www.linkedin.com/in/synthetic-author/">Synthetic Author</a>
    <p>A synthetic post body long enough to look like content.</p>
    <div class="post-actions">
      <button type="button" id="post-like" aria-pressed="false">{labels.like}</button>
      <button type="button">{labels.comment}</button>
      <button type="button" aria-haspopup="true">{labels.repost}</button>
    </div>
    <form id="post-box"><div contenteditable="true" role="textbox">{draft}</div><button type="submit" {"" if draft else "disabled"}>{labels.post}</button></form>
    {comment_list if flat else ""}
  </section>
  {"" if flat else comment_list}
</main>
{_SCRIPT}
</body></html>"""


@pytest.fixture
async def dom_page():
    """Real chromium page, or skip when no browser is installed."""
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(channel="chromium", headless=True)
            page = await browser.new_page()
        except Exception as exc:  # browser binary missing
            pytest.skip(f"chromium unavailable: {exc}")
        try:
            yield page
        finally:
            await browser.close()


async def _serve(page: Any, html: str) -> CommentScraper:
    await page.unroute("https://www.linkedin.com/**")
    await page.route(
        "https://www.linkedin.com/**",
        lambda route: route.fulfill(content_type="text/html", body=html),
    )
    session = _Session(cast(Page, page))
    return CommentScraper(
        session,
        cast(Any, _Navigator(page)),
        PageContentReader(session),
    )


@pytest.fixture(autouse=True)
def fast_waits(monkeypatch):
    monkeypatch.setattr(comments_module, "_CONFIRMATION_WAIT_SECONDS", 2.0)
    monkeypatch.setattr(comments_module, "_REPLY_BOX_WAIT_SECONDS", 1.0)
    monkeypatch.setattr(comments_module, "_CONTENT_WAIT_MS", 1_000)


def _refs(result: dict[str, Any]) -> list[dict[str, Any]]:
    return result.get("references", {}).get("comments", [])


@pytest.mark.parametrize("labels", LOCALES, ids=lambda item: item.locale)
class TestReadingComments:
    async def test_every_comment_carries_its_urn_author_and_parent(
        self, dom_page, labels
    ):
        scraper = await _serve(dom_page, _document(labels))
        result = await scraper.get_post_comments(POST_URL)

        refs = _refs(result)
        assert [ref["value"] for ref in refs] == [
            f"urn:li:comment:({THREAD},{FIRST})",
            f"urn:li:comment:({THREAD},{NESTED})",
            f"urn:li:comment:({THREAD},{SECOND})",
        ]
        assert refs[0]["url"] == "/in/synthetic-alice/"
        assert refs[0]["text"] == "Alice Example"
        assert refs[0]["context"] == "Synthetic headline for Alice Example"
        assert "parent" not in refs[0]
        assert refs[1]["parent"] == f"urn:li:comment:({THREAD},{FIRST})"
        assert refs[1]["permalink"].startswith(f"/feed/update/urn:li:{THREAD}/")
        # A comment's own text, never its replies' or its buttons'.
        assert refs[0]["excerpt"] == "First top-level comment"
        assert "Second comment" in result["sections"]["comments"]

    async def test_replies_can_be_left_out(self, dom_page, labels):
        scraper = await _serve(dom_page, _document(labels))
        result = await scraper.get_post_comments(POST_URL, include_replies=False)
        assert [ref.get("parent") for ref in _refs(result)] == [None, None]

    async def test_load_more_is_clicked_but_no_toggle_is(self, dom_page, labels):
        scraper = await _serve(dom_page, _document(labels, load_more=True))
        result = await scraper.get_post_comments(POST_URL)

        names = [ref.get("text") for ref in _refs(result)]
        assert "Loaded One" in names and "Loaded Two" in names
        pressed = await dom_page.evaluate(
            "() => document.querySelectorAll('[aria-pressed=\"true\"]').length"
        )
        assert pressed == 0
        # Nor did it open a reply box or leave a marker behind.
        assert (
            await dom_page.evaluate(
                "() => document.querySelectorAll('form.reply-box').length"
            )
            == 0
        )
        assert (
            await dom_page.evaluate(
                "() => document.querySelectorAll('[data-linkedin-mcp-loader]').length"
            )
            == 0
        )

    async def test_loader_scope_never_widens_into_the_post_action_bar(
        self, dom_page, labels
    ):
        # The comment list sits inside the post's own container, next to the
        # post's reaction toggle and its plain Comment button. Only the list's
        # own loader may be clicked.
        scraper = await _serve(dom_page, _document(labels, load_more=True, flat=True))
        result = await scraper.get_post_comments(POST_URL)

        assert "Loaded One" in [ref.get("text") for ref in _refs(result)]
        clicked = await dom_page.evaluate(
            "() => JSON.parse(document.body.getAttribute('data-test-clicks') || '[]')"
        )
        assert clicked == ["load-more"]

    async def test_reply_loaders_follow_include_replies(self, dom_page, labels):
        scraper = await _serve(dom_page, _document(labels, replies_loader=True))
        without = await scraper.get_post_comments(POST_URL, include_replies=False)
        assert "Late Reply" not in [ref.get("text") for ref in _refs(without)]

        scraper = await _serve(dom_page, _document(labels, replies_loader=True))
        with_replies = await scraper.get_post_comments(POST_URL)
        late = [ref for ref in _refs(with_replies) if ref.get("text") == "Late Reply"]
        assert late and late[0]["parent"] == f"urn:li:comment:({THREAD},{FIRST})"


@pytest.mark.parametrize("labels", LOCALES, ids=lambda item: item.locale)
class TestReplyingToComments:
    async def test_reply_is_typed_under_the_target_and_confirmed(
        self, dom_page, labels
    ):
        scraper = await _serve(dom_page, _document(labels))
        result = await scraper.reply_to_comment(
            POST_URL,
            f"urn:li:comment:({THREAD},{SECOND})",
            "Thanks for this\nsecond line",
            confirm=True,
        )

        assert result["status"] == "posted", result
        assert result["posted"] is True and result["retry_safe"] is False
        assert result["posted_comment_urn"].startswith(f"urn:li:comment:({THREAD},")
        assert result["target"]["author_name"] == "Bob Example"
        assert (
            await dom_page.evaluate(
                """urn => document.querySelector(`[data-id="${urn}"] .body`).textContent""",
                result["posted_comment_urn"],
            )
            == "Thanks for this\nsecond line"
        )
        parent = await dom_page.evaluate(
            """urn => {
                const node = document.querySelector(`[data-id="${urn}"]`);
                return node.parentElement.closest('article').getAttribute('data-id');
            }""",
            result["posted_comment_urn"],
        )
        assert parent == f"urn:li:comment:({THREAD},{SECOND})"
        assert (
            await dom_page.evaluate(
                """() => [...document.querySelectorAll('*')].some(element =>
                [...element.attributes].some(a => a.name.startsWith('data-linkedin-mcp')))"""
            )
            is False
        )

    async def test_reply_to_a_reply_lands_in_the_parent_thread(self, dom_page, labels):
        scraper = await _serve(dom_page, _document(labels))
        result = await scraper.reply_to_comment(
            POST_URL,
            f"urn:li:comment:({THREAD},{NESTED})",
            "Nested answer",
            confirm=True,
        )
        assert result["status"] == "posted", result

    async def test_a_box_opening_under_another_comment_is_refused(
        self, dom_page, labels
    ):
        scraper = await _serve(dom_page, _document(labels, mode="foreign"))
        result = await scraper.reply_to_comment(
            POST_URL, f"urn:li:comment:({THREAD},{SECOND})", "Wrong place", confirm=True
        )

        assert result["status"] == "reply_box_unavailable"
        assert result["retry_safe"] is True
        typed = await dom_page.evaluate(
            "() => [...document.querySelectorAll('[contenteditable]')].map(e => e.innerText).join('')"
        )
        assert typed.strip() == ""

    async def test_an_unconfirmed_submission_is_not_retry_safe(self, dom_page, labels):
        scraper = await _serve(dom_page, _document(labels, mode="silent"))
        result = await scraper.reply_to_comment(
            POST_URL, f"urn:li:comment:({THREAD},{FIRST})", "Lost reply", confirm=True
        )
        assert result["status"] == "unconfirmed"
        assert result["posted"] is False and result["retry_safe"] is False

    async def test_missing_and_duplicate_comments_fail_closed(self, dom_page, labels):
        scraper = await _serve(dom_page, _document(labels))
        missing = await scraper.reply_to_comment(
            POST_URL,
            f"urn:li:comment:({THREAD},7300000000000000999)",
            "x",
            confirm=True,
        )
        assert missing["status"] == "comment_not_found"

        scraper = await _serve(dom_page, _document(labels, duplicate=True))
        duplicate = await scraper.reply_to_comment(
            POST_URL, f"urn:li:comment:({THREAD},{SECOND})", "x", confirm=True
        )
        assert duplicate["status"] == "comment_ambiguous"
        assert (
            await dom_page.evaluate(
                "() => document.querySelectorAll('form.reply-box').length"
            )
            == 0
        )


@pytest.mark.parametrize("labels", LOCALES, ids=lambda item: item.locale)
class TestTopLevelAndReactions:
    async def test_comment_on_post_is_confirmed_as_a_top_level_comment(
        self, dom_page, labels
    ):
        scraper = await _serve(dom_page, _document(labels))
        result = await scraper.comment_on_post(POST_URL, "Great post", confirm=True)
        assert result["status"] == "posted", result
        top_level = await dom_page.evaluate(
            """urn => !document.querySelector(`[data-id="${urn}"]`)
                .parentElement.closest('article')""",
            result["posted_comment_urn"],
        )
        assert top_level is True

    async def test_a_draft_in_the_post_box_is_left_untouched(self, dom_page, labels):
        scraper = await _serve(dom_page, _document(labels, draft="Someone's draft"))
        result = await scraper.comment_on_post(POST_URL, "Great post", confirm=True)
        assert result["status"] == "composer_occupied"
        assert (
            await dom_page.evaluate(
                "() => document.querySelector('#post-box [contenteditable]').innerText"
            )
        ) == "Someone's draft"

    async def test_like_presses_only_the_target_toggle_and_is_idempotent(
        self, dom_page, labels
    ):
        scraper = await _serve(dom_page, _document(labels))
        urn = f"urn:li:comment:({THREAD},{FIRST})"
        first = await scraper.react_to_comment(POST_URL, urn, confirm=True)
        assert first["status"] == "reacted", first
        pressed = await dom_page.evaluate(
            """() => [...document.querySelectorAll('[aria-pressed="true"]')]
                .map(button => button.closest('article')?.getAttribute('data-id') || 'post')"""
        )
        assert pressed == [urn]

        second = await scraper.react_to_comment(POST_URL, urn, confirm=True)
        assert second["status"] in {"reacted", "already_reacted"}
