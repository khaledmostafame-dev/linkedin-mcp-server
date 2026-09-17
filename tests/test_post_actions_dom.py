"""Browser-DOM tests for the programs that act on a single post page.

Covers save_post and get_post_reactions, delete_post/edit_post's route to
the post's own control menu, and the comment programs (post comment box,
top-level loaders, reply candidates). The unit suite mocks
``page.evaluate``, so these programs never execute there. These tests run
the real ones in headless chromium.

The page around the controls is a claim about LinkedIn: its shape mirrors
the structure of a signed-in, English-UI post page captured on 2026-09-17
(structure only, every name, slug and id here is synthetic). What it keeps
is what defeated the earlier probes: dozens of ``button[aria-expanded]``
(the post's menu opener with its collapsed panel beside it, the reactor
facepile's "see more", the Like picker, Repost, the emoji picker, the
comment sort control and one menu opener per comment), action-row
controls wrapped one per span/div and carrying ``data-finite-scroll-hotkey``,
and a single ``data-reaction-details`` button that comments do not carry.

Also mirrored from the capture: the comment box is a role=textbox editor
with an empty contenteditable helper beside it; the top-level comment
loaders sit in a sibling of the comment list's third ancestor; and for some
comments a reaction-count button sits one level closer to the Like toggle
than the Reply button.

The open menus, the delete confirmation, the edit dialog and the reactions
dialog are a claim about the algorithm only: none was in the capture. The menu items follow the shape
``post_composer.py`` ports from upstream (``[role="button"]`` items with
``data-test-icon`` hooks), and a click on an item closes the menu, which
is what forces ``save_post`` to reopen it before confirming.

Every fixture renders with English labels and with opaque tokens; a
decision that differs between them read a word.
"""

from __future__ import annotations

from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

import asyncio

import pytest
from patchright.async_api import Page, async_playwright

from linkedin_mcp_server.scraping import post_composer as composer_module
from linkedin_mcp_server.scraping.comments import (
    _EXPANSION_SCAN_JS,
    _POST_COMPOSER_JS,
    _REPLY_PREPARE_JS,
)
from linkedin_mcp_server.scraping.post_actions import PostActions
from linkedin_mcp_server.scraping.post_composer import PostComposer
from linkedin_mcp_server.scraping.post_content import build_post_edit
from linkedin_mcp_server.scraping.reactions import _FIND_SOCIAL_COUNTS_CONTROL_JS
from linkedin_mcp_server.scraping.session import ScrapingSession

pytestmark = [
    pytest.mark.browser_dom,
    pytest.mark.xdist_group("browser_runtime"),
]

POST_URL = "https://www.linkedin.com/feed/update/urn:li:activity:7000000000000000001/"

LABELS = {
    "en": {
        "menu": "Open control menu for post",
        "panel": "Control menu for post",
        "reactions": "12 reactions",
        "comments": "3 comments",
        "like": "React Like",
        "picker": "Open reactions menu",
        "comment": "Comment",
        "repost": "Repost",
        "send": "Send in a private message",
        "more": "See more reactors",
        "save": "Save",
        "copy": "Copy link to post",
        "reply": "Reply to comment",
        "load": "Load more comments",
        "edit": "Edit post",
        "delete": "Delete post",
    },
    "opaque": {
        key: f"k{index}"
        for index, key in enumerate(
            [
                "menu",
                "panel",
                "reactions",
                "comments",
                "like",
                "picker",
                "comment",
                "repost",
                "send",
                "more",
                "save",
                "copy",
                "reply",
                "load",
                "edit",
                "delete",
            ]
        )
    },
}


def _comment(
    index: int, label: dict[str, str], *, reaction_count: bool, menu: bool
) -> str:
    # Live shape (2026-09-17): the like toggle and its picker share a span;
    # for some comments the reaction-count button sits beside that span,
    # one level closer to the toggle than the Reply button's own wrapper.
    count = (
        f'<button aria-label="{label["reactions"]}" type="button" data-fixture="count">'
        '<img alt=""><span aria-hidden="true">1</span></button>'
        if reaction_count
        else ""
    )
    menu_block = (
        f"""<div><div id="cm{index}">
    <button id="cmb{index}" class="comment-menu" aria-expanded="false" type="button" tabindex="0">
      <svg role="img" aria-hidden="true" aria-label="{label["more"]}"
        data-test-icon="overflow-web-ios-small"></svg></button>
    <div id="cmp{index}" aria-hidden="true" tabindex="-1"></div>
  </div></div>"""
        if menu
        else ""
    )
    return f"""
<div><article id="c{index}" data-id="urn:li:comment:(activity:7000000000000000001,{index})" tabindex="-1">
  <div><a aria-label="{label["comment"]}" href="/in/synthetic-commenter-{index}/">
    <h3><span>Commenter {index}</span></h3></a></div>
  <div><time>1d</time>{menu_block}</div>
  <div><section><span dir="ltr">A synthetic comment.</span></section></div>
  <div><div><div><div>
    <span>
      <button aria-label="{label["like"]}" aria-pressed="false"><span>Like</span></button>
      <button aria-label="{label["picker"]}" aria-expanded="false"
        data-finite-scroll-hotkey="l" tabindex="0"><svg data-test-icon="caret-small"></svg></button>
    </span><span></span>{count}
  </div>
  <div></div>
  <div><button id="rp{index}" aria-label="{label["reply"]}" data-fixture="reply">
    <span><span aria-hidden="true">Reply</span></span></button></div>
  </div></div></div>
</article></div>"""


def _post_page(
    locale: str = "en",
    *,
    saved: bool = False,
    second_reaction_hook: bool = False,
    second_opener: bool = False,
    open_dialog: bool = False,
    second_save_item: bool = False,
    open_panel: str = "unhide",
    opener_reports_expanded: bool = True,
    post_opener: str = "standard",
    post_action_row: bool = True,
    comment_in_panel: bool = False,
    helper_text: str = "",
    post_text: str = "A synthetic post body.",
    with_comments: bool = True,
    comment_menus: bool = True,
    post_owner_items: bool = True,
    open_comment_menu: bool = False,
    author_href: str = "/in/synthetic-author/",
    leading_profile_link: bool = False,
    render_delay_ms: int = 0,
) -> str:
    label = LABELS[locale]
    extra_hook = (
        f'<li><button aria-label="{label["reactions"]}" type="button" '
        'data-reaction-details=""><span aria-hidden="true">1</span></button></li>'
        if second_reaction_hook
        else ""
    )
    extra_opener = (
        f'<div id="x1"><button id="x2" aria-expanded="false" aria-label="{label["menu"]}" '
        'type="button" tabindex="0"><svg></svg></button>'
        f'<div id="x3" aria-hidden="true" aria-label="{label["panel"]}" tabindex="-1"></div></div>'
        if second_opener
        else ""
    )
    dialog = (
        '<div role="dialog"><a href="/in/synthetic-other/">Other</a></div>'
        if open_dialog
        else ""
    )
    # "no_panel": a differently shaped post menu (its panel rendered elsewhere),
    # which the opener rule must not recognise.
    post_panel = (
        f'<div id="panel" aria-hidden="true" aria-label="{label["panel"]}" tabindex="-1"></div>'
        if post_opener == "standard"
        else ""
    )
    detached_panel = (
        ""
        if post_opener == "standard"
        else f'<div id="panel" aria-hidden="true" aria-label="{label["panel"]}" tabindex="-1"></div>'
    )
    hotkeys = post_action_row
    action_row = (
        f"""
  <div><span>
    <button aria-label="{label["like"]}" aria-pressed="false"><span>Like</span></button>
    <button aria-expanded="false" aria-label="{label["picker"]}"
      {'data-finite-scroll-hotkey="l"' if hotkeys else ""}
      tabindex="0"><span><svg data-test-icon="caret-small"></svg></span></button>
  </span>
  <span tabindex="-1"><div><button role="button" aria-label="{label["comment"]}"
    {'data-finite-scroll-hotkey="c"' if hotkeys else ""} tabindex="0"><span>Comment</span></button></div></span>
  <div><span tabindex="-1"><button aria-expanded="false" aria-label="{label["repost"]}"
    type="button" {'data-finite-scroll-hotkey="r"' if hotkeys else ""} tabindex="0"><span>Repost</span></button>
    <div aria-hidden="true" tabindex="-1"></div></span></div>
  <div><span tabindex="-1"><button aria-label="{label["send"]}" type="button"
    {'data-finite-scroll-hotkey="s"' if hotkeys else ""}><span>Send</span></button></span></div>
  </div>"""
        if post_action_row
        else ""
    )
    comments = (
        "".join(
            _comment(index, label, reaction_count=index % 2 == 1, menu=comment_menus)
            for index in range(1, 4)
        )
        if with_comments
        else ""
    )
    save_item = f"item(bookmark, {label['save']!r}, 'save')"
    save_items = f"{save_item} + {save_item}" if second_save_item else save_item
    stray_comment = (
        '<div data-id="urn:li:comment:(activity:7000000000000000001,9)"></div>'
        if comment_in_panel
        else ""
    )
    # A profile link <main> renders ahead of the post unit (a sidebar card
    # rendered first): "the first profile link in <main>" would take it.
    leading_link = (
        '<div><a href="/in/synthetic-sidebar-member/"><span>Someone</span></a></div>'
        if leading_profile_link
        else ""
    )
    # A page still rendering, in the order measured live (2026-09-17): the
    # boot loader with no <main> at all, then a laid-out <main> (a visible
    # placeholder, so a wait for a visible <main> is already satisfied) before
    # the post inside it, then the post. The page's controls are wired up only
    # once the post exists, as a client-rendered page does.
    deferred = render_delay_ms > 0
    loader = (
        '<div id="app-boot-bg-loader"><div><svg></svg></div></div><div id="app-root"></div>'
        if deferred
        else ""
    )
    template_open = '<template id="deferred-root">' if deferred else ""
    template_close = "</template>" if deferred else ""
    boot_open = "function boot() {" if deferred else ""
    boot_close = (
        f"""}}
  setTimeout(() => {{
    document.getElementById('app-boot-bg-loader').remove();
    document.getElementById('app-root').innerHTML =
      '<div class="application-outlet"><main aria-label="Main Feed">'
      + '<div aria-busy="true"><div style="height: 240px"></div></div></main></div>';
    setTimeout(() => {{
      document.getElementById('app-root').replaceWith(
        document.getElementById('deferred-root').content.cloneNode(true));
      boot();
    }}, {render_delay_ms});
  }}, 100);
"""
        if deferred
        else ""
    )
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"></head>
<body dir="ltr"><div id="artdeco-toasts__wormhole"></div>{dialog}
{loader}{template_open}<div id="app-root"><div class="application-outlet"><main aria-label="Main Feed">
{leading_link}<div aria-label="Feed update"><section>
<a aria-label="{label["menu"]}" data-test-boost-header="" href="/ad-beta/">Boost</a>
<div data-view-name="feed-full-update"><div>
<div id="post" role="article" data-urn="urn:li:activity:7000000000000000001"><div><div>
  <h2>Feed post</h2>
  <div><div><div><div>
    <a aria-label="{label["comment"]}" href="{author_href}"><img alt=""></a>
    <div><a aria-label="{label["comment"]}" href="{author_href}"><span>Synthetic Author</span></a></div>
  </div></div>
  <div><div id="menu-root">
    <button id="opener" aria-expanded="false" aria-label="{label["menu"]}" type="button" tabindex="0">
      <svg width="24" height="24"></svg></button>{post_panel}
  </div>{extra_opener}</div></div>
  <div tabindex="-1"><div dir="ltr"><span dir="ltr">{post_text}</span></div></div>
  <div id="social"><div><div><div><ul>
    <li><button id="reactions" aria-label="{label["reactions"]}" type="button" data-reaction-details="">
      <img alt=""><span aria-hidden="true">12</span></button></li>
    {extra_hook}
    <li data-non-reaction-details=""><ul><li>
      <button aria-label="{label["comments"]}" type="button"><span aria-hidden="true">3</span></button>
    </li></ul></li>
  </ul></div></div></div>
  <section><h3>Reactions</h3><ul>
    <li><a aria-label="{label["comment"]}" href="/in/synthetic-reactor-1/"><img alt=""></a></li>
    <li><button aria-expanded="false" aria-label="{label["more"]}" type="button"
      data-jump-link-target="reactors-facepile-see-more-jump-target">+</button></li>
  </ul></section>
  {action_row}</div>
  <div id="comments"><div>
  <div data-scroll-name="true"><div><form><div><div>
    <div role="textbox" aria-label="{label["comment"]}" aria-multiline="true"
      contenteditable="true" data-test-ql-editor-contenteditable="true"><p><br></p></div>
    <div contenteditable="true" tabindex="-1" data-fixture="helper">{helper_text}</div>
  </div>
    <span><button aria-controls="emoji-panel" aria-expanded="false" aria-label="{label["picker"]}"
      type="button" data-reaction-summary__emoji-hoverable=""><svg></svg></button></span>
  </div></form></div></div>
  <div><div><button id="sort" aria-expanded="false" type="button" tabindex="0"><span>Most relevant</span></button>
    <div id="sort-panel" aria-hidden="true" tabindex="-1"></div></div></div>
  <div id="comment-list">
    <div><div><div>{comments}</div><div></div></div></div>
    <div>
      <button id="load-icon" aria-label="{label["load"]}" data-fixture="loader">
        <svg role="none" aria-hidden="true" data-test-icon="maximize-small"></svg><span></span></button>
      <button id="load-text" data-fixture="loader"><span>{label["load"]}</span></button>
    </div>
  </div>
  </div></div>
</div></div></div></div></div></section></div>
</main></div></div>{template_close}
{detached_panel}
<output id="counters" hidden data-opener="0" data-save="0" data-reactions="0"
  data-comment-menu="0" data-sort="0"></output>
<script>{boot_open}
  // Page globals are invisible to patchright's isolated evaluate world, so
  // the counters live in the DOM; irreversible actions are also reported to
  // the test server, because the flows navigate away afterwards.
  const counters = document.getElementById('counters');
  function count(name) {{
    counters.dataset[name] = String(Number(counters.dataset[name]) + 1);
  }}
  function record(name, value) {{
    fetch('/__record/' + name + '?value=' + encodeURIComponent(value || ''));
  }}
  let saved = {"true" if saved else "false"};
  const opener = document.getElementById('opener');
  const panel = document.getElementById('panel');
  function item(glyph, text, action) {{
    return '<li><div role="button" tabindex="0" data-action="' + action + '">'
      + '<svg role="none" aria-hidden="true" data-test-icon="' + glyph + '"></svg>'
      + '<div><h5 role="none">' + text + '</h5></div></div></li>';
  }}
  const ownerItems = () => item('edit-medium', {label["edit"]!r}, 'edit')
    + item('trash-medium', {label["delete"]!r}, 'delete');
  function render() {{
    const bookmark = saved ? 'bookmark-fill-medium' : 'bookmark-outline-medium';
    panel.innerHTML = '<div><ul>'
      + {save_items}
      + ({"ownerItems()" if post_owner_items else "''"})
      + '<li><div role="button" tabindex="0"><svg data-test-icon="link-medium"></svg>'
      + '<div><h5 role="none">' + {label["copy"]!r} + '</h5></div></div></li>'
      + '</ul>{stray_comment}</div>';
  }}
  function close() {{
    opener.setAttribute('aria-expanded', 'false');
    panel.setAttribute('aria-hidden', 'true');
    panel.innerHTML = '';
  }}
  function confirmDialog(onConfirm) {{
    const dialog = document.createElement('div');
    dialog.setAttribute('role', 'alertdialog');
    dialog.innerHTML = '<p>Sure?</p><button>No</button><button>Yes</button>';
    dialog.lastElementChild.addEventListener('click', () => {{
      onConfirm();
      dialog.remove();
    }});
    document.body.appendChild(dialog);
  }}
  opener.addEventListener('click', () => {{
    count('opener');
    if (opener.getAttribute('aria-expanded') === 'true') {{ close(); return; }}
    if ({"true" if opener_reports_expanded else "false"}) {{
      opener.setAttribute('aria-expanded', 'true');
    }}
    if ({"true" if open_panel == "remove" else "false"}) {{
      panel.removeAttribute('aria-hidden');
    }} else {{
      panel.setAttribute('aria-hidden', 'false');
    }}
    render();
  }});
  panel.addEventListener('click', event => {{
    const target = event.target.closest('[data-action]');
    if (!target) return;
    const action = target.getAttribute('data-action');
    if (action === 'save') {{
      count('save');
      saved = !saved;
    }} else if (action === 'delete') {{
      confirmDialog(() => record('post-delete'));
    }} else if (action === 'edit') {{
      const editDialog = document.createElement('div');
      editDialog.setAttribute('role', 'dialog');
      editDialog.innerHTML = '<div role="textbox" contenteditable="true"><p>'
        + {post_text!r} + '</p></div>'
        + '<button aria-label="Back">Back</button><button>Save</button>';
      const editor = editDialog.firstElementChild;
      editDialog.lastElementChild.addEventListener('click', () => {{
        record('edit-save', editor.innerText);
        editDialog.remove();
      }});
      document.body.appendChild(editDialog);
    }}
    close();
  }});
  for (const menu of document.querySelectorAll('.comment-menu')) {{
    const commentPanel = menu.nextElementSibling;
    const openComment = () => {{
      menu.setAttribute('aria-expanded', 'true');
      commentPanel.setAttribute('aria-hidden', 'false');
      commentPanel.innerHTML = '<div><ul>' + ownerItems() + '</ul></div>';
    }};
    menu.addEventListener('click', () => {{
      count('commentMenu');
      openComment();
    }});
    if ({"true" if open_comment_menu else "false"} && menu.id === 'cmb1') openComment();
    commentPanel.addEventListener('click', event => {{
      const target = event.target.closest('[data-action]');
      if (!target) return;
      if (target.getAttribute('data-action') === 'delete') {{
        record('comment-delete');
        confirmDialog(() => record('comment-delete-confirmed'));
      }} else {{
        record('comment-edit');
      }}
    }});
  }}
  document.getElementById('sort').addEventListener('click', () => {{
    count('sort');
    const sort = document.getElementById('sort');
    sort.setAttribute('aria-expanded', 'true');
    const sortPanel = document.getElementById('sort-panel');
    sortPanel.setAttribute('aria-hidden', 'false');
    sortPanel.innerHTML = '<ul><li><div role="button">A</div></li></ul>';
  }});
  document.getElementById('reactions').addEventListener('click', () => {{
    count('reactions');
    const dialog = document.createElement('div');
    dialog.setAttribute('role', 'dialog');
    dialog.innerHTML = '<ul><li><a href="/in/synthetic-reactor-1/">Reactor</a></li></ul>';
    document.body.appendChild(dialog);
  }});
{boot_close}</script>
</body></html>"""


class _Session(ScrapingSession):
    async def delay(self, seconds: float) -> None:
        await asyncio.sleep(0)

    async def check_rate_limit(self) -> None:
        return None


class _Navigator:
    def __init__(self, page: Any):
        self._page = page

    async def _navigate_to_page(self, url: str) -> None:
        await self._page.goto(url)
        await self._page.wait_for_load_state("load")


@pytest.fixture
async def dom_page():
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


async def _serve(page: Any, html: str) -> PostActions:
    async def handle(route: Any) -> None:
        await route.fulfill(content_type="text/html; charset=utf-8", body=html)

    await page.route("https://www.linkedin.com/**", handle)
    session = _Session(cast(Page, page))
    return PostActions(session, cast(Any, _Navigator(page)))


async def _counters(page: Any) -> dict[str, int]:
    return await page.evaluate(
        "() => { const c = document.getElementById('counters').dataset;"
        " return {opener: Number(c.opener), save: Number(c.save),"
        " reactions: Number(c.reactions),"
        " dialogs: document.querySelectorAll('[role=\"dialog\"]').length}; }"
    )


@pytest.mark.parametrize("locale", list(LABELS))
async def test_reactions_control_is_the_single_reaction_details_hook(dom_page, locale):
    await dom_page.set_content(_post_page(locale))

    assert await dom_page.evaluate(_FIND_SOCIAL_COUNTS_CONTROL_JS) is True
    counters = await _counters(dom_page)
    assert (counters["reactions"], counters["dialogs"], counters["opener"]) == (
        1,
        1,
        0,
    )


@pytest.mark.parametrize(
    "variant", [{"second_reaction_hook": True}, {"open_dialog": True}]
)
async def test_reactions_probe_refuses_ambiguity_without_clicking(dom_page, variant):
    await dom_page.set_content(_post_page(**variant))

    assert await dom_page.evaluate(_FIND_SOCIAL_COUNTS_CONTROL_JS) is False
    assert (await _counters(dom_page))["reactions"] == 0


@pytest.mark.parametrize("locale", list(LABELS))
async def test_confirm_false_reads_state_without_clicking_the_item(dom_page, locale):
    actions = await _serve(dom_page, _post_page(locale))

    result = await actions.save_post(POST_URL, confirm=False)

    assert (result["status"], result["saved"], result["retry_safe"]) == (
        "confirmation_required",
        False,
        True,
    )
    counters = await _counters(dom_page)
    assert (counters["opener"], counters["save"]) == (1, 0)


# An open panel may either say aria-hidden="false" or drop the attribute;
# the opener is still found by its aria-expanded="true" either way.
@pytest.mark.parametrize("open_panel", ["unhide", "remove"])
@pytest.mark.parametrize("locale", list(LABELS))
async def test_confirm_true_saves_and_reopens_the_menu_to_confirm(
    dom_page, locale, open_panel
):
    actions = await _serve(dom_page, _post_page(locale, open_panel=open_panel))

    result = await actions.save_post(POST_URL, confirm=True)

    assert (result["status"], result["saved"], result["retry_safe"]) == (
        "saved",
        True,
        False,
    )
    counters = await _counters(dom_page)
    # Opened once to read, once more after the click closed it.
    assert (counters["opener"], counters["save"]) == (2, 1)


async def test_already_saved_post_is_not_clicked(dom_page):
    actions = await _serve(dom_page, _post_page(saved=True))

    result = await actions.save_post(POST_URL, confirm=True)

    assert (result["status"], result["saved"]) == ("already_in_desired_state", True)
    assert (await _counters(dom_page))["save"] == 0


async def test_unsave_reads_the_fill_glyph_as_saved(dom_page):
    actions = await _serve(dom_page, _post_page(saved=True))

    result = await actions.save_post(POST_URL, confirm=True, unsave=True)

    assert (result["status"], result["saved"]) == ("unsaved", False)
    assert (await _counters(dom_page))["save"] == 1


async def test_a_second_menu_opener_above_the_action_row_refuses(dom_page):
    actions = await _serve(dom_page, _post_page(second_opener=True))

    result = await actions.save_post(POST_URL, confirm=True)

    assert result["status"] == "structural_signal_not_found"
    counters = await _counters(dom_page)
    assert (counters["opener"], counters["save"]) == (0, 0)


async def test_two_bookmark_items_refuse_before_clicking(dom_page):
    actions = await _serve(dom_page, _post_page(second_save_item=True))

    result = await actions.save_post(POST_URL, confirm=True)

    assert (result["status"], result["saved"]) == ("structural_signal_not_found", None)
    counters = await _counters(dom_page)
    assert (counters["opener"], counters["save"]) == (1, 0)


async def test_a_panel_shown_without_its_opener_expanding_is_not_trusted(dom_page):
    actions = await _serve(dom_page, _post_page(opener_reports_expanded=False))

    result = await actions.save_post(POST_URL, confirm=True)

    assert (result["status"], result["saved"]) == ("structural_signal_not_found", None)
    assert (await _counters(dom_page))["save"] == 0


# --- comment programs on the same page (scraping/comments.py) -------------


async def _load(page: Any, html: str) -> None:
    async def handle(route: Any) -> None:
        await route.fulfill(content_type="text/html; charset=utf-8", body=html)

    await page.route("https://www.linkedin.com/**", handle)
    await page.goto(POST_URL)


@pytest.mark.parametrize("locale", list(LABELS))
async def test_post_comment_box_is_the_textbox_beside_its_empty_helper(
    dom_page, locale
):
    await _load(dom_page, _post_page(locale))

    state = await dom_page.evaluate(_POST_COMPOSER_JS, {"token": "t"})

    assert state["status"] == "ready"
    marked = await dom_page.evaluate(
        "() => Array.from(document.querySelectorAll("
        "'[data-linkedin-mcp-comment-editor]')).map(e => e.getAttribute('role'))"
    )
    assert marked == ["textbox"]


async def test_a_helper_holding_text_keeps_the_comment_box_ambiguous(dom_page):
    await _load(dom_page, _post_page(helper_text="draft"))

    state = await dom_page.evaluate(_POST_COMPOSER_JS, {"token": "t"})

    assert state["status"] == "ambiguous_editor"


async def test_top_level_loaders_beside_the_comment_list_are_found(dom_page):
    await _load(dom_page, _post_page())

    scan = await dom_page.evaluate(
        _EXPANSION_SCAN_JS, {"includeReplies": False, "token": "t", "targetId": None}
    )

    assert (scan["count"], scan["loaders"]) == (3, 2)
    marked = await dom_page.evaluate(
        "() => Array.from(document.querySelectorAll('[data-linkedin-mcp-loader]'))"
        ".map(e => e.getAttribute('data-fixture'))"
    )
    assert marked == ["loader"]


@pytest.mark.parametrize("locale", list(LABELS))
async def test_reply_comes_before_a_closer_reaction_count(dom_page, locale):
    await _load(dom_page, _post_page(locale))

    # Comment 1 carries a reaction-count button one level closer than Reply;
    # comment 2 does not.
    shapes = []
    for comment_id in ("1", "2"):
        prepared = await dom_page.evaluate(
            _REPLY_PREPARE_JS,
            {
                "id": comment_id,
                "kind": "activity",
                "threadId": "7000000000000000001",
                "token": f"t{comment_id}",
                "maxCandidates": 3,
            },
        )
        assert prepared["status"] == "candidates"
        shapes.append(
            await dom_page.evaluate(
                "(token) => Array.from(document.querySelectorAll("
                "'[data-linkedin-mcp-reply-candidate]'))"
                ".filter(e => e.getAttribute('data-linkedin-mcp-reply-candidate')"
                ".startsWith(token + '-'))"
                ".sort((a, b) => a.getAttribute('data-linkedin-mcp-reply-candidate')"
                ".localeCompare(b.getAttribute('data-linkedin-mcp-reply-candidate')))"
                ".map(e => e.getAttribute('data-fixture'))",
                f"t{comment_id}",
            )
        )
    assert shapes == [["reply", "count"], ["reply"]]


# --- delete_post / edit_post through the post's own menu (post_composer.py) ---


class _ComposerNavigator(_Navigator):
    async def _navigate_to_page(self, url: str) -> None:
        if url.endswith("/in/me/"):
            url = "https://www.linkedin.com/in/synthetic-author/"
        await super()._navigate_to_page(url)


@pytest.fixture
def fast_steps(monkeypatch):
    monkeypatch.setattr(composer_module, "_STEP_TIMEOUT_MS", 1500)
    monkeypatch.setattr(composer_module, "_OPEN_TIMEOUT_MS", 3000)


@pytest.fixture(autouse=True)
def no_composer_traces(monkeypatch):
    # The composer's step traces screenshot a real page; nothing here reads them.
    monkeypatch.setattr(composer_module, "trace_enabled", lambda: False)


async def _composer(
    page: Any, **variant: Any
) -> tuple[PostComposer, list[tuple[str, str]]]:
    records: list[tuple[str, str]] = []

    async def handle(route: Any) -> None:
        url = route.request.url
        path = urlsplit(url).path
        if path.startswith("/__record/"):
            value = parse_qs(urlsplit(url).query).get("value", [""])[0]
            records.append((path.removeprefix("/__record/"), value))
            await route.fulfill(status=204, body="")
            return
        if path.startswith("/feed/update/"):
            if any(name == "post-delete" for name, _ in records):
                body = "<html><body><main><p>Unavailable.</p></main></body></html>"
            else:
                edited = [value for name, value in records if name == "edit-save"]
                shown: dict[str, Any] = dict(variant)
                if edited:
                    shown["post_text"] = edited[-1]
                body = _post_page(**shown)
        else:
            body = "<html><body><main><p>Profile.</p></main></body></html>"
        await route.fulfill(content_type="text/html; charset=utf-8", body=body)

    await page.route("https://www.linkedin.com/**", handle)
    session = _Session(cast(Page, page))
    return PostComposer(session, cast(Any, _ComposerNavigator(page))), records


async def _composer_counters(page: Any) -> dict[str, int]:
    return await page.evaluate(
        "() => { const c = document.getElementById('counters').dataset;"
        " return {opener: Number(c.opener), commentMenu: Number(c.commentMenu),"
        " sort: Number(c.sort)}; }"
    )


@pytest.mark.parametrize("with_comments", [True, False])
@pytest.mark.parametrize("locale", list(LABELS))
async def test_delete_preview_opens_only_the_posts_own_menu(
    dom_page, fast_steps, locale, with_comments
):
    composer, records = await _composer(
        dom_page, locale=locale, with_comments=with_comments
    )

    result = await composer.delete_post(POST_URL, confirm=False)

    assert result["status"] == "preview"
    counters = await _composer_counters(dom_page)
    assert (counters["opener"], counters["commentMenu"], records) == (1, 0, [])


async def test_confirmed_delete_goes_through_the_posts_own_menu(dom_page, fast_steps):
    composer, records = await _composer(dom_page)

    result = await composer.delete_post(POST_URL, confirm=True)

    assert result["status"] == "deleted"
    assert [name for name, _ in records] == ["post-delete"]


@pytest.mark.parametrize(
    "variant",
    [
        # The post's menu is shaped differently (its panel is not beside it).
        {"post_opener": "no_panel"},
        # ...and the post's own action row is missing too, so the first
        # action anchor on the page belongs to a comment.
        {"post_opener": "no_panel", "post_action_row": False, "comment_menus": False},
    ],
)
async def test_a_post_menu_not_recognised_never_reaches_another_menu(
    dom_page, fast_steps, variant
):
    composer, records = await _composer(dom_page, **variant)

    result = await composer.delete_post(POST_URL, confirm=True)

    assert result["status"] == "post_unavailable"
    counters = await _composer_counters(dom_page)
    assert (counters["commentMenu"], counters["sort"], records) == (0, 0, [])


async def test_a_post_panel_holding_a_comment_is_refused(dom_page, fast_steps):
    composer, records = await _composer(dom_page, comment_in_panel=True)

    result = await composer.delete_post(POST_URL, confirm=True)

    # The menu that opened is not provably the post's: refused, but it says
    # nothing about who wrote the post.
    assert result["status"] == "menu_unavailable"
    assert records == []


async def test_edit_types_into_the_dialog_editor_not_the_comment_box(
    dom_page, fast_steps
):
    composer, records = await _composer(dom_page)

    result = await composer.edit_post(
        POST_URL,
        build_post_edit("Edited synthetic text", allow_schedule=False),
        confirm=True,
    )

    assert result["status"] == "edited"
    assert records == [("edit-save", "Edited synthetic text")]


async def test_owner_items_come_only_from_the_posts_own_panel(dom_page, fast_steps):
    # The post's menu offers no owner actions while a comment's menu, already
    # open further down, does.
    composer, records = await _composer(
        dom_page, post_owner_items=False, open_comment_menu=True
    )

    result = await composer.delete_post(POST_URL, confirm=True)

    assert result["status"] == "not_own_post"
    assert records == []


# --- ownership: rendering and the author's own links ----------------------


async def _preview(composer: PostComposer, tool: str) -> dict[str, Any]:
    if tool == "delete_post":
        return await composer.delete_post(POST_URL, confirm=False)
    return await composer.edit_post(
        POST_URL,
        build_post_edit("New synthetic text", allow_schedule=False),
        confirm=False,
    )


@pytest.mark.parametrize("tool", ["delete_post", "edit_post"])
async def test_a_post_still_rendering_is_judged_once_it_renders(
    dom_page, fast_steps, tool
):
    # Live run 2026-09-17: the page was still the boot loader right after the
    # navigation, and <main> attaches before the post inside it does. Read at
    # that moment the author link does not exist yet.
    composer, records = await _composer(dom_page, render_delay_ms=900)

    result = await _preview(composer, tool)

    assert result["status"] == "preview"
    counters = await _composer_counters(dom_page)
    assert (counters["opener"], counters["commentMenu"], records) == (1, 0, [])


@pytest.mark.parametrize("tool", ["delete_post", "edit_post"])
async def test_a_profile_link_ahead_of_the_post_is_not_its_author(
    dom_page, fast_steps, tool
):
    composer, records = await _composer(dom_page, leading_profile_link=True)

    result = await _preview(composer, tool)

    assert result["status"] == "preview"
    assert records == []


@pytest.mark.parametrize("tool", ["delete_post", "edit_post"])
async def test_another_members_post_is_refused_by_both_tools(
    dom_page, fast_steps, tool
):
    composer, records = await _composer(
        dom_page, author_href="/in/synthetic-other-member/?miniProfileUrn=x"
    )

    result = await _preview(composer, tool)

    assert result["status"] == "not_own_post"
    counters = await _composer_counters(dom_page)
    assert (counters["opener"], records) == (0, [])


@pytest.mark.parametrize("tool", ["delete_post", "edit_post"])
async def test_an_author_named_only_by_profile_id_is_unverified_not_foreign(
    dom_page, fast_steps, tool
):
    # The member is known here by vanity only, so a profile-id link can be
    # neither matched nor contradicted.
    composer, records = await _composer(
        dom_page, author_href="/in/ACoAASyntheticProfileId000000000000/"
    )

    result = await _preview(composer, tool)

    assert result["status"] == "author_unverified"
    counters = await _composer_counters(dom_page)
    assert (counters["opener"], records) == (0, [])
