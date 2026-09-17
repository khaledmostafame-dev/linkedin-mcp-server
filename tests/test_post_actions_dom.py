"""Browser-DOM tests for the post-page probes of save_post and get_post_reactions.

The unit suite mocks ``page.evaluate``, so the programs in
``scraping/post_actions.py`` and ``scraping/reactions.py`` never execute
there. These tests run the real ones in headless chromium.

The page around the controls is a claim about LinkedIn: its shape mirrors
the structure of a signed-in, English-UI post page captured on 2026-09-17
(structure only, every name, slug and id here is synthetic). What it keeps
is what defeated the earlier probes: dozens of ``button[aria-expanded]``
(the post's menu opener with its collapsed panel beside it, the reactor
facepile's "see more", the Like picker, Repost, the emoji picker, the
comment sort control and one menu opener per comment), action-row
controls wrapped one per span/div and carrying ``data-finite-scroll-hotkey``,
and a single ``data-reaction-details`` button that comments do not carry.

The open menu and the reactions dialog are a claim about the algorithm
only: neither was in the capture. The menu items follow the shape
``post_composer.py`` ports from upstream (``[role="button"]`` items with
``data-test-icon`` hooks), and a click on an item closes the menu, which
is what forces ``save_post`` to reopen it before confirming.

Every fixture renders with English labels and with opaque tokens; a
decision that differs between them read a word.
"""

from __future__ import annotations

from typing import Any, cast

import asyncio

import pytest
from patchright.async_api import Page, async_playwright

from linkedin_mcp_server.scraping.post_actions import PostActions
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
            ]
        )
    },
}


def _comment(index: int, label: dict[str, str]) -> str:
    return f"""
<article id="c{index}" data-id="urn:li:comment:(activity:7000000000000000001,{index})">
  <div><a aria-label="{label["comment"]}" href="/in/synthetic-commenter-{index}/">
    <h3><span>Commenter {index}</span></h3></a></div>
  <div><time>1d</time><div><div id="cm{index}">
    <button id="cmb{index}" aria-expanded="false" type="button" tabindex="0">
      <svg role="img" aria-hidden="true" aria-label="{label["more"]}"
        data-test-icon="overflow-web-ios-small"></svg></button>
    <div id="cmp{index}" aria-hidden="true" tabindex="-1"></div>
  </div></div></div>
  <div><section><span dir="ltr">A synthetic comment.</span></section></div>
  <div><span>
    <button aria-label="{label["like"]}" aria-pressed="false"><span>Like</span></button>
    <button aria-label="{label["picker"]}" aria-expanded="false"
      data-finite-scroll-hotkey="l" tabindex="0"><svg data-test-icon="caret-small"></svg></button>
  </span><span></span>
  <button aria-label="{label["reactions"]}" type="button"><img alt=""><span aria-hidden="true">1</span></button>
  <div><button aria-label="{label["comment"]}"><span>Reply</span></button></div></div>
</article>"""


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
    comments = "".join(_comment(index, label) for index in range(1, 4))
    save_item = f"item(bookmark, {label['save']!r})"
    save_items = f"{save_item} + {save_item}" if second_save_item else save_item
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"></head>
<body dir="ltr"><div id="artdeco-toasts__wormhole"></div>{dialog}
<div><div class="application-outlet"><main aria-label="Main Feed">
<div aria-label="Feed update"><section>
<a aria-label="{label["menu"]}" data-test-boost-header="" href="/ad-beta/">Boost</a>
<div data-view-name="feed-full-update"><div>
<div id="post" role="article" data-urn="urn:li:activity:7000000000000000001"><div><div>
  <h2>Feed post</h2>
  <div><div><div><div>
    <a aria-label="{label["comment"]}" href="/in/synthetic-author/"><img alt=""></a>
    <div><a aria-label="{label["comment"]}" href="/in/synthetic-author/"><span>Synthetic Author</span></a></div>
  </div></div>
  <div><div id="menu-root">
    <button id="opener" aria-expanded="false" aria-label="{label["menu"]}" type="button" tabindex="0">
      <svg width="24" height="24"></svg></button>
    <div id="panel" aria-hidden="true" aria-label="{label["panel"]}" tabindex="-1"></div>
  </div>{extra_opener}</div></div>
  <div tabindex="-1"><div dir="ltr"><span dir="ltr">A synthetic post body.</span></div></div>
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
  <div><span>
    <button aria-label="{label["like"]}" aria-pressed="false"><span>Like</span></button>
    <button aria-expanded="false" aria-label="{label["picker"]}" data-finite-scroll-hotkey="l"
      tabindex="0"><span><svg data-test-icon="caret-small"></svg></span></button>
  </span>
  <span tabindex="-1"><div><button role="button" aria-label="{label["comment"]}"
    data-finite-scroll-hotkey="c" tabindex="0"><span>Comment</span></button></div></span>
  <div><span tabindex="-1"><button aria-expanded="false" aria-label="{label["repost"]}"
    type="button" data-finite-scroll-hotkey="r" tabindex="0"><span>Repost</span></button>
    <div aria-hidden="true" tabindex="-1"></div></span></div>
  <div><span tabindex="-1"><button aria-label="{label["send"]}" type="button"
    data-finite-scroll-hotkey="s"><span>Send</span></button></span></div>
  </div></div>
  <div><div><div data-scroll-name="true"><form>
    <div role="textbox" aria-label="{label["comment"]}" aria-multiline="true"
      contenteditable="true" data-test-ql-editor-contenteditable="true"><p><br></p></div>
    <span><button aria-controls="emoji-panel" aria-expanded="false" aria-label="{label["picker"]}"
      type="button" data-reaction-summary__emoji-hoverable=""><svg></svg></button></span>
  </form></div>
  <div><div><button aria-expanded="false" type="button" tabindex="0"><span>Most relevant</span></button>
    <div aria-hidden="true" tabindex="-1"></div></div></div>
  <div>{comments}</div></div></div>
</div></div></div></div></section></div>
</main></div></div>
<output id="counters" hidden data-opener="0" data-save="0" data-reactions="0"></output>
<script>
  // Page globals are invisible to patchright's isolated evaluate world, so
  // the counters live in the DOM.
  const counters = document.getElementById('counters');
  function count(name) {{
    counters.dataset[name] = String(Number(counters.dataset[name]) + 1);
  }}
  let saved = {"true" if saved else "false"};
  const opener = document.getElementById('opener');
  const panel = document.getElementById('panel');
  function item(glyph, text) {{
    return '<li><div role="button" tabindex="0" class="save-item">'
      + '<svg role="none" aria-hidden="true" data-test-icon="' + glyph + '"></svg>'
      + '<div><h5 role="none">' + text + '</h5></div></div></li>';
  }}
  function render() {{
    const bookmark = saved ? 'bookmark-fill-medium' : 'bookmark-outline-medium';
    panel.innerHTML = '<div><ul>'
      + {save_items}
      + '<li><div role="button" tabindex="0"><svg data-test-icon="link-medium"></svg>'
      + '<div><h5 role="none">' + {label["copy"]!r} + '</h5></div></div></li>'
      + '</ul></div>';
  }}
  function close() {{
    opener.setAttribute('aria-expanded', 'false');
    panel.setAttribute('aria-hidden', 'true');
    panel.innerHTML = '';
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
    if (!event.target.closest('.save-item')) return;
    count('save');
    saved = !saved;
    close();
  }});
  document.getElementById('reactions').addEventListener('click', () => {{
    count('reactions');
    const dialog = document.createElement('div');
    dialog.setAttribute('role', 'dialog');
    dialog.innerHTML = '<ul><li><a href="/in/synthetic-reactor-1/">Reactor</a></li></ul>';
    document.body.appendChild(dialog);
  }});
</script>
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
