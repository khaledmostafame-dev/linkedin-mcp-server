"""Browser-DOM tests for the LINKEDIN_TRACE_DOM skeleton script.

The page is synthetic and served from a made-up origin, so this is a claim
about the walk only: what it keeps (structure, attribute names, state values,
data-* values, link path shapes) and what it must never keep (text, labels,
query values), not about LinkedIn's markup.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from patchright.async_api import async_playwright

from linkedin_mcp_server.debug_trace import (
    DOM_SKELETON_ARIA_VALUES,
    DOM_SKELETON_MAX_DEPTH,
    DOM_SKELETON_MAX_VALUE,
    DOM_SKELETON_SCRIPT,
    get_trace_dir,
    record_page_trace,
    reset_trace_state_for_testing,
    sanitize_dom_skeleton,
)

pytestmark = [
    pytest.mark.browser_dom,
    pytest.mark.xdist_group("browser_runtime"),
]

ORIGIN = "https://skeleton.test"
URL = f"{ORIGIN}/search/results/people/?keywords=Jane%20Doe&origin=GLOBAL"
SECRETS = (
    "Jane",
    "Doe",
    "Secret title",
    "Photo alt",
    "Type here",
    "hunter2",
    "tracking-secret",
    "person@example.com",
    "console.log",
    "circle",
)


def _html() -> str:
    nested = (
        "<div>" * (DOM_SKELETON_MAX_DEPTH + 5)
        + "deep"
        + "</div>" * (DOM_SKELETON_MAX_DEPTH + 5)
    )
    return f"""<!DOCTYPE html>
<html lang="ar" dir="rtl"><head><meta charset="utf-8"><title>Jane Doe</title>
<style>.a {{ color: red }}</style></head>
<body>
<main id="workspace" class="a b c">
  <h1>Jane Doe</h1>
  <button aria-label="Follow Jane Doe" aria-pressed="true" aria-expanded="false"
          aria-describedby="tip" title="Secret title" data-urn="urn:li:activity:1"
          type="button" class="x" disabled>Follow Jane Doe</button>
  <a href="/in/sample-slug-123/?miniProfileUrn=tracking-secret&amp;trk=hunter2#frag"
     aria-haspopup="menu" aria-controls="menu-1">Jane Doe</a>
  <a href="mailto:person@example.com">Mail Jane</a>
  <a href="https://other.test/feed/update/urn:li:activity:2/?x=hunter2">x</a>
  <img alt="Photo alt" src="/img.png">
  <input type="text" name="q" placeholder="Type here" value="hunter2">
  <div role="textbox" contenteditable="true" aria-label="Jane Doe">Jane Doe</div>
  <svg viewBox="0 0 1 1"><circle r="1"></circle></svg>
  <script>console.log("Jane Doe")</script>
</main>
<section id="nest">{nested}</section>
</body></html>"""


@pytest.fixture
async def dom_page():
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(channel="chromium", headless=True)
            page = await browser.new_page()
        except Exception as exc:  # browser binary missing
            pytest.skip(f"chromium unavailable: {exc}")
        try:

            async def handle(route: Any) -> None:
                await route.fulfill(
                    content_type="text/html; charset=utf-8", body=_html()
                )

            # Every request is answered here; nothing leaves the machine.
            await page.route("**/*", handle)
            await page.goto(URL)
            yield page
        finally:
            await browser.close()


def _args(**overrides: Any) -> dict[str, Any]:
    return {
        "maxDepth": DOM_SKELETON_MAX_DEPTH,
        "maxNodes": 15000,
        "maxValue": DOM_SKELETON_MAX_VALUE,
        "ariaValues": list(DOM_SKELETON_ARIA_VALUES),
        **overrides,
    }


async def _skeleton(page: Any, **overrides: Any) -> dict[str, Any]:
    return await page.evaluate(DOM_SKELETON_SCRIPT, _args(**overrides))


async def test_the_raw_script_output_carries_no_text_or_label_values(dom_page):
    raw = await _skeleton(dom_page)
    serialized = json.dumps(raw)

    for secret in SECRETS:
        assert secret not in serialized, secret
    assert raw["path"] == "/search/results/people/"
    assert raw["query_params"] == ["keywords", "origin"]
    assert (raw["lang"], raw["dir"]) == ("ar", "rtl")
    # The sanitizer is a second line, not the only one: it keeps all of this.
    assert sanitize_dom_skeleton(raw)["nodes"] == raw["nodes"]


async def test_structure_state_and_link_shapes_are_kept(dom_page):
    nodes = (await _skeleton(dom_page))["nodes"]
    by_tag: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        by_tag.setdefault(node["tag"], []).append(node)

    assert nodes[0] == {"tag": "body", "depth": 0}
    main = by_tag["main"][0]
    assert (main["id"], main["class_count"], main["depth"]) == ("workspace", 3, 1)
    assert by_tag["h1"][0]["text_length"] == len("Jane Doe")

    button = by_tag["button"][0]
    assert button["aria_attrs"] == [
        "aria-describedby",
        "aria-expanded",
        "aria-label",
        "aria-pressed",
    ]
    assert button["aria"] == {"pressed": "true", "expanded": "false"}
    assert button["data"] == {"data-urn": "urn:li:activity:1"}
    assert button["other_attrs"] == ["title"]
    assert (button["type"], button["disabled"]) == ("button", True)

    profile, mail, external = by_tag["a"]
    assert profile["href"] == {
        "path": "/in/sample-slug-123/",
        "params": ["miniProfileUrn", "trk"],
        "same_origin": True,
    }
    assert profile["aria"] == {"haspopup": "menu", "controls": "menu-1"}
    assert mail["href"] == {"scheme": "mailto"}
    assert external["href"] == {
        "path": "/feed/update/urn:li:activity:2/",
        "params": ["x"],
        "same_origin": False,
    }

    assert by_tag["img"][0]["other_attrs"] == ["alt", "src"]
    field = by_tag["input"][0]
    assert (field["type"], field["name"]) == ("text", "q")
    assert "value" in field["other_attrs"] and "placeholder" in field["other_attrs"]
    textbox = by_tag["div"][0]
    assert (textbox["role"], textbox["contenteditable"]) == ("textbox", "true")

    # svg is a leaf, script and style are skipped outright.
    assert "svg" in by_tag and "circle" not in by_tag
    assert "script" not in by_tag and "style" not in by_tag


async def test_depth_and_node_caps_are_reported(dom_page):
    deep = await _skeleton(dom_page)
    assert deep["truncated_depth"] is True
    assert deep["truncated_nodes"] is False
    assert max(node["depth"] for node in deep["nodes"]) == DOM_SKELETON_MAX_DEPTH

    capped = await _skeleton(dom_page, maxNodes=5)
    assert capped["truncated_nodes"] is True
    assert len(capped["nodes"]) == 5


async def test_record_page_trace_writes_the_skeleton_file(
    dom_page, monkeypatch, tmp_path
):
    reset_trace_state_for_testing()
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))
    monkeypatch.setenv("LINKEDIN_TRACE_DOM", "true")

    await record_page_trace(dom_page, "people search")

    trace_dir = get_trace_dir()
    assert trace_dir is not None
    path = trace_dir / "dom" / "001-people-search.json"
    text = path.read_text(encoding="utf-8")
    skeleton = json.loads(text)
    assert skeleton["schema_version"] == 1
    assert skeleton["node_count"] == len(skeleton["nodes"]) > 10
    for secret in SECRETS:
        assert secret not in text, secret
