import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from linkedin_mcp_server.debug_trace import (
    _safe_source_profile_dir,
    cleanup_trace_dir,
    get_trace_dir,
    mark_trace_for_retention,
    record_page_trace,
    reset_trace_state_for_testing,
)


def setup_function():
    reset_trace_state_for_testing()


def teardown_function():
    reset_trace_state_for_testing()


def test_get_trace_dir_creates_ephemeral_dir_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))

    trace_dir = get_trace_dir()

    assert trace_dir is not None
    assert trace_dir.exists()
    assert "trace-runs" in str(trace_dir)


def test_cleanup_trace_dir_removes_ephemeral_dir_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))
    trace_dir = get_trace_dir()
    assert trace_dir is not None

    cleanup_trace_dir()

    assert not trace_dir.exists()


def test_mark_trace_for_retention_keeps_trace_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))
    trace_dir = mark_trace_for_retention()
    assert trace_dir is not None

    cleanup_trace_dir()

    assert trace_dir.exists()


def test_explicit_trace_dir_is_preserved(monkeypatch, tmp_path):
    trace_dir = tmp_path / "explicit-trace"
    monkeypatch.setenv("LINKEDIN_DEBUG_TRACE_DIR", str(trace_dir))

    resolved = get_trace_dir()
    assert resolved == trace_dir
    trace_dir.mkdir(parents=True, exist_ok=True)

    cleanup_trace_dir()

    assert trace_dir.exists()


def test_trace_mode_off_disables_trace_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))
    monkeypatch.setenv("LINKEDIN_TRACE_MODE", "off")

    assert get_trace_dir() is None


@pytest.mark.asyncio
async def test_reset_trace_state_resets_step_counter(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))

    page = MagicMock()
    page.url = "https://www.linkedin.com/feed/"
    page.title = AsyncMock(return_value="LinkedIn")
    page.evaluate = AsyncMock(return_value="Feed")
    locator = MagicMock()
    locator.count = AsyncMock(return_value=0)
    page.locator = MagicMock(return_value=locator)
    page.context.cookies = AsyncMock(return_value=[])
    page.screenshot = AsyncMock()

    await record_page_trace(page, "first")
    trace_dir = get_trace_dir()
    assert trace_dir is not None
    first_payload = json.loads((trace_dir / "trace.jsonl").read_text().splitlines()[0])
    assert first_payload["step_id"] == 1

    reset_trace_state_for_testing()
    monkeypatch.setenv("USER_DATA_DIR", str((tmp_path / "second") / "profile"))

    await record_page_trace(page, "first-again")
    second_trace_dir = get_trace_dir()
    assert second_trace_dir is not None
    second_payload = json.loads(
        (second_trace_dir / "trace.jsonl").read_text().splitlines()[0]
    )
    assert second_payload["step_id"] == 1


def test_safe_source_profile_dir_ignores_generic_env_fallback(monkeypatch):
    monkeypatch.setenv("USER_DATA_DIR", "/tmp/unrelated-user-data")
    monkeypatch.setattr(
        "linkedin_mcp_server.debug_trace.get_source_profile_dir",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert _safe_source_profile_dir() == Path("~/.linkedin-mcp/profile").expanduser()


# --- LINKEDIN_TRACE_DOM ------------------------------------------------------

_LEAKS = ("Jane Doe", "Secret title", "Photo of Jane", "Type a name", "hunter2")


def _trace_page(dom_result: object = None, dom_error: Exception | None = None):
    from linkedin_mcp_server.debug_trace import DOM_SKELETON_SCRIPT

    async def evaluate(script, *args):
        if script == DOM_SKELETON_SCRIPT:
            if dom_error is not None:
                raise dom_error
            return dom_result
        return "Feed"

    page = MagicMock()
    page.url = "https://www.linkedin.com/feed/"
    page.title = AsyncMock(return_value="LinkedIn")
    page.evaluate = AsyncMock(side_effect=evaluate)
    locator = MagicMock()
    locator.count = AsyncMock(return_value=0)
    page.locator = MagicMock(return_value=locator)
    page.context.cookies = AsyncMock(return_value=[])
    page.screenshot = AsyncMock()
    return page


def _payload(trace_dir: Path) -> dict:
    return json.loads((trace_dir / "trace.jsonl").read_text().splitlines()[-1])


@pytest.mark.parametrize("value", [None, "", "0", "false", "off", "no", "maybe"])
async def test_dom_trace_is_off_unless_asked_for(monkeypatch, tmp_path, value):
    from linkedin_mcp_server.debug_trace import DOM_SKELETON_SCRIPT

    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))
    if value is None:
        monkeypatch.delenv("LINKEDIN_TRACE_DOM", raising=False)
    else:
        monkeypatch.setenv("LINKEDIN_TRACE_DOM", value)
    page = _trace_page({"nodes": []})

    await record_page_trace(page, "feed")

    trace_dir = get_trace_dir()
    assert trace_dir is not None
    assert not (trace_dir / "dom").exists()
    assert "dom_skeleton" not in _payload(trace_dir)
    scripts = [call.args[0] for call in page.evaluate.await_args_list]
    assert DOM_SKELETON_SCRIPT not in scripts


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
async def test_dom_trace_writes_a_sanitized_skeleton(monkeypatch, tmp_path, value):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))
    monkeypatch.setenv("LINKEDIN_TRACE_DOM", value)
    # A script that misbehaves: every field the schema does not allow, and
    # values where only names belong, must be dropped on the Python side.
    raw = {
        "path": "/feed/",
        "query_params": ["keywords", 7],
        "lang": "en",
        "dir": "ltr",
        "text": "Jane Doe",
        "nodes": [
            {
                "tag": "button",
                "depth": 3,
                "text": "Jane Doe",
                "text_length": 8,
                "aria_attrs": ["aria-label", "Secret title"],
                "aria": {"pressed": "true", "label": "Jane Doe"},
                "aria-label": "Jane Doe",
                "title": "Secret title",
                "alt": "Photo of Jane",
                "placeholder": "Type a name",
                "value": "hunter2",
                "data": {"data-urn": "urn:li:activity:1", "title": "Secret title"},
                "other_attrs": ["title", "aria-label"],
                "href": {
                    "path": "/in/sample/",
                    "params": ["trk"],
                    "query": "trk=hunter2",
                    "same_origin": True,
                },
                "class_count": 2,
                "disabled": True,
            },
            "Jane Doe",
            {"depth": 1},
        ],
    }
    page = _trace_page(raw)

    await record_page_trace(page, "Feed step")

    trace_dir = get_trace_dir()
    assert trace_dir is not None
    path = trace_dir / "dom" / "001-feed-step.json"
    assert _payload(trace_dir)["dom_skeleton"] == str(path)
    text = path.read_text(encoding="utf-8")
    skeleton = json.loads(text)
    for leak in _LEAKS:
        assert leak not in text
    assert skeleton["path"] == "/feed/"
    assert skeleton["query_params"] == ["keywords"]
    assert skeleton["node_count"] == 1
    assert skeleton["nodes"] == [
        {
            "tag": "button",
            "depth": 3,
            "text_length": 8,
            "aria_attrs": ["aria-label"],
            "aria": {"pressed": "true"},
            "data": {"data-urn": "urn:li:activity:1"},
            "other_attrs": ["title"],
            "href": {"path": "/in/sample/", "params": ["trk"], "same_origin": True},
            "class_count": 2,
            "disabled": True,
        }
    ]


async def test_a_failing_dom_capture_is_recorded_and_never_raised(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))
    monkeypatch.setenv("LINKEDIN_TRACE_DOM", "1")
    page = _trace_page(dom_error=RuntimeError("Target page closed\ncall log"))

    await record_page_trace(page, "feed")

    trace_dir = get_trace_dir()
    assert trace_dir is not None
    payload = _payload(trace_dir)
    assert payload["dom_skeleton"] == "<error: RuntimeError: Target page closed>"
    recorded = json.loads((trace_dir / "dom" / "001-feed.json").read_text())
    assert recorded == {
        "schema_version": 1,
        "error": "RuntimeError: Target page closed",
    }
    # The rest of the trace step is still written.
    assert payload["screenshot"].endswith("001-feed.png")


async def test_a_non_object_dom_result_is_recorded_as_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))
    monkeypatch.setenv("LINKEDIN_TRACE_DOM", "1")

    await record_page_trace(_trace_page("Jane Doe"), "feed")

    trace_dir = get_trace_dir()
    assert trace_dir is not None
    assert _payload(trace_dir)["dom_skeleton"].startswith("<error: TypeError")
    assert "Jane Doe" not in (trace_dir / "dom" / "001-feed.json").read_text()
