"""Every tool that can change LinkedIn state is declared, paced and gated as a write.

The inventory is read from the registered server, never from a list kept here,
so a tool added later is held to the same contract the moment it is registered.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linkedin_mcp_server.pacing import ToolKind, classify_tool
from linkedin_mcp_server.server import create_mcp_server

ROOT = Path(__file__).resolve().parents[1]

# Name shapes that change LinkedIn state. A read tool never starts with one.
WRITE_PREFIXES = (
    "accept_",
    "archive_",
    "comment_",
    "connect_",
    "create_",
    "delete_",
    "edit_",
    "follow",
    "mark_",
    "react_",
    "reply_",
    "respond_",
    "save_",
    "send_",
    "unfollow",
    "withdraw_",
)
CONFIRM_PARAMETERS = ("confirm", "confirm_send")
# Drives the browser without touching LinkedIn; pacing exempts it by name.
BROWSER_ONLY = {"close_session"}


def _is_write_name(name: str) -> bool:
    return name.startswith(WRITE_PREFIXES)


@pytest.fixture(scope="module")
def tools() -> dict[str, Any]:
    import asyncio

    registered = asyncio.run(create_mcp_server().list_tools())
    return {tool.name: tool for tool in registered}


def test_the_inventory_contains_the_known_writes(tools):
    writes = {name for name in tools if _is_write_name(name)}

    # Guards the enumeration itself: an empty or renamed inventory would make
    # every other assertion here pass vacuously.
    assert {"send_message", "connect_with_person"} <= writes
    assert len(writes) >= 2


def test_every_write_is_destructive_tagged_and_paced_as_a_write(tools):
    problems = []
    for name, tool in sorted(tools.items()):
        if not _is_write_name(name):
            continue
        tags = set(tool.tags or ())
        if getattr(tool.annotations, "destructiveHint", None) is not True:
            problems.append(f"{name}: destructiveHint is not True")
        if getattr(tool.annotations, "readOnlyHint", None) is True:
            problems.append(f"{name}: readOnlyHint is True")
        if "write" not in tags:
            problems.append(f"{name}: missing the 'write' tag")
        if "local" in tags:
            problems.append(f"{name}: tagged 'local', which skips pacing")
        if classify_tool(name, tool.annotations, tool.tags) is not ToolKind.WRITE:
            problems.append(f"{name}: pacing does not classify it as a write")

    assert problems == []


def test_every_write_requires_an_explicit_boolean_confirmation(tools):
    problems = []
    for name, tool in sorted(tools.items()):
        if not _is_write_name(name):
            continue
        schema = tool.parameters or {}
        properties = schema.get("properties", {})
        present = [p for p in CONFIRM_PARAMETERS if p in properties]
        if len(present) != 1:
            problems.append(f"{name}: expected one of {CONFIRM_PARAMETERS}")
            continue
        parameter = present[0]
        if parameter not in schema.get("required", []):
            problems.append(f"{name}: {parameter} is optional")
        if "default" in properties[parameter]:
            problems.append(f"{name}: {parameter} has a default")
        if properties[parameter].get("type") != "boolean":
            problems.append(f"{name}: {parameter} is not a boolean")

    assert problems == []


def test_read_tools_do_not_claim_to_be_destructive(tools):
    problems = []
    for name, tool in sorted(tools.items()):
        if _is_write_name(name) or name in BROWSER_ONLY:
            continue
        if getattr(tool.annotations, "destructiveHint", None) is True:
            problems.append(f"{name}: a read tool with destructiveHint")
        if "write" in set(tool.tags or ()):
            problems.append(f"{name}: a read tool tagged 'write'")

    assert problems == []


_POST = "urn:li:activity:7300000000000000000"
_COMMENT = "urn:li:comment:(activity:7300000000000000000,7300000000000000101)"

# Writes whose confirm=False answer is documented as browser-free. Previews of
# own-post and scheduled-post edits, and the dry runs of send_message,
# reply_to_conversation, mark_conversation_read, archive_conversation, save_job
# and save_post, read LinkedIn to report the current state without changing it,
# so they are not listed here.
BROWSER_FREE_PREVIEWS: dict[str, dict[str, Any]] = {
    "connect_with_person": {"linkedin_username": "ada-lovelace"},
    "follow": {"target_url": "https://www.linkedin.com/company/analytical-engine/"},
    "withdraw_invitation": {"linkedin_username": "ada-lovelace"},
    "respond_to_invitation": {"linkedin_username": "ada-lovelace", "action": "accept"},
    "comment_on_post": {"post_url": _POST, "text": "Synthetic comment"},
    "reply_to_comment": {
        "post_url": _POST,
        "comment_urn": _COMMENT,
        "text": "Synthetic reply",
    },
    "react_to_comment": {"post_url": _POST, "comment_urn": _COMMENT},
    "create_post": {"text": "Synthetic post body"},
    "create_poll": {
        "question": "Synthetic question?",
        "options": ["Yes", "No"],
        "duration_days": 7,
    },
}


@pytest.mark.parametrize("name", sorted(BROWSER_FREE_PREVIEWS))
async def test_confirm_false_never_acquires_a_browser(name):
    mcp = create_mcp_server()
    tool = await mcp.get_tool(name)
    if tool is None:
        pytest.skip(f"{name} is not registered in this build")

    refuse = AsyncMock(side_effect=AssertionError("a preview acquired a browser"))
    ctx = MagicMock()
    ctx.report_progress = AsyncMock()
    module = tool.fn.__module__
    with (
        patch(f"{module}.get_ready_extractor", refuse, create=True),
        patch("linkedin_mcp_server.dependencies.get_ready_extractor", refuse),
        patch("linkedin_mcp_server.drivers.browser.get_or_create_browser", refuse),
    ):
        result = await tool.fn(ctx=ctx, confirm=False, **BROWSER_FREE_PREVIEWS[name])

    refuse.assert_not_awaited()
    assert result["status"] == "preview"


def _readme_tool_rows() -> list[str]:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    return re.findall(r"^\| `([a-z_]+)` \|", readme, flags=re.MULTILINE)


def test_manifest_lists_every_registered_tool_exactly_once(tools):
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    names = [tool["name"] for tool in manifest["tools"]]

    assert sorted(names) == sorted(set(names)), "duplicate manifest entries"
    assert set(names) == set(tools)


def test_readme_tool_table_lists_every_registered_tool_exactly_once(tools):
    rows = [row for row in _readme_tool_rows() if row in tools]

    assert sorted(rows) == sorted(set(rows)), "duplicate README rows"
    assert set(rows) == set(tools)
