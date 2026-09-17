"""Best-effort trace capture with on-error retention."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Literal

from linkedin_mcp_server.common_utils import (
    secure_mkdir,
    secure_write_text,
    slugify_fragment,
)
from linkedin_mcp_server.session_state import auth_root_dir, get_source_profile_dir

TraceMode = Literal["off", "on_error", "always"]

_TRACE_COUNTER = itertools.count(1)
_TRACE_DIR: Path | None = None
_TRACE_KEEP = False
_EXPLICIT_TRACE_DIR = False


def _trace_mode() -> TraceMode:
    raw = os.getenv("LINKEDIN_TRACE_MODE", "").strip().lower()
    if raw in {"off", "false", "0", "no"}:
        return "off"
    if raw in {"always", "keep", "persist"}:
        return "always"
    return "on_error"


def _trace_root() -> Path:
    source_profile = _safe_source_profile_dir()
    root = auth_root_dir(source_profile) / "trace-runs"
    secure_mkdir(root)
    return root


def trace_enabled() -> bool:
    return (
        bool(os.getenv("LINKEDIN_DEBUG_TRACE_DIR", "").strip())
        or _trace_mode() != "off"
    )


def get_trace_dir() -> Path | None:
    global _TRACE_DIR, _EXPLICIT_TRACE_DIR

    explicit = os.getenv("LINKEDIN_DEBUG_TRACE_DIR", "").strip()
    if explicit:
        _EXPLICIT_TRACE_DIR = True
        if _TRACE_DIR is None:
            _TRACE_DIR = Path(explicit).expanduser().resolve()
        return _TRACE_DIR

    if _trace_mode() == "off":
        return None

    if _TRACE_DIR is None:
        _TRACE_DIR = Path(
            tempfile.mkdtemp(
                prefix="run-",
                dir=_trace_root(),
            )
        ).resolve()
    return _TRACE_DIR


def mark_trace_for_retention() -> Path | None:
    global _TRACE_KEEP
    trace_dir = get_trace_dir()
    if trace_dir is not None:
        secure_mkdir(trace_dir)
        _TRACE_KEEP = True
    return trace_dir


def should_keep_traces() -> bool:
    return _EXPLICIT_TRACE_DIR or _TRACE_KEEP or _trace_mode() == "always"


def cleanup_trace_dir() -> None:
    global _TRACE_DIR, _TRACE_KEEP, _EXPLICIT_TRACE_DIR

    trace_dir = _TRACE_DIR
    if trace_dir is None or should_keep_traces():
        return
    try:
        shutil.rmtree(trace_dir)
    except OSError:
        return
    _TRACE_DIR = None
    _TRACE_KEEP = False
    _EXPLICIT_TRACE_DIR = False


def reset_trace_state_for_testing() -> None:
    global _TRACE_COUNTER, _TRACE_DIR, _TRACE_KEEP, _EXPLICIT_TRACE_DIR
    _TRACE_COUNTER = itertools.count(1)
    _TRACE_DIR = None
    _TRACE_KEEP = False
    _EXPLICIT_TRACE_DIR = False


def _slugify_step(step: str) -> str:
    return slugify_fragment(step)


# --- DOM skeleton ------------------------------------------------------------
#
# LINKEDIN_TRACE_DOM writes, beside each trace screenshot, the *structure* of
# the page: what a selector fix needs to see of LinkedIn's real markup, and
# nothing a reader of the trace could learn a person from. So the walk emits
# tags, roles, attribute names, a handful of state-attribute values, data-*
# values and link path shapes, and never text: not node text (only its length),
# and not aria-label, title, alt, placeholder or value, which is where names
# and message snippets live. The Python side re-checks the result against an
# allowlist, so a script change that starts leaking a field is dropped here
# rather than written to disk.

_TRUTHY = frozenset({"1", "true", "yes", "on"})

DOM_SKELETON_MAX_DEPTH = 40
DOM_SKELETON_MAX_NODES = 15000
DOM_SKELETON_MAX_VALUE = 120

#: aria-* attributes whose *value* is a state or a reference, never prose.
DOM_SKELETON_ARIA_VALUES = (
    "aria-pressed",
    "aria-expanded",
    "aria-checked",
    "aria-haspopup",
    "aria-controls",
    "aria-selected",
)

DOM_SKELETON_SCRIPT = """
({maxDepth, maxNodes, maxValue, ariaValues}) => {
  const cut = (value) => String(value).slice(0, maxValue);
  const names = (params) => [...new Set([...params.keys()])].sort();
  const SKIP = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "LINK", "META"]);
  const LEAF = new Set(["SVG", "TEMPLATE", "IFRAME", "OBJECT", "CANVAS"]);
  const FORM = new Set(["INPUT", "TEXTAREA", "SELECT", "BUTTON"]);
  const COVERED = new Set([
    "id", "role", "class", "style", "href", "type", "name",
    "contenteditable", "disabled",
  ]);
  const hrefShape = (raw) => {
    let url;
    try {
      url = new URL(raw, location.href);
    } catch (error) {
      return {invalid: true};
    }
    if (url.protocol !== "http:" && url.protocol !== "https:") {
      return {scheme: url.protocol.replace(":", "")};
    }
    return {
      path: cut(url.pathname),
      params: names(url.searchParams),
      same_origin: url.origin === location.origin,
    };
  };
  const describe = (el, depth) => {
    const node = {tag: el.tagName.toLowerCase(), depth};
    if (el.id) node.id = cut(el.id);
    const role = el.getAttribute("role");
    if (role !== null) node.role = cut(role);
    const aria = [];
    const ariaState = {};
    const data = {};
    const other = [];
    for (const attr of el.attributes) {
      const name = attr.name.toLowerCase();
      if (name.startsWith("aria-")) {
        aria.push(name);
        if (ariaValues.includes(name)) ariaState[name.slice(5)] = cut(attr.value);
      } else if (name.startsWith("data-")) {
        data[name] = cut(attr.value);
      } else if (!COVERED.has(name)) {
        other.push(name);
      }
    }
    if (aria.length) node.aria_attrs = aria.sort();
    if (Object.keys(ariaState).length) node.aria = ariaState;
    if (Object.keys(data).length) node.data = data;
    if (other.length) node.other_attrs = other.sort();
    const href = el.getAttribute("href");
    if (href !== null) node.href = hrefShape(href);
    if (FORM.has(el.tagName)) {
      const type = el.getAttribute("type");
      const name = el.getAttribute("name");
      if (type !== null) node.type = cut(type);
      if (name !== null) node.name = cut(name);
    }
    if (el.classList.length) node.class_count = el.classList.length;
    const editable = el.getAttribute("contenteditable");
    if (editable !== null) node.contenteditable = cut(editable);
    if (el.hasAttribute("disabled")) node.disabled = true;
    let textLength = 0;
    for (const child of el.childNodes) {
      if (child.nodeType === Node.TEXT_NODE) {
        textLength += child.nodeValue.replace(/\\s+/g, " ").trim().length;
      }
    }
    if (textLength) node.text_length = textLength;
    return node;
  };

  const page = {
    path: location.pathname,
    query_params: names(new URLSearchParams(location.search)),
    lang: document.documentElement.lang || null,
    dir: document.documentElement.dir || document.dir || null,
    truncated_depth: false,
    truncated_nodes: false,
    nodes: [],
  };
  if (!document.body) return page;
  const stack = [[document.body, 0]];
  while (stack.length) {
    const [el, depth] = stack.pop();
    if (page.nodes.length >= maxNodes) {
      page.truncated_nodes = true;
      break;
    }
    page.nodes.push(describe(el, depth));
    if (LEAF.has(el.tagName.toUpperCase())) continue;
    const children = [...el.children].filter((child) => !SKIP.has(child.tagName.toUpperCase()));
    if (!children.length) continue;
    if (depth + 1 > maxDepth) {
      page.truncated_depth = true;
      continue;
    }
    for (let i = children.length - 1; i >= 0; i -= 1) {
      stack.push([children[i], depth + 1]);
    }
  }
  return page;
}
"""

_NODE_STRING_FIELDS = frozenset(
    {"tag", "id", "role", "type", "name", "contenteditable"}
)
_NODE_INT_FIELDS = frozenset({"depth", "class_count", "text_length"})
_ARIA_STATE_KEYS = frozenset(name[len("aria-") :] for name in DOM_SKELETON_ARIA_VALUES)


def dom_trace_enabled() -> bool:
    """Whether ``LINKEDIN_TRACE_DOM`` asks for DOM skeletons (off by default)."""
    return os.getenv("LINKEDIN_TRACE_DOM", "").strip().lower() in _TRUTHY


def _cut(value: Any) -> str:
    return str(value)[:DOM_SKELETON_MAX_VALUE]


def _names(value: Any, prefix: str = "") -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted(
        {
            _cut(item)
            for item in value
            if isinstance(item, str) and item.startswith(prefix)
        }
    )


def _sanitize_href(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if value.get("invalid") is True:
        return {"invalid": True}
    if isinstance(value.get("scheme"), str):
        return {"scheme": _cut(value["scheme"])}
    if not isinstance(value.get("path"), str):
        return None
    return {
        "path": _cut(value["path"]),
        "params": _names(value.get("params")),
        "same_origin": value.get("same_origin") is True,
    }


def _sanitize_node(raw: Any) -> dict[str, Any] | None:
    """Keep only allowlisted fields, so no text can reach the trace file."""
    if not isinstance(raw, dict) or not isinstance(raw.get("tag"), str):
        return None
    node: dict[str, Any] = {}
    for key in _NODE_STRING_FIELDS:
        if isinstance(raw.get(key), str):
            node[key] = _cut(raw[key])
    for key in _NODE_INT_FIELDS:
        value = raw.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            node[key] = value
    if raw.get("disabled") is True:
        node["disabled"] = True
    if aria_attrs := _names(raw.get("aria_attrs"), "aria-"):
        node["aria_attrs"] = aria_attrs
    if isinstance(raw.get("aria"), dict):
        state = {
            key: _cut(value)
            for key, value in raw["aria"].items()
            if key in _ARIA_STATE_KEYS and isinstance(value, str)
        }
        if state:
            node["aria"] = state
    if isinstance(raw.get("data"), dict):
        data = {
            _cut(key): _cut(value)
            for key, value in raw["data"].items()
            if isinstance(key, str) and key.startswith("data-")
        }
        if data:
            node["data"] = data
    other = [
        name
        for name in _names(raw.get("other_attrs"))
        if not name.startswith(("aria-", "data-"))
    ]
    if other:
        node["other_attrs"] = other
    if (href := _sanitize_href(raw.get("href"))) is not None:
        node["href"] = href
    return node


def sanitize_dom_skeleton(raw: Any) -> dict[str, Any]:
    """Reduce an evaluate result to the documented skeleton schema."""
    if not isinstance(raw, dict):
        raise TypeError(f"DOM skeleton script returned {type(raw).__name__}")
    raw_nodes = raw.get("nodes")
    nodes = [
        node
        for node in (
            _sanitize_node(item)
            for item in (raw_nodes if isinstance(raw_nodes, list) else [])
        )
        if node is not None
    ][:DOM_SKELETON_MAX_NODES]
    return {
        "schema_version": 1,
        "path": _cut(raw["path"]) if isinstance(raw.get("path"), str) else None,
        "query_params": _names(raw.get("query_params")),
        "lang": _cut(raw["lang"]) if isinstance(raw.get("lang"), str) else None,
        "dir": _cut(raw["dir"]) if isinstance(raw.get("dir"), str) else None,
        "truncated_depth": raw.get("truncated_depth") is True,
        "truncated_nodes": raw.get("truncated_nodes") is True,
        "node_count": len(nodes),
        "nodes": nodes,
    }


def _skeleton_json(skeleton: dict[str, Any]) -> str:
    """One node per line: valid JSON that still greps and diffs usefully."""
    header = {key: value for key, value in skeleton.items() if key != "nodes"}
    lines = [json.dumps(header, ensure_ascii=True)[:-1] + ', "nodes": [']
    nodes = skeleton["nodes"]
    for index, node in enumerate(nodes):
        suffix = "," if index < len(nodes) - 1 else ""
        lines.append(json.dumps(node, ensure_ascii=True, sort_keys=True) + suffix)
    lines.append("]}")
    return "\n".join(lines) + "\n"


async def _record_dom_skeleton(page: Any, path: Path) -> str:
    """Write the page's DOM skeleton to *path*; never raises.

    Returns the file path, or ``<error: ...>`` naming what failed. A failed
    capture still leaves a file saying so when the directory is writable, so a
    missing skeleton is never mistaken for a step that was not traced.
    """
    try:
        raw = await page.evaluate(
            DOM_SKELETON_SCRIPT,
            {
                "maxDepth": DOM_SKELETON_MAX_DEPTH,
                "maxNodes": DOM_SKELETON_MAX_NODES,
                "maxValue": DOM_SKELETON_MAX_VALUE,
                "ariaValues": list(DOM_SKELETON_ARIA_VALUES),
            },
        )
        content = _skeleton_json(sanitize_dom_skeleton(raw))
        result = str(path)
    except Exception as exc:
        # The type and the first line only: an evaluate error can quote the
        # script, never the page, but there is no reason to keep more.
        lines = str(exc).splitlines()
        error = f"{type(exc).__name__}: {lines[0][:200] if lines else ''}"
        content = json.dumps({"schema_version": 1, "error": error}) + "\n"
        result = f"<error: {error}>"
    try:
        secure_write_text(path, content)
    except Exception as exc:
        return f"<error: {type(exc).__name__}: could not write DOM skeleton>"
    return result


def _safe_source_profile_dir() -> Path:
    try:
        return get_source_profile_dir()
    except Exception:
        return Path("~/.linkedin-mcp/profile").expanduser()


async def record_page_trace(
    page: Any, step: str, *, extra: dict[str, Any] | None = None
) -> None:
    """Persist a screenshot and basic page state when trace capture is enabled."""
    trace_dir = get_trace_dir()
    if trace_dir is None:
        return

    secure_mkdir(trace_dir)
    screenshot_dir = trace_dir / "screens"
    secure_mkdir(screenshot_dir)
    step_id = next(_TRACE_COUNTER)
    slug = _slugify_step(step) or "step"

    try:
        title = await page.title()
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        title = f"<error: {exc}>"

    try:
        body_text = await page.evaluate("() => document.body?.innerText || ''")
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        body_text = f"<error: {exc}>"

    if not isinstance(body_text, str):
        body_text = ""

    try:
        remember_me = (await page.locator("#rememberme-div").count()) > 0
    except Exception:  # pragma: no cover - best effort diagnostics
        remember_me = False

    try:
        cookies = await page.context.cookies()
    except Exception:  # pragma: no cover - best effort diagnostics
        cookies = []

    linkedin_cookie_names = sorted(
        {
            cookie["name"]
            for cookie in cookies
            if "linkedin.com" in cookie.get("domain", "")
        }
    )

    screenshot_path = screenshot_dir / f"{step_id:03d}-{slug}.png"
    screenshot: str | None = None
    try:
        await page.screenshot(path=str(screenshot_path), full_page=True)
        screenshot = str(screenshot_path)
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        screenshot = f"<error: {exc}>"

    payload = {
        "step_id": step_id,
        "step": step,
        "url": getattr(page, "url", ""),
        "title": title,
        "remember_me": remember_me,
        "body_length": len(body_text),
        "body_marker": " ".join(body_text.split())[:200],
        "linkedin_cookie_names": linkedin_cookie_names,
        "screenshot": screenshot,
        "extra": extra or {},
    }
    if dom_trace_enabled():
        payload["dom_skeleton"] = await _record_dom_skeleton(
            page, trace_dir / "dom" / f"{step_id:03d}-{slug}.json"
        )

    trace_jsonl = trace_dir / "trace.jsonl"
    try:
        fd = os.open(str(trace_jsonl), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
    except FileExistsError:
        pass
    with trace_jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
