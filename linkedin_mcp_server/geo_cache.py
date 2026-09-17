"""Persisted cache of resolved LinkedIn geo URN lookups.

``geo_resolver.py`` drives a full jobs-search-page typeahead to turn a free-
text place name into LinkedIn's numeric geo id, and every drive of it costs
account-safety budget the same way any other LinkedIn navigation does (see
that module's docstring). Once a query has been resolved unambiguously, the
answer is unlikely to change -- LinkedIn's geo taxonomy is effectively
static -- so it is worth remembering by normalized query text rather than
resolved again on every repeat of the same search.

This module owns only the *storage*: a key-value cache with a TTL, backed by
``geo-resolution-cache.json`` beside ``pacing-state.json`` in the auth root
(see ``session_state.auth_root_dir``), following the same fail-open shape as
``pacing.py``'s ``PacingStateStore`` -- a missing or corrupt file starts
fresh with a warning, because losing this cache costs navigations, never
correctness.

Deliberately outside ``linkedin_mcp_server.scraping``: that package's own
dependency-layering test (``tests/scraping/test_facade_structure.py``)
refuses any of its modules importing ``session_state`` or the other daemon/
profile/session-owning modules directly, precisely so that kind of process
state stays reachable from one place. ``geo_resolver.py`` normalizes its own
query text (see its ``_normalize``) and passes the already-normalized string
here as ``key`` -- this module does not know or care that a key came from a
place name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import json
import logging
import time
from collections.abc import Callable
from typing import Any, cast

from linkedin_mcp_server.common_utils import secure_write_text

logger = logging.getLogger(__name__)

#: Kept beside ``pacing-state.json`` in the auth root so a restart does not
#: lose it.
GEO_CACHE_FILE = "geo-resolution-cache.json"
_GEO_CACHE_VERSION = 1
#: How long a cached resolution is trusted before it is resolved again.
#: LinkedIn's geo taxonomy is effectively static, but nothing here promises
#: it never changes, so this is a refresh interval rather than "forever".
GEO_CACHE_TTL_SECONDS = 30 * 24 * 3600.0


@dataclass(frozen=True)
class CachedGeoResolution:
    """One persisted resolution: a normalized query's confirmed answer."""

    name: str
    geo_urn_id: str
    resolved_at: float


def default_geo_cache_path() -> Path:
    # Imported here, not at module level: session_state pulls in the config
    # singleton at import (see pacing.py's ``_default_state_path``, the same
    # shape for the same reason -- and unlike pacing.py, this module could
    # import it at module level without a layering violation, but keeping
    # the same shape means one fewer thing to explain differently).
    from linkedin_mcp_server.session_state import auth_root_dir

    return auth_root_dir() / GEO_CACHE_FILE


class GeoResolutionCache:
    """``geo-resolution-cache.json``: fail open for the file, never guess.

    One entry per key, holding the name and id LinkedIn's typeahead
    confirmed and when. A missing or corrupt file starts fresh with a
    warning, exactly like pacing state.
    """

    def __init__(
        self,
        path: Callable[[], Path] = default_geo_cache_path,
        *,
        clock: Callable[[], float] = time.time,
        ttl_seconds: float = GEO_CACHE_TTL_SECONDS,
    ) -> None:
        self._path = path
        self._clock = clock
        self._ttl_seconds = ttl_seconds

    def path(self) -> Path:
        return self._path()

    def get(self, key: str) -> CachedGeoResolution | None:
        """The cached entry for ``key``, or ``None`` on a miss.

        A miss covers everything that is not a fresh, well-formed hit: no
        entry, a corrupt file, or an entry older than the TTL -- the last of
        those is not distinguished from "never cached" because the caller's
        only correct response to either is to resolve it live again.
        """
        entry = self._load().get(key)
        if entry is None:
            return None
        if self._clock() - entry.resolved_at > self._ttl_seconds:
            return None
        return entry

    def put(self, key: str, *, name: str, geo_urn_id: str) -> None:
        """Remember ``name``/``geo_urn_id`` as the confirmed answer for ``key``."""
        entries = self._load()
        entries[key] = CachedGeoResolution(
            name=name, geo_urn_id=geo_urn_id, resolved_at=self._clock()
        )
        self._save(entries)

    def _load(self) -> dict[str, CachedGeoResolution]:
        try:
            raw = self._path().read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:
            logger.warning(
                "Could not read geo resolution cache, starting fresh: %s", exc
            )
            return {}
        try:
            return self._parse(json.loads(raw))
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning(
                "Ignoring corrupt geo resolution cache at %s, starting fresh: %s",
                self._path(),
                exc,
            )
            return {}

    @staticmethod
    def _parse(data: object) -> dict[str, CachedGeoResolution]:
        if not isinstance(data, dict):
            raise ValueError("geo resolution cache is not an object")
        payload = cast(dict[str, Any], data)
        if payload.get("version") != _GEO_CACHE_VERSION:
            raise ValueError("unknown geo resolution cache version")
        raw_entries = payload.get("entries", {})
        if not isinstance(raw_entries, dict):
            raise ValueError("entries is not an object")
        parsed: dict[str, CachedGeoResolution] = {}
        for key, value in cast(dict[str, Any], raw_entries).items():
            if not isinstance(value, dict):
                raise ValueError(f"entry {key!r} is not an object")
            name = value["name"]
            geo_urn_id = value["geo_urn_id"]
            resolved_at = value["resolved_at"]
            if not isinstance(name, str) or not isinstance(geo_urn_id, str):
                raise ValueError(f"entry {key!r} has a non-string name or id")
            if isinstance(resolved_at, bool) or not isinstance(
                resolved_at, (int, float)
            ):
                raise ValueError(f"entry {key!r} has a non-numeric resolved_at")
            parsed[key] = CachedGeoResolution(
                name=name, geo_urn_id=geo_urn_id, resolved_at=float(resolved_at)
            )
        return parsed

    def _save(self, entries: dict[str, CachedGeoResolution]) -> None:
        payload = {
            "version": _GEO_CACHE_VERSION,
            "entries": {
                key: {
                    "name": entry.name,
                    "geo_urn_id": entry.geo_urn_id,
                    "resolved_at": entry.resolved_at,
                }
                for key, entry in entries.items()
            },
        }
        try:
            secure_write_text(self._path(), json.dumps(payload, sort_keys=True) + "\n")
        except OSError as exc:
            logger.warning(
                "Could not persist geo resolution cache; this resolution stays "
                "in memory only: %s",
                exc,
            )
