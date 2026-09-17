"""A Chromium singleton lock nobody can still hold is cleared before a launch.

Incident this covers: a container recreated while its Chromium was running left
``SingletonLock -> <old-hostname>-<pid>`` in the mounted profile. The new
container has a new hostname, Chromium refuses a lock from "another computer"
without checking whether it is still there, and every tool call then failed as
"Network error. Check your connection" until the links were deleted by hand.
"""

from __future__ import annotations

import logging
import socket
import sys
from pathlib import Path
from unittest import mock

import pytest
from fastmcp.exceptions import ToolError

from linkedin_mcp_server.core.browser import BrowserManager
from linkedin_mcp_server.core.exceptions import NetworkError
from linkedin_mcp_server.error_handler import raise_tool_error
from linkedin_mcp_server.exceptions import ProfileLockedError, ProfileRootRefusedError
from linkedin_mcp_server.profile_lease import get_profile_lease
from linkedin_mcp_server.session_state import (
    clear_stale_chromium_singleton,
    runtime_profiles_root,
)

_SINGLETONS = ("SingletonLock", "SingletonSocket", "SingletonCookie")

#: What Playwright put into the launch error on the NAS, stderr included.
_INCIDENT_ERROR = (
    "BrowserType.launch_persistent_context: Target page, context or browser has "
    "been closed\nBrowser logs:\n\n[pid=57][err] [0917/081512.330:ERROR:"
    "process_singleton_posix.cc(358)] The profile appears to be in use by another "
    "Chromium process (39) on another computer (d0dda23ca75e). Chromium has "
    "locked the profile so that it doesn't get corrupted."
)


def _plant(profile: Path, lock_target: str) -> None:
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "SingletonLock").symlink_to(lock_target)
    (profile / "SingletonSocket").symlink_to(
        "/tmp/org.chromium.Chromium.luffee/SingletonSocket"
    )
    (profile / "SingletonCookie").symlink_to("10001647406132631536")
    (profile / "Local State").write_text("{}")


def _present(profile: Path) -> set[str]:
    return {name for name in _SINGLETONS if (profile / name).is_symlink()}


@pytest.fixture
def profile(isolate_profile_dir, tmp_path):
    """The claimed source profile, skipped where symlinks cannot be made."""
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to("host-1")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")
    probe.unlink()
    isolate_profile_dir.mkdir(parents=True, exist_ok=True)
    return isolate_profile_dir


@pytest.fixture
def lease(profile):
    held = get_profile_lease(profile)
    assert held.try_acquire()
    return held


def _foreign_host() -> str:
    return f"not-{socket.gethostname()}"


class TestWhichLocksAreStale:
    def test_a_lock_from_another_host_is_removed_while_the_lease_is_held(
        self, profile, lease, caplog
    ):
        _plant(profile, f"{_foreign_host()}-39")

        with caplog.at_level(logging.INFO, logger="linkedin_mcp_server.session_state"):
            reason = clear_stale_chromium_singleton(profile)

        assert reason is not None and "another host" in reason
        assert _present(profile) == set()
        # Nothing but the three links: the profile's own files stay.
        assert (profile / "Local State").read_text() == "{}"
        line = next(r.getMessage() for r in caplog.records if r.levelno == logging.INFO)
        assert "SingletonLock, SingletonSocket, SingletonCookie" in line
        assert "another host" in line
        assert str(profile) not in line

    def test_the_link_is_removed_not_its_target(self, profile, lease, tmp_path):
        outside = tmp_path / "outside-socket"
        outside.write_text("not ours")
        profile.mkdir(parents=True, exist_ok=True)
        (profile / "SingletonLock").symlink_to(f"{_foreign_host()}-39")
        (profile / "SingletonSocket").symlink_to(outside)

        clear_stale_chromium_singleton(profile)

        assert _present(profile) == set()
        assert outside.read_text() == "not ours"

    def test_a_dead_pid_on_this_host_is_removed(self, profile, lease, monkeypatch):
        monkeypatch.setattr(
            "linkedin_mcp_server.session_state._pid_is_alive", lambda pid: False
        )
        _plant(profile, f"{socket.gethostname()}-39")

        reason = clear_stale_chromium_singleton(profile)

        assert reason is not None and "no longer exists" in reason
        assert _present(profile) == set()

    @pytest.mark.parametrize("alive", [True, None], ids=["alive", "unknown"])
    def test_a_pid_on_this_host_that_may_be_alive_is_kept(
        self, profile, lease, monkeypatch, alive
    ):
        monkeypatch.setattr(
            "linkedin_mcp_server.session_state._pid_is_alive", lambda pid: alive
        )
        _plant(profile, f"{socket.gethostname()}-39")

        assert clear_stale_chromium_singleton(profile) is None
        assert _present(profile) == set(_SINGLETONS)

    @pytest.mark.skipif(sys.platform == "win32", reason="signal 0 is CTRL_C there")
    def test_the_real_probe_tells_a_live_pid_from_a_dead_one(self, profile, lease):
        import os

        _plant(profile, f"{socket.gethostname()}-{os.getpid()}")
        assert clear_stale_chromium_singleton(profile) is None
        assert _present(profile) == set(_SINGLETONS)

        for name in _SINGLETONS:
            (profile / name).unlink()
        _plant(profile, f"{socket.gethostname()}-999999")
        assert clear_stale_chromium_singleton(profile) is not None
        assert _present(profile) == set()

    def test_without_the_lease_even_a_foreign_lock_is_kept(self, profile):
        """Another server of ours may be on the profile; only the lease says not."""
        _plant(profile, f"{_foreign_host()}-39")
        assert not get_profile_lease(profile).held

        assert clear_stale_chromium_singleton(profile) is None
        assert _present(profile) == set(_SINGLETONS)

    def test_without_the_lease_a_dead_local_pid_is_kept(self, profile, monkeypatch):
        monkeypatch.setattr(
            "linkedin_mcp_server.session_state._pid_is_alive", lambda pid: False
        )
        _plant(profile, f"{socket.gethostname()}-39")

        assert clear_stale_chromium_singleton(profile) is None
        assert _present(profile) == set(_SINGLETONS)

    def test_a_root_nobody_claimed_is_refused_untouched(self, profile, tmp_path):
        foreign_root = tmp_path / "somebody-else" / "profile"
        _plant(foreign_root, f"{_foreign_host()}-39")
        assert get_profile_lease(foreign_root).try_acquire()

        with pytest.raises(ProfileRootRefusedError):
            clear_stale_chromium_singleton(foreign_root, foreign_root)

        assert _present(foreign_root) == set(_SINGLETONS)

    def test_a_directory_outside_the_owned_root_is_kept(self, profile, lease, tmp_path):
        stray = tmp_path / "stray" / "profile"
        _plant(stray, f"{_foreign_host()}-39")

        assert clear_stale_chromium_singleton(stray) is None
        assert _present(stray) == set(_SINGLETONS)

    def test_a_derived_runtime_profile_is_covered(self, profile, lease):
        derived = runtime_profiles_root(profile) / "linux-amd64-container" / "profile"
        _plant(derived, f"{_foreign_host()}-39")

        assert clear_stale_chromium_singleton(derived) is not None
        assert _present(derived) == set()

    def test_an_unattributable_lock_is_kept(self, profile, lease):
        _plant(profile, "noseparator")

        assert clear_stale_chromium_singleton(profile) is None
        assert _present(profile) == set(_SINGLETONS)

    def test_an_entry_that_is_not_a_link_is_not_chromiums(self, profile, lease):
        profile.mkdir(parents=True, exist_ok=True)
        (profile / "SingletonLock").symlink_to(f"{_foreign_host()}-39")
        (profile / "SingletonCookie").write_text("a file somebody put here")

        clear_stale_chromium_singleton(profile)

        assert not (profile / "SingletonLock").is_symlink()
        assert (profile / "SingletonCookie").read_text() == "a file somebody put here"


def _fake_playwright(recorder: dict, *, error: str | None = None):
    class _Page:
        url = "about:blank"

    class _Context:
        pages = [_Page()]

    class _Chromium:
        async def launch_persistent_context(self, user_data_dir, **kwargs):
            recorder["lock_at_launch"] = (
                Path(user_data_dir) / "SingletonLock"
            ).is_symlink()
            if error is not None:
                raise RuntimeError(error)
            return _Context()

    class _Playwright:
        chromium = _Chromium()

        async def stop(self):
            return None

    async def start():
        return _Playwright()

    return start


class TestTheLaunch:
    """Through ``BrowserManager.start()``, with the driver and its OS side faked.

    Containment and the drain are stood in for because a fake driver has no
    process to put in a Job or scan for, and neither is what is under test.
    """

    @pytest.fixture
    def launch(self, monkeypatch):
        monkeypatch.setattr(
            "linkedin_mcp_server.core.browser.contain_browser_launch",
            lambda driver: None,
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.core.browser.drain_browser_process_marker",
            lambda marker, **kwargs: True,
        )

        async def run(profile: Path, recorder: dict, *, error: str | None = None):
            manager = BrowserManager(user_data_dir=profile, headless=False)
            with mock.patch(
                "linkedin_mcp_server.core.browser.async_playwright"
            ) as playwright:
                playwright.return_value.start = _fake_playwright(recorder, error=error)
                await manager.start()
            return manager

        return run

    async def test_the_stale_lock_is_gone_before_chromium_starts(
        self, profile, lease, launch
    ):
        _plant(profile, f"{_foreign_host()}-39")
        recorder: dict = {}

        await launch(profile, recorder)

        assert recorder["lock_at_launch"] is False

    async def test_a_live_lock_reaches_chromium_untouched(self, profile, launch):
        """Without the lease the launch must not clear anything itself."""
        _plant(profile, f"{_foreign_host()}-39")
        recorder: dict = {}

        await launch(profile, recorder)

        assert recorder["lock_at_launch"] is True

    async def test_a_refused_lock_is_reported_as_a_locked_profile(
        self, profile, launch
    ):
        """Not "Network error. Check your connection", which was the incident."""
        with pytest.raises(ProfileLockedError) as caught:
            await launch(profile, {}, error=_INCIDENT_ERROR)

        assert not isinstance(caught.value, NetworkError)
        assert "SingletonLock" in str(caught.value)
        assert caught.value.__cause__ is not None

    async def test_any_other_launch_failure_is_still_a_network_error(
        self, profile, launch
    ):
        with pytest.raises(NetworkError):
            await launch(
                profile, {}, error="Executable doesn't exist at /ms-playwright/chrome"
            )


def test_the_tool_error_names_the_lock_without_diagnostics():
    error = ProfileLockedError("/home/pwuser/.linkedin-mcp/profile")

    with pytest.raises(ToolError) as caught:
        raise_tool_error(error, "get_person_profile")

    surfaced = str(caught.value)
    assert surfaced == str(error)
    assert "Network error" not in surfaced
    assert "Diagnostics:" not in surfaced
