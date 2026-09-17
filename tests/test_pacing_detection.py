"""Where pacing learns LinkedIn pushed back, and how pacing is configured."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linkedin_mcp_server import pacing_signals
from linkedin_mcp_server.config.loaders import load_config
from linkedin_mcp_server.config.schema import AppConfig, ConfigurationError
from linkedin_mcp_server.core.auth import detect_auth_barrier_quick
from linkedin_mcp_server.core.exceptions import RateLimitError
from linkedin_mcp_server.core.utils import detect_rate_limit
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession


class TestChallengeRoutes:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.linkedin.com/checkpoint/challenge/AgF1?x=token",
            "https://www.linkedin.com/checkpoint/lg/login-submit",
            "https://www.linkedin.com/authwall?trk=bf",
            "https://www.linkedin.com/uas/consumer-email-challenge",
        ],
    )
    def test_challenge_routes_are_recognised(self, url):
        assert pacing_signals.is_challenge_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            # An expired session lands here too; that alone is no reason to stop.
            "https://www.linkedin.com/login",
            "https://www.linkedin.com/in/checkpoint-consulting/",
            "https://www.linkedin.com/feed/",
        ],
    )
    def test_other_routes_are_not(self, url):
        assert not pacing_signals.is_challenge_url(url)


class TestDetectorsReport:
    async def test_the_auth_barrier_check_reports_a_checkpoint(self, mock_page):
        mock_page.url = "https://www.linkedin.com/checkpoint/challenge/abc"

        with pacing_signals.collecting() as signals:
            barrier = await detect_auth_barrier_quick(mock_page)

        assert barrier is not None
        assert signals == {pacing_signals.SECURITY_CHALLENGE}

    async def test_a_login_redirect_is_a_barrier_but_not_a_signal(self, mock_page):
        mock_page.url = "https://www.linkedin.com/login"

        with pacing_signals.collecting() as signals:
            barrier = await detect_auth_barrier_quick(mock_page)

        assert barrier is not None
        assert signals == set()

    async def test_the_rate_limit_check_reports_an_authwall(self, mock_page):
        mock_page.url = "https://www.linkedin.com/authwall?trk=x"

        with pacing_signals.collecting() as signals:
            with pytest.raises(RateLimitError):
                await detect_rate_limit(mock_page)

        assert signals == {pacing_signals.SECURITY_CHALLENGE}


class TestNavigationSees429:
    @staticmethod
    def _quiet_barriers():
        return patch(
            "linkedin_mcp_server.scraping.navigation.detect_auth_barrier_quick",
            new_callable=AsyncMock,
            return_value=None,
        )

    async def test_a_committed_429_raises_and_reports(self, mock_page):
        response = MagicMock()
        response.status = 429
        response.headers = {"retry-after": "120"}
        mock_page.goto = AsyncMock(return_value=response)
        navigator = PageNavigator(ScrapingSession(mock_page))

        with self._quiet_barriers(), pacing_signals.collecting() as signals:
            with pytest.raises(RateLimitError) as excinfo:
                await navigator._goto_with_auth_checks(
                    "https://www.linkedin.com/in/someone/"
                )

        assert excinfo.value.suggested_wait_time == 120
        assert signals == {pacing_signals.HTTP_429}

    async def test_a_healthy_response_reports_nothing(self, mock_page):
        response = MagicMock()
        response.status = 200
        response.headers = {}
        mock_page.goto = AsyncMock(return_value=response)
        navigator = PageNavigator(ScrapingSession(mock_page))

        with self._quiet_barriers(), pacing_signals.collecting() as signals:
            await navigator._goto_with_auth_checks("https://www.linkedin.com/feed/")

        assert signals == set()

    async def test_a_refused_429_is_read_off_the_error_page(self, mock_page):
        mock_page.goto = AsyncMock(
            side_effect=Exception("net::ERR_HTTP_RESPONSE_CODE_FAILURE at url")
        )
        mock_page.evaluate = AsyncMock(return_value="This page isn't working\n429")
        navigator = PageNavigator(ScrapingSession(mock_page))

        with self._quiet_barriers(), pacing_signals.collecting() as signals:
            with pytest.raises(RateLimitError):
                await navigator._goto_with_auth_checks(
                    "https://www.linkedin.com/in/someone/"
                )

        assert signals == {pacing_signals.HTTP_429}

    async def test_a_refused_404_stays_a_navigation_error(self, mock_page):
        mock_page.goto = AsyncMock(
            side_effect=Exception("net::ERR_HTTP_RESPONSE_CODE_FAILURE at url")
        )
        mock_page.evaluate = AsyncMock(return_value="This page isn't working\n404")
        navigator = PageNavigator(ScrapingSession(mock_page))

        with (
            self._quiet_barriers(),
            patch(
                "linkedin_mcp_server.scraping.navigation.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "linkedin_mcp_server.scraping.navigation.detect_auth_barrier",
                new_callable=AsyncMock,
                return_value=None,
            ),
            pacing_signals.collecting() as signals,
        ):
            with pytest.raises(Exception) as excinfo:
                await navigator._goto_with_auth_checks(
                    "https://www.linkedin.com/in/typo/"
                )

        assert not isinstance(excinfo.value, RateLimitError)
        assert signals == set()


class TestPacingConfig:
    def test_pacing_is_on_by_default(self):
        config = load_config()

        assert config.pacing.enabled is True
        assert config.pacing.min_interval_seconds > 0
        assert config.pacing.max_writes_per_day > 0
        assert config.pacing.cooldown_base_seconds > 0

    def test_the_defaults_are_the_account_owners_choices(self):
        # Chosen by the account owner on 2026-09-17 (config/schema.py). A change
        # to any of them is a change to the risk the account runs.
        pacing = load_config().pacing

        assert {
            "min_interval_seconds": pacing.min_interval_seconds,
            "jitter_seconds": pacing.jitter_seconds,
            "max_calls_per_minute": pacing.max_calls_per_minute,
            "max_reads_per_hour": pacing.max_reads_per_hour,
            "write_min_interval_seconds": pacing.write_min_interval_seconds,
            "write_jitter_seconds": pacing.write_jitter_seconds,
            "max_writes_per_hour": pacing.max_writes_per_hour,
            "max_writes_per_day": pacing.max_writes_per_day,
            "private_write_min_interval_seconds": (
                pacing.private_write_min_interval_seconds
            ),
            "private_write_jitter_seconds": pacing.private_write_jitter_seconds,
            "max_private_writes_per_hour": pacing.max_private_writes_per_hour,
            "max_private_writes_per_day": pacing.max_private_writes_per_day,
            "cooldown_base_seconds": pacing.cooldown_base_seconds,
        } == {
            "min_interval_seconds": 8,
            "jitter_seconds": 7,
            "max_calls_per_minute": 20,
            "max_reads_per_hour": 60,
            "write_min_interval_seconds": 35,
            "write_jitter_seconds": 25,
            "max_writes_per_hour": 40,
            "max_writes_per_day": 50,
            "private_write_min_interval_seconds": 10,
            "private_write_jitter_seconds": 10,
            "max_private_writes_per_hour": 60,
            "max_private_writes_per_day": 300,
            "cooldown_base_seconds": 1800,
        }

    def test_environment_overrides(self, monkeypatch):
        monkeypatch.setenv("PACING_ENABLED", "false")
        monkeypatch.setenv("PACING_MIN_INTERVAL_SECONDS", "2.5")
        monkeypatch.setenv("PACING_JITTER_SECONDS", "0")
        monkeypatch.setenv("PACING_WRITE_MIN_INTERVAL_SECONDS", "120")
        monkeypatch.setenv("PACING_MAX_READS_PER_HOUR", "10")
        monkeypatch.setenv("PACING_MAX_WRITES_PER_HOUR", "3")
        monkeypatch.setenv("PACING_MAX_WRITES_PER_DAY", "0")
        monkeypatch.setenv("PACING_COOLDOWN_BASE_SECONDS", "600")
        monkeypatch.setenv("PACING_WRITE_JITTER_SECONDS", "4")
        monkeypatch.setenv("PACING_PRIVATE_WRITE_MIN_INTERVAL_SECONDS", "3")
        monkeypatch.setenv("PACING_PRIVATE_WRITE_JITTER_SECONDS", "2.5")
        monkeypatch.setenv("PACING_MAX_PRIVATE_WRITES_PER_HOUR", "11")
        monkeypatch.setenv("PACING_MAX_PRIVATE_WRITES_PER_DAY", "0")
        monkeypatch.setenv("PACING_MAX_CALLS_PER_MINUTE", "7")

        pacing = load_config().pacing

        assert pacing.enabled is False
        assert pacing.min_interval_seconds == 2.5
        assert pacing.jitter_seconds == 0
        assert pacing.write_min_interval_seconds == 120
        assert pacing.max_reads_per_hour == 10
        assert pacing.max_writes_per_hour == 3
        assert pacing.max_writes_per_day == 0
        assert pacing.cooldown_base_seconds == 600
        assert pacing.write_jitter_seconds == 4
        assert pacing.private_write_min_interval_seconds == 3
        assert pacing.private_write_jitter_seconds == 2.5
        assert pacing.max_private_writes_per_hour == 11
        assert pacing.max_private_writes_per_day == 0
        assert pacing.max_calls_per_minute == 7

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("PACING_ENABLED", "maybe"),
            ("PACING_MIN_INTERVAL_SECONDS", "-1"),
            ("PACING_MIN_INTERVAL_SECONDS", "soon"),
            ("PACING_MIN_INTERVAL_SECONDS", "inf"),
            ("PACING_MAX_WRITES_PER_DAY", "2.5"),
            ("PACING_MAX_READS_PER_HOUR", "-3"),
            ("PACING_MAX_CALLS_PER_MINUTE", "1.5"),
            ("PACING_MAX_PRIVATE_WRITES_PER_HOUR", "-1"),
            ("PACING_PRIVATE_WRITE_JITTER_SECONDS", "nan"),
            ("PACING_WRITE_JITTER_SECONDS", "-2"),
        ],
    )
    def test_unusable_values_are_refused(self, monkeypatch, key, value):
        monkeypatch.setenv(key, value)

        with pytest.raises(ConfigurationError, match=key):
            load_config()

    def test_cli_flags_win_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("PACING_MAX_WRITES_PER_DAY", "5")

        monkeypatch.setenv("PACING_MAX_CALLS_PER_MINUTE", "5")

        config = load_config(
            [
                "--no-pacing",
                "--pacing-max-writes-per-day",
                "9",
                "--pacing-jitter",
                "1",
                "--pacing-max-calls-per-minute",
                "12",
                "--pacing-write-jitter",
                "3",
                "--pacing-private-write-min-interval",
                "4",
                "--pacing-private-write-jitter",
                "6",
                "--pacing-max-private-writes-per-hour",
                "13",
                "--pacing-max-private-writes-per-day",
                "14",
            ]
        )

        assert config.pacing.enabled is False
        assert config.pacing.max_writes_per_day == 9
        assert config.pacing.jitter_seconds == 1
        assert config.pacing.max_calls_per_minute == 12
        assert config.pacing.write_jitter_seconds == 3
        assert config.pacing.private_write_min_interval_seconds == 4
        assert config.pacing.private_write_jitter_seconds == 6
        assert config.pacing.max_private_writes_per_hour == 13
        assert config.pacing.max_private_writes_per_day == 14

    def test_validation_refuses_a_negative_value_set_directly(self):
        config = AppConfig()
        config.pacing.max_writes_per_hour = -1

        with pytest.raises(ConfigurationError, match="max_writes_per_hour"):
            config.validate()

    def test_a_daemon_owner_receives_the_pacing_it_was_configured_with(self):
        import json

        from linkedin_mcp_server import daemon_config

        config = AppConfig()
        config.pacing.max_writes_per_day = 3
        config.pacing.enabled = False

        rebuilt = daemon_config.decode(daemon_config.encode(config))

        assert rebuilt.pacing.max_writes_per_day == 3
        assert rebuilt.pacing.enabled is False

        # A frontend that predates pacing sends no section; the owner keeps the
        # defaults, which are on.
        legacy = json.loads(daemon_config.encode(config))
        del legacy["pacing"]
        assert daemon_config.decode(json.dumps(legacy)).pacing.enabled is True
