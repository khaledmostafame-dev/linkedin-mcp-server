"""Browser-DOM tests for the analytics reads, against synthetic pages.

The fixture is a claim about the algorithm only: stat cards are a number beside
a label, in English, Arabic (with Arabic-Indic digits and RTL marks) and opaque
tokens. Pairing must not depend on which. Redirects are simulated by serving a
page whose script moves ``location`` off the analytics route.
"""

from __future__ import annotations

from typing import Any, cast

import asyncio

import pytest
from patchright.async_api import Page, async_playwright

from linkedin_mcp_server.scraping import analytics as analytics_module
from linkedin_mcp_server.scraping.analytics import AnalyticsScraper
from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.session import ScrapingSession

pytestmark = [
    pytest.mark.browser_dom,
    pytest.mark.xdist_group("browser_runtime"),
]

ACTIVITY = "urn:li:activity:7300000000000000000"
RLM = chr(0x200F)


def _arabic(number: str) -> str:
    return RLM + "".join(
        chr(0x660 + int(d)) if d.isdigit() else chr(0x66C) for d in number
    )


LOCALES = {
    "en": ("Impressions", "Members reached", lambda n: n),
    "ar": ("مرات الظهور", "الأعضاء الذين تم الوصول إليهم", _arabic),
    "opaque": ("k1", "k2", lambda n: n),
}


class _Session(ScrapingSession):
    async def delay(self, seconds: float) -> None:
        await asyncio.sleep(0)

    async def check_rate_limit(self) -> None:
        return None

    async def dismiss_modal(self) -> bool:
        return False

    async def scroll_body(self, pause_time: float = 1.0, max_scrolls: int = 10) -> None:
        return None


class _Navigator:
    def __init__(self, page: Any):
        self._page = page

    async def _navigate_to_page(self, url: str) -> None:
        await self._page.goto(url)
        await self._page.wait_for_load_state("load")


def _page(first: str, second: str, fmt: Any) -> str:
    filler = "Synthetic analytics filler text. " * 10
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"></head><body><main>
<h1>Synthetic dashboard</h1><p>{filler}</p>
<section><div><span>{fmt("1,234")}</span></div><div><span>{first}</span></div></section>
<section><div><span>{fmt("900")}</span><span>{second}</span></div></section>
</main></body></html>"""


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


@pytest.fixture(autouse=True)
def fast_waits(monkeypatch):
    monkeypatch.setattr(analytics_module, "_CONTENT_WAIT_MS", 500)


async def _scraper(page: Any, routes: dict[str, str]) -> AnalyticsScraper:
    async def handle(route: Any) -> None:
        path = route.request.url.split("linkedin.com", 1)[1]
        body = next(
            (html for prefix, html in routes.items() if path.startswith(prefix)),
            "<html><body><main>elsewhere</main></body></html>",
        )
        await route.fulfill(content_type="text/html; charset=utf-8", body=body)

    await page.route("https://www.linkedin.com/**", handle)
    session = _Session(cast(Page, page))
    return AnalyticsScraper(
        session, cast(Any, _Navigator(page)), PageContentReader(session)
    )


@pytest.mark.parametrize("locale", list(LOCALES))
async def test_post_analytics_pairs_counts_with_their_labels(dom_page, locale):
    first, second, fmt = LOCALES[locale]
    scraper = await _scraper(
        dom_page, {"/analytics/post-summary/": _page(first, second, fmt)}
    )
    result = await scraper.get_post_analytics(ACTIVITY)

    assert "post_analytics" in result["sections"]
    assert result["metrics"]["post_analytics"] == [
        {"label": first, "value": 1234},
        {"label": second, "value": 900},
    ]
    if locale == "en":
        assert result["named_metrics"]["post_analytics"] == {
            "impressions": 1234,
            "members_reached": 900,
        }
    else:
        assert "named_metrics" not in result


async def test_a_redirect_off_the_post_summary_is_not_authorized(dom_page):
    redirect = (
        "<html><body><main>x</main><script>"
        f"location.replace('/feed/update/{ACTIVITY}/')</script></body></html>"
    )
    scraper = await _scraper(
        dom_page,
        {
            "/analytics/post-summary/": redirect,
            "/feed/update/": _page("Impressions", "Members reached", lambda n: n),
        },
    )
    result = await scraper.get_post_analytics(ACTIVITY)

    assert result["sections"] == {}
    assert "metrics" not in result
    error = result["section_errors"]["post_analytics"]
    assert error["error_type"] == "not_authorized"
    assert error["landed_path"] == f"/feed/update/{ACTIVITY}/"


async def test_ugcpost_resolves_through_the_posts_own_analytics_link(dom_page):
    post_page = (
        "<html><body><main><a href='/analytics/post-summary/"
        f"{ACTIVITY}/'>x</a></main></body></html>"
    )
    scraper = await _scraper(
        dom_page,
        {
            "/feed/update/urn:li:ugcPost:": post_page,
            "/analytics/post-summary/": _page("Impressions", "Reactions", lambda n: n),
        },
    )
    result = await scraper.get_post_analytics("urn:li:ugcPost:7300000000000000001")
    assert result["url"].endswith(f"/analytics/post-summary/{ACTIVITY}/")
    assert result["named_metrics"]["post_analytics"]["reactions"] == 900


async def test_profile_dashboards_are_one_section_each_and_redirects_are_reported(
    dom_page,
):
    premium = (
        "<html><body><main>x</main><script>location.replace('/premium/products/')"
        "</script></body></html>"
    )
    scraper = await _scraper(
        dom_page,
        {
            "/analytics/profile-views/": _page("Profile viewers", "k", lambda n: n),
            "/analytics/search-appearances/": premium,
        },
    )
    result = await scraper.get_profile_analytics("profile_viewers,search_appearances,x")

    assert list(result["sections"]) == ["profile_viewers"]
    assert result["named_metrics"]["profile_viewers"] == {"profile_viewers": 1234}
    assert result["section_errors"]["search_appearances"]["error_type"] == "redirected"
    assert result["unknown_sections"] == ["x"]
