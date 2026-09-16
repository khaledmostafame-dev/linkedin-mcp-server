"""Raw page content reads shared by every scraping workflow."""

from __future__ import annotations

import logging
import re
from typing import Any

from linkedin_mcp_server.scraping.session import ScrapingSession
from linkedin_mcp_server.scraping.text import strip_linkedin_noise


logger = logging.getLogger(__name__)

# Shared JS function returning the largest image variant an <img> offers.
#
# `currentSrc` alone is not enough: it is whatever the layout engine resolved
# for the current viewport and device pixel ratio, so the same profile yields
# a 200px photo in one window and 800px in another, while the element's
# srcset lists every size LinkedIn will serve. The CDN URLs are signed, so a
# larger variant cannot be constructed after the fact — it has to be read
# here.
#
# Kept as its own module constant so a DOM test can evaluate this exact
# source against a real browser instead of a copy of it.
LARGEST_IMAGE_VARIANT_FN_JS = r"""
function largestImageVariant(img) {
  const candidates = [];
  for (const attr of ['src', 'data-delayed-url', 'data-ghost-url', 'data-src']) {
    const value = (img.getAttribute(attr) || '').trim();
    if (value) candidates.push(value);
  }
  if (img.currentSrc) candidates.push(img.currentSrc.trim());
  // srcset is "<url> <descriptor>, <url> <descriptor>, ..."
  for (const part of (img.getAttribute('srcset') || '').split(',')) {
    const url = part.trim().split(/\s+/)[0];
    if (url) candidates.push(url);
  }

  let best = '';
  let bestSize = -1;
  for (const url of candidates) {
    const match = url.match(/_(\d+)_(\d+)\//);
    const size = match ? Math.max(+match[1], +match[2]) : 0;
    if (size > bestSize) {
      bestSize = size;
      best = url;
    }
  }
  return best;
}
"""


class PageContentReader:
    """Read innerText and raw anchor metadata off the bound page."""

    def __init__(self, session: ScrapingSession):
        self._session = session

    async def get_page_text(self) -> str:
        """Extract innerText from the main content area of the current page."""
        text = await self._session.page.evaluate(
            "() => (document.querySelector('main') || document.body).innerText || ''"
        )
        return strip_linkedin_noise(text) if isinstance(text, str) else ""

    async def click_button_by_text(
        self, text: str, *, scope: str = "main", timeout: int = 5000
    ) -> bool:
        """Click the first button or link whose visible text exactly matches."""
        matches = (
            self._session.page.locator(scope)
            .locator("button, a, [role='button']")
            .filter(has_text=re.compile(rf"^{re.escape(text)}$"))
        )
        count = await matches.count()
        logger.debug("click_button_by_text(%r): %d matches in %s", text, count, scope)
        if count == 0:
            return False
        target = matches.first
        try:
            await target.scroll_into_view_if_needed(timeout=timeout)
        except Exception:
            logger.debug("Scroll failed for button '%s'", text, exc_info=True)
        try:
            await target.click(timeout=timeout)
            return True
        except Exception:
            logger.debug("Click failed for button '%s'", text, exc_info=True)
            return False

    async def _extract_root_content(
        self,
        selectors: list[str],
    ) -> dict[str, Any]:
        """Extract innerText and raw anchor metadata from the first matching root."""
        result = await self._session.page.evaluate(
            "({ selectors }) => {\n"
            + LARGEST_IMAGE_VARIANT_FN_JS
            + """
                const normalize = value => (value || '').replace(/\\s+/g, ' ').trim();
                const containerSelector = 'section, article, li, div';
                const headingSelector = 'h1, h2, h3';
                const directHeadingSelector = ':scope > h1, :scope > h2, :scope > h3';
                const MAX_HEADING_CONTAINERS = 300;
                const MAX_REFERENCE_ANCHORS = 500;

                const getHeadingText = element => {
                    if (!element) return '';

                    const heading =
                        element.matches && element.matches(headingSelector)
                            ? element
                            : element.querySelector
                              ? element.querySelector(directHeadingSelector)
                              : null;

                    return normalize(heading?.innerText || heading?.textContent);
                };

                const getPreviousHeading = node => {
                    let sibling = node?.previousElementSibling || null;
                    for (let index = 0; sibling && index < 3; index += 1) {
                        const heading = getHeadingText(sibling);
                        if (heading) {
                            return heading;
                        }
                        sibling = sibling.previousElementSibling;
                    }
                    return '';
                };

                const root = selectors
                    .map(selector => document.querySelector(selector))
                    .find(Boolean);
                const source = root ? 'root' : 'body';
                const container = root || document.body;
                const text = container ? (container.innerText || '').trim() : '';
                const headingMap = new WeakMap();

                const candidateContainers = [
                    container,
                    ...Array.from(container.querySelectorAll(containerSelector)).slice(
                        0,
                        MAX_HEADING_CONTAINERS,
                    ),
                ];
                candidateContainers.forEach(node => {
                    const ownHeading = getHeadingText(node);
                    const previousHeading = getPreviousHeading(node);
                    const heading = ownHeading || previousHeading;
                    if (heading) {
                        headingMap.set(node, heading);
                    }
                });

                const findHeading = element => {
                    let current = element.closest(containerSelector) || container;
                    for (let depth = 0; current && depth < 4; depth += 1) {
                        const heading = headingMap.get(current);
                        if (heading) {
                            return heading;
                        }
                        if (current === container) {
                            break;
                        }
                        current = current.parentElement?.closest(containerSelector) || null;
                    }
                    return '';
                };

                const references = Array.from(container.querySelectorAll('a[href]'))
                    .slice(0, MAX_REFERENCE_ANCHORS)
                    .map(anchor => {
                        const rawHref = (anchor.getAttribute('href') || '').trim();
                        if (!rawHref || rawHref === '#') {
                            return null;
                        }

                        const href = rawHref.startsWith('#')
                            ? rawHref
                            : (anchor.href || rawHref);

                        return {
                            href,
                            text: normalize(anchor.innerText || anchor.textContent),
                            aria_label: normalize(anchor.getAttribute('aria-label')),
                            title: normalize(anchor.getAttribute('title')),
                            heading: findHeading(anchor),
                            in_article: Boolean(anchor.closest('article')),
                            in_nav: Boolean(anchor.closest('nav')),
                            in_footer: Boolean(anchor.closest('footer')),
                        };
                    })
                    .filter(Boolean);

                // Every <img>, not img[src]: LinkedIn defers loading, so an
                // image below the fold has no src attribute yet at
                // extraction time and the narrower selector silently
                // matches nothing.
                const images = Array.from(container.querySelectorAll('img'))
                    .slice(0, MAX_REFERENCE_ANCHORS)
                    .map(img => ({
                        src: largestImageVariant(img),
                        alt: normalize(img.getAttribute('alt')),
                    }))
                    .filter(image => image.src);

                return { source, text, references, images };
            }""",
            {"selectors": selectors},
        )
        return result
