"""The switch for the experimental company-page tools.

EXPERIMENTAL, not fully tested and known not working: a live check on
2026-09-17 had LinkedIn answer the admin analytics routes with
``not_authorized``. The code is kept, but off by default: without
``ENABLE_COMPANY_PAGE_TOOLS`` the server registers no
``get_company_page_analytics`` and ``create_post``/``create_poll`` refuse
``post_as`` before any browser work.
"""

from fastmcp.exceptions import ToolError

from linkedin_mcp_server.config import get_config

COMPANY_PAGE_POSTING_DISABLED_MESSAGE = (
    "posting as a company page is disabled: experimental, not fully tested; "
    "set ENABLE_COMPANY_PAGE_TOOLS=true to enable at your own risk"
)


def company_page_tools_enabled(override: bool | None = None) -> bool:
    """Whether to register and accept the company-page features.

    *override* is for a caller that registers tools without the process
    configuration (tests, mostly); ``None`` reads ``ENABLE_COMPANY_PAGE_TOOLS``
    through the configuration singleton.
    """
    if override is not None:
        return override
    return get_config().server.enable_company_page_tools


def refuse_post_as_when_disabled(post_as: str | None, *, enabled: bool) -> None:
    """Raise a ToolError for a non-empty ``post_as`` while the switch is off."""
    if not enabled and post_as is not None and post_as.strip():
        raise ToolError(COMPANY_PAGE_POSTING_DISABLED_MESSAGE)
