---
name: linkedin-mcp
description: Use LinkedIn data for profile, company, job, post, feed, inbox, recruiting, and outreach work through the bundled LinkedIn MCP server. Use only for an explicit LinkedIn request.
---

# LinkedIn MCP

Use the bundled `linkedin` MCP server only when the user explicitly asks for
LinkedIn data or an action on LinkedIn. Do not call LinkedIn tools for unrelated
work, health checks, or background maintenance.

## Operating rules

- Start with the smallest read operation that can answer the request.
- Treat profile, company, job, post, feed, and message results as live LinkedIn
  evidence. Distinguish retrieved facts from inference.
- Keep searches and result pages modest. Do not bulk scrape or spam.
- Never enable the plugin or its MCP server, edit Codex configuration, or start
  a login flow merely because LinkedIn might be useful. If either component is
  disabled, explain that state and stop.
- `send_message`, `reply_to_conversation`, `mark_conversation_read`,
  `archive_conversation`, `connect_with_person`, the comment writes,
  `create_post`, `create_poll`, `edit_scheduled_post`, `delete_scheduled_post`,
  `edit_post`, `delete_post`, `follow`, `withdraw_invitation`,
  `respond_to_invitation`, `save_post` and `save_job` are write actions. Each
  requires `confirm` (`confirm_send` for `send_message`). Use them only when the user
  explicitly authorizes the exact recipient and action (for a post, the exact
  post). Call `create_post` with `confirm=false` first and show the preview. Confirm the
  final message or connection note unless the user has already supplied it.
- Do not retry a failed write action unless the result proves it was not sent.

## Browser and authentication

The server uses a managed Chromium browser and the user's own LinkedIn session.
Installing or enabling the plugin does not sign in. The first LinkedIn data
request may prepare the browser, import an existing local session, or require a
visible login window. If LinkedIn presents a captcha, two-factor prompt, or
other user-owned authentication step, stop and ask the user to complete it.

If browser setup or authentication is still in progress, report that exact
state and retry once after it finishes. Do not loop, launch extra browser
instances, clear profiles, or replace the user's browser session.

## Tool selection

- Profiles: `get_my_profile`, `get_person_profile`, `search_people`,
  `resolve_geo_location`, and `get_sidebar_profiles`.
- Sales Navigator (read-only, needs a seat): `sales_nav_search_leads`,
  `sales_nav_search_accounts`, `sales_nav_get_lists`, and `sales_nav_get_list`.
- Pacing: `get_pacing_status` reports counters and cooldowns without contacting
  LinkedIn.
- Companies: `get_company_profile`, `get_company_posts`, `search_companies`,
  and `get_company_employees`.
- Jobs: `search_jobs`, `get_saved_jobs`, `get_job_details`, and
  `get_job_alerts`; `save_job` (write).
- Content: `get_feed`, `get_hashtag_feed`, `get_notifications`, `search_posts`,
  `get_saved_posts`, `get_post_reactions`, and `get_post_comments`; `save_post`
  (write).
- Analytics (the signed-in member's own only): `get_post_analytics` and
  `get_profile_analytics`. `get_company_page_analytics` is EXPERIMENTAL, not
  fully tested and known not working (live check 2026-09-17: admin analytics
  routes returned not_authorized); it is only registered when the server runs
  with `ENABLE_COMPANY_PAGE_TOOLS=true`.
- Messages: `get_inbox`, `get_conversation`, and `search_conversations`;
  `reply_to_conversation`, `mark_conversation_read`, and `archive_conversation`
  (writes).
- Posting: `get_scheduled_posts` (read); `create_post`, `create_poll`,
  `edit_scheduled_post`, `delete_scheduled_post`, `edit_post` and
  `delete_post` (writes). Posting as a company page (`post_as`) is
  EXPERIMENTAL and refused unless `ENABLE_COMPANY_PAGE_TOOLS=true`; do not
  pass it otherwise.
- Network: `list_connections`, `get_mutual_connections`, `get_invitations`,
  `search_groups`, `get_group_posts`, `get_group_members`, `search_events`,
  `get_event_details`, and `get_event_attendees` (reads);
  `follow`, `withdraw_invitation`, and `respond_to_invitation` (writes).
- Writes: `send_message`, `connect_with_person`, `comment_on_post`,
  `reply_to_comment`, and `react_to_comment`, subject to the explicit
  authorization rules above. Call the comment writes with `confirm=false` first
  to preview.
- Cleanup: use `close_session` only when the user asks to end the managed
  browser session or when the current LinkedIn task is finished and no follow-up
  call is expected.
