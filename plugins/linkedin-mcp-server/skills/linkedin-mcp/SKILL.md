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
- `send_message`, `connect_with_person`, the comment writes, `create_post`,
  `edit_scheduled_post`, `delete_scheduled_post`, `edit_post`, `delete_post`,
  `follow`, `withdraw_invitation` and `respond_to_invitation` are write
  actions. Each requires `confirm` (`confirm_send` for `send_message`). Use them only when the user
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

- Profiles: `get_my_profile`, `get_person_profile`, `search_people`, and
  `get_sidebar_profiles`.
- Companies: `get_company_profile`, `get_company_posts`, `search_companies`,
  and `get_company_employees`.
- Jobs: `search_jobs`, `get_saved_jobs`, and `get_job_details`.
- Content: `get_feed`, `search_posts`, and `get_post_comments`.
- Analytics (the signed-in member's own only): `get_post_analytics`,
  `get_profile_analytics`, and `get_company_page_analytics` (pages they
  administer).
- Messages: `get_inbox`, `get_conversation`, and `search_conversations`.
- Posting: `get_scheduled_posts` (read); `create_post`,
  `edit_scheduled_post`, `delete_scheduled_post`, `edit_post` and
  `delete_post` (writes).
- Network: `list_connections`, `get_mutual_connections`, `get_invitations`,
  `search_groups`, `get_group_posts`, and `get_group_members` (reads);
  `follow`, `withdraw_invitation`, and `respond_to_invitation` (writes).
- Writes: `send_message`, `connect_with_person`, `comment_on_post`,
  `reply_to_comment`, and `react_to_comment`, subject to the explicit
  authorization rules above. Call the comment writes with `confirm=false` first
  to preview.
- Cleanup: use `close_session` only when the user asks to end the managed
  browser session or when the current LinkedIn task is finished and no follow-up
  call is expected.
