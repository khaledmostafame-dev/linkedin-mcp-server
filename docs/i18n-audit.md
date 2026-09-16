# Non-English LinkedIn UI audit

An inventory of every place classification, control-finding, or text parsing
in this codebase could depend on LinkedIn's UI language, done for upstream
[#890](https://github.com/stickerdaniel/linkedin-mcp-server/issues/890)
("Messaging selectors decide identity from English text") and the account
this fork drives, which may render LinkedIn in Arabic (RTL, Arabic-Indic
digits). The rule this measures against is `AGENTS.md` -> Scraping Rules:
detection must be locale-independent (URL patterns, attribute presence,
structural counts); where text is genuinely the only signal, guard it behind
an explicit per-locale table and document the limitation in code.

None of this was checked against a live non-English LinkedIn session — that
would mean driving a real account into a locale it doesn't use, which is out
of scope here (see the safety rules this work operated under). Every "fixed"
item below is a mechanical improvement (structural signal instead of an
English word, or a documented-best-effort table) verified with synthetic
fixtures, not a live capture. Every Arabic string added is a best-effort
transcription. **Live verification against a real Arabic-locale LinkedIn
session is still required** before trusting any of it in production — see
the per-item notes and the summary at the bottom.

## Already compliant (found during the audit, no change needed)

- **Connection-state detection** — `scraping/connection.py`,
  `scraping/connection_actions.py`. Fully structural since upstream #504/#629
  landed: 1st-degree vs. follow-only vs. incoming-request is decided by
  `aria-label` *presence*, anchor `href` patterns
  (`/messaging/compose/`, `/preload/custom-invite/`), and button-row counts —
  never by the text of "Connect", "Follow" or "1st". No `INCOMING_REQUEST_LABELS`
  or similar text table exists in this codebase; upstream's own docstring at
  `connection.py:10-14` states the rule explicitly.
- **Message composition and sending** — `scraping/message_sender.py`. The
  selectors upstream #890 originally reported (`aria-label*="Send"`,
  `placeholder*="Type a name"`, `aria-label*="Close your draft conversation"`,
  `aria-label*="Write a message"`) do not exist in this file as it stands
  today — it was rewritten around a URN-verified composer/owner model that
  identifies the recipient from the compose URL and the DOM structure
  (`[role="textbox"][contenteditable="true"]`,
  `button[type="submit"], button[data-control-name="send"]`), never from
  English microcopy. Confirmed by grep: no `aria-label*=`/`placeholder*=`
  substring selector remains anywhere in `linkedin_mcp_server/`. #890 is still
  open upstream but appears to have been resolved here as a side effect of
  that rewrite — flagged for Khaled to close or comment on upstream if this
  fork's history confirms it.
- **Login/authwall/rate-limit detection** — `scraping/contracts.py:14-29`,
  `scraping/capture.py`. Gated on URL patterns
  (`/login`, `/authwall`, `/checkpoint`) with an explicit comment ruling out
  body-text detection: "body text would be a per-locale guess, and this
  project's rule is that classification never depends on text values."
- **Job id extraction** — `scraping/link_metadata.py` `JOB_PATH_RE`. Already
  uses `[0-9]` rather than `\d` specifically because Python's `\d` also
  matches Arabic-Indic digits while the JS extractor's `\d` does not — the
  existing comment documents this asymmetry and the code is already correct
  for it (a job id from a URL is never Arabic-Indic in practice, since the
  URL itself is ASCII; this guards against Python and JS disagreeing on what
  `\d` means, not against a real Arabic-Indic id).

## Fixed in this branch

| # | File:line | What broke under Arabic | Fix |
|---|---|---|---|
| 1 | `scraping/job_pages.py` `_get_total_search_pages` (was line 409) | Matched the English word "of" in `.jobs-search-pagination__page-state`'s "Page X of Y" text (`re.search(r"of\s+(\d+)", text)`). A translated "of" ("من" in Arabic) never matches, so total-page count silently degrades to `None` (documented fallback to `max_pages`, so not a crash — but a real loss of a usable signal). | Replaced the word match with a structural one: extract every digit-group via `re.findall(r"\d+", ...)` and, when there are exactly two, take the second as the total — the shape of "Page X of Y" (current, then total) rather than any locale's spelling of "of". Digits are normalized through the new `normalize_localized_digits` helper first, so Arabic-Indic counts parse too. Extracted into a pure, browser-free `parse_total_from_page_state_text()` for unit testing. Verified: `tests/scraping/test_job_pages.py::TestParseTotalFromPageStateText` (pure Python) + the existing `tests/test_job_pagination_dom.py::TestTotalSearchPages` (real Chromium) all pass unchanged in shape. |
| 2 | `scraping/job_pages.py` `_get_total_list_pages` (saved-jobs pager) | Read each pager button's label through JS `parseInt`, which cannot parse Arabic-Indic digits (`parseInt('١')` is `NaN`) — documented and accepted upstream as a locale loss (`tests/test_job_pagination_dom.py`'s prior `test_non_ascii_numerals_degrade_to_no_count`). | Moved the parse from JS `parseInt` to Python `int()` after `normalize_localized_digits`, which does understand Arabic-Indic and Extended Arabic-Indic digits. The DOM test was updated: `test_non_ascii_numerals_are_normalized` now asserts a **correct** count instead of asserting the old degradation, and a new `test_an_unlisted_digit_script_degrades_to_no_count` pins the still-accepted gap (a digit script outside the helper's table, e.g. Devanagari, still fails safe to "not a page number" for that one button rather than crashing or guessing). |
| 3 | `core/utils.py` `handle_modal_close` (was lines 447-450) | Matched `button[aria-label="Dismiss"]` / `button[aria-label="Close"]` literally. An Arabic modal's close control never carries that exact English string, so the popup-dismiss best-effort silently does nothing on it (falls through to the `button.artdeco-modal__dismiss` class selector, which is locale-independent but not guaranteed present on every modal LinkedIn renders). | Added a `_MODAL_DISMISS_ARIA_LABELS` per-locale table (`en`, `ar`) built into `_modal_dismiss_selector()`; the class-based selector stays and is tried alongside it, never replaced. Verified: `tests/test_core_utils.py::TestModalDismissSelector`, `TestHandleModalClose`. |
| 4 | `scraping/link_metadata.py` `_GENERIC_LABELS` (was module-level set) | A LinkedIn anchor whose only text is a generic control word ("Follow", "Connect", "Comment", ...) is meant to be dropped as noise rather than surfacing as a person/company's display name. The set was English-only, so the equivalent Arabic control word passed through unfiltered and could shadow a better candidate label (e.g. an `aria-label` carrying the real name). | Restructured into `_GENERIC_LABELS_BY_LOCALE` (`en`, `ar`) unioned into `_GENERIC_LABELS`, since this module has no locale signal of its own to pick one table over the other. Verified: `tests/test_link_metadata.py::TestLocaleGuardedLabels::test_drops_an_arabic_generic_action_label`. |
| 5 | `scraping/link_metadata.py` `_CONTEXT_LABELS` / `clean_heading` (was module-level set + straight membership check) | A profile section heading ("About", "Experience", ...) is used to tag a reference's `context` field, matched against a lowercased `h1`/`h2`/`h3`. English-only, so an Arabic heading never matched and `context` silently came back `None` (or fell to a generic "top card"/`None` fallback) instead of naming the section. | Restructured into `_CONTEXT_LABEL_TRANSLATIONS` (canonical English tag -> `{locale: localized text}`) plus a reverse `_CONTEXT_LABEL_CANONICAL` lookup, so `clean_heading` always returns the fixed English vocabulary the API already contracts on, regardless of which locale's heading text matched. Verified: `tests/test_link_metadata.py::TestLocaleGuardedLabels::test_arabic_heading_maps_to_the_canonical_english_context` and `test_an_uncovered_locale_heading_degrades_to_top_card` (regression: an unlisted locale, German here, still degrades to the existing "top card" fallback rather than crashing). |
| 6 | `scraping/text.py` (new) | No shared helper existed for Arabic-Indic/Extended Arabic-Indic digits, Arabic thousands (`٬`) / decimal (`٫`) separators, or bidi marks (LRM/RLM/ALM) that LinkedIn's Arabic rendering can embed around numbers. | Added `normalize_localized_digits()` — see its docstring for exactly which code points it rewrites, and its documented gap (other numeral scripts, e.g. Devanagari, CJK, are out of scope; extend the table if one turns up against a real session). Used at both `job_pages.py` sites above. Verified: `tests/scraping/test_text.py::TestNormalizeLocalizedDigits`. |

## Found, not fixed — needs a live non-English DOM capture

These are real gaps, but the text they'd need to match is a specific,
multi-word LinkedIn microcopy string this project has no verified transcription
for (unlike the single common control words fixed above). Guessing a full
phrase wrong is worse than leaving the documented fallback in place: a wrong
guess either fails to match (no better than today) or, worse, could
coincide with real message content in ways a short exact-match boundary is
not designed to rule out. This mirrors upstream's own conclusion on #890:
"Nobody has read LinkedIn's non-English compose DOM, so the structural
replacement cannot be designed from here yet... A capture of that page's
compose DOM is the missing input for the fix."

- **`scraping/text.py` `_MESSAGING_CHROME_STRINGS`** (`strip_conversation_chrome`).
  Boundaries for stripping inbox-sidebar and composer chrome around an opened
  conversation thread: `"Load more conversations"`,
  `"Open the options list in your conversation with"`,
  `"Maximize compose field"`, `"Open GIF Keyboard"`, `"Open Emoji Keyboard"`,
  `"Open send options"`. English-only (`"en"` is the only key). On a
  non-English session this falls through to returning the text unstripped
  (documented in the function's own docstring: "Each boundary independently
  falls back to keeping the text when its marker is absent"), so
  `get_conversation` would return the thread with inbox-sidebar and composer
  chrome still attached, not a crash.
- **`scraping/text.py` `_SIDEBAR_CHROME_STRINGS`** (used by `scraping/person.py`
  `_SIDEBAR_PROFILES_JS`). Sidebar recommendation headings
  (`"More profiles for you"`, `"Explore premium profiles"`,
  `"People you may know"`) and the `"show all"`/`"see all"` expand-link
  prefixes. English-only. On a non-English session, `get_person_profile`'s
  sidebar-recommendation collection (an optional, best-effort section) would
  simply collect nothing for that profile rather than erroring.
- **`scraping/text.py` `_DETAIL_CAPTURE_TEXT`** (`DETAIL_CAPTURE_EN_US`).
  Readiness-blocking prefixes (`"Load more"`, `"More profiles for you"`,
  `"Explore premium profiles"`) and the `"Show (more|all)"` expansion-button
  pattern used when hydrating a detail overlay. English-only; the module
  comment already states browser locale is forced to `en-US` and "unsupported
  locales are deliberately not inferred." Same caveat as everywhere else in
  this audit: forcing the *browser's* locale does not force LinkedIn's
  *account* display language, so this can still be reached on an Arabic
  account.
- **`scraping/conversations.py` `_SELECT_CONVERSATION_PREFIX_RE`**
  (`strip_select_conversation_prefix`). Matches the `"Select conversation
  with "` prefix on a sidebar row's `aria-label` to isolate the participant
  name. English-only; already documented in the module as "Best-effort ...
  falls through silently for any other locale, in which case the full
  aria-label flows into the ref's text field rather than a stripped name" —
  i.e. a cosmetic degradation (a longer, unstripped label), not a functional
  break.
- **`scraping/link_metadata.py` `_CONNECTIONS_FOLLOW_RE`**. Drops a company-page
  noise label matching `"connections follow this page"` (e.g. "500
  connections follow this page"). English-only; not extended here for the
  same reason as the four items above — it is a full sentence pattern, and
  the cost of leaving it unfixed is purely cosmetic (the sentence leaks
  through as a reference's `text` instead of being dropped).

## Note on the browser-locale assumption

Several of the tables above (and the module comments they were already
carrying, e.g. `text.py`'s `_MESSAGING_CHROME_STRINGS`) reason from
`BrowserManager` forcing the Chromium context `locale` to `en-US`
(`core/browser.py`). That setting controls `navigator.language` and the
`Accept-Language` header, which biases anonymous/first-visit rendering
toward English — it does **not** override a signed-in LinkedIn account's own
saved display-language preference (Settings -> Account preferences ->
Language), which is a server-side setting LinkedIn appears to honor
regardless of what the browser sends. An account whose LinkedIn display
language is set to Arabic will see an Arabic UI through this server exactly
as it would through any other browser. Every finding in this document should
be read with that in mind: "the browser is forced to en-US" is not a reason
a given text-matching site is safe, only a historical explanation for why it
was written English-only.

## Digit and separator coverage (`normalize_localized_digits`)

| Covered | Not covered (documented gap) |
|---|---|
| Arabic-Indic ٠-٩ (U+0660-0669) | Devanagari, CJK, and other numeral scripts |
| Extended Arabic-Indic / Persian-Urdu ۰-۹ (U+06F0-06F9) | |
| Arabic thousands separator ٬ (U+066C) -> `,` | |
| Arabic decimal separator ٫ (U+066B) -> `.` | |
| Bidi marks LRM/RLM/ALM (U+200E, U+200F, U+061C) stripped | |

No existing call site in this codebase parses a follower/connection/reaction
count or a relative date ("2w", "3mo") into a number — per `AGENTS.md` ->
Tool Return Format, every scraping tool returns raw `innerText` in
`sections: {name: raw_text}` rather than a server-side parsed count or date,
so that parsing (and any locale handling it would need) belongs to whatever
reads the tool's output, not to this server. `normalize_localized_digits` is
provided as infrastructure for the two pagination call sites above and for
any future code that does need to parse a LinkedIn-rendered number.

## Summary for Khaled (live verification needed)

Everything below needs checking with an actual Arabic-locale LinkedIn
session (Settings -> Account preferences -> Language) before being trusted:

1. The exact Arabic transcriptions added (`core/utils.py`
   `_MODAL_DISMISS_ARIA_LABELS["ar"]`, `scraping/link_metadata.py`
   `_GENERIC_LABELS_BY_LOCALE["ar"]` and `_CONTEXT_LABEL_TRANSLATIONS[...]["ar"]`)
   — these are best-effort, not copied from a live DOM capture.
2. Whether `_get_total_search_pages`'s two-numbers-in-order assumption
   ("Page X of Y" keeps current-then-total ordering under RTL rendering)
   actually holds for LinkedIn's Arabic pagination text.
3. Whether LinkedIn's Arabic UI renders counts in Arabic-Indic digits at all
   (some Arabic-locale software renders Western digits by default depending
   on an OS/browser numbering-system setting) — `normalize_localized_digits`
   is a no-op on Western digits, so this is safe either way, but worth
   knowing for the "not fixed" items above.
4. The five "found, not fixed" items: a real capture of an Arabic
   conversation-thread page, an Arabic profile's sidebar recommendations,
   and an Arabic company page's follower-count sentence would turn each of
   those into a fixable, verified per-locale table entry.
