# Browser / Computer-Use Agent Mission Profile — Draft 0.1

**Status:** Draft starting point pending validation by deployers shipping browser and computer-use agents in production. The verb overrides, exception conditions, and escalation chain below reflect failure modes we have seen or expect in current-generation tooling.

**Companion file:** `browser_agent.json` — the deployable Mission Profile template.

---

## What this profile is for

A "browser / computer-use agent" is an autonomous system that operates a browser or a full desktop on behalf of a human user. Examples: Claude's computer-use API, OpenAI's Operator, browser-controlling agents from Adept / Cohere / xAI, the various open-source clones (Browser Use, OpenInterpreter, Open Interpreter Code-Interpreter, autogen browser tools).

The profile assumes the agent has tool access for: navigating URLs, reading rendered page content (DOM + screenshots), clicking elements, typing into form fields, downloading and uploading files, executing browser shortcuts. Some computer-use agents also have OS-level access (Finder, Terminal, native applications). This profile speaks primarily to the browser surface; an extended deployment with OS access should layer additional restrictions.

The deployment context the profile assumes: the user is logged into their own browser session with their own cookies and credentials. The agent acts in the user's name. Consent for the agent's session is implicit (the user started it), but consent for each specific consequential action is not.

## The failure modes this profile targets

**Silent purchases.** The agent is asked to "find me a nice gift under $50" and ends up clicking through checkout — sometimes successfully, sometimes with the wrong product, sometimes recurring-subscription. The user wakes up to a credit card charge they did not approve. This is the single highest-frequency failure mode in current computer-use deployments.

**Account modification without notice.** The agent navigates to a settings page (Gmail filters, Google Workspace admin, AWS IAM, GitHub repo permissions, bank account preferences) and changes a value in pursuit of the user's stated goal — say "fix my email" leads to deleting incoming-filter rules. Reversal is hard or impossible.

**Credential and 2FA collection.** A page renders a login form. The agent fills it. The credentials come from the user's password manager autofill, the URL is genuine, the form is fine — but the action of completing authentication in contexts the user did not specifically authorize (e.g. third-party "Sign in with Google" prompts that grant scope to apps the user does not know) is consequential and gets normalized as "routine."

**Manipulation of vulnerable users.** A browser agent assisting someone in a vulnerable state — financial distress, medical crisis, recent bereavement — can apply persuasion pressure that, while accurate by the letter of the spec, exploits the situation. `COMMUNICATE.PERSUADE.CONVINCE` against `vulnerability.level = HIGH` warrants extra scrutiny.

**Cross-site exfiltration.** The agent opens a sensitive page (banking dashboard, health portal, internal company tool) and then in a separate tab POSTs to an unallowlisted endpoint. The browser DOM exposes both — a malicious or compromised agent can stitch them.

## Verb-by-verb design

### Reads — `OBSERVE.SENSE.QUERY`, `OBSERVE.MONITOR.WATCH`

Default `ALLOW`. Reading pages, screenshots, and DOM is the agent's primary capability.

Two patterns escalate. Reads from browser-internal pages (`chrome://settings`, `chrome://extensions`, `chrome://passwords`, `edge://settings`) carry password-manager content, installed extension lists, and saved cards. Reads from authentication-flow pages (`/login`, `/signin`, `/account`, `/password`, `/2fa`, `/otp`, `/mfa`) shouldn't happen on the agent's own initiative; if they do, escalate so a human verifies the agent should be authenticating right now.

Monitoring financial sites (banking, brokerage, exchange, wallet) escalates unless the user explicitly authorized financial monitoring — accidentally polling a brokerage page in a loop is the kind of thing that gets a deployment shut down.

### Writes via form-fill and click — `TRANSFORM.ALTER.MODIFY`, `CREATE.GENERATE.PRODUCE`

Default `ALLOW`. Form fills and click actions are how the agent does its job.

Hard `BLOCK` on writing to `chrome://settings` and `chrome://extensions` — these change the user's browser configuration in ways the user almost certainly did not request as part of "book me a flight."

`ESCALATE` for paths ending in `/admin`, `/settings`, `/account/security`. Site settings, security panels, and admin pages are where consequential state changes live. The agent might be asked to do legitimate work in these, but the action should be confirmed.

### Money — `EXCHANGE.TRANSFER.PAY`, `EXCHANGE.TRADE.BARTER`

Default `ESCALATE` (not `BLOCK`) for `EXCHANGE.TRANSFER.PAY`. Browser agents *do* legitimately make purchases; blocking would defeat the use case. The exception structure is:

- Amounts under $50 to merchants on a user-approved allowlist `ALLOW` with justification. Routine.
- Amounts at or above $500 `ESCALATE` to a `BILLING_AUTHORITY` role regardless of merchant. Heads-up: even with allowlisted merchants, larger amounts surface.
- Recurring subscriptions ALWAYS `ESCALATE`. The most common silent-purchase failure mode is the agent enrolling the user in a subscription rather than a one-time charge.

`EXCHANGE.TRADE.BARTER` defaults `BLOCK`. Browser agents shouldn't be executing stock or crypto trades; brokerages and exchanges have purpose-built agent interfaces that should be used instead.

### Persuasion — `COMMUNICATE.PERSUADE.CONVINCE`

Default `ALLOW`. The agent advising the user is normal.

Exception: when `context.vulnerability.level` is `HIGH` or `CRITICAL`, escalate. This catches the deployment-context where a user in distress is being talked into something. The condition relies on the deployment correctly populating vulnerability context — the spec mandates context be system-attested rather than agent-claimed.

### Authority and authentication — `BOND.ATTACH.COMMIT`, `GOVERN.AUTHORITY.OBEY`

`BOND.ATTACH.COMMIT` defaults `ESCALATE` rather than `ALLOW`. Browser agents create commitments by signing up for accounts, accepting Terms of Service, subscribing to services, joining mailing lists. Each of these binds the user. The default position is "ask before binding," with the exception that pre-authorized commitment contexts (the user said "sign me up for the newsletter") `ALLOW`.

`GOVERN.AUTHORITY.OBEY` carries the same "I was just following orders" safety override as the code-agent profile. Even if the user said to, the agent does not produce `HARM.*` or `EXCHANGE.CORRUPTION.*` effects.

### Flagged verbs

Same shape as code-agent: `HARM.DECEIVE.LIE`, `HARM.COERCE.THREATEN`, `HARM.DAMAGE.STRIKE`, `COMMUNICATE.OBFUSCATE.CONFUSE`, `EXCHANGE.CORRUPTION.BRIBE` all default `BLOCK`. Browser agents have no legitimate need for any of these.

## Escalation chain

Three tiers, deliberately compressed compared to code-agent because browser sessions are usually short-lived and the user is present:

1. `DEPLOYMENT_OPERATOR` — the human running the session. 2-minute timeout. Most escalations land here.
2. `BILLING_AUTHORITY` — for purchases over the auto-approve threshold. 15-minute timeout.
3. `SECURITY_TEAM` — for account-modification or credential-flow escalations the operator can't resolve. 1-hour timeout.

Real deployments will substitute concrete identities (the operator is often "the user themselves," paged via push notification or modal).

## Fail-safe and retention

`fail_safe_mode: FAIL_CLOSED`. Browser agents whose policy decision gets stuck should not "guess" — the cost of completing the wrong purchase is high.

`audit_retention: P1Y` — a defensible baseline for consumer-facing applications under most jurisdictions. Deployers under PCI DSS (for the payment events) extend to retention windows their QSA requires.

## What this profile does not cover

Computer-use agents with OS-level capabilities (terminal commands, native applications, file-system access outside the browser) need additional verb overrides covering shell execution and file I/O. See `code_agent.json` for a starting point on those patterns; combine.

The profile does not list specific allowlisted merchants or external endpoints. Deployers maintain those lists and reference them in the `effects[].target in user_approved_merchants` and `not_matches_allowlist external_endpoints` conditions.

The profile assumes one user per session. Multi-user contexts (a browser agent working on behalf of a team) need an additional layer of authority-chain validation not encoded here.

## Open questions for reviewers

1. Is the $50 auto-approve threshold right? Should it be currency-relative, jurisdiction-relative, or per-user-configurable rather than a single number?
2. Should `EXCHANGE.TRADE.BARTER` `BLOCK` be conditional? Some users *do* want browser-driven trading agents — what's the right escalation path for that case?
3. Are there browser-internal pages we missed in the `chrome://` / `edge://` block list?
4. The `BOND.ATTACH.COMMIT` default of `ESCALATE` is aggressive. Should it be `ALLOW` with exceptions for explicit-consent-needed cases (Terms of Service acceptance, billing enrollment) instead?
5. For computer-use agents (not just browser), what additional verbs need overrides — `RESOURCE.DAMAGE`? `INFO.WITHHOLD`?
