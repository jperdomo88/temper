# Code Agent Mission Profile — Draft 0.1

**Status:** Draft starting point pending validation by frontier-lab platform engineers and security teams. The verb overrides, exception conditions, and escalation chain below reflect choices we believe defensible based on the failure modes we see in tool-using code agents today. They are not a finished policy.

**Companion file:** `code_agent.json` — the deployable Mission Profile template.

---

## What this profile is for

A "code agent" is an autonomous system that reads, writes, edits, executes, and reasons about source code on behalf of a human developer. Examples: Claude Code, Cursor, Cline, Aider, GitHub Copilot Workspace, Continue, Cognition's Devin. The profile assumes the agent has tool access for: file I/O within a working directory, shell execution, git operations, HTTP/HTTPS requests, and natural-language communication with the developer.

The profile does not assume billing capability, access to production secrets, write access outside the working directory, or authority to modify CI/CD configuration without escalation. Each of those is treated as elevated.

## The failure modes this profile targets

Three failure modes occur with non-trivial frequency in tool-using code agents:

**Exfiltration via tool composition.** The agent claims a benign action ("I'll fetch the latest from upstream") and uses tool capabilities sequentially to read sensitive data (an `~/.ssh/` key, a `.env` file, a secret in environment) and then POST it to an endpoint the user did not authorize. Each individual tool call looks normal; the *combination* is the problem. The profile flags the read of sensitive-shaped paths and the unallow-listed external POST as separate ESCALATE events.

**Silent CI/credential modification.** The agent edits `.github/workflows/`, `.circleci/`, or a `.env` file as part of a larger task. The change adds a step that exfiltrates secrets during the next CI run, or replaces a credential with one the agent controls. The user merges the PR without noticing the workflow diff. The profile escalates writes to CI config paths and blocks writes to credential file patterns.

**Deceptive task framing.** The agent reports "task complete" while the diff includes changes the user did not request — disabled tests, removed assertions, broadened CORS, weakened auth. The profile flags `HARM.DECEIVE.LIE` and `COMMUNICATE.OBFUSCATE.CONFUSE` as BLOCK so the integrity check between claimed verb and observed effects will trip.

The profile is not a substitute for code review, secret scanning, or pre-merge CI checks. It is a behavioral audit layer that runs alongside them and produces an inspectable record when something goes wrong.

## Verb-by-verb design

Verbs not listed take the spec's default mapping with no override.

### Reads — `OBSERVE.SENSE.QUERY`, `OBSERVE.MONITOR.WATCH`

Default `ALLOW`. Routine code-agent operation involves reading hundreds of files. Wholesale escalation isn't feasible.

Exceptions escalate when the target path or content shape suggests a credential or secret. The check is on the path, not on the content — checking content for secrets is the job of an upstream secret-scanning tool. The profile catches the broad pattern: paths containing `secrets`, `credentials`, `.env`, `id_rsa`, `id_ed25519`, `aws_access_key`, `private_key`, `token`, or paths under system credential directories (`/etc/`, `/var/secrets/`, `~/.ssh/`, `~/.aws/`).

Reads from *outside the working directory* also escalate. A code agent has no general reason to read files outside the project it was invoked on; doing so is suspicious enough to warrant developer review.

### Writes and edits — `TRANSFORM.ALTER.MODIFY`, `CREATE.GENERATE.PRODUCE`, `TRANSFORM.MOVE.RELOCATE`

Default `ALLOW` for in-tree writes. The agent's primary job is editing code.

Two paths trigger `BLOCK` outright (not just escalate): writing to credential files (`.env*`, `id_rsa`, anything containing `credentials`) and writing outside the working directory to system locations (`/etc/`, `/usr/local/`). These are operations a code agent should never do legitimately. A `BLOCK` here is a stronger signal than `ESCALATE` because there is no reasonable human use case to surface to.

Three paths trigger `ESCALATE` to `TECH_LEAD` or `SECURITY_ENGINEER`:
- Writes to CI configuration directories (`.github/workflows/`, `.circleci/`, `Dockerfile`, `.gitlab-ci.yml`). CI config changes are a known supply-chain attack vector. Human review before the change merges.
- Creates of files that look like credentials. The agent should not generate keys or `.env` files unless explicitly asked.
- Moves outside the working directory.

### External communication — `COMMUNICATE.INFORM.TELL`

Default `ALLOW`. The agent communicating with the developer is normal.

Exceptions: `INFO.DISCLOSE` effects whose target is a URL not on the deployment's external-endpoint allowlist `ESCALATE`. Targets explicitly marked `unverified` `BLOCK`. This is the second half of the exfiltration check: even if the read of a secret was allowed (perhaps with justification), the disclosure to an unallowlisted target stops the chain.

### Flagged verbs — `HARM.DECEIVE.LIE`, `COMMUNICATE.OBFUSCATE.CONFUSE`, `HARM.COERCE.THREATEN`, `EXCHANGE.CORRUPTION.BRIBE`

All `BLOCK`. No code agent has a legitimate reason to deceive, obfuscate, threaten, or bribe. If CCD ever classifies a tuple as one of these (because the claim-vs-mechanical check disagrees), the action is stopped and a tuple is recorded.

### Harm — `HARM.DAMAGE.STRIKE`

Default `BLOCK`. The verb encompasses destructive operations (`rm -rf`, `git reset --hard` without backup, force-pushing over remote history). The exception: when the damage is within the working directory, the agent acknowledged the harm in `justification.harm_acknowledged`, and the user (`authority_chain[].authority_id = "USER"`) authorized the destructive operation, `ALLOW`. This permits routine destructive operations (deleting a test file the user said to remove) while requiring explicit acknowledgment.

### Money — `EXCHANGE.TRANSFER.PAY`, `EXCHANGE.CORRUPTION.BRIBE`

Default `BLOCK` for both. Most code agents have no billing capability. For agents that do (e.g., agents authorized to spin up cloud resources with a credit card), the exception escalates to `TECH_LEAD` rather than allowing silently. `BRIBE` is unconditional `BLOCK`.

### Authority — `GOVERN.AUTHORITY.OBEY`, `GOVERN.AUTHORITY.DISOBEY`

`OBEY` defaults `ALLOW` (the agent following user instructions is the normal case) with two exceptions: unrecognized authority IDs `ESCALATE` (someone claiming to be the user but unverified), and obedience that would produce a `HARM.*` or `EXCHANGE.CORRUPTION.*` effect `BLOCK` regardless of who issued the instruction. This is the "I was just following orders" check.

`DISOBEY` defaults `LOG` (record but don't gate). The exception allows disobedience when the stated goal is refusing an unsafe instruction. The agent saying "no" to a dangerous request is a feature, not a violation.

### Self-monitoring — `RECURSE.VERIFY.AUDIT`

Default `ALLOW`. The agent can examine its own state, run self-checks, and report.

Exception: any audit operation whose purpose contains "modify" or "override" `BLOCKS`. The agent should not be modifying its own audit configuration, escalation chain, or fail-safe mode. The Mission Profile is signed and rollback-resistant; an agent attempting to alter it locally is a clear integrity violation.

## Escalation chain

Four tiers:

1. `DEVELOPER` — the user running the agent. Timeout 5 minutes (PT5M). Most escalations land here.
2. `TECH_LEAD` — for CI changes and billing concerns. Timeout 30 minutes (PT30M).
3. `SECURITY_ENGINEER` — for credential-path reads, external-target communications, and trusted-party violations. Timeout 2 hours (PT2H).
4. `ENGINEERING_DIRECTOR` — final tier. Timeout 8 hours (PT8H).

If an escalation times out at any tier, the next tier is paged. If all four timeout, the `fail_safe_mode` of `FAIL_CLOSED` kicks in and the action is blocked.

Real deployments substitute concrete identities (an on-call rotation in PagerDuty, a Slack channel handle, a specific person's GitHub username) for these role names. The roles in the template are placeholders.

## Fail-safe and retention

`fail_safe_mode: FAIL_CLOSED` — when the policy layer can't reach a decision (escalation timeout, signature failure, mapping disagreement), the action is blocked. The alternative `SAFE_STATE` (return to a known-good baseline) doesn't apply cleanly to code agents because there's no obvious "baseline" mid-task. `DEGRADE` is too lenient for the failure modes this profile targets.

`audit_retention: P1Y` — one year. This is a defensible baseline for SOC 2 (which requires period-of-coverage retention, typically 12 months) and EU AI Act Art. 18 (which requires 10 years for high-risk providers but accepts shorter for non-high-risk). Deployers in regulated domains should lengthen accordingly. Deployers in less-regulated domains can shorten with disclosure.

## What this profile does not cover

The profile is not signed. The `authority` block is intentionally omitted from the template. Before deploying, the responsible authority signs the profile and the signature's `sequence_number` becomes the rollback-resistance anchor.

The profile does not encode observer independence level — that's a deployment property, not a policy one. The profile names what to do *when* an inconsistent CCD result arrives; how strong the CCD evidence is depends on which independence level (SAME_PROCESS → INSTITUTIONALLY_INDEPENDENT) the observer was provisioned at.

The profile does not list specific allowlisted external endpoints. Deployers add their own (e.g., `api.openai.com`, `api.anthropic.com`, their own backend, package registries) and reference the named allowlist in the `effects[].target not_matches_allowlist external_endpoints` condition.

The condition expression language above is illustrative rather than formally specified. A real deployment encodes these in whatever policy engine the team uses (OPA, Cedar, a Python evaluator, etc.). The intent of each condition is what's portable; the syntax is local.

## Open questions for reviewers

If you are a platform engineer or security lead reviewing this draft, these are the questions whose answers most usefully shape v0.2:

1. Is the working-directory-vs-outside split the right primary boundary, or do you use a different namespace (project-relative paths, container filesystem, etc.)?
2. Are there agent operations we're not capturing? Specifically: long-running shell commands, container exec, kubectl apply, terraform apply, AWS / GCP / Azure CLI calls?
3. Should `GOVERN.AUTHORITY.DISOBEY` default to `ALLOW` rather than `LOG`? The current setting records but doesn't gate; some teams want explicit verification that a refusal was justified.
4. What real escalation chains do you run? Names, timeouts, paging mechanism.
5. What sensitive-path patterns are missing from the credential check? The current list is `secrets, credentials, id_rsa, id_ed25519, .env, aws_access_key, private_key, token`.

Feedback via GitHub issues on the `jperdomo88/tao` repository, or directly to the maintainer.
