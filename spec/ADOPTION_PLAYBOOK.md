# TAO adoption playbook

This is a practical guide for a team adopting TAO in an existing deployment. It is written for the engineer or operator who has read the spec and is asking: *what do I actually do on Monday?*

The playbook assumes nothing about your stack beyond the existence of an AI agent that takes effectful actions (tool calls, API requests, database writes, message sends, file operations) and a place where you currently log those actions.

## Three operational tiers

TAO adoption is not a single thing. Three distinct tiers give three distinct assurance levels, and they should not be confused with each other. The most common adoption failure is treating the decorator tier as if it were the audit-grade tier.

| Tier | What you ship | What it gives you | What it does NOT give you |
|---|---|---|---|
| **TAO-Log** | Decorator-only structured claims emitted to a sink you control. | A behavioral record in a known schema; queryable across agents; usable for routine debugging, incident reconstruction, and regulatory inquiries that just need "what happened." | Audit-grade evidence. The decorator runs inside the agent, so a buggy or hostile adapter produces a clean-looking record of what the adapter *says* happened. |
| **TAO-Check** | TAO-Log plus an independent observer producing check tuples plus CCD evaluation. | A claim-vs-check delta on every action: divergences between what the agent says and what was actually observed get flagged. Catches adapter errors and a meaningful class of agent misbehavior. | Runtime enforcement. CCD-INCONSISTENT cases get logged, not blocked. Effects still land. Coverage is bounded by what the observer can see (see TAO v0.11 §6.6). |
| **TAO-Governed** | TAO-Check plus the CCD result and Mission Profile evaluation wired into the agent's effect surface as a blocking or escalation gate. | Pre-effect or in-flight halt on the failure modes the profile flags. The substrate is now *governing* the agent, not just observing it. | A safety system. TAO does not certify model alignment. Even at this tier, the substrate is necessary, not sufficient; safety work is its own discipline. |

A two-sentence summary if you're skimming: TAO-Log is cheap and audit-grade-curious; TAO-Check is the dimension that distinguishes TAO from ordinary logging; TAO-Governed is what catches the failure before it lands. Most teams start at TAO-Log, run there for a release cycle, then move to TAO-Check on the highest-cost effect surface first.

## Stage 1 — TAO-Log: Drop in the decorator

**Goal:** Start emitting TAO-conformant tuples for every effectful action your agent takes. Do not change the agent's behavior. Do not add a check tuple yet. Do not change your existing logs.

**Effort:** A few hours to a day for a typical Python agent stack. Proportionally similar for other languages once an emitter is written for that stack.

**Steps:**

1. **Install the reference package.** `pip install tao-spec` (or pin to a commit hash for production). The package includes the JSON Schema, the verb-to-effect mappings, and the `@tao_emit` decorator.

2. **Identify your agent's effect surface.** List the tool calls or API methods your agent invokes. For each, identify the closest TAO verb from the 29-verb taxonomy ([`TAO_v0_11.md`](TAO_v0_11.md) §6). When in doubt, pick the verb whose REQUIRED effect best matches what the tool call actually does.

3. **Decorate each effect-surface function.**

   ```python
   from tao.adapters import tao_emit

   @tao_emit(verb="COMMUNICATE.INFORM.TELL",
             actor_id="customer_support_agent_v2")
   def send_email(to, subject, body):
       # existing implementation
       ...
   ```

   The decorator emits one tuple per call. The actor, verb, and effects are filled from the decorator and the function's signature; context and justification fields can be supplied or defaulted.

4. **Wire up a sink.** The decorator writes to a sink you configure. For the first integration, the `JsonlSink` writing to a file is the lowest-friction choice. Production teams typically route to their existing log pipeline (a structured-logging library, a Kafka topic, an S3 prefix, a SIEM).

   ```python
   from tao.adapters.sinks import JsonlSink
   from tao.adapters import configure_emitter

   configure_emitter(sink=JsonlSink("/var/log/agent/tao.jsonl"),
                     actor_id="customer_support_agent_v2")
   ```

5. **Verify with the reference validator.** Run `tao validate /var/log/agent/tao.jsonl` after a few minutes of agent activity. Every line should parse against the schema. Mismatches usually mean the verb you picked doesn't permit one of the effects you're claiming — re-check the verb mapping.

**What you have at the end of Stage 1:** a behavioral record of your agent's actions in a known schema. You can answer questions like "which agents called `EXCHANGE.TRANSFER.PAY` in the last 24 hours?" without learning a vendor-specific log format. This alone is materially useful for incident response and regulatory inquiries.

## Stage 2 — TAO-Check: Add an independent observer

**Goal:** Emit Check tuples from an observer that has independent visibility into what the agent did. Compute Claim-Check Deltas.

**Effort:** One to four engineering weeks, depending on what observability you already have and how much of the agent's effect surface you want to cover. Most teams start with a single high-cost effect family (payments, external network egress, message sends to external parties) and expand.

**Steps:**

1. **Pick the effect family with highest cost-of-quiet-failure.** Where would an unrecorded effect hurt the most? For a code agent, network egress. For a customer-support agent, message sends. For a browser agent, payments and subscriptions. Start there.

2. **Build (or buy) the observer.** The observer needs visibility *independent* of the agent's adapter — i.e., it must see what actually happened, not what the agent says happened. Common implementations:
   - A sidecar process reading from a system-level egress monitor for network effects
   - A second process reading from the database WAL for write effects
   - A separate logging agent watching the messaging API for send effects

   The observer emits a Check tuple in the same schema, with `observer_independence_level` set to the appropriate ladder rung (TAO v0.11 §6.5).

3. **Declare the observer's coverage.** Independence is where the observer runs; coverage is what it can see. A `PRIVILEGE_ISOLATED` observer that only watches wrapper-level tool returns is independent and uninformative. The check tuple's `provenance.coverage` block (TAO v0.11 §6.6) declares the channels the observer watches and — equally importantly — the channels it does not. Reviewers reading your CCD output need this to interpret the result. Without a coverage declaration, an attested-citation CCD is not valid.

4. **Configure the CCD evaluator.** The reference Python package's `tao.ccd` module compares paired Claim and Check tuples and emits a CCD result. Configure it to consume the two streams and emit a third stream (or annotate the Claim).

5. **Run in shadow mode for two sprints.** Don't wire CCD output to enforcement yet. Watch the rate of CCD-INCONSISTENT cases on a representative slice of traffic. False positives almost always come from coverage gaps the observer didn't know it had; tune those first. After two sprints, the inconsistency rate should be a stable signal rather than noise, and the residual cases are the real ones.

6. **Set thresholds for inconsistency response.** Default after shadow mode: log CCD-INCONSISTENT cases for human review. Graduate to escalation or blocking via Mission Profile (Stage 3) once the rate is stable and the residual mismatches are real.

**What you have at the end of Stage 2:** the dimension that distinguishes TAO from ordinary logging. You can catch effects the agent didn't mention (the `COMMITMENT.MAKE` in the browser-agent scenario) and purposes the agent claimed but didn't realize (the deflected refund in the customer-service scenario).

## Stage 3 — TAO-Governed: Write a Mission Profile and wire it to enforcement

**Goal:** Codify your organization's policy — what's allowed, what escalates, what blocks — as a Mission Profile that the substrate enforces.

**Effort:** Two to six weeks, mostly *not* engineering work. The work is alignment between the engineering team, the compliance/legal team, and (where applicable) the medical director, risk officer, or registered advisor.

**Steps:**

1. **Start from a published profile.** The `mission_profiles/` directory contains starting points for code agents, browser agents, customer service, financial services, healthcare, enterprise tools, and education. None are finished policies for your organization; they're drafts to negotiate from.

2. **Convene the right people.** A Mission Profile is policy expressed in a machine-readable form. Your compliance team and your engineering team need to read it together. Where applicable, your regulator-facing counsel should review the profile before it goes live.

3. **Document deviations.** When you weaken a default (e.g., changing an `ESCALATE` to `ALLOW` because you have an additional control elsewhere), the profile schema requires a deviation report explaining why. This isn't bureaucracy — it's what makes the profile auditable. The next compliance officer to read the file should understand why each choice was made.

4. **Wire the profile back to the agent's effect surface.** The simplest integration: a middleware step that reads the agent's intended action, evaluates it against the profile, and either passes it through, escalates, or blocks. The reference validator's profile-evaluation logic gives you the function signature.

5. **Decide your fail-open vs. fail-closed default.** The profile schema is explicit about this. For low-cost effects (e.g., a draft email), failing open (passing through and logging) is reasonable. For high-cost effects (e.g., a payment), failing closed (blocking until human review) is the prudent default. Be explicit; don't leave it implicit.

**What you have at the end of Stage 3:** the substrate is not just observing your agent, it's *governing* it within the bounds your organization has explicitly authorized. The audit trail now includes not just behavior, but the policy that produced each behavior. An incident review six months later can see not just what happened, but what the policy was at the time.

## Who does what

A common failure mode is treating TAO as "the engineer adds a decorator and governance happens." The substrate spans multiple functions and no single role owns it end-to-end. The work splits roughly as follows.

| Role | Owns |
|---|---|
| **Agent / platform engineer** | Emitting claim tuples. Selecting verbs for the agent's effect surface. Wiring the decorator into existing tool calls. Routing emissions to the configured sink. |
| **Security / platform team** | Building or buying the independent observer. Operating the observer at the declared independence level. Owning the coverage declaration honestly. Routing check tuples into the same audit pipeline. |
| **Compliance / risk** | Approving the Mission Profile. Reviewing deviation reports on profile overrides. Setting retention and access policy on the audit log. |
| **Domain owner** (medical director, registered advisor, etc.) | Defining the agent's authorized scope. Owning the `permitted_actions` / `prohibited_actions` lists in the relevant profile. Resolving cases where an authority chain is plausible but not specifically authorized. |
| **Audit / procurement** | In vendor relationships, requiring TAO-conformant emissions to a customer-controlled sink with a defined coverage declaration. Setting retention requirements. |
| **Incident response** | Using CCD outputs and deviation reports during investigations. Defining playbooks for which CCD inconsistency classes trigger which escalation paths. |

The substrate gives each role a shared schema to work in. None of these roles can be elided. If your compliance team isn't reading Mission Profiles, you have logging, not governance. If your security team isn't owning the observer, your CCD output is whatever the adapter chose to emit.

## A note on sequencing

Skipping Stage 1 to go directly to Stage 2 or 3 is a common temptation and almost always a mistake. The decorator is cheap, the value compounds, and most teams discover during Stage 1 that their effect surface is more diverse than they assumed. Building an observer for an effect surface you haven't characterized is expensive.

The other common mistake is skipping Stage 3 — staying at "we log everything" without writing down the policy. Mission Profiles are how you avoid the dynamic where every incident requires reconstructing what the policy *should have been*. Write it down once, before you need it.

## The "drop into Claude" prompt template

When you hand the TAO repository to an AI assistant (Claude, GPT, etc.) and ask it to help your team adopt the substrate, a prompt like this works:

> I'm adopting TAO in an existing deployment of [describe agent: language, framework, effect surface]. I've read TAO_v0_11.md, the mission_profiles/ directory, and the scenarios/ directory.
>
> My agent's effect surface is roughly: [list tool calls or API methods].
>
> Walk me through Stage 1 of ADOPTION_PLAYBOOK.md for my specific stack. For each effect-surface function, suggest the closest TAO verb from §6 and flag any cases where the mapping isn't clean.

The AI assistant has the whole spec in context. It can write the decorator wiring for your specific tool calls, map your effect surface to verbs, and suggest where the verb taxonomy doesn't quite fit (which is useful feedback to send upstream as a spec issue).

This is the intended adoption pattern: the substrate is small enough to fit in a model's context, the patterns are explicit enough to be applied, and the work that requires human judgment (which verb is right, what should escalate, where to weaken a default) is the work the human team should be doing anyway.

## Links

- Full spec: [`TAO_v0_11.md`](TAO_v0_11.md)
- Mission Profile starting points: [`mission_profiles/`](mission_profiles/)
- Worked scenarios: [`scenarios/`](scenarios/)
- Stakeholder one-pagers: [`stakeholders/`](stakeholders/)
- Reference Python package: [`../tao/`](../tao/)
- Validator spec: [`REFERENCE_VALIDATOR_SPEC.md`](REFERENCE_VALIDATOR_SPEC.md)
- Compliance crosswalk: [`TAO_COMPLIANCE_CROSSWALK.md`](TAO_COMPLIANCE_CROSSWALK.md)
