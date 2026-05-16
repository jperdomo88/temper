# Healthcare Provider Agent Mission Profile — Draft 0.1

**Status:** Draft starting point pending validation by clinicians, healthcare privacy officers, hospital compliance teams, and medical ethics committees. The profile reflects U.S. HIPAA constraints and general clinical-practice norms. Deployers operating under different regimes (EU MDR, UK MHRA, etc.) layer their own additional constraints. **This profile is not a substitute for clinical judgment or regulatory legal review.**

**Companion file:** `healthcare_provider.json`.

---

## What this profile is for

A "healthcare provider agent" is an autonomous system embedded in clinical workflow — EHR-integrated clinical decision support, patient-facing triage and follow-up, clinician-facing summarization, prior-auth automation, medical billing AI. Examples: AI scribes (Abridge, Nuance DAX), clinical decision support agents in Epic / Cerner, patient-facing triage chatbots, AI radiology pre-reads, AI-augmented mental-health platforms with crisis routing.

The profile assumes the agent operates in a context where a licensed clinician retains medical decision authority. The AI surfaces information, summarizes, drafts documentation, flags concerns — but it does not diagnose, prescribe, or treat on its own authority. Deployments that operate without a clinician in the loop are outside this profile's scope and may not be lawful in the deployer's jurisdiction.

## The failure modes this profile targets

**Practicing medicine without a license.** An AI making a diagnosis or prognosis to a patient, recommending a medication, or prescribing a treatment without a licensed clinician's authorization is unauthorized practice. This is the most-cited concern in regulatory commentary on healthcare AI.

**HIPAA breach via the path of least resistance.** PHI travels easily. An agent retrieving a record without a treatment relationship, sharing data with research/analytics destinations without de-identification, posting PHI into chat threads that aren't part of the chart — each is a potentially reportable breach.

**Coerced consent under medical vulnerability.** A patient in a medical crisis, in significant pain, or recently diagnosed with serious illness is in a high-vulnerability context. Persuasion under those conditions — even toward genuinely correct medical choices — needs human judgment, not AI optimization.

**Treatment side effects without acknowledgment.** Therapeutic actions (medications, procedures, restraints) routinely have side effects that are `RESOURCE.DAMAGE` under TAO's vocabulary. The mapping rules permit them only when `harm_acknowledged` is present and the action chain points at informed consent or emergency-doctrine authority. The profile makes this enforcement explicit.

**Therapeutic deception.** Lying to a patient — even with benevolent intent ("don't tell them they're dying, it'll make them lose hope") — is a long-running ethics issue. The profile blocks `HARM.DECEIVE.LIE` and `COMMUNICATE.OBFUSCATE.CONFUSE` outright. Whether or not to disclose to a patient is a human clinical judgment, not an AI action.

**Capability restrictions on patients.** AI placing safety holds, restricting medication access, ordering involuntary confinement — these require human clinical authorization. The agent surfaces the concern; the clinician makes the call.

## Verb-by-verb design

### Treatment — `PROTECT.HEAL.TREAT`, `HARM.DAMAGE.STRIKE`

`PROTECT.HEAL.TREAT` defaults `ESCALATE`. Treatment is a clinician verb. The exception allows clinical-decision-support uses where the agent's role is information surfacing — reminders, protocol cues, alerts. The condition explicitly excludes "prescribe," "order," "diagnose" from the allowed purposes.

`HARM.DAMAGE.STRIKE` defaults `BLOCK`. The exception is the canonical surgery / aggressive-therapy case: the harm occurs in a `PROTECT.HEAL.TREAT` clinical context, harm is acknowledged in justification, and the authority chain cites informed consent or emergency doctrine. This is the spec's surgical-incision example operationalized.

### Communication — `COMMUNICATE.INFORM.TELL`, `PERSUADE.CONVINCE`

`INFORM.TELL` defaults `ALLOW` but escalates on diagnosis, prognosis, and medication communication. These three categories are practice-of-medicine territory.

`PERSUADE.CONVINCE` defaults `ESCALATE`. Influencing a patient's medical decision requires a clinician. The exception allows informed-consent persuasion when the explicit informed-consent protocol is being followed.

### PHI — `INFO.DISCLOSE`

Default `ESCALATE` to `PRIVACY_OFFICER`. The exceptions encode HIPAA's "Treatment, Payment, Operations" (TPO) framework:

- Disclosing to the patient about their own record `ALLOW`. Patients have access rights.
- Disclosing to a treating clinician or care team member with care-team authorization `ALLOW`. This is TPO's "Treatment" prong.
- Disclosing to insurance / payer with HIPAA TPO authorization `ALLOW`. This is "Payment."
- Disclosing to research / registry / analytics `BLOCKS` unless the data is HIPAA Safe Harbor de-identified. This is the area HIPAA breaches most commonly originate.

### Restrictions — `CAPABILITY.RESTRICT`

Default `ESCALATE`. The exception allows clinician-authorized or emergency-protocol-authorized safety holds. The asymmetry: the AI cannot decide to restrict a patient; only flag for clinician decision.

### Money — `EXCHANGE.TRANSFER.PAY`

`BLOCK`. Clinical agents shouldn't be handling payment. Billing AI is a different deployment with a different profile.

### Authority — `GOVERN.AUTHORITY.OBEY`

Two safety overrides:

- If obeying would produce `PROTECT.HEAL.TREAT` and the authority chain doesn't cite a licensed provider or a signed order, `BLOCK`. The AI cannot be "talked into" treatment by anyone who isn't licensed to order it.
- The standard "no harm or deception even when ordered" block.

## Escalation chain

Four tiers tailored to clinical contexts:

1. `ATTENDING_CLINICIAN` (2 minutes) — the responsible clinician for the patient. Short timeout because clinical workflow is real-time.
2. `PRIVACY_OFFICER` (15 minutes) — HIPAA / privacy escalations.
3. `MEDICAL_DIRECTOR` (1 hour) — for clinical-policy questions or novel cases.
4. `ETHICS_COMMITTEE` (24 hours) — for fundamental conflicts (capacity-impaired patient, religious-objection care, end-of-life issues).

## Fail-safe and retention

`fail_safe_mode: SAFE_STATE`. In healthcare, halting outright (FAIL_CLOSED) is rarely the safest action — better to revert to a known-safe baseline (e.g., "this AI cannot act, escalating to the human clinician with current context preserved").

`audit_retention: P10Y` — ten years. Aligns with HIPAA's six-year minimum and most state malpractice statute-of-limitations windows. Pediatric records often require longer (until the patient reaches age of majority + statute window); deployers in pediatric contexts extend.

## What this profile does not cover

This profile is *not* a clinical AI safety framework. It addresses HIPAA-adjacent governance and verb-level constraints, but does not certify the underlying model's clinical safety. FDA's Software-as-a-Medical-Device (SaMD) framework, EU MDR, and equivalent regimes have separate certification regimes that complement, not replace, this profile.

The profile assumes the agent is operating in a clinical context with treatment relationships properly established. Patient-facing direct-to-consumer healthcare AI (without a treatment relationship) has additional regulatory complications.

The profile does not address controlled substances, behavioral health emergencies, or pediatric-specific protections in depth. Each warrants its own extension.

## Open questions for reviewers

1. The line between "information surfacing" and "diagnosis/prognosis" in `COMMUNICATE.INFORM.TELL` is hard to draw cleanly. Where does "your labs suggest X" fall — surfacing or diagnosis?
2. `EXCHANGE.TRANSFER.PAY` is `BLOCK` because payment is a different deployment. But what about cost-of-care discussions ("this procedure costs $X")? Is that `INFORM.TELL` and fine, or something else?
3. Crisis routing (suicide / overdose / acute psychiatric) deserves a dedicated path. How should this profile handle it — escalate to a specific role, or defer to a separate "crisis profile" overlay?
4. For pediatric, mental health, and substance-use deployments, what additional restrictions should appear?
5. The HIPAA-centric framing won't fit non-US jurisdictions. Should there be a generic "patient data protection" condition that deployers fill in with their regional regime, rather than HIPAA-specific terms?
