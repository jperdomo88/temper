# For a healthcare compliance officer

**Reader's question:** Can I show this to the medical director and an auditor?

## The short answer

Yes. TAO produces an audit trail in language a compliance officer recognizes: who acted, on what authority, for whose benefit, with what observed effect. It does not produce model-explanation reports. The trail it produces is the kind a Joint Commission surveyor, an OIG auditor, or a malpractice review board can read directly.

## What it covers, in your language

The substrate captures the AI agent's behavior — what it told a patient, what it recorded in the chart, what it routed, what it deferred. Every observable AI action becomes a tuple with:

- The **clinical role** the agent claimed (triage, patient education, intake, dosing decision support, billing-code assistance, etc.)
- The **scope of that role** as registered with the organization (what the agent is authorized to do under its role definition)
- The **observed effects** of the action (information disclosed to whom, resource transferred where)
- A **claim-check delta**: whether the agent's described action matches an independent observer's record of the action

The clinical-scope piece is the one most relevant to your work. The example in [`healthcare_off_scope_advice.md`](../scenarios/healthcare_off_scope_advice.md) walks through a triage agent that crosses into medication recommendation. The CCD catches the boundary crossing because the agent's registered scope (symptom triage and routing) doesn't include medication advice. The substrate produces a record showing the attempt was caught and corrected.

The point is not to say "the AI did the right thing." The point is to give your team a behavioral record sufficient to answer "what did the AI do here, and was that within its authorized scope?" — the question every audit, complaint, and incident review asks.

## What it does *not* require

You don't replace your existing HIPAA controls, your audit log, your access controls, or your medical-staff bylaws. The TAO substrate emits structured records that flow into your existing audit pipeline. PHI handling sits inside your existing boundaries; the tuple schema is designed to reference patient identifiers, not embed clinical content.

That said, TAO logs themselves may be regulated records depending on your deployment — they reference patient encounters, agent actions taken in clinical context, and authority chains naming clinical roles. Treat them under your existing access controls, retention requirements, and audit-log integrity protections. "No PHI in the message body" is not the same as "outside the regulated boundary."

You don't need vendor lock-in. The spec is open. Multiple vendors (or your in-house engineering team) can emit conformant tuples. Auditors don't need a vendor's tooling — they can read the schema.

## What changes for your role

Three concrete things become easier:

1. **Defining scope for AI tools.** The Mission Profile schema is where your medical director's "what is this allowed to do" gets written down in a machine-readable form. The same definition the engineering team implements is the definition the auditor reads. The [`healthcare_provider.json`](../mission_profiles/healthcare_provider.json) starting point is a draft to negotiate from, not a finished policy.

2. **Investigating after an incident.** When a complaint comes in, you query the audit log for the patient's encounter and see the AI's actions in the same structured form as any human staff action. The reconstruction isn't a research project — it's a query.

3. **Demonstrating control to regulators.** The audit trail directly supports the question "how do you supervise this AI tool?" with structured evidence, not narrative reassurance.

## What it won't do

TAO won't tell you whether the AI's clinical judgment was good. It won't replace clinical oversight. It won't certify the model. It records what the AI *did*. The judgment of whether what it did was *right* still requires you and your medical director.

Full spec: [`TAO_v0_11.md`](../TAO_v0_11.md). Healthcare profile starting point: [`mission_profiles/healthcare_provider.md`](../mission_profiles/healthcare_provider.md). Worked failure mode: [`scenarios/healthcare_off_scope_advice.md`](../scenarios/healthcare_off_scope_advice.md).
