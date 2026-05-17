# Mission Profile templates

**Status: draft starting points pending industry validation.** Every profile in this directory is published as a working draft. None has been signed off by domain practitioners. They exist to give organizations adopting TAO a concrete shape to react to, adapt, and harden — not a ready-to-deploy artifact.

Each profile is two files: a `.json` deployable template and a `.md` companion that explains the choices made.

## A note on what Mission Profiles are

Mission Profiles are not a new institutional category. Every regulated domain already has the equivalent under different names: Rules of Engagement in military contexts, clinical protocols and scope-of-practice rules in hospitals, standard operating procedures in factories, fiduciary duty rules at financial firms, treaties and conventions with specified escalation ladders in international relations. These rules already exist as paper documents, training materials, and case law. They specify who is authorized to do what, what requires escalation, and what is prohibited regardless of circumstance.

What the substrate contributes is making those existing rules machine-readable. The same rules a human resident or junior analyst is trained on become the rules the AI tool actually follows, producing the same audit-grade record when applied. The templates here are not propositions about what an organization's values *should* be — those choices remain the responsibility of the medical directors, compliance officers, risk committees, and policy boards who already own them. The templates are propositions about *what shape the rule looks like* when written in a form a substrate can enforce.

## What's here

| Profile | Status | Audience for first review |
|---|---|---|
| `code_agent.json` · `code_agent.md` | Draft 0.1 | Frontier-lab platform engineers and security teams |
| `browser_agent.json` · `browser_agent.md` | Draft 0.1 | Computer-use deployers and the operators of agentic browsers |
| `customer_service.json` · `customer_service.md` | Draft 0.1 | Contact-center ops, consumer-protection regulators |
| `financial_services.json` · `financial_services.md` | Draft 0.1 | Registered advisors, compliance, AML officers |
| `healthcare_provider.json` · `healthcare_provider.md` | Draft 0.1 | Clinicians, privacy officers, hospital compliance |
| `enterprise_tool_agent.json` · `enterprise_tool_agent.md` | Draft 0.1 | Enterprise IT, info security, data protection |
| `education.json` · `education.md` | Draft 0.1 | Teachers, district IT, FERPA / COPPA compliance |

Each profile is a working draft inviting domain-practitioner pushback. The profiles deliberately differ in `fail_safe_mode`, `audit_retention`, and escalation-chain structure to match the operational realities of their domains. They are not interchangeable — picking the wrong profile for your context produces wrong outcomes by design.

Defense, government, retail, and research-assistant profiles are not in this initial batch and will be added when domain practitioners engage.

## What a Mission Profile does

A Mission Profile is the policy layer TAO operates under. The spec defines TAO as policy-neutral; the profile is where policy lives. A profile names which verbs are allowed, blocked, or escalated, who gets called on escalation, how long audit logs are retained, and which observer independence level the deployment commits to.

The profile is signed, versioned, and bound to emitted tuples by `profile_hash` so the audit trail records *what policy was in force* at the time an action occurred. A reviewer six months later can answer "who signed off on this configuration?" by looking at the profile chain.

## How to use these templates

1. Read the `.md` companion. It describes the threat model the profile assumes, the verbs it focuses on, and the choices made for each exception.
2. Copy the `.json` file into your repository.
3. Replace placeholder fields (`profile_name` to match your org, `effective_date`, `audit_retention` per your retention policy, `escalation_chain` with real role identifiers in your organization, `authority` block with a real signature).
4. Adapt verb overrides and exception conditions to your deployment. The templates are designed to be a thoughtful starting point, not a complete answer for your specific stack.
5. Sign the profile with the responsible authority's key. Bind tuples to the profile's hash by adding `provenance.profile_hash` to each emitted tuple.
6. Validate the profile structure with `tao validate-profile your_profile.json` (planned for v0.12) or directly against `spec/tao_mission_profile.schema.json` with any JSON Schema validator.

## What these templates are NOT

- They are **not** legal advice. Compliance with EU AI Act, ISO 42001, SOC 2, HIPAA, GDPR, PCI DSS, etc. requires organizational processes, documentation, and audits beyond what any technical profile encodes. The companion `TAO_COMPLIANCE_CROSSWALK.md` discusses which provisions TAO substrate satisfies.
- They are **not** comprehensive threat models. Each profile reflects the threats that are common in the named domain, not every possible one.
- They are **not** signed. The `authority` block is intentionally omitted from the templates. A deployment signs the profile before it goes live.
- They are **not** the only way. A domain may need different verb overrides, different escalation chains, different fail-safe modes. The templates demonstrate the shape; the content is the deployer's call.

## Contributing

If you operate in one of the named domains and want to harden a template against real-world threats, open a pull request or an issue. Useful contributions:

- A real adversarial scenario the current exception rules would miss.
- A real escalation chain from a regulated organization (with role names you can publish).
- A real audit retention policy from your jurisdiction.
- An entire new domain profile with the same `.json` + `.md` two-file structure.

Profiles will be added or revised in response to real deployment experience, not hypothetical refinement.
