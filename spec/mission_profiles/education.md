# Education Agent Mission Profile — Draft 0.1

**Status:** Draft starting point pending validation by teachers, school administrators, district IT, learning-science researchers, and FERPA / COPPA compliance officers. The profile addresses both K-12 and higher-education contexts but acknowledges they differ significantly.

**Companion file:** `education.json`.

---

## What this profile is for

An "education agent" is an autonomous system used in teaching and learning — AI tutors, AI teaching assistants, AI grading assistants, AI study tools deployed in schools or platforms, AI accessibility supports (text-to-speech, real-time captioning, IEP-aligned assistive tools). Examples: Khan Academy's Khanmigo, Duolingo's AI tutors, Quizlet AI, university LMS-integrated tutors, AI proctoring tools, district-deployed AI assistants like MagicSchool / SchoolAI / Brisk.

The profile assumes the agent is operating in an educational context where the learner may be a minor, where student records are protected under FERPA (U.S.) or equivalent (EU GDPR Article 6 for children's data, UK age-appropriate-design code, etc.), and where academic integrity has institutional standing.

## The failure modes this profile targets

**Academic-integrity laundering.** Student asks the AI to "help with my essay." The AI writes the essay. The student submits it. The school's academic-integrity policy is violated, but neither the student nor the AI explicitly framed the interaction as cheating. The profile blocks the literal write-the-work pattern and escalates ambiguous "help" patterns in graded contexts.

**FERPA disclosure.** AI agent retrieves student records to answer a question and shares specifics with someone who isn't the student, parent (for minors), or a school official with legitimate educational interest. FERPA violations are reportable; willful violations can cost a school its federal funding.

**Unconsented biometric monitoring.** AI-driven proctoring tools watch students through webcams and microphones during exams. The consent framework varies by jurisdiction and student age. The profile blocks proctoring without explicit consent and escalates biometric monitoring of minors.

**Persuasion of vulnerable students.** A student in crisis (suicidal ideation, abuse disclosure, eating-disorder indicators) interacts with an AI tutor. Even well-meaning AI advice in those contexts requires human escalation to a counselor, not AI-driven persuasion.

**Identity confusion / authority bypass.** A child claims to be a parent / teacher / adult to access restricted functionality. The agent has weak identity verification. Authority chain check fails.

## Verb-by-verb design

### Teaching — `COMMUNICATE.INFORM.TELL`, `COOPERATE.ASSIST.HELP`, `CREATE.GENERATE.PRODUCE`

The core academic-integrity guard lives across these three.

`COMMUNICATE.INFORM.TELL` defaults `ALLOW` (teaching is the whole point). The exception `BLOCK`s when purpose explicitly includes "answer for student," "complete assignment," "write paper for." This is the literal cheating-enabling pattern.

`COOPERATE.ASSIST.HELP` defaults `ALLOW` with two `BLOCK` exceptions:
- "Do the work for / complete on behalf of / submit as own" — explicit substitution.
- Any assistance in `GRADED_ASSESSMENT` context unless the purpose is "prepare for" rather than "do." This requires the deployment to set `environment.domain = "GRADED_ASSESSMENT"` when active assessments are happening; the agent then refuses to do the assessment work.

`CREATE.GENERATE.PRODUCE` defaults `ALLOW` with a `BLOCK` for essays/papers/theses generated in graded-assessment context.

The pattern: the agent can teach extensively, but it cannot produce student work as the student.

### Persuasion under vulnerability — `COMMUNICATE.PERSUADE.CONVINCE`

Default `ALLOW`. Persuasion is part of teaching ("you should keep trying").

Exception: `vulnerability.level = HIGH` or `CRITICAL` escalates to a `TEACHER`. Educational vulnerability covers a wide range: struggling student, learning disability, mental health crisis, family emergency, abuse disclosure. Each requires human judgment, not AI persuasion.

### FERPA-equivalent — `INFO.DISCLOSE`, `OBSERVE.SENSE.QUERY`

`INFO.DISCLOSE` has two strict `BLOCK` conditions:
- Student records to anyone other than the student/parent without `ferpa_authorized_disclosure` cited.
- Minor's records to external/third-party destinations unconditionally.

`OBSERVE.SENSE.QUERY` reading student records escalates unless `legitimate_educational_interest` is cited (the FERPA-defined standard for internal access).

### Monitoring — `OBSERVE.MONITOR.WATCH`

The hardest area in modern educational AI.

Camera / microphone / biometric monitoring of minors escalates to `PRINCIPAL` for case-by-case decision. The condition catches the most-litigated patterns: proctoring tools watching K-12 students.

Proctoring `BLOCKS` outright unless `proctoring_consent_obtained` is cited in justification. This pushes the consent mechanism upstream — the deployment proves consent was obtained before the agent watches anyone.

### Authority and identity — `GOVERN.AUTHORITY.OBEY`

The minor-specific exception: when the student is a minor and the instructing authority isn't a verified educator or verified parent/guardian, escalate to `PRINCIPAL`. The pattern catches the "AI talks to child without supervision" scenario.

Standard "no harm or deception even when ordered" safety override applies.

### Capability restrictions — `CAPABILITY.RESTRICT`

Default `ESCALATE` to `TEACHER`. The exception allows safe-search / content-filter activations under their explicit purpose.

### Flagged verbs

`HARM.DECEIVE.LIE` and `COMMUNICATE.OBFUSCATE.CONFUSE` default `BLOCK`. An education AI lying to students is a particularly bad failure mode.

## Escalation chain

Five tiers reflecting educational hierarchy:

1. `TEACHER` (15 min) — primary educator responsible for the student.
2. `REGISTRAR` (1 hour) — student-records authority.
3. `PRINCIPAL` (4 hours) — for monitoring decisions and identity-verification escalations.
4. `DPA_OFFICER` (4 hours) — Data Protection Authority designate (FERPA officer in U.S., DPO in EU contexts).
5. `SCHOOL_BOARD` (1 day) — for systemic or policy-level escalations.

Real deployments adapt — higher education has different roles (advisor, ombuds, registrar, provost) and many K-12 districts consolidate roles.

## Fail-safe and retention

`fail_safe_mode: SAFE_STATE` — students whose tutoring session gets stuck shouldn't be left hanging.

`audit_retention: P5Y` — five years. Aligns with most FERPA retention periods and U.S. state education-records retention. Higher education extends for credential-verification purposes (sometimes indefinitely for transcripts).

## What this profile does not cover

The profile blurs K-12 and higher-education contexts. They differ significantly: minors vs. adults, in loco parentis vs. autonomous learners, FERPA's parental-access provisions, COPPA implications for under-13 users, IDEA / IEP requirements for students with disabilities. A real deployment splits this profile into two (`k12.json` and `higher_ed.json`) and adds the regulatory frameworks specific to each.

The profile does not address research ethics for AI-augmented educational research (IRB requirements, informed consent for data collection, etc.).

The profile does not address content-classification questions — what content is age-appropriate, what subjects can be discussed with what age groups. Those are content-moderation rules, not behavioral rules.

The profile does not cover AI used by teachers as a tool (lesson planning, grading assistance, parent communication drafting). Those are a different deployment with different risk patterns; layer the `enterprise_tool_agent.json` profile.

## Open questions for reviewers

1. The "GRADED_ASSESSMENT" context flag depends on the deployment setting it correctly. How is this likely to fail in practice — students working on graded material outside official assessment windows, or assessment windows not marked correctly?
2. The K-12 / higher-ed split: better to fork this into two profiles, or keep one and parameterize via context fields?
3. What's the right pattern for IEP-aligned accessibility supports? Currently the profile treats them like any educational interaction; some accommodations need higher-trust paths.
4. AI-driven proctoring is heavily contested. Should this profile take a stronger position (e.g., default-BLOCK on biometric monitoring regardless of consent), or is the current escalation-based approach correct?
5. What additional protections are needed for under-13 users (COPPA)? The current profile relies on the deployment setting `vulnerability.factors` correctly, which is fragile.
