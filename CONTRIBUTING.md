# Contributing to TAO

TAO is a working draft. Contributions, hard critique, and adversarial review
are all welcome and explicitly invited. The goal of this document is to make it
easy for you to engage at whatever depth makes sense.

## Ways to contribute

**File an issue.** The fastest contribution. Categories that are especially
useful right now:

- **Verb gap.** You've shipped an action class that doesn't fit any of the 29
  normative verbs and can't be covered by a registered extension. File an issue
  describing the action and the closest verbs that didn't fit.
- **Effect gap.** The nine mechanical effect types miss something your system
  produces that has audit relevance.
- **Observer model gap.** Your platform's isolation architecture doesn't map
  cleanly to one of the five observer-independence levels.
- **Override discipline edge case.** You can construct a Mission Profile that
  passes the §7.3 checks but launders something we'd prefer to catch.
- **CCD false positive or false negative.** Concrete claim/check pair that the
  validator classifies wrong.

**Add a test vector.** If you have a concrete claim/check or tuple/profile pair
that should pass or fail, open a PR adding it to `spec/test_vectors.json` with
the expected result and a short description. The conformance test suite is the
authoritative test of what the spec means; growing the suite tightens it.

**Register an extension.** If your domain needs verbs or effects beyond the core,
the extension registry (spec §9.2) is the mechanism. Add a YAML file under
`spec/extensions/` (TBD path) with the namespace, maintainer, verbs, mappings,
and definitions. Open a PR. There is no central authority; merge is by
repository maintainers under the published contribution criteria.

**Improve the implementation.** The Python validator is the reference; alternate
implementations are encouraged. The validator passes 21/21 published vectors.
A PR that adds a vector and the code to handle it is the cleanest contribution.

**Push back on the spec.** Spec text changes are accepted via PR with a
migration note. A MAJOR or MINOR version bump is expected if your change
affects tuple validity; PATCH for editorial-only changes. The public comment
period for proposed normative changes is at least 14 days.

## What kinds of contributions are NOT useful right now

- Vocabulary disputes that don't touch a real deployment. "Should verb X be
  named Y instead?" is interesting eventually but not the current bottleneck.
- Bikeshedding on the semantic-mechanical mapping for edge cases nobody is
  shipping. The reference mapping is a working draft; it will get revised
  in response to deployment evidence, not aesthetic preferences.
- New conformance levels. The current two (TAO, TAO-Attested) are
  deliberately minimal. Domain-specific levels belong in domain regulator
  documents, not the core spec.

## Quick start for code contributors

```bash
git clone <repo>
cd tao-spec
pip install -e .[dev]
tao check-suite spec/test_vectors.json
```

If `tao check-suite` returns 21/21, the environment is good.

## Style notes

- The Python is Python 3.10+ with type hints. No dependency on anything more
  exotic than `jsonschema`.
- Specification text is strict CommonMark with blank lines around lists,
  headers, and code blocks.
- Issue and PR descriptions: name the spec section you're touching (e.g.,
  "§4.6 mapping rules", "§7.3 override discipline") so reviewers can locate
  context quickly.

## Governance and disclosure

TAO is a working draft. There is no formal standards body, no governance
committee, and no funding source. Maintenance and review are done by the
project maintainers in their own time. If TAO accumulates enough adoption to
warrant a formal governance structure, that structure will be defined
publicly in this repository before being adopted.

If you find a security issue, please email the maintainer directly rather
than opening a public issue. Contact: jorgeperdom@gmail.com.

## Code of conduct

Be direct, be technical, and treat the work seriously. Don't make personal
critiques where technical ones would do. Disagreement is welcome; contempt
is not. Substantive critique of the spec, the vocabulary, or the
implementation is exactly what this project is for.
