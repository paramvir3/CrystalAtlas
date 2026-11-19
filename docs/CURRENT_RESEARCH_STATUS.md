# Research snapshot — 15 October 2026

This is a development/research checkpoint, not a production-ready release or a
claim of superior crystal-generation accuracy.

## Included work since the previous Git snapshot

- Historical MP20 source-scope enforcement, offline bundle assembly and auditing.
- Periodic enumeration precision, bounded-work and input-validation regressions.
- Convex-hull feasibility/energy checks and bounded bootstrap calculations.
- A frozen real-data compilation runner and bounded, factor-first training runner.
- Test receipts, provenance records and failed-run evidence as well as successes.

The 512-input MP20 pilot compiled locally to 193 accepted and 319 rejected
structures, without replacement. Composition/prototype/scaffold/element-set split
views were frozen. A separate reviewed declaration permits the bounded training
use; other redistribution/commercial uses remain unresolved.

The pilot uses eight fixed training records and plans 192 optimizer updates across
separate composition, scaffold-selection and geometry phases, with no joint
training. Its explicit all-118 composition vocabulary retains a selected ThSe3
record; geometric tolerances and the rest of the physical domain are unchanged.

## Latest GPU outcome and open blocker

All 15 runner regressions passed on pc10, including CUDA RNG preservation.
The real-data job then failed before any optimizer update when Linux rebuilding
did not reproduce the stored Mac float64 lattice latents bit-for-bit. The measured
eight-record differences are at floating-point rounding scale; no tolerance was
relaxed to force a pass. No learned pilot checkpoint or real-data loss improvement
is claimed.

A separate code review identified a broader SVD-nullspace coordinate-frame risk:
sign fixing does not eliminate orthogonal basis mixing in a multidimensional
nullspace. A production portability fix must bind/version chart frames and
separate byte authentication from numerical semantic validation. That fix remains
unimplemented; same-stack recompilation is only a research workaround.

Next: recompile the original 512 inputs on pc10's numerical stack into a new store,
compare accepted identities, rejections, split assignments and all eight selected
records, then rerun the bounded pilot with fresh pins. The user approved transfer
of the source bundle, but the last attempted transfer did not execute because the
approval service reported a usage-limit error. Recompilation/retraining remain
pending. GPU work is paused for this user-requested repository snapshot.

## Deliberately local-only files

Raw/source bundles, compiled datasets, checkpoints and the two coordinate-bearing
`portability-*-r1.json` diagnostics are excluded from Git. Their scripts, summary
findings, hashes and appropriate audit records are included. Nothing is published
as a public dataset, trained model or release by this private-repository snapshot.

Detailed evidence: `artifacts/research_pilot_2026-10-12/STATUS_2026-10-14.md`.

## Snapshot verification

The focused local regression suite completed with 174 passed, one CUDA-only test
skipped, and one full-catalogue builder-contract test deliberately deselected.
Ruff and the source/documentation whitespace checks passed. The archived failed
compiler-test XML retains its original traceback whitespace and therefore reports
trailing-whitespace warnings in an unrestricted staged diff check; that evidence
was not rewritten. This is not a rerun of the entire repository's test suite;
the prior GPU result is separately preserved evidence. Receipt:
`artifacts/research_pilot_2026-10-12/github-snapshot-tests-2026-10-15.xml`.
