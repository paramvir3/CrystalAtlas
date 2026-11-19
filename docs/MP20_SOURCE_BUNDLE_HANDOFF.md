# Historical MP-20 source-bundle boundary

This is an implementation handoff, not an authorization or completed source
manifest. The frozen source review and provider evidence are retained under
`artifacts/mp20_pilot_2026-09-14/`. No new provider permissions are asserted here.

The existing dataset orchestration supports compilation, publication and
authenticated reopening. The remaining MP-20-specific work is an offline,
exclusive-output source-bundle builder with these properties:

- Leave the original extraction immutable. Bind all 512 selected IDs in their
  original order; do not refill preprocessing failures.
- Bind CIF and metadata hashes, row identifiers, container digests, source
  revision and ordered indexes to `selected-bindings.json` and the actual bytes.
  The generic loader carries a metadata digest but does not independently require
  an archived metadata artifact with that digest; this builder must check it.
- Archive the dated provider clarification as the terms evidence. Retain the
  license text, source review, original README, acquisition/extraction receipts,
  metadata and container pins as auxiliary evidence. Do not relabel the dated
  clarification as current website terms or a provider-signed review.
- Use a narrowly versioned historical-CDVAE-MP20 scope declaration. The current
  `materials_project_core_source()` helper is not appropriate unchanged: its
  current terms URL was not retrieved, and the MP-20 rows do not supply the
  modern core/MPContribs/access-control/license fields its scope policy requires.
  Label reviewed provenance inference explicitly; never invent those fields.
- Keep the original Materials Project database version unknown, preserve the
  disclosed ICSD-derived starting lineage and DFT relaxation, and attribute
  Materials Project and CDVAE. Request only the intended research-training use.
- Retain the source-manifest fingerprint outside the bundle. A self-fingerprint
  establishes internal integrity, not legal permission or a provider signature.

Only after that declaration is supported may `compile_local_cif_dataset()` run
under its authorized use request. Use STRUCTURE deduplication to preserve
polymorphs and four separately audited composition/prototype/scaffold/element-set
split views. Preserve the 512-source denominator. Retain the local build receipt
and reopen the store using external source/build/storage pins before constructing
`StoredTrainingDataset` from explicit training-partition IDs.

Tests must reject changed evidence, mismatched metadata/CIF/ID bindings, changed
ordering, unsupported permissions and wrong external fingerprints. This source
binding work and current crystallographic preprocessing require CPU, not GPU.

## Concrete API boundary reviewed on 24 September

The smallest implementation adds a new
`RecordScopePolicy.CDVAE_MP20_HISTORICAL` with versioned value
`cdvae-mp20-historical-f857f598-v1`; existing policies and receipt schemas must
not be weakened. Existing string-valued scope claims can distinguish reviewed
assertions from byte-derived metadata. Require an exact historical claim set:

| Claim | Required content |
| --- | --- |
| `benchmark` | `cdvae-mp20` |
| `source_revision` | `f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea` |
| `source_partition` | `train` |
| `source_container_sha256` | Externally pinned original train.csv digest |
| `scope_assertion_basis` | `reviewed-historical-provenance` |
| `reviewed_lineage` | `icsd-initialized-mp-dft-relaxed` |
| `reviewed_upstream_license` | `CC-BY-4.0` |
| `record_license_metadata` | `not-provided` |
| `upstream_database_version` | `unknown` |
| `provider_signed` | `false` |
| `review_artifact_sha256` | Externally pinned source-review bytes |
| `selected_bindings_sha256` | Pinned original selected-bindings.json digest |

These are proposed policy fields, not an instantiated authorization. Keep
`source_metadata_sha256` equal to the original unchanged row-metadata digest.
A historical source factory should default training permission to UNRESOLVED
and require an explicit reviewed declaration for research training; possession
of evidence files alone must not set ALLOWED. Unrequested uses stay unresolved.

The archive-binding validator must run in **both**
`load_local_cif_candidates()` and `DatasetCompiler._verify_snapshot_contract()`.
A check only in the bundle-builder script can be bypassed by directly creating
candidates. Rehash the archived metadata after reading, and bind its ID, CIF,
partition, row index, container digest and selected rank to the ordered binding
index. Require the cited review/binding artifacts and dated TERMS evidence by
digest and semantic role. Reject unsupported extra claims, modern MP-core flags,
wrong metadata relationships, reordered IDs and substituted terms evidence.

This design audit created no new policy, source manifest, permission grant, or
training dataset. Implementation remains pending. The frozen 512-record
preprocessing run has now completed; its results and coverage limitations are
retained in `artifacts/mp20_pilot_2026-09-24/README.md`.

## Implementation follow-up — 30 September 2026

The paragraph above records the state of the original design audit. The following
foundation is now implemented and tested:

- `RecordScopePolicy.CDVAE_MP20_HISTORICAL` has the exact reviewed claim set,
  including the externally pinned review and selected-binding hashes. Extra
  modern MP-core flags and invented version/license/signature claims reject.
- `historical_cdvae_mp20_source()` defaults training permission to UNRESOLVED.
  An explicit `Permission.ALLOWED` argument represents a reviewer assertion,
  not a provider signature or a permission inferred from file possession.
  Unrequested uses remain unresolved.
- `historical_mp20.require_historical_mp20_archive()` binds actual metadata and
  CIF bytes to the ordered selected-binding archive. It also requires the pinned
  original training container, source review and dated TERMS evidence with their
  correct semantic roles. It verifies consumed bytes again after the preliminary
  snapshot check, rejects symlinks/nonregular inputs, and uses bounded reads.
- Both `load_local_cif_candidates()` and the compiler's snapshot-contract check
  enforce this boundary. Direct candidate construction does not bypass it.
- Artifact lookup is indexed by `(role, digest)` once per snapshot, rather than
  scanning the full artifact collection separately for every record.

Existing source-policy behavior and receipt schemas are unchanged. No numerical
physics, crystallographic tolerances, targets, model parameters or training
objectives were modified in this follow-up. Prior GPU evidence is historical
evidence for its pinned wheel, not a new GPU execution of these changed sources.

Validation: 132 passing tests across historical scope enforcement, provenance,
builder/orchestration, storage, splits, identity and benchmark parsing; Ruff
passed on changed production modules and tests. Mypy was not available in the
selected local Python runtime, so no static type-check pass is claimed.
Report: `artifacts/provenance_2026-09-30/regression-tests.xml`.

Still pending: the exclusive-output offline bundle assembler, inclusion of the
complete auxiliary acquisition/extraction/license/README evidence, creation of
an externally pinned real source manifest, and compilation/reopening of the
frozen pilot into audited split views. The current guard tests use explicitly
synthetic trust anchors where needed; they neither authorize real training nor
claim a completed real-data archive audit. The v6 pilot baseline remains
193 accepted / 319 rejected out of 512, documented in
`artifacts/mp20_pilot_2026-09-25/README.md`.

### Offline assembly completed (2026-09-30 follow-up)

`scripts/prepare_mp20_source_bundle.py` now assembles the original frozen pilot
without network access, changing inputs, filtering records, or granting training
permission. It includes the full acquisition/extraction, license, source-review,
provider clarification and upstream README evidence. Input bytes are rehashed
at consumption; output paths must be new, external pins cannot modify input
trees, and the source manifest is written last as the completion marker.

Local result: `data/source-bundles/mp20-512-historical-unresolved-v1`, excluded
from Git. All 512 records and 1,039 artifacts passed archive loading. A separate
process reopened the saved manifest, verified its SHA and both manifest
fingerprints against `artifacts/provenance_2026-09-30/source-bundle-pin.json`,
loaded all 512 candidates, and confirmed training is denied. The external pin
is an integrity anchor, not a provider signature or a legal authorization.

The expanded regression suite passed 140 tests (no failures/errors/skips),
recorded separately in `regression-tests-r2.xml`; the earlier report is retained.
Ruff passed. No GPU, numerical-equation change or tolerance relaxation was used.
Still pending: resolve the intended training-use authorization explicitly, then
compile/reopen audited training split views. Archive loading is not canonical
target compilation or evidence of prediction quality; the v6 preprocessing
baseline remains 193 accepted / 319 rejected.

### Reusable independent auditor (2026-10-02)

`scripts/audit_mp20_source_bundle.py` now repeats the independent read-only
check using a separately trusted pin SHA. It passed on the frozen 512-record,
1,039-artifact bundle. Fourteen synthetic tests exercise valid loading and
corrupted inputs; reproduction instructions and evidence are recorded in
`artifacts/provenance_2026-10-02/README.md`. Training permission remains
unresolved, and no canonical targets or training splits were produced by this
audit. Permission review must precede changing the source declaration; ordinary
archive integrity success must never serve as that approval.
