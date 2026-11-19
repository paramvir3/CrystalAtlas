# Dataset pipeline contract

CrystalAtlas treats dataset construction as a scientific compilation step, not
as permissive data cleaning. A retained target must pass five independent
boundaries:

1. an authorized reviewer must approve the declared use under an archived
   licence declaration;
2. the exact terms, selection specification, ordered record index, ordered
   payload index, per-record scope index, and data artifacts must be archived
   with their semantic roles and SHA-256 digests;
3. every row must lie in its declaration's machine-checkable scope and carry
   immutable source metadata, an evidence URI, and attribution;
4. strict parsing, FullPrim, the pinned Hall setting, canonical origin,
   tolerance-stable symmetry classification, exact Wyckoff inversion,
   continuous target lifting, and decoder round-trip must succeed; and
5. deduplication and each requested train/validation/test view must use frozen,
   audited identity keys.

This is conservative engineering, not legal advice. A factory in this package
is a review template, not a legal determination and not permission to acquire a
database. Terms and record-level rights can change. Every acquisition must
archive the terms and source metadata that applied on its retrieval date and be
reviewed for the intended use and jurisdiction.

No database dump is bundled with CrystalAtlas.

`RecordScopePolicy.SYNTHETIC_INTERNAL` exists only for structures authored as
internal synthetic fixtures with the exact claims `origin=synthetic` and
`third_party_content=false`. It is not a shortcut for acquired data, does not
confer rights on third-party content, and must never be used to relabel a
database export whose licence or lineage is unresolved.

## Provenance, scope, and trust boundary

`DatasetSourceDeclaration` records the provider, narrowly selected data scope,
licence identifier, reviewed terms digest, citations, attribution/notice
requirements, and tri-state permissions. Only `Permission.ALLOWED` authorizes a
requested operation; `UNRESOLVED` fails closed. `DatasetPolicyDecision` freezes
the exact `DatasetUseRequest`, source decisions, and resulting obligations.

Each `DatasetSnapshot` requires:

- at least one `DATA` artifact;
- exactly one `TERMS` artifact whose digest is the declaration's reviewed terms
  digest;
- exactly one `SELECTION_SPEC`, `RECORD_INDEX`, `PAYLOAD_INDEX`, and
  `SCOPE_INDEX` artifact, each bound to its correspondingly named digest; and
- a non-mutable snapshot identifier, retrieval timestamp and URI, exact record
  count, byte counts, and hashes for every artifact.

Artifact roles are part of the contract: a digest found in a data file cannot
silently satisfy the terms or index obligation. The selection/index digests are
pairwise distinct. Before crystallographic work begins, `DatasetCompiler`
re-hashes every local artifact and requires the supplied candidates to match the
archived record identifiers, payload digests, and scope evidence in exact
snapshot order.

Every `DatasetCandidate` carries `SourceRecordScopeEvidence`. It binds a
source-specific policy, exact scope claims, the archived upstream metadata
digest, an HTTPS evidence URI, and nonempty attribution. The compiler verifies
those claims against the declared source; it does not infer licence scope from
geometry, a record identifier, or the database name.

Receipt, build, split, and storage fingerprints detect accidental or malicious
modification only when their expected values come from somewhere trusted. A
self-fingerprint is not a signature: anyone able to replace a JSON file can
recompute it. Production consumers must retain the reviewed fingerprint outside
the dataset tree or verify a reviewer signature. The `read_json(...,
expected_fingerprint=...)` and `DatasetStore.open(..., expected_*_fingerprint=...)`
arguments provide this external trust-pin boundary. JSON readers also reject
duplicate keys and non-standard nonfinite numbers.

## Initial source review templates

The following factories encode narrow candidate scopes. An actual snapshot is
eligible only if contemporaneous archived evidence and every row's scope claims
establish the stated conditions; the table itself grants no rights.

| Declaration | Required row-level scope | Mandatory exclusion |
|---|---|---|
| `cod` | main COD crystallographic CIF collection identified as `CC0-1.0` | linked third-party payloads, software, and the separate predicted collection |
| `nomad-open` | explicitly open, reusable structure identified as `CC-BY-4.0` | restricted records and metadata-only rows without a reusable structure |
| `alexandria` | member of the declared versioned Alexandria release identified as `CC-BY-4.0` | material absent from that frozen release |
| `oqmd-generated-prototypes` | generated-prototype lineage identified as `CC-BY-4.0`, with no restricted upstream structure | ICSD-derived or other restricted third-party coordinates/prototypes |
| `materials-project-core` | non-access-controlled core record identified as `CC-BY-4.0`, outside MPContribs | MPContribs, access-controlled records, GNoME/BY-NC batches, and any differently licensed batch |

Restricted or proprietary sources such as ICSD and CSD are not covered by these
templates. Adding one requires a separate declaration of the applicable grant,
reviewed evidence, an authorized policy decision, and compliance with its
redistribution restrictions.

Official pages used to define these narrow review templates include:

- [COD CC0 statement](https://crystallography.net/cod/new.html)
- [NOMAD Terms of Use](https://cloud.nomad-lab.eu/nomad-lab/terms.html)
- [Alexandria licence page](https://alexandria.icams.rub.de/about.html)
- [OQMD downloads and data-licence statement](https://www.oqmd.org/download/)
- [OQMD structure lineages](https://www.oqmd.org/documentation/structures)
- [Materials Project terms](https://materialsproject.org/about/terms)
- [Materials Project database versions and special batches](https://docs.materialsproject.org/changes/database-versions)

## Crystallographic preprocessing gate

For a raw coloured periodic set \(C\), preprocessing constructs

\[
\bar C = \operatorname{PinnedHall}\!\left(
  \operatorname{FullPrim}(C)
\right)
\]

and records the raw-to-primitive transform, translation index, pinned Hall
setting, origin shifts, tolerances, backend version, and catalogue fingerprint.
Hidden translations, equivalent integer bases, declared/hidden supercells,
origin shifts, orbit permutations, and alternate settings therefore disappear
before identity construction, but only after their certificates and decoder
round-trip pass. Reflections are not quotiented: matching uses proper rotations
\(SO(3)\), so mirror-related chiral crystals remain distinct.

The pinned numerical policy also guards the classification boundary. With the
default policy, the nominal spglib classification must agree at
\(0.5\,\mathrm{symprec}\) and \(2\,\mathrm{symprec}\). Agreement covers the
space-group type, primitive/conventional counts, Wyckoff and site-symmetry
assignments, orbit partition, and species membership. A disappearing or changed
classification raises `AmbiguousPreprocessingError`; the receipt must state
`symmetry_stability_verified=True` before the compiler will retain the target.
This finite window is an explicit reproducibility test, not a proof that a noisy
experimental structure has a unique physical symmetry at every tolerance.

The accepted raw domain is intentionally strict: static, ordered, fully
occupied structures with supported species and finite, nonsingular geometry.
Invalid syntax, unsupported/disordered input, tolerance-unstable
crystallography, or identity-cell boundary ambiguity is recorded as an input-row
rejection with source reference, payload hash, stage, exception type, and
reason. Catalogue corruption, failed global certification, digest mismatch, or
an internal invariant violation aborts the build rather than being converted
into convenient row loss.

This compiler never reads an energy, stability label, model prediction, relaxed
model output, novelty score, or benchmark result to decide retention. It rejects
ambiguous or out-of-domain *preprocessing inputs*, not inconvenient generated
structures or model outputs.

## Policy-defined identities and deduplication

Before identity construction, the primitive translation lattice is put in a
proper-basis canonical presentation and repinned to the catalogue's Hall
setting. The finite search minimizes the largest squared basis-vector length,
then the sorted Gram diagonals, off-diagonals, and origin-quotiented coloured
motif. Every improving unimodular basis lies in the ellipsoid bounded by the
supplied basis, so the accepted search is complete within its declared numerical
guards. Only determinant-positive basis changes and Cartesian rotations are
used; mirror images are not deliberately identified.

The implementation uses fixed ranking grids, floating roundoff estimates, and
explicit resource ceilings, not interval-certified real arithmetic. Bin-wall
ambiguity, resolved complete-key ties, and exceeded search limits reject the
input. The downstream identity partition has its own guards. Version
`crystalatlas.canonical-structure-identity.v2` is not interchangeable with old
identities: rebuild canonical targets, deduplication, and split manifests from
the authenticated raw corpus rather than relabelling old hashes.

Floating tolerance is not used as a non-transitive pairwise "close enough"
relation. For each scalar component the identity partition is

\[
Q_q(x)=\left\lfloor x/q + \tfrac12\right\rfloor.
\]

A value whose guard intersects a bin wall is rejected as ambiguous. Intrinsic
fractions are partitioned exactly on \(\mathbb R/\mathbb Z\); their guard is at
least the certified Cartesian preprocessing tolerance divided by the smallest
singular value of the conventional lattice (plus a floating-point floor).
Accepted records therefore occupy deterministic, versioned identity cells.

Three deduplication keys are available:

- `EXACT`: exact rational Wyckoff parameters and hexadecimal metric values;
- `STRUCTURE`: binned full metric and binned intrinsic coordinates; and
- `POLYMORPH`: binned determinant-one metric shape and binned intrinsic
  coordinates, deliberately quotienting uniform volume rescaling.

The default compiler groups by `STRUCTURE`, so isotropic volume differences,
pressure-dependent snapshots, and other full-metric distinctions remain
separate. Within each structure cell it chooses one deterministic representative
while retaining every source alias, its aligned per-record scope evidence, and
attribution. Records with different structure keys at the same reduced
composition are retained; this is the pipeline's conservative operational form
of polymorph preservation. Deduplication never selects a single representative
for an entire composition: multiple geometrically distinct candidate polymorphs
survive as separate training records.

These are declared computational equivalence relations, not a universal
physical phase classifier. Quantization can neither prove thermodynamic phase
identity nor model experimental uncertainty in every lattice measurement. In
particular, metric guards are fixed numerical guards, not experimental error
bars. Choose and fingerprint `IdentityPolicy` for the scientific question.
`POLYMORPH` is an explicit, more aggressive scale-free option and should be used
only when uniformly rescaled observations are intended to share a training
record.

## Leakage-resistant split views

For split mode \(m\), records are grouped by an exact frozen key \(k_m\), and
whole groups—not individual rows—are assigned to train, validation, or test.
The compiler materializes a separate, authenticated split manifest for each
requested mode:

- `COMPOSITION`: exact reduced composition;
- `PROTOTYPE`: a conservative anonymous crystallographic protostructure—the
  pinned setting, formula-unit multiplier, and sorted species-role blocks of
  Wyckoff labels, with species names, lattice metric, and free Wyckoff
  coordinates removed;
- `SCAFFOLD`: the element-decorated discrete Wyckoff assignment in the pinned
  setting, including reduced composition and formula-unit multiplier, but no
  free coordinates or metric; and
- `ELEMENT_SET`: the exact all-and-only set of atomic numbers.

Thus prototype-disjoint evaluation withholds an entire discrete symmetry/orbit
template, not merely a near-duplicate geometry. It is intentionally
conservative. The four manifests are independent evaluation views; one should
not describe a composition-mode partition as simultaneously prototype-,
scaffold-, and element-set-disjoint.

Regardless of mode, records sharing a canonical polymorph key are coupled and
cannot cross partitions. Distinct polymorphs are not collapsed merely because
they share a composition. The allocator minimizes the deterministic Pearson
discrepancy

\[
L(n)=\sum_{s:f_s>0}\frac{(n_s-f_sN)^2}{f_sN}
\]

when assigning each next whole group, with SHA-256 ranks resolving ties.
Balanced group partitioning is combinatorial, so exact requested fractions are
not promised. Total assignment, mode-key disjointness, polymorph coupling,
deterministic replay, and manifest binding are promised and independently
re-audited.

## Compilation and authenticated storage

```python
from crystalatlas import (
    CatalogueDatasetTargetBuilder,
    CrystallographyCertification,
    CrystalAtlasConfig,
    DatasetCompiler,
    DatasetSourceManifest,
    DatasetUseRequest,
    IdentityPolicy,
    SpglibPolicy,
    SpglibStandardizer,
    build_full_catalogue,
    pinned_hall_numbers,
)
from crystalatlas.dataset_storage import DatasetStore, write_compiled_dataset
from crystalatlas.ingestion import CatalogueTargetAssigner

catalogue = build_full_catalogue()
crystallography = CrystallographyCertification.read_json(
    "crystallography-receipt.json"
)
sources = DatasetSourceManifest.read_json(
    "dataset-sources.json",
    expected_fingerprint=trusted_source_manifest_fingerprint,
)
config = CrystalAtlasConfig()
standardizer = SpglibStandardizer(pinned_hall_numbers(), SpglibPolicy())
assigner = CatalogueTargetAssigner(catalogue)

compiler = DatasetCompiler(
    catalogue=catalogue,
    crystallography_certification=crystallography,
    source_manifest=sources,
    use_request=DatasetUseRequest(train_model=True),
    target_builder=CatalogueDatasetTargetBuilder(
        standardizer=standardizer,
        target_assigner=assigner,
        domain=config.domain,
    ),
    identity_policy=IdentityPolicy(require_all_230=True),
)

dataset = compiler.compile(
    candidates,  # every row, in archived order, from every declared snapshot
    artifact_roots=roots,  # keyed by (source_id, snapshot_id)
    split_seed=20260826,
)

storage_manifest = write_compiled_dataset(
    dataset,
    "compiled-dataset",
    records_per_shard=1024,
)
# Persist storage_manifest.fingerprint in a separate trusted system.

with DatasetStore.open(
    "compiled-dataset",
    expected_build_manifest_fingerprint=dataset.manifest.fingerprint,
    expected_source_manifest_fingerprint=sources.fingerprint,
    expected_storage_fingerprint=storage_manifest.fingerprint,
) as store:
    first = store[0]                 # lazy, read-only memory-mapped arrays
    same = store[first.record_id]    # stable identity lookup
```

`DatasetCandidate` may contain strict CIF text or a validated
`RawPeriodicStructure`. The compiler binds the exact target-builder,
crystallography receipt, catalogue, identity policy, use request, policy
decision, retained targets, rejection receipts, aliases/scope evidence, and
split-manifest fingerprints into the build manifest.

`write_compiled_dataset` publishes deterministic typed `.npy` shards with
explicit `int64` ragged offsets and canonical JSON metadata; pickle and object
arrays are forbidden. It also stores the full source-receipt manifest, build
manifest, record index, licence/attribution `NOTICE.txt`, and requested split
receipts. Publication is transactional at directory granularity: a private
sibling tree is written and fsynced, authenticated with the production reader,
and then renamed; an existing destination is never overwritten.

`DatasetStore.open` validates the externally expected fingerprints when
provided, the storage manifest, embedded source and build manifests, record
index, notice, split receipts, every file byte count and SHA-256, shard
schemas/dtypes/shapes, ragged-offset invariants, and record-to-manifest bindings
before exposing read-only memory maps. Symlinks, extra files, missing files,
duplicate JSON keys, and unauthenticated mutations fail closed.

Compilation currently constructs the canonical record set in memory before
writing shards. Very large acquisitions should be divided into independently
pinned source snapshots and deterministic compilation jobs, while still
performing the global identity merge and freezing final split manifests before
training.

### Reproducible local direct-CIF entry point

`load_local_cif_candidates` closes the filesystem seam between archived source
receipts and `DatasetCandidate`. It reads the canonical record, payload, and
scope indexes; requires identical ordering; revalidates each row's declared
scope; and maps every payload digest to exactly one individually pinned `DATA`
artifact. It will not infer how an opaque archive was extracted. An archive or
API export therefore needs a separate, content-addressed extraction/indexing
receipt before this narrow entry point can consume it.

`compile_local_cif_dataset` then runs the existing compiler, publishes the
authenticated `DatasetStore`, reopens the final store with all expected
fingerprints, and returns a path-independent `LocalDatasetBuildReceipt`. The
receipt embeds the complete `LocalCifInputManifest`, including the source-row
to data-artifact bindings, and has canonical `to_dict()` and `to_json()` forms
for CLI integration:

```python
from crystalatlas import compile_local_cif_dataset

result = compile_local_cif_dataset(
    compiler,
    expected_source_manifest_fingerprint=trusted_source_manifest_fingerprint,
    expected_crystallography_receipt_fingerprint=(
        trusted_crystallography_receipt_fingerprint
    ),
    artifact_roots=roots,
    destination="compiled-dataset",
)
print(result.to_json(), end="")
```

Both expected fingerprints must come from outside the acquisition/storage tree
or from a verified reviewer signature. The returned receipt is self-hashed but
not signed; persist `result.fingerprint` separately as the build trust anchor.

## Declared domain limits

Each limit below is a declared support boundary; it is not a statement about physics. A record outside it is rejected with the quoted reason, never repaired.

| Limit | Legacy v6 value | What it excludes | Versioned alternative |
|---|---|---|---|
| Primitive-ranking resolution | `256·eps·‖G‖₂ < 10⁻¹⁰/8 Å²`, i.e. ‖G‖₂ < 219.9 Å² or σ_max(L) < 14.8 Å. Reason: `primitive metric exceeds canonical numerical resolution` | Every cell with one long axis, whatever its symmetry | `SpglibPolicy(relative_metric_quantum=1e-9)`: scale-free, so this guard can no longer trigger (`docs/PEER_REVIEW_REMEDIATION_2026-10.md`, B2/M7) |
| Lattice chart ball | ‖log(G₀^{-1/2} G G₀^{-1/2})‖_F < B = 2, with G₀ the setting's reference shape at 20 Å³/atom. Reason: `coefficient lies outside the open bounded chart ball` | At fixed volume, c/a may differ from the reference by at most a factor e^{√6/2} ≈ 3.40 for hexagonal, trigonal and tetragonal commutants. The reference c/a is 0.866 for hexagonal axes and 1 for tetragonal ones, so the maximum is about 2.95 and 3.40 | `DomainConfig(lattice_chart="loewner-v2")`, the default since 2026-10-20 (`ball-v1` is the explicit legacy option): the open Loewner interval l(N)G₀ < G < uG₀. The lower bound l(N) = s_min²/λ_min(G₀(N)) certifies σ_min(L) ≥ 0.75 Å exactly, independent of the upper bound. The upper bound u = e^{4.5} admits long axes up to e^{2.25} ≈ 9.5× the reference length. It covered 496 of 496 held-out records against 450 for the ball (B2) |
| Atom bound | ν·\|c\| ≤ 40 conventional atoms, i.e. 40/I_G primitive atoms: 10 for F-centred and 13 for R-centred settings | Large F- and R-centred primitive cells | `DomainConfig(max_primitive_atoms=20, max_raw_atoms=80)`: MP-20's own definition, ν·\|c\| ≤ 20·I_G |
| Raw-to-ideal lattice fidelity | Componentwise rtol = atol = 2·10⁻⁸. Reason: `reported idealized lattice exceeds the frozen raw-frame input-fidelity tolerance` | DFT-relaxed cells about 10⁻⁷ from ideal; most trigonal and hexagonal inputs | `SpglibPolicy(idealization_tolerance_angstrom=τ)`: every cell point moves at most τ Å (B1) |
