# Frozen MP-20 pilot extraction

This is a bounded real-data **inspection and preprocessing handoff**, not a
training authorization, compiled dataset, or predictive-quality result.

## Source and scope

The source is the CDVAE MP-20 CSV release at commit
`f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea`, the revision explicitly linked in
[CrystaLLM's benchmark instructions](https://github.com/lantunes/CrystaLLM/blob/main/BENCHMARKING.md).
The frozen CSV files contain **27,136 training, 9,047 validation and 9,046 test
records**, for 45,229 total. These counts are parsed CSV records, not text lines;
each quoted CIF field contains multiple lines. The upstream README's 45,231
description is not substituted for the actual container counts.

`scripts/acquire_mp20.py` verifies upstream Git blob identities and byte counts.
`scripts/prepare_mp20_pilot.py` independently pins the revision, each container's
Git blob identity, byte count and SHA-256, and requires the acquisition receipt's
SHA-256 as an explicit argument. A self-hash stored alongside a modified source
tree is not an external trust anchor or a licence authorization.

The default pilot selects 512 records from the published **training split only**,
using seed `20260913`. Each training ID is ranked by SHA-256 of the compact JSON
array `["crystalatlas.train-pilot-selection.v1", seed, material_id]`, with the ID
as tie breaker. All selected rows are preserved in rank order. Selection does
not inspect energies, structures, species, space groups, model outputs or target
conversion success. Rejected preprocessing records must not be replaced by new
draws. The command allows smaller counts but rejects counts above 512.

## Offline entry point

Supply the separately retained acquisition SHA-256; the placeholder below is
deliberately not an executable value:

```bash
PYTHONPATH=src python scripts/prepare_mp20_pilot.py \
  data/benchmarks/mp20-cdvae-f857f598 \
  data/pilots/mp20-512-seed20260913 \
  --expected-acquisition-sha256 TRUSTED_ACQUISITION_SHA256 \
  --count 512 --seed 20260913
```

No network access, upstream code execution, `eval`, pickle loading, CIF
reinterpretation, physical validation or optimization occurs in this command.
Raw property metadata remain strings; a CIF's UTF-8 bytes are preserved exactly
after CSV field decoding, including its original line endings.

## Output contract

| Artifact | Binding |
| --- | --- |
| `selection.json` | Frozen revision, acquisition digest, train-only selection algorithm, count and seed; no refill policy. |
| `source-rows.jsonl` | Every source row in train/val/test order, zero-based CSV row index, original ID, original metadata strings, CIF digest, metadata digest, container digest and selected/not-selected flag. |
| `selected-bindings.json` | Every selected rank and ID linked to original train row, source URI, container digest, exact CIF file and metadata file. |
| `record-index.json` | Selected IDs in deterministic rank order, using the dataset pipeline's canonical record-index encoding. |
| `record-payload-index.json` | The same ordered IDs bound to exact CIF SHA-256 values. |
| `cifs/<SHA256>.cif` | Byte-preserving selected CIF payloads. Identical bytes are stored once without collapsing their source-row bindings. |
| `metadata/<SHA256>.json` | Canonical source-row metadata; its file SHA-256 equals `BenchmarkCifRow.metadata_sha256`. |
| `extraction-receipt.json` | Source identities, all parsed counts, selected count, every output artifact's path/role/size/hash, and a canonical self-fingerprint. |

All three original containers are verified and every row is indexed, but holdout
CIF files are not extracted for training. Cross-split identifier overlap aborts
the gate. Neither the original random splits nor this subset are represented as
composition-, prototype-, scaffold- or element-set-disjoint partitions.

Publication reserves a fresh directory and writes files exclusively. Existing
destinations are never overwritten, including incomplete earlier attempts.
Every output is re-hashed before `extraction-receipt.json` is written last. An
interrupted directory without that receipt is incomplete and retained for
inspection; it must not be consumed. This is a completion-marker protocol, not
an atomic directory publication or a guarantee against a hostile concurrent
filesystem writer. Downstream dataset publication uses the stricter existing
`DatasetStore` contract.

## Rights and downstream training boundary

The extraction receipt always records `source_rights_status="unresolved"` and
`training_authorized_by_this_receipt=false`. It deliberately creates no source
scope evidence, `Permission.ALLOWED` flags, reviewer identity, or
`DatasetSourceManifest`. Software repository licensing is not a grant of rights
to upstream crystal data. The MP-20 README discloses ICSD-derived initial
lineage and DFT relaxation; preserve that provenance rather than silently
labeling these records as independently acquired ICSD data or as new MP data.

A separately reviewed declaration must bind the archived provider evidence to
the exact selected data and intended uses. Dated provider clarification must
be labeled with its actual date and origin, not relabeled as current website
terms. Once that boundary is satisfied, the extracted files and metadata can
populate the existing snapshot, scope index and source-manifest contracts.

Then run canonical target compilation, STRUCTURE deduplication with polymorph
preservation, frozen split-view audits and the exact decoder gates before
opening authenticated `StoredTrainingDataset` batches. Record every rejection
and retain the 512 selected-source denominator. This handoff does not claim
that every selected structure satisfies the current atom-count, formula-unit,
species, metric-support, symmetry-stability or canonical-identity domain.

## Completed extraction, 14 September 2026

The offline command was executed with the separately supplied acquisition file
SHA-256
`3efeb934ea6fdc4800242c54b05c1211d89e432f92553533e1a6563632fbd095`,
count 512 and seed 20260913. Output is
`data/pilots/mp20-512-seed20260913`.

The completed extraction has receipt fingerprint
`a91ccc2eb224a9aeb3f77f840bbdba3c0a10ac5523424359e6d245dc98557276`
and receipt-file SHA-256
`512585859b2def771cb7cf31cd1df72884ba0fe17f225b8f371a3c6857932894`.
It contains 1,029 manifest artifacts (512 CIF files, 512 metadata files and five
indexes/specifications), plus the completion receipt. Manifest-listed artifacts
total 37,486,832 bytes; the all-source metadata index accounts for most of this
space. None of the 512 selected source IDs was replaced or filtered.

A separate standard-library audit reconstructed every one of the 45,229 source
row metadata bindings from the original CSV containers, recomputed the selected
512 IDs without calling the selection API, compared every selected CIF byte for
byte with its original CSV field, and verified all artifact lengths and hashes.
All checks passed. The first five selected IDs are `mp-697047`, `mp-634945`,
`mp-1226790`, `mp-1223088`, and `mp-1220781`.

The preparation script and benchmark CSV reader tests passed **31 tests**;
the XML result is
`artifacts/mp20_pilot_2026-09-14/mp20-extraction-tests.xml`.
Ruff checks passed for the preparation script and its new tests. This is a
transport/extraction validation result only; it is not a successful canonical
target compilation, a source-rights authorization, or training evidence.

## Preprocessing checkpoint, 23 September 2026

The fixed first 32 of the original 512 selected records have now completed the
canonical target, exact-decoder and identity preflight without an internal
exception: **8 passed, 24 rejected**, no refills. The remaining 480 records were
not audited by this run. Rejections comprise 11 primitive ranking-bin boundaries,
6 symmetry-stability failures, 4 lattice-fidelity failures, 2 metric resolution
limits and 1 bounded chart-domain failure. These are preprocessing limitations,
not a statement that the rejected crystals are physically invalid.

The follow-up corrected origin/identity species-order mismatches, separated exact
primitive topology from reported lattice idealization, and verified proper
affine pinned-setting round trips instead of demanding coordinate identity.
Current standardization algorithm is `fullprim-minimax-proper-primitive-pinned-hall-v4`,
explicitly bound by dataset-builder schema v3; identity policy fingerprint is v2.
Numerical tolerance values were not increased. Prior fatal reports remain
archived, and previously passing targets retain their fingerprints.

Evidence is `artifacts/mp20_pilot_2026-09-23/preprocessing32_r5.json` and
`regression-tests.xml` in that directory (**267 focused tests passed**).
Before training, complete the frozen-pilot audit, investigate coverage, and
finish the separate historical source-bundle authorization/binding described in
`docs/MP20_SOURCE_BUNDLE_HANDOFF.md`. No GPU is required for these remaining
preprocessing and provenance steps.

## Full frozen-pilot checkpoint, 24–25 September 2026

All **512** selected records have now been audited: **187 passed, 325 rejected,
zero internal failures and zero not-run**. The first full attempt stopped at
record 287 and remains archived. A separate internal-composition audit now
distinguishes a corrupted basis/rotation map from accumulated input idealization
exceeding the unchanged endpoint fidelity predicate. Standardizer algorithm v5
is bound by target-builder schema v3. All 103 targets that passed the first full
attempt retained identical fingerprints in the completed rerun.

Evidence and rejection counts are in
`artifacts/mp20_pilot_2026-09-24/README.md` and `preprocessing512_r2.json`.
Three retained suites contain 365 distinct passing cases: 274 focused regressions,
17 extended crystallography tests and 74 report-summary tests, with no skips.
The strict offline `scripts/summarize_mp20_preflight.py` preserves complete,
aborted and not-run denominators using an externally supplied report digest.

Only 36.5% passed the current contracts. Numerical ranking/resolution rejected
169 records, symmetry instability 87, lattice fidelity 66, and declared atom/
chart domains 3. This attrition must be investigated before representative
training or performance claims. Twelve passed records also disagree with the
upstream space-group metadata under the current reconstruction policy; that
requires investigation, not automatic relabeling to match upstream.

No training DatasetStore or source authorization has yet been created. Coverage
analysis and historical source-bundle binding remain CPU tasks; a GPU is still
not needed for this stage.
