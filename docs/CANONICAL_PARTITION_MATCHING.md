# Strict canonical-partition matching

CrystalAtlas diagnostics now provide a deliberately narrow production boundary
for exact equality of accepted canonical partitions. This is not a
tolerance-based physical structure-similarity matcher and is not a substitute
for a separately frozen benchmark criterion.

For every dataset target and valid generated candidate, the experiment must
first run the same certified path:

\[
C \longmapsto \operatorname{FullPrim}(C)
\longmapsto \operatorname{ProperPrimitiveSection}(C)
\longmapsto \operatorname{PinnedHall}(C)
\longmapsto \operatorname{CanonicalOrigin}(C)
\longmapsto \operatorname{ExactWyckoff}(C)
\longmapsto Q(C).
\]

`CatalogueDatasetTargetBuilder` performs the all-230 target construction, and
`build_canonical_identity` constructs the versioned partition. Only after those
steps should `encode_structure_partition` or `encode_scaffold_partition` be
called. A generated candidate that fails decoding, standardization, exact orbit
assignment, round-trip, or identity construction is invalid and retains its
ranked diagnostic slot.

Each immutable envelope binds the complete canonical payload and its verified
key to the following externally supplied digest claims:

- the authenticated dataset-manifest digest;
- target-builder, catalogue, identity-policy, and crystallography-receipt
  fingerprints;
- CrystalAtlas and crystallography-backend artifact digests; and
- the frozen execution-environment digest.

Digest binding is not artifact authentication. The experiment harness must
resolve these digests against externally trusted receipts and prove dataset
membership before encoding cases; a self-hash in the same output tree is not a
signature.

`encode_structure_partition`, `encode_scaffold_partition`,
`validate_canonical_partition_payload`, and both production evaluators require
the live `CertifiedCatalogue` and `IdentityPolicy` objects in addition to their
fingerprints. Validation resolves the pinned Hall tuple and every Wyckoff row
against that catalogue, checks catalogue multiplicities and intrinsic
dimensions, enforces policy bin domains and canonical orbit order, and checks
that the reconstructed quantized metric is positive definite and satisfies the
selected point-group metric constraints within the exact quantization-error
bound. A self-consistent JSON object naming an invented setting or row is
therefore rejected even if its inner SHA-256 is recomputed.

Production metrics must call `evaluate_canonical_scaffold_retrieval` or
`evaluate_canonical_geometry_decomposition`; these wrappers prevalidate every
payload and bind the evaluation dataset manifest to the matcher provenance
before invoking the generic diagnostic engine. A provenance, policy,
catalogue, level, inner-schema, or payload/key mismatch is a protocol error; it
aborts evaluation rather than becoming a model miss. Canonicalization failures
are represented as invalid ranked attempts with `payload=None`; raw failure
artifacts belong in the separately authenticated generation log, not in the
canonical matcher input. The wrappers then use
the closure-free `canonical_structure_partition_equal` or
`canonical_scaffold_partition_equal` callable. These functions perform only
exact immutable-byte equality, allowing the diagnostic machinery to
fingerprint their implementation and recheck determinism and symmetry.

The structure and scaffold contracts are intentionally separate. Structure
payloads retain the guarded intrinsic-coordinate and full conventional-metric
partitions. Scaffold payloads are derived from the verified structure payload,
remove continuous parameter bins and the metric, and are checked against the
stored `scaffold_key`. Scaffold retrieval therefore compares the maximal
post-standardization pinned scaffold, not merely the cover selected by the
model.

## Scientific limitation

The current canonical identity propagates certified positional uncertainty
into fractional-coordinate bin guards. It does not yet propagate a certified
lattice uncertainty into the metric-bin guards. Consequently every provenance
manifest, envelope, and matcher contract states:

```text
metric_uncertainty_certified = false
representation_invariant_physical_equivalence_claim = false
```

Exact equality is a valid equivalence relation on the accepted canonical
payload bytes. Until lattice uncertainty is propagated and metric-wall
intersections fail closed, this boundary must be described only as strict
canonical-partition membership. It must not be advertised as proof of
representation-invariant physical equivalence or as a conventional CSP
similarity score.
