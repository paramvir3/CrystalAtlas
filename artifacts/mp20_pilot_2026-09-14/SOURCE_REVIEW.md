# Frozen MP-20 source review — 2026-09-14

## Decision and scope

Use the historical CDVAE MP-20 snapshot for the bounded CrystalAtlas research
pilot. Record Materials Project as the upstream computed-data source, CDVAE as
the benchmark curator, and CC-BY-4.0 as the reported upstream data licence.
This engineering review is not a legal opinion, a provider signature, or a
licence grant for unrelated Materials Project batches. The byte-acquisition and
CSV-extraction receipts deliberately do **not** authorize training by themselves:
a separately bound dataset source declaration is required by DatasetCompiler.

The public benchmark uses DFT-relaxed Materials Project results with ICSD
lineage, as disclosed in the archived README. It is not a direct ICSD export.
Do not describe these data as current Materials Project, a new experimental
measurement set, or as free of third-party lineage. The original MP database
release/API query is not supplied; its value remains **unknown**. Reproducibility
is instead bound to the exact distributed CSV bytes and repository revision.

## Evidence and immutable pins

- CDVAE revision: `f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea`, the revision linked
  in [CrystaLLM's benchmark instructions](https://github.com/lantunes/CrystaLLM/blob/main/BENCHMARKING.md).
- The GitHub contents API at this revision gives the same four file blob IDs as
  the initial CDVAE commit `73874c48a46b74fedaa161fdac512b0a75e2d336`, dated
  2021-12-09. This historical release is not a modern GNoME/MPContribs export.
- [Archived MP staff clarification](https://matsci.org/t/is-it-ok-to-post-snapshots-of-materials-project-structures-on-e-g-figshare-i-e-for-manuscript-reproducibility/45437/2):
  Matthew Horton, 2022-11-14, confirms reproducibility snapshots and CC-BY-4.0
  attribution. Read the provider's second post, not just the question's quotation.
  The retrieved JSON is `mp-provider-clarification-45437.json`, SHA-256
  `67da73530a0444b2135dcc8aaae869349644a4567da45411cfbbcbb82acbd4b6`.
- The current `materialsproject.org/about/terms` endpoint returned HTTP403.
  The archived clarification must not be presented as a retrieved copy of the
  current terms. The historical provider clarification and the published
  benchmark lineage are the evidence used here.
- [CC-BY-4.0 legal text](https://creativecommons.org/licenses/by/4.0/legalcode.txt)
  is separately archived. Retain attribution, licence/disclaimer notices and
  source links; identify transformations; do not imply provider endorsement.
- The CDVAE software licence is **not** used as a substitute for a data licence.

## Required attribution and transformation record

Upstream: Jain et al., *Commentary: The Materials Project: A materials genome
approach to accelerating materials innovation*, APL Materials1,011002 (2013),
[DOI](https://doi.org/10.1063/1.4812323).

Benchmark curation: Xie et al., *Crystal Diffusion Variational Autoencoder for
Periodic Material Generation*, [paper](https://arxiv.org/abs/2110.06197),
[frozen MP-20 description](https://github.com/txie-93/cdvae/blob/f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea/data/mp_20/README.md).

CrystalAtlas transformations: strict CSV decoding, transport-index labeling,
SHA-256-ranked training-only subset selection, and lossless UTF-8 extraction of
the CIF field. Crystallographic canonicalization is a later, separately audited
transformation. Every rejected input must remain in the denominator; no failed
input is replaced. Model-generated failures are not dataset-preprocessing rows.

This is a historical reproducibility snapshot. For current materials information
or new database-wide studies, obtain current data and terms from Materials
Project directly. No claim of current energetic stability is inferred from the
historical benchmark metadata.
