# CrystalAtlas

Crystal structure generation and prediction with exact symmetry.

CrystalAtlas generates crystals in three modes:

- **CSP**: given a formula, predict its structures.
- **Species DNG**: given a set of elements, generate new compositions and structures.
- **DNG**: generate new materials from scratch.

Every sample is built from a space group and Wyckoff positions that are checked exactly,
so the symmetry and stoichiometry of a generated crystal are correct by construction.
A learned model chooses the symmetry, and a flow-matching model places the atoms.

## Status

This is an early research release (`0.1.0a1`).

- The code is complete and tested: the exact symmetry engine for all 230 space groups,
  the learned selector and geometry model, training, and the CSP evaluation protocol.
- No trained model is included yet. The current model is a pilot trained on 404 MP-20
  structures. On its 50 held-out test formulas it finds the right structure for 13
  within 20 attempts.
- There are no full MP-20 benchmark numbers yet, so no claim is made that it beats
  other generators.

The research record, with every experiment registered before it was run, is in
[`artifacts/research_2026-10-30/R1_LOG.md`](artifacts/research_2026-10-30/R1_LOG.md).

## How it works

1. A composition model proposes a formula (DNG only).
2. A cover model chooses the space group and the Wyckoff positions. Exact masks allow
   only choices that can complete the formula.
3. A periodic Transformer, trained by flow matching, moves the lattice and the free
   atomic coordinates from random noise to a crystal.
4. The result is decoded exactly to a full periodic structure. Failures are reported,
   never silently repaired or filtered.

The equations and proofs are in [`theory/`](theory/).

## Install

```bash
pip install -e '.[ml,crystallography]'
```

Python 3.11 or newer. The core needs only NumPy; `ml` adds PyTorch and
`crystallography` adds spglib.

## Quick start

```bash
crystalatlas status
crystalatlas formula Si2O4
crystalatlas reference-generate --mode csp --formula LiFePO4 --samples 10 --seed 9 --output out/lifepo4
```

`reference-generate` uses an untrained reference model. It checks the pipeline end to
end, but its structures are not predictions.

Training, datasets and evaluation are described in [`docs/`](docs/), starting with
[`TRAINING_AND_DIAGNOSTICS.md`](docs/TRAINING_AND_DIAGNOSTICS.md) and
[`DATASET_PIPELINE.md`](docs/DATASET_PIPELINE.md).

## Limits

Ordered, fully occupied periodic crystals only: no disorder, magnetism or defects. A
valid structure is not necessarily stable; stability needs separate energy calculations.

## Licence

Free for academic and other noncommercial use under the
[PolyForm Noncommercial License 1.0.0](LICENSE). That covers universities, public research
institutes, charities and government bodies, whatever their funding.

Commercial use needs a paid licence: write to paramvir.chem@gmail.com.

The bundled spglib tables keep their own BSD-3-Clause licence
(`src/crystalatlas/_data/spglib_v2_7_0/COPYING`).
