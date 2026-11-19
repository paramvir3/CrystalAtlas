# MP-20 pilot preparation

Work in progress. The frozen CDVAE benchmark snapshot is stored separately at
`data/benchmarks/mp20-cdvae-f857f598`; no upstream code is executed. The repository
revision is `f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea`, matching the revision linked
by CrystaLLM's benchmarking instructions. Dataset byte identity is checked against
published Git blob hashes and recorded independently with SHA-256.

The first pilot selects 512 rows from the published **training** split by a fixed
SHA-256 ordering of seed 20260913 and material ID, before parsing or target
acceptance. Preprocessing failures must be retained and never refilled. Published
validation/test rows are not training inputs. New disjoint split views are separate
from published benchmark splits and cannot be presented as the same protocol.

Rights evidence in this directory is a historical provider clarification, not a
retrieved copy of the current Materials Project terms page (which returned HTTP
403). The staff response by Matthew Horton dated 2022-11-14 explicitly identifies
CC-BY-4.0 and permits scientific-reproducibility snapshots with attribution and
historical-version labeling. It must not be mislabeled as a new permission grant
from the CDVAE software's MIT licence. The CDVAE description discloses ICSD lineage
and subsequent DFT relaxation. These are distributed MP computed results, not
directly acquired ICSD database files. The precise original MP database release
is not disclosed by this CSV snapshot; do not invent it or call this current MP.

Training has not started. A completed preparation receipt and exact-decoder audit
will be required before creating the training dataset store.
