# Private Git repository scope

Initial snapshot: 28 September 2026. The working directory had no `.git`
directory or recoverable commit history. The initial commit therefore records
the current state, not a reconstruction of the earlier development sequence.
Dated scientific evidence, including failed runs, remains in `artifacts/`.

Tracked: source, tests, theory, documentation, workflows, acquisition scripts,
verification reports/logs, and retained versioned package/evidence archives.
The latest verified candidate is documented in
`artifacts/gpu_rtx5060ti_2026-09-26_v13/README.md`; its wheel and reports are
included. Production standardization is v6, learned representation v2, and
training objective v4.

Excluded by `.gitignore`: caches, transient build/runtime directories,
credentials, raw benchmark/pilot copies under `data/benchmarks/` and
`data/pilots/`, and model checkpoints (`*.pt`, `*.pth`, `*.safetensors`).
Raw data redistribution/authorization review is unfinished. Dataset acquisition
and provenance instructions remain in `docs/MP20_PILOT.md` and related source
reviews; these do not authorize training by themselves. Checkpoint paths and
hashes remain in the GPU reports.

Excluded files are not deleted. This repository is a version-controlled code
and evidence snapshot, **not a complete backup of all local or remote data**.
Large checkpoints require a separately chosen artifact-backup/LFS policy.
No project-wide open-source license is granted by creating this private repo;
existing third-party attribution and license notices remain in place.
