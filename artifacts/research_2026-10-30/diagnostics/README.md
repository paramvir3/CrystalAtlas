# Diagnostics of the 2026-10-30 research program

Scripts that produced evidence cited in `../R1_LOG.md`. They were run on tcm-pc38 from
`~/crystalatlas-gpu-20261025/r3/` with `PYTHONPATH` set to a frozen code snapshot
(`r3/v7_eval_*_code/src` and `/scripts`) and the working directory `repo3`. They read
private attributes of the generator and are research tools, not package API.

| file | log section |
|---|---|
| `compare_csp_samplers.py` | Fused geometry sampler `mixed-v1` |
| `repro_case_sampler_gap.py` | Fused geometry sampler `mixed-v1` |
| `diagnose_orbit_collapse.py`, `analyse_traj.py` | Orbit collapse: measured mechanism |
| `matrix_exp_check.py` | Numerical finding: `torch.matrix_exp` for one matrix |
| `ab_periodic/` | Registered A/B: periodic prior on the cover selector |
| `ab_carrier/` | Registered A/B: pair-carrier smoothing on the geometry field |
| `ab_selector/` | Registered A/B: selector inputs and regularisation |
| `pc10/` | Move to tcm-pc10 and cross-host check |
| `ab_carrier/verdict_2026-11-15.txt`, `ab_selector/verdict_2026-11-15.txt` | Results of the carrier and selector A/Bs (read 2026-11-15) |
| `m1_contacts.py`, `ab_carrier/m1_targets.json`, `ab_carrier/m1_analysis.py`, `pc10/m1_queue.sh` | Measurement M1: contact pairs of the carrier A/B |
| `ab_carrier/m1_verdict_2026-11-15.txt`, `ab_carrier/m1_checks/` | Result of measurement M1 (read 2026-11-15) |
| `ab_carrier/m1b_analysis.py` | Measurement M1b: separations of the collapses the smoothing removed |
| `ab_carrier/m1b_verdict_2026-11-15.txt`, `ab_carrier/m1b_checks/` | Result of measurement M1b (read 2026-11-15) |
| `m2_starts.py`, `ab_carrier/m2_targets.json`, `ab_carrier/m2_analysis.py`, `pc10/m2_queue.sh` | Measurement M2: starting pair separations and validity |
| `ab_carrier/m2_verdict_2026-11-15.txt`, `ab_carrier/m2_checks/` | Result of measurement M2 (read 2026-11-15) |
| `m3_redraw.py`, `ab_carrier/m3_targets.json`, `ab_carrier/m3_analysis.py`, `pc10/m3_queue.sh` | Measurement M3: paired redraw of near-coincident starts |
| `ab_carrier/m3_verdict_2026-11-15.txt`, `ab_carrier/m3_checks/` | Result of measurement M3 (read 2026-11-15) |
| `m4_rejection.py`, `ab_carrier/m4_analysis.py`, `pc10/m4_queue.sh` | Measurement M4: opt-in start-rejection sampler (option `start_min_separation` in `src/crystalatlas/learned.py`, tests `tests/test_start_rejection.py`) |
| `ab_carrier/m4_verdict_2026-11-16.txt`, `ab_carrier/m4_checks/` | Result of measurement M4 (read 2026-11-16) |

The result files stay on the host under `r3/`; the log quotes the numbers used. The two
`verdict_2026-11-15.txt` files are the printed outputs of the repository copies of
`ab_carrier/analysis.py` (run in a copy of `r3/ab_carrier`) and `ab_selector/analysis.py`
(run in a copy of `r3/` as `analysis.py ab_periodic ab_selector LEVER 0.2` for each lever).
