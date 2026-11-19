# Primitive ranking guard: independent review

Review date: 20 September 2026. This review changes no production equations,
ranking grids, tolerances, rejection rules, or accepted training records.

## Observed scope

The frozen MP-20 pilot contains **512 extracted training-source records**. The
first preprocessing preflight scheduled only its first **32** records and
attempted **four**: three ranking-guard rejections, followed by one fatal
`InvariantViolation` (`continuous free-origin section is not canonical`). The
remaining **28 scheduled records were not run**; another 480 pilot records were
outside that preflight. There were zero successful target compilations in this
run. Neither a four-record nor a 512-record successful crystallographic audit
has been established.

Evidence: `artifacts/mp20_pilot_2026-09-14/preprocessing32_r1.json` and
`artifacts/mp20_pilot_2026-09-14/ranking_guard_observations.json`. The extraction
checks described in `docs/MP20_PILOT.md` concern transport, selection, and byte
identity, not successful canonical target construction. The preprocessing
runner correctly stops on an unexpected internal invariant failure instead of
silently counting it as a rejected material.

## Equation and guard interpretation

For a row-vector lattice matrix \(L\), the metric is \(G=LL^T\). An integer
lattice vector \(v\) has squared length \(v^TGv\); a proper unimodular basis
\(B\) has metric \(BGB^T\). These equations and the code's determinant-positive
basis restriction are consistent with retaining handedness. The supplied basis
bounds the minimax objective, giving a finite vector search that includes the
global minimizer when the numerical enclosure and search limits hold.

The current grid is \(q=10^{-\min(d,10)}\), hence \(10^{-10}\) Å² for metric
entries under the default `canonical_decimals=12`. Coordinates use a separate
dimensionless grid with the same numerical spacing. For a scalar value \(x\),
the nearest-bin integer is \(k=\lfloor x/q+1/2\rfloor\). The guard rejects when

\[
\tfrac12-|x/q-k|\leq e/q,
\]

where \(e\) is its floating-point uncertainty allowance. This is the correct
intersection condition for a symmetric uncertainty interval and a nearest-bin
wall, subject to the numerical validity of the enclosure itself. The three
recorded failures satisfy that inequality; none is a spurious branch caused by
an inverted comparison. The allowance propagates a metric perturbation using
\(|v^T\Delta Gv|\leq\|\Delta G\|_2\|v\|_2^2\) and
\(\|B\Delta GB^T\|_2\leq\|B\|_2^2\|\Delta G\|_2\), which are valid bounds.

However, the factor `256 * eps * max(norm(G, 2), 1)` is a conservative numerical
allowance, **not a complete directed-rounding interval-arithmetic proof** for
the entire spglib-to-ranking pipeline. This review does not independently
certify all upstream errors or every floating-point operation. Likewise,
bin-wall proximity does **not** establish physically ambiguous crystal data;
it establishes that the present numerical ranking contract cannot certify its
decision at that point. The norm and the error allowance depend on the supplied
basis; equivalent presentations need not have identical rejection behavior.

## The three retained failures

| Source ID | First failure under the unchanged guard | Distance to wall, in bins | Allowance, in bins |
| --- | --- | ---: | ---: |
| `mp-697047` | Vector `(1,2,0)`, squared length 52.35953968133032 Å² | 0.1968384 | 0.2224297 |
| `mp-634945` | Basis off-diagonal −13.94024688525549 Å² | 0.0549011 | 0.1029177 |
| `mp-1226790` | Basis off-diagonals 7.682274910748781 and 15.434979155246179 Å² | 0.0121918 / 0.0382080 | 0.2294255 |

The first failure occurs before basis ranking. The other two occur on the first
basis tested, before any best-key incumbent exists. The diagnostic log's
`vector_if_present` is a stale vector-enumeration loop variable for these latter
two failures; it is **not** the vector responsible for the off-diagonal error.
Use `basis_if_present` and `caller_line` to interpret them.

A further read-only enumeration of the captured metrics evaluated ordinary
nearest-point scores for diagnosis only; it bypassed no production acceptance
gate and produced no training targets. It found 7, 6, and 6 unoriented vectors,
respectively. Every basis attaining the smallest ordinary point-estimate Gram
key had at least one off-diagonal interval crossing a wall: four such bases for
`mp-697047`, 24 for `mp-634945`, and four for `mp-1226790`.

For `mp-697047`, the long vectors `(1,2,0)` and `(2,1,0)` cannot improve the
best point-estimate primary key. A proper competing basis has squared lengths
approximately `(14.23942302, 23.88069363, 78.26050917)` Å², versus the ambiguous
vectors' approximately 52.35953968 Å². Nevertheless, its best point-estimate
off-diagonal is approximately −7.11971151187 Å² and also crosses a wall under
the present allowance. Removing irrelevant long-vector failures therefore does
not by itself establish acceptance of this record.

These diagnostic minima are **not interval-certified winning representatives**.
They neither prove that a specific ambiguous off-diagonal is unavoidable under
every possible improved certification method nor justify choosing a representative
by ordinary rounding. They do show that no improved acceptance rate can presently
be promised from simple lazy rejection or irrelevant-vector pruning.

## Bounded next action

First fix and regress the independent free-origin invariant failure, then rerun
the same scheduled 32 IDs, preserving this original evidence and the no-refill
selection rule. Do not infer a dataset-wide rejection rate from the three
ranking failures preceding the internal abort.

A separate, versioned follow-up may prototype **certified interval pruning**
without changing the objective or grid:

1. Enclose each scalar ranking value by outward-rounded bounds \([a,b]\),
   including division and floor-boundary arithmetic. Its possible bins lie in
   \([\lfloor a/q+1/2\rfloor,\lfloor b/q+1/2\rfloor]\). A singleton interval
   certifies that component; a multi-bin interval does not authorize selecting
   its midpoint.
2. Obtain a certified feasible incumbent. Propagate interval bounds through
   sorted diagonals, the maximum diagonal, and the off-diagonal lexicographic
   key. Ordering ambiguities require considering every feasible ordering, not
   sorting only the midpoint estimates.
3. Prune a candidate only if its entire best-possible key is strictly worse
   than the incumbent's certified key. Equality, overlapping intervals, or a
   potentially better prefix cannot be pruned. A vector can be discarded only
   after excluding every proper unimodular completion that could compete.
4. Preserve all surviving metric ties for the origin-quotiented colored-motif
   comparison. Any unresolved candidate capable of changing the winning output
   must still reject. Preserve original physical coordinates; ranking bins are
   not permission to round the returned structure.
5. Test against exhaustive small-lattice enumeration, exact bin-wall and
   near-wall cases, proper basis changes and rotations, motif translations,
   mirrors, and adversarial overlapping intervals. Retain resource-limit
   failures explicitly. Report measured yield separately from correctness.

Coarsening the grid changes the canonicalization/identity contract and can
create new unresolved ties. It is not a bug fix or an interchangeable shortcut
to interval pruning. Higher-precision arithmetic alone also cannot be assumed
to erase uncertainty already introduced by upstream floating-point
standardization. Neither change is recommended without a separately stated
error model, versioned policy, and equivalence/round-trip evidence.

## Follow-up, 26 September 2026

The frozen 512-record v5 audit is now complete. All 169 numerical rejections
were traced without bypassing a guard: 115 vector norms, 36 basis off-diagonals,
18 initial metric-resolution failures. Exact-rational checks of the stored
binary64 values reproduce every recorded bin-wall intersection. This does not
turn the heuristic error allowance into a full uncertainty certificate.

Of the first offending vectors, 23 have coordinate gcd greater than one. Such
a vector v=g w cannot appear in a unimodular basis because g divides every
integer determinant containing it. Version v6 excludes these inadmissible rows
before numerical ranking, with synthetic shear/rotation regression tests and
unchanged quantum/error predicates. Later guards can still reject those records;
23 first failures does not establish 23 recovered targets. General interval
pruning and the remaining resolution/representation limitations are still open.

An independent input-wrapper coordinate-snapping bug was also found and fixed
in v6. See `artifacts/mp20_pilot_2026-09-25/` for retained before/after evidence;
none of this modifies the historical observations above.
