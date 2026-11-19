# 0. Scope and scientific contract

## 0.1 What the theory is trying to solve

The package has three first-class tasks:

1. **composition-conditioned CSP**, which receives a reduced elemental
   composition \(c\) and returns a calibrated distribution over possible
   periodic crystal structures; and
2. **chemical-system-conditioned de-novo generation (S-DNG)**, which receives
   an exact unordered element set \(S\), generates its reduced stoichiometry,
   and then samples from the same conditional crystal law; and
3. **unconditional de-novo generation (DNG)**, which first generates \(S\),
   then its canonical reduced stoichiometry, and finally the same conditional
   crystal law.

All three tasks should return distributions rather than single deterministic
answers.  The representation should enforce:

- exact reduced elemental composition and exact advertised conventional-cover
  counts \(N_Z=\nu c_Z\);
- positive-volume periodic lattices;
- exact crystallographic orbit constraints;
- positive probability for exact selected-group symmetry-containment sets;
- exchangeability of physically identical orbit instances; and
- a normalized training and sampling law.

It should not confuse these guarantees with energetic stability.

## 0.2 Declared physical class and computational support

Version 1 concerns static, ordered, fully occupied, nonmagnetic,
three-dimensional, commensurate periodic crystals.  A physical object is
always reduced to its full primitive translation lattice before equivalence is
assessed.

The finite bounds below are **raw-cover resource bounds**, not invariants of a
physical crystal.  Fix \(N_{\max}^{\rm raw}\) and
\(\nu_{\max}\in\mathbb N\) with \(\nu_{\max}\ge1\).  A selected
Hall/Wyckoff cover may expand at most \(N_{\max}^{\rm raw}\) sites in its pinned
conventional cell.  For each supported selected setting, also fix a metric
bound.  If \(G_0\) is that cover's reference invariant metric, retained raw
cells obey

\[
\left\|
\log\!\left(G_0^{-1/2}GG_0^{-1/2}\right)
\right\|_{\mathrm{F}}
< \mathcal B_g < \infty.
\]

This bounds relative cell scale and condition number in the pinned raw-cover
gauge.  It is required for finite computation and certified periodic-image
enumeration; it is neither a law of crystallography nor a function on the
physical quotient.

Let \(\mathscr X_c\) denote the full physical class above.  Let
\(\mathcal X_c^{\rm target}\subset\mathscr X_c\) contain those physical
crystals for which at least one presentation in the declared
actual-stabilizer pinned Hall/Wyckoff convention satisfies the raw atom and
metric bounds and has conventional formula-unit multiplier
\(\nu(C)=\gcd_Z N_Z^{\rm conv}\le\nu_{\max}\).  Let
\(\mathcal X_c^{\rm model}\) be the union of
all decoded, fully primitive-reduced outputs of every bounded raw cover.  The
target domain is contained in the model support, but the inclusion can be
strict: a lower-symmetry raw cover may decode to a higher-symmetry crystal
whose own pinned conventional presentation lies outside the raw bound.  Such
an output remains a valid generated crystal and is reported as outside the
declared target domain; it is not silently rejected.

Fix a finite chemical-element vocabulary \(\mathcal E\).  A reduced
composition is

\[
c=(c_Z)_{Z\in\mathcal E},
\qquad c_Z\in\mathbb N_0,
\qquad c\ne0,
\qquad \gcd\{c_Z:c_Z>0\}=1.
\tag{0.1}
\]

The version-1 composition-conditioned input domain is

\[
\mathcal C_{\rm CSP}
=
\left\{c\text{ satisfying (0.1)}:
\sum_Zc_Z\le N_{\max}^{\rm raw}\right\}.
\]

The \(P1\) witness below makes the arithmetic selector nonempty on this whole
domain.  A supplied formula outside \(\mathcal C_{\rm CSP}\) is out of scope,
not a failed stochastic sample.

For DNG, fix a finite allowed element alphabet
\(\mathcal E_{\rm DNG}\subseteq\mathcal E\) with a total order and a maximum
number \(K_{\max}\) of distinct elements.  The declared canonical composition
support is

\[
\mathcal C_{\rm DNG}
=
\left\{
c:
\begin{array}{l}
\operatorname{supp}(c)\subseteq\mathcal E_{\rm DNG},\\
1\le |\operatorname{supp}(c)|\le K_{\max},\\
\sum_Zc_Z\le N_{\max}^{\rm raw},\\
\gcd\{c_Z:c_Z>0\}=1
\end{array}
\right\}.
\tag{0.1a}
\]

The valid exact chemical systems and their stoichiometry fibres are

\[
\mathfrak S_{\rm DNG}
=
\left\{S\subseteq\mathcal E_{\rm DNG}:
1\le |S|\le\min(K_{\max},N_{\max}^{\rm raw})\right\},
\qquad
\mathcal C(S)
=
\{c\in\mathcal C_{\rm DNG}:\operatorname{supp}(c)=S\}.
\tag{0.1d}
\]

Every \(\mathcal C(S)\) is nonempty because the all-one coefficient vector is
reduced and obeys the raw-count bound.  In version 1, a supplied bare species
list means **all and only** those elements must occur.  An “allowed palette”
with optional elements is a different conditional event and may not be
silently reported as exact-system generation.

This set is finite.  Every member has nonempty downstream arithmetic support
without an additional learned or model-dependent feasibility test: take
\(\nu=1\), the pinned \(P1\) setting, and \(c_Z\) repeatable general-position
orbits of multiplicity one for every present element \(Z\).  Distinct orbit
representatives exist on the three-torus, and their collision set has measure
zero.  This is only a support witness, not a claim that \(P1\) is the likely or
maximal symmetry.

In particular, \(\mathcal C_{\rm DNG}\subseteq\mathcal C_{\rm CSP}\); the DNG
alphabet and cardinality bound may be narrower than those accepted by CSP.
The ordered representation

\[
s(c)=(K,Z_1,\ldots,Z_K,c_{Z_1},\ldots,c_{Z_K},\mathrm{EOS}),
\qquad Z_1<\cdots<Z_K,
\tag{0.1b}
\]

is a support-first flat sequence of typed tokens and is unique.  Its complete
prefix \(a(S)=(K,Z_1,\ldots,Z_K)\) identifies the exact chemical system; the
remaining coefficient suffix identifies the reduced stoichiometry.  Element
ordering removes sequence permutations and the gcd rule removes formula-scaling
duplicates.  These are representation guarantees, not chemical plausibility,
charge-state, stability, or synthesizability guarantees.  Universal hard
oxidation-state masks are excluded: formal oxidation states are nonunique and
need not describe metals, intermetallics, or mixed-valence crystals.

Across this finite composition set, use the coproduct Borel spaces

\[
\mathcal X_{\rm DNG}^{\rm target}
=
\bigsqcup_{c\in\mathcal C_{\rm DNG}}\mathcal X_c^{\rm target},
\qquad
\mathcal X_{\rm DNG}^{\rm model}
=
\bigsqcup_{c\in\mathcal C_{\rm DNG}}\mathcal X_c^{\rm model}.
\tag{0.1c}
\]

The outer components are disjoint because complete primitive reduction
preserves and uniquely determines reduced composition.  Raw symmetry covers
may still overlap within one component.

The pinned conventional cell contains

\[
N_Z=\nu c_Z,
\qquad
1\leq\nu\leq\nu_{\max},
\qquad
\sum_ZN_Z\leq N_{\max}^{\rm raw}.
\tag{0.2}
\]

For a selected setting \(g\), let \(I_g\) be its centering index: the number of
known conventional-cell lattice points.  Removing those declared centering
copies gives

\[
N_Z^{(g)}=\frac{\nu c_Z}{I_g},
\tag{0.3}
\]

which must be integral.  A particular decoded motif can possess additional
species-preserving translations.  If their full conventional-cell translation
index is \(J(C)\ge I_g\), complete primitive reduction instead gives

\[
N_Z^{\rm prim}=\frac{\nu c_Z}{J(C)}.
\tag{0.3a}
\]

The reduced composition remains \(c\), but the actual primitive multiplier may be
smaller than \(\nu/I_g\).  If a benchmark supplies counts in the same pinned
conventional setting, those counts, not merely \(c\), are the condition and
\(\nu\) is not inferred.  Conditioning on an exact primitive-cell atom count
alone is **outside version 1**, because the full translation index \(J(C)\) is
known only after geometry is decoded.  A future extension must encode that
constraint structurally and renormalize its support; post-filtering is not a
valid implementation.  Conventional Wyckoff multiplicities and primitive-cell
counts must never be inserted into the same constraint equation.

Version 1 explicitly excludes:

- partial occupancy and occupational disorder;
- magnetic moments and magnetic space groups;
- incommensurate or modulated phases;
- quasicrystals;
- finite-temperature displacement distributions;
- defects not represented by a bounded ordered supercell; and
- claims about thermodynamic stability or synthesis pathways.

Physical equivalence uses proper Cartesian rotations and right-handed
primitive bases.  Enantiomers are therefore distinct outputs.

These exclusions are mathematical boundaries, not statements that the omitted
physics is unimportant.

## 0.3 Evidence labels

Every future claim must carry one label:

- **P — proved:** follows analytically from stated assumptions.
- **O — oracle verified:** exhaustively checked on a finite bounded problem.
- **N — numerically tested:** checked within a declared floating tolerance.
- **E — empirical:** supported only by data, training runs, or hardware tests.

A numerical test is not a proof.  A proof does not establish that code follows
the equations.  A benchmark result does not establish a universal law.

## 0.4 Five levels that must remain separate

\[
\boxed{
\text{formula legal}
\not\Rightarrow
\text{geometrically legal}
\not\Rightarrow
\text{correct maximal symmetry}
\not\Rightarrow
\text{chemically plausible}
\not\Rightarrow
\text{stable or synthesizable}.
}
\tag{0.4}
\]

The discrete grammar will prove the first property.  The exact decoder and
global verifier can prove the second and determine the third.  The last two
require physical evidence that is not part of this theory.

## 0.5 Falsifiable central hypothesis

The mathematical construction itself makes no accuracy prediction.  The later
empirical hypothesis is:

> A normalized exact-reduced-formula distribution over overlapping crystallographic
> symmetry-containment covers, preceded in the de-novo modes by a support-first
> normalized
> chemical-system and stoichiometry law and coupled to a shared exchangeable
> intrinsic flow with cutoff-complete periodic-image enumeration, improves
> fixed-budget recovery and DFT-verified discovery yield relative to the
> strongest reproducible task-matched generative and traditional-search
> baselines.

This hypothesis must be rejected if matched experiments do not support it.

## 0.6 Stop rule

No implementation begins until:

1. the physical quotient and cell convention are frozen;
2. the exact catalogue source and symmetry tolerance policy are frozen;
3. every theorem assumption is mapped to a verifier;
4. the prior-art boundary has been independently reviewed; and
5. the proof-obligation document contains a finite oracle for every discrete
   algorithm and a property test for every numerical identity.

The selector weights in version 1 are latent **cover** probabilities.  Because
different covers can decode to the same physical crystal, they are not claimed
to equal probabilities of actual maximal space groups or physical scaffolds.
The DNG weights \(p_\omega(S)\) and \(p_\omega(c\mid S)\) are
probabilities under the declared learned or reweighted database composition
law, not thermodynamic prevalence.
