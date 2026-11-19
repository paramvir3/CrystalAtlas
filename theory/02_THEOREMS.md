# 2. Theorems and proofs

All results are conditional on the domain and definitions in
[00_SCOPE.md](00_SCOPE.md) and [01_FOUNDATIONS.md](01_FOUNDATIONS.md).  A theorem
is not an implementation claim until its proof obligations pass.
The crystallographic results are first proved for each supplied reduced
composition.  Theorem 2.2c and Theorem 2.5 then lift those conditional kernels
to a normalized first-class de-novo law through an outer canonical-composition
mixture.

## Theorem 2.1 — Bounded-domain overlapping-cover coverage

Assume:

1. the catalogue contains every pinned space-group realization and every exact
   Wyckoff position needed within the declared bounds;
2. every input cell is reduced to its full primitive translation lattice;
3. conventional decodes use the exact centering map and then complete primitive
   reduction;
4. the continuous origin subtorus is quotiented exactly;
5. for every supported arithmetic cover, the singular set \(\Sigma_\alpha\) is
   product-volume null; and
6. the full primitive decoder is measurable and preserves reduced composition.

Then

\[
\mathcal X_c^{\rm target}
\subseteq
\mathcal X_c^{\rm model}
=
\bigcup_{\alpha\in\mathcal A_{\rm arith}(c)}
\bar D_\alpha(\widehat Q_\alpha\setminus\Sigma_\alpha).
\tag{2.1}
\]

### Proof

Take \(C\in\mathcal X_c^{\rm target}\).  Its complete coloured stabilizer
\(\overline G_C\) is defined by equation (1.7).  Transform it to the pinned Hall
realization.  Its coloured sites split into group orbits.  Each orbit has one
species, exact Wyckoff position, multiplicity, and intrinsic parameter.
Counting those orbits gives \(h_{wZ}\), while the canonically represented metric
belongs to \(\mathcal K_g^{\mathcal B}\).  Therefore these
data define an arithmetic raw cover \(\alpha\) and a regular point \(q\) with
\(\bar D_\alpha(q)\sim C\).

The equality defining \(\mathcal X_c^{\rm model}\) is the support definition
from section 1.9.  Every regular raw-cover point expands to the declared
conventional formula, contains the selected infinite group, and maps under
complete primitive reduction to a member of the broad physical class
\(\mathscr X_c\) with reduced composition \(c\).  It need not belong to
\(\mathcal X_c^{\rm target}\), because reclassification can change the pinned
conventional presentation.  The union is generally overlapping: two covers
may represent the same physical crystal, and a nominal cover may decode
generically to a larger stabilizer. ∎

### Important limits

- A rational-grid search for special positions does not prove catalogue
  completeness.
- A standardized cell is not automatically unique under its Euclidean
  normalizer.
- \(P1\) gives redundant coverage only when its atom and metric bounds admit the
  corresponding primitive presentation; a one-atom nominal P1 cover decodes
  to at least inversion symmetry.
- Cover weights are not identifiable physical space-group probabilities.
- The theorem proves coverage, not a unique representation.

Taking the finite coproduct over \(c\in\mathcal C_{\rm DNG}\) gives the DNG
representability statement

\[
\mathcal X_{\rm DNG}^{\rm target}
\subseteq
\mathcal X_{\rm DNG}^{\rm model}.
\tag{2.1a}
\]

This is set coverage, not positive probability for any exact crystal point.
Positive composition mass, full branch support, and decoder continuity or a
positive-measure preimage-neighbourhood condition are separate probability-
support questions.

## Theorem 2.2a — Arithmetic suffix reachability is sound and complete

Fix \((g,\nu,c)\), put \(\mathbf N=\nu c\), and use equations (1.16)--(1.17).
Then \(B_{g,r}(\rho)=1\) exactly when at least one capacity-respecting suffix
consumes the residual \(\rho\).

### Proof

Backward induction.  At \(r=R_g+1\), equation (1.16) accepts exactly the zero
residual.  Assuming the claim at \(r+1\), equation (1.17) is true exactly when
one allowed row action leaves a residual accepted by the suffix.  Appending
that witness proves soundness; the first row of any legal suffix proves
completeness. ∎

The dense residual state count is

\[
S_N=\prod_{Z:c_Z>0}(N_Z+1),
\tag{2.2}
\]

and the direct arithmetic recursion costs

\[
O\!\left(S_N\sum_r|\mathcal H_{gr}|\right).
\tag{2.3}
\]

This is pseudo-polynomial and exponential in the number of present elements.
It proves formula feasibility only.

## Theorem 2.2b — Arithmetic completion decoding is normalized and dead-end free

Assume \(\mathcal V(c)\ne\varnothing\), the arithmetic table \(B\) is exact, and
every retained global or row action has a finite logit.  Then:

1. every allowed prefix has at least one arithmetic continuation;
2. every sampled table satisfies the exact conventional-cell formula and row
   capacities;
3. the complete selector (1.20) is normalized over all arithmetic raw covers;
   and
4. every arithmetic raw cover has positive probability.

### Proof

Theorem 2.2a makes every retained suffix action set nonempty.  The definitions
of \(\mathcal V(c)\) and \(\mathcal G(c,\nu)\) make the two global softmax supports
nonempty as well.  Every action retained by \(\mathcal A^{\rm arith}\) leaves a
residual accepted by the suffix table, so induction reaches zero residual
without a dead end.  Iterated total probability normalizes the product of
global and row conditionals.  Every arithmetic path is retained at every
prefix; finite logits make its probability positive. ∎

## Theorem 2.2c — Canonical de-novo composition decoding is normalized and dead-end free

Assume \(\mathcal C_{\rm DNG}\ne\varnothing\), the live-prefix set
\(\mathcal P_{\rm comp}\) in (1.20a) is exact, and every allowed token has a
finite logit.  Then equation (1.20c):

1. defines a normalized probability mass function on exactly
   \(\mathcal C_{\rm DNG}\);
2. has no nonterminal dead-end prefix;
3. produces each formula in one unique ordered serialization;
4. gives positive probability to every formula in the declared support; and
5. factors as (1.20d), with \(p_\omega(S)\) a normalized positive pmf over
   \(\mathfrak S_{\rm DNG}\) and \(p_\omega(c\mid S)\) a normalized
   positive pmf on exactly \(\mathcal C(S)\) for every \(S\).

### Proof

Every nonterminal \(u\in\mathcal P_{\rm comp}^{\circ}\) is, by definition, a
proper prefix of at least one accepted leaf.  Hence
\(\Omega_{\rm comp}(u)\ne\varnothing\), and every allowed child remains live.
Induction reaches exactly one EOS-terminated leaf in the finite prefix tree,
where generation stops.  Iterated total probability makes the sum of all leaf
probabilities one.  Ordering by the frozen element order and the gcd-one
condition in the leaf set make the serialization injective; every \(c\) has
its declared serialization, so it is also surjective onto
\(\mathcal C_{\rm DNG}\).  Finite logits make every edge, and therefore every
leaf, have positive probability.

For fixed \(S\), its support-first string \(a(S)\) is a complete
positive-probability prefix.  The descendant leaves below this prefix are
exactly \(\{s(c):c\in\mathcal C(S)\}\).  Applying the same subtree induction
from those leaves to \(a(S)\) proves that the continuation probabilities sum
to one, which is precisely the conditional factor in (1.20d).  The support
subtrees are disjoint and partition all accepted leaves, so their prefix
probabilities sum to one; finite logits make each one positive.

The downstream support condition is constructive rather than learned.  For
any \(c\in\mathcal C_{\rm DNG}\), choose \(\nu=1\), \(g=P1\), and
\(h_{1a,Z}=c_Z\).  The repeatable \(P1\) general position has multiplicity one,
so this table satisfies every element-count equation and the raw atom bound.
Distinct representatives exist; for two or more advertised sites, the
collision set is a finite union of pairwise diagonal subtori of codimension
three and is therefore proper and Haar-null.  Finally, any hidden
species-preserving translation subgroup of index \(J\) would act freely and
force \(J\mid c_Z\) for every \(Z\).  Since \(\gcd(c)=1\), \(J=1\).  Thus the
witness preserves its primitive counts and every generated formula has at
least one regular arithmetic cover. ∎

## Corollary 2.2d — Uniform-mixture stoichiometry exploration is normalized

Fix \(S\in\mathfrak S_{\rm DNG}\), set
\(M_S=|\mathcal C(S)|\), and let \(0\le\epsilon\le1\).  Then

\[
p_{\omega,\epsilon}(c\mid S)
=
(1-\epsilon)p_\omega(c\mid S)
+\epsilon\frac{\mathbf1[c\in\mathcal C(S)]}{M_S}
\]

is a normalized probability mass function on exactly \(\mathcal C(S)\).
Every legal formula has mass at least \(\epsilon/M_S\), and \(B\)
independent draws hit any fixed \(c\in\mathcal C(S)\) with probability at
least \(1-(1-\epsilon/M_S)^B\).

### Proof

Both mixture components are normalized on the same finite set, giving the
first claim by convexity.  The second follows from the uniform component.
For independent draws, the miss probability is
\((1-p_{\omega,\epsilon}(c\mid S))^B\le
(1-\epsilon/M_S)^B\). ∎

## Theorem 2.3 — Complete invariant positive-definite metric chart

Let \(P_g\) be the finite fractional point group.  Let \(G_0\succ0\) satisfy

\[
WG_0W^{\mathsf T}=G_0
\quad\forall W\in P_g.
\]

Set \(S=G_0^{1/2}\), \(Q_W=S^{-1}WS\), and let

\[
\mathfrak p_g
=
\{A=A^{\mathsf T}:Q_WAQ_W^{\mathsf T}=A\ \forall W\in P_g\}.
\]

For a basis \(E_1,\ldots,E_{d_g}\), define

\[
G_g(b)=S\exp\!\left(\sum_jb_jE_j\right)S.
\tag{2.4}
\]

Then \(G_g\) is a smooth bijection from \(\mathbb R^{d_g}\) onto

\[
\mathcal K_g
=
\{G\succ0:WGW^{\mathsf T}=G\ \forall W\in P_g\},
\tag{2.5}
\]

and its inverse is smooth.

### Proof

First,

\[
Q_WQ_W^{\mathsf T}
=
S^{-1}WS^2W^{\mathsf T}S^{-1}
=I,
\]

so \(Q_W\) is orthogonal.  For \(A=\sum_jb_jE_j\), \(A\) is symmetric and
\(e^A\succ0\), hence \(G_g(b)\succ0\).  Also

\[
WG_g(b)W^{\mathsf T}
=
S Q_W e^A Q_W^{\mathsf T}S
=
S e^{Q_WAQ_W^{\mathsf T}}S
=G_g(b).
\]

Thus the image lies in \(\mathcal K_g\).

Conversely, take \(G\in\mathcal K_g\) and set

\[
H=S^{-1}GS^{-1}\succ0.
\]

It satisfies \(Q_WHQ_W^{\mathsf T}=H\).  Orthogonal functional calculus gives

\[
Q_W(\log H)Q_W^{\mathsf T}=\log H,
\]

so \(\log H\in\mathfrak p_g\).  It therefore has a unique basis expansion
\(\sum_jb_jE_j\), and \(G=G_g(b)\).  Uniqueness of the principal symmetric
logarithm proves injectivity.  Matrix exponential and principal logarithm are
smooth on the SPD cone. ∎

### Scope of “complete”

The theorem covers every invariant SPD metric in one fixed fractional
representation.  It does not quotient unimodularly equivalent bases or
residual normalizer operations.

### Corollary — Bounded latent chart

If the \(E_j\) are Frobenius-orthonormal, then

\[
z\longmapsto
b=\frac{\mathcal B_gz}{\sqrt{1+\|z\|_2^2}}
\longmapsto G_g(b)
\]

is a smooth bijection from \(\mathbb R^{d_g}\) onto the declared bounded
invariant metric domain
\(\|\log(S^{-1}GS^{-1})\|_{\rm F}<\mathcal B_g\).  The first map is radial, injective,
and has the smooth inverse
\(z=b/\sqrt{\mathcal B_g^2-\|b\|_2^2}\); the second is Theorem 2.3 restricted to that
ball.

## Theorem 2.3b — Loewner interval chart (production `loewner-v2`)

Keep \(G_0\), \(S\), \(Q_W\) and \(\mathfrak p_g\) from Theorem 2.3 and fix
\(0<l<1<u\).  Put \(c=(u-1)/(1-l)\) and define, for \(Z\in\mathfrak p_g\),

\[
W(Z)=l\,I+(u-l)\,\sigma(Z-\ln c\,I),
\qquad
\sigma(x)=\frac1{1+e^{-x}},
\qquad
G_g^{\rm L}(Z)=S\,W(Z)\,S,
\tag{2.5a}
\]

where \(\sigma\) acts by spectral calculus on the symmetric matrix
\(Z-\ln c\,I\); equivalently \(W=(ue^Z+cl\,I)(e^Z+c\,I)^{-1}\).  Then
\(G_g^{\rm L}\) is a smooth bijection from \(\mathfrak p_g\) onto the open Loewner
interval of the invariant cone,

\[
\mathcal K_g(l,u)=\{G\in\mathcal K_g:\ lG_0\prec G\prec uG_0\},
\tag{2.5b}
\]

with smooth inverse and \(W(0)=I\), so \(Z=0\) decodes the reference metric.
Every decoded lattice \(L\) with \(LL^{\mathsf T}=G\) satisfies the shortest-vector
floor

\[
\sigma_{\min}(L)\ \ge\ \sqrt{\lambda_{\min}(G_0)\,l}.
\tag{2.5c}
\]

### Proof

The scalar map \(x\mapsto l+(u-l)\sigma(x-\ln c)\) is a smooth increasing
bijection of \(\mathbb R\) onto \((l,u)\) with nonvanishing derivative, and it
sends \(0\) to \(l+(u-l)/(1+c)=1\) by the choice of \(c\).  Spectral calculus with a
smooth real function is smooth on symmetric matrices, is well defined at
repeated eigenvalues, and commutes with orthogonal conjugation, so
\(Q_WZQ_W^{\mathsf T}=Z\) gives \(Q_WW(Z)Q_W^{\mathsf T}=W(Q_WZQ_W^{\mathsf T})=W(Z)\):
\(W(Z)\) lies in the commutant, and \(G_g^{\rm L}(Z)\in\mathcal K_g\) by the argument
of Theorem 2.3.  The eigenvalues of \(W(Z)\) are the images of those of \(Z\),
hence lie in \((l,u)\); thus \(lI\prec W(Z)\prec uI\), which is
\(lG_0\prec G\prec uG_0\) after conjugation by \(S\).  Conversely, for
\(G\in\mathcal K_g(l,u)\) the whitened matrix \(H=S^{-1}GS^{-1}\) is symmetric,
commutes with every \(Q_W\), and has spectrum in \((l,u)\); applying the inverse
scalar map \(w\mapsto\ln c+\ln\frac{w-l}{u-w}\) spectrally gives the unique
\(Z\in\mathfrak p_g\) with \(W(Z)=H\).  Both directions are smooth because both
scalar maps are.  For (2.5c), \(\sigma_{\min}(L)^2=\lambda_{\min}(G)\ge
\lambda_{\min}(lG_0)=l\,\lambda_{\min}(G_0)\) because \(G\succ lG_0\). ∎

### Remarks

* Only the bijection and the floor are used.  The logistic function is not
  operator monotone, so no Loewner-order monotonicity of \(Z\mapsto W(Z)\) is
  claimed.
* The latent of section 1.7 is \(Z=\sum_jz_jE_j\) in the Frobenius-orthonormal
  basis.  The Gaussian base \(\mathcal N(0,I_{d_g})\) of (1.30a) therefore lives in
  a *logit* coordinate of the whitened spectrum, not in the log-metric
  coordinate \(b\) of (1.25).  The log-metric coefficients remain available as
  \(b=\operatorname{coeff}\log W(Z)\), with spectrum in \((\ln l,\ln u)\); they are
  the bounded lattice coordinate the field receives (section 3.5).
* Unlike the radial ball of the corollary above, which bounds the shortest and
  longest directions together through \(\|\log W\|_{\rm F}<\mathcal B_g\), the floor
  (2.5c) depends on \(l\) alone, so a layered (long-axis) cell is admitted
  through \(u\) without lowering the physical shortest-vector support.  The
  implementation (`lattice.py`, `loewner_lower_bound`) raises \(l\) by whole ulps
  until the floor holds exactly in float64.

## Theorem 2.4 — Exact decoder guarantees group containment

For a scaffold satisfying equation (1.10), let \(G=G_g(b)\), let
\(\mathscr G_{g,L}^{\rm Euc}\) denote the selected infinite Euclidean space
group defined after equation (1.28), and expand every
orbit using (1.26)--(1.28).  Assume each parameter lies in its intended
nonspecialized locus, each orbit has its declared multiplicity, and different
orbits do not collide.  Then the decoded structure:

1. has positive cell volume;
2. is periodic;
3. has exactly \(N_Z^{\rm conv}=\nu c_Z\) conventional-cell atoms;
4. contains the selected infinite Euclidean space group as a
   species-preserving subgroup;
5. maps through complete translation reduction to a well-defined primitive
   physical crystal; and
6. has exact primitive counts
   \(N_Z^{\rm prim}=\nu c_Z/J(C)\), where
   \(J(C)=|T(C)|\ge I_g\), and therefore reduced composition \(c\).

### Proof

Positive definiteness gives an invertible lattice factor and positive volume.
Fractional positions modulo \(\mathbb Z^3\) define a periodic point set.  The
conventional species count is

\[
N_Z
=
\sum_{i:Z_i=Z}\mu_i
=
\sum_w\mu_{gw}h_{wZ}
=
\nu c_Z.
\]

Every selected orbit contains the declared centering copies, so division by
\(I_g\) gives the count after declared centering reduction.  Complete primitive
reduction instead quotients the full species-preserving translation group.
Translations act freely on torus sites, so \(J(C)\) divides every
\(\nu c_Z\).  Since \(\gcd\{c_Z:c_Z>0\}=1\), it also divides \(\nu\), and the final
primitive counts are \((\nu/J(C))c_Z\).  Their reduced ratio is exactly \(c\).

Right multiplication by any selected fractional Hall operation permutes the
right cosets \(H_i\backslash\overline G_g^{\rm conv}\) in (1.28), modulo
conventional translations.  Thus every expanded infinite orbit is preserved.  Species is
constant on each orbit, so the decoration is preserved.  The metric invariance
equation makes its Cartesian realization an isometry.  Therefore, at the level
of symmetry operations of the infinite coloured point set,

\[
\mathscr G_{g,L}^{\rm Euc}
\subseteq
\operatorname{Sym}_{\infty}
\!\left(\operatorname{Per}(D_\alpha^{\rm conv}(q))\right).
\]

After complete primitive reduction, the induced map into the finite
translation-quotiented stabilizer can have a kernel when additional
translations are present.  Thus no injective subgroup claim between the two
finite quotient presentations is required.

∎

### Why equality needs a global check

Within the theorem's nonspecialized, noncollision domain, the selected group
can be a strict subgroup because:

- an orbit retains its declared Wyckoff row and multiplicity but has enhanced
  inherent symmetry;
- several generic orbits jointly create a supergroup;
- the lattice specializes jointly with coordinates; or
- a hidden translation reduces the primitive cell.

A true free-coordinate specialization that collapses the advertised
multiplicity lies outside the theorem's hypotheses and belongs to
\(\Sigma_\alpha\).

The selected cover does not claim equality.  The actual maximal stabilizer is
a measurable post-decode observable in exact arithmetic and must be determined
by an independent complete calculation.

## Proposition 2.4b — Catalogue-certified defect loci are null

Assume that for every arithmetic cover \(\alpha\), the set \(\Sigma_\alpha\) of
orbit collapses and cross-orbit collisions is a finite union of proper smooth
or affine-algebraic subsets of lower dimension in \(\widehat Q_\alpha\).  Then it
has zero product volume.  Any branch law absolutely continuous with respect to
that volume assigns

\[
\rho_{\theta,\alpha}(\Sigma_\alpha\mid c)=0.
\]

If the branch law is obtained from an absolutely continuous base by a smooth
diffeomorphic ODE flow, the same conclusion holds at every time.

### Proof

Every proper lower-dimensional smooth subset has zero ambient volume; so does a
finite union.  Absolute continuity gives the probability statement.  A smooth
diffeomorphism and its inverse are locally Lipschitz and preserve null sets in
local charts. ∎

The word **proper** is essential.  It requires a catalogue proof or exact
regular witness for every arithmetic branch in the declared bounded domain.
Failure for one branch invalidates the declared support rather than creating a
post-hoc rejection rule.

## Theorem 2.5 — Normalized conditional and de-novo overlapping-cover crystal laws

Let \(\mathscr X_c^\dagger\) be standard Borel, let the arithmetic cover set be
finite, let every total extension \(\bar D_\alpha^\dagger\) be measurable, and
assume \(\mathcal X_c^{\rm model}\) is measurable as required by C5.
Suppose

\[
\pi_\phi(\alpha\mid c)\geq0,
\qquad
\sum_\alpha\pi_\phi(\alpha\mid c)=1,
\qquad
\rho_{\theta,\alpha}(\widehat Q_\alpha\mid c)=1,
\qquad
\rho_{\theta,\alpha}(\Sigma_\alpha\mid c)=0.
\tag{2.6}
\]

Then equation (1.29) defines a probability measure on
\(\mathscr X_c^\dagger\), even when cover dimensions differ and decoder images
overlap.  Moreover,

\[
P_\Theta(\{\dagger\}\mid c)=0,
\]

so its restriction to \(\mathscr X_c\) is a probability measure supported on
\(\mathcal X_c^{\rm model}\).

If, additionally, \(\mathcal C_{\rm DNG}\) is finite and nonempty, hypotheses
(2.6) hold for every \(c\in\mathcal C_{\rm DNG}\), and the canonical
composition decoder of Theorem 2.2c supplies the following weights, assume
also that \(\widetilde{\mathscr X}_{\rm DNG}^\dagger\) and
\(\mathscr X_{\rm phys,DNG}^\dagger\) are standard Borel and that
\(J^\dagger\) is measurable:

\[
p_\omega(c)>0\quad\forall c\in\mathcal C_{\rm DNG},
\qquad
\sum_{c\in\mathcal C_{\rm DNG}}p_\omega(c)=1,
\tag{2.6a}
\]

then equation (1.29a) defines a probability measure
\(\widetilde P_\Theta^{\rm DNG}\) on
\(\widetilde{\mathscr X}_{\rm DNG}^\dagger\), its total failure mass is zero,
and its conditional kernel given any formula \(c\) is exactly equation (1.29).
The pushforward (1.29b) is a normalized physical DNG probability measure on
\(\mathscr X_{\rm phys,DNG}^\dagger\).  For every supplied
\(S\in\mathfrak S_{\rm DNG}\), equation (1.29c) likewise defines normalized
tagged and physical S-DNG laws with zero failure mass and

\[
P_\Theta^{\rm S\text{-}DNG}
\!\left(
\left\{C\in\mathscr X_{\rm phys,DNG}:
\operatorname{supp}(r(C))=S\right\}
\middle| S
\right)=1.
\tag{2.6b}
\]

The tagged S-DNG law is also the ordinary conditional kernel in (1.29d).

### Proof

Each pushforward

\[
(\bar D_\alpha^\dagger)_\#\rho_{\theta,\alpha}(A)
=
\rho_{\theta,\alpha}((\bar D_\alpha^\dagger)^{-1}(A))
\]

is a probability measure.  A nonnegative countable weighted sum of measures is
countably additive, and its total mass is

\[
\sum_\alpha
\pi_\phi(\alpha\mid c)
\rho_{\theta,\alpha}(\widehat Q_\alpha\mid c)
=1.
\]

No disjointness or injectivity is used.

For DNG, apply the same argument once more to the finite outer mixture over
compositions.  The tagged law's total mass is

\[
\sum_{c\in\mathcal C_{\rm DNG}}p_\omega(c)
\sum_{\alpha\in\mathcal A_{\rm arith}(c)}
\pi_\phi(\alpha\mid c)
\rho_{\theta,\alpha}(\widehat Q_\alpha\mid c)
=1.
\]

The total failure mass is the same sum with
\(\rho_{\theta,\alpha}(\Sigma_\alpha\mid c)=0\).  Strict positivity in (2.6a)
makes ordinary conditioning on every discrete label \(c\) well defined and
recovers the inner law (1.29).  Finally, a measurable pushforward preserves
normalization and sends all tagged failures to the single global failure
symbol, proving the claim for (1.29b).

For S-DNG, Theorem 2.2c gives
\(\sum_{c\in\mathcal C(S)}p_\omega(c\mid S)=1\).  Replacing the outer sum in
the preceding proof by this fibre sum proves normalization and zero failure
mass for (1.29c).  Every nonfailure term lies in \(\mathscr X_c\) with
\(\operatorname{supp}(c)=S\), proving (2.6b).  Finally,
\(p_\omega(c)/p_\omega(S)=p_\omega(c\mid S)\) on the fibre, so finite-event
conditioning of (1.29a) gives (1.29d). ∎

The identity \(r(C)=c\) almost surely shows that two different outer
composition labels cannot describe the same decoded physical crystal.

The failure mass is

\[
\sum_\alpha
\pi_\phi(\alpha\mid c)
\rho_{\theta,\alpha}(\Sigma_\alpha\mid c)
=0
\]

by (2.6).  Removing a null atom therefore leaves total physical mass one.

For any total measurable endpoint classifier \(M\) with values in
\(\mathcal Y\)—including explicit ambiguity and failure values—the induced mass
is

\[
w_y(c)
=
\sum_\alpha\pi_\phi(\alpha\mid c)
\rho_{\theta,\alpha}
\{q:M(\bar D_\alpha^\dagger(q))=y\}.
\tag{2.7}
\]

Then \(\sum_yw_y(c)=1\) for a countable partition.  In general
\(w_y\ne\pi_\phi(\alpha\mid c)\); cover weights are latent and non-identifiable.

For the physical event \(E_g\) that the infinite coloured-point-set symmetry
group contains a conjugate of the selected group type \(g\), Theorem 2.4 gives

\[
P_\Theta(E_g\mid c)
\ge
\sum_{\alpha:\,g_\alpha=g}\pi_\phi(\alpha\mid c).
\]

Thus a feasible high-symmetry branch gives positive probability to exact
selected-group containment.  If that containment event is lower-dimensional
in a separately specified fixed-\(N\) ambient atom chart, this probability is a
singular component relative to ambient volume.  Without that codimension
hypothesis—one-atom inversion is a counterexample—no singularity claim follows.
This also does not imply positive mass on the event that \(g\) is the
**maximal** group.

## Theorem 2.6 — Positive mass on a lower-dimensional symmetry stratum requires a singular construction

Let \(M\) be a smooth full-dimensional ambient crystal-coordinate manifold, and
let \(S\subset M\) be a lower-dimensional embedded exact-symmetry stratum, or a
countable union of such strata.  If a probability law \(P\) is absolutely
continuous with respect to the volume
measure on \(M\), then

\[
P(S)=0.
\tag{2.8}
\]

Consequently, a diffeomorphic flow of an absolutely continuous base on \(M\)
cannot assign positive probability to \(S\).  To model \(P(S)>0\), one needs a
singular component, for example the discrete chart mixture in equation
(1.29).

### Proof

A lower-dimensional embedded submanifold has zero ambient volume.  Absolute
continuity therefore implies \(P(S)=0\).  A diffeomorphism maps sets of zero
volume to sets of zero volume locally and preserves absolute continuity of a
base density.  Hence an ordinary full-dimensional flow cannot create positive
mass on \(S\). ∎

### Interpretation

This is a necessity result only when the target stratum has lower ambient
dimension.  Some symmetries are forced on an entire fixed-\(N\) state space (the
one-atom inversion example), so dimensionality must be checked rather than
assumed from a group label.  The theorem is not a claim that the chosen chart
parameterization is the only possible implementation.  Discrete
mixtures, spike-and-slab measures, or other explicitly singular constructions
can all satisfy the requirement.

## Theorem 2.7 — Equivariant vector fields descend to quotient flows

Let a group \(\Gamma\) act smoothly and properly on a covering space
\(\widetilde Q\).  Suppose a time-dependent vector field satisfies

\[
v_t(\gamma q)=d\gamma_qv_t(q)
\quad\forall\gamma\in\Gamma,
\tag{2.9}
\]

and its ODE has a unique flow \(\Phi_{t,t_0}\).  Then

\[
\Phi_{t,t_0}(\gamma q)
=
\gamma\Phi_{t,t_0}(q),
\tag{2.10}
\]

so

\[
[q]\longmapsto[\Phi_{t,t_0}(q)]
\tag{2.11}
\]

is well defined on \(Q=\widetilde Q/\Gamma\).

### Proof

The curves \(t\mapsto\Phi_{t,t_0}(\gamma q)\) and
\(t\mapsto\gamma\Phi_{t,t_0}(q)\) start at the same point.  Equation (2.9)
shows they solve the same ODE.  Uniqueness makes them identical, proving
(2.10).  Equivalent representatives therefore map to equivalent
representatives. ∎

### Corollary — Exchangeability

If the base law is \(\Gamma\)-invariant, then its transported law is also
\(\Gamma\)-invariant:

\[
\gamma_\#(\Phi_t)_\#\rho_0
=
(\Phi_t)_\#\gamma_\#\rho_0
=
(\Phi_t)_\#\rho_0.
\tag{2.12}
\]

Taking

\[
\Gamma\supseteq\prod_{w,Z}S_{h_{wZ}}
\]

gives exact exchangeability of repeated identical orbit instances.

## Theorem 2.8 — Expand--perceive--reduce preserves repeated-orbit equivariance

Let \(\sigma\) permute repeated identical orbit instances and let
\(\widetilde\sigma\) be the atom permutation induced after exact expansion.
Assume

\[
E_\alpha(\sigma q)=\widetilde\sigma E_\alpha(q),
\tag{2.13}
\]

\[
T_\theta(\widetilde\sigma x)
=
\widetilde\sigma T_\theta(x),
\tag{2.14}
\]

and

\[
R_{\alpha,G}(\widetilde\sigma v)
=
\sigma R_{\alpha,G}(v).
\tag{2.15}
\]

Then

\[
V_\theta(q)
=
R_{\alpha,G}T_\theta(E_\alpha(q))
\]

satisfies

\[
V_\theta(\sigma q)=\sigma V_\theta(q).
\tag{2.16}
\]

### Proof

Substitute (2.13), then (2.14), then (2.15):

\[
\begin{aligned}
V_\theta(\sigma q)
&=R_{\alpha,G}T_\theta(E_\alpha(\sigma q))\\
&=R_{\alpha,G}T_\theta(\widetilde\sigma E_\alpha(q))\\
&=R_{\alpha,G}\widetilde\sigma T_\theta(E_\alpha(q))\\
&=\sigma V_\theta(q).
\end{aligned}
\]

∎

A Transformer without atom-index embeddings is permutation equivariant only if
all pair features and masks transform covariantly.  This theorem does not
establish residual origin or affine-normalizer equivariance.

## Theorem 2.9 — Current-state metric weighting preserves the CFM population target

Fix one raw cover \(\alpha\), so every velocity below belongs to one fixed
finite-dimensional tangent space.  Let

\[
X_\alpha=(t,q_t,c)
\]

contain all current conditioning variables, let \(U_t\) be the conditional-path
velocity, and let the measurable finite matrix
\(W_\alpha(X_\alpha)\succ0\) depend only on \(X_\alpha\), almost surely under
\(P_{X_\alpha}\).  Assume \(U_t\) is Bochner integrable,
\(\mathbb E\|U_t\|<\infty\), and

\[
\mathbb E[U_t^{\mathsf T}W_\alpha(X_\alpha)U_t]<\infty.
\]

Over measurable vector fields in the weighted space
\(L^2(P_{X_\alpha};W_\alpha)\), the unique minimizer, up to
\(P_{X_\alpha}\)-almost-sure equality, of

\[
\mathcal R(v)
=
\mathbb E[(v(X_\alpha)-U_t)^{\mathsf T}
W_\alpha(X_\alpha)(v(X_\alpha)-U_t)]
\tag{2.17}
\]

is

\[
v_\alpha^\star(X_\alpha)=\mathbb E[U_t\mid X_\alpha].
\tag{2.18}
\]

### Proof

Let \(m(X_\alpha)=\mathbb E[U_t\mid X_\alpha]\).  Conditional on
\(X_\alpha\),

\[
\begin{aligned}
\mathbb E[\|v-U_t\|_{W_\alpha}^2\mid X_\alpha]
&=
\|v-m\|_{W_\alpha}^2
+
\mathbb E[\|U_t-m\|_{W_\alpha}^2\mid X_\alpha],
\end{aligned}
\]

because \(W_\alpha\) is fixed after conditioning on \(X_\alpha\) and the cross term has zero
conditional expectation.  Weighted Jensen's inequality places \(m\) in
\(L^2(P_{X_\alpha};W_\alpha)\).  The second term does not depend on \(v\); the
first is uniquely minimized at \(v=m\), up to \(P_{X_\alpha}\)-null sets,
because \(W_\alpha\succ0\) almost surely. ∎

Summing this risk over a discrete distribution of covers preserves the same
per-cover conclusion; it does not require treating different-dimensional
cover tangents as one vector space.  This is an unrestricted function-space
statement.  It does not assert that a finite shared neural parameterization can
attain every per-cover conditional mean simultaneously.

If instead the weight depends on the unknown endpoint after conditioning on
\(X_\alpha\), assume the conditional expectations below are finite and
\(\mathbb E[W\mid X_\alpha]\) is invertible almost surely.  The minimizer is
generally

\[
v^\star
=
\mathbb E[W\mid X_\alpha]^{-1}
\mathbb E[WU_t\mid X_\alpha],
\tag{2.19}
\]

which is not the ordinary conditional flow velocity.  This is why equation
(1.37) uses the current-state metric.

## Theorem 2.9b — Conditional velocities transport the marginal path

Let \(Z\) index conditional probability-measure paths \(p_t^Z\) on one smooth
raw cover, with \((t,Z)\mapsto p_t^Z\) a measurable probability kernel.  Let
\(u_t^Z\) be a jointly measurable tangent field satisfying

\[
t\longmapsto \int\varphi(q)\,p_t^Z(dq)
\quad\text{is absolutely continuous on }[0,1]
\]

for every compactly supported smooth test function \(\varphi\), for almost
every \(Z\), with

\[
\frac{d}{dt}\int\varphi(q)\,p_t^Z(dq)
=
\int\nabla\varphi(q)\!\cdot u_t^Z(q)\,p_t^Z(dq)
\tag{2.20}
\]

holding for almost every \(t\).  Assume the joint integrability needed for
Fubini and for averaging these absolutely continuous functions, in particular

\[
\int_0^1
\mathbb E_Z\!\left[
\int\|u_t^Z(q)\|p_t^Z(dq)
\right]dt
<\infty.
\]

Define the marginal probability measure and vector-valued current by

\[
p_t(A)=\mathbb E_Z[p_t^Z(A)],
\qquad
M_t(A)
=
\mathbb E_Z\!\left[\int_Au_t^Z(q)\,p_t^Z(dq)\right].
\tag{2.21}
\]

Then, for almost every \(t\), \(M_t\ll p_t\).  Let
\(v_t=dM_t/dp_t\) be its Radon--Nikodym velocity at those times.  For every
test function above, \(t\mapsto\int\varphi\,dp_t\) is absolutely continuous and
the marginal satisfies, for almost every \(t\),

\[
\frac{d}{dt}\int\varphi(q)\,p_t(dq)
=
\int\nabla\varphi(q)\!\cdot v_t(q)\,p_t(dq),
\tag{2.22}
\]

which is the weak continuity equation.  Under the joint law
\(P(dZ)p_t^Z(dq)\), a version of this derivative is
\(v_t(q)=\mathbb E[u_t^Z(q)\mid q_t=q]\).  Equip
\(\widehat Q_\alpha=\mathbb R^{d_g}\times\mathbb T^{d_\alpha}\) with its
declared complete flat product metric \(d_\alpha^{\rm prod}\), canonical tangent
trivialization, and a fixed base point \(q_\star\).  Suppose additionally that
a version of \(v(t,q)\) is measurable in \(t\), continuous in \(q\), and
satisfies the Carathéodory bounds

\[
\|v(t,q)-v(t,q')\|\le L(t)d_\alpha^{\rm prod}(q,q'),
\qquad
\|v(t,q)\|\le a(t)(1+d_\alpha^{\rm prod}(q,q_\star)),
\]

with \(L,a\in L^1([0,1])\).  Then the
ODE has a unique global absolutely continuous flow on \([0,1]\), and that flow
transports \(p_0\) to \(p_1\).

### Proof

At every time for which the current is finite, if \(p_t(A)=0\), nonnegativity
gives \(p_t^Z(A)=0\) for almost every \(Z\), so the integrability assumption
gives \(M_t(A)=0\); hence \(M_t\ll p_t\) for almost every \(t\).  Absolute
continuity in time and the stated joint integrability permit averaging the
fundamental theorem of calculus for (2.20) over \(Z\).  Fubini and
\(M_t(dq)=v_t(q)p_t(dq)\) then give (2.22).  The conditional-expectation
identity is the defining disintegration of the same current.  The final
statement is the standard uniqueness result for the continuity equation
induced by the assumed global ODE flow. ∎

This theorem applies on the redundant covering completion from section 1.10.
It does not prove that arbitrary neural parameters are globally Lipschitz or
that an image-list cutoff is smooth; those are architecture obligations.

## Proposition 2.9d — Gauge-aligned minibatch couplings keep both endpoint marginals

Fix one cover \(\alpha\), the variable the geometry field is conditioned on.  Let
\(\Gamma\) be its certified finite residual group, acting on the intrinsic torus
by affine unimodular maps and trivially on \(z\); let the base \(\rho_0\) be
\(\Gamma\)-invariant (it is: Haar on the torus and \(\mathcal N(0,I)\) in \(z\)),
and let the target \(\bar\rho_1\) be the symmetrized endpoint law of (1.30b).
Draw \(B\) base samples \(q_0^{(i)}\sim\rho_0\) i.i.d. and, independently, \(B\)
records \(q_1^{(j)}\sim\bar\rho_1\) of that cover.  Give every record a candidate
list \((\gamma_{jk})_{k=1}^{K_j}\) of one of two kinds:

* (i) \(K\ge1\) actions drawn i.i.d. from \(\operatorname{Unif}(\Gamma)\),
  independently of everything else; or
* (ii) a fixed list in which every element of \(\Gamma\) occurs equally often
  (for example \(\Gamma\) itself).

For any measurable cost \(c(q_0,q_1)\) set
\(C_{ij}=\min_k c(\gamma_{jk}^{-1}q_0^{(i)},q_1^{(j)})\), take a minimizing
assignment \(\pi\) of \(\sum_iC_{i\pi(i)}\) by a deterministic rule, and for
record \(j\) let \(\gamma_j^\star\) be its minimizing candidate against base
\(\pi^{-1}(j)\), the first one in the list among equal costs.  In case (ii) assume
that for \(\gamma\ne\gamma'\) in \(\Gamma\) the tie set
\(\{q_0:c(\gamma^{-1}q_0,q_1)=c(\gamma'^{-1}q_0,q_1)\}\) is \(\rho_0\)-null for
almost every \(q_1\).  Then the coupling that connects \(q_0^{(\pi^{-1}(j))}\) to
\(\gamma_j^\star q_1^{(j)}\) (by the pushed path of section 3.7, built from the
unpushed base \((\gamma_j^\star)^{-1}q_0^{(\pi^{-1}(j))}\)) has the base marginal
\(\rho_0\) and the endpoint marginal \(\bar\rho_1\), for every \(B\), every
candidate rule and every such cost.  Here the marginals are those of the pair at
a uniformly chosen batch position, which is the law the summed CFM loss
averages over.

### Proof

The base marginal: \(\pi\) is a bijection, so the starts of the \(B\) paths are
the drawn base samples in another order, and
\(\sum_jf(q_0^{(\pi^{-1}(j))})=\sum_if(q_0^{(i)})\) for every \(f\).  The start at
a uniformly chosen position therefore has the law \(\rho_0\).

The endpoint marginal: fix the records.  For \(g\in\Gamma\) let \(T_g\) replace
every base sample by \(gq_0^{(i)}\) and, in case (i), every candidate by
\(g\gamma_{jk}\).  In case (i) every cost is unchanged, by the identity
\[
c\bigl((g\gamma_{jk})^{-1}gq_0^{(i)},q_1^{(j)}\bigr)
=c\bigl(\gamma_{jk}^{-1}q_0^{(i)},q_1^{(j)}\bigr),
\]
which uses no property of \(c\).  Hence the matrix \(C\), the assignment and the
minimizing indices are unchanged, and the chosen action of record \(j\) becomes
\(g\gamma_j^\star\).  In case (ii) the list is fixed, and left multiplication by
\(g\) permutes it up to the equal multiplicities; the candidate \(g\gamma\) has
against \(gq_0^{(i)}\) the cost that \(\gamma\) had against \(q_0^{(i)}\).  So the
set of candidate costs of every pair \((i,j)\) is unchanged, \(C\) and the
assignment are unchanged, and outside the tie sets the minimizing element
becomes \(g\gamma_j^\star\).  The joint law of the base samples and the
candidates is \(T_g\)-invariant (\(\rho_0\) is invariant, uniform candidates stay
uniform), so
\(\mathbb P(\gamma_j^\star=\gamma\mid\text{records})=\mathbb P(\gamma_j^\star=g\gamma\mid\text{records})\)
for all \(g\): \(\gamma_j^\star\) is uniform on \(\Gamma\) and independent of the
records.  The pushed endpoint \(\gamma_j^\star q_1^{(j)}\) therefore has the law
\(|\Gamma|^{-1}\sum_\gamma(\gamma)_\#\bar\rho_1=\bar\rho_1\) by the invariance of
\(\bar\rho_1\) (and for records drawn from the unsymmetrized data law it has the
symmetrized law, which is how the objective uses it).  Every record is used
exactly once. ∎

### Remarks

* Theorem 2.9b needs only the endpoint marginals, so the conditional-expectation
  field of any such coupling still transports the base to \(\bar\rho_1\).  What
  the coupling changes is the conditional variance of the target velocity,
  which section 2.9's gradient-noise argument identifies as the limit of the
  learned endpoint accuracy (the F10 loss decomposition: more than 99% of the
  loss is target variance).  The alignment removes the variance that the
  uniform gauge push alone injects, and the assignment reduces the crossing of
  paths within the batch (minibatch optimal transport).
* One cover per assignment.  The proposition is about records of one cover.  An
  assignment across covers would still permute the base samples of the batch,
  but the base samples that reach the records of a given cover would no longer
  be \(\rho_0\)-distributed: each cover competes for the bases nearest to its own
  records.  The field is conditioned on the cover and sampling starts every
  cover from \(\rho_0\), so such a coupling would train a different conditional
  problem from the one that is sampled.  The same failure appears when the
  residual groups of two covers differ, since no \(T_g\) then acts on both.
  The implementation couples a batch block by block, one block per cover; a
  block of one record is the pure gauge alignment (\(B=1\)).
* The cost need not be \(\Gamma\)-invariant.  For an invariant cost
  \(c(\gamma^{-1}q_0,q_1)=c(q_0,\gamma q_1)\) and rule (ii) makes
  \(C_{ij}=\min_{\gamma\in\Gamma}c(q_0^{(i)},\gamma q_1^{(j)})\) the exact quotient
  cost; rule (i) minimizes over a random subset.  The implemented cost is the
  squared length of the path actually trained on (the half-open lift of
  \(\eta\) in the reference translation-quotient metric plus \(d_z\) in the
  chart-origin pullback metric), a positive definite quadratic form of an affine
  image of \(q_0\); its tie sets between distinct torus maps are null because the
  dequantized record has a density (\(\sigma_\eta>0\)).
* The exact Fourier oracle of F10 evaluates the endpoint law and is unchanged;
  its velocity field is that of the independent coupling and is not a valid
  reference for \(\rho_v\) under another coupling.
* Implementation: ``coupling="gauge-ot-v1"`` of the batched objective.  The
  candidates come from the digit-coded action table of the cover
  (``ResidualActionTable``): rule (ii) with every digit row once when the table
  has at most ``coupling_candidates`` rows, rule (i) otherwise.  Research run R1
  (2026-10-30; block structure and rule (ii) added 2026-11-10 before any
  aligned result was observed).

## Proposition 2.9e — Exact model likelihood of the cover law

For a cover \(\alpha\) with \(d_g\) lattice coordinates and intrinsic torus
\(\mathbb R^{d_\alpha}/\Lambda_\alpha\) (period basis \(B\)), let
\(\Phi_{1,0}\) be the flow of the learned field and \(\rho_{\theta,\alpha}=(\Phi_{1,0})_\#\rho_{0,\alpha}\)
as in (1.30c).  Then for every \(q_1=(z_1,\eta_1)\),

\[
\log\rho_{\theta,\alpha}(q_1)
=
\log\rho_{0,\alpha}(q_0)
-\int_0^1\operatorname{div}v_\theta(t,q_t)\,dt,
\qquad
q_t=\Phi_{t,1}(q_1),
\tag{2.17a}
\]

with \(\log\rho_{0,\alpha}(z,\eta)=-\tfrac12\|z\|^2-\tfrac{d_g}2\log2\pi-\log|\det B|\),
the divergence taken in the flat coordinates \((z,\eta)\).  This is the
instantaneous change of variables on \(\mathbb R^{d_g}\times\mathbb R^{d_\alpha}/\Lambda_\alpha\)
(a flat manifold, so the Riemannian correction vanishes); the residual-group
average (1.30b) does not change \(\rho_{0,\alpha}\) because the raw base is
invariant.  The value is the density of the dequantized endpoint law that the
flow was trained to match, with respect to the cover's own product volume.  It
is exact up to the quadrature of the backward ODE, and the exact Fourier oracle
certifies the implementation: run on the oracle's own field, (2.17a) reproduces
the oracle's closed-form density (`tests/test_flow_log_density.py`).

### Use and limits

Within one cover, \(\log\pi_\phi(\alpha\mid c)+\log\rho_{\theta,\alpha}(q_1)\) is the
model's exact log probability of a generated candidate in its generative
parametrization, available without an energy model; it ranks candidates of one
cover and of covers of equal dimension.  Densities of covers of different
dimension are taken with respect to different volumes and are not comparable
as numbers (Theorem 2.6); ranking across them uses \(\pi_\phi\) alone or a
declared common smoothing, which this document does not define.  Implementation:
`LearnedCrystalGenerator.geometry_log_density` (exact trace or Hutchinson
divergence) and `rank_samples` (research R2, 2026-10-30).

## Proposition 2.9c — A diffeomorphic flow preserves full support up to null sets

Let the base on one raw cover have a density that is strictly positive almost
everywhere, and let \(\Phi\) be a smooth diffeomorphism of that cover.  Then
\(\Phi_\#\rho_0\) also has a density strictly positive almost everywhere.  In
particular, such a flow cannot assign zero mass density to a nonempty open
region.

### Proof

The change-of-variables density is

\[
p_1(q)=p_0(\Phi^{-1}q)
\left|\det D\Phi^{-1}(q)\right|.
\]

Both factors are positive almost everywhere for a diffeomorphism. ∎

This is why section 1.10 uses Gaussian noise and a torus heat kernel rather
than a component-truncated or compact-support endpoint.

## Proposition 2.10 — Exact pre-origin-quotient fixed-cell pullback

At fixed \(G\), the expanded Cartesian displacement generated by
\(\delta\xi_i\) obeys

\[
\sum_{k=1}^{\mu_i}\|\delta r_{ik}\|_2^2
=
\delta\xi_i
\left(\mu_iA_iGA_i^{\mathsf T}\right)
\delta\xi_i^{\mathsf T}.
\tag{2.23}
\]

### Proof

From equation (1.34),

\[
\begin{aligned}
\sum_k\|\delta r_{ik}\|_2^2
&=
\delta\xi_i
\left(\sum_kA_iW_{ik}GW_{ik}^{\mathsf T}A_i^{\mathsf T}\right)
\delta\xi_i^{\mathsf T}\\
&=
\delta\xi_i
\left(\sum_kA_iGA_i^{\mathsf T}\right)
\delta\xi_i^{\mathsf T}\\
&=
\delta\xi_i
\left(\mu_iA_iGA_i^{\mathsf T}\right)
\delta\xi_i^{\mathsf T},
\end{aligned}
\]

using \(W_{ik}GW_{ik}^{\mathsf T}=G\). ∎

When the lattice also changes, the full Cartesian differential contains a
lattice term and cross terms.  Equation (2.23) is therefore exact for the
fixed-cell internal-coordinate block before the continuous origin quotient,
not for the full joint manifold.

### Corollary — Translation-quotient pullback

Let \(J_\alpha^{\rm cart}\) map gauge-reduced internal velocities to all expanded
Cartesian velocities and
\(P_N=(I_N-\mathbf1\mathbf1^{\mathsf T}/N)\otimes I_3\).  Then

\[
W_\alpha^{\rm quot}
=
(J_\alpha^{\rm cart})^{\mathsf T}
P_NJ_\alpha^{\rm cart}
\]

is the exact fixed-cell pullback after quotienting global translations.

### Proof

For stacked displacement \(d\),

\[
\min_{t\in\mathbb R^3}\sum_a\|d_a-t\|_2^2
=d^{\mathsf T}P_Nd.
\]

Substituting \(d=J_\alpha^{\rm cart}\delta\eta\) gives the stated quadratic form.
To close the rank statement, write the affine reconstruction as
\(\Xi_i(\eta)=b_i+\eta C_i\).  The stacked fractional advertised-slot Jacobian
is constant in \(\eta\).  With column-vector stacking,

\[
J_\alpha^{\rm cart}(G)
=
(I_N\otimes L^{\mathsf T})J_\alpha^{\rm frac},
\qquad
P_N(I_N\otimes L^{\mathsf T})
=
(I_N\otimes L^{\mathsf T})P_N.
\]

Since \(L\) is invertible,

\[
\operatorname{rank}(P_NJ_\alpha^{\rm cart}(G))
=
\operatorname{rank}(P_NJ_\alpha^{\rm frac}).
\]

The tangent-kernel certificate (1.15a) makes the gauge-reduced right-hand side
full column rank.  Hence \(W_\alpha^{\rm quot}\succ0\) for every \(\eta\) and
every decoded \(G\succ0\), including advertised-slot collision or
specialization points.  It is a smooth cover-training metric; at a
hidden-translation point it is not asserted to be the metric of the further
reduced physical quotient.  Dividing by \(\ell_0^2\) gives the dimensionless
weight (1.35a) without changing rank. ∎

## Theorem 2.11 — Irreducible top-1 polymorph error

Fix any declared metric \(d_{\mathscr X}\) on the physical quotient.  Suppose
that under the same condition \(c\), two measurable polymorph basins
\(\mathcal B_1,\mathcal B_2\) have probabilities \(p_1,p_2>0\) and

\[
\inf_{C_1\in\mathcal B_1,\ C_2\in\mathcal B_2}
d_{\mathscr X}(C_1,C_2)>2\varepsilon.
\tag{2.24}
\]

Every deterministic composition-only prediction \(\widehat C(c)\) then obeys

\[
P[d_{\mathscr X}(C,\widehat C(c))>\varepsilon\mid c]
\geq\min(p_1,p_2).
\tag{2.25}
\]

### Proof

The radius-\(\varepsilon\) ball around one prediction cannot intersect both
basins by (2.24).  It therefore misses at least the smaller positive basin
mass. ∎

The theorem rules out guaranteed unique prediction in this case.  A
conditional distribution or candidate set can represent the uncertainty, but
its calibration and coverage remain empirical properties to be tested.

## Theorem 2.12 — Unrestricted OOD correctness is unidentifiable

Let an unseen condition \(c_\star\) admit at least two distinct crystals
\(C_a,C_b\) in the declared physical class.  For any finite training support
that excludes \(c_\star\), there are two admissible conditional data-generating
laws that agree on every training condition but put unit mass on different
crystals at \(c_\star\).  No learner using only the training observations can
guarantee the correct conditional law at \(c_\star\).

### Proof

Construct two laws \(P_1,P_2\) that are identical on the training support and
satisfy

\[
P_1(C_a\mid c_\star)=1,
\qquad
P_2(C_b\mid c_\star)=1,
\qquad C_a\ne C_b.
\]

The learner receives exactly the same data in both worlds, hence produces the
same prediction.  That prediction cannot be correct with probability one in
both worlds. ∎

Exact constraints can guarantee legality out of distribution.  They cannot
guarantee learned probabilities or geometry without additional assumptions or
information.

The same no-free-lunch argument applies to DNG.  Two admissible joint laws can
agree on every observed formula and structure while assigning different
probability or different conditional geometry to an unobserved formula.
Positive grammar support is therefore not evidence of physical realizability
or calibrated prevalence.

## Proposition 2.13 — Crystallographic legality does not imply stability

The decoder theorems impose no condition equivalent to

\[
\nabla_CE(C)=0,
\tag{2.26}
\]

no positivity condition on the mass-weighted dynamical matrix

\[
D_C(k)\succeq0\quad\forall k,
\tag{2.27}
\]

and no convex-hull free-energy condition

\[
\Delta G_{\rm hull}(C)=0.
\tag{2.28}
\]

Therefore stability and synthesizability cannot be deduced from the preceding
theorems.  Any later stability claim requires a separately validated physical
model and evaluation protocol.

## Status of the mathematical contribution

The individual tools used above—group actions, quotient measures, dynamic
programming, invariant SPD exponentials, equivariant ODEs, and flow matching—are
established mathematics.  The candidate theoretical contribution is the
crystallographic synthesis and its exact separation of:

1. a standard normalized canonical-composition outer law for DNG;
2. formula-safe discrete cover support;
3. an overlapping mixed-dimensional containment-cover measure;
4. continuous-origin reduction and repeated-orbit exchangeability; and
5. current-state metric intrinsic flow.

Item 1 is an ordinary chain-rule/grammar-constrained construction and is not
claimed as novelty.  It makes DNG mathematically explicit without altering the
candidate crystallographic contribution in items 2--5.

These theorems give conditional implications under their stated assumptions;
they do not by themselves certify the catalogue, implementation, literature
priority, or predictive superiority.
