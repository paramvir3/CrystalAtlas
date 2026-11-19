# 1. Mathematical foundations

## 1.1 Row-vector crystallographic convention

Fractional coordinates are row vectors.  A cell is represented by

\[
C=(L,F,Z),
\qquad
L\in GL^+(3,\mathbb R),
\qquad
F=(f_1,\ldots,f_N)^{\mathsf T}\in(\mathbb T^3)^N.
\tag{1.1}
\]

The lattice vectors are the rows of \(L\), so

\[
r_i=f_iL,
\qquad
G=LL^{\mathsf T},
\qquad
\|r_i-r_j\|_2^2=(f_i-f_j)G(f_i-f_j)^{\mathsf T}.
\tag{1.2}
\]

A fractional symmetry operation is

\[
f\mapsto fW+w\pmod{\mathbb Z^3},
\qquad W\in GL(3,\mathbb Z),
\quad w\in\mathbb T^3.
\tag{1.3}
\]

It is a Cartesian isometry exactly when

\[
WGW^{\mathsf T}=G.
\tag{1.4}
\]

These equations are the normative convention.  An external library using
column vectors must be converted explicitly by transposing the rotational
part.

## 1.2 Primitive physical object and conventional chart cover

The physical crystal is an infinite coloured periodic point set whose
translation lattice is the **full primitive translation lattice**.  Choose a
primitive presentation.  Atom indices, its origin, primitive basis, and
Cartesian orientation are not physical labels.  Let \(\sigma\) be a
species-preserving permutation, \(\tau\in\mathbb T^3\) an origin shift,
\(U\in SL(3,\mathbb Z)\) a right-handed unimodular primitive-basis change, and
\(Q\in SO(3)\) a proper Cartesian rotation.  Define

\[
L'=ULQ,
\qquad
f_i'=(f_{\sigma(i)}+\tau)U^{-1}\pmod{\mathbb Z^3},
\qquad
Z_i'=Z_{\sigma(i)}.
\tag{1.5}
\]

Then \(C\sim C'\).  Proper rotations keep enantiomorphs distinct.  Replacing
\(SO(3)\) by \(O(3)\) would identify mirror images and would define a different
scientific target.

The restriction \(U\in SL(3,\mathbb Z)\) concerns right-handed **basis gauge**
only.  It does not exclude determinant-minus-one matrices \(W\) that are actual
improper symmetries of a crystal.

Equation (1.5) defines equivalence between primitive presentations.  A pinned
Hall/Wyckoff catalogue instead uses a standard **conventional cover**.  For its
setting \(g\), let \(\overline G_g^{\rm conv}\) be its exact pinned conventional
operation group and define the known finite centering subgroup

\[
T_g
=
\{t\in\mathbb T^3:(I,t)\in\overline G_g^{\rm conv}\},
\qquad
I_g=|T_g|.
\tag{1.6}
\]

The selected Hall setting's **declared** translation lattice, expressed in
conventional fractional coordinates, is \(\mathbb Z^3+T_g\).  Thus a centered conventional cell is not a
spurious supercell: its noninteger centering translations are required by the
pinned setting.  Every conventional decode is reduced through this known
subgroup before it enters the primitive physical quotient.  Any **additional**
pure translation beyond \(T_g\) enlarges the decoded translation lattice to
\(\mathbb Z^3+T(C)\), signals a smaller primitive cell, and must cause
reclassification.

Concretely, the catalogue supplies an exact rational centering-reduction matrix
\(C_g\in GL(3,\mathbb Q)\) with \(\det C_g=I_g^{-1}>0\),
\(C_g^{-1}\in M_3(\mathbb Z)\), and kernel exactly \(T_g\) for the induced torus
map, such that

\[
L_{\rm dcr}=C_gL_{\rm conv},
\qquad
f_{\rm dcr}=f_{\rm conv}C_g^{-1}\pmod{\mathbb Z^3}.
\]

Here “dcr” means **declared-centering-reduced**.  The map
\(\operatorname{CentRed}_g\) applies these equations and deduplicates the known
centering copies exactly.  It need not yet be a full primitive reduction when a
decoded motif has additional translations.  Let
\(\operatorname{FullPrim}\) determine and quotient the complete
species-preserving translation lattice.  The physical decoder used below is

\[
\bar D_\alpha
=
\left[\operatorname{FullPrim}\circ
\operatorname{CentRed}_g\circ D_\alpha^{\rm conv}\right]_{\sim},
\]

where the brackets denote the right-handed primitive quotient (1.5).  The map
is indexed by the raw cover \(\alpha\); different covers may have the same image.

The resulting broad physical class at composition \(c\) is denoted
\(\mathscr X_c\).  The target and model subsets introduced in section 0.2 are
\(\mathcal X_c^{\rm target}\) and \(\mathcal X_c^{\rm model}\).
For DNG, use the finite standard-Borel coproduct

\[
\widetilde{\mathscr X}_{\rm DNG}
=
\bigsqcup_{c\in\mathcal C_{\rm DNG}}\mathscr X_c.
\]

Let \(\mathscr X_{\rm phys,DNG}=\bigcup_c\mathscr X_c\) be the corresponding
untagged physical union.  It is assumed standard Borel, and the forgetful map
\(J(c,C)=C\) from the coproduct is assumed measurable.  The measurable
reduced-composition map
\(r:\mathscr X_{\rm phys,DNG}\to\mathcal C_{\rm DNG}\) satisfies
\(r(\bar D_\alpha(q))=c\) for every regular decode from a cover in
\(\mathcal A_{\rm arith}(c)\).  Hence \(J\) is injective on physical states:
the outer composition components are intrinsically disjoint even though raw
cover images overlap inside one component.

## 1.3 Actual coloured stabilizer

The normative stabilizer is computed after exact reduction to a primitive
basis.  Let \(\operatorname{Sym}_\infty(C)\) be the complete Euclidean isometry
group of the infinite coloured point set.  In a primitive fractional
presentation, define its finite species-preserving quotient modulo primitive
integer lattice translations by

\[
\overline G_C
=
\left\{
(W,w)\in GL(3,\mathbb Z)\ltimes\mathbb T^3:
WGW^{\mathsf T}=G,
\quad
\{(f_iW+w,Z_i)\}_i=\{(f_i,Z_i)\}_i
\right\}.
\tag{1.7}
\]

The complete stabilizer is computed in the full primitive basis and may then
be transformed to **its own** pinned conventional setting.  When hidden
translations are present, its rotations need not act through integer matrices
on the originally selected conventional supercell.  Therefore the complete
actual finite stabilizer is not assumed to act on that raw selected torus.

For a decoded motif in the raw conventional cover, define its
species-preserving pure-translation subgroup directly by

\[
T(C)
=
\left\{
t\in\mathbb T^3:
\{(f_i+t,Z_i)\}_i=\{(f_i,Z_i)\}_i
\right\}.
\tag{1.8}
\]

Every regular decode in setting \(g\) contains \(T_g\).  The full subgroup \(T(C)\)
may strictly contain it; complete primitive reduction must detect this rather
than assuming \(T(C)=T_g\).  The equality \(T(C)=T_g\) is required only for the
special formula \(N_Z^{\rm prim}=\nu c_Z/I_g\); version 1 does not condition on
a supplied primitive atom count.

The phrase **maximal symmetry** means the complete primitive group (1.7), or
its exactly conjugate conventional representation.  It does not
mean a maximal element of the list of 230 space-group types.  Under (1.5), the
stabilizer changes by conjugacy, so its physical type is well defined.

Use one pinned Hall realization for each of the 230 physical space-group types.
Hall settings are operation-table and coordinate-setting choices; they are not
530 distinct physical labels to be learned.

For a pinned realization, its finite fractional point group is

\[
P_g
=
\{W:\exists w\text{ with }(W,w)\in\overline G_g^{\rm conv}\}.
\]

## 1.4 Discrete scaffold

A raw scaffold is

\[
\alpha=(\nu,g,h),
\qquad
h=(h_{wZ})_{w,Z},
\tag{1.9}
\]

where \(g\) identifies the pinned realization and \(h_{wZ}\) counts independent
orbits of species \(Z\) on the explicit Wyckoff position \(w\).  The necessary
arithmetic constraints are

\[
\sum_w\mu_{gw}h_{wZ}=\nu c_Z
\quad\forall Z,
\tag{1.10}
\]

and

\[
\sum_Zh_{wZ}\leq1
\quad\text{when the position has zero free dimensions.}
\tag{1.11}
\]

Because every conventional Wyckoff orbit contains complete centering copies,
equation (1.10) implies that \(\nu c_Z/I_g\) is integral.  This is the count after
removing the **declared** centering copies.  If the decoded motif has full
translation index \(J(C)>I_g\), its final primitive count is instead
\(\nu c_Z/J(C)\); the reduced composition is unchanged.

Positive-dimensional positions may occur repeatedly with distinct parameters.
The atom-count bound makes the action space finite.

A residual chirality-preserving affine-normalizer operation can exchange
Wyckoff labels, origin choices, or right-handed cell descriptions.  Version 1
pins one exact catalogue convention and permits overlapping raw covers rather
than pretending that \(h\) is a unique physical label.  Orientation-reversing
maps are excluded.

## 1.5 Bounded arithmetic symmetry-containment cover

Let

\[
\mathcal K_g^{\mathcal B}
=
\{G\in\operatorname{Sym}_{++}(3):
WGW^{\mathsf T}=G\ \forall W\in P_g,\quad
\|\log(G_0^{-1/2}GG_0^{-1/2})\|_{\rm F}<\mathcal B_g\}
\tag{1.12}
\]

be the bounded symmetry-compatible metric domain declared in
`00_SCOPE.md`.  The norm is evaluated only after the declared right-handed
pinned conventional standardization.  It is a computational domain definition,
not a basis-invariant physical observable.  For each explicit Wyckoff position,
use the compact intrinsic torus from equation (1.27).  Let

\[
\mathbb T_\alpha^{\rm full}
=
\prod_{w,Z}
\left(\mathbb R^{d_{gw}}/\Lambda_{gw}\right)^{h_{wZ}}.
\tag{1.13}
\]

Any continuous origin freedom acts through a closed subtorus
\(H_\alpha^{\rm org}\).  Exact integer normal forms define the compact quotient

\[
\mathbb T_\alpha^{\rm int}
=
\mathbb T_\alpha^{\rm full}/H_\alpha^{\rm org}.
\tag{1.14}
\]

The version-1 continuous cover is the full product

\[
\widehat Q_\alpha
=
\mathbb R^{d_g}
\times
\mathbb T_\alpha^{\rm int},
\tag{1.15}
\]

with latent \(z\in\mathbb R^{d_g}\) decoded by
\(G=G_g(\kappa_g(z))\) into \(\mathcal K_g^{\mathcal B}\).  Thus Gaussian measure
on the first factor and Haar measure on the second are literal reference
measures on \(\widehat Q_\alpha\).

The origin quotient is required to be exact also at tangent level.  If
\(J_{\alpha,\rm full}^{\rm frac}\) maps velocities on
\(\mathbb T_\alpha^{\rm full}\) to the stacked advertised-slot fractional
velocities, then the catalogue must certify

\[
\ker\!\left(P_NJ_{\alpha,\rm full}^{\rm frac}\right)
=T H_\alpha^{\rm org},
\tag{1.15a}
\]

where \(P_N\) removes one common translation as defined in section 1.11.
Consequently the induced gauge-reduced Jacobian is injective.  This
certificate—not a numerical ridge—is what makes the intrinsic positional
metric positive definite.

It is deliberately **not** cut into disjoint maximal-stabilizer components.
Let \(\Sigma_\alpha\) contain the parameters where an advertised orbit loses
multiplicity, two distinct advertised orbits collide, or the exact decoder is
otherwise not a coloured crystal with the declared conventional count.  The
catalogue obligation is to prove, for **every** arithmetic table inside the
declared bounds, that \(\Sigma_\alpha\) is a finite union of proper
lower-dimensional affine/algebraic subsets and hence has zero product volume.
This is an admissibility axiom for the declared catalogue/domain, not a
post-hoc branch filter.  If any arithmetic branch fails it, the domain is not
frozen and the selector theorem does not apply.

Extra translations or a larger actual stabilizer are **not** singular failures.
They remain in the cover and are discovered by complete primitive reduction and
post-decode stabilizer analysis.  Consequently different covers may overlap in
physical crystal space, and an entire nominal cover may generically decode to a
larger group.  This redundancy is the price of avoiding an intractable
maximal-stratum emptiness and connected-component oracle.

Repeated identical orbit instances still have a finite permutation redundancy.
The base law and vector field will be exchangeable under it.  Other residual
chirality-preserving normalizer representatives are either fixed by one exact
catalogue convention or explicitly averaged; version 1 does not claim that the
latent cover itself is a unique physical quotient.

Precisely, let \(\Pi_\alpha=\prod_{w,Z}S_{h_{wZ}}\) be the
repeated-identical-orbit permutation group.  It is handled by exchangeable
measures and set-equivariant maps, never by factorial enumeration.  Any
additional retained within-cover representative group
\(\Gamma_\alpha^{\rm res}\) is a **finite, explicitly bounded** group of smooth
diffeomorphisms

\[
\gamma:\widehat Q_\alpha\longrightarrow\widehat Q_\alpha
\]

such that, for every \(\gamma\in\Gamma_\alpha^{\rm res}\):

1. \(\gamma(\Sigma_\alpha)=\Sigma_\alpha\) and \(\gamma\) preserves the
   declared bounded cover;
2. \(\bar D_\alpha^\dagger(\gamma q)=\bar D_\alpha^\dagger(q)\), so the
   action changes only the stored representative of the same physical output;
3. its action on the intrinsic torus is affine with a constant invertible
   derivative \(D_\eta\gamma\) preserving \(\Lambda_\alpha\); and
4. its action on \(z\) and its inverse are smooth; and
5. it normalizes the repeated-orbit action,
   \(\gamma\Pi_\alpha\gamma^{-1}=\Pi_\alpha\), including the corresponding
   tangent representation.

It contains only residual representatives that act inside the same raw cover.
A transformation that
changes \((\nu,g,h)\) relates two overlapping covers and is **not** an element
of \(\Gamma_\alpha^{\rm res}\).  Infinite or continuous normalizer freedom must be removed
by the pinned catalogue/origin quotient rather than placed in this finite
average.  If the catalogue proves a unique section, set
\(\Gamma_\alpha^{\rm res}=\{e\}\).

## 1.6 Exact formula grammar

For fixed \(g,\nu\), set \(\mathbf N=\nu c\).  Order the explicit Wyckoff positions by
\(r=1,\ldots,R_g\), and let \(a_r\in\mathbb N_0^{|\mathcal E|}\) denote the
row occupation vector.  Its local finite action set \(\mathcal H_{gr}\) applies
the capacity rule (1.11).

For a residual element-count vector \(\rho\), define

\[
B_{g,R_g+1}(\rho)=\mathbf1[\rho=0],
\tag{1.16}
\]

\[
B_{g,r}(\rho)
=
\bigvee_{\substack{a\in\mathcal H_{gr}\\
\mu_{gr}a\leq\rho}}
B_{g,r+1}(\rho-\mu_{gr}a).
\tag{1.17}
\]

Equations (1.16)--(1.17) certify arithmetic formula completion.  Their
completion-safe action set is

\[
\mathcal A^{\rm arith}_{gr}(\rho)
=
\{a\in\mathcal H_{gr}:
\mu_{gr}a\leq\rho,
B_{g,r+1}(\rho-\mu_{gr}a)=1\}.
\tag{1.18}
\]

For finite logits \(\ell_{\phi,r}\), the local probability is

\[
p_\phi(a_r\mid a_{<r},g,\nu,c)
=
\frac{
\mathbf1[a_r\in\mathcal A^{\rm arith}_{gr}(\rho_r)]
e^{\ell_{\phi,r}(a_r)}
}{
\sum_{a\in\mathcal A^{\rm arith}_{gr}(\rho_r)}
e^{\ell_{\phi,r}(a)}
}.
\tag{1.19}
\]

Define

\[
\mathcal V(c)
=
\left\{
\nu\in\{1,\ldots,\nu_{\max}\}:
\nu\sum_Zc_Z\leq N_{\max}^{\rm raw},\quad
\exists g,\ B_{g,1}(\nu c)=1
\right\},
\qquad
\mathcal G(c,\nu)
=
\{g\in\mathcal G_{\rm pin}:B_{g,1}(\nu c)=1\}.
\]

where \(\mathcal G_{\rm pin}\) is the finite set of supported pinned
realizations.

The supports of \(p(\nu\mid c)\) and \(p(g\mid\nu,c)\) are respectively
\(\mathcal V(c)\) and \(\mathcal G(c,\nu)\).  If \(\mathcal V(c)\) is empty, the
condition lies outside the declared domain.  The raw-cover law is

\[
\pi_\phi(\alpha\mid c)
=
p_\phi(\nu\mid c)
p_\phi(g\mid\nu,c)
\prod_{r=1}^{R_g}
p_\phi(a_r\mid a_{<r},g,\nu,c).
\tag{1.20}
\]

Equation (1.19) defines a direct constrained autoregressive law on arithmetic
completions.  It is not in
general the global conditional of an unconstrained autoregressive model given
eventual legality.

### Canonical composition grammar for de-novo generation

Let \(\mathcal C_{\rm DNG}\) be the finite set in equation (0.1a), and let
\(s(c)\) be its unique serialization (0.1b).  Include the number of distinct
elements \(K\) as the first generated token.  BOS is an external context
symbol and is not part of \(s(c)\).  Let

\[
\mathcal P_{\rm comp}
=
\{u:\exists c\in\mathcal C_{\rm DNG}\text{ such that }u
\text{ is a prefix of }s(c)\}
\tag{1.20a}
\]

be the finite live-prefix set.  For a prefix \(u\), define the exact
completion-safe token set

\[
\Omega_{\rm comp}(u)
=
\{x:u\mathbin{\Vert}x\in\mathcal P_{\rm comp}\},
\qquad
u\in\mathcal P_{\rm comp}^{\circ}
:=
\mathcal P_{\rm comp}\setminus\{s(c):c\in\mathcal C_{\rm DNG}\}.
\tag{1.20b}
\]

The token vocabulary contains the bounded \(K\), element labels, positive
integer coefficients, and EOS in their frozen typed positions.  Equivalently,
\(\mathcal P_{\rm comp}\) is a finite prefix automaton whose accepted leaves
are exactly the canonical formulas in \(\mathcal C_{\rm DNG}\).  It may be
constructed by an exact dynamic program rather than explicit enumeration, but
the accepted language must be identical.  No next-token distribution is
defined at a complete EOS leaf; sampling stops there.

For finite logits \(r_\omega(x;u)\) computed from the causal context
\([\mathrm{BOS},u]\), define

\[
p_\omega(x\mid u)
=
\frac{
\mathbf1[x\in\Omega_{\rm comp}(u)]
e^{r_\omega(x;u)}
}{
\sum_{y\in\Omega_{\rm comp}(u)}e^{r_\omega(y;u)}
},
\qquad
p_\omega(c)=\prod_{j=1}^{|s(c)|}p_\omega(s_j(c)\mid s_{<j}(c)).
\tag{1.20c}
\]

For \(S=\{Z_1<\cdots<Z_K\}=\operatorname{supp}(c)\), let
\(a(S)=(K,Z_1,\ldots,Z_K)\) be the complete support prefix.  Because the
serialization is support-first, the same autoregressive law factors exactly:

\[
p_\omega(c)
=
p_\omega(S)\,p_\omega(c\mid S),
\qquad
p_\omega(S)
=
\prod_{j=1}^{K+1}p_\omega(s_j(c)\mid s_{<j}(c)),
\qquad
p_\omega(c\mid S)
=
\prod_{j=K+2}^{2K+2}p_\omega(s_j(c)\mid s_{<j}(c)).
\tag{1.20d}
\]

The conditional continuation in (1.20d) is normalized on exactly
\(\mathcal C(S)\): supplying \(S\) means clamping the deterministic prefix
\(a(S)\), then sampling only positive coefficients and EOS with the same
masked logits.  It is therefore the true conditional of the joint
autoregressive law, not a second network or a rejection/repair heuristic.
The disjoint support-prefix subtrees partition the formula leaves, hence

\[
\sum_{S\in\mathfrak S_{\rm DNG}}p_\omega(S)=1,
\qquad
p_\omega(S)>0,
\qquad
\sum_{c\in\mathcal C(S)}p_\omega(c\mid S)=1.
\]

Thus \(p_\omega\) is a direct normalized law over canonical reduced formulas;
it never generates an unreduced formula and then repairs it.  In particular,
\(\nu\) remains a downstream cover variable.  Since \(\gcd(c)=1\), the
advertised conventional count vector \(N=\nu c\) uniquely satisfies
\(\nu=\gcd\{N_Z:N_Z>0\}\).

## 1.7 Bijective invariant lattice chart

Choose \(G_0\succ0\) satisfying

\[
WG_0W^{\mathsf T}=G_0
\quad\forall W\in P_g.
\tag{1.21}
\]

One construction is the positive group average

\[
G_0=\frac1{|P_g|}\sum_{W\in P_g}WHW^{\mathsf T},
\qquad H\succ0.
\tag{1.22}
\]

Let \(S=G_0^{1/2}\) and

\[
Q_W=S^{-1}WS.
\tag{1.23}
\]

Then \(Q_W\) is orthogonal.  Define the symmetric commutant

\[
\mathfrak p_g
=
\{A=A^{\mathsf T}:Q_WAQ_W^{\mathsf T}=A\ \forall W\in P_g\}.
\tag{1.24}
\]

For a basis \(E_1,\ldots,E_{d_g}\), define

\[
G_g(b)
=
S\exp\!\left(\sum_{j=1}^{d_g}b_jE_j\right)S,
\qquad b\in\mathbb R^{d_g}.
\tag{1.25}
\]

This chart is complete only for the invariant SPD cone in the pinned
fractional representation.  Alternative lattice bases still belong to the
quotient (1.5).

For the bounded generative domain, take the \(E_j\) Frobenius-orthonormal and use
the smooth map

\[
\kappa_g(z)
=
\frac{\mathcal B_g z}{\sqrt{1+\|z\|_2^2}},
\qquad z\in\mathbb R^{d_g}.
\]

Then \(b=\kappa_g(z)\) lies in the open coefficient ball
\(\|\sum_jb_jE_j\|_{\rm F}<\mathcal B_g\).  This gives a normalized unbounded latent base
while every decoded metric remains inside the declared bounded raw-cover
computational domain.

The production chart (`loewner-v2`, Theorem 2.3b) replaces the radial squash by
the spectral logistic map

\[
W(Z)=l\,I+(u-l)\,\sigma\!\left(Z-\ln c\,I\right),
\qquad
c=\frac{u-1}{1-l},
\qquad
G_g^{\rm L}(z)=S\,W\Big(\sum_jz_jE_j\Big)\,S,
\tag{1.25a}
\]

a smooth bijection of \(\mathbb R^{d_g}\) onto the open Loewner interval
\(\{lG_0\prec G\prec uG_0\}\) of the invariant cone, with \(W(0)=I\).  Its
shortest-vector floor \(\sigma_{\min}(L)\ge\sqrt{\lambda_{\min}(G_0)\,l}\) depends on
\(l\) alone, so a long axis is admitted through \(u\) without lowering the floor.
The bounded coordinate handed to the field is
\(b^{\rm L}(z)=\operatorname{coeff}\log W\), whose spectrum lies in
\((\ln l,\ln u)\).  Wherever \(b=\kappa_g(z)\) appears below, read
\(b=b^{\rm L}(z)\) under `loewner-v2`; the ball chart is the documented legacy
option.  In both charts the latent \(z\) is unbounded and the Gaussian base of
(1.30a) is a Gaussian in it: under `loewner-v2` that is a logit coordinate of
the whitened spectrum.

## 1.8 Intrinsic Wyckoff coordinates and decoder

All catalogue translations and intrinsic affine data are exact rationals.
For each explicit Wyckoff position choose

\[
u_i(\xi_i)
=
a_{w_i}+\xi_iA_{w_i}pmod{\mathbb Z^3},
\qquad
A_{w_i}\in\mathbb Q^{d_i\times3},
\tag{1.26}
\]

where \(a_{w_i}\in\mathbb Q^3\), \(A_{w_i}\) has full row rank, and the catalogue
proves that the intrinsic parameter lives on

\[
\mathbb R^{d_i}/\Lambda_{w_i},
\qquad
\Lambda_{w_i}
=
\{\lambda:\lambda A_{w_i}\in\mathbb Z^3\}.
\tag{1.27}
\]

with \(\Lambda_{w_i}\) a full-rank lattice in \(\mathbb R^{d_i}\).  Hence the
quotient is a compact torus with normalized Haar measure.  The pairs
\((W_{ik},w_{ik})\) represent the **right cosets**
\(H_i\backslash\overline G_g^{\rm conv}\), where \(H_i\) is the representative's
site stabilizer modulo conventional lattice translations.  This coset side is
forced by the row-coordinate right action
\(f\cdot(W,w)=fW+w\): composition is defined by
\((f\cdot g)\cdot a=f\cdot(ga)\), and applying \(a\) sends
\(H_ig\mapsto H_iga\).  Below, \(A_i:=A_{w_i}\).

For coset representatives \((W_{ik},w_{ik})\), expand

\[
x_{ik}
=
u_i(\xi_i)W_{ik}+w_{ik}pmod{\mathbb Z^3},
\qquad
k=1,\ldots,\mu_i.
\tag{1.28}
\]

The continuous-origin quotient (1.14) has an exact coordinate presentation

\[
\mathbb T_\alpha^{\rm int}
\cong
\mathbb R^{d_\alpha}/\Lambda_\alpha.
\]

Write its coordinate as \(\eta\).  Each representative parameter is recovered
by an exact affine map \(\xi_i=\Xi_i(\eta)\) in the chosen gauge.  This removes
continuous origin directions before a positive-definite positional metric or
flow is defined.

All members of one orbit receive one species.  Use the frozen smooth
right-handed section \(L=\operatorname{chol}(G_g(b))\), the lower-triangular
Cholesky factor with strictly positive diagonal, so
\(LL^{\mathsf T}=G_g(b)\).  This gives a measurable, smooth conventional
Cartesian presentation; denote this exact expanded object by
\(D_\alpha^{\rm conv}(q)=(L,F,Z)\).  Its infinite periodic extension is

\[
\operatorname{Per}(D_\alpha^{\rm conv}(q))
=
\{((f_i+n)L,Z_i):i=1,\ldots,N,\ n\in\mathbb Z^3\}.
\]

For this decoded lattice, the selected infinite Euclidean realization is

\[
\mathscr G_{g,L}^{\rm Euc}
=
\left\{
x\mapsto xL^{-1}WL+(w+n)L:
(W,w)\in\overline G_g^{\rm conv},\ n\in\mathbb Z^3
\right\}.
\]

Equation (1.4) makes every linear part orthogonal.  The physical decoder
\(\bar D_\alpha\) first reduces the known
centering subgroup \(T_g\), then finds and quotients every additional
species-preserving translation, and finally returns the primitive infinite
coloured point set in the quotient of section 1.2.

The independent verifier must check:

1. exact element counts;
2. declared orbit multiplicities;
3. noncollision using a complete closest-vector calculation;
4. the complete pure-translation subgroup, including whether it strictly
   contains the declared centering subgroup \(T_g\); and
5. the actual coloured stabilizer (1.7).

An output with enhanced stabilizer but unchanged advertised orbit
multiplicities, or with additional translations, is assigned its actual
post-decode metadata.  A true Wyckoff specialization that collapses an
advertised orbit belongs to \(\Sigma_\alpha\) instead.  Neither case invalidates
normalization because the latter has zero branch probability under the stated
assumptions.

## 1.9 Overlapping-cover probability measure

Let \(\mathcal A_{\rm arith}(c)\) be the finite set of arithmetic-legal raw
covers.  For every \(\alpha\), let
\(\rho_{\theta,\alpha}(\cdot\mid c)\) be a normalized probability measure on
\(\widehat Q_\alpha\), absolutely continuous with respect to its product volume.
Adjoin an isolated failure point \(\dagger\) and set
\(\mathscr X_c^\dagger=\mathscr X_c\sqcup\{\dagger\}\).  Extend each physical
decoder measurably by

\[
\bar D_\alpha^\dagger(q)
=
\begin{cases}
\bar D_\alpha(q),&q\notin\Sigma_\alpha,\\
\dagger,&q\in\Sigma_\alpha.
\end{cases}
\]

Let \(\Theta=(\omega,\phi,\theta)\) collect the composition-generator,
selector, and continuous-flow parameters.  For every measurable
\(A\subseteq\mathscr X_c^\dagger\), define the composition-conditioned law

\[
\boxed{
P_\Theta(A\mid c)
=
\sum_{\alpha\in\mathcal A_{\rm arith}(c)}
\pi_\phi(\alpha\mid c)
\int_{\widehat Q_\alpha}
\mathbf1_A\!\left(\bar D_\alpha^\dagger(q)\right)
\rho_{\theta,\alpha}(dq\mid c).
}
\tag{1.29}
\]

For de-novo generation, form the finite disjoint union

\[
\widetilde{\mathscr X}_{\rm DNG}^\dagger
=
\bigsqcup_{c\in\mathcal C_{\rm DNG}}\mathscr X_c^\dagger
\]

and, for measurable \(B\subseteq\widetilde{\mathscr X}_{\rm DNG}^\dagger\),
write \(B_c=\{C:(c,C)\in B\}\) and define the tagged joint law

\[
\boxed{
\widetilde P_\Theta^{\rm DNG}(B)
=
\sum_{c\in\mathcal C_{\rm DNG}}
p_\omega(c)
P_\Theta(B_c\mid c).
}
\tag{1.29a}
\]

Here \(P_\Theta(B_c\mid c)\) denotes the conditional measure in (1.29), not a
density.  Define the total measurable failure-collapsing map

\[
J^\dagger(c,C)=C,
\qquad
J^\dagger(c,\dagger)=\dagger,
\]

with codomain
\(\mathscr X_{\rm phys,DNG}^\dagger
=\mathscr X_{\rm phys,DNG}\sqcup\{\dagger\}\).  The untagged physical DNG law is

\[
P_\Theta^{\rm DNG}
=
J^\dagger_\#\widetilde P_\Theta^{\rm DNG}.
\tag{1.29b}
\]

For a supplied exact chemical system \(S\in\mathfrak S_{\rm DNG}\), define
the tagged system-conditioned law

\[
\widetilde P_\Theta^{\rm S\text{-}DNG}(B\mid S)
=
\sum_{c\in\mathcal C(S)}
p_\omega(c\mid S)P_\Theta(B_c\mid c),
\qquad
P_\Theta^{\rm S\text{-}DNG}(\cdot\mid S)
=
J^\dagger_\#\widetilde P_\Theta^{\rm S\text{-}DNG}(\cdot\mid S).
\tag{1.29c}
\]

Equation (1.29) is used when \(c\) is supplied for CSP.  Equation (1.29c)
clamps the exact support prefix and samples stoichiometry for S-DNG.  Equation
(1.29a) samples both support and stoichiometry for unconditional DNG.  All
three use exactly the same conditional cover, geometry, and decoder kernels;
once \(c\) is realized, neither \(S\) nor a task-mode token is passed
downstream.  Replacing \(c\) by a null token would not define the arithmetic
support and is not this model.

On the tagged DNG space define the measurable support projection
\(\Pi(c,C)=\operatorname{supp}(c)\), including tagged failure states.  Since
\(p_\omega(S)>0\), the S-DNG law is exactly the finite-event conditional of
the same tagged joint law:

\[
\widetilde P_\Theta^{\rm S\text{-}DNG}(B\mid S)
=
\widetilde P_\Theta^{\rm DNG}(B\mid\Pi=S)
=
\frac{
\widetilde P_\Theta^{\rm DNG}
\!\left(B\cap\Pi^{-1}\{S\}\right)
}{
p_\omega(S)
}.
\tag{1.29d}
\]
Since the measurable reduced-composition map satisfies \(r(C)=c\) almost
surely, (1.29b) loses no physical composition identity; only the multiple
tagged failure atoms are identified with the one global failure symbol.

Proposition 2.4b will imply \(P_\Theta(\{\dagger\}\mid c)=0\); hence the
restriction of (1.29) to \(\mathscr X_c\) is again a probability measure.  The
model support is

\[
\mathcal X_c^{\rm model}
=
\bigcup_{\alpha\in\mathcal A_{\rm arith}(c)}
\bar D_\alpha(\widehat Q_\alpha\setminus\Sigma_\alpha).
\]

Different covers may have different dimensions and overlapping images.
Therefore \(\pi_\phi(\alpha\mid c)\) is a latent proposal-path probability,
**not** the probability of an actual maximal space group or physical scaffold.
For any physical event \(E\), including an actual maximal group,

\[
P_\Theta(E\mid c)
=
\sum_\alpha\pi_\phi(\alpha\mid c)
\rho_{\theta,\alpha}
\{q:\bar D_\alpha^\dagger(q)\in E\}.
\]

An exact physical likelihood, if a suitable dominating measure is declared,
would have to integrate every raw-cover fibre with the appropriate
coarea/Jacobian and quotient multiplicity factors.  A finite mixture always
admits an artificial dominating measure, for example the sum of its component
pushforwards, but there is generally no canonical ambient-volume or Lebesgue
density across different dimensions.  Version 1 therefore claims normalized
generation, not a unique physical-density decomposition.

## 1.10 Intrinsic conditional flow

For a fixed raw cover \(\alpha\), the flow is defined on the full smooth product
completion \(\widehat Q_\alpha\).  It may cross specialization or collision sets;
therefore the vector field itself must possess a smooth extension there.  The
base and generated endpoints avoid \(\Sigma_\alpha\) almost surely when the flow
is diffeomorphic, because null sets remain null.
In DNG, \(c\) and \(\alpha\) are sampled before this stage and remain fixed
throughout the continuous trajectory; no flow crosses composition or cover
dimensions.

The continuous state is

\[
q=(z,\eta),
\tag{1.30}
\]

The unsymmetrized reference base and its residual-group average are

\[
\rho_{0,\alpha}^{\rm raw}
=
\mathcal N(0,I_{d_g})
\otimes
\operatorname{Haar}(\mathbb R^{d_\alpha}/\Lambda_\alpha),
\tag{1.30a}
\]

\[
\rho_{0,\alpha}
=
\frac1{|\Gamma_\alpha^{\rm res}|}
\sum_{\gamma\in\Gamma_\alpha^{\rm res}}
(\gamma)_\#\rho_{0,\alpha}^{\rm raw}.
\tag{1.30b}
\]

Repeated identical orbits receive exchangeable noise and no instance-index
embedding.  Equation (1.30b) makes residual invariance exact even when the
action on \(z\) is not orthogonal.

Under the global-existence assumptions of F9, let
\(\Phi_{t,s}^{\theta,\alpha,c}\) be the unique flow of the final symmetrized
cover-conditioned vector field.  The continuous model measure used in
equations (1.29) and (3.1) is, by definition,

\[
\rho_{\theta,\alpha}(\cdot\mid c)
:=
(\Phi_{1,0}^{\theta,\alpha,c})_\#\rho_{0,\alpha}.
\tag{1.30c}
\]

Conditional flow matching trains this endpoint toward the full-support
dequantized data law below; it does not introduce a separate endpoint
probability head.

The target law on each raw cover is required to have a strictly positive smooth
density with respect to its product volume.  A finite training set is atomic
and does not satisfy this assumption.  Therefore a future implementation uses
the predeclared symmetry-preserving full-support kernel

\[
K_\sigma(dz,d\eta\mid q_{\rm data},\alpha)
=
\mathcal N(z;z_{\rm data},\sigma_z^2I)\,dz
\otimes
k_{\sigma_\eta}^{\rm heat}
(\eta-\eta_{\rm data})\,d\operatorname{Haar}(\eta),
\qquad \sigma>0,
\]

symmetrized over repeated-orbit permutations and any explicitly retained finite
representative action.  The torus heat kernel is strictly positive.  Hence the
dequantized endpoint and Gaussian--Haar base are equivalent to cover volume.

Two torus kernels are implemented (`CFMDequantization.intrinsic_kernel`).  The
legacy `fractional-v1` kernel is \(k^{\rm heat}_{\sigma_\eta}\) with the isotropic
covariance \(\sigma_\eta^2I\) in the intrinsic coordinate \(\eta\); in space its
displacement is anisotropic by the cell's axial ratios (0.09 Å in plane against
0.9 Å along \(c\) for a 3 × 3 × 30 Å cell).  The `cartesian-v2` kernel is the
torus heat kernel with the record-dependent covariance

\[
\Sigma_\alpha(q_{\rm data})
=
\sigma^2\big(\widehat J_\alpha^{\mathsf T}P_N\widehat J_\alpha\big)^{-1}
\Big|_{q_{\rm data}},
\tag{1.30d}
\]

the inverse of the translation-quotient coordinate metric of section 1.11 at the
record, so that \(\operatorname{tr}(M\Sigma)=\sigma^2d_\alpha\): \(\sigma\) is the
root-mean-square centred Cartesian displacement per free coordinate in units of
\(\ell_0\), the same in every cell.  Both kernels are strictly positive on the
torus, so the equivalence above holds for either; the exact Fourier oracle of
F10 evaluates the per-record covariance.

**Basis augmentation** (`shear-v1`, review 2026-10-29 C2).  Theorem 2.3 does
not quotient \(GL(3,\mathbb Z)\), and preprocessing fixes one canonical
presentation (section 1.2), so for \(P1\), \(P\bar1\) and the monoclinic settings
the target has a sharp boundary inside the support of the base.  With a declared
probability \(p\) a record is moved to a neighbouring presentation: for an
elementary shear \(W=I\pm E_{ij}\) such that \((W,0)\) normalizes \(G\) exactly
and maps every row of \(\alpha\) onto itself,

\[
q_{\rm data}=(z,\eta)\ \longmapsto\
\big(z(W^{-1}GW^{-\mathsf T}),\ \eta_W(\eta)\big),
\tag{1.30e}
\]

the same crystal in the basis \(x\mapsto xW\), with \(\eta_W\) the exact affine
row map of section 1.5 and the image accepted only inside the chart's interval.
The endpoint law becomes the mixture
\((1-p_\alpha(q))\,K_\sigma(\cdot\mid q)+p_\alpha(q)\,|S_\alpha(q)|^{-1}\sum_W
K_\sigma(\cdot\mid W\!\cdot q)\) over the admissible in-support shears
\(S_\alpha(q)\); it is strictly positive wherever the kernel is, Theorem 2.9b
applies to any endpoint mixture, and a sample generated in a non-canonical basis
decodes to the same physical crystal, which the primitive reduction of section
1.2 canonicalizes.  `none` makes no draw and is the legacy law.
The learned endpoint is this dequantized law, not the literal empirical
measure.  Preprocessing assigns each datum to the raw cover defined by its
actual stabilizer.  A lower-symmetry cover generally fails to preserve the full
target stabilizer under dequantization, although unavoidable-symmetry
exceptions exist; actual-stabilizer assignment is the frozen convention.

For the training coupling, draw
\(\pi\sim\operatorname{Unif}(\Pi_\alpha)\) by independently shuffling each
identical-orbit block and
\(\gamma\sim\operatorname{Unif}(\Gamma_\alpha^{\rm res})\), then set
\(\widetilde q_1=\gamma(\pi q_1)=(\widetilde z_1,\widetilde\eta_1)\).  The
composition order \(\gamma\circ\pi\) is frozen.  Because the endpoint law is
invariant under both actions and the residual action normalizes \(\Pi_\alpha\),
this stochastic gauge lift changes the coupling but not either endpoint
marginal.  It requires no permutation optimization or arbitrary matching cost.
Use

\[
z_t=(1-t)z_0+t\widetilde z_1,
\qquad
b_t=\kappa_g(z_t)\ \text{or}\ b^{\rm L}(z_t)\ \text{(1.25a)}.
\tag{1.31}
\]

The path is linear in the unbounded latent \(z\) of whichever chart is
configured; under `loewner-v2` that is the logit coordinate of Theorem 2.3b.

Choose an exact row-basis matrix \(B_\alpha^\Lambda\) for the intrinsic period
lattice,

\[
\Lambda_\alpha
=
\{mB_\alpha^\Lambda:m\in\mathbb Z^{d_\alpha}\}.
\]

For the gauge-reduced intrinsic coordinate, use the half-open fundamental-cell
lift

\[
a=(\widetilde\eta_1-\eta_0)(B_\alpha^\Lambda)^{-1},
\qquad
\Delta\eta
=
\operatorname{wrap}_{[-1/2,1/2)}(a)B_\alpha^\Lambda,
\tag{1.32}
\]

and set

\[
\eta_t=\eta_0+t\Delta\eta\pmod{\Lambda_\alpha}.
\tag{1.33}
\]

where wrapping is coordinatewise with the deterministic half-open tie rule.
This is a measurable topological coupling on the exact intrinsic torus.  It is
not a componentwise minimum-image rule in fractional Cartesian coordinates,
nor is it claimed to minimize Cartesian length or to be a geodesic of the fully
coupled lattice-coordinate manifold.  The physically meaningful metric enters
the regression loss below, not the lift selection.

The affine reconstruction \(\xi_i=\Xi_i(\eta)\) makes the lifted velocity legal
exactly, not merely modulo an integer.  If an integer-valued continuous
function has the form \(k+tm\) for every real \(t\in[0,1]\), then \(m=0\).

## 1.11 Fixed-cell Cartesian block metric

At fixed lattice, an intrinsic perturbation gives

\[
\delta r_{ik}=\delta\xi_iA_iW_{ik}L.
\tag{1.34}
\]

Using (1.4),

\[
\sum_{k=1}^{\mu_i}\|\delta r_{ik}\|_2^2
=
\delta\xi_i
\underbrace{\mu_iA_iGA_i^{\mathsf T}}_{W_i(G)}
\delta\xi_i^{\mathsf T}.
\tag{1.35}
\]

Thus \(W_i(G)\) is the exact **pre-origin-quotient**, fixed-cell pullback for one
orbit-coordinate block.  Let \(J_\alpha^{\rm cart}(G)\) map a
gauge-reduced velocity \(\delta\eta\) to the stacked Cartesian displacement of
all \(N\) expanded atoms, and define

\[
P_N=
\left(I_N-\frac1N\mathbf1\mathbf1^{\mathsf T}\right)
\otimes I_3.
\]

The exact fixed-cell, translation-quotient pullback is

\[
W_\alpha^{\rm quot}(G)
=
(J_\alpha^{\rm cart})^{\mathsf T}
P_NJ_\alpha^{\rm cart}.
\]

Equivalently, its squared norm is
\(\min_{t\in\mathbb R^3}\sum_a\|\delta r_a-t\|_2^2\).  It is positive definite
on the **entire advertised-slot cover** under the tangent-kernel certificate
(1.15a).  The fractional advertised-slot Jacobian is constant in \(\eta\), and
multiplication by an invertible lattice factor cannot change its
translation-quotient rank.  Collisions or specializations change coordinate
values but do not deduplicate the advertised slots inside this smooth cover.

Fix one physical reference length \(\ell_0\), for example
\(1\,\text{\AA}\), and define the dimensionless exact training weight

\[
\widehat W_\alpha^{\rm quot}(G)
=
\ell_0^{-2}W_\alpha^{\rm quot}(G).
\tag{1.35a}
\]

This makes the coordinate and affine-invariant lattice terms dimensionless and
leaves the objective unchanged under a mere change of length units.  It is not
the complete joint pullback when lattice and coordinates change together,
because the Cartesian differential then contains lattice-coordinate cross
terms.

For the lattice block, use the affine-invariant SPD metric

\[
g_G(U,V)=\operatorname{tr}(G^{-1}UG^{-1}V),
\tag{1.36}
\]

pulled back through \(G_g\circ\kappa_g\) to the positive matrix

\[
[H_z(z)]_{jk}
=
\operatorname{tr}\!\left(
G^{-1}\frac{\partial G}{\partial z_j}
G^{-1}\frac{\partial G}{\partial z_k}
\right).
\]

The conditional flow-matching loss is

\[
\mathcal L_{\rm CFM}
=
\mathbb E\!\left[
(v_z-\dot z)^{\mathsf T}H_z(z_t)(v_z-\dot z)
+
\frac1N
(v_\eta-\Delta\eta)^{\mathsf T}
\widehat W_\alpha^{\rm quot}(G_t)
(v_\eta-\Delta\eta)
\right].
\tag{1.37}
\]

The weights are evaluated at the current state.  Equation (1.37) is a
co-moving block metric, not an energy, force field, or complete physical metric
on joint crystal space.

## 1.12 Periodic geometric field

The simplest shared field is an exact expand--perceive--reduce map:

\[
q
\xrightarrow{E_\alpha}
F_{\rm full}
\xrightarrow{T_\theta}
V_{\rm full}
\xrightarrow{R_{\alpha,G}}
v_q.
\tag{1.38}
\]

\(E_\alpha\) expands all symmetry orbits.  In the primary candidate,
\(T_\theta\) has atom-permutation-equivariant, \(SO(3)\)-invariant learned
scalar channels and a deterministic \(SO(3)\)-covariant Cartesian
readout.  Thus the complete \(T_\theta\) remains equivariant without learned
vector hidden channels.  \(R_{\alpha,G}\) is the exact smooth
translation-quotient metric adjoint whose output lies in the legal intrinsic
Wyckoff tangent space.  Equations (6.10)--(6.13a) specify the implemented dense all-pairs
construction.

The following cutoff-complete image-resolved construction is the retained
`equivariant_local` ablation, not the default field.  It remains an admissible
comparison object and its independent completeness obligations still apply
when selected.

A cutoff-complete, image-resolved periodic edge first uses the same frozen
length \(\ell_0\) to form dimensionless displacements and squared distances.
Regular solid harmonics are homogeneous polynomials and remain defined at a
collision:

\[
d_{abn}=(f_b-f_a+n)L,
\qquad
\widehat d_{abn}=d_{abn}/\ell_0,
\qquad
\widehat s_{abn}=\|\widehat d_{abn}\|_2^2,
\qquad
e_{abn}
=
\left[
\varphi_p(\widehat s_{abn}),
\mathcal S_{\ell m}(\widehat d_{abn})
\right]_{p,\ell,m}.
\tag{1.39}
\]

All noncentral images inside the cutoff must be enumerated using a certified
cell-dependent bound; \((a=b,n=0)\) is excluded from the geometric edge set,
while \((a=b,n\ne0)\) is retained.  A fixed image cube is not exact for arbitrary
skew cells.  Put \(\widehat r_c=r_c/\ell_0\).  Let
\(\chi\in C^1([0,\infty))\) be a nonnegative compact cutoff with
\(\chi(\widehat r_c^2)=\chi'(\widehat r_c^2)=0\).  Gate the **entire unnormalized edge weight**, not
merely an input feature:

\[
w_{abn}
=
\chi(\widehat s_{abn})
\exp\!\left(q_a^{\mathsf T}k_b/\sqrt d+\psi_{\rm edge}(e_{abn}^{(0)})\right),
\qquad
a_{abn}
=
\frac{w_{abn}}{w_a^{\rm self}+\sum_{b',n'}w_{ab'n'}},
\tag{1.40}
\]

with a persistent uniform bound \(w_a^{\rm self}\ge\epsilon_{\rm self}>0\) and
\(a_a^{\rm self}=w_a^{\rm self}/[w_a^{\rm self}+\sum_{b,n}w_{abn}]\).
The complete message is

\[
m_a
=
a_a^{\rm self}V H_a
+
\sum_{b,n}a_{abn}
\left[VH_b+\Phi(e_{abn},H_b)\right].
\]

Here \(\Phi\) is \(SO(3)\)-equivariant, uses regular tensor features, and produces
channels including a dimensionless proper-rotation vector measured in units of
\(\ell_0\); no \(O(3)\) parity is claimed.  All neural branches are bounded
and \(C^1\) on the closed bounded feature domain.  Since the whole edge weight
and its first derivative vanish at the cutoff, adding or removing an image does
not make the field discontinuous.  This is neither an injective description of all periodic
geometry nor a long-range interaction.  The periodic-attention form is
established prior art.  Its role here is to parameterize the
quotient-compatible vector field, not to serve as a novelty claim.
