# 3. Architecture implied by the theory

This is a mathematical interface specification, not code.  Every learned
module exists only where the preceding theorems leave a statistical choice.
Crystallographic identities remain deterministic.

## 3.1 End-to-end factorization

For a supplied reduced composition \(c\), the CSP kernel is

\[
P_\Theta(dC\mid c)
=
\sum_{\alpha\in\mathcal A_{\rm arith}(c)}
\pi_\phi(\alpha\mid c)
(\bar D_\alpha^\dagger)_\#
\rho_{\theta,\alpha}(dq\mid c).
\tag{3.1}
\]

The first-class de-novo law adds only the canonical composition generator:

\[
P_\Theta^{\rm DNG}(dC)
=
\sum_{c\in\mathcal C_{\rm DNG}}
p_\omega(c)P_\Theta(dC\mid c),
\qquad
\Theta=(\omega,\phi,\theta).
\tag{3.1a}
\]

Equation (3.1a) is the untagged physical marginal (1.29b); its tagged joint
counterpart \(\widetilde P_\Theta^{\rm DNG}(dc,dC)\) retains the sampled
composition explicitly.

For a supplied exact chemical system \(S\), the same model gives

\[
P_\Theta^{\rm S\text{-}DNG}(dC\mid S)
=
\sum_{c\in\mathcal C(S)}
p_\omega(c\mid S)P_\Theta(dC\mid c).
\tag{3.1b}
\]

The parameter blocks \(\omega\), \(\phi\), and \(\theta\) respectively belong
to the composition generator, discrete cover selector, and continuous geometry
model.  The block \(\omega\) samples both support and stoichiometry in DNG,
only stoichiometry in S-DNG, and is bypassed when \(c\) is supplied for CSP.
Equation (3.1) is first a law on \(\mathscr X_c^\dagger\); Theorem 2.5 proves
that its failure-point mass is zero under the stated assumptions, so its
physical restriction has total mass one.

The corresponding shared computation is

```text
optional support generation (DNG only)
    -> optional stoichiometry generation (DNG and S-DNG)
    -> composition
    -> normalized arithmetic-completion selector
    -> overlapping symmetry-containment cover
    -> exchangeable intrinsic orbit state
    -> shared periodic conditional flow
    -> exact decoder
    -> independent canonical verifier
```

No second symbolic denoiser, parent-phase hierarchy, ACE descriptor, Mamba
sequence, energy model, or post-hoc symmetry projector is part of version 1.
Composition is generated explicitly before the arithmetic selector; replacing
it by a null condition cannot define either de-novo mode because the selector's
legal support depends on the realized formula.

## 3.2 Preprocessing contract

Before learning, every structure must be transformed into the exact tuple

\[
(c,\alpha,q)
=
(c,\nu,g,h,z,\eta).
\tag{3.2}
\]

The pipeline must:

1. determine the full primitive translation lattice;
2. determine the complete species-preserving stabilizer in that primitive
   basis;
3. classify its physical space-group type and select that type's own pinned
   Hall realization;
4. apply the corresponding exact primitive-to-conventional transformation;
5. decompose the atoms into stabilizer orbits and identify explicit Wyckoff
   positions and intrinsic parameters;
6. quotient continuous origin freedom, use the pinned right-handed catalogue
   convention, and store repeated identical orbits as an unordered block;
7. encode the invariant metric with the inverse of
   \(G_g\circ\kappa_g\); and
8. verify that \(\bar D_\alpha(q)\) returns the original physical
   crystal.

The symmetry finder, version, standardization policy, and tolerance are part of
the dataset identity.  For finite-precision structural databases, preprocessing
must be repeated over a declared tolerance sweep.  Structures whose labels
change materially are reported as symmetry-ambiguous rather than being treated
as exact ground truth at one arbitrary tolerance.

The main v1 model assigns each idealized structure to the cover defined by its
**actual** complete stabilizer.  Gaussian--heat-kernel dequantization then stays
inside that symmetry-constrained cover.  A lower-symmetry cover does not in
general preserve the datum's full stabilizer under dequantization, although
exceptional cases such as a one-atom P1 cover retain unavoidable inversion.
Actual-stabilizer assignment is therefore the declared training convention; a later
experimental-observation model may marginalize label uncertainty; that is not
silently inserted into this theory.

## 3.3 Canonical composition generator and composition encoders

For DNG, use one causal Transformer to generate the unique flat typed token
sequence

\[
[K,Z_1,\ldots,Z_K,c_{Z_1},\ldots,c_{Z_K},\mathrm{EOS}],
\qquad Z_1<\cdots<Z_K.
\]

The external BOS symbol initializes the empty prefix and is never scored.  For
each nonterminal generated prefix \(u\), the Transformer supplies next-token
logits only:

\[
H_\omega(u)
=
\operatorname{CausalTr}_\omega([E_{\rm BOS},E_u],E_{\rm type})_{\rm last},
\qquad
r_\omega(x;u)
=
\operatorname{MLP}_\omega(H_\omega(u),E_x),
\tag{3.3a}
\]

where \(E_{\rm type}\) distinguishes the \(K\), element, coefficient, and EOS
token roles.  The exact live-prefix oracle \(\Omega_{\rm comp}(u)\), not the
Transformer, defines legal support.  It enforces the element/count bounds,
strict element order, gcd-one reduction, and EOS placement.  The constructive
\(P1\) witness in Theorem 2.2c proves downstream arithmetic support for every
accepted formula.  The target token is never included in the context that
produces its own logit.  Because all element tokens precede every coefficient,
the complete prefix \(a(S)=[K,Z_1,\ldots,Z_K]\) gives the exact chain-rule
factorization

\[
p_\omega(c)=p_\omega(S)p_\omega(c\mid S).
\tag{3.3b}
\]

Unconditional DNG samples the entire string.  S-DNG validates and sorts the
supplied exact \(S\), clamps \(a(S)\), and samples only its coefficient/EOS
continuation.  CSP canonicalizes the supplied formula and bypasses the whole
composition Transformer.  Once \(c\) is realized, all modes use the same
encoders and every downstream module; neither \(S\) nor a task token is passed
further.  A small marginal \(p_\omega(S)\) makes that system rare under
unconditional DNG but does not penalize S-DNG: clamping the prefix samples the
renormalized continuation \(p_\omega(c\mid S)\).

The input is an unordered collection of element/count pairs.  A simple pair of
permutation-invariant encoders is

\[
e_c^{\rm sel}
=
\rho_\phi^{\rm sel}\!\left(
\sum_{Z:c_Z>0}
\psi_\phi^{\rm sel}(e_Z^{\rm sel},\log(1+c_Z))
\right),
\qquad
e_c^{\rm flow}
=
\rho_\theta^{\rm flow}\!\left(
\sum_{Z:c_Z>0}
\psi_\theta^{\rm flow}(e_Z^{\rm flow},\log(1+c_Z))
\right),
\tag{3.3}
\]

where \(e_Z^{\rm sel}\) and \(e_Z^{\rm flow}\) each concatenate a separate
learned element embedding with the same optional fixed elemental descriptors.
Only the set invariance is a theorem-level requirement; the descriptor choice
is empirical.  The two encoders share no learned weights in version 1, so the
conditional parameter split \((\phi,\theta)\) and gradient ownership are exact;
the separate composition-generator block is \(\omega\).

The reduced-composition model receives neither the target space group nor a
target cell atom count at inference.  Counts supplied in the same pinned
conventional convention can replace \(\nu\); primitive-count-only conditioning
is outside version 1, as declared in section 0.2.

The composition Transformer, selector encoder, and flow encoder are kept
parameter-separate in the proof-first specification.  Sharing their element
embeddings or reusing the composition EOS representation is an empirical
ablation, not part of the probability factorization.

## 3.4 Arithmetic-completion-certified cover selector

Use one of the 230 physical space-group types, mapped deterministically to one
pinned Hall realization.  The normalized raw-path selector is

\[
\pi_\phi(\alpha\mid c)
=
p_\phi(\nu\mid c)
p_\phi(g\mid\nu,c)
\prod_{r=1}^{R_g}
p_\phi(a_r\mid a_{<r},g,\nu,c),
\tag{3.4}
\]

with every support restricted by the arithmetic suffix table \(B\) and equation
(1.19).  The catalogue admissibility obligation in sections 1.5 and 4.3 is a
frozen domain assumption for every arithmetic terminal table; it is not a learned or
sample-time rejection step.

Version 1 uses one small causal Transformer for these logits.  With a fixed
catalogue row order, its row context and action score are

\[
H_r
=
\operatorname{CausalTr}_\phi
\left(e_c^{\rm sel},E_\nu,E_g,E_{a_{<r}},E_{g,r}^{\rm row}\right),
\qquad
\ell_{\phi,r}(a)
=
\operatorname{MLP}_\phi
\left(H_r,E_{g,r}^{\rm row},E^{\rm count}(a)\right).
\]

Here the row embedding is derived from its multiplicity, site-symmetry
operations, and free-coordinate dimension; it is not merely an arbitrary
letter ID.  Those three cannot tell Pm-3m 12h \((x,\tfrac12,0)\) from 12i
\((0,y,y)\), which are different crystal chemistry, so the implementation can
add the exact chart of the row's normalizer-canonical representative (its
offset and coordinate matrix) together with the canonical and own row indices
(`selector_row_geometry_features`, review 2026-10-29 M1): rows equivalent under
the full Euclidean normalizer share the chart, inequivalent rows differ, and
the index reproduces the data's canonical choice among equivalent rows.  One explicit permutation-invariant count embedding is

\[
E^{\rm count}(a)
=
\sum_{Z:c_Z>0}\chi_\phi(e_Z^{\rm sel},a_Z).
\]

Separate masked categorical heads produce \(\nu\) and \(g\).  The Transformer
supplies scores only: the exact suffix table, not the network, defines legal
support.

The row action \(a_r\) is itself an unordered element-count vector.  If its
joint action space is large, replace the joint row conditional by a nested
species-count autoregression in a fixed element order while augmenting the
feasibility state with the remaining row capacity.  This is a different local
parameterization on the same legal table support, but it remains exactly
normalized when every nested conditional uses its completion-safe mask.

For the deterministic preprocessing assignment
\(\alpha^\star=\operatorname{Enc}(C^\star)\), the exact joint-label discrete loss is

\[
\mathcal L_{\rm cover}
=
-\log\pi_\phi(\alpha^\star\mid c)
\tag{3.5}
\]

If preprocessing deliberately retains a finite set
\(\mathcal L_{\rm eq}(C^\star)\) of exactly equivalent catalogue labels, the label-set
loss may instead be

\[
\mathcal L_{\rm cover}
=
-\log
\sum_{\alpha\in\mathcal L_{\rm eq}(C^\star)}
\pi_\phi(\alpha\mid c)
\tag{3.6}
\]

These are likelihoods for the declared cover-label target.  They are not the
physical crystal likelihood, because several overlapping covers and continuous
fibres can decode to the same crystal and may have different dimensions.  The normalized generator needs no
maximal-locus predicate, quotient-class head, or connected-component variable.

## 3.5 Intrinsic state constructor

Once a raw cover \(\alpha\) is selected, instantiate one token for each
independent orbit.
The continuous state is

\[
q=(z,\eta),
\qquad
z\in\mathbb R^{d_g},
\quad
\eta\in\mathbb R^{d_\alpha}/\Lambda_\alpha.
\tag{3.7}
\]

An orbit token contains:

- species;
- multiplicity;
- operation-derived site-symmetry descriptors;
- free-coordinate dimension and active-coordinate mask;
- character Fourier features of the reconstructed intrinsic torus point: if
  \(B_i^\Lambda\) is a row basis of \(\Lambda_i\), set
  \(\vartheta_i=\xi_i(B_i^\Lambda)^{-1}\pmod{\mathbb Z^{d_i}}\) and use
  \(\sin(2\pi k\!\cdot\!\vartheta_i)\) and
  \(\cos(2\pi k\!\cdot\!\vartheta_i)\) for integer modes \(k\); the retained
  mode set is closed under the finite residual action;
- flow composition context \(e_c^{\rm flow}\), time, and the current bounded lattice coordinate
  \(b_t=\kappa_g(z_t)\), or \(b_t=b^{\rm L}(z_t)\) under the production `loewner-v2`
  chart (1.25a).

It contains no arbitrary instance index.  Tokens for repeated identical
\((w,Z)\) orbits share all categorical embeddings.  Distinction arises only
from their current exchangeable coordinates.

Define

\[
\rho_{0,\alpha}^{\rm raw}
=
\mathcal N(0,I_{d_g})
\otimes
\operatorname{Haar}(\mathbb R^{d_\alpha}/\Lambda_\alpha).
\]

\(\operatorname{Haar}\) is normalized Haar probability on the compact intrinsic
torus.  The product base is already invariant under the repeated-orbit set
action \(\Pi_\alpha\).  Let \(\Gamma_\alpha^{\rm res}\) be the separately
bounded residual representative group from section 1.5; take it to be the
identity group when the catalogue proves a unique section.  The actual base is

\[
\rho_{0,\alpha}
=
\frac1{|\Gamma_\alpha^{\rm res}|}
\sum_{\gamma\in\Gamma_\alpha^{\rm res}}
(\gamma)_\#\rho_{0,\alpha}^{\rm raw}.
\tag{3.8}
\]

It is exactly \(\Pi_\alpha\)- and \(\Gamma_\alpha^{\rm res}\)-invariant whether
or not the residual action on \(z\) is orthogonal.  It remains full-support,
and \(b=\kappa_g(z)\), or \(b^{\rm L}(z)\) under `loewner-v2`, keeps every metric
inside the declared finite raw-cover domain.

## 3.6 Shared expand--perceive--reduce scalar-invariant Transformer

At every flow evaluation:

1. decode \(b_t\) and \(G_t\) from \(z_t\) by the configured chart: (1.25) with
   \(\kappa_g\), or the Loewner chart (1.25a) in production;
2. expand every intrinsic representative to the full conventional cell, or,
   under `geometry_expansion_cell="primitive-v2"` (review 2026-10-29, M3), to
   the primitive cell: one coset representative per class modulo the centring
   translations, lattice \(PL\) with \(P\) the centring reduction and
   coordinates \(xP^{-1}\).  Cartesian positions and Jacobians are the same
   geometry with \(N/I_c\) atoms, so in (3.9)--(3.10) \(N\) and \(P_N\) count the
   expansion cell's atoms, the conventional metric is
   \(\widehat W^{\rm quot}_{\rm conv}=I_c\,\widehat W^{\rm quot}_{\rm prim}\), and the
   \(1/N\) of (3.14) makes the coordinate loss identical.  The chart, the decoder
   and the kernel covariance (1.30d) keep the conventional cell;
3. construct exact fractional-torus Fourier pair features and deterministic
   dimensionless direct, reciprocal, direct--reciprocal cross-product, and
   pinned lattice-frame covariant carriers for all \(N^2\) atom pairs;
4. run a bidirectional dense scalar atom-set Transformer with no atom-index
   embeddings, neighbour graph, cutoff, sparse edge index, or scatter
   aggregation;
5. use invariant learned coefficients on deterministic polar and axial
   covariant carriers to predict a dimensionless \(SO(3)\) vector with no
   imposed \(O(3)\) parity,
   \(\widehat s_{ik}\in\mathbb R^3\), measured in units of \(\ell_0\), for every
   expanded atom, and predict an intrinsic lattice velocity \(v_z\) from pooled
   scalar channels;
6. reduce those vectors to intrinsic Wyckoff velocities by the exact smooth
   translation-quotient metric adjoint.

Stack all Cartesian outputs into \(\widehat s\in\mathbb R^{3N}\).  Let
\(J_\alpha^{\rm cart}\) be the exact Jacobian from the gauge-reduced intrinsic
velocity \(v_\eta\) to all expanded Cartesian velocities, and let
\(P_N=(I_N-\mathbf1\mathbf1^{\mathsf T}/N)\otimes I_3\).  The legal
translation-quotient intrinsic velocity is defined after scaling Cartesian
lengths by the frozen reference \(\ell_0\).  Put
\(\widehat J_\alpha^{\rm cart}=J_\alpha^{\rm cart}/\ell_0\).  Then

\[
v_\eta
=
\left[
(\widehat J_\alpha^{\rm cart})^{\mathsf T}
P_N\widehat J_\alpha^{\rm cart}
\right]^{-1}
(\widehat J_\alpha^{\rm cart})^{\mathsf T}P_N\widehat s.
\tag{3.9}
\]

Equation (3.9) is the unique minimizer of

\[
\min_{v,\tau\in\mathbb R^3}
\sum_{a=1}^{N}
\|[\widehat J_\alpha^{\rm cart}v]_a+\tau-\widehat s_a\|_2^2.
\tag{3.10}
\]

The tangent-kernel certificate (1.15a) makes the normal matrix positive
definite on the entire advertised-slot cover.  Because the affine fractional
Jacobian is constant and every \(L(G)\) is invertible, collisions and
specializations do not create a rank-defect wall in this redundant smooth
representation.  The output is always an exact legal intrinsic velocity; no
ridge hyperparameter is needed.

For this expand--reduce interface, a nonzero covariant Cartesian output basis is
essential; learned equivariant hidden vector channels are not.  Scalar tokens
alone cannot be mapped directly to a nonzero Cartesian polar vector while
guaranteeing rotation covariance.  The deterministic direct/reciprocal moments
and their cross products supply configuration-dependent covariants.  The three
rows of the pinned right-handed lattice section supply additional polar carriers
when every sine-odd pair carrier vanishes at an advertised-slot collision.  Every
learned hidden channel and coefficient remains an \(SO(3)\)-invariant scalar.

The model predicts \(v_z\in\mathbb R^{d_g}\) directly.  Therefore every
integrated lattice is decoded through the bounded SPD chart; no projection of
a raw matrix is necessary.  The vector field must be globally Lipschitz or
otherwise satisfy a proved global-existence condition on \([0,1]\).

Repeated-orbit permutations are enforced architecturally.  If the pinned
catalogue retains a finite within-cover representative action
\(\Gamma_\alpha^{\rm res}\), enforce it exactly by

\[
\bar v_\alpha(q)
=
\frac1{|\Gamma_\alpha^{\rm res}|}
\sum_{\gamma\in\Gamma_\alpha^{\rm res}}
d\gamma^{-1}_{\gamma q}\,
v_\alpha(\gamma q).
\]

This averaging is omitted only when the catalogue proves a unique section.
Denote the resulting shared, cover-conditioned intrinsic field by
\(\bar v_\theta(t,q,\alpha,c)\).  The quotient state \(q\) and reduced velocity
\(v_q\) have no Cartesian rotation index, so this intrinsic field is invariant
under a change of external orientation.  Its expanded Cartesian intermediate
must instead be \(SO(3)\)-covariant, so that reducing it gives the same intrinsic
velocity for every properly rotated presentation.  The current candidate admits signed
triple products in the scalar path and axial cross-product covariants in the
readout, so it is not forced to obey an \(O(3)\) parity law.  A transformer-level
right-handed mirror witness exercises the scalar and vector paths; it is not a
canonicalized all-pipeline P1 enantiomorph proof or an injectivity claim.  This design freedom is not a
claim that the physical law breaks parity.  A parity-typed \(O(3)\)-equivariant
readout can still keep enantiomeric inputs distinct and is a required ablation.

### Dense scalar-invariant periodic attention

The implemented dense construction and its covariance proof are equations
(6.10)--(6.13a).  Equation (6.15) is a future learned-moment ablation.  Pair geometry enters both the attention logits and values, so
identical input tokens do not erase geometry.  Full within-crystal attention is
\(O(N^2)\); the declared raw-cell limit \(N\le 40\) makes this a bounded design
choice.  It is free of cutoff topology and minimum-image discontinuities.  A
finite Fourier basis is not automatically injective, so reciprocal-radius and
harmonic convergence remain empirical ablations.

Self-attention still exchanges information between tokens.  The precise
architectural claim is removal of **sparse local graph message passing**, not
removal of all interatomic information aggregation.

### Legacy sparse equivariant ablation

The former cutoff graph is retained only as a checkpoint-incompatible,
explicitly selected `equivariant_local` ablation for the future parameter- and
measured-compute-matched experiment in
section 6.8.  It is not the default geometry backbone.  Its equations are
recorded below so the comparison remains reproducible.

For each expanded reference atom \(a\), source atom \(b\), and lattice translation
\(n\), define the image-resolved edge

\[
d_{abn}=(f_b-f_a+n)L_t,
\qquad
\widehat d_{abn}=d_{abn}/\ell_0,
\qquad
\widehat s_{abn}=\|\widehat d_{abn}\|_2^2,
\qquad
e_{abn}
=
\left[\varphi_p(\widehat s_{abn}),\mathcal S_{\ell m}(\widehat d_{abn})\right]_{p,\ell,m}.
\tag{3.11}
\]

Here \(\mathcal S_{\ell m}\) are regular solid harmonics, so no division by
\(\|d\|\) occurs at a collision.  Put \(\widehat r_c=r_c/\ell_0\) and enumerate
all \(n\) with \(\widehat s_{abn}<\widehat r_c^2\).  Exclude only the central geometric self-edge
\((a=b,n=0)\); retain nonzero periodic self-images.  A persistent ordinary
self-token path is required.  Choose a nonnegative compact cutoff
\(\chi\in C^1([0,\infty))\) with
\(\chi(\widehat r_c^2)=\chi'(\widehat r_c^2)=0\).

Let token channels transform in irreducible representations of \(SO(3)\).
Define the whole-edge unnormalized weight

\[
\omega_{abn}^{(h)}
=
\chi(\widehat s_{abn})
\exp\!\left[
q_a^{(h)\mathsf T}k_b^{(h)}/\sqrt{d_h}
+\psi_h^{\rm edge}(e_{abn}^{(0)})
\right],
\tag{3.12}
\]

and the equivariant tensor message

\[
m_a^{(h)}
=
\frac{
\omega_{a,{\rm self}}^{(h)}V_hH_a
+
\sum_{b,n}\omega_{abn}^{(h)}
\left[V_hH_b+\Phi_h(e_{abn},H_b)\right]
}{
\omega_{a,{\rm self}}^{(h)}
+
\sum_{b,n}\omega_{abn}^{(h)}
},
\qquad
\omega_{a,{\rm self}}^{(h)}\ge\epsilon_{\rm self}>0.
\tag{3.13}
\]

The regular tensor map obeys, for every proper rotation \(R\),

\[
\Phi_h(R\!\cdot\!e,R\!\cdot\!H)
=R\!\cdot\!\Phi_h(e,H).
\]

The \(\ell=1\) output channel supplies \(\widehat s_{ik}\); invariant \((\ell=0)\) pooled
channels supply \(v_z\).  All neural branches are bounded and \(C^1\) on the
closed bounded feature domain.  The complete edge weight and its first
derivative vanish at the cutoff, so dynamic-list values and first derivatives
agree.  The
directional value term prevents periodic geometry
from disappearing in a one-neighbour softmax.  This is a
**cutoff-complete image enumeration**, not complete or injective geometry and
not a long-range physical interaction.  Equations (3.11)--(3.13) are derived
here from the declared periodicity, cutoff smoothness, and proper-rotation
covariance requirements.  The prior-art audit identifies closely overlapping
equivariant and periodic-attention constructions, so these equations are not
claimed as independent inventions and no external implementation is a core
dependency.

## 3.7 Conditional flow-matching training

For each canonical datum \((\alpha,q_{\rm data})\) assigned by its actual
stabilizer:

1. sample a symmetry-preserving dequantized endpoint
   \(q_1\sim K_\sigma(\cdot\mid q_{\rm data},\alpha)\) using the Gaussian--torus
   heat kernel in section 1.10, with the `fractional-v1` or the Cartesian
   `cartesian-v2` torus covariance (1.30d);
2. sample \(q_0\sim\rho_{0,\alpha}\);
3. draw the uniform repeated-orbit shuffle and uniform bounded residual
   representative from section 1.10 after continuous-origin quotienting;
4. sample \(t\sim\operatorname{Unif}[0,1]\);
5. construct \(q_t\) and target velocity \(u_t\) from equations
   (1.31)--(1.33), then push forward the complete path and tangent by that
   representative: \(\eta_t\mapsto\operatorname{canon}_B(\eta_tT)\),
   \(u_\eta\mapsto u_\eta T\), without wrapping the velocity;
6. evaluate the shared field; and
7. minimize the current-state block loss

\[
\mathcal L_{\rm flow}
=
\mathbb E\!\left[
(v_z-u_z)^{\mathsf T}H_z(z_t)(v_z-u_z)
+
\frac1N
(v_\eta-u_\eta)^{\mathsf T}
\widehat W_\alpha^{\rm quot}(G_t)
(v_\eta-u_\eta)
\right].
\tag{3.14}
\]

The two blocks may carry explicit weights \(\lambda_z,\lambda_\eta>0\)
(`CFMBlockWeights`, review 2026-10-29 m2; default \(1,1\)): the coordinate
block is in \(\ell_0^2\) per atom and the lattice block is a dimensionless
squared strain, so their trade-off is a declared hyperparameter, recorded in
the objective contract.  They are current-state constants, hence Theorem 2.9
keeps the population minimizer.

For a certified finite residual group \(H\), step 5 is essential even when
the endpoint kernel has already been symmetrized. The half-open coordinate
lift need not commute with a nonorthogonal torus automorphism. If \(p_t,j_t\)
are the marginal density and probability current of an unaugmented coupling,
uniformly pushing forward the **whole path and tangent** gives
\(\bar p_t=|H|^{-1}\sum_T T_\#p_t\) and
\(\bar j_t=|H|^{-1}\sum_T T_\#j_t\), where current pushforward includes the
tangent action. Each pushed-forward continuity equation remains valid, so
\(\partial_t\bar p_t+\operatorname{div}\bar j_t=0\). Group closure makes
\(\bar p_t\) invariant and its conditional-mean velocity covariant. The
Haar base is unchanged; the terminal law is the desired averaged noisy
endpoint. The implementation applies this construction only to its certified
actions. It does not assert general-group certification from P1/P-1 witnesses.

The exact support and conditional-stoichiometry losses are

\[
\mathcal L_{\rm support}=-\log p_\omega(S),
\qquad
\mathcal L_{\rm stoich}=-\log p_\omega(c\mid S),
\qquad
\mathcal L_{\rm comp}
=\mathcal L_{\rm support}+\mathcal L_{\rm stoich}
=-\log p_\omega(c).
\tag{3.14a}
\]

On retained DNG records with \(c\in\mathcal C_{\rm DNG}\), the joint objective
is

\[
\mathcal L
=
\lambda_{\rm comp}\mathcal L_{\rm comp}
+\lambda_{\rm cover}\mathcal L_{\rm cover}
+\lambda_{\rm flow}\mathcal L_{\rm flow}.
\tag{3.15}
\]

For a CSP-only training run, omit \(\mathcal L_{\rm comp}\).  In a unified run,
only records in the frozen DNG domain train \(p_\omega\); additional
\(c\in\mathcal C_{\rm CSP}\setminus\mathcal C_{\rm DNG}\) records are explicitly
labelled auxiliary conditional data and train only the selector and flow.  A
strict joint-model experiment instead restricts all records to
\(\mathcal C_{\rm DNG}\).  For an exact joint MLE, the support and
stoichiometry terms use the same declared record/formula measure.  Fitting
\(p(S)\) record-weighted while fitting \(p(c\mid S)\) with a system/formula-
balanced measure defines a composite two-target search prior, not one joint
likelihood.  Record weighting and formula balancing change the outer
composition target, while polymorph balancing can also change the conditional
target \(P(C\mid c)\); every policy must be labelled explicitly.

A proposed **discovery-oriented S-DNG experiment** would use the frozen search
measure

\[
S\sim\operatorname{Unif}(\mathfrak S_{\rm train}),\qquad
c\sim\operatorname{Unif}(\mathcal C_{\rm obs}(S)),\qquad
C\sim\operatorname{Unif}(\mathcal P_{\rm obs}(c)),
\tag{3.15a}
\]

after exact structure, formula, and polymorph deduplication.  Here
\(\mathcal C_{\rm obs}(S)\) is the frozen set of observed reduced formulas and
\(\mathcal P_{\rm obs}(c)\) is the frozen set of observed polymorphs for that
formula. Since 2026-10-20 the training path implements this measure as the
opt-in `TrainingMeasure.SYSTEM_FORMULA_POLYMORPH` (review m6); no such trained
checkpoint is bundled. A separate
record-weighted checkpoint could estimate the retained database
law for unconditional density modelling. These are different statistical
targets; the first is a search-prior objective, not joint maximum likelihood.
Matched learned baselines receive exactly the same record weights.

The implemented alpha objective instead trains composition on uniform distinct
formulas, and conditional cover/geometry on uniform retained structures. In
joint training the composition term for a structure with formula \(c\) carries
\(N_s/(N_c n(c))\), preserving the uniform-formula marginal. Topology batching
uses a fixed nominal gradient denominator \(RA(N/B)\), with \(N,B\) the global
retained epoch record and microbatch counts. This prevents unequal bucket
sizes from silently changing that declared record measure. See
[the executable training contract](../docs/TRAINING_AND_DIAGNOSTICS.md).

Equations (3.14a) and (3.5) or (3.6) are exact discrete label negative
log-likelihoods.
Equation (3.14) is conditional-flow regression.  Their sum is a composite
training objective, not an exact negative log-likelihood of the physical
crystal unless a continuous change-of-variables likelihood is separately
computed with the quotient and decoder measures.

The endpoint of the continuous model is the declared full-support dequantized
raw-cover law.
Setting \(\sigma=0\) on a finite empirical dataset is incompatible with a
globally regular finite-time diffeomorphic flow and is not allowed under the
present theorem assumptions.

The metric weights in (3.14) use \(q_t\), not the target endpoint.  Theorem 2.9
then preserves the standard conditional-regression target, and Theorem 2.9b
connects that target to marginal transport under its regularity and global
existence assumptions.  Interpolation may cross a measure-zero enhancement or
collision locus in the redundant cover.  Proposition 2.4b makes singular
endpoints null under the exact flow assumptions.  The model does not target a
preselected maximal component.

## 3.8 Inference

For one ancestral sample:

1. in **DNG mode**, sample \(S\sim p_\omega(S)\), then
   \(c\sim p_\omega(c\mid S)\); in **S-DNG mode**, canonicalize and validate
   the supplied exact \(S\), clamp \(a(S)\), and sample only
   \(c\sim p_\omega(c\mid S)\); in **CSP mode**, canonicalize and validate the
   supplied \(c\) and bypass \(p_\omega\);
2. sample \(\nu\) and \(g\) from their feasible masked distributions;
3. sample every row action using the arithmetic completion table \(B\);
4. instantiate the unordered intrinsic state from (3.8);
5. integrate

   \[
   \dot q_t=\bar v_\theta(t,q_t,\alpha,c)
   \tag{3.16}
   \]

   from \(t=0\) to \(t=1\);
6. map \(z_1\) through \(\kappa_g\) and \(G_g\), reconstruct every
   \(\xi_i=\Xi_i(\eta_1)\), and expand the Hall/Wyckoff orbits with equations
   (1.25)--(1.28); and
7. apply \(\operatorname{CentRed}_g\) to remove the selected setting's declared
   centering copies, then apply \(\operatorname{FullPrim}\) to determine and
   quotient any additional species-preserving translations; independently
   determine formula, collisions, and actual maximal symmetry in the resulting
   primitive presentation.

All one-sample protocols return the raw result even when it fails a separate
chemical-validity diagnostic.  Retrying until valid changes the model
distribution and must be reported as rejection sampling.  A DNG run supplied
with held-out target formulas is instead an oracle-composition diagnostic, not
de-novo generation.  An S-DNG run must be labelled as chemical-system
conditioned and every compared method must receive the identical exact \(S\).

Under the regularity assumptions, the collision/multiplicity defect set has
probability zero.  A larger actual stabilizer or additional translation is a
valid overlapping-cover outcome and is recorded as metadata.  A numerical
collision or multiplicity collapse is a conformance event, not a sample to
repair silently.

For a fixed candidate budget \(K_{\rm cand}\), one may instead allocate candidates over
distinct raw covers.  Because cover images overlap, deduplication happens only
after full physical decoding.  This is a candidate-set protocol, not a
one-sample result.

For deliberate rare-stoichiometry exploration in S-DNG, an optional normalized
proposal is

\[
p_{\omega,\epsilon}(c\mid S)
=
(1-\epsilon)p_\omega(c\mid S)
+\epsilon\,\frac{\mathbf 1[c\in\mathcal C(S)]}{M_S},
\qquad
M_S=|\mathcal C(S)|,
\qquad 0\le\epsilon\le1.
\tag{3.16a}
\]

It guarantees \(p_{\omega,\epsilon}(c\mid S)\ge\epsilon/M_S\) for every
legal stoichiometry, so \(B\) independent coefficient draws hit any specified
\(c\in\mathcal C(S)\) with probability at least
\(1-(1-\epsilon/M_S)^B\).  This is a transparent exploration proposal, not
a thermodynamic prior and not a guarantee that the corresponding structure is
stable.  The exact ancestral reference sampler uses \(\epsilon=0\); any
nonzero value is reported and compared at the same total candidate budget.

It induces a separately named search law

\[
P_{\Theta,\epsilon}^{\rm S\text{-}DNG}(dC\mid S)
=
\sum_{c\in\mathcal C(S)}
p_{\omega,\epsilon}(c\mid S)P_\Theta(dC\mid c),
\qquad
P_{\Theta,0}^{\rm S\text{-}DNG}=P_\Theta^{\rm S\text{-}DNG}.
\]

For exact uniform sampling without materializing the complete fibre, compute
suffix leaf counts on the live subtree:

\[
m(u)
=
\begin{cases}
1,&u\text{ is an accepted leaf},\\
\displaystyle\sum_{x\in\Omega_{\rm comp}(u)}m(u\mathbin{\Vert}x),
&u\text{ is nonterminal},
\end{cases}
\]

and choose child \(x\) with probability
\(m(u\mathbin{\Vert}x)/m(u)\), starting at \(a(S)\).  The hit bound above
concerns \(B\in\mathbb N\) independent raw formula draws; it is not a bound
on stable structures or DFT discoveries.

### Optional fixed-budget discovery controller

The normalized ancestral law and the policy used to spend expensive
relaxation/DFT evaluations are different mathematical objects.  After drawing
a declared raw pool and applying the same inexpensive relaxation and scoring
pipeline to every compared generator, let \(\mathcal P_{\rm score}\) be the
deterministically ordered set of candidates with a finite surrogate result and
put \(K'=\min(K_{\rm DFT},|\mathcal P_{\rm score}|)\).  Define the batch
utility

\[
U(\mathcal A)
=
\sum_{x\in\mathcal A}
\left(
-\bigl(\widehat\mu_{\rm hull}(x)
       -\beta_{\rm unc}\widehat\sigma_{\rm hull}(x)\bigr)
+\lambda_{\rm nov}d_{\rm nov}(x,\mathcal D_{\rm nov}^{t_0})
\right)
+\lambda_{\rm div}
\log\det(I+\eta K_{\mathcal A}),
\tag{3.16b}
\]

and use the explicit greedy controller

\[
\mathcal A_0=\varnothing,qquad
x_j=\operatorname*{arg\,max}_{x\in
\mathcal P_{\rm score}\setminus\mathcal A_{j-1}}^{\prec}
\bigl[U(\mathcal A_{j-1}\cup\{x\})-U(\mathcal A_{j-1})\bigr],qquad
\mathcal A_j=\mathcal A_{j-1}\cup\{x_j\},quad 1\le j\le K'.
\tag{3.16c}
\]

The superscript \(\prec\) denotes a frozen canonical-hash tie rule.  A failed
surrogate is assigned no finite score, is retained in the all-attempt
denominator with its incurred cost, and is never silently replaced; if
\(K'<K_{\rm DFT}\), the missing submission slots are reported as failures.

Here \(K_{\mathcal A}\) is a frozen positive-semidefinite structural or
composition--structure similarity kernel normalized by \(K_{ii}=1\), and the
novelty corpus \(\mathcal D_{\rm nov}^{t_0}\) is the prospective corpus defined
in E3.  The surrogate supplies a calibrated mean and uncertainty.  The optimistic
lower-confidence term is an optional exploration rule and is ablated against
\(\beta_{\rm unc}=0\).  Freeze
\(\eta,\lambda_{\rm nov},\lambda_{\rm div},\beta_{\rm unc}\ge0\), so
\(I+\eta K_{\mathcal A}\succ0\).  The deterministic greedy rule is a declared
approximation, not a claim to solve cardinality-constrained log-determinant
maximization exactly.  This policy balances predicted low hull distance,
novelty, and nonredundancy; it is not part of
\(P_\Theta\), does not alter any normalization theorem, and makes no claim
that the surrogate score equals DFT.  Its training cutoff, model, kernel,
hyperparameters, pool size, and cost are preregistered.  Compare it with random
and lowest-predicted-energy selection.  Report two non-interchangeable tracks:
(i) one **common controller** applied to every generator, which attributes
differences to generation, and (ii) each **method-native controller** under the
same end-to-end resource budget, which compares discovery systems.  The count
\(K_{\rm DFT}\) is not the heterogeneous protocol budget
\(\mathbf B_{\rm cost}\) defined in E3.  Charge the controller's measured
runtime and memory to the protocol trace.  Report ancestral generation
separately so search cannot conceal a weak probability model.

## 3.9 Statistical target and database bias

Let \(K_{\rm obs}(dy\mid C,c)\) be a measurable Markov probability kernel
describing measurement, refinement, and record noise, so
\(K_{\rm obs}(\mathcal Y\mid C,c)=1\).  Let
\(s_{\rm obs}(C,c)\ge0\) describe selection into the database.  The observed
law is generally

\[
p_{\rm data}(dy\mid c)
=
\frac{
\int K_{\rm obs}(dy\mid C,c)\,
s_{\rm obs}(C,c)\,
p_{\rm physical}(dC\mid c)
}{
\int s_{\rm obs}(C,c)\,p_{\rm physical}(dC\mid c)
},
\tag{3.17}
\]

when the denominator is finite and nonzero.  The simpler multiplicative-bias
formula is recovered only for noiseless identity observations.  Database
frequency is therefore not automatically a thermodynamic polymorph probability.

Balancing polymorphs or compositions during training deliberately changes the
target distribution.  Such a sampling policy must be declared and evaluated
as a modelling choice; it is not a correction derived from the present
theory.

For DNG the observed joint law factors tautologically as

\[
p_{\rm data}(dc,dy)
=
p_{\rm data}^{\rm comp}(dc)\,p_{\rm data}(dy\mid c).
\tag{3.17a}
\]

Because \(S=\operatorname{supp}(c)\) is deterministic, the composition
marginal further factors as

\[
p_{\rm data}^{\rm comp}(c)
=
p_{\rm data}^{\rm sys}(S)\,
p_{\rm data}^{\rm stoich}(c\mid S).
\tag{3.17c}
\]

Here \(p_{\rm data}^{\rm comp}\) is the normalized retained or deliberately
reweighted marginal on the frozen \(\mathcal C_{\rm DNG}\); records outside
that domain do not train this factor.  Because finite logits give
\(p_\omega(c)>0\) on every accepted formula, the learned law is a full-support
approximation to a sparse empirical marginal, not an exact copy of empirical
zeros.  It is not a thermodynamic prevalence distribution over all possible
compositions.  In particular, training \(p(c\mid S)\) directly on structure
records weights formulas by their recorded polymorph counts.  A unique-formula
or system/formula-balanced search prior must therefore be declared separately
from record likelihood.

## 3.10 Complexity

Let \(C_{\rm build}^{\rm comp}\), \(M_{\rm oracle}^{\rm comp}\), and
\(C_{\rm load}^{\rm comp}\) be the one-time construction cost, stored memory,
and load cost of the exact canonical-formula prefix oracle.  Let
\(C_{\rm lookup}^{\rm comp}\) be one local successor lookup,
\(S_{\rm comp}\) the sampled composition-token length, \(d_{\rm comp}\) its
Transformer width, and \(H_{\rm comp}\) its layer count.  After setup, one DNG
composition draw costs

\[
O\!\left(
S_{\rm comp}C_{\rm lookup}^{\rm comp}
+H_{\rm comp}
[S_{\rm comp}d_{\rm comp}^2+S_{\rm comp}^2d_{\rm comp}]
+A_{\rm comp}
\right),
\tag{3.17b}
\]

where \(A_{\rm comp}\) is the total number of locally scored legal tokens.  A
materialized prefix tree can be combinatorial in the element and coefficient
bounds; no tractability claim is made until the exact oracle is benchmarked.
Report \(C_{\rm build}^{\rm comp}\), \(M_{\rm oracle}^{\rm comp}\),
\(C_{\rm load}^{\rm comp}\), and their amortization over a declared sample
count separately from (3.17b).
CSP bypasses this sampling cost.  Full DNG runtime is this outer cost plus the
downstream selector/flow cost averaged over \(c\sim p_\omega\).
S-DNG clamps the first \(|S|+1\) support tokens and pays only the coefficient/
EOS continuation cost plus the same downstream computation.

Let

\[
\mathcal P(c)
=
\left\{(\nu,g):1\le\nu\le\nu_{\max},\quad
\sum_Z\nu c_Z\le N_{\max}^{\rm raw}\right\}.
\]

For \(E\) elements present in the formula, define
\(S_\nu=\prod_{Z=1}^{E}(\nu c_Z+1)\) and
\(A_g=\sum_r|\mathcal H_{gr}|\).  Building every dense arithmetic table needed
to normalize \(p(\nu,g\mid c)\) costs

\[
O\!\left(
\sum_{(\nu,g)\in\mathcal P(c)}S_\nu A_g
\right).
\tag{3.18}
\]

Caching all tables uses
\(O(\sum_{(\nu,g)\in\mathcal P(c)}R_gS_\nu)\) memory; streaming needs the
largest such table but recomputes discarded suffixes.  This is
pseudo-polynomial, not polynomial in binary-encoded counts.  For selector
sequence length \(S_\alpha^{\rm sel}\), width \(d_{\rm sel}\), and
\(H_{\rm sel}\) cached causal-Transformer layers, neural evaluation adds

\[
O\!\left(H_{\rm sel}
[S_\alpha^{\rm sel}d_{\rm sel}^2
+(S_\alpha^{\rm sel})^2d_{\rm sel}]
+|\mathcal P(c)|\right),
\]

plus enumeration of each locally feasible action set.  The implementation must
measure this term rather than report only the dynamic program.

For \(N\) expanded atoms, scalar width \(d\), pair width \(p\), \(H\)
attention layers, \(M_K=(2K+1)^3\) reciprocal modes, \(B\) deterministic
covariant bases, \(C_v\) vector readout channels, and \(T\) ODE evaluations,
let \(d_\alpha\) be the intrinsic-coordinate dimension.  A direct Reynolds
implementation evaluates each of the \(|\Gamma_\alpha^{\rm res}|\) residual
images.  A conservative inference bound for the dense construction is

\[
O\!\left(
T|\Gamma_\alpha^{\rm res}|
\left[
N^2M_Kp+N^2p^2
+H\{Nd^2+N^2(d+p^2+pd)\}
+N^2d(d+p+C_vB)
+Nd_\alpha^2+d_\alpha^3
\right]
+C_{\rm FullPrim}
\right),
\tag{3.19}
\]

before batching.  The first two terms construct and embed dense Fourier pair
features; the third is dense attention and pair-value processing; the fourth
is the covariant coefficient readout.  The final two terms inside the brackets
form and solve the exact adjoint normal equations; \(C_{\rm FullPrim}\) is the
post-ODE exact primitive-reduction/stabilizer cost.  Batched group actions or
cached geometry may reduce wall time but not erase the declared work.  The
architecture is explicitly quadratic in \(N\), not linear.  Its deployment
contract therefore retains the finite \(N\le40\) raw-cell bound.  Report the
empirical NFE distribution \(T\), wall time, and peak memory.  With residual
actions accumulated sequentially, a single-sample inference working set is at
least \(O(Nd+N^2(M_K+p+d)+d_\alpha^2)\); training activations add the
layer/batch factors unless checkpointed.  The legacy local ablation instead
obeys the former \(E_{\rm img}\)-dependent bound from equations
(3.11)--(3.13).

## 3.11 Minimum ablations implied by the architecture

Once implementation is authorized, keep the data, sampler, parameter budget,
and evaluator fixed while testing:

1. support-first \(p(S)p(c\mid S)\) versus the former interleaved or an
   unconditional-then-filtered chemical-system generator;
2. exact completion-safe stoichiometry versus unconstrained/repaired formulas,
   and record-weighted versus unique-formula/system-balanced training;
3. learned \(p_\omega(c\mid S)\) versus the explicit exploration mixture
   (3.16a) at frozen \(\epsilon\);
4. generated compositions versus an oracle held-out composition marginal;
5. unconstrained scaffold scores versus arithmetic-completion-certified scores;
6. one canonical cover label versus finite equivalent-label marginalization;
7. arbitrary orbit ordering versus exchangeable orbit sets;
8. unconstrained lattice parameters versus the complete chart (1.25);
9. fractional MSE versus current-state Cartesian block weighting;
10. no geometry, logit-only geometry, and logit-plus-value geometry;
11. finite dense Fourier features versus the legacy cutoff-complete
    periodic-image features;
12. actual-stabilizer oracle cover versus predicted raw cover;
13. CFM versus diffusion with the identical Transformer, state, data,
    parameters, and matched NFE/compute curves;
14. the dense invariant-scalar/covariant-readout Transformer versus both the
    parameter- and compute-matched local equivariant graph and an AF3-style
    raw-coordinate non-equivariant Transformer; and
15. the fixed-budget controller (3.16b) versus random and
    lowest-predicted-energy selection.

For a fixed reference \(C^\star\) and evaluator tolerance \(\varepsilon\), the
model's exact one-sample match probability is

\[
P_\Theta(\operatorname{match}_\varepsilon C^\star\mid c^\star)
=
\sum_\alpha
\pi_\phi(\alpha\mid c^\star)
\int
\mathbf1\!\left[
q\notin\Sigma_\alpha,
\quad
d_{\mathscr X}(\bar D_\alpha(q),C^\star)\le\varepsilon
\right]
\rho_{\theta,\alpha}(dq\mid c^\star).
\tag{3.20}
\]

If the matcher requires exact reduced composition, \(C^\star\) has composition
\(c^\star\), and \(c^\star\in\mathcal C_{\rm DNG}\), the corresponding
unconditional one-sample DNG event factorizes as

\[
P_\Theta^{\rm DNG}(\operatorname{match}_\varepsilon C^\star)
=
p_\omega(c^\star)
P_\Theta(\operatorname{match}_\varepsilon C^\star\mid c^\star).
\tag{3.20a}
\]

For \(S^\star=\operatorname{supp}(c^\star)\), exact-system generation instead
has

\[
P_\Theta^{\rm S\text{-}DNG}
\!\left(\operatorname{match}_\varepsilon C^\star\mid S^\star\right)
=
p_\omega(c^\star\mid S^\star)
P_\Theta(\operatorname{match}_\varepsilon C^\star\mid c^\star).
\tag{3.20b}
\]

Under the optional exploration law, the corresponding identity is

\[
P_{\Theta,\epsilon}^{\rm S\text{-}DNG}
\!\left(\operatorname{match}_\varepsilon C^\star\mid S^\star\right)
=
p_{\omega,\epsilon}(c^\star\mid S^\star)
P_\Theta(\operatorname{match}_\varepsilon C^\star\mid c^\star).
\tag{3.20c}
\]

This identity exposes composition selection, cover selection, and geometry as
three distinct empirical bottlenecks.  It does not assign positive point mass
to \(C^\star\); the event is the evaluator's measurable tolerance set.

There is no selector-only hard ceiling in the overlapping-cover model: P1 and
other covers can represent the same target.  Reporting the terms of (3.20) by
raw cover is nevertheless a useful attribution diagnostic.  No extra module
should be added until the oracle-cover and predicted-cover tests locate the
actual bottleneck.
