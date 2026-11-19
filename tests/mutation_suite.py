# ruff: noqa: E501  -- entries hold verbatim source snippets and pytest node ids.
"""Executable counterpart of the theory 4.9 mandatory mutation suite (review m5).

Each entry is one deliberately introduced defect, written as an exact textual
source replacement, together with the frozen theory obligations that must kill
it and the concrete tests that embody them.  ``scripts/run_mutation_suite.py``
applies every executable entry to a temporary copy of ``src/``.  It requires
the killing tests to pass on the unmutated copy and to fail on the mutated
copy.  ``tests/test_mutation_suite_manifest.py`` keeps the manifest from
rotting: every snippet must occur exactly once, and every killing test must
exist.

An entry with ``status="obligation-open"`` is a defect whose killing
obligation is not implemented yet.  It is listed and reported as a gap, never
counted as a kill.

This module is not collected by pytest (no ``test_`` prefix).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Mutation:
    number: int
    defect: str
    obligations: str
    path: str = ""
    find: str = ""
    replace: str = ""
    killers: tuple[str, ...] = ()
    status: str = "executable"
    note: str = ""


MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        number=1,
        defect="transpose the metric identity away from W G W^T = G",
        obligations="C1, L2",
        path="src/crystalatlas/lattice.py",
        find=(
            "                rotation @ self.reference_metric @ rotation.T,\n"
            "                self.reference_metric,"
        ),
        replace=(
            "                rotation.T @ self.reference_metric @ rotation,\n"
            "                self.reference_metric,"
        ),
        killers=(
            "tests/test_lattice.py::test_l1_chart_dimension_matches_the_crystal_system_for_all_230_settings",
        ),
        note=(
            "Whitened rotations are orthogonal, so the commutant itself cannot see the "
            "transpose; the fractional reference-metric check is where W^T G W differs."
        ),
    ),
    Mutation(
        number=2,
        defect="admit a determinant-minus-one basis gauge under the chirality-preserving target",
        obligations="C4, L4, F6b",
        path="src/crystalatlas/primitive_canonicalization.py",
        find="                if _determinant(basis) != 1:",
        replace="                if abs(_determinant(basis)) != 1:",
        killers=(
            "tests/test_standardization.py::test_primitive_canonical_section_compares_different_metrics_and_keeps_mirrors",
        ),
    ),
    Mutation(
        number=3,
        defect="omit declared centering or complete primitive reduction",
        obligations="C3, W3, W4, S1",
        path="src/crystalatlas/decoder.py",
        find="        if translated == coloured:\n            accepted.append(translation)",
        replace=(
            "        if translated == coloured and not any(translation):\n"
            "            accepted.append(translation)"
        ),
        killers=(
            "tests/test_wyckoff_decoder.py::ExactDecoderTests::test_hidden_translation_is_detected_without_changing_reduced_formula",
            "tests/test_geometry.py::test_continuous_decoder_reports_hidden_translation_index",
        ),
    ),
    Mutation(
        number=5,
        defect="replace full-support dequantization by a truncated kernel",
        obligations="F1, F8b, Proposition 2.9c oracle",
        path="src/crystalatlas/flow.py",
        find=(
            "        eta_noise = torch.randn(\n"
            "            eta_data.shape,\n"
            "            dtype=eta_data.dtype,\n"
            "            device=eta_data.device,\n"
            "            generator=generator,\n"
            "        )"
        ),
        replace=(
            "        eta_noise = torch.randn(\n"
            "            eta_data.shape,\n"
            "            dtype=eta_data.dtype,\n"
            "            device=eta_data.device,\n"
            "            generator=generator,\n"
            "        ).clamp(-2.0, 2.0)"
        ),
        killers=(
            "tests/test_flow_oracle.py::test_dequantization_kernel_has_the_full_gaussian_support",
        ),
        note=(
            "The F8b law test alone does not kill a 2-sigma clamp at sigma_eta = 0.125: "
            "the moment shift (~0.005) lies inside its 5-SE window.  The kernel test checks "
            "tail masses directly."
        ),
    ),
    Mutation(
        number=15,
        defect="weight CFM by the unknown endpoint metric",
        obligations="F8 analytic counterexample",
        path="src/crystalatlas/flow.py",
        find="    lattice_term = _quadratic(z_error, current_lattice_metric)",
        replace=(
            "    lattice_term = _quadratic(z_error, current_lattice_metric) * (\n"
            "        1.0 + target.z.square().sum(dim=-1)\n"
            "    )"
        ),
        killers=(
            "tests/test_flow_population_target.py::test_current_state_weighting_returns_the_conditional_mean",
        ),
        note="The weight now depends on the target velocity u = q_1 - q_0, i.e. on the endpoint.",
    ),
    Mutation(
        number=4,
        defect="include a branch whose Sigma_alpha is the whole cover",
        obligations="D1b, G1",
        status="obligation-open",
        note=(
            "No table-wide proper-null (D1b) certificate is implemented; the theory ledger "
            "lists it as open.  There is therefore no killing test to run."
        ),
    ),
    Mutation(
        number=6,
        defect="dequantize a cubic datum in a P1 cover",
        obligations="S1, actual-stabilizer cover-assignment test",
        path="src/crystalatlas/ingestion.py",
        find="        setting = self.catalogue.get(numerical.international_number)",
        replace=(
            "        setting = self.catalogue.get(\n"
            "            1 if numerical.international_number >= 195 else numerical.international_number\n"
            "        )"
        ),
        killers=(
            "tests/test_catalogue_assignment.py::test_centered_fm3m_is_end_to_end_canonical_target",
        ),
    ),
    Mutation(
        number=7,
        defect="impose primitive atom count only by post-filtering",
        obligations="S3, primitive-count contract test",
        path="src/crystalatlas/nn/cover_sampling.py",
        find=(
            "            sample = self._sample_once(\n"
            "                composition,\n"
            "                temperature=temperature,\n"
            "                generator=generator,\n"
            "                device=device,\n"
            "                condition=condition,\n"
            "            )\n"
            "            if condition is None:\n"
            "                return sample\n"
        ),
        replace=(
            "            sample = self._sample_once(\n"
            "                composition,\n"
            "                temperature=temperature,\n"
            "                generator=generator,\n"
            "                device=device,\n"
            "                condition=None,\n"
            "            )\n"
            "            if condition is None:\n"
            "                return sample\n"
            "            if not condition.admits(\n"
            "                nu=sample.cover.nu,\n"
            "                formula_atoms=composition.total_atoms,\n"
            "                centering_index=self.catalogue.get(\n"
            "                    sample.cover.space_group\n"
            "                ).centering_index,\n"
            "            ):\n"
            "                continue\n"
        ),
        killers=(
            "tests/test_primitive_count_conditioning.py::test_primitive_count_enters_the_masked_law_not_a_post_filter",
        ),
        note="Draws then satisfy the count, but from the post-filtered law, not the masked law.",
    ),
    Mutation(
        number=8,
        defect="report raw pi_alpha as actual maximal-group mass",
        obligations="D4, E1",
        status="obligation-open",
        note=(
            "No actual-maximal-group classifier or event-probability report exists, so "
            "there is no code to mutate and no D4/E1 test; both obligations are open."
        ),
    ),
    Mutation(
        number=9,
        defect="replace the summed pushforward event probability by a maximum over covers",
        obligations="D4 tiny overlapping-cover oracle",
        status="obligation-open",
        note="The D4 tiny overlapping-cover oracle is not implemented.",
    ),
    Mutation(
        number=10,
        defect=(
            "allow repeated element labels, nonincreasing element order, gcd greater than "
            "one, or an out-of-domain composition in the DNG prefix tree"
        ),
        obligations="D0a-D0b composition canonicalization/prefix oracle",
        path="src/crystalatlas/composition.py",
        find="        return int(prefix_gcd == 1)",
        replace="        return 1",
        killers=(
            "tests/test_composition.py::CompositionGrammarTests::test_language_equals_independent_reduced_formula_enumeration",
        ),
        note="The gcd-greater-than-one variant: completions with a common divisor are counted.",
    ),
    Mutation(
        number=11,
        defect="feed held-out formulas to the purported de-novo sampler",
        obligations="S3 DNG input-provenance trace",
        path="src/crystalatlas/pipeline.py",
        find="            if self.species is not None or self.composition is not None:",
        replace="            if self.species is not None:",
        killers=(
            "tests/test_pipeline.py::PipelineTests::test_dng_receives_neither_a_formula_nor_a_species_set",
        ),
    ),
    Mutation(
        number=12,
        defect="leave a softmax edge/value ungated at the cutoff",
        obligations="W5, F6 cutoff value/Jacobian test",
        path="src/crystalatlas/nn/equivariant_transformer.py",
        find="        cutoff = smooth_whole_edge_cutoff(distance_squared, cutoff_squared)\n",
        replace="        cutoff = torch.ones_like(distance_squared)\n",
        killers=(
            "tests/test_equivariant_transformer.py::test_soft_normalized_local_transformer_is_c1_across_the_cutoff",
            "tests/test_equivariant_transformer.py::test_production_local_field_is_continuous_only_with_soft_normalization",
        ),
        note=(
            "Killed under the opt-in soft-v2 normalization.  The legacy rms-v1 "
            "normalization is itself discontinuous at r_c (strict xfail), so it cannot "
            "witness this defect."
        ),
    ),
    Mutation(
        number=13,
        defect="use d/|d| at a collision",
        obligations="F6 collision-limit test",
        path="src/crystalatlas/nn/invariant_transformer.py",
        find=(
            "    return vectors / torch.sqrt(\n"
            "        vectors.square().sum(dim=-1, keepdim=True) + epsilon\n"
            "    )"
        ),
        replace="    return vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)",
        killers=(
            "tests/test_invariant_transformer.py::test_collision_forward_and_coordinate_lattice_gradients_are_finite",
            "tests/test_invariant_transformer.py::test_smooth_covariant_normalization_has_the_declared_collision_jacobian_bound",
        ),
    ),
    Mutation(
        number=14,
        defect="use a fixed periodic-image cube",
        obligations="W5, F7 skew-cell exhaustive oracle",
        path="src/crystalatlas/nn/equivariant_transformer.py",
        find="    component_bound = cutoff * torch.linalg.vector_norm(inverse, dim=0)",
        replace="    component_bound = torch.ones(3, dtype=inverse.dtype, device=inverse.device)",
        killers=(
            "tests/test_equivariant_transformer.py::test_periodic_enumeration_is_complete_for_a_skew_cell",
            "tests/test_equivariant_transformer.py::test_periodic_enumeration_is_complete_for_a_strongly_skewed_cell",
        ),
        note=(
            "The mildly skewed oracle needs only |n_j| <= 2, which a fixed cube still "
            "covers; the strongly skewed oracle needs |n_j| = 4."
        ),
    ),
    Mutation(
        number=16,
        defect="add an orbit-instance index embedding",
        obligations="F1, F4 permutation test",
        path="src/crystalatlas/nn/bridge.py",
        find="                float(row.dimension == 0),",
        replace="                float(row.dimension == 0) + 0.25 * index,",
        killers=(
            "tests/test_nn_bridge.py::test_field_is_covariant_to_identical_p1_anchor_exchange",
        ),
    ),
    Mutation(
        number=17,
        defect="silently truncate an image list or resource-heavy sample",
        obligations="F7 resource-bound test",
        path="src/crystalatlas/nn/equivariant_transformer.py",
        find=(
            "            if accepted_edge_count > max_edges:\n"
            "                raise PeriodicImageBudgetError(\n"
            '                    "periodic edge list exceeds max_edges; no edge was silently truncated"\n'
            "                )"
        ),
        replace=(
            "            if accepted_edge_count > max_edges:\n"
            "                keep = torch.nonzero(accepted).flatten()[\n"
            "                    : count - (accepted_edge_count - max_edges)\n"
            "                ]\n"
            "                accepted = torch.zeros_like(accepted)\n"
            "                accepted[keep] = True\n"
            "                accepted_edge_count = max_edges"
        ),
        killers=(
            "tests/test_equivariant_transformer.py::test_edge_budget_overflow_fails_closed_instead_of_truncating",
        ),
    ),
    Mutation(
        number=18,
        defect="retry generation until a validity check passes",
        obligations="D4, E3 invalid-as-failure accounting",
        path="src/crystalatlas/learned.py",
        find="                    decoded = self.model.layout(item.cover_sample.cover).decode(\n",
        replace=(
            "                    layout = self.model.layout(item.cover_sample.cover)\n"
            "                    try:\n"
            "                        layout.decode(\n"
            "                            geometry,\n"
            "                            collision_tolerance=(\n"
            "                                self.model.scientific_config.numerics.collision_atol_angstrom\n"
            "                            ),\n"
            "                            symmetry_tolerance=(\n"
            "                                self.model.scientific_config.numerics.symmetry_atol\n"
            "                            ),\n"
            "                        )\n"
            "                    except GeneratedSampleError:\n"
            "                        pass\n"
            "                    decoded = layout.decode(\n"
        ),
        killers=(
            "tests/test_learned.py::test_a_decode_failure_keeps_its_slot_and_is_never_retried",
        ),
        note="A failed decode is silently retried, so an invalid first draw can be reported as a success.",
    ),
    Mutation(
        number=19,
        defect=(
            "mix conventional Wyckoff multiplicities with primitive counts in one "
            "Diophantine equation"
        ),
        obligations="C3, D1, W3",
        path="src/crystalatlas/cover.py",
        find="        WyckoffRowSpec(row.label, row.multiplicity, row.dimension)",
        replace=(
            "        WyckoffRowSpec(row.label, row.multiplicity // setting.centering_index, "
            "row.dimension)"
        ),
        killers=(
            "tests/test_primitive_atom_bound.py::test_conventional_multiplicities_fix_the_rocksalt_formula_units",
        ),
    ),
    Mutation(
        number=20,
        defect=(
            "delete one authoritative Wyckoff row or pinned setting while leaving all "
            "remaining stored records internally consistent"
        ),
        obligations="C2 authoritative-key bijection, G1",
        path="src/crystalatlas/full_catalogue.py",
        find="    for source_row in source.rows:",
        replace="    for source_row in source.rows[:-1] if international_number == 225 else source.rows:",
        killers=(
            "tests/test_full_catalogue.py::test_complete_catalogue_passes_exact_constructor_certification",
        ),
        note="Setting and expected rows stay mutually consistent; only the frozen manifest can tell.",
    ),
    Mutation(
        number=21,
        defect=(
            "give DNG, S-DNG, and CSP separate downstream selector/flow weights or a hidden "
            "task/species-set token after composition is fixed"
        ),
        obligations="D0b shared-path conformance, S3",
        path="src/crystalatlas/learned.py",
        find="                generator=self._torch_generator(int(seeds[2]), device),\n",
        replace=(
            "                generator=self._torch_generator(int(seeds[2]), device),\n"
            "                temperature=0.5 if request.mode is GenerationMode.DNG else 1.0,\n"
        ),
        killers=(
            "tests/test_learned.py::test_three_modes_share_the_downstream_path_after_composition",
        ),
    ),
    Mutation(
        number=22,
        defect="allow a null or absent composition to enter the DNG scaffold selector",
        obligations="D0b null-composition rejection, S3",
        path="src/crystalatlas/nn/cover_sampling.py",
        find="        if not isinstance(composition, Composition):\n",
        replace="        if composition is not None and not isinstance(composition, Composition):\n",
        killers=(
            "tests/test_nn_cover_sampling.py::test_cover_sampler_rejects_an_absent_composition",
        ),
    ),
    Mutation(
        number=23,
        defect=(
            "use an interleaved element/count sequence, force future element tokens, and "
            "report that suffix as the exact conditional p(c | S)"
        ),
        obligations="D0a-D0b support-prefix conditional identity",
        path="src/crystalatlas/composition.py",
        find="            CompositionToken.element(z) for z in ordered\n",
        replace="            CompositionToken.element(z) for z in ordered[:1]\n",
        killers=(
            "tests/test_composition.py::CompositionGrammarTests::test_clamped_support_is_exact_sdng_fibre",
        ),
        note="The clamped prefix forces only the first element token of S.",
    ),
    Mutation(
        number=24,
        defect=(
            "pass an unused allowed element or task-mode token to the downstream cover "
            "selector or continuous field after c is fixed"
        ),
        obligations="D0b three-mode shared-path conformance, S3",
        path="src/crystalatlas/learned.py",
        find="                generator=self._torch_generator(int(seeds[2]), device),\n",
        replace=(
            "                generator=self._torch_generator(\n"
            "                    int(seeds[2]) + len(request.species or ()), device\n"
            "                ),\n"
        ),
        killers=(
            "tests/test_learned.py::test_three_modes_share_the_downstream_path_after_composition",
        ),
        note="The S-DNG species-set size leaks into the downstream random stream.",
    ),
)


MUTATIONS = tuple(sorted(MUTATIONS, key=lambda mutation: mutation.number))
