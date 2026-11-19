"""Optional PyTorch neural components for CrystalAtlas."""

from .bridge import (
    CoverConditionedGeometryField,
    TorchCoverGeometry,
    TorchExpandedGeometry,
)
from .composition import (
    CanonicalCompositionTransformer,
    CompositionEncoder,
    CompositionTransformerConfig,
    apply_legal_mask,
)
from .cover_sampling import (
    NeuralCoverSample,
    NeuralCoverSampler,
    canonical_rows,
    operation_derived_row_features,
)
from .equivariant_transformer import (
    PeriodicEdges,
    PeriodicImageBudgetError,
    PeriodicSE3Transformer,
    PeriodicTransformerConfig,
    PeriodicTransformerOutput,
    VectorLinear,
    concatenate_periodic_edges,
    enumerate_packed_periodic_edges,
    enumerate_periodic_edges,
    oriented_triple_product,
    regular_solid_harmonics,
    smooth_whole_edge_cutoff,
)
from .field import (
    IntrinsicFieldConfig,
    IntrinsicFieldOutput,
    IntrinsicGeometryField,
    PackedIntrinsicFieldOutput,
    TangentRankError,
    reduce_cartesian_to_intrinsic,
    translation_quotient_normal_matrix,
)
from .invariant_transformer import (
    InvariantPeriodicTransformer,
    InvariantPeriodicTransformerConfig,
    InvariantPeriodicTransformerOutput,
)
from .sampling import sample_composition
from .selector import (
    ArithmeticCoverSelector,
    CountVectorEncoder,
    CoverSelectorConfig,
)

__all__ = [
    "ArithmeticCoverSelector",
    "CanonicalCompositionTransformer",
    "CompositionEncoder",
    "CompositionTransformerConfig",
    "CoverConditionedGeometryField",
    "NeuralCoverSample",
    "NeuralCoverSampler",
    "CountVectorEncoder",
    "CoverSelectorConfig",
    "IntrinsicFieldConfig",
    "IntrinsicFieldOutput",
    "IntrinsicGeometryField",
    "InvariantPeriodicTransformer",
    "InvariantPeriodicTransformerConfig",
    "InvariantPeriodicTransformerOutput",
    "PackedIntrinsicFieldOutput",
    "PeriodicEdges",
    "PeriodicImageBudgetError",
    "PeriodicSE3Transformer",
    "PeriodicTransformerConfig",
    "PeriodicTransformerOutput",
    "TangentRankError",
    "TorchCoverGeometry",
    "TorchExpandedGeometry",
    "VectorLinear",
    "apply_legal_mask",
    "canonical_rows",
    "concatenate_periodic_edges",
    "enumerate_packed_periodic_edges",
    "enumerate_periodic_edges",
    "oriented_triple_product",
    "operation_derived_row_features",
    "reduce_cartesian_to_intrinsic",
    "regular_solid_harmonics",
    "smooth_whole_edge_cutoff",
    "sample_composition",
    "translation_quotient_normal_matrix",
]
