from dataclasses import dataclass
from typing import Optional , Literal , Union
from torch.distributions import Distribution
import nflows.transforms as nflows_tf
import zuko.flows as zuko_flows
from torch import Tensor, nn

@dataclass(frozen=True)
class ZScoreConfig:
    """User-facing z-score intent for both parameters and observations.

    Specifies *how* each input should be standardised before being passed to
    a density estimator or classifier.  Used by the z-score resolver in
    to produce the appropriate``ZScoreContext``.

    Attributes:
        theta: Standardisation strategy for the parameter (``theta``) input.
        x: Standardisation strategy for the observation (``x``) input.
        x_dist: Reference distribution used when ``x='transform_to_unconstrained'``.
                Required if and only if ``x='transform_to_unconstrained'``.
    """
    theta: Optional[
        Literal["independent", "structured", "transform_to_unconstrained", "none"]
    ] = "independent"
    x: Optional[
        Literal["independent", "structured", "transform_to_unconstrained", "none"]
    ] = "independent"

    # Zuko unconstrained-transform support
    x_dist: Optional[Distribution] = None

    def __post_init__(self):
        if self.x == "transform_to_unconstrained" and self.x_dist is None:
            raise ValueError(
                "x_dist must be provided when x='transform_to_unconstrained'"
            )
        if self.x == "transform_to_unconstrained" and not hasattr(self.x_dist, "support"):
            raise ValueError(
                "`x_dist` requires a `.support` attribute for"
                "an unconstrained transformation."
            )
        if self.x != "transform_to_unconstrained" and self.x_dist is not None:
            raise ValueError(
                "x_dist should only be provided when x='transform_to_unconstrained'"
            )
    
@dataclass(frozen=True)
class TransformNormalization:
    """Normalisation backed by an explicit invertible transform.

    Used when the z-score resolver produces either an NFlows
    ``PointwiseAffineTransform`` or a Zuko ``UnconditionalTransform``.
    """

    transform: Union[
        nflows_tf.PointwiseAffineTransform,
        zuko_flows.UnconditionalTransform,
    ]


@dataclass(frozen=True)
class StatsNormalization:
    """Normalisation backed by precomputed mean and standard deviation tensors.

    Used for the standard affine z-score path (``'independent'`` /
    ``'structured'``) where statistics are estimated from the training batch
    and stored directly.
    """

    mean: Tensor
    std: Tensor


@dataclass(frozen=True)
class InputTransformNormalization:
    """Normalisation backed by a fixed input-space transform tensor.

    Stores a pre-computed transform applied directly to the raw input before
    it enters the network, distinct from the affine stats path.
    """

    transform_input: Tensor


# Union type for all supported normalisation representations.
Normalization = Union[
    TransformNormalization,
    StatsNormalization,
    InputTransformNormalization,
]

@dataclass(frozen=True)
class ZScoreContext:
    """Resolved z-score artefacts shared by all estimator types.

    Produced by the z-score resolver after interpreting a ``ZScoreConfig``
    and passed into every ``EstimatorBuilder.build()`` call via
    ``BuildContext``.  Subclasses extend this with estimator-specific fields.

    Attributes:
        x_normalization: Normalisation to apply to the observation (``x``)
            input, or ``None`` when no normalisation is required.
        y_embedding: Embedding network applied to the conditioning variable
            (``theta`` / ``y``), or ``None``.
    """

    x_normalization: Optional[Normalization] = None
    y_embedding: Optional[nn.Module] = None


@dataclass(frozen=True)
class ClassifierZScoreContext(ZScoreContext):
    """Z-score context for classifier-based estimators (NRE).

    Extends ``ZScoreContext`` with a second embedding network for the
    observation side, since classifiers embed both inputs independently.

    Attributes:
        x_embedding: Embedding network applied to the observation (``x``)
            input, or ``None``.
    """

    x_embedding: Optional[nn.Module] = None


@dataclass(frozen=True)
class MixedZScoreContext(ZScoreContext):
    """Z-score context for mixed (discrete + continuous) estimators (MNLE / MNPE).

    Extends ``ZScoreContext`` with separate embedding networks for the
    discrete and continuous observation components.

    Attributes:
        embedding_net_discrete: Embedding network for the discrete part of
            the observation, or ``None``.
        embedding_net_continuous: Embedding network for the continuous part
            of the observation, or ``None``.
    """

    embedding_net_discrete: Optional[nn.Module] = None
    embedding_net_continuous: Optional[nn.Module] = None