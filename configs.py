from dataclasses import dataclass
from typing import Optional , Literal
from torch.distributions import Distribution

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