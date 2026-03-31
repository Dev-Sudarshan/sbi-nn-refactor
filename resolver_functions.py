from typing import Literal, Optional

import torch
from torch import Tensor, nn
from dataclasses import replace

from configs import ClassifierZScoreContext, InputTransformNormalization, MixedZScoreContext, TransformNormalization, ZScoreConfig, ZScoreContext , StatsNormalization
from sbi.utils.sbiutils import (
    biject_transform_zuko,
    mcmc_transform,
    standardizing_net,
    standardizing_transform,
    standardizing_transform_zuko,
    z_score_parser,
    z_standardization,
)

def resolve_zscore_classifier(
    batch_x: Tensor,
    batch_y: Tensor,
    zscore_config: ZScoreConfig,
) -> ClassifierZScoreContext:
    """Resolve z-score artefacts for classifier-based estimators (NRE).

    Returns a ``ClassifierZScoreContext`` with independent embedding nets for
    both the observation (``x_embedding``) and the parameter (``y_embedding``).
    """

    embedding_net_x = resolve_embedding_net(batch_x, zscore_config.x)
    embedding_net_y = resolve_embedding_net(batch_y, zscore_config.theta)


    return ClassifierZScoreContext(
        x_embedding=embedding_net_x,
        y_embedding=embedding_net_y,
    )

def resolve_nflows_input_transform(
    batch_x: Tensor,
    batch_y: Tensor,
    zscore_config: ZScoreConfig,
) -> ZScoreContext:
    """Resolve an NFlows input transform from a ``ZScoreConfig``.

    Populates ``x_normalization`` with a ``TransformNormalization`` when
    z-scoring is enabled; sets ``y_embedding`` via ``resolve_embedding_net``.
    """

    x_normalization = None

    z_score_x_bool, structured_x = z_score_parser(zscore_config.x)

    if z_score_x_bool:
        transform = standardizing_transform(batch_x, structured_x)
        x_normalization = TransformNormalization(transform=transform)

    embedding_net_y = resolve_embedding_net(batch_y, zscore_config.theta)
    return ZScoreContext(
        x_normalization=x_normalization,
        y_embedding=embedding_net_y
    )


def resolve_zuko_x_transforms(
    batch_x: Tensor,
    batch_y: Tensor,
    zscore_config: ZScoreConfig,
) -> ZScoreContext:
    """Resolve Zuko-specific x transforms from a ``ZScoreConfig``.

    Handles both the ``'transform_to_unconstrained'`` path (bijective MCMC
    transform via ``biject_transform_zuko``) and the standard affine z-score
    path.  Sets ``x_normalization`` and ``y_embedding`` accordingly.
    """

    x_normalization = None

    if zscore_config.x == "transform_to_unconstrained":
        if zscore_config.x_dist is None:
            raise ValueError("x_dist required")
        transform = biject_transform_zuko(mcmc_transform(zscore_config.x_dist))
        x_normalization = TransformNormalization(transform=transform)
    else:
        z_score_x_bool, structured_x = z_score_parser(zscore_config.x)
        if z_score_x_bool:
            transform = standardizing_transform_zuko(batch_x, structured_x)
            x_normalization = TransformNormalization(transform=transform)

    embedding_net = resolve_embedding_net(batch_y, zscore_config.theta)

    return ZScoreContext(
        x_normalization=x_normalization,
        y_embedding=embedding_net
    )


def resolve_mdn_transform_input(
    batch_x: Tensor,
    batch_y: Tensor,
    zscore_config: ZScoreConfig,
) -> ZScoreContext:
    """Resolve MDN input normalisation tensors from a ``ZScoreConfig``.

    Populates ``x_normalization`` with an ``InputTransformNormalization``
    (stacked mean/std tensor) when z-scoring is enabled.
    """
    x_normalization = None

    z_score_x_bool, structured_x = z_score_parser(zscore_config.x)

    if z_score_x_bool:
        x_mean, x_std = z_standardization(batch_x, structured_x)
        x_normalization = InputTransformNormalization(
            transform_input=torch.stack([x_mean, x_std], dim=0)
        )
    embedding_net = resolve_embedding_net(batch_y, zscore_config.theta)

    return ZScoreContext(
        x_normalization=x_normalization,
        y_embedding=embedding_net
    )


def resolve_x_normalization_stats(
    batch_y: Tensor,
    batch_x: Tensor,
    zscore_config: ZScoreConfig,
) -> ZScoreContext:
    """Resolve x normalisation statistics for score / vector-field estimators.

    Populates ``x_normalization`` with a ``StatsNormalization`` (mean, std).
    Falls back to mean=0, std=1 when z-scoring is disabled, preserving
    backward compatibility.
    """

    z_score_x_bool, structured_x = z_score_parser(zscore_config.x)
    if z_score_x_bool:
        mean, std = z_standardization(batch_x, structured_x)
    else:
        mean, std = 0, 1

    x_normalization = StatsNormalization(mean=mean, std=std)
    embedding_net = resolve_embedding_net(batch_y, zscore_config.theta)

    return ZScoreContext(
        x_normalization=x_normalization,
        y_embedding=embedding_net
    )


def resolve_embedding_net(
    batch: Tensor,
    z_score: Literal["none", "independent", "structured", "transform_to_unconstrained"] = "independent",
) -> Optional[nn.Module]:
    """Resolve a standardising embedding net for a conditioning variable.

    Returns a ``standardizing_net`` when z-scoring is enabled, otherwise
    ``None``.
    """

    z_score_bool, structured = z_score_parser(z_score)
    if z_score_bool:
        embedding_net = standardizing_net(batch, structured)
        return embedding_net
    
    return None


def resolve_unconditional_zuko_transforms(
    batch_x: Tensor,
    zscore_config: ZScoreConfig,
) -> ZScoreContext:
    """Resolve x transform for unconditional Zuko flows.

    Populates ``x_normalization`` with a ``TransformNormalization`` when
    z-scoring is enabled; ``y_embedding`` is always ``None`` (no conditioning).
    """

    x_normalization = None

    z_score_x_bool, structured_x = z_score_parser(zscore_config.x)
    if z_score_x_bool:
        transform = standardizing_transform_zuko(batch_x, structured_x)
        x_normalization = TransformNormalization(transform=transform)

    return ZScoreContext(
        x_normalization=x_normalization
    )


def z_score_resolver_mixed_density_estimator(
    batch_x: Tensor,
    batch_y: Tensor,
    zscore_config: ZScoreConfig,
    flow_model: Optional[Literal["nsf", "maf", "mafrqs", "made", "zuko_nsf", "zuko_maf", "zuko_mafrqs", "zuko_made","mdn"]] = "nsf",
) -> MixedZScoreContext:
    """Resolve z-score artefacts for mixed (discrete + continuous) density estimators.

    Dispatches to the NFlows or Zuko resolver based on ``flow_model`` and
    returns a ``MixedZScoreContext`` with ``x_normalization``, ``y_embedding``,
    and ``embedding_net_continuous`` populated; ``embedding_net_discrete`` is
    reserved for future use and always ``None``.
    """


    embedding_net_y = resolve_embedding_net(batch_y, zscore_config.theta)

    zscore_config_local = replace(zscore_config, theta="none")

    if flow_model in ["nsf", "maf", "mafrqs", "made"]:
        ctx = resolve_nflows_input_transform(
            batch_x,
            batch_y,
            zscore_config_local,
        )

        return MixedZScoreContext(
            x_normalization =ctx.x_normalization,
            y_embedding=embedding_net_y,
            embedding_net_discrete=None,
            embedding_net_continuous=ctx.y_embedding,
        )

    elif flow_model.startswith("zuko"):
        ctx = resolve_zuko_x_transforms(
            batch_x,
            batch_y,
            zscore_config_local,
        )

        return MixedZScoreContext(
            x_normalization =ctx.x_normalization,
            y_embedding=embedding_net_y,
            embedding_net_discrete=None,
            embedding_net_continuous=ctx.y_embedding,
        )


    elif flow_model == "mdn":
        
        ctx = resolve_mdn_transform_input(
            batch_x,
            batch_y,
            zscore_config_local,
        )

        return MixedZScoreContext(
            x_normalization = ctx.x_normalization,
            y_embedding=embedding_net_y,
            embedding_net_discrete=None,
            embedding_net_continuous=None,
        )

    else:
        raise ValueError(f"Unsupported flow_model: {flow_model}")
    

def resolve_z_transform(self, batch_x, batch_y, z_score_config):
    """Dispatch z-score resolution to the resolver matching the model backend.

    Selects the appropriate resolver based on ``self.model``:

    * ``"mdn"``                   → ``resolve_mdn_transform_input``
    * ``"maf"``, ``"nsf"``, ``"maf_rqs"``, ``"made"`` → ``resolve_nflows_input_transform``
    * ``"zuko_*"``                → ``resolve_zuko_x_transforms``
    * ``"mnle"``, ``"mnpe"``      → ``z_score_resolver_mixed_density_estimator``

    Args:
        batch_x: Sample batch of observations used for x-normalisation
            statistics and discrete/continuous column detection.
        batch_y: Sample batch of parameters used for y-embedding statistics.
        z_score_config: Z-scoring intent for both x and theta.

    Returns:
        A resolved ``ZScoreContext`` (or ``MixedZScoreContext`` for mixed
        models) with pre-computed transforms and/or embedding nets.
    """
    if self.model == "mdn":
        return resolve_mdn_transform_input(batch_x, batch_y, z_score_config)

    elif self.model in ("maf", "nsf", "maf_rqs", "made"):
        return resolve_nflows_input_transform(batch_x, batch_y, z_score_config)

    elif self.model.startswith("zuko_"):
        return resolve_zuko_x_transforms(batch_x, batch_y, z_score_config)

    elif self.model in ("mnle", "mnpe"):
        num_disc = (
            len(self.config.num_categories_per_variable)
            if self.config.num_categories_per_variable is not None
            else int(torch.sum(_is_discrete(batch_x)))
        )
        cont_x, _ = _separate_input(batch_x, num_discrete_columns=num_disc)
        return z_score_resolver_mixed_density_estimator(
            cont_x, batch_y, z_score_config, flow_model=self.config.flow_model
        )