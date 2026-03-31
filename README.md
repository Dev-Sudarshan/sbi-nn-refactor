# sbi Neural Network API Refactor — Prototype

A working prototype implementing the proposed neural-network configuration
and z-score resolver refactor for the [`sbi`](https://github.com/sbi-dev/sbi)
library, built as part of my **Google Summer of Code (GSoC) proposal** .

This repository demonstrates early implementation of the proposed design,
showing that the ideas are not only conceptual but already partially realized
and ready for integration into the main codebase.

## Motivation

The current `sbi` codebase resolves z-scoring, embedding networks, and
normalisation inline inside each `build_*` function. This leads to:

- duplicated logic across NRE, NPE (NFlows/Zuko), MDN, MNLE, and score-based estimators
- tight coupling between data preprocessing and model construction
- difficulty in testing, extending, and reasoning about normalization behavior

This prototype introduces a clean architectural separation:

1. **User declares intent** via a `ZScoreConfig` dataclass.
2. **Resolver computes artifacts** → `ZScoreContext` (normalisation + embeddings).
3. **Builders consume structured input** via `BuildContext`.

## What This Prototype Demonstrates

This is not a minimal mock — it includes real implementation of core components:

- ✅ Typed configuration system (`ZScoreConfig`)
- ✅ Multiple `ZScoreContext` variants for different estimator families
- ✅ Resolver functions covering NFlows, Zuko, MDN, classifier, and mixed estimators
- ✅ Unified `BuildContext` abstraction

This validates the core refactor direction and reduces uncertainty for full integration.

## Files

| File | Purpose |
|------|---------|
| `configs.py` | Frozen dataclasses: `ZScoreConfig`, normalization types (`StatsNormalization`, `TransformNormalization`, `InputTransformNormalization`), context types (`ZScoreContext`, `ClassifierZScoreContext`, `MixedZScoreContext`), and `BuildContext`. |
| `resolver_functions.py` | Resolver functions that interpret a `ZScoreConfig` and return the appropriate `ZScoreContext`. |

## Key Design Decisions

**Frozen dataclasses** — All config and context objects are immutable, making the data flow explicit and safe to cache.

**Typed normalization system**
```python
Normalization = TransformNormalization | StatsNormalization | InputTransformNormalization
```
Builders operate on types rather than string flags.

**Dedicated resolver per estimator family** — Eliminates scattered conditional logic (`if z_score_x == ...`) across the codebase.

**Unified `BuildContext`** — Encapsulates shapes, resolved normalization, device/dtype, and sample batches into a single interface.

## Estimator Families Covered

| Resolver | Estimator |
|----------|-----------|
| `resolve_nflows_input_transform` | NFlows-based NPE |
| `resolve_zuko_x_transforms` | Zuko-based NPE |
| `resolve_mdn_transform_input` | MDN |
| `resolve_x_normalization_stats` | Score / vector-field estimators |
| `resolve_zscore_classifier` | NRE (classifier) |
| `resolve_unconditional_zuko_transforms` | Unconditional Zuko flows |
| `z_score_resolver_mixed_density_estimator` | MNLE / MNPE (discrete + continuous) |

## Status

- ✅ Core architecture implemented
- 🔄 Builder integration (next step)
- 🔄 Trainer integration (planned in GSoC timeline)

This prototype was developed before proposal submission to demonstrate
implementation readiness and reduce execution risk.

I have also contributed to `sbi` previously with merged PRs.

## Dependencies

- [`sbi`](https://github.com/sbi-dev/sbi)
- `torch`
- `nflows`
- `zuko`