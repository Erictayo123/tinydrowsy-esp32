# Decision: Dataset Strategy

**Date:** 2026-07-15
**Status:** Accepted

## Context

Eye-state classification models trained on a single dataset tend to overfit to that dataset's capture conditions (camera type, lighting, subject demographics, framing). Since TinyDrowsy needs to generalize to a live OV2640 camera feed in a car — very different conditions from most lab-collected datasets — the training data needed to be broad enough to cover that gap.

## Decision

Train on a **combined dataset** of 46,000+ images pooled from multiple sources:
- **MRL Eye Dataset** — large-scale, varied-condition eye images
- **CEW (Closed Eyes in the Wild)** — closed-eye images captured in unconstrained conditions
- **DDD (Driver Drowsiness Dataset)** — driving-specific footage, closer to the deployment domain
- **Selfie-captured images** — collected specifically to include conditions similar to the target device's camera and typical in-cabin framing

## Rationale

1. **Domain coverage.** Combining a driving-specific dataset (DDD) with general eye-state datasets (MRL, CEW) gives the model both volume and domain relevance.
2. **Generalization over a single-source ceiling.** A model trained only on lab datasets tends to perform worse when deployed on an embedded camera with different optics, resolution, and lighting response; mixing in selfie-captured images partially closes that gap.
3. **Class balance.** Sourcing from multiple datasets made it easier to balance the open/closed eye classes rather than being constrained by a single dataset's natural distribution.

## Alternatives Considered

| Approach | Data Volume | Domain Match | Verdict |
|---|---|---|---|
| MRL only | Large | Low (lab conditions) | Rejected — insufficient domain match |
| DDD only | Small | High (driving-specific) | Rejected — insufficient volume |
| **Combined (MRL + CEW + DDD + selfies)** | **46,000+** | **Moderate–High** | **✅ Selected** |

## Consequences

- ✅ 99.87% test accuracy achieved
- ✅ Better robustness to varied lighting and camera angle than a single-source dataset would provide
- ❌ Some residual **domain shift** observed during live simulation — real-world camera conditions on the ESP32-S3 + OV2640 setup still diverge somewhat from the training distribution, which is flagged as an area for continued refinement (e.g., collecting more on-device calibration images)
