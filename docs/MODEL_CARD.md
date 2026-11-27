# Model card: card-fraud benchmark model

## Model details

- **Type:** standardized, class-weighted logistic regression (LightGBM was the
  alternative and scored lower on validation average precision).
- **Inputs:** 28 anonymized PCA components (`V1`-`V28`) and `Amount`. `Time` is not a feature.
- **Output:** an uncalibrated score in [0, 1] and an illustrative action label.
- **Artifact:** local joblib file, never committed; its SHA-256 is pinned in production mode.
- **Seed:** 42.

## Intended use

Studying evaluation design for extremely imbalanced, time-ordered fraud data, and
demonstrating auditable serving patterns (idempotency, durable decisions, delayed labels).

## Out-of-scope use

Approving, challenging or blocking real transactions; estimating INR losses or ROI;
inferring behavior of current card or UPI traffic.

## Data

ULB / Worldline credit-card dataset via OpenML 1597 v1: 284,807 transactions over two
days in September 2013, 492 fraud cases. Features are PCA-anonymized, the currency of
`Amount` is undocumented, and there are no account identifiers.

| Window | Rows | Fraud cases |
| --- | ---: | ---: |
| Train | 199,364 | 384 |
| Validation | 42,722 | 56 |
| Test | 42,721 | 52 |

## Evaluation

| Metric (test window) | Value |
| --- | ---: |
| Average precision | 0.708 (prevalence 0.00122) |
| ROC-AUC | 0.977 |
| Brier score (uncalibrated) | 0.0224 |
| Precision / recall / F1 at threshold | 0.459 / 0.750 / 0.569 |
| Confusion matrix | TN 42,623, FP 46, FN 13, TP 39 |
| Relative cost vs approve-all | 306 vs 1,040 |

The threshold and model were chosen on validation only. With 52 positive cases the
estimates carry wide uncertainty, which is not quantified here.

## Limitations and risks

- One short temporal split; no repeated-period validation.
- No account-disjoint evaluation, no device, merchant or geographic context.
- Scores are not calibrated probabilities; do not read them as fraud likelihoods.
- Cost weights are assumptions, not measured operating costs.
- Dataset age and narrow collection window limit any claim about current traffic.
- Fairness and subgroup analysis is not possible with anonymized features.

## Operational evidence

Single-row in-process latency p95 1.43 ms. A 300-request local HTTP replay measured
about 68.8 req/s with P95 177.7 ms and recovered state after a forced restart. These
are single-machine measurements and not an SLA.
