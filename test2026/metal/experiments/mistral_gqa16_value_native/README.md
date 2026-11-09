# Native shared-value Mistral verification attention

The verifier previously reloads value data separately for query tokens and GQA
heads. The new optional Metal kernel shares each value/probability tile across
four query tokens and four query heads per KV head. Every output retains the
same ascending FP32 accumulation chain and separate FP32 probability boundary.
It uses the screened 64-key tile for B1 and 32-key tile for B4, with 128 threads.

`GARNET_METAL_MISTRAL_VERIFY_GQA16_VALUES` defaults OFF and requires bounded
verification and the Mistral layout. The prepared profile is ungated B1/B4,
T4, H32/KV8/D128, C4096. Actual visible tokens remain runtime data; the fixed
layout does not pad computation to the context maximum. Selection requires
validated immutable contiguous controls, bounded uint32 matrix spans and a
maximum visible prefix of at least 512. Other profiles use their existing route.
Both PSOs are compiled at initialization, with native default compiler options.
Dynamic threadgroup storage is 12,288 bytes for B1 and 6,144 for B4; no global
GPU workspace or weight copy is added. xModel operators are unchanged.

The source includes upstream `main` through `879ed6651`, merged in `176f28846`.
`native1.patch` is the value-only increment over that reconciled native QK source.
`source-reference.json` pins complete sources, base/patch inputs and libraries.
The backend capability is `native_mistral_verification_gqa16_values64_32_optional_v1`.
C++ diagnostics add `VerificationSetGqaValuesEnabled` (closed/drained batches),
`LastVerificationGqaValueDispatches` and stat `target_attention_gqa16_value_dispatches`.
The legacy vector4 counter counts only its actual dispatch. Begin/Abort reset
both counters; the new route retains existing output/cache ownership.

## Qualification

ON and default-OFF builds each pass 56 native tests. Invalid dependencies are
rejected. The new unit compares 1,883,915,904 FP32 words across 62 cases per
build, including full probabilities/context/KV, mixed and inactive NaN rows,
short/unsupported fallback profiles, committed/live KV, pending queries,
foreign writers, partial submission and Abort/recovery. Native owners end at zero.

Five fresh full-model engines ran serially. B1 uses the actual Qwen coding helper
and produces exactly 128 target IDs with identical proposals, accepted commits
and full state. C4 ordinary/verification/reverse-preparation arms match 5,460
state digest pairs; 1,040 digest pairs prove invalid calls leave state unchanged.
C4 covers all 1..16 submitted physical-token counts, cancellation/refill and
late owners. It deliberately uses zero proposals for correctness: productive
learned-helper C4 speculation and serving throughput remain unqualified.
The qualification auditor rejects 13 deliberate corruptions.

## Paired native decode measurement

Four new engines ran OFF/ON then ON/OFF, with unchanged 2K target input,
128 outputs, helper C512/fixed user tail192 and exact qualified histories.
No builds/source/model hashes run during collection; comparators are unchanged.

| Cohort | Value OFF tok/s | Value ON tok/s | Paired ratio |
|---|---:|---:|---:|
| 1 | 28.192 | 30.502 | 1.082x |
| 2 | 26.889 | 29.646 | 1.103x |
| Median rate | 27.540 | 30.074 | |

Both cohorts pass the declared 1.05x native-stage gate. These are native decode
stages, not HTTP TTFT or aggregate serving measurements. Absolute rates differ
from earlier qualification/score runs; thermal and OS conditions are uncontrolled.
This does not establish a serving win over vLLM/SGLang, coding-quality success,
or the overall 2x goal. The matched HTTP comparison remains unchanged.

All source files reconstruct exactly from base + upstream patch + native patch.
The independent CPU replay reproduces complete source, qualification and timing
reports exactly. Run `replay_audits1.py FRESH_OUTPUT_DIR` in the qualified asset
Python environment; it verifies the package/raw evidence, reconstructs a fresh
source tree and runs CPU auditors without executing any GPU model.

Next: productive causal B4 verification, matched max-four HTTP measurements,
and prepared prefill/quality work before returning to the Qwen27B goal.
