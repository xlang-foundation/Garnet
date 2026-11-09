# Native Mistral GQA16 verification scores

The native Metal verifier now has an opt-in shared-key score route for
ungated B1/B4, T4, H32/KV8/D128, C4096 attention. It shares each 32-key tile
across four query tokens and four query heads per KV head. It preserves the
per-lane dot-product order, SIMD reduction, separate FP32 probability
boundary and original full-row softmax. The uniform all-inactive-row guard
avoids converting inactive NaN slots. Inputs, K/V ownership and xModel
operators remain unchanged.

`GARNET_METAL_MISTRAL_VERIFY_GQA16_SCORES` defaults OFF and requires bounded
verification plus the Mistral layout. Both generic and B1 layout PSOs are
prepared at context initialization, with the same native compiler options.
The route requires the exact geometry above, validated immutable contiguous
controls, proven uint32 matrix spans and maximum visible prefix >=512.
Other shapes and short prefixes use their existing route. It adds no global
GPU workspace or weight copies. The two encoders reuse the existing FP32
probability matrix and its pending/Abort ownership registration.

The backend advertises `native_mistral_verification_gqa16_scores32_optional_v1`.
Native C++ diagnostics add `VerificationSetGqaScoresEnabled` (closed/drained
batches only), `LastVerificationGqaScoreDispatches`, and the verification
stat `target_attention_gqa16_score_dispatches`. The xModel operator API does
not change. Dispatch counters reset on Begin/Abort.

## Qualification

Both ON and default-OFF builds pass all 55 native tests. Invalid dependency
configuration is rejected. The new unit compares 1,401,980,224 FP32 words in
48 cases against ordinary sequential attention, including full K/V and
context results. It tests mixed/inactive rows, all-inactive NaN slots,
unsupported shape fallbacks, committed or uncommitted current K/V, pending
queries, partial Flush, foreign writers, rejected controls and Abort/recovery.
It finishes with zero native owners.

Five fresh model engines ran serially: B1 ordinary plus real causal-helper
verification with GQA ON/OFF, then C4 ordinary/verify/reverse-preparation
lifecycle arms. All 5,460 K/V/hidden digest pairs match ordinary inference;
1,040 digest pairs prove rejected native calls leave state unchanged. The
C4 lifecycle covers 1..16 submitted token positions, depths 1..4,
cancellation, row refill, surviving owners and late admission. Those C4
fixtures explicitly use zero proposals and are correctness-only. Real
learned-helper batching beyond B1 remains unqualified.

## Native performance

Mistral-7B-Instruct-v0.3 MLX affine4bit/g64/FP16 coefficients, FP32 KV/decode,
FP16 prefill, unchanged 2048-token target prompt and 128 output tokens. The
real Qwen2.5-Coder-0.5B helper keeps the declared fixed 192-user-token tail
and appends only already-generated target text. Its context is 512. No
future-output oracle enters proposals. All output IDs, full final K/V and
hidden state, helper histories and target commits repeat exactly.

Four fresh engines ran OFF/ON then ON/OFF, strictly serially. No source or
model hashing, builds, or concurrent model processes ran during collection.

| Native decode stage | Cohort 1 tok/s | Cohort 2 tok/s | Median tok/s |
|---|---:|---:|---:|
| GQA OFF, existing split-128 scores | 25.07 | 24.55 | 24.81 |
| GQA ON | 32.13 | 31.93 | 32.03 |
| Paired gain | 1.282x | 1.301x | — |

Native prefill remains about 3.4 seconds. These are native decoding stages,
not HTTP TTFT or matched serving throughput. Thermal/OS conditions remain
uncontrolled. Frozen vLLM-Metal and SGLang cohorts were neither rerun nor
changed. No new full-serving comparison or coding-quality admission is
claimed; the 2x decode/throughput and half-TTFT goals remain open.

## Evidence and reproduction

`native1.patch` applies with `git apply --unidiff-zero` to the frozen base in
`source-reference.json`. All 26,304 resulting source files were reconstructed
and checked exactly. Native code lives in the isolated source/build roots;
the original 11 dirty Garnet paths remain unchanged. The nine-file native
patch changes no xModel operators.

Raw logs, binaries, complete source bindings and fixtures remain at the
immutable roots listed in `source-reference.json`. The final native raw
index is `provenance2.json`. The earlier `provenance1.json`, failing replay1
and its outputs are preserved: replay1 loaded a historical audit module due
to an import-path conflict. `replay2.py` restores the current root before
each audit; it reproduces all complete reports exactly without GPU work.
No model inference was repeated for this harness fix.

Run `replay_audits1.py FRESH_OUTPUT_DIR` with the pinned asset Python
(NumPy, tokenizers, Jinja2). It verifies every package/raw hash, reconstructs
source comparisons, and runs the full source, qualification and measurement
CPU auditors with only report destinations redirected. Entire reports must
match their frozen originals. It never runs Metal or a model.

For a new GPU study, use a fresh directory and the recorded preparation,
build and worker scripts, preserving the pinned inputs and isolated source
base. Never rerun into frozen evidence roots. `decision1.json` records
remaining work: shared-value attention, real C4 causal drafting and matched
HTTP serving, larger prepared prefill families, coding quality, ANE/Gemma
and return to Qwen 27B.
