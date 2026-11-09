# Mistral native checkpoint, 2026-11-08

This checkpoint includes the native Metal work through source commit
`bd0c48b2533575c37839d01e1a153bed5e4333f7`. It preserves the experimental
history and negative results. It does not claim the 2x serving target is met.

## Native runtime and scheduling

- Optional native lookahead attention, prepared projections, candidate pool,
  row plans, staged KV commit and Session/scheduler integration are present.
  `GARNET_METAL_LOOKAHEAD_ATTENTION`, `GARNET_METAL_LOOKAHEAD_PROJECTIONS` and
  `GARNET_METAL_LOOKAHEAD_RUNTIME` default to OFF. The separate target-Jacobi
  and indexed-clear experiments also remain optional and default OFF.
- Native lookahead qualification retained 56 exact generated IDs and 6,890
  full canonical KV/final-hidden state comparisons. The KV commit unit covered
  1,129 cases and 179,666,944 FP32 word comparisons. These are correctness
  results, not measured serving speed gains.
- `continuous_scheduler.set_prefill_completion_owners(1|2|4)` is available
  under the existing wide-owner Metal profile. It requires an idle scheduler,
  bounded chunks and completion-first mode. The default remains one owner.
  Ready decodes are scheduled first; actual work stays within the existing
  512-token global budget and fixed row leases.
- The one-owner/two-owner native pair completed five queued requests, including
  physical KV row reuse. All 59 output IDs matched; 5,330 full-state digest
  comparisons at equal progress passed. All captured ready/generated states
  matched. Nineteen different partial-prefill boundaries were explicitly not
  compared. Seven corruption controls were rejected. CPU scheduling checks
  covered 12 variants, including cancellation and cached head hits.

Qualification summaries and their hashes are in
[`mistral_native_runtime_checkpoint`](../../test2026/metal/experiments/mistral_native_runtime_checkpoint/manifest.json).
Model artifacts and native libraries are not included in Git.

## Serving and ANE status

The matched HTTP cohort comparison is prepared but has not run. The current
profile already enables the earlier decode kernels; the new cohort policy
changes how prefill work overlaps ready decoding. Historical TTFT and decode
maxima must not be combined into a synthetic result. vLLM-Metal and SGLang
reference measurements remain frozen.

The retained Core ML CPU+ANE Mistral MLP operator study improved from about
17.47 ms to 14.22 ms, while the Metal control was about 10.36 ms. Thus the
optimized ANE path improved over its previous implementation but remained
slower than Metal. This is an operator study using embedding proxy rows,
not a full-model or separate MPS comparison. Its numerical error and placement
limits remain documented in
[`coreml-ane-plugin.md`](coreml-ane-plugin.md).

Useful coding quality, a whole-model ANE speed gain, and the full Mistral/Qwen
performance targets remain unqualified. This is a code and evidence checkpoint.
