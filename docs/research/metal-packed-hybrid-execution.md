# Packed hybrid execution proposal

Status: proposed, not implemented or benchmarked. This is an engineering hypothesis,
not a claim of research novelty or a measured model speedup.

## Evidence and references

The frozen native operator probe attributes 94% of GPU time to projections.
The register cooperative-tensor candidate improves isolated prefill projections
by approximately 1.23–1.86x over the tall FP32 control; its BF16 operands are an
explicit precision change and full-model qualification remains required.
Fixed FP32 chunk sweeps favor 32 among 8/16/32/64 in the two observed contexts.
Neither result establishes the serving target.

Pinned reference sources, inspected locally:

- vLLM-Metal a44be48731005932a24dd30b55c21e992dc90b9a:
  `vllm_metal/platform.py` enables chunked prefill with a unified variable-length
  paged path; `vllm_metal/v1/model_runner.py` represents packed prefill and decode.
- SGLang 57abc8a831abb83009ab543fc89df1d39a1306ff:
  `python/sglang/srt/managers/scheduler.py` manages chunked prefill and mixed work;
  `hardware_backend/mlx/model_runner.py::_select_auxiliary_state_track_len`
  selects valid recurrent-state tracking boundaries. Source support does not
  imply that every configuration was exercised in our frozen baselines.

## Proposed Garnet method

At each transformer layer, gather active decode tokens and admitted prompt
tokens into one compact activation matrix. Apply shared quantized projections
to this matrix once. Route projected segments to each request's attention or
linear recurrent operation, preserving causal token order and independent
leased state. Gather the resulting activations for the next shared projection.
Decode rows must finish a token in the same iteration; prompt segments may end
at a native checkpoint boundary.

Unlike the current alternating fixed graphs, projection execution would share
work across decode and prefill requests. The intended gain is amortizing weight
reads and reducing padded activation work. It is not a claim that matrix-unit
execution will benefit one-row decode; low-row kernel dispatch must remain
available and be measured.

Select the prompt token budget from a measured per-layer cost table, active
decode count, remaining prompt lengths, and a decode latency budget. Start with
bounded 8/16/32/64 candidates rather than an unbounded adaptive controller.
Admit a candidate only if its workspace fits the hardware profile and its
predicted iteration time fits the latency budget. If none fits, issue decode
alone. Record predicted versus actual times and fall back on prediction error.

Do not instantiate all existing full-model graph shapes to implement this:
their persistent scratch can exceed this Mac's budget. Reuse bounded layer
workspaces and retain a fixed-shape control for comparison.

Prefix reuse must restore a jointly valid tuple of token length, full-attention
KV, convolution history, recurrent state, and positional metadata. Publish a
checkpoint only after the corresponding native iteration completes. Never
truncate recurrent state to an arbitrary common prefix. Immutable cache entries
and row epochs protect against concurrent reuse and stale completion.

## Implementation and acceptance sequence

1. Finish the live register-kernel coding gate and preserve token divergences.
2. Measure its serving admission with bounded timeouts and graceful close.
3. Add an explicit packed-segment native contract: request ID, row, epoch,
   packed offset, token count, absolute position, and prefill/decode kind.
   Validate disjoint ranges, causal progress, stale leases, and total bounds.
4. Implement layer-level gather/projection/scatter on the existing state format;
   compare every request against isolated execution, including ragged chunks,
   cancellation, refill, and concurrent prefix restoration.
5. Compare fixed chunks against adaptive budgets with identical prompts,
   precision, output limits, cache policy, and memory cap. Report TTFT, decode
   token latency percentiles, aggregate output tokens/s, and peak memory.
6. Run the complete frozen coding suite before accepting any speed result;
   run repeated serving cohorts only after admission passes. Keep vLLM-Metal
   and SGLang baseline results unchanged. Run all engines sequentially.

Reject the method if packing overhead erases the gain, decode tail latency
worsens beyond the declared budget, memory exceeds the profile, or correctness
fails. Single-request decode still needs its own optimization if unchanged;
prefill gains alone cannot satisfy the 2x decode target.
