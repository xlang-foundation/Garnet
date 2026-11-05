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

## Implementation checkpoint

The portable packed-plan helper is implemented and its initial mixed-tick
range/lease/budget rejection tests pass. Follow-up source changes expose this
metadata from `MetalContinuousScheduler::NextJson` with an explicit
`execution: metadata_only` marker, and additionally require each segment to
start at the allocator's committed logical length. Native integration tests
now compare every packed segment with its source scheduler request.
These follow-up changes await compilation and execution after
`native27b-registers-chunk16-http-admission1` exits; that live campaign uses
its own frozen pre-change library. No packed GPU execution is implemented yet.

The worker now has an opt-in `mixed_forward: true` source path requiring the
native `packed_token_plan` backend extension. A mixed native tick materializes
one existing chunk graph input: decode rows have one active token; prefill rows
have the scheduled prompt segment. A single session execution updates disjoint
leased rows, then predicts decode/final-prefill rows and acknowledges each
request with its original stage. `native_mixed_forward` records valid tokens
and padded projection rows. This is an intermediate masked execution method,
not compact matrix execution. It may change decode precision when a BF16
prefill kernel is selected and therefore requires fresh coding qualification.

The quality auditor requires mixed-forward evidence when enabled and joins its
segments/counts/leases against preceding native plans; three new tamper cases
cover missing mixed calls, changed counts and stale epochs. The archived
non-mixed register coding run still passes all eight existing tamper checks.
Python syntax has been checked. Compilation, native integration tests, the
three mixed tamper cases and actual mixed coding execution remain pending until
the frozen serving campaign releases the device. Default behavior stays off.

## Compact projection implementation choice

Inspection of `MetalContext::EncodeAffineInt4MatMul` and the register cooperative
shader suggests a smaller implementation than allocating gathered matrices.
Pass an immutable active-token row index table to an indexed affine kernel.
Its logical M is the number of valid tokens; input loads and output stores map
logical rows back to the existing physical activation rows. Attention,
convolution, recurrence and residual operations retain their existing layout.
This can remove both padded projection arithmetic and separate activation
pack/unpack copies. The existing masked mixed path is the correctness control.

Required safeguards before implementation/acceptance:

- The session derives indices from validated active flags after epoch checks;
  indices must be unique, in range and immutable through command completion.
- Apply indexing only to token projections with the exact controlled row
  layout. Exclude the vocabulary head, state matrices and any incompatible
  reshape. An equal row count alone is insufficient proof of layout identity.
- Define inactive outputs explicitly, preferably zero, so reused scratch never
  leaks old values or NaNs into later operations. Include that write cost.
- Keep precision selection explicit: compacting a BF16 matrix to four or fewer
  rows must not silently switch to a different arithmetic contract. Report
  selected kernel, logical/physical rows and actual indexed dispatch counts.
- Compare active results and untouched state with the masked control across
  noncontiguous row selections, one-token decode plus prompt tails, empty/full
  masks and reused buffers. Then run the full learned coding gate.
- Account for index storage and scratch in the xModel memory profile. Compare
  total request latency, not just an indexed projection microbenchmark.

This indexed kernel is not implemented yet. First qualify the optional masked
mixed execution and its native metadata integration after the live campaign.

## Direct comparison with the frozen vLLM-Metal run

The preserved `baseline-phase1/vllm/cohort-1/server.log` confirms the actual
baseline enabled chunked prefill with a **512-token total scheduler budget**.
This is not equivalent to Garnet's fixed16-token-per-row graph. The log also
records native paged attention and NAX prefill-attention kernel loading; that
log message does not establish the matrix projection instruction path.

At the pinned vLLM-Metal source revision, inspect:

- `vllm_metal/v1/model_runner.py::_start_paged_forward`: builds one unpadded
  token sequence with decode segments first and prompt segments afterward;
  `cu_seqlens` provides request boundaries. Populates per-request hybrid state
  positions before one lazy target forward and asynchronous output submission.
- `vllm_metal/attention/impls/linear.py::_project_inputs`: projects the full
  packed token matrix first, then `_run_conv` and recurrent operations use
  request-specific state slots. This directly supports the intended Garnet
  architecture, rather than treating batching as separate graph calls.
- `vllm_metal/compiled_mlp.py::CompiledMLPBlocks`: compiles bounded decode-shaped
  stateless MLP blocks, with a four-token routing threshold. Source capability
  alone does not prove that this option was enabled in the frozen run.

The most actionable gap is projection amortization over a larger **valid-token
matrix**, without allocating a full fixed512-token scratch family per layer.
Do not merely raise Garnet's existing graph size to512: its measured scratch
budget and fixed-row layout are different. Qualify masked mixed execution,
then indexed valid-row projections and bounded shared workspace. Recheck
weight-read cost and actual layer/command timing before choosing larger
scheduler budgets. Asynchronous submission is another explicit implementation
contrast to inspect; it is not a standalone explanation for the full gap.

## Workspace lifetime candidate

`GARNET_METAL_PREFILL_LIFETIME_REUSE` is a new default-OFF source candidate,
requiring serialized shared prefill scratch. The existing eligibility checks
still exclude protected storage, existing aliases, incompatible layouts and
incomplete first writers. A portable interval planner additionally tracks the
last read **and write** of eligible temporaries and reuses equal-shape storage
only when the prior lifetime ends strictly before the next first write.
Same-operation read/write intervals cannot alias. The owner retains the maximum
required pool across graphs and the existing execution lock remains mandatory.

A new native test covers overlap boundaries, nonoverlap reuse, independent
shapes, deterministic ordering, duplicate slots and invalid ranges. Both this
test and the Metal candidate await compilation/execution after the ongoing
serving run. No memory saving or quality result has been measured. Qualify this
flag separately from the default-OFF mixed-forward worker option so their
correctness and performance effects remain distinguishable.

The first real event in `native27b-registers-mixed-chunk16-quality1` confirms
one mixed session execution containing a decode token at slot100 and a final
prefill token at slot112, with distinct rows/epochs. Its trace reports2 valid
tokens versus32 physical projection rows. The coding run remains live; this
is execution evidence only, not quality or performance acceptance.

A separate chunk16 lifetime-only coding profile is prepared with
`require_lifetime_reuse: true` and `mixed_forward: false`. The worker validates
that explicit policy against the native backend tag before loading model
weights. This avoids accidentally qualifying the control build as the lifetime
candidate. The profile has not yet been executed.

## Large-chunk native guard preparation

A default-OFF `GARNET_METAL_QWEN38_LARGE_PREFILL` source candidate now couples
128/256-token native limits to signed16-bit attention indices and the
corresponding threadgroup scratch accounting. It requires bounded27B wide
prefill, compact4K attention and nonresident pack retirement. Existing profiles
keep their previous native64-token scheduler limit when disabled. Extra native
oracle cases target indices127/255, contiguous and overwritten slots, inactive
queries,4K attention, and256/512-row real-inner-dimension projections.
These changes are not yet compiled or admitted as xModel profiles; compilation
waits for the live pack-retirement coding run. The xModel factory still rejects
chunks above64. Passing component cases alone will not establish memory-safe
learned-model execution at128 or256.
