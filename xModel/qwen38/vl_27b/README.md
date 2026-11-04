# Qwen3.8-27B Metal qualification

Production qualification is incomplete. Keep these profiles out of the
production catalog until their remaining gates pass. The current machine is an
Apple M5 with24GiB unified memory; results do not admit other Apple hardware.
CUDA factories remain separate.

The Metal path consumes pinned `mlx-community/Qwen3.8-27B-4bit`, revision
`10c35caafbb80f7dc6a7a432cdd11af10a6d4818`: affine INT4/group64, eight low-nibble-first
values perU32, BF16 scales and additive biases. Native kernels keep projections
compressed and use FP32 accumulation. The control uses FP32 activations/KV;
optional BF16 tensor operands are separate experimental precision profiles.
Requested embedding rows round to BF16 before host FP32 conversion, matching
the artifact's quantized embedding output contract. Independent real-row parity,
packing and shape references are retained in the research evidence.

The adapter validates names, convolution dimension order and pre-shifted
normalization weights. It uses zero additional norm offset for this conversion;
original HF formats retain their existing offset. The untied `lm_head` must not
project logits through the input embedding matrix. Text input allocates no dummy
image or vision encoder.

## Current evidence

| Area | Verified scope | Remaining scope |
| --- | --- | --- |
| Text coding | Eight frozen sandboxed tasks pass for single-native and concurrent batch2 controls; full generated sequences match | Broader coding qualification and final optimized candidate |
| Batch state | Native stable row/epoch leases, ragged batch2/4, row refill, cancellation and inactive-state preservation | Production integration and final serving performance |
| Prefill scheduling | Optional native chunks interleave existing decode; batch2/4K/chunk64 passes8/8 coding tasks,1,058 combined decode plans and9 interleavings | Serving timing and restored-prefix integration with this optional schedule |
| Prefix reuse | Native token matching restores full-attention KV, convolution, recurrence and final hidden state; exact/append/edit/refill cases pass | Complete matched cache performance cohorts and chunked-schedule combination |
| Memory | Batch2/4K/chunk64 allocates18,259,361,792 Metal bytes and returns to720,896 bytes after close on this M5 | Execution to every context boundary and other hardware profiles |
| Vision | All333 artifact tensors mapped; native finite vision parity and two image-grounding cases pass | General VLM factory admission, reference VLM qualification and screenshot-to-code quality |
| Performance | Repeated vLLM-Metal/SGLang text references frozen; native timing admissions and projection diagnostics preserved | Full repeated optimized Garnet campaign; all2x goals remain unmet |

The latest FP32 batch2/wide64 timing admission has five completed cells and a
failed four-client long-prompt cell: one queued request exceeded600s. Preserve
that failure; it is not a passing campaign. Loaded native chunk-selection probes
and serialized operator traces are not HTTP serving measurements.

## Entry points and limits

- `model.create_decoder` is the bounded batch1 Metal study factory, capacity64..4096.
- `metal_batch_study.create_decoder` is a separate experimental batch2/4 factory.
  Batch2 permits chunks2/4/8/16/32/64; batch4 remains chunk<=8. Wider shapes require
  native guarded workspace, shared scratch, in-place caches and per-row hidden
  support. Separate fixed-shape admissions do not admit all shapes resident at once.
  The default-off `qwen38_large_prefill` extension additionally exposes experimental
  batch2 chunks128/256 with signed16-bit attention indices and nonresident pack
  retirement. Component tests pass; learned chunk128 qualification is pending.
  Each requested shape must still pass the native working-set guard; this does
  not guarantee that chunk256 fits this Mac or admit these shapes for production.
- `garnet.continuous_scheduler()` owns allocation and lifecycle in native code.
  The optional native chunk schedule retains leases while committing partial
  prompt lengths. Python study transports supply inputs and route native events.
- Native host prefix snapshots are bounded and identity-scoped. The current
  chunked-prefill worker profile explicitly rejects prefix reuse until restored
  progress is integrated; the whole-prompt prefix study remains available.
- Current compact chunk attention is limited to4096 context positions.8K/16K,
  paged or BF16 KV, and arbitrary larger batches are unsupported profiles.
- `create_vision` still rejects general runtime admission. Finite grounding
  evidence is not a general VLM qualification.

`metal-qualification-profile.json` and `metal-context-study-profile.json` retain
historical bring-up definitions; they are not production profiles or an assertion
that all later gates are missing. Source, raw outputs, model hashes, frozen
requests, audits and limitations are described in
[the research record](../../../docs/research/local-coder-metal-benchmark.md).

Build extensions default off. Register/staged Metal tensor candidates require
explicit operand/layout identities and actual native dispatch evidence; API
selection alone does not prove physical NAX instruction use or a model speedup.
