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
| Prefill scheduling | Optional native chunks interleave existing decode; alias-group batch2/4K/chunk128 passes8/8 coding tasks,1,050 combined decode plans and3 interleavings | Repeated performance qualification of this memory candidate and restored-prefix integration |
| Prefix reuse | Native token matching restores full-attention KV, convolution, recurrence and final hidden state; exact/append/edit/refill cases pass | Complete matched cache performance cohorts and chunked-schedule combination |
| Memory | Opt-in alias-group/lifetime-reuse batch2/4K/chunk128 reports17,297,309,696 prepared Metal bytes and1,507,328 after close; all4 Garnet owners close to zero | Peak allocation, execution to every context boundary and other hardware profiles |
| Vision | All333 artifact tensors mapped; native finite vision parity and two image-grounding cases pass | General VLM factory admission, reference VLM qualification and screenshot-to-code quality |
| Performance | Frozen vLLM-Metal/SGLang references and a complete native three-cohort campaign retained:90 measured waves/210 requests, accounting and full-token audits pass | All2x speed/throughput and half-TTFT goals remain unmet; repeat the new memory candidate separately |

The earlier FP32 batch2/wide64 timing admission has five completed cells and a
failed four-client long-prompt cell: one queued request exceeded600s. Preserve
that failure; it is not a passing campaign. Loaded native chunk-selection probes
and serialized operator traces are not HTTP serving measurements.

The completed native campaign uses strict BF16 prefill operands and FP32 decode,
with prepared context buckets and chunk128. The newer memory candidate enables
both lifetime reuse and proven same-layout storage alias groups. Its8 coding
outputs exactly match the original single-native control. Preparation saves
678,100,992 bytes relative to the campaign candidate; this is sampled allocation,
not a peak measurement or a speedup. Detailed results, negative findings and
evidence paths are in `docs/research/local-coder-metal-benchmark.md`.

An additional default-OFF active-token chunk policy skips masked convolution
arithmetic and recurrent barriers without changing storage or active-token math.
It passes20 enabled/19 disabled native tests,238.8million original FP32 word
comparisons, learned session references and all8 unchanged coding tasks with
full token equality. Its isolated masked-convolution gains do not establish a
model speedup: all six single-wave serving admissions pass accounting and full
token equality, but five cells report lower overall output rates than the
control. All2x targets remain unmet. The policy stays disabled by default and
outside the production profile catalog.

The default-OFF native FP32 verification-attention primitive now matches repeated
ordinary decode for probabilities, outputs and full KV contents in226 cases.
Existing dense/selected four-row FP32 projections also match separate one-row
decodes, including the full vocabulary head. These are component gates only;
compiled verification graphs, per-token outputs and hybrid-state rollback remain
unfinished. No speculative generation or verification speedup is admitted.

## Entry points and limits

Profile sizes are upper bounds. Actual input lengths and occupied request slots
may be smaller. Prepared memory layouts and kernel variants support those
lengths; input capacity, prefill chunk budget and total context capacity are
separate limits. A512-token chunk budget permits a longer prompt to progress
through several chunks when the context profile allows it.

- `model.create_decoder` is the bounded batch1 Metal study factory, capacity64..4096.
- `metal_batch_study.create_decoder` is a separate experimental batch2/4 factory.
  Batch2 permits chunks2/4/8/16/32/64; batch4 remains chunk<=8. Wider shapes require
  native guarded workspace, shared scratch, in-place caches and per-row hidden
  support. Separate fixed-shape admissions do not admit all shapes resident at once.
  The default-off `qwen38_large_prefill` extension additionally exposes experimental
  batch2 chunks128/256 with signed16-bit attention indices and nonresident pack
  retirement. The opt-in alias-group/lifetime-reuse batch2/4K/chunk128
  indexed/mixed coding study passes8/8 tasks with all full token sequences
  matching the single-request reference,1,050 combined decode plans and53 audit
  corruption checks. Ready Metal allocation is17,297,309,696 bytes. Two cold C1
  and four C2/C4 serving checks pass accounting and31 corruption checks per
  archive. All14 serving outputs exactly match the preceding candidate, with498
  combined decode plans in the concurrent archive. These single-wave checks do
  not establish repeated performance or production admission.
  A separate alias-group/lifetime-reuse chunk256 profile also passes the unchanged
  eight coding contracts and full token equality. Its short coding prompts fit
  in one chunk, so their original audit retains two missing-coverage failures.
  Eight additional canonical short/long requests supply the missing multi-chunk
  and sparse-head coverage, with full token equality,16 interleavings and53
  corruption checks. A combined qualification audit requires identical native
  libraries, implementations and runtime profiles and passes18 more corruption
  checks. Prepared allocation is18,027,708,416 bytes; largest sampled allocation
  is18,038,292,480 bytes, within the configured guard. Each requested profile must
  still pass its native working-set guard. Repeated performance and production
  admission remain incomplete.
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
