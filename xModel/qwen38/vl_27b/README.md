# Qwen3.8-27B qualification

The existing CUDA factories and research gates remain separate from the new
Metal bring-up. Initial full-model execution passes a bounded smoke;
**production qualification is incomplete**. Do not register
the qualification profile as a production catalog model or claim a speedup.

The draft Metal path consumes the pinned
`mlx-community/Qwen3.8-27B-4bit` conversion directly: affine INT4, group64,
eight low-nibble-first values per U32, with BF16 scales and additive biases.
It keeps projection matrices compressed and decodes only requested embedding
rows on the CPU. Native Metal kernels perform decode and tiled prefill
contractions with FP32 accumulation; this is not bit-identical MLX arithmetic.
Requested embedding rows round to BF16 before conversion to host FP32, matching
the output dtype inferred by MLX's quantized embedding from BF16 coefficients.
This rounding is separate from compressed projection accumulation. Boundary
goldens are prepared; real-row numerical parity still requires execution.

The conversion changes tensor names, convolution dimension order and certain
normalization weights. The native checkpoint adapter checks that representation
and uses zero normalization offset for its already shifted weights. Original
HF checkpoints keep their existing offset. The 27B model has an untied
`lm_head`; it must not project logits through the input embedding matrix.

The draft vision adapter maps `vision_tower` names and explicitly transposes
the small Conv3d patch projection from O,T,H,W,C to O,C,T,H,W. Header inspection
matches all 333 vision tensors, but numerical and end-to-end validation remain
pending and `create_vision` still rejects runtime admission. Text requests
continue to allocate no image encoder or dummy image.

Build integration is behind `GARNET_METAL_MLX_AFFINE_INT4=ON`, default OFF.
The initial `metal-qualification-profile.json` uses batch1 and capacity512 to
establish correctness without the current flat FP32 state's large capacity
copies. This restricted smoke profile is not an equivalent arm of the frozen
4096-context reference benchmark. The requested 4K/8K/16K profiles, continuous
batching, native prefix reuse and multimodal execution remain required work.

The `garnet_affine_int4_test` covers independent packing goldens, grouped
coefficients, matrix tails, batches1/2/4/8, prefill shapes, rejection and resource
lifetime. These are component gates; full checkpoint parity, coding quality,
memory measurements and frozen-baseline comparisons are separate. The native
package compiles and all eight configured component tests pass on this M5.
Seven benchmark/evaluator tests also pass, including sandbox containment and
early-exit rejection. Independent original-format linear attention and decoder
references pass for batches1/2/4, including inactive-row preservation and the
three-to-one value/key head ratio. The INT4-disabled control package also builds
and passes all seven configured component tests.
The opt-in request-row build passes nine component tests. Two reset-separated
79-input/four-output native runs produce identical token IDs, including after
small-row projection specialization. That smoke uses capacity512 and is not
the matched 4K benchmark, a coding-quality score, continuous batching, prefix
reuse or vision qualification.

`GARNET_METAL_REQUEST_ROWS=ON` adds synchronous native `reset_row` lifecycle
support. An independent mixed linear/KV batch4 fixture verifies A-to-E row
replacement while B/C/D retain their state, including pause/reactivation and
optional head prediction. This is a prerequisite for continuous scheduling;
the serving scheduler and native hybrid prefix capture/restore remain pending.
