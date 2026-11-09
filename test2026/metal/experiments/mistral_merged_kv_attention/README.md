# Read-only merged-KV attention screen

Rejected for broad native integration: the extra temporary K/V layout does not
provide a repeated long-prefix gain against the already-tested exact contiguous
softmax control. This is a standalone FP32 operator experiment, not a runtime
selector or xModel change. All raw commands, failures/preparations, input/output
parents, binaries and samples remain in the retained study
`out/research/mistral7b-local-coder/mistral-merged-kv-attention1`.

The candidate packs each owner's old prefix and live K/V into temporary buffers,
then uses one QK product, causal softmax and one PV product. Original input KV is
read-only. Admission uses the exact previously tested native contiguous-prefix
proof. Future/empty-owner materialized entries are zero. The two existing
controls retain four MPS products and respectively original or exact contiguous
softmax. All operands/results remain FP32; the existing FP64 attention bound is
unchanged. MPS reduction grouping can change output bits.

Prepare with `python3 prepare_shader.py ADMITTED_CLEAN_SOURCE FRESH_OUTPUT`.
It reuses the pinned standalone control preparation and appends this candidate.
Compile either `.mm` with C++20/O2/ObjC ARC, the admitted clean source's
`third_party` include directory, and Foundation/Metal/MetalPerformanceShaders.
The copied policy header is pinned byte-for-byte to its admitted native source.
Do not build the dirty Garnet working checkout.

Run `HELPER COMBINED_SHADER - FRESH_FIXTURES correctness` first.
Then `python3 audit_merged_attention1.py FRESH_FIXTURES` (requires NumPy).
Seven GPU cases cover B1/B4, T8/17/128, context128/256/2048/4096, ragged/empty
owners, gating, padding and parent offsets. The auditor independently reconstructs
all144,966,080 materialized parent words, checks complete output bounds and
28,672 FP64 values per variant, and rejects23 output corruptions.

Only after that audit succeeds, run
`HELPER COMBINED_SHADER AUDITED_FIXTURES FRESH_SCREEN screen`.
All pipelines/shapes are warmed before the first measurement; each cell has nine
rotating three-variant rounds. The two helper sources differ only by reversed
variant order; run them in separate strictly serial processes. Include query
packing and materialization in GPU spans. Finish all builds/hashing beforehand.
`python3 audit_screens1.py ABSOLUTE_STUDY` validates the saved two-process
collection, source/fixture bindings, actual process order and all378 observations.
Its schema names match the retained study; use the documented collector layout.
No comparator inference, GPU profiling or unrelated work may overlap timing.

B4/T128/4K full-owner paired contiguous-control/merged ratios are0.849/0.905.
At2K they are0.974/0.936, with substantial variation. Cold-prefix observations
favor two products but remain noisy and do not establish complete-prefill gain.
The B4/C4096 temporary K/V allocation is128MiB plus guards. Ragged timings use
full-row products and cannot replace native active-span measurements. Fewer
products do not guarantee a win once staging/shapes/dispatches change; these
experiments do not isolate copy time as the sole cause.

The generic native primitive must retain read-only input semantics. A future
no-copy rewrite requires an immutable graph proof of logical cache aliases,
liveness and original commit ordering before hoisting cache writes; it must
also qualify native failure/abort, prefix-return, batching, full-state/logit,
quality and serving behavior. This prototype supplies none of those admissions.
Serving tables and strict coding0/8 remain unchanged; the2x goal remains open.
