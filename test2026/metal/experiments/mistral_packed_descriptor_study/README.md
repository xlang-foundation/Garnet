# Packed projection descriptor screen: correct, no useful speed gain

This isolated C++/Objective-C++/Metal experiment follows the half-prefill GPU
attribution. It tests relaxed MPP descriptors in the **current** packed
M128/M256, N64, K-stage128 kernels. The older reuse64 relaxed experiment already
found no meaningful large-shape gain; its result is preserved and not rerun.
This test changes neither geometry, arithmetic order, operand storage, input
gather, quantized decoding nor row selection. It changes only the descriptor
flag in copied candidate kernels. Original kernels and all runtime files stay
unchanged. No xModel, tokenizer, model artifact, cache or serving API changes.

The source generator verifies the native half-prefill checkpoint and three
original shader headers, then keeps their complete strict kernels alongside
the candidate in one Safe Metal4 library. Same-library guards check strict and
relaxed cooperative maps and an intentional capacity fault. Atomic counters
prove all1,280 cooperative operand/result coordinates have exactly one owner.
Separate writer variants change only the six output-store sites to atomic
increments, proving actual selected outputs have one writer while inactive
rows and full parent guards remain unchanged.

Frozen synthetic native capture fixtures cover gate/up, down and Q/O shapes.
Thirty-nine cells exercise actual counts128/129/255/256/257/383/384/511/512,
permuted/reversed indices and dense rows. Native tile selection is preserved;
both tiles execute in their actual eligible shapes. Every arithmetic call
matches the complete frozen native output bits, including guards and inactive
rows. The full14,680,064-byte compact-input arena, redzones, unused tail and
read-only input/weight/coefficient parents are checked after every call.
No short input is padded to the maximum logical row count.

The independent saved-data audit verifies150,532,096 arithmetic parent words,
the same number of writer-parent words and2,340 FP64 values at the unchanged
`2e-6+4e-6*sum_abs` bound. Maximum bound fraction is0.019113. Thirteen deliberate
metadata/output/guard/writer corruptions reject. These are finite synthetic
operator checks, not trained-model numerical or coding-quality qualification.

Only after that audit do two fresh serial processes collect opposite-order
timing cohorts: three shapes, three warmups and nine rotating paired rounds.
All pipelines specialize before measurement. Input gather is inside each GPU
span; CPU poison, readback, comparisons and FP64 checks are outside. There are
108 measured GPU commands and36 warmups in total. No builds, source hashing,
other task inference or reference inference overlaps collection. Postflight
source/asset checks pass, retained outputs stay exact, and ten deliberate
timing corruptions reject.

| 512-row shape | Strict GPU median, cohort range | Relaxed GPU median, cohort range | Strict/relaxed ratio |
|---|---:|---:|---:|
| Gate/up, K4096/N14336 | 4.891–4.957ms | 4.901–4.960ms | 0.998–0.999× |
| Down, K14336/N4096 | 5.515–5.520ms | 5.535–5.567ms | 0.991–0.997× |
| Q/O, K4096/N4096 | 1.693–1.701ms | 1.688–1.699ms | 1.001–1.003× |

Within-round ratios also fail to show a reliable useful benefit. Do not
integrate or rerun this candidate unchanged. This rules out the descriptor-only
change for these measured kernels; it does not establish an instruction,
occupancy or bandwidth explanation for the remaining projection cost. Earlier
XOR-swizzle/shared-stage, loop-control, full-tile, packed64 and bulk-product
studies must be reviewed before selecting a distinct next hypothesis.

The native half-prefill body gain remains1.18–1.22×. There is no new TTFT,
decode, throughput, HTTP, ANE or coding-quality result. The overall2× and
half-TTFT targets, Gemma and original Qwen objectives remain open. Frozen
vLLM/SGLang backends and baseline data are untouched.

Artifacts are retained in
`out/research/mistral7b-local-coder/mistral-packed-descriptor-study1`.
The generator, host, collectors and auditors reproduce this local study with
fresh destinations and the pinned sibling native/capture studies. Scripts take
an absolute study directory; auditors optionally take a fresh report path.
NumPy is required for the correctness auditor. Large payloads and binaries
remain local; the checkpoint contains sources, the complete generator and
assessment/hashes. An initial preparation assertion expected four store sites;
inspection established six across the two kernels. It failed before shader
creation or GPU execution; the corrected generator explicitly requires six.
