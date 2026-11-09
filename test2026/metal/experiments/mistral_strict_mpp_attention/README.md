# Strict FP32 MPP online attention

This standalone experiment directly investigates the M5 matrix/online-softmax
approach in pinned vLLM-Metal. Both candidate products use FP32 operands,
FP32 probability/accumulator/output and `relaxed_precision=false`, under Safe
Metal 4 math. It does not change Garnet runtime, xModel operators or defaults.
Evidence: `out/research/mistral7b-local-coder/mistral-strict-mpp-attention1`.

The first candidate uses public cooperative coordinates and threadgroup score
exchange. It passes the original FP64 bound but loses long-context speed.
The follow-up keeps scores in registers and uses XOR-1/XOR-8 row reductions.
Before using raw register indices, its SAME-library GPU guard verifies the
complete score/P-left/output maps. The attempted relaxed-style map correctly
rejects before attention. A separate public-coordinate probe identifies the
strict FP32 two-column/eight-stride map; all observed coordinates are checked
independently. The new guard preserves this contract and fails on incompatibility.

Both final candidates pass seven finite shapes, B1/B4, T8/17/128, 2K/4K extents,
gating, ragged/empty owners, real source/output offsets, padding and guards.
Each variant has 28,672 independent FP64 values under the unchanged
`1e-5 + 2e-4 * abs(reference)` bound, plus complete original-control error checks.
Every input and the prepared committed KV parents remain unchanged. Both MPS
controls match the prior qualified full payloads bit-for-bit. The original
four-product path sees old cache and live K/V; no-copy two-product MPS and the
candidate use the same CPU-prepared immutable committed cache. Cache preparation
is outside both timed paths; MPS query pack/softmax/unpack stays inside.

For each candidate, two fresh processes run strictly serially in opposite
rotating three-variant orders. Every shape/variant gets three warmups before
collection, then nine measured triples, 378 GPU observations per candidate.
All timed outputs match the corresponding retained full parent. Coordinate
coverage has exactly one writer per logical element. The result auditor rejects
ten metadata corruptions; numerical audits retain their separate guard/nonfinite/
inactive/truncation negatives. Failed first compilation and rejected mapping
are preserved, with no changed numerical bounds or precision switches.

| Register candidate vs no-copy MPS | Cohort 0 paired ratio | Cohort 1 paired ratio |
|---|---:|---:|
| B4/T128, cold prefix | 1.988x | 1.997x |
| B4/T128, 2K full owners | 0.979x | 0.937x |
| B4/T128, 4K full owners | 1.030x | 1.019x |
| B4/T128, 4K ragged/gated | 1.642x | 1.674x |

Ratios are medians of paired original/candidate GPU times, not ratios of medians.
Above one favors the candidate. These synthetic component results do not justify
a broad native selector: dense long-prefill benefit is absent/small, and this
MPS helper computes full physical owners/rows while native Garnet already skips
empty owners. Ragged gains can overstate actual native benefit. There is no
full-model speed/correctness, TTFT, decode, serving, coding-quality or 2x admission.
The unchanged full-model strict score remains 0/8; comparator results are frozen.

Prepare shaders from the hash-pinned admitted source (e.g. the selected
prefix-policy clean source), without modifying it:

```sh
python3 prepare_shader.py ADMITTED_GARNET_SOURCE FRESH_OUTPUT_DIRECTORY
```

`public.metal` and `register.metal` exactly reproduce the qualified sources.
`diagnostic-rejected-map.metal` is ONLY for `coordinate_dump1.mm`: its deliberately
incompatible register guard is expected to report 14 in every lane, and no
attention kernel is launched by that probe. Do not use it for attention.
Compile the `.mm` helpers with C++20/ARC, the admitted `third_party` include,
Foundation, Metal and MetalPerformanceShaders. `strict_attention2.mm` handles
the public-coordinate version; `strict_attention3.mm` handles the final guarded
register version. Retained compiler/collection commands provide exact flags.
The helper arguments are SHADER, `-` or audited fixture directory, fresh output
directory, and `correctness`, `screen` or `screen_reverse`.

Audit the retained evidence without GPU execution:

```sh
python3 audit_strict_attention1.py STUDY/correctness2
python3 audit_register_attention1.py STUDY/correctness4
python3 audit_screens1.py STUDY
```

Collectors are provenance templates bound to the saved study files and paths;
they do not download model assets or bootstrap those bindings. All GPU work is
serial. No vLLM/SGLang inference is repeated. The main optimization/qualification
goal remains open, including actual-model numerical/quality, cache/batch,
ANE/Gemma and later Qwen work.
