# Real trained Mistral MLP replay: exact through the down projection

This follows the fused gate/up–SiLU screen at checkpoint `60f4ac611`. The
Fast candidate now matches complete native MLP observations at layers0/15/31
on real512-row activations from the last full prefill chunk. No runtime
fusion, production default or serving benchmark changes in this study.
The retained study is `out/research/mistral7b-local-coder/mistral-fused-mlp-trained1`.

## Native observations

`prepare_capture1.py` copies the pinned clean half-prefill source and adds a
five-path diagnostic overlay. `GARNET_METAL_DIAGNOSTIC_MLP_CAPTURE` defaults
OFF and requires the existing Mistral half-prefill/layer-bulk profile. Source
changes are limited to CMake, compiled-graph and Metal-context declarations/
implementation. Existing shader headers and all xModel files remain exact.

The existing exact Mistral FFN matcher identifies normalized input, gate/up,
SiLU and down output. Five independent snapshots are allocated per selected
layer before inference. Bounded native blit copies observe values before
shared scratch can overwrite them. They do no floating-point arithmetic and
do not alter route counters. Snapshots use the owning command's normal
pending-output/readiness tracking and are released on Close.

A native unit checks18,948,096 complete parent words, including offsets,
guards, signed zeros, subnormals, infinities and NaN payloads. Eighteen invalid
operations reject aliases, wrong shapes, missing commands, uninitialized
sources, premature reads and cross-command ownership violations. Abort
invalidates results; recovery and zero native owners pass.

One untimed native model process runs the unchanged four frozen long prompts
with eight outputs each. It captures call16, positions1920–2047, with all512
actual token rows active. Native events and frozen token IDs bind this
boundary independently. Eight API gates reject unarmed, premature, duplicate,
wrong-layer/invocation and invalid-path capture operations.

All32 generated tokens/text and all260 complete physical KV/hidden digest
pairs match the frozen selected half-prefill engine exactly. Close reports
zero native owners and1,900,544 Metal bytes. Eight corruptions reject. These
are sampled memory/readiness checks, not peak-memory or general capture-API
admission. CoreAnalytics context-leak warnings remain in this run and the
frozen control; zero owner counters do not explain them.

## Exact native full-MLP replay

The replay uses captured normalized activations and exact trained gate/up/
down quantized weights. Original packed projections use Safe MSL4 with the
native512-row geometry. The candidate fuses gate/up, SiLU and direct half
down input, then uses the original Safe down GEMM. Shadow, normal and actual
atomic-writer variants are retained.

| Independently audited comparison | Words |
|---|---:|
| Original gate/up/SiLU/down FP32 versus native captures | 72,351,744 |
| Fused gate/up FP32 shadows versus native captures | 44,040,192 |
| Fused half down-input comparisons | 66,060,288 |
| Fused final down FP32 comparisons | 18,874,368 |
| Actual half-output writer-map entries | 22,020,096 |

All are exact, including guards. Normal fused execution leaves unused FP32
intermediates untouched, as checked by the collector. The independent audit
also recomputes243 FP64 projection samples including down GEMM under the
unchanged2e-6+4e-6*sum_abs allowance. The maximum sampled fraction is0.01723.
Nine corrupted reports reject.

## Compiler-setting correction

The earlier standalone screen explicitly selected MSL3.1 for SiLU. The
selected native build passes `compileOptions=nil`: its MSL3.1 assignment is
conditional on `GARNET_METAL_NATIVE_BFLOAT_BODY`, which is OFF. The old control
was a compiler-setting surrogate, not the exact native setting described
earlier.

This replay uses actual nil/default options and matches every original
FP32 result to native captures. A separate explicit-MSL3.1 check also matches
all observed SiLU FP32/half values at these three layers. That proves
numerical agreement on this data, not performance equivalence. Preserve the
previous timings as surrogate block comparisons; they do not establish a
new native runtime or serving gain.

## Remaining admission

Compiled-runtime fusion is unimplemented. It must prove that skipped
intermediates have no observable consumers, retain safe scratch/alias and
command ownership, preserve failure/Abort behavior, and pass full model,
continuous-batch and prefix-return/cache-pressure checks before new serving
measurements. No2× or half-TTFT target is achieved here. Strict coding remains
0/8. ANE/Gemma, multimodal and the Qwen goal remain open.

Both packaged auditors reproduce the retained reports exactly. The package
preparer reproduces the overlay and worker bytes without repeating inference.
Original11 dirty checkout paths, frozen comparators and xlang3 are untouched.
Hashes and failed compiler/auditor attempts are preserved.

Audit retained artifacts without inference, using fresh report destinations:

```sh
python3 -B audit_capture1.py ABSOLUTE_STUDY FRESH_REPORT
python3 -B audit_replay1.py ABSOLUTE_STUDY FRESH_REPORT
```

For a new change or unresolved concern, choose an unused sibling study under
the pinned `out/research/mistral7b-local-coder` directory:

```sh
python3 -B prepare_capture1.py FRESH_STUDY
python3 -B build_capture1.py FRESH_STUDY
python3 -B FRESH_STUDY/run_capture1.py
python3 -B audit_capture1.py FRESH_STUDY
python3 -B FRESH_STUDY/prepare_replay1.py
python3 -B compile_replay1.py FRESH_STUDY
python3 -B FRESH_STUDY/run_replay1.py
python3 -B audit_replay1.py FRESH_STUDY
```

Preparation/audits use the existing asset Python environment with NumPy and
safetensors. Native builds use Xcode/Metal on the same Apple10/M5 machine.
The recipe refers to pinned local evidence. Do not rerun these unchanged
correctness cases or any frozen comparator inference.
