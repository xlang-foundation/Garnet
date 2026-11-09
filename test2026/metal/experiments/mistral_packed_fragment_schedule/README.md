# Strict fragment scheduling: exact, no runtime replacement selected

This experiment tests two concrete differences informed by the pinned MLX
source used with vLLM-Metal: explicit threadgroup RHS half4 loads, and K32
operand preloading with two increasing K16 updates per accumulator. The latter
changes interleaving between independent accumulators; their arithmetic order,
half operands/coefficient decode, strict M16/N32/K16 MPP operations and FP32
outputs remain unchanged. This differs from the earlier tile-size, descriptor,
branch-free and compiler loop-hint screens.

All kernels use the existing M128/M256 geometry,256 threads and17,408 bytes
of threadgroup memory. The original native packed kernel is preserved as the
timing control. A separate copied control checks the disabled writer observer.
Counted variants are used only for correctness, never timing. No runtime,
xModel, tokenizer, model, precision policy or frozen reference changes.

## Correctness

Real native layer0/15/31 norm inputs, SiLU inputs and gate/down outputs are
replayed against the pinned trained Q4/group64 weights. The4096×4096 shape
uses a prefix of the trained gate weights/output; it is explicitly a shape
proxy, not an actual Q/O capture. Both row tiles cover127/128/129/255/256/257/
383/384/511/512 selected rows, with a unique reordered physical-row map.
The127-row case probes the standalone kernel boundary; it does not enable
the native packed runtime below its128-row threshold.

All1,620 cases pass. An independent auditor reconstructs every complete
parent output and writer-map hash from the original captures:6,228,593,280
words in each category. Sixteen complete raw arrays remain available for
separate verification. Input/gather/output guards, inactive physical rows
and exact one-writer coverage pass. All243 independent FP64 samples satisfy
the unchanged2e-6 +4e-6×sum-absolute-products bound; maximum observed fraction
is0.018899. Ten deliberate audit corruptions reject.

## Timings and decision

Two fresh GPU processes run serially. Each mode/cell has five warmups and
twenty measured rounds. Rotating/reversing order places each mode in each of
five positions exactly four times per process. There are3,600 measured and
900 warmup samples, covering128/256/512 rows and contiguous/permuted maps.
All added source/model hashing and compilation precede the full collection;
postflight follows both terminal processes.

The table reports control time / candidate time for512 contiguous rows.
Values above1 favor the candidate; ranges span the two process medians.

| Candidate | Trained gate | Trained down |
|---|---:|---:|
| Explicit RHS half4 loads |1.001–1.001× |1.001–1.002× |
| Scalar RHS, K32 scheduling |0.942–0.942× |0.973–0.975× |
| Vector RHS, K32 scheduling |0.992–0.993× |0.982–0.984× |

**No replacement is selected.** Vector-only is essentially flat, and K32
variants regress on the important largest real MLP shapes. Some128-row cells
and the4096×4096 proxy improve; all observations are retained. Proxy gains
do not establish actual Q/O, full-model or serving gains, and the currently
selected fused gate/up producer was not modified by this screen.

Durations come directly from completed Metal command-buffer GPU timestamps
and include gather/projection plus inactive-output clearing when needed.
Only their millisecond differences are retained, not absolute start/end
timestamps. They are operator GPU durations, not wall latency or token speed.
Seven timing-evidence corruptions reject. The source comparison does not
prove which per-operator kernel the frozen reference actually executed.

Local evidence is `mistral-packed-fragment-schedule1`; the independent
preparation replay is `mistral-packed-fragment-replay-preparation1`, under
`out/research/mistral7b-local-coder`. The packaged generator reproduces the
qualified shader exactly, and both packaged auditors reproduce the complete
reports without inference. Original11 dirty files remain byte-identical.

The latest serving results remain the [fused MLP campaign](../mistral_fused_mlp_serving_benchmark/README.md).
No new serving inference is justified by this screen. Strict coding remains
0/8, and the2×/half-TTFT, production integration, ANE/Gemma and original Qwen
goals remain open. No vLLM/SGLang inference is repeated.

## Reproduction

The tools require the pinned sibling native/capture studies. Use an absolute
fresh destination. All GPU work is serial; timing requires the numerical audit.

```sh
python3 -B prepare1.py /absolute/path/to/mistral7b-local-coder/FRESH-STUDY
python3 -B /absolute/path/to/FRESH-STUDY/run_correctness1.py
python3 -B audit_correctness1.py /absolute/path/to/FRESH-STUDY /absolute/path/to/FRESH-STUDY/correctness-audit1.json
python3 -B /absolute/path/to/FRESH-STUDY/run_timing1.py
python3 -B audit_timing1.py /absolute/path/to/FRESH-STUDY /absolute/path/to/FRESH-STUDY/timing-audit1.json
```

Saved-data auditing uses the last two auditor commands with fresh report
destinations and executes no models or GPU kernels. Do not repeat unchanged
timings. A distinct follow-up can evaluate preparing tiled half-weight caches
once under an explicit compile-profile memory budget, retaining this strict
MPP consumer rather than the previously tested MPS cached/streamed product.
