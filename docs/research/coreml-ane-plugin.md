# Optional native Core ML / ANE plugin

`GARNET_APPLE_COREML=ON` adds `garnet.coreml_session()` alongside Metal. It is
OFF by default. Model loading, prediction, tensor transport, persistent state,
reset and snapshot/restore are C++/Objective-C++; Python/coremltools is used only
to prepare offline model artifacts. No Rust runtime or tokenizer is introduced.

This is an initial functional plugin, not Mistral support or ANE performance
qualification. A compiled `.mlmodelc` asset is required; the session does not
export or compile an xModel graph. Mistral conversion, useful ANE placement,
profile-prepared model families, native continuous-batch/prefix-cache integration
for Core ML, and xMind routing remain pending. Metal retains its existing native
scheduler/cache implementations and measured comparator baselines.

## Native contract

Open a precompiled ML Program with a JSON profile containing:

```json
{
  "compute_units": "cpu_ane",
  "function": "main",
  "maximum_io_bytes": 2048,
  "maximum_state_bytes": 1024,
  "require_ane_plan": false,
  "inputs": {"x": {"dtype": "float16", "maximum_shape": [1, 32, 1, 16]}},
  "outputs": {"y": {"dtype": "float16", "maximum_shape": [1, 32, 1, 16]}}
}
```

`cpu` selects CPUOnly; `cpu_ane` selects CPUAndNeuralEngine, excluding GPU.
Supported transport types are float16, float32 and int32. Actual shapes must be
positive, have the profiled rank and fit every maximum dimension. Core ML also
checks its own compiled model constraints. These maxima bound transport; they
do not prove that flexible shapes avoid internal Core ML compilation. The
initial fixture has fixed shapes.

Profiles name every required tensor input/output and reject optional/non-array
features. Declared maximum I/O bytes are checked before execution; declared
state bytes are checked before allocating the session's state. These are tensor
bounds, not a total memory cap: weights, compiler/driver allocations, duplicate
snapshots and host transport staging are additional. In particular, FP32 host
transport takes twice the bytes of its FP16 model representation.

`predict(inputs)` uses exact declared transport types. `predict_float32(inputs)`
explicitly allows FP32 host tensors for FP16 inputs, converts with the native
round-to-nearest-even codec, and expands FP16 outputs to FP32; int32 features
remain int32. This avoids requiring low-precision Python tensor construction.
Core ML still computes with its compiled model types. Input copy/conversion is
outside `last_prediction_and_output_copy_ms`; that diagnostic is not complete
request latency or a performance benchmark.

Each session owns its `MLState`. Prediction and state access are serialized.
Snapshots own independent raw typed bytes; restore validates every state and
fills a fresh state before publishing it. Invalid input/restore calls leave
current state unchanged. Prediction/output failures poison the session until
`reset()` creates fresh state. `close()` is idempotent. The current state owner
is a whole model session; no row-level lease or cache-hit claim follows from it.

`report_json()` includes the selected function's anticipated/preferred device
counts from `MLComputePlan`. Unknown operations are retained. With
`require_ane_plan=true`, loading fails if no operation prefers ANE. This is a
plan check, not observed hardware execution; neither compute-unit selection nor
a preferred-device count proves actual ANE residency or a speed gain.

## Initial M5 /24 GiB evidence

The offline fixture is a tiny FP16 stateful accumulator, identity convolution
and half scaling, compiled with coremltools9.0 without loading a Python inference
model. Native C++ executes predictions under CPU and CPU+ANE policies in serial.
`mistral-coreml-draft1/ctest.log` records19/19 tests: the18 prior native tests and
the new Core ML state reference. The latter passes6,193 checks for independent
exact outputs, state ownership/reset/restore, invalid control rejection, budgets
and rejection of a required-ANE plan when none is available.

The Garnet API check in `coreml-runtime-api2.json` passes3,079 checks, including
raw FP16 snapshot handles, explicit FP32 transport, known independent half
rounding, invalid dtype/extent/state rejection and calls after close. Its loaded
library is `mistral-coreml-library2/libgarnet.dylib`, SHA256
`67cccac769be863949dd7d24f73ffc1c0d790ad66346aeec70023ce9555a063e`.
The source snapshot is `mistral-coreml-source2`; the captured test driver adds
negative cases after the earlier snapshot's copy. The final native runtime
source is identical between the two Core ML builds; the second build adds the
explicit transport wrapper. Three serial-engine guard tests also pass.

For this fixture, both compute policies report5 CPU-preferred operations and7
unknown operations, with zero ANE-preferred operations. Thus this evidence
establishes Core ML execution and state/API correctness, not ANE acceleration.
The first API attempt stopped before prediction because the XLang list
constructor intentionally rejects low-precision payloads; the explicit native
FP32 transport method fixes that boundary. The earlier library/probes remain
preserved rather than overwritten.

These APIs require macOS15+. This qualification used macOS26.6 on Apple M5;
the local JPEG/PNG dependencies target macOS26, so this build does not establish
deployment compatibility with older macOS versions.

## Reproduction

Use an isolated offline conversion environment with coremltools9.0, then run
`test2026/coreml/build_state_fixture.py OUTPUT_DIRECTORY`. Configure the existing
Metal/xModel build with `GARNET_APPLE_COREML=ON` and
`GARNET_COREML_STATE_TEST_MODEL=/absolute/path/state_fixture.mlmodelc`.
`garnet_coreml_state_runtime_test` is the native reference executable.

The XLang API check is `test2026/coreml/test_runtime_api.py`; its arguments are
the absolute library directory, source directory, compiled fixture directory and
a new output JSON path. Remove DYLD library overrides when selecting an isolated
library; the test verifies `native_library_path()` before running.

Apple references: [stateful models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)
and [MLComputePlan](https://developer.apple.com/documentation/coreml/mlcomputeplan).
The implementation was checked against the installed public Core ML SDK headers
for state serialization, buffer lifetimes and anticipated device usage.

## Actual-weight Mistral MLP study

The optional `garnet_mistral_mlp_probe` now compares one layer0 MLP with native
Metal and Core ML CPU / CPU+ANE policies. This is an operator study, not an
end-to-end model result. It needs the affine INT4, FP16 coefficient, FP16 MPP
and prepared affine profile build options. The Python exporter prepares offline
artifacts; native C++ executes every timed prediction. No dummy image is used:
the channel-first four-dimensional array holds text hidden values for1x1
convolutions.

Model: `mlx-community/Mistral-7B-Instruct-v0.3-4bit`, revision
`a4b8f870474b0eb527f466a03fbc187830d271f5`, safetensors SHA256
`d7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d`.
All176,160,768 FP16 weight words across the three projections match the native
dequantizer exactly. Full-K LUTs retain the original Q4 code and per64-element
FP16 scale/bias through exact16-value palettes. Dense FP16 uses the same decoded
weights. Tiled LUT6/K256 and LUT8/K64/output-group16 representations independently
check the same entire weight set after index/palette remapping; they introduce
no additional weight quantization.

The128 input rows are frozen prompt embeddings, optionally RMS-normalized with
the actual post-attention normalization weight. They are a proxy, not actual
post-attention hidden activations. All formats use the same normalized input
(RMS0.24167). Native Metal uses FP16 prefill operands and FP32 accumulation,
intermediates and output. The ordinary Core ML graph uses FP16 intermediates and
output with framework-selected accumulation. Host timing includes input/output
layout and dtype conversion, prediction and native I/O; it excludes model open
and compute-plan inspection. Three warmups and nine measurements run in each
fresh process, with two serial cohorts in reversed engine order.

Normalized-input results on this M5 /24GiB Mac:

| Representation / compute policy | Host median, cohorts0 /1 (ms) | Relative RMS error vs Metal | Convolutions preferring ANE |
|---|---:|---:|---:|
| Metal |10.420 /10.396 | Reference | N/A |
| Full-K LUT / CPU |20.943 /20.909 |2.097% |0 /3 |
| Full-K LUT / CPU+ANE |21.165 /21.058 |2.512% |0 /3 |
| Dense FP16 / CPU |21.156 /20.920 |2.097% |0 /3 |
| Dense FP16 / CPU+ANE |16.685 /16.633 |7.236% |3 /3 |

Every engine's output is bit-identical between its fresh cohorts. Full-K LUT
and dense CPU outputs are also identical. A separate FP32 diagnostic graph
keeps exact FP16 weights and down-projection operands, while computing the
projections and activation in FP32. Its CPU result has0.0202% relative RMS error
and a32.139ms host median (one process). This supports the weight/layout mapping
and implicates arithmetic precision in the ordinary graph's larger error; it
does not identify a specific framework accumulation implementation.

Tiling the inner dimension to256 with balanced partial sums improved CPU
relative RMS error to0.0939%, at18.884ms vs10.480ms for its Metal control. This
is only a partial cohort. Its CPU+ANE child exited1 after906.97s with
`Core ML compute-plan inspection timeout`; it produced no prediction result.
The collector stopped and retained the failure. The second tiled representation
and reversed cohort were not executed. The timeout is not a zero-latency sample
or evidence that the tiled model is unsupported. Model load precedes the
30-second asynchronous plan-inspection timeout and can itself take much longer.

The earlier unnormalized embedding study is retained as an underflow stress
case: Metal output RMS was4.44e-7 and Core ML relative RMS errors were25.9–33.7%.
Its absolute errors were small, which is why relative error and input scale are
reported together. No tested Core ML path has established useful Mistral speed
or numerical quality. Anticipated device plans do not prove hardware execution;
that collection did not include an execution trace. The subsequent trace study
below addresses model-labelled ANE activity. A system ANE compiler service was active
during this work, so these descriptive operator timings do not establish fully
isolated hardware performance. No frozen vLLM-Metal or SGLang result was rerun.

Evidence under the external `out/research/mistral7b-local-coder` directory:
`mistral-coreml-mlp-lut128-campaign1`,
`mistral-coreml-mlp-normalized-campaign1` (both complete),
`mistral-coreml-mlp-fp32-control128-run1` (diagnostic only), and
`mistral-coreml-mlp-tiled-campaign1` (failed partial collection). Their raw model
artifacts, inputs, outputs, logs and process identities are preserved separately.
`audit_mistral_mlp_probe.py` rejects incomplete collections, checks serial cohorts,
artifact hashes, independent FP64 Metal samples, output finiteness and cohort
repeatability. Its collection pass deliberately does not assert numerical
quality, observed ANE execution or whole-model performance. Eight positive and
tampered-evidence controls exercise those boundaries.

Reproduce with `test2026/coreml/export_mistral_mlp.py`,
`run_mistral_mlp_probe.py` and `audit_mistral_mlp_probe.py --help`. The exporter
requires pinned native-decoder assets; the collector rejects concurrent engines
and changed artifacts. `--comparison-export` compares two representations with
one shared native Metal control per cohort. Runtime reports retain per-operation
preferred/supported devices and use a contiguous memcpy fast path for matching
MLMultiArray layouts, with a stride-aware fallback.

## Observed ANE activity and a channel/range profile

The next study uses Xcode Instruments27.1's `Core AI` template and exports its
`ane-hw-intervals` table. Two serial collections, each launching dense CPU+ANE,
dense CPU-only and LUT CPU+ANE controls, reproduce the same output bytes as the
earlier uninstrumented collection. The dense graph has12 model-labelled ANE
prediction intervals, matching three warmups plus nine native predictions;
the CPU-only control has zero. The LUT graph also has12 intervals, despite zero
convolutions preferring ANE in its plan. Its SiLU and multiplication prefer ANE;
its three convolutions prefer CPU.

In the collector-qualified second trace collection, the dense ANE interval
median is15.793ms, while the LUT interval median is0.218ms. Instrumented host
medians are17.896 /24.428ms; CPU-only is24.604ms. These measurements include
instrumentation effects and do not replace the uninstrumented latency tables.
They support observed model-labelled ANE activity and distinguish limited LUT
offload from the larger dense graph. The table does not expose a per-row PID
or hardware-cycle utilization. Attribution uses the model labels, launched
target arguments, exact prediction counts and CPU-only negative control;
per-operation residency and pure ANE compute time are not established.

An additional offline profile splits the14336 intermediate channels into seven
2048-channel branches. Each branch computes gate/up, SiLU/product and its down
projection contribution; a balanced sum combines the contributions. This
produces21 convolutions and covers every original weight once. An explicit
power-of-two range adjustment multiplies the up-projection weights by64 and
applies the reciprocal scale to the final output. Every one of58,720,256 up
weight values roundtrips bitwise through the scale/inverse scale. Together
with the176,160,768 original native-dequantizer weight checks, this establishes
the base values and reversibility. Exported scaled up-weight bits differ from
the base bits; the transformation preserves the real-valued MLP algebra while
changing FP16 intermediate range and rounding. It is not an exact FP32
execution replacement, and safe ranges for arbitrary activations are unproven.

The final direct collection holds input, original model/weight identity,
native probe and engine policy fixed and compares the original dense export
with the combined2048-channel/64x profile. Ten fresh native processes run
serially in two reversed cohorts, each with three warmups and nine samples.
The input remains128 normalized embedding proxy rows, not actual
post-attention activations or a complete model request.

| Final direct operator comparison | Cohort0 /1 host medians (ms) | Relative RMS error vs Metal |
|---|---:|---:|
| Metal control |10.358 /10.364 |Reference |
| Original dense Core ML CPU |20.938 /20.998 |2.097% |
| Original dense Core ML CPU+ANE |17.587 /17.353 |7.236% |
| Channel/range Core ML CPU |20.351 /20.299 |1.972% |
| Channel/range Core ML CPU+ANE |14.310 /14.121 |1.104% |

The candidate is1.2291x /1.2289x the original ANE operator speed, or about18.6%
lower latency. Metal remains faster. Complete outputs repeat bitwise between
cohorts within each profile. This is a two-cohort operator result, not a
statistical serving qualification, full-model ANE gain, coding-quality
admission or achievement of the2x target.

The staged studies are also retained. Channel tiling alone improves ANE
latency but leaves error near7.236%; adding64x range scaling reduces it to
1.104% with negligible timing change. Increasing to256x only reduces error to
1.090%, with no demonstrated speed advantage, so64x remains the research
candidate. The range-scaling evidence supports a range-loss explanation; it
does not identify a specific internal accumulator or prove the precise
location of underflow. A separate trace of the selected-source64x candidate
records12 model-labelled ANE prediction intervals with a12.262ms interval
median. All21 convolutions prefer ANE; its instrumented output matches the
uninstrumented candidate bitwise. Per-operation residency remains unproven.

The optional exporter flags are `--intermediate-tile 2048 --activation-scale
64`. Existing defaults remain unchanged. The selected exporter reproduces
the original default and measured candidate protobuf semantics, raw weight
files and input exactly. Protobuf map serialization is not byte-deterministic;
the proof compares complete parsed protobufs and deterministic serialization,
plus raw weights/input. Four additional serial native smoke probes reproduce
the original CPU/CPU+ANE and candidate CPU/CPU+ANE outputs bitwise. No C++
runtime API, xModel operator, default inference route or serving library is
changed by these offline flags. Core ML model-level scheduling, KV/prefix
ownership, full-model quality/memory and xMind integration remain open.

Reproduction tools are `run_mistral_ane_trace.py`,
`audit_mistral_ane_trace.py`, `test_mistral_ane_trace.py`, the extended
`export_mistral_mlp.py` / `audit_mistral_mlp_probe.py`, and
`test_mistral_channel_profile.py`. The trace collector accepts `--probe`,
`--checkpoint`, `--dense-export`, `--lut-export` and a new `--output` directory.
Audit it with `audit_mistral_ane_trace.py CAMPAIGN --output NEW_JSON`; its
reference outputs are the retained normalized campaign beside that directory.
The trace negative controls reject21 corruptions, including unresolved XML
references, missing predictions, CPU-only ANE events and changed outputs.
Fourteen profile controls reject inconsistent tile/scale/coverage claims;
the eight existing collection controls pass for both legacy and new data.

Evidence roots are `mistral-coreml-observed-trace1` (initial manual capture),
`mistral-coreml-observed-trace3` (collector reproduction) and
`mistral-coreml-channel-tiles1` (four serial paired campaigns, exports,
selected-source proof/smokes and candidate trace). The failed trace2 collector
preflight is retained: an older export omitted the explicit arithmetic field;
the collector now accepts its established FP16 default. No inference ran in
that failed attempt. Frozen vLLM-Metal/SGLang results are not rerun or replaced.

The aggregate post-run `mistral-coreml-channel-tiles1/provenance1.json` SHA256
is `0566e22a4a00cee0af990a910c3126a4e2e52d13a327ec0fc89f37ee3f9fcec1`.
It binds final/staged reports, retained source copies, selected-source checks,
trace-bundle file manifests and protected-source/baseline hash checks. Two
invalid exporter flag combinations are rejected before output creation.
The protected serving DLL, target checkpoint and frozen comparator summary
remain unchanged. Pre-existing dirty work is preserved separately.
