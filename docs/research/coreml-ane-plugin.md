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
