# Exact decode threadgroup geometry screen

This study tests two and eight SIMD groups per threadgroup against the existing
four-group blocked decode kernel. Four adjacent output columns per SIMD group,
the word/lane traversal, FP32 fused operations, SIMD reduction, FP16 coefficients
and fixed32-bit profile indices stay unchanged. The only shader edit changes
the group-to-column stride; host dispatch width changes with it. The unmodified
scalar decode kernel independently anchors the blocked control and both
candidates. This is an isolated Objective-C++/Metal research executable, built
from frozen headers. It changes no runtime default or xModel operator.

The earlier blocked-column/vector-input screen is not repeated. The current
engine already has those changes. Here the distinct variable is threadgroup
width:64,128 or256 threads.

## Correctness

Frozen trained gate/up/down weights and activations from layers0,15,31 cover
actual4096/14336 dimensions, plus1/7/17/33-output tails. Ordinary1–4 rows and
all16 selected-row masks are tested in both padded and contiguous layouts.
The latter uses serving input/output strides. All2520 candidate/control cases
match complete FP32 output parents, including inactive rows, row padding and
outer guards:47,375,280 compared words. Read-only input/weight checks pass.

There are5292 independent FP64 checks against the original
`2e-6 + 4e-6*sum_abs` bound. The largest observed error uses0.00606 of that
bound. All420 contiguous legacy parents are retained and independently audited;
the padded arm retains two representative parents and all case digests. Fourteen
deliberate correctness-evidence corruptions reject. Both correctness processes
close with458,752 allocated Metal bytes.

## Timing and decision

Four fresh processes run serially, with opposite paired candidate order in two
warm and two pressure cohorts. Each configuration has five warmups and20 measured
rounds. The5040 measured samples cover gate/up/down, ordinary1/4 rows, indexed
single rows0/3, and indexed2/3/4 rows. All samples and wider-batch cells are
retained. Pressure runs execute a separate256MiB read/write command immediately
before each projection. Its GPU interval is excluded from the projection time;
this does not prove that every hardware cache is cold. OS/thermal state is
uncontrolled.

The integration screen was declared before inspecting timing metrics. For each
single-active-row placement, the sum of gate/up/down median GPU times must
improve by at least3% in all four processes, with no individual projection
regression exceeding2%. Neither candidate passes.

| Cohort | Two groups: control/candidate ratio | Eight groups: control/candidate ratio |
|---|---:|---:|
| First warm process, rows0/3 |0.9927–1.1046 |0.9991–1.0994 |
| Second warm process, rows0/3 |1.0043–1.0056 |0.9997–0.9998 |
| Both pressure processes, rows0/3 |1.0012–1.0038 |0.9950–1.0004 |

Ratios above1 indicate lower candidate operator time. The isolated approximately
10% first-warm observation does not repeat. Pressure results are essentially
flat. **Keep the four-group kernel; do not integrate either candidate or run
unchanged full-model/comparator benchmarks for them.** These operator ratios
are not token rates and do not change the saved serving comparison or2× goals.
All four timing processes close with720,896 allocated Metal bytes. Fifteen
deliberate timing-evidence corruptions reject.

The first serial timing pass prepared pipelines between configuration groups.
It remains retained as diagnostics. A new host prepares the complete64-pipeline
family before any measured configuration and rejects every preparation miss
after the barrier. The reported decision uses only this second campaign. No
source/model hashing or host compilation occurs during collection; integrity
checks bracket terminal process execution.

## Reproduction

Pinned sibling native-wide and trained-operand studies are required under
`out/research/mistral7b-local-coder`. The generator reproduces both shader
libraries and all three host versions byte for byte without inference. The
recorded runners preserve the collection commands. Run only changed candidates
when collecting new operator measurements.

```sh
python3 -B prepare1.py /absolute/path/to/mistral7b-local-coder/FRESH-SOURCE-REPLAY
python3 -B audit_correctness3.py /absolute/path/to/mistral-decode-threadgroup-screen1 /absolute/path/to/FRESH-CORRECTNESS.json
python3 -B audit_timing3.py /absolute/path/to/mistral-decode-threadgroup-screen1 /absolute/path/to/FRESH-TIMING.json
```

The packaged auditors reproduce both complete saved reports without inference.
Raw evidence lives in `mistral-decode-threadgroup-screen1`. The11 pre-existing
dirty paths remain unchanged. Prefix lifecycle passed in the separate
[wide-profile prefix study](../mistral_wide_half_prefix_lifecycle/README.md).
Coding quality, hybrid-state qualification, full serving targets, production
integration, ANE/Gemma and original Qwen goals remain incomplete.
