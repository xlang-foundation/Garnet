# Compensated half MPP attention: accurate, slower

This standalone C++/Objective-C++/Metal study tests two-half decomposition
against the existing FP32 attention contract. Evidence is retained under
`out/research/mistral7b-local-coder/mistral-compensated-mpp-attention1`.
No native runtime/default, xModel operation or reference engine is changed.

Scaled Q/K/V and unnormalized online-softmax probabilities are split into high
and low half components. Main and correction products accumulate separately in
FP32, with FP32 softmax and final output. Q/K/V use an exact scale of 16;
probabilities use 32768. Three products omit low×low; four retain it. The earlier
1024 probability-scale draft is retained locally and was never collected.
This is different from the previous compensated BF16 affine-verification study.

Actual half/half/float cooperative coordinates and the raw score/P/output maps
are checked in the same compiled library before attention. Capacity checks alone
are insufficient. Strict descriptors and Safe Metal4 math remain. Prototype
inputs must be finite and bounded by 2048; no general fallback or runtime input
domain admission is claimed. The official MSL conversion rule and tiny-P GPU
cases are recorded locally; no assumption about all tensor arithmetic preserving
subnormals substitutes for the actual GPU tests.

Ten synthetic cases cover B1/B4, cold/2K/4K prefixes, ragged/empty owners, gating,
padding, offsets, a half midpoint and tiny probabilities. Four products pass all
original bounds. Three products fail midpoint case7, reaching 1.873× the fixed
FP64 tolerance, and remain rejected despite passing the trained fixtures.
The independent bound stays `1e-5 + 2e-4*abs(FP64 reference)`.

Trained inputs come bit-for-bit from the qualified original native Mistral
capture at layers0/15/31, B4/T128/C4096, offset1920. All complete outputs are
compared with captured native gold; 18,432 FP64 values are checked per variant,
including native gold. The original four-product and no-copy two-product MPS
outputs retain 12,595,392 exact parent-word pairs from the preceding replay.
Four-product compensated attention passes, but this is an operator gate rather
than full-model quality, cache-policy or serving qualification.

Two fresh processes run serially with opposite rotating orders. Each prepares
all pipelines/inputs before three warmups per case/variant and nine measured
triples. Only variants0/1/3 are dispatched during timing; rejected variant2 is
excluded. Timed outputs must equal their retained correctness payloads exactly.
Read-only parents, immutable committed cache and guards are checked outside the
GPU timestamp spans. CPU preparation of the committed cache is outside both
no-copy MPS and candidate timing. No reference inference, source hashing or build
overlaps timed collection.

| Layer | Cohort | Original MPS ms | No-copy MPS ms | Compensated ms | Paired no-copy/candidate |
|---|---|---|---|---|---|
| 0 | 1 | 9.629 | 7.195 | 21.902 | 0.3335× |
| 15 | 1 | 8.777 | 7.837 | 22.230 | 0.3496× |
| 31 | 1 | 8.881 | 9.400 | 22.088 | 0.4150× |
| 0 | 2 | 9.807 | 9.528 | 22.198 | 0.4169× |
| 15 | 2 | 8.693 | 7.205 | 22.062 | 0.3413× |
| 31 | 2 | 8.650 | 8.410 | 21.916 | 0.3877× |

These are component GPU medians, not token speed or HTTP TTFT. The median of
paired ratios differs from the ratio of medians. All six comparisons regress;
the candidate is not integrated. Extra arithmetic is explicit in the source,
but the measurements do not establish a hardware occupancy/stall explanation.
The timing auditor checks 162 measured commands and rejects14 metadata/sample
corruptions. No useful-coding, ANE/Gemma, 2×/half-TTFT or subsequent Qwen goal is
qualified; the authoritative serving table remains unchanged.

Prepare exact source without building the dirty checkout:

```sh
python3 prepare_shader.py ADMITTED_GARNET_SOURCE FRESH_OUTPUT_DIRECTORY
```

The helper delegates the adjacent strict-attention preparation and appends the
hash-pinned compensated shader. It copies both native hosts and the policy
header. Compile C++20/ARC with the admitted `third_party` include directory and
Foundation/Metal/MetalPerformanceShaders frameworks, as in retained compile
commands. Host1 accepts `SHADER - FRESH_OUTPUT correctness` for synthetic data,
or `SHADER TRAINED_FIXTURE FRESH_OUTPUT trained` for retained trained inputs.
Host2 accepts `SHADER AUDITED_CORRECTNESS FRESH_OUTPUT forward|reverse`.

For CPU-only replay, the numeric auditor takes `STUDY/synthetic1` or
`STUDY/trained1` and writes `audit2.json`/`payload-hashes2.json`, preserving the
original audit1 evidence. Binding and timing auditors take the absolute study
directory. They retain the explicit workspace/adjacent trained-fixture layout;
they are research replay tools rather than a generic public runtime API.
The serial collector also takes that study directory and requires fresh timing
destinations. Do not rerun it against already collected cohort names.
