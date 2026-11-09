# Current verification GPU census and centered-Q4 pilot

This checkpoint profiles the current native target and tests a larger arithmetic
change. **Neither prototype is selected for integration.** The frozen vLLM and
SGLang baselines, existing native engine and original 11 dirty paths are unchanged.

## Current target GPU regions

The existing `prepare_gpu_counters` native API records calibrated direct encoder
timestamps while preserving normal graph fusions and submission boundaries.
No per-operation serialized profiler or new native build is used. The disabled
four-row-fusion library from the preceding checkpoint replays all target calls
from the saved first control coding stage: 1,125 verification frames and 54
ordinary fallback frames across eight cases. These are saved proposals for
diagnostics; the draft model is not executed.

All 3,052 output IDs and complete KV/final hidden states match that saved current
target. Every verification retains two commands, target then accepted-prefix
commit. The audit validates calibrated clocks, raw encoder intervals, all 225
projection encoders, both attention encoders per layer, logical dispatch/restore
counts and unchanged accepted outputs. Eight deliberate evidence corruptions
reject. Native owners release within the measured 4 MiB close bound.

| Region | Summed region unions / target GPU command time |
|---|---:|
| MLP gate/up/down projections | 50.97% |
| Verification score/value attention | 17.13% |

Direct encoder unions cover 95.27% of target GPU duration: 69,753.967 ms of
73,214.687 ms. Individual region unions can overlap; this table is not an
exclusive partition and its rows must not be added as disjoint elapsed time.
MPS-owned encoders and unattributed command gaps are not observed. Counter
overhead is unmeasured; these values are not serving performance, TTFT, coding
quality or a speedup comparison. The auxiliary non-verification command count
was corrected in a second saved-data auditor without rerunning inference.

## New centered group64 arithmetic policy

The prototype factors the affine 4-bit dot within each 64-value quantization
group. It chooses integer `z = rint(-bias/scale)` only if the ratio is in
`[0,15]`, otherwise zero, and sets `residual = fma(z, scale, bias)`. A small
preallocated scratch buffer holds each input row's sequential FP32 group sum.
Each output then combines the centered integer dot with that shared sum:

```text
scale * sum(input * (quant - z)) + residual * sum(input)
```

Inputs remain FP32 and coefficient bits remain FP16. There is no new weight
copy or operand rounding. Accumulation order changes, so this is a distinct
arithmetic policy, not a lossless replacement for the original FP32 traversal.
A future engine would need its own producer identity and complete ordinary,
verification, cache and coding qualification. No runtime operator is changed here.

Two standalone Metal pilots use trained layer 15 gate and down projections,
ordinary one/four-row controls, selected occupancy-one rows and output tiles of
one/four columns per SIMD group. Pipeline function constants bind actual shapes
and row counts before collection. Command time includes input-sum preparation.
All original scalar/blocked control shaders remain present and unchanged.

The pinned C++ host reports zero coefficient-reconstruction failures across all
16 nibble values, zero original-bound failures over 276,480 output comparisons,
zero fixed-one-row/batch word differences and 1,033,088 parent words checked per
pilot. Its independent CPU FP64 oracle uses the original dequantized FP32 weights
and unchanged `2e-6 + 4e-6 * sum_abs` bound. It also checks shared-sum words and
source/weight/guard integrity. Candidate output bits differ from the old FP32
control, as expected. The second prototype vector-loads packed words and unrolls
the word loop; every candidate output hash must equal the first prototype.

Each pilot records 252 measured projection commands after five warmups, with
21 rounds per cell and rotating/reversed method order. These are preliminary
single-process observations; full masks, tails, offsets, adversarial inputs,
other trained layers, independent retained-word replay and repeated warm/pressure
timing are not qualified. Raw output word parents and GPU clock endpoints were
not retained, so the saved-data auditor validates coverage/order and recomputes
medians, but does not independently replay the host's numerical assertions.
Seven evidence corruptions reject. Both pilots release their Metal allocations
within the 4 MiB close bound.

| Prototype / tile columns | Gate, 1 row | Down, 1 row | Gate, 4 rows | Down, 4 rows |
|---|---:|---:|---:|---:|
| Original group64, 1 column | 1.046× | 1.035× | 0.496× | 0.445× |
| Original group64, 4 columns | 0.990× | 0.914× | 0.933× | 0.785× |
| Vector weight loads, 1 column | 1.090× | 0.980× | 0.582× | 0.475× |
| Vector weight loads, 4 columns | 0.975× | 0.908× | 0.298× | 0.299× |

Ratios are original control median / candidate median; higher is better. Large
four-row regressions persist. **Do not integrate this arithmetic change or infer
a model speed gain from it.** Register pressure or memory-transaction causes are
not established by these timestamp-only observations. The initial compiled host
used the wrong control function name; source inspection corrected it before any
GPU pilot, and both source/binary versions remain retained in the raw study.

## A profile gap affecting the actual 2K benchmark

The earlier learned-coder experiment uses a 512-token draft profile and a guard
requiring `draft_prompt_tokens + 8 <= 512`. CPU-only tokenization of the matched
2K target workloads shows that all five requests already exceed that limit:

| Target workload | Full draft prompt | Fixed 192-user-token tail with chat framing |
|---|---:|---:|
| C1 request | 1,664 tokens | 221 tokens |
| Four C4 requests | 1,664–1,667 tokens | 221 tokens each |

Thus the existing full-prompt proposer is not initially eligible for these long
benchmark prompts, even though its eight short coding cases fit. This report
opens no target output/control file and runs no model. The bounded tail is a
new unmeasured proposer policy. The next experiment should keep the target's
complete prompt and 4K context, keep intact draft chat framing, append only the
already-generated target prefix and retain native resident reconciliation and
the 512-token fallback guard. Only the helper draft sees a bounded prompt.
Target verification must still establish exact complete target outputs/state,
and actual matched 2K serving is required before any comparison claim.

This checkpoint establishes no new default, native runtime change, hybrid
recurrent/MM coverage, coding-quality admission or 2× result. The original
Qwen objectives and remaining Mistral/Gemma/ANE work remain open.

## Source reproduction and saved replay

Pinned sibling studies under `out/research/mistral7b-local-coder` are required.
Use the existing `qwen-local-coder/asset-env/bin/python` environment for
`tokenizers` and Jinja dependencies. No packages need installing.

```sh
python3 -B prepare_fresh1.py /absolute/path/to/mistral7b-local-coder/FRESH-PROFILE /absolute/path/to/mistral7b-local-coder/FRESH-PILOT
python3 -B replay_saved1.py /absolute/path/to/FRESH-REPORT-DIRECTORY
```

Preparation reproduces every source byte; only fresh host compile-command paths
are relocated. It builds nothing and executes no inference. Saved replay runs
the complete corrected profile auditor, preliminary pilot auditor and CPU
admission check, requiring exact report equality without inference.

For an explicitly needed fresh collection, run fresh profile `launch1.py` then
`audit2.py`. Compile both fresh standalone hosts with `build_pilots1.py
FRESH-PILOT`; run fresh pilot `run1.py`, then `run2.py`, then `audit1.py`.
Keep GPU processes serial. Do not rerun unchanged inference merely to replay
saved evidence. Production integration is intentionally not enabled for the
rejected prototypes.
