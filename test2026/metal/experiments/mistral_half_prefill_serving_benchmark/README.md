# Native half-prefill serving: measured improvement, targets unmet

The opt-in Mistral half-prefill engine now has audited HTTP serving results
at one and four concurrent requests. This experiment changes no runtime,
xModel operator, production default, comparator source or frozen baseline.
The retained study is `out/research/mistral7b-local-coder/mistral-half-serving2`.

Mistral-7B-Instruct-v0.3 uses the pinned MLX affine 4-bit/group64 weights.
C1 has a 2,048-token prompt. C4 has frozen prompts of 2,048/2,051/2,048/2,049
tokens. Every response generates exactly128 tokens; prefix retention is off.
The native profile bounds B4/C4096/T128, global prefill budget512, and the
qualified private half-QKV/P16 attention policy. Canonical KV/decode remains
FP32; the frozen Apple references use FP16 state. These are different
numerical policies, and their thermal/OS conditions are uncontrolled.

## Warmed comparison

Each entry is the median of wave metrics. Native has four warmed waves per
cell from two fresh engines. Each frozen reference has15 measured waves per
cell from three process cohorts. Cold native observations are separate below.
TTFT is first nonempty content latency. Decode is per-request incremental
token rate after the first token. Aggregate output throughput includes
prefill and all requests in the wave.

| Concurrent requests | Engine | TTFT, seconds ↓ | Decode/request, tokens/s ↑ | Aggregate output, tokens/s ↑ |
|---:|---|---:|---:|---:|
| 1 | Garnet half-prefill | 4.012 | 17.300 | 11.258 |
| 1 | vLLM-Metal, frozen | 4.219 | 20.751 | 12.345 |
| 1 | SGLang-MLX, frozen | 4.159 | 20.737 | 12.518 |
| 4 | Garnet half-prefill | 15.606 | 12.623 | 19.871 |
| 4 | vLLM-Metal, frozen | 11.570 | 10.484 | 21.179 |
| 4 | SGLang-MLX, frozen | 10.884 | 6.307 | 16.337 |

The2× single-request decode,2× aggregate throughput and half-TTFT targets
are unmet against both references. C4 decode/request is2.001× SGLang's
median, but that is neither the single-request decode target nor aggregate
throughput. C1 TTFT ranges overlap the references, so the small median
difference does not establish a causal advantage. Full ranges and ratios
are retained in `comparison1.json`.

## First round in each native process

| Concurrent requests | Observations | Median TTFT, seconds | Median decode/request, tokens/s | Median aggregate, tokens/s |
|---:|---:|---:|---:|---:|
| 1 | 2 | 3.969 | 17.723 | 11.476 |
| 4 | 2 | 15.289 | 12.532 | 20.075 |

These first rounds follow readiness and engine preparation. They are not
hardware-cold measurements, and they are not compared as cold/cold reference
observations. Historical FP32 serving medians were slower at C4, but these
separate campaigns do not isolate a causal speedup from the new kernel.

## Integrity and remaining limits

- Four fresh engines run serially in order C1,C4,C4,C1. Each engine runs
  cold,warm1,warm2. No vLLM/SGLang inference is repeated.
- Preflight verifies55,546 input files, including exact native source and
  independently prepared source copies, trained weights, native binary,
  original11 dirty paths, correctness gates and frozen reference artifacts.
  No model/source hashing or additional profiling runs during timed waves.
  The unchanged client hashes its small request body after its submitted
  clock, as in the frozen baseline protocol.
- All30 native responses,3,840 generated IDs and complete text match the
  saved full-length FP32 controls. Prompt IDs, request bodies, SSE IDs,
  row epochs, scheduler bounds, actual dispatches and zero-retention policy
  are audited. There are6,336 half-attention layer dispatches and no executed
  MPS attention products or scalar FP32 chunk-attention fallback.
- Every worker exits0 through expected close. All native owner counters
  return to zero. Post-close allocations are720,896 bytes for C1 and
  1,245,184 for C4. Sampled working-set checks are not a peak-memory proof.
- Raw SSE clocks independently reconstruct TTFT, incremental decode rate
  and complete-wave throughput. All30 selected frozen reference waves are
  re-audited from saved streams for each engine, without executing models.
  Ten deliberate corruptions reject, including wrong tokens/prompts,
  stale epochs, missing half dispatch, MPS fallback, leaks, fake cache hits,
  false occupancy, TTFT and decode speed.
- The packaged auditor reproduces the entire retained audit byte-for-byte.
  Postflight verifies all preflight hashes after every engine is terminal.
- The unchanged strict coding suite remains0/8. Exact output regression
  is not a coding-quality pass. No model, production default, ANE/Gemma,
  multimodal or Qwen goal is qualified by this experiment.

The prior `mistral-half-serving1` attempt is retained separately. Its worker
rejected an inherited profile missing `prefix_capture_policy=prefill_only_v1`
before readiness or any timed request. The owned bridge was interrupted and
the collector terminated1. The corrected campaign uses a fresh destination
and adds equality checks against the previously admitted coding profile,
allowing only identifiers, documented scope and disabled cache budgets.
The C1 barrier was also corrected before the first launch to use actual
concurrency. Neither fix changes native inference arithmetic.

## Replay and future collection

Audit saved artifacts without inference, using a fresh report destination:

```sh
python3 -B audit1.py /absolute/path/to/mistral-half-serving2 /absolute/path/to/fresh-report.json
```

Preparing a new campaign requires the same pinned artifacts and an unused
study name directly under `out/research/mistral7b-local-coder`. Preparation
copies the exact frozen transport and admitted uninstrumented worker,
generalizes only C1/C4 selection and barrier width, and adds the explicit
native producer policy. Do not repeat collection unchanged without a new
change or unresolved concern. Baseline inference must remain frozen.

```sh
python3 -B prepare1.py /absolute/path/to/out/research/mistral7b-local-coder/FRESH-STUDY
python3 -B /absolute/path/to/out/research/mistral7b-local-coder/FRESH-STUDY/run1.py
```

The collector never reuses a campaign directory and stops on failure. It
waits each owned child to termination before the next wave and hashes only
before the full campaign and after all four terminal waves. The scripts
retain the workspace-specific original evidence paths. This is a
reproducible local experiment, not a portable serving product.

The projection/MLP path remains the next optimization priority, informed by
the earlier GPU attribution and the closed descriptor-only negative result.
