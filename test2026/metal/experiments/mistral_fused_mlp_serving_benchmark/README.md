# Native fused MLP serving: modest medians, targets unmet

Mistral-7B-Instruct-v0.3 uses the pinned MLX affine4-bit/group64 weights.
The native experimental fused MLP engine has complete-model, cache/batching,
pressure and unchanged coding-output regression checks. The retained campaign
is `out/research/mistral7b-local-coder/mistral-fused-mlp-serving1`.

All four fresh native engines run serially in C1,C4,C4,C1 order, each with
first,warm1,warm2 rounds. C1 has2048 prompt tokens; C4 has2048/2051/2048/2049.
Every response generates128 tokens. Prefix retention and state/GPU profiling
are off. Original small-request hashing remains inside the unchanged client;
all added source/model/artifact hashing occurs before collection and after
all four engines are terminal. No comparator inference is repeated.

## Warmed comparison

TTFT measures first nonempty content. Decode is per-request incremental token
rate after the first token. Aggregate output throughput includes prefill and
all requests in the wave. Native has four warmed waves per cell from two
fresh engines; each frozen reference has15 measured waves per cell from
three cohorts. Thermal/OS conditions are uncontrolled. Native canonical
KV/decode is FP32 with half-prefill/fused MLP; reference state is FP16.

| Concurrent requests | Engine | TTFT, seconds ↓ | Decode/request, tokens/s ↑ | Aggregate output, tokens/s ↑ |
|---:|---|---:|---:|---:|
|1 |Garnet fused MLP |3.829 |17.536 |11.552 |
|1 |vLLM-Metal, frozen |4.219 |20.751 |12.345 |
|1 |SGLang-MLX, frozen |4.159 |20.737 |12.518 |
|4 |Garnet fused MLP |14.542 |12.531 |20.695 |
|4 |vLLM-Metal, frozen |11.570 |10.484 |21.179 |
|4 |SGLang-MLX, frozen |10.884 |6.307 |16.337 |

**The2× single-request decode,2× aggregate and half-TTFT targets remain
unmet against both references.** At C4, native aggregate is0.977× vLLM and
1.267× SGLang. C4 decode/request is1.987× SGLang, which neither reaches2×
nor establishes the single-request target. C1 TTFT ranges overlap the
references; the small median difference is not a proven causal advantage.

The preceding half-only native campaign reported TTFT medians4.012s at C1
and15.606s at C4. The present medians are lower, but these separate campaigns
do not isolate a causal fusion speedup. Full ranges, cold observations and
ratios remain in the raw audit. First rounds follow engine preparation and
are not hardware-cold measurements.

## Integrity and limits

- Preflight verifies55,340 bound inputs, including the selected native and
  independently reproduced source, trained artifacts, native library,
  regression gates and frozen reference data. Postflight verifies them after
  every owned engine is terminal. No builds run during collection.
- All30 responses and3,840 generated IDs/text exactly match the saved full
  FP32 controls. Original requests, prompt IDs, row epochs, native scheduling,
  route bounds, no-retention policy and SSE clock calculations are audited.
- Each engine executes1,536 fused blocks:6,144 total producer kernels and
  6,144 down kernels. Logical affine counters preserve the existing product
  convention; these counts exclude gathers/clears. Each prepared inventory
  has one private14,680,064-byte fused arena. No MPS attention products run.
- All native workers exit0 through expected close, and native owners return
  to zero. Post-close allocations are1,769,472 bytes for C1 and2,293,760 for
  C4. These snapshots are not peak-memory evidence.
- Thirteen deliberate corruptions reject, including missing/wrong fused
  routes/sites, wrong tokens/prompts/epochs, cache hits, MPS fallback,
  owner leaks, false occupancy, TTFT and decode rate.
- Raw saved streams independently reconstruct all reference metrics, with
  the existing15 warmed waves per cell and unchanged baseline hashes.
- The packaged auditor reproduces the complete retained report exactly.
  An initial local audit invocation passed a study directory as its output
  filename and correctly hit the no-overwrite guard. The corrected invocation
  uses unchanged audit logic and saved data; no inference is repeated. The
  packaged CLI explicitly accepts the study directory followed by an output.

Strict coding remains0/8, with exact output regression. This is a text-only
experimental native overlay, not production enablement or model-quality
qualification. ANE/Gemma, multimodal/hybrid recurrent Qwen and the original
overall objectives remain open. The original11 dirty files are preserved.

## Reproduction

Saved-data audit, without inference:

```sh
python3 -B audit1.py /absolute/path/to/mistral-fused-mlp-serving1 /absolute/path/to/fresh-audit.json
```

Prepare/collect only when a new candidate or unresolved concern justifies
another run. The preparer requires the pinned sibling half-serving and fused
native/gate studies, clones the unchanged transport, and uses the admitted
uninstrumented fused worker. It accepts an absolute fresh study directory.

```sh
python3 -B prepare1.py /absolute/path/to/mistral7b-local-coder/FRESH-SERVING
python3 -B /absolute/path/to/FRESH-SERVING/run1.py
python3 -B audit1.py /absolute/path/to/FRESH-SERVING /absolute/path/to/fresh-audit.json
```

The tools retain local artifact paths. They never rerun vLLM/SGLang, overwrite
collected campaigns, build the dirty root, or run inference engines together.
