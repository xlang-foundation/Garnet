# Mistral 7B comparator admission and fixed reference plan

This is separate from the preserved Qwen27B baselines. Mistral now has matched
vLLM-Metal and SGLang admission evidence, including full serving prompt-ID
checks and unchanged coding failures. Single-wave admission numbers below are
not repeated baseline medians or a Garnet speedup claim.

## Pinned configuration

The checkpoint is the original MLX affine INT4/group64 Mistral-7B-Instruct-v0.3
artifact documented in [onboarding](mistral7b-metal-onboarding.md). Its FP16
coefficients are retained. The model SHA256 is
`d7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d`.
Both engines explicitly request `float16`. Garnet currently retains FP32 KV
and intermediate state, so these are not identical whole-model precision paths.

| Property | vLLM-Metal | SGLang MLX |
|---|---|---|
| Source revision | `a44be48731005932a24dd30b55c21e992dc90b9a` | `57abc8a831abb83009ab543fc89df1d39a1306ff` |
| MLX / mlx-lm | 0.32.4 / 0.32.3.dev12+g4cfc64784 | 0.32.4 / 0.32.2 |
| Transformers | 5.19.0 | 5.19.0 |
| Context capacity / max active requests | 4096 / 4 | 4096 / 4 |
| Global prefill token budget | 512 | 512 |
| Prefix reuse | Disabled | Radix cache disabled |
| KV/cache memory policy | Explicit 2 GiB KV budget | 16,384 scheduler tokens; 512 MiB recycled-buffer cap |

The global prefill budget is shared across active requests, not 512 tokens per
row. A future Garnet scheduler must observe the same global limit for matched
comparisons, even if its compiled per-row profile can accept larger chunks.
Capacity 4096 is an upper bound, not mandatory input padding.

At FP16, dense Mistral KV needs
`32 layers * 2 caches * 8 heads * 128 dimensions * 2 bytes = 131072 bytes/token`.
The stock vLLM `--kv-cache-memory-bytes 2147483648` setting consequently admits
16,384 tokens, confirmed by the actual server log: exactly four 4K requests.
This excludes weights, scratch and other process/device memory. SGLang's
`--max-total-tokens 16384` is a scheduler ceiling; its MLX backend uses private
request caches rather than a full shared GPU pool. `SGLANG_MLX_CACHE_LIMIT_GB=0.5`
limits freed buffers and does not truncate live request KV.

vLLM's core log labels its control-plane device/cache as CPU, while the Metal
worker explicitly selects `Device(gpu,0)` and loads MLX-LM. These are actual
Apple GPU backends. Comparator runtime source is unchanged; the existing
SGLang `srt_mps` packaging overlay is captured as a patch. Full package lists,
native binary hashes, source revisions and overlay hashes are preserved.

All runs use this Apple M5/24 GiB Mac on AC power, with Cantor and simulator
stopped and one inference engine at a time. Models, packages, builds and
downloads are not changed during repeated measurements.

## Admission evidence and retained alternatives

The original auto-dtype vLLM trial selected BF16 KV and is retained separately.
The explicit-FP16 trial passed all six cells but reserved 98,960 cache tokens,
well beyond this profile, with material system paging/compression. The final
2 GiB policy retains enough KV for every admitted request and avoids that
excess reservation. Its six performance cells plus eight coding replays all
complete, with zero accounting/memory-bracketing audit issues.

Uncapped SGLang also passed its six cells. Its four-request 2K cell had 34,208
system swap-out pages and 353,975 compression pages in the bracketed window.
The final stock 512 MiB freed-buffer cap passes six performance cells and eight
coding replays. Its admission windows show no swap-out/compression increase,
but some swap-ins/page-ins and three page-outs remain in the long/C4 window.
System activity includes other apps; neither engine is declared paging-free.

Both final configurations have periodic server-log observations of four active
requests. Client concurrency alone is not used as proof of actual batching.
vLLM exits 0; SGLang's own cleanup exits -9 after group TERM. The collector records
no forced kill for either, and their process groups disappear. This is not a
claim of graceful instrumented native-owner cleanup inside the comparators.

The actual `/tokenize` message route agrees with every frozen ID for all 22
requests per engine: 14 performance rows and 8 coding rows. Independent audits
reject five corruptions each. Package/source/binary/admission guards also reject
12 corruptions before an engine is launched. Positive preflight rehashes the
actual checkpoint files as well as checking the comparator environment.

Both final serving configurations score **0/8** on the unchanged strict
code-only coding gate, as do native Garnet and the standalone MLX reference.
Explanations/examples violate the format contract; no prompts, output parsing
rules or tests are changed to promote a pass. This checkpoint is not admitted
as a high-quality coding model by these results.

## Single-wave observations, not baseline medians

Requests are greedy with a 128-token output cap and natural EOS. Actual short
prompt counts are 263/266/263/264 for the four rows; long counts are
2048/2051/2048/2049. Each concurrency cell uses its corresponding prefix of
these rows. All six admission waves generate 128 tokens per request.

The table is aggregate output tokens/s, including prefill and queuing. It is
not the per-request decode rate. There are no warmups or repetitions in these
admission values.

| Nominal prompt / clients | vLLM-Metal | SGLang MLX |
|---|---:|---:|
| 256 / 1 | 27.580 | 27.455 |
| 256 / 2 | 49.333 | 46.913 |
| 256 / 4 | 65.559 | 64.044 |
| 2048 / 1 | 16.531 | 16.939 |
| 2048 / 2 | 22.767 | 20.299 |
| 2048 / 4 | 25.095 | 21.629 |

Single-client short/long decode observations are 30.260/27.632 tokens/s for
vLLM and 29.919/27.831 for SGLang. Corresponding TTFT is 0.444/3.147 seconds and
0.417/2.993 seconds. Actual incremental usage timestamps supply decode timing;
SSE event counts are not treated as token counts.

## Repeated collection and evidence paths

Local evidence root:
`/Users/shawn/CantorAI/out/research/mistral7b-local-coder/`.

- `serving-fixtures1`: all request bodies, rendered prompts, full input IDs and
  budget/hash manifests, including the unchanged eight-task quality fixture.
- `baseline-phase1/*candidate.json`: preserved dtype/cache alternatives.
- `baseline-phase1/vllm-engine.json`, `sglang-engine.json`: final fixed specs.
- `baseline-phase1/vllm-fp16-kv2g-admission1`,
  `sglang-fp16-cache512m-admission1`: final 14-cell admission data and raw logs.
- `baseline-phase1/*prompt-token-proof1*`: raw serving tokenizer responses and
  independent full-ID audits, including earlier uncapped SGLang proof.
- `baseline-phase1/*quality-report1.json`: strict failures, linked to untouched
  raw responses. Secondary single-fence diagnoses do not replace these scores.
- `baseline-phase1/performance-manifest.json`: six cells, two warmup and five
  measured waves per cell, in three fresh processes per engine.
- `baseline-phase1/reference-preflight1`, `reference-guard-corruptions1.json`:
  positive environment/checkpoint guards and 12 rejected mutations.
- `reference-campaign1/status.json`: authoritative progress for the sequential
  repeated run, vLLM first and then SGLang. Each engine's nested campaign has
  per-cohort state, raw SSE and bracketed memory samples. Completion requires
  the final status plus zero issues in the independent accounting audit.

Each complete repeated engine campaign must contain 90 measured waves and 210
measured requests, plus 36 warmup waves and 84 warmup requests. Baselines are
frozen for these configurations; subsequent Garnet iterations reuse them.
Material configuration changes or invalid measurements require explicit new
evidence, not silent replacement. No matched Mistral Garnet HTTP campaign or
2x speed/throughput result exists yet; the ANE plugin remains separate work.
