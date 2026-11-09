# Mistral profile follow-ups

The 64-token helper upper bound is the fastest observed native option in the clean confirmation;32 is nearly tied. The gain is small and the 2× target remains unmet. This checkpoint preserves the rejected kernel alternatives and the existing-kernel profile experiments without changing runtime defaults.

## Helper profile sweep

Apple M5/24 GiB; one Mistral request with the frozen 2K prompt and128 target outputs; real Qwen2.5-Coder-0.5B drafts; fixed target configuration and model files. Each profile precompiles every supported power-of-two graph family up to its bound. Ten fresh engines run strictly serially in ascending then descending order.

| Helper upper bound | Median native decode, target tokens/s | Extra prepared Metal memory vs16 |
|---|---:|---:|
|16|34.73|0 MiB|
|32|35.61|4 MiB|
|64|35.64|12 MiB|
|128|35.33|29 MiB|
|256|35.45|67 MiB|

The paired speed ratios for 64 versus 16 are 1.012× and 1.042×; the ratio of median rates is 1.026×. These two orderings do not establish that 64 is generally faster than 32. The initial 222-token helper prompt uses 64/64/64/32 profiles, with only 30 active tokens in the last chunk. Native token-count witnesses end at 222. Initial helper GPU diagnostics are about 126 ms for 16 and 65 ms for 64; those qualification diagnostics are separate from the clean end-to-end decode measurement.

All five bounds match 1,280 target output IDs and 650 full target KV/hidden digest comparisons against ordinary decoding. Causal proposals, acceptance, native prefix reconciliation and the absence of host KV snapshots are independently audited. Different helper numerical profiles are not required to have identical helper-state bits; the target verifier certifies every committed output.

The first timing sweep overlapped another research chat's already-running archive/hash process. Its raw results and the unknown overlap interval are retained and explicitly excluded from timing admission. The table uses the subsequent clean confirmation after that process terminated and a work hold was acknowledged. No model/source hashing or builds ran in our timed collection, and qualification-only per-chunk diagnostics were disabled during generation.

The historical worker declared a legacy `B1D512T16` cache label for every bound. Its private host-snapshot caches were always empty, so no cross-profile payload was reused. `b1_worker2.py` corrects the label to the actual bound, giving `B1D512T64` for 64. Its mathematical functions are AST-identical to the timed worker. A fresh corrected-identity run matches another 256 target IDs, 130 full-state digests and the complete causal proposal/acceptance trace. Historical timings were neither overwritten nor relabelled.

## Other completed experiments

| Experiment | Result | Decision |
|---|---|---|
|Larger direct-tensor row tiles|Both512-row down variants regress against the qualified256-row tensor kernel|Exclude from native integration|
|Direct tensor operands inside fused gate/up|Full learned intermediates, down outputs and GPU writer maps pass; whole-MLP ratios0.989–0.995×|Exclude from native integration|
|Existing FP32 four-row fusion plus shared-KV verification|57 native tests;512 qualification IDs and260 state digests; paired native decode ratios1.015×/1.014×|Keep optional; below the declared3% stage gate|

The fused-verifier study's separate cohorts measured35.23→35.75 native tokens/s. Do not mix those absolute rates with the helper table: they came from separate collections. The complete native source is unchanged from the qualified base; one compiled profile option enables the already-implemented four-row fusion. It executes on actual B1/T4 verifier frames and preserves all observed proposals and acceptance decisions.

These are native-stage studies. The matched vLLM-Metal/SGLang HTTP comparison remains unchanged; neither comparator was rerun. No HTTP TTFT, aggregate-throughput, broader coding-quality, ANE gain or production-default admission follows from these measurements. The studies do not qualify the larger helper bounds for C4 serving.

## Evidence and replay

The four subfolders contain executed sources, collection helpers, independent auditors and reports. Two copied auditor files have a trailing empty EOF line removed; their ASTs are identical and the raw sources are untouched, as recorded in `package-normalization1.json`. Complete raw arrays, model/source bindings, libraries, process records and source/build snapshots are frozen outside Git. `source-reference1.json` binds those roots; `package-contents1.json` binds every copied file. Collection helpers retain their original paths and refuse overwriting evidence.

For a CPU-only replay with the existing frozen roots, run from this folder:

```sh
/Users/shawn/CantorAI/out/research/qwen-local-coder/asset-env/bin/python -B audit_package1.py \
  /absolute/path/to/a/fresh/replay-directory
```

The replay isolates each raw root in a fresh Python process to prevent similarly named auditor modules from contaminating one another. It must reproduce eleven report files byte-for-byte. No model or GPU execution is requested.

Next, the candidate needs admission into a serving profile with the same maximum concurrency as the frozen HTTP references. Single-request native results alone do not establish that comparison. Runtime integration, matched HTTP measurements, the full coding/multimodal qualification and the 2× objective remain incomplete.
