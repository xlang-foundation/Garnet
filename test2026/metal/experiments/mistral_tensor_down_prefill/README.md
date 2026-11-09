# Bounded Mistral tensor-operand down projection

The optional native kernel passes exact model and row-lifecycle qualification. Its measured native prefill improvement is small; the 2× target remains unmet.

| Native C4 prefill diagnostic | Existing kernel | Tensor kernel | Speed ratio |
|---|---:|---:|---:|
| First prediction, forward cohort | 14.249 s | 14.135 s | 1.008× |
| First prediction, reverse cohort | 14.786 s | 14.577 s | 1.014× |
| Full prefill burst, forward cohort | 14.336 s | 14.216 s | 1.008× |
| Full prefill burst, reverse cohort | 14.872 s | 14.660 s | 1.014× |

These are native diagnostics over the original four ~2K prompts, with one prediction per request and prefix reuse disabled. They include all prefill and first-prediction work. Two serial fresh engines alternate enabled/disabled configurations within the same prepared library, with opposite orders, four warmup bursts and eight measured bursts per engine. No source/model hashing or builds run during timing. These values do not replace the matched HTTP comparison, measure decode throughput, or establish an ANE gain. Frozen vLLM/SGLang inference was never rerun.

`native-review1.patch` implements `GARNET_METAL_MISTRAL_DOWN_TENSOR512`, disabled by default, against the qualified `mistral-causal-batch4-helper1/source1` snapshot. It uses Metal MPP tensor operands over the existing bounded Q4 shared tile. Admission requires Apple10, FP16/group64 affine weights, K14336/N4096, 512 actual physical rows and an identity selection of all 512 rows. Partial and permuted profiles retain the existing route. Both plain packed projection and the fused MLP down consumer select the new pipeline. The gate/up path stays unchanged. There are no additional GPU buffers or xModel operator changes.

Both build profiles pass 56 native tests. Three fresh ordinary engines (enabled, same binary disabled, default-off binary) and an enabled cancellation/reuse/late-arrival lifecycle run match 494 output IDs and 40,820 full-row KV/hidden digest comparisons against frozen controls. Lifecycle checks exercise compact owner rows and fallback paths. The audits reject 11 qualification corruptions and 10 timing corruptions. Independent CPU report and source/patch reconstruction replays pass.

The `operator` folder records the preceding learned-weight experiments. Direct operands over the existing shared tile improve the isolated down projection about 6.8–7.3%. Global half-weight expansion loses when its cost is included; alternate threadgroup ordering adds no useful gain. Failed compilation and audit attempts remain in the frozen raw provenance. A source-manifest replay issue was corrected by binding an explicit input set; no GPU collection was repeated for that repair. Operator writer-map instrumentation remains unqualified and is not claimed.

This package checkpoints an isolated prototype. The repository runtime defaults are unchanged. Full raw tensors, results, libraries and complete source/build trees remain outside Git, bound by `source-reference1.json` and the content index. Collection scripts preserve their original paths and refuse overwriting evidence. For a CPU-only package replay with the existing frozen roots:

```sh
/Users/shawn/CantorAI/out/research/qwen-local-coder/asset-env/bin/python -B audit_package1.py \
  /Users/shawn/CantorAI/out/research/mistral7b-local-coder/mistral-down-tensor-native1 \
  /absolute/path/to/a/fresh/replay-directory
```

The next experiment should increase reuse of each unpacked weight across more actual token rows, screening larger direct-tensor row tiles before another native integration. This checkpoint does not admit a production default, HTTP improvement, coding-quality suite, or goal completion.
