# Fused MLP cache-pressure and coding regressions

These serial experiments use the immutable experimental native library from
`mistral-fused-mlp-native1`. Python transports requests and observes native
state; scheduling, row leases, prefix matching, retention/restore and the MLP
fusion execute in native Garnet. No model, tokenizer, xModel operator, frozen
reference engine, prompt or scorer changes.

The unchanged nine-request protocol exercises four live owners, queued E,
then exact/append/late-edit/early-edit A on A's recycled row while B/C/D
continue. The80MiB pressure budget forces14 evictions. Maximum observed
accounted retained host bytes are68,300,676, below83,886,080. This is not a
peak physical-memory or general OOM/failure-path proof.

| Returned request | Cached tokens under pressure |
|---|---:|
| Exact A, after eviction |0 |
| Append |128 |
| Late edit |128 |
| Early edit |0 |

All752 output IDs and text match the saved uncached control. All1,170 target
KV/hidden digest pairs match, and1,950 survivor-state observations remain
unchanged. Actual routes cover320 half-attention dispatches and160 fused
MLP blocks, with no executed MPS attention products. The six inherited
pressure corruptions and three added fusion/inventory corruptions reject.

The unchanged eight-task coding run collects3,052 IDs and complete text,
exact to the saved FP32 outputs; all tasks reach EOS within their1024-token
ceilings. At most four rows are live. There are seven actual prefill body
calls,224 half-attention dispatches and64 fused blocks. Arrival scheduling
can change the number of partial prefill calls, so this is not a timing
comparison with the preceding collection.

**Strict coding remains0/8.** The separate single-fence diagnostic remains
1/8 and does not replace the strict score. Collection/audit success means no
observed output regression on these tasks, not coding-model qualification.
Sandbox startup passes. Seven inherited coding corruptions and three added
fusion/inventory corruptions reject. Native owners close to zero; device
allocations remain below4MiB. CoreAnalytics warnings remain in raw logs.

The local studies are `mistral-fused-mlp-prefix-pressure1`,
`mistral-fused-mlp-coding-quality1` and their tools in
`mistral-fused-mlp-gates1`, under `out/research/mistral7b-local-coder`.
No vLLM/SGLang inference is rerun. Original11 dirty Garnet files remain
byte-identical. No ANE, Gemma, multimodal, hybrid recurrent Qwen, production
integration or overall goal is qualified here.

The packaged preparation was replayed without inference: worker, fixture,
suite/evaluator and effective coding policy reproduce exactly, allowing the
new inventory destination. Use absolute fresh study directories and the
pinned sibling studies. Preparation/launch retains local evidence paths.

```sh
python3 -B prepare1.py pressure /absolute/path/to/mistral7b-local-coder/FRESH-PRESSURE
python3 -B /absolute/path/to/FRESH-PRESSURE/launch1.py
python3 -B /absolute/path/to/FRESH-PRESSURE/audit_pressure1.py
python3 -B prepare1.py quality /absolute/path/to/mistral7b-local-coder/FRESH-QUALITY
python3 -B /absolute/path/to/FRESH-QUALITY/launch1.py
python3 -B run_quality_audits1.py /absolute/path/to/FRESH-QUALITY
```

The quality launcher requires the saved admitted pressure audit. All inference
is serial. These observers and quality runs establish no serving speed. See
the [separate serving campaign](../mistral_fused_mlp_serving_benchmark/README.md)
for TTFT, decode and aggregate output throughput.
