# Committed-KV two-product attention experiment

This is an opt-in native patch and reproducible research tooling. It is not a
production selection or a serving-speed/quality qualification. Evidence lives in
`out/research/mistral7b-local-coder/mistral-committed-kv-attention1`.

The existing compiler-proved early KV commits let a private attention entry read
one unified cache. The candidate packs Q only, performs one FP32 QK product and
one FP32 PV product per owner/KV-head, and directly unpacks the result. It removes
both live-only products and uses the existing prepared GPU buffers. Generic
public attention semantics, xModel operators and scalar-tail fallback remain.
`GARNET_METAL_COMMITTED_KV_MPS` defaults OFF and requires the prepared MPS,
index32, context-bucket, active-row and commit-hoist options. The numeric backend
tag is `committed_kv_mps_fp32_v1`; native prefix identity appends
`|committed-mps-fp32-v1` to separate this reduction policy from existing snapshots.

The new primitive passes six finite-input shapes: full/ragged/gated owners,
2K/4K extents and the scalar tail. It checks 7,127,040 complete output words,
864 independently calculated FP64 reference values at the original
`1e-5 + 2e-4 * abs(reference)` bound, complete physical cache commits, readonly
inputs, unsafe/pending controls, partial submission and owner cleanup. Native
ON/OFF suites pass 51/50 tests.

Two fresh serial full-model engines use four frozen 2048/2051/2048/2049-token
prompts and eight outputs each. Both reproduce all 32 frozen native IDs/texts.
512 candidate attention dispatches halve observed matrix products from 65,536
to 32,768; both variants retain 1,088 hoisted commits. The OFF control's complete
state matches prior qualified state. ON is not bit-identical after later chunks:
max final-hidden-bank absolute error is 0.006295 at the observed boundaries.
Both close with zero native owners and 1,900,544 Metal bytes. CoreAnalytics
context-leak warnings remain in both logs and are not explained by owner counts.
The saved-data auditor rejects eight corrupted records.

Eight additional fresh serial engines run two opposite-order cohorts, one/four
active owners, three warmups and nine measured calls each. The actual 32-layer
late chunk at offset 1920 follows 15 real prefix chunks. Wall speed ratios are
1.035/1.044 for one owner and 1.023/1.052 for four. Next predictions match; full
KV/hidden state is exact across cohorts within each policy. Across policies,
the full late-chunk hidden tensor has max absolute difference 0.060588. These
are descriptive ratios of process medians, not paired estimates or complete
prefill/TTFT/decode/HTTP results. The result supports retaining the experiment,
not default promotion or a 2x claim. Cached returns, full-model numerical/logit
and coding qualification, exception/cancellation and serving gates remain open.

Prepare a clean source tree, supplying the existing admitted source/manifest:

```sh
python3 prepare_native_source.py GARNET_REPOSITORY ADMITTED_SOURCE ADMITTED_MANIFEST NEW_SOURCE
```

This delegates the hash-pinned hoist preparation helper in the adjacent
`mistral_kv_commit_hoist` directory, verifies seven exact parent files, applies
this nine-file patch and verifies every candidate hash. It never builds the
dirty checkout. Configure/build flags and complete logs are retained in the
study; use its configure command with separate ON/OFF build directories.

Audit the retained study without GPU work:

```sh
python3 audit_model_state.py STUDY
python3 audit_body.py STUDY/body1
```

The worker/collector copies are research templates bound to the study's
`bindings1.json`, profiles, fixture, libraries and source. Preserve their original
study filenames when replaying; they do not bootstrap model assets or bindings.
No vLLM/SGLang inference is rerun. Their frozen FP16 attention/KV remains distinct
from this candidate's FP32 policy. Existing serving and strict 0/8 coding results
are unchanged. The Mistral/Gemma/ANE and original Qwen goals remain open.
