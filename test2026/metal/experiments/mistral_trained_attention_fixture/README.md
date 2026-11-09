# Actual trained attention fixtures

This is a test-only native patch and retained-data tooling. It captures actual
Mistral Q/K/V, private cache, controls, gate and attention result at the 16th
prefill chunk, B4/C4096/T128, offset1920, layers0/15/31. Evidence is under
`out/research/mistral7b-local-coder/mistral-trained-attention-fixture1`.
No xModel operator, neural shader string, tokenizer, math, precision or serving
default changes. No Rust runtime is introduced. This is not a timing campaign.

`GARNET_METAL_ATTENTION_FIXTURE_CAPTURE` defaults OFF. Use the complete admitted
hoist configure recipe, including packed Mistral, diagnostic layer-bulk and
up-to512 dense-token policies; other compile flag combinations are unqualified.
The flag exposes test-only native arm/finish calls. Arm requires a fresh absolute
output directory and drained ready context. It prepares490,749,312 requested
shared-buffer bytes, below512MiB. One frame must contain exactly32 eligible
attention calls with completed, all-active contiguous controls at slots1920..2047.
Only the selected three layers get ordered read-only input/result blits. Finish
requires drained successful work, verifies complete capture guards/tails/finite
values, writes27 guarded FP32 tensors plus metadata, then releases the buffers.
Abort poisons capture; reset/cancel discards it. No prefix numerical-domain change
is needed because the observer does not change the model calculation.

The first source passes its original native tests and reproduces original
model state, but the independent close audit rejects493,109,248 Metal bytes
against the existing4MiB limit. Native tracked owners alone miss retained blit
encoders. That source, library, raw capture and failed audit remain. Revision2
adds scoped autorelease pools and checks actual device allocation after partial
abort/cancel, finish and context close. Both native suites pass51ON/50OFF. The
capture test verifies122,687,328 exported parent words, actual source offsets,
full input immutability, physical guards, invalid/open/duplicate/pending/abort/
cancel/finish controls and bounded post-close device resources.

A separate native test also destroys armed and exception-failed captures without
an explicit cancel. An unprepared attention error occurs after input blits are
encoded; failed export rejects. Closed device allocations are475,136/1,523,712
bytes, both below the same4MiB limit. The test links the exact revision2 static
library with its CMake flags; no real hardware fault or full-model failure is
claimed.

One fresh revision2 engine reproduces all32 frozen generated IDs/texts and all
260 complete KV/hidden digest pairs at four frame boundaries. Six captured cache
hashes exactly match the actual layer0/15/31 state at frame16. The27 payloads are
bit-identical to the rejected first capture, but cleanup now returns to1,900,544
Metal bytes and zero native owners. The original framework CoreAnalytics leak
warnings remain and are not explained by these bounded resource observations.

Native attention gold passes18,432 independent FP64 values at unchanged
`1e-5 + 2e-4 * abs(reference)`; the largest fraction is0.038745. Actual Q reaches
about-14 and absolute K about19.5, substantially beyond the earlier small random fixtures.
Row padding for the operator replay preserves all116,395,008 input words bitwise.
Original four-product MPS, no-copy two-product MPS and guarded strict FP32 MPP
replay all meet the native gold bound over18,874,368 complete comparisons.
The independent oracle checks18,432 values per variant, including native gold:
73,728 variant comparisons. No precision/tolerance switch is used. These are
bounded trained-operator gates, not full-model logits/quality qualification for
the new numerical policies. A serving speed gain has not been demonstrated for these paths.

Prepare clean source using the hash-verified admitted source/manifest:

```sh
python3 prepare_native_source.py GARNET_REPO ADMITTED_SOURCE ADMITTED_MANIFEST FRESH_OUTPUT
```

It delegates the adjacent hoist preparation helper and applies the exact
revision2 eight-file patch. Use the retained configure commands with separate
ON/OFF builds. The qualified binary/source are revision2; revision1 is rejected.
`autorelease-fix2.patch` records only the diagnosed lifetime/test correction.
The native test is part of the patch, rather than a replacement xModel operation.

Audit the retained study without GPU work:

```sh
python3 audit_trained_capture.py STUDY
python3 audit_trained_replay.py STUDY/replay1
python3 audit_replay_bindings.py STUDY
```

The fixture preparer is a study-bound research template: it requires the
successful revision2 capture/audit and the frozen strict-MPP study. The native
standalone replay helper uses the retained Safe-Metal4 shader, policy header,
C++20/ARC and Foundation/Metal/MPS links from the saved compile command. Its
`trained` mode performs only untimed correctness collection. Large model/payload
files remain in the local evidence directory; raw inputs are never embedded in
this repository. No vLLM/SGLang inference is repeated. The strict coding0/8,
serving comparison,2x/half-TTFT, ANE/Gemma and original Qwen goals remain open.
