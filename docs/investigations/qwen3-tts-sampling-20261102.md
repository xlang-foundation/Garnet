# Qwen3-TTS sampling qualification, 2026-11-02

This is an incomplete qualification. The candidate DLL was not promoted to
production, and default model sampling was not changed.

The native sampler permitted reserved codec IDs and compounded the repetition
penalty for duplicate occurrences. The candidate masks the final 1024 IDs except
EOS, prevents EOS during the first two frames, and penalizes each distinct
history token once. This follows the
[Qwen generation implementation](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/core/models/modeling_qwen3_tts.py)
and [Transformers repetition semantics](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py).
CPU regression cases cover forbidden-token selection, duplicate positive and
negative logits, EOS admission and invalid values. They pass in Release builds.

The isolated build changed the compiled-model runtime and the package entry
objects. It preserves the original build DLL, runtime and vendor libraries.
The model-tested candidate SHA256 is
`2e4f70ce94691758eaf97ecc22b127e35a026bf7834a8868438e62babb904138`.
The final staged build appends the identity accessor after existing API members
to preserve their order: SHA256
`2b6323f83194a57ec65a6d90455891d17e80ad3ac64923480aba735f7e2c9d8c`.
Its model algorithm is unchanged; it has an identity/API smoke check, not a new
claim of complete model qualification.

Qualification exposed a loader hazard: directories named `lib` are excluded
from XLang3 native package search. The initial worker loaded Cantor's default
Garnet instead. No inference acceptance was counted from that worker. The
candidate moved to a `native` directory. A new `native_library_path()` accessor
and optional required-identity admission guard verify the selected artifact.
The actual watched worker's mapped DLL path/hash and zero CPython modules were
also checked. Older package objects lack Python `__file__`; absence is not a
valid identity assertion for them. Legacy deployments still require their
external module audit until they gain native identity support.

Unchanged stochastic defaults produced 11 exact round trips out of 12; the
first request exhausted the 128-frame profile. Fixing seed zero produced three
identical length-limit results, including cold and warm requests. This does
not support blaming first-call initialization alone.

Greedy decoding passed all 12 Ryan English phrase cases, including the first
request. A separate test across all nine voices and three Chinese phrases
passed only 7 of 12 cases: Uncle Fu, Ono Anna, Sohee and Dylan exhausted the
profile on English input, and Eric added speech. Do not promote greedy decoding
as a general default or retry away these failures. Further work needs explicit
language/reference-frontend comparisons and voice coverage, including the
reference's code-predictor sampling policy. The current wrapper uses `Auto`.

Optional sampling settings are validated immutable model configuration, not
arbitrary native call arguments. A distinct profile contract is required for
different settings. The managed catalog rejects mixing different sampling
contracts in one local resource family. Existing configuration without sampling
settings retains its current behavior.

All attempts and generated-audio reports are retained under
`D:/CantorAI/out/tts-sampling-v2/`. The candidate ran only in temporary watched
Cantor workers on the existing SHAWN3080TI node. Coordinator revisions 16 and 17
drained and restored its original TTS slot, preserving five other placements.

A subsequent predeclared 30-case language/voice/seed comparison passed 23 exact
full-audio ASR round trips. English input passed 10/12 with Auto and 7/12 with
explicit English; Chinese passed 3/3 under each setting. Four failures reached
the profile cap and three produced incorrect speech within the cap. Every
generated waveform was retained and completely transcribed in bounded ASR
chunks. No retry or seed exclusion was used. Explicit English is not a repair.

This test also exposed a separate Serving loader defect: adding an import root
did not guarantee the configured host source was selected. The native loader
now loads that file explicitly and retains its module for worker callbacks.
The model-tested Serving alias was `d3e164be4cdd250b1796ed7636772d8606e790e76dba6398bc580cca43786c82`;
the final staged build adds companion-import priority and rollback tests and has
CPU qualification only. Production defaults and Garnet were not replaced.
The original TTS placement was restored at revision 19, generation 5. Evidence,
including setup failures and cleanup, is in `D:/CantorAI/out/tts-sampling-v3/`.
