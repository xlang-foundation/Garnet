# Bounded wide half-attention profiles for native prefill

This checkpoint extends the optional native completion-first experiment to
prepared owner profiles up to512 tokens. The per-tick actual-token budget and
the maximum GPU projection workspace remain512 rows. Kernels and xModel
operators are unchanged. The new CMake option
`GARNET_METAL_CHUNK_HALF_WIDE` defaults OFF and requires half attention,
prepared wide-owner profiles and a512-token compiled budget.

The first profile-only attempt failed its unchanged runtime coverage guard:
the compiler's immutable KV-commit-hoist proof and native half-attention guards
still capped each owner at128 tokens. The first1×512 operation therefore took
the FP32 attention path. The run stopped without publishing output tokens;
it supplies no performance evidence. Its raw artifacts and terminal state
remain retained.

The repaired engine adds an explicit128/512 compile-time half-profile limit.
The graph proof accepts the wider limit only in the corresponding compiled
engine, while retaining its alias, producer, consumer and ordering checks.
The context still checks at most four owners,512 total token rows,32 query
heads,8 KV heads,128 head dimensions and context capacity4096. The private half
arena stays68MiB plus384 guard bytes. There is no new Metal shader source or
arithmetic change.

## Numerical and lifecycle evidence

Both builds pass all52 native tests serially: wide enabled and the new option
off. Enabling the extension with half attention off rejects during configuration.
The expanded half-attention suite adds1×256,2×256 and1×512 shapes, retaining
complete parent/cache/scratch/guard checks and the existing numerical bounds.
Three additional partition tests compare a512-token call with128-token calls
bit for bit, including all output words and complete canonical KV parents:
first position0/full512, position1920/active257, and position3584/full512/gated.

Two fresh native engines then run serially with untimed state witnesses. The
128-token control reproduces the frozen whole-engine state:260 KV/hidden
digests and65,536 hidden values. The up-to512 completion-first schedule matches
all32 output tokens/text and all520 per-request KV/hidden digests, plus32,768
hidden values. These cover full single-row32-layer canonical FP32 caches at
prefill completion and final generation, including physical row reuse with
changed epochs. The candidate exercises512/256/4/2-token profiles. Ten deliberate
state-audit corruptions reject. Prefix-cache reuse and coding-quality gates
are separate and remain pending for this candidate.

An unprofiled admission process then serves the unchanged frozen2K-input/C4/
128-output requests in cold,warm1,warm2 rounds. All12 responses and1,536 output
IDs/text match the frozen native FP32 control. The audit observes four-row
decode batches,76 prefill steps with ready decoders,512-token maximum actual
ticks and2,688 fused MLP blocks. All GPU projection extents are at most512.
Twelve deliberate admission-evidence corruptions reject. Sources/models are
hashed before collection and after terminal shutdown; no builds, comparator
inference or state-witness copying occur during the admission collection.

The original128-token audit needed two explicit saved-data corrections. A wide
single-call prefill is not a grouped bulk plan; bulk classification now also
requires the bulk profile. The worker's fused `physical_rows` field describes
the full CPU input parent before compaction. Its maximum is2048 for B4/T512;
the audited compact GPU extent is512. The revised audit binds both extents to
the native plan/compact trace rather than treating the parent as a GPU dispatch.
All original stream, token, lease, alias, route and close checks are retained.
No model was rerun to repair an auditor. The initial runner-path preflight
failure and an audit-derivation metadata correction are also retained; neither
changed collected model observations.

## Admission observations

These are diagnostics from one fresh engine, with two warmed rounds. Different
campaign times and uncontrolled OS/thermal conditions prevent attributing an
exact improvement to this change. This is not a repeated or matched-control
serving qualification.

| Round | Median TTFT s | Worst request TTFT s | Decode/request tokens/s | Aggregate tokens/s | Delivery-gap p95 s |
|---|---:|---:|---:|---:|---:|
| cold |9.00 |14.65 |9.33 |21.76 |0.462 |
| warm1 |9.34 |15.36 |8.90 |20.81 |0.489 |
| warm2 |9.61 |15.93 |8.60 |20.15 |0.504 |

The previous128-token completion-first candidate observed warmed median TTFT
12.50–12.86s and aggregate15.86–16.24 tokens/s. The selected frozen Garnet
campaign has median TTFT14.54s and aggregate20.69 tokens/s at C4. These separate
campaigns motivate further evaluation; they do not establish a causal gain.
The candidate's early responses still encounter long token-delivery gaps while
prefill runs. Decode speed, tail latency and throughput must be assessed along
with median TTFT. **No default promotion or2× claim is made.**

The latest established comparison remains the
[fused MLP campaign](../mistral_fused_mlp_serving_benchmark/README.md).
Strict coding remains0/8; cached-prefix lineage, coding qualification, repeated
serving, production integration, ANE/Gemma and original Qwen goals remain open.
No vLLM/SGLang inference was repeated, and the original11 dirty files remain
byte-identical.

## Reproduction and saved-data auditing

The source generator requires the pinned sibling completion-policy and native
capture studies under `out/research/mistral7b-local-coder`, along with their build
dependencies. It reproduces all26,295 qualified source files and the native
patch exactly without compiling or executing a model. The native edits are
retained as both a reviewable patch and exact guarded textual transformations.

```sh
python3 -B prepare1.py /absolute/path/to/mistral7b-local-coder/FRESH-STUDY
python3 -B /absolute/path/to/FRESH-STUDY/build_native2.py
python3 -B /absolute/path/to/FRESH-STUDY/prepare_model2.py
python3 -B /absolute/path/to/FRESH-STUDY/run_model2.py
python3 -B audit_model2.py /absolute/path/to/FRESH-STUDY
```

Do not rerun unchanged inference just to verify evidence. Packaged auditors
accept `STUDY FRESH-REPORT.json` and reproduce both complete saved model and
admission reports without inference. The recorded admission driver, profiles,
all attempted generations and raw responses remain in the frozen local study.
The package above reproduces native unit and full-model state qualification;
it does not claim to launch a new serving campaign automatically.

Local evidence is `mistral-prefill-completion-wide1`, with replay studies
`mistral-prefill-completion-wide-replay1` and
`mistral-prefill-completion-wide-replay-source1`.
