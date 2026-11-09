# Four-row fusion: cache lifecycle and learned-draft validation

This extends the default-off
[native fusion checkpoint](../mistral_fused_q4_decode_native/README.md) without
changing native source, checkpoints, tokenizer output, scheduler plans or frozen
vLLM/SGLang results. The additional evidence qualifies returning-request cache
reuse and measures the existing learned-coder draft with fusion enabled versus
disabled. **The declared repeatable performance gate fails. Keep fusion OFF.**

## Prefix cache with three live survivors

Two serial fresh engines run uncached and cached versions of the same nine
requests. B/C/D are submitted concurrently. The driver waits for actual native
final-prefill acknowledgments for all three before A submission; client arrival
order or one client's first token is not treated as a readiness guarantee.
A finishes, a fifth queued E takes its row, then A returns with identical,
appended, late-edited and early-edited prompts. All six newcomers finish while
the three original survivors remain live, with fresh row epochs.

| Returning request | Reused prompt tokens |
|---|---:|
| A unchanged | 2,048 |
| Append | 1,792 |
| Late edit | 1,792 |
| Early edit | 0 |

All 720 output IDs/text per arm match. The complete cached/uncached request
states match across 1,170 layer/plane KV and hidden digest pairs and 73,728 hidden
values. Three-survivor witnesses surround every newcomer's actual native
lease/restore tick and remain unchanged. The full A hit uses a head-only plan;
it is excluded from body-kernel observations. Each arm exercises 42 actual
four-request decode frames and 1,344 fused blocks, with zero input copies.
Partial decode uses the original route. Twenty deliberate evidence corruptions
reject. The cache's retained host accounting is bounded to 2.5 GiB and 128
entries. These read-only full-state observations perturb execution: this is
correctness evidence, not timing evidence. Mistral is dense; no hybrid recurrent
model is tested here.

## Learned coding draft: serial on/off comparison

The target is the same pinned Mistral 7B 4-bit checkpoint. Its existing draft is
Qwen2.5-Coder-0.5B-Instruct, revision
`6b16732e5af5cd9bd600186ad59fa618867ef7a4`, packed Q4/g64 with FP16 coefficients.
The unchanged eight-case input, tokenizer translation, resident draft-prefix
reconciliation, maximum eight draft tokens and maximum three target proposals
are retained. No policy is tuned against these cases.

Both native libraries come from final source3 in the preceding checkpoint.
Their `GARNET_*` CMake options differ only in `GARNET_METAL_FUSED_Q4_DECODE`.
Four fresh processes run disabled/enabled/enabled/disabled, serially. All
pipelines are prepared before collection. Sources/models are hashed before and
after the complete collection, with no builds or source/model hashing during
the timed stages. Each stage decodes 3,044 tokens after the first token across
eight complete outputs (3,052 output tokens including first tokens).

| Pair | Disabled tokens/s | Enabled tokens/s | Enabled / disabled |
|---|---:|---:|---:|
| 1: disabled then enabled | 35.747 | 34.832 | 0.9744× |
| 2: enabled then disabled | 32.909 | 34.276 | 1.0415× |

The predeclared gate requires at least 1.03× aggregate stage speed in both
pairs, and no case below 0.98×. It fails. Rates also drift across the fresh
processes; neither a causal thermal explanation nor a repeatable fusion gain
is established. These are native complete-output stage measurements, not HTTP,
TTFT or a matched comparison with vLLM/SGLang.

Every stage uses 1,121 full four-input verification frames, four partial frames
and 54 ordinary fallback frames. Each enabled stage executes 35,872 fused
blocks and zero input copies. Draft proposals, emitted IDs, accepted counts,
cursors and resident draft histories match exactly between enabled/disabled
runs. Prepared Metal allocation is 5,744,803,840 bytes disabled and
5,744,869,376 enabled: exactly 64 KiB of additional private scratch.

The saved cost census covers all 1,125 instrumented verification frames per
stage. Target GPU duration accounts for approximately 97% of their target wall
time. Commit/rollback GPU duration is only 0.21–0.22% of complete stage time;
draft wall time is 23.0–24.9 seconds. The 54 ordinary fallback frames lack
separate GPU timing and are explicitly excluded from GPU coverage. This directs
the next performance investigation toward larger verification-kernel gains,
rather than a rollback-copy micro-optimization. It does not attribute time to
an individual kernel or normalize GPU clocks. The arithmetic and raw input hashes
are retained in `evidence/cost-census1.json`.

## State reference and retained audit failures

The first saved-data auditor compared a planned library directory with the
runtime's full dylib filename and rejected. A filename comparison correction
was made in a new auditor; no collected file was changed or inference repeated.

The corrected auditor then rejected internal state against an older frozen
native build. Both current enabled and disabled builds retain its exact output
IDs, but both differ in 62 KV-plane digests and the final hidden digest for each
case. They agree with each other. The older/current engines have different
prefill and attention options; no single cause is isolated, and internal-state
equivalence across those producers is not claimed. Both failed auditors and
their stderr remain retained. The historical-state gate remains **failed**.

A separate new reference uses the current disabled build, the same prepared
prefill and eight prompts, and ordinary one-token target decoding only. It
executes no proposer or target-verification rounds. All original frozen output
IDs still match. All four learned-draft stages match its full KV and hidden
state: 2,080 layer/plane/hidden digest pairs. The common causal validator uses
the role label `control`; only the ordinary reference's in-memory role label is
adapted for it. No raw digest or reference file is modified. Thirty evidence
corruptions reject. Original four timed stages are never rerun for audit repairs.

This qualifies current-producer ordinary/verification parity, not equivalence
to every older precision/route history. Strict coding score remains 0/8, and
coding quality, repeated matched HTTP speed, production integration, ANE/Gemma
and the original Qwen goals remain open. No 2× result is claimed.

## Reproduction and replay

Pinned sibling studies and checkpoints under `out/research/mistral7b-local-coder`
are required. Preparation reconstructs the exact observed prefix worker and
copies the unchanged learned-draft input/worker, reference and auditors. It
performs no native build or inference.

```sh
python3 -B prepare_fresh1.py /absolute/path/to/mistral7b-local-coder/FRESH-PREFIX /absolute/path/to/mistral7b-local-coder/FRESH-CODER
python3 -B replay_saved1.py /absolute/path/to/FRESH-REPORT-DIRECTORY
```

Saved replay runs both complete independent audits and their corruption
controls, requiring exact report equality without inference. For an explicitly
needed fresh qualification, run fresh prefix `launch1.py`, then
`audit_lifecycle1.py`; run fresh coder `launch1.py`, then `run_reference3.py`
and `audit3.py`. The original `audit1.py` and `audit2.py` are included to retain
their historical failures. Keep GPU processes serial. Do not rerun unchanged
inference merely to replay saved evidence. The initial prefix setup copied
read-only files and failed before its first write; destination copies were
verified unchanged and made writable. Original frozen inputs and the working
repository's 11 protected dirty paths remain unchanged.
