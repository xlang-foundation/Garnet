# Long-prefix reuse with wide native prefill profiles

This study qualifies the optional wide-profile engine's prefix-cache lifecycle
for Mistral 7B. It uses the frozen engine from
[the wide half-attention study](../mistral_prefill_completion_wide/README.md).
It changes only the Python research observer and arrival driver. Prefix matching,
cache retention/restoration, row ownership, epochs and scheduling execute in the
existing native C++ runtime. No kernel, xModel operator or production default is
changed here.

Two fresh engines run serially: uncached control, then cached. Three long-lived
requests B/C/D occupy different physical rows. A runs and finishes; a fifth
request E is admitted before A finishes and waits for its row. A then returns,
appends text, changes its ending, and changes its beginning. All six newcomers
execute and finish while B/C/D remain live. Native events prove that all three
seed prefills finish before each newcomer's first native plan, and all six use
A's row with strictly increasing epochs.

| Incoming request | Input tokens | Retained tokens |
|---|---:|---:|
| A returns unchanged |2048 |2048 |
| A appends a requirement |2061 |1792 |
| A changes its ending |2049 |1792 |
| A changes its beginning |2052 |0 |

Appending message content changes the rendered prompt before the closing
instruction delimiter, so the entire old prompt is not a token prefix. The
1792-token hits correspond to a captured256-token checkpoint. The exact2048
hit uses a head-only prefill plan: no prompt-body recomputation.

Both arms produce the same720 output IDs and text. All1170 full per-row KV/
hidden digest pairs and73,728 hidden values match, at prefill completion and
final generation. These compare all32 layers' complete4096-capacity canonical
FP32 row buffers, including their unused suffix. Survivor witnesses compare
8320 digests in the uncached arm and5070 in the cached arm before/after native
scheduler calls. Each newcomer has a witness bound to its actual restore tick,
with all three seed leases and full prefill histories intact.

The cache budget is2.5GiB/128 entries, with direct capture and shared64-token
blocks. The final cached state has40 entries and2,246,337,120 accounted host
bytes after five evictions. Every observed accounting sample stays within the
budget and reconciles private plus shared bytes. Actual GPU token workspaces
remain at most512 rows; a larger CPU input parent is not treated as a GPU
dispatch extent. Both engines close with all tracked owners at zero and
2,949,120 Metal bytes, below the4MiB close bound. Sixteen deliberate evidence
corruptions reject.

One preparation-time statement needed clarification. Waiting for D's first
token after concurrent B/C/D submission does not prove the other two are ready
at A's client submission. The original record is retained. The audit instead
requires actual native final-prefill acknowledgments before each newcomer
native plan and binds survivor witnesses to its restore tick. This proves
the tested runtime boundary without assuming client arrival order. See
`collection-clarification1.json`.

This is an untimed correctness study: state copying perturbs execution. It
makes no prefill/decode speed claim, and does not qualify hybrid recurrent
state or coding quality. Strict coding remains0/8 in the separate frozen
quality campaign. Repeated serving qualification, default promotion,
production integration, ANE/Gemma and the original Qwen targets remain open.
No vLLM/SGLang inference was rerun. All11 pre-existing dirty files remain
byte-identical.

## Reproduce inputs or audit saved evidence

The input generator requires the pinned sibling wide, previous prefix and
selected serving studies under `out/research/mistral7b-local-coder`. It creates
a fresh directory, derives the observer from the frozen wide worker, renders
prompts with the checkpoint template and independent tokenizer oracle, and
reproduces the exact qualified worker, fixture, profile and binding hashes.
It does not execute inference. Its preparation statement uses the corrected
native boundary; the original recorded preparation script is also retained.

```sh
python3 -B prepare1.py /absolute/path/to/mistral7b-local-coder/FRESH-INPUT-REPLAY
python3 -B audit_lifecycle3.py /absolute/path/to/mistral-wide-half-prefix-http1 /absolute/path/to/FRESH-REPORT.json
```

The packaged auditor and helpers reproduce the full saved report without
inference. `run_http1.py` and `launch1.py` preserve the exact collection used;
they are research records, not an automatic benchmark launcher. The original
and strengthened auditors/reports are retained locally. The local campaign is
`mistral-wide-half-prefix-http1`.

The replay auditor verifies the frozen original auditor hash and compares helper
contents with trailing whitespace normalized. Only a surplus blank line at the
end of the copied fused-route helper was removed for the repository checkpoint.
Its replay report still matches the complete frozen report exactly.
