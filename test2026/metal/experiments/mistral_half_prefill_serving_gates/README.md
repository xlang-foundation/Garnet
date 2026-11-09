# Half-prefill cache and coding regression gates

These experiments use the immutable native half-prefill library from
`644b5bb6d`. They change no C++ runtime, kernel, xModel operator, tokenizer,
model weights or frozen reference engine. Python transports requests and
observes native state; scheduling, leases, prefix matching, cache storage and
restore remain inside Garnet's native runtime.

Two fresh serial engines run the unchanged nine-request protocol with retention
disabled/enabled. A/B/C/D occupy four rows; E queues before A finishes. E and
exact/append/late-edit/early-edit A then reuse A's physical row at increasing
epochs while B/C/D continue. All752 generated IDs/text per engine match each
other and their frozen FP32 protocol controls. Cached and uncached target
prefill/final states match across1,170 full-KV/hidden digest pairs. Five restore
observations preserve1,950 survivor hash observations per engine. There are no
executed MPS attention products;384/256 half dispatches cover the12/8 actual
prefill body calls. Native owners close to zero under the4MiB device limit.

| Returned request | 384MiB host-prefix budget | 80MiB pressure budget |
|---|---:|---:|
| Exact A | 211 cached tokens | 0, after eviction |
| Appended A | 128 | 128 |
| Late edit | 128 | 128 |
| Early edit | 0 | 0 |

The pressure engine is a third fresh serial process compared with the saved
uncached half-policy control. It forces14 evictions. Maximum observed accounted
retained host bytes are68,300,596, below the83,886,080-byte budget; this is not
a peak physical-RAM measurement. All752 outputs/text and1,170 target-state
pairs remain exact, with the same queued-fifth/row-reuse/survivor invariants.
No evicted A snapshot is incorrectly reused. The available128-token nodes
still serve append/edit requests correctly. This is one bounded pressure
profile, not proof of every OOM, cancellation or model-failure path.

The ordinary cache audit rejects eight deliberate corruptions; the pressure
audit rejects six, including budget overrun, suppressed eviction and wrong
state/lease/route/reuse. The original protocol fixture and all client bodies
are unchanged. The half-policy producer namespace prevents accidental use of
the old FP32 producer domain. These text-only Mistral tests do not qualify a
hybrid recurrent Qwen or multimodal cache.

The coding run uses the unchanged eight-task suite,1024-token ceilings and
strict sandbox evaluator. All3,052 generated IDs/text are exact to the frozen
FP32 native control; all eight terminate at EOS. At most four rows are live,
and256 half-prefill dispatches cover the actual prefill bodies. Sandbox startup
passes and native owners close to zero. Seven corrupted collections reject.

**Strict coding remains0/8.** Every answer still fails the unchanged code-only
format gate. The separate single-fence diagnostic remains1/8 and does not
replace that score. This is no observed coding-output regression on these tasks,
not qualification of Mistral as a good coding model or a general quality proof.
No scorer, prompt, token ceiling, answer or frozen reference is altered.

State observers perturb scheduling and latency. None of these HTTP/quality
timings establishes TTFT, token speed or throughput. The separate prefill-body
gain remains1.18–1.22×; the2× and half-TTFT goals remain unmet. ANE/Gemma and
the original Qwen objectives remain open.

Local evidence directories are `mistral-half-prefix-http1`,
`mistral-half-prefix-pressure1` and `mistral-half-coding-quality1`, under
`out/research/mistral7b-local-coder`. They retain raw SSE, native events, full
state digests, process exit/shutdown, profiles, frozen-input bindings and audit
faults. No comparator inference is rerun. Existing root modifications remain
untouched; checkpoints are local to the feature branch.

Preparation/launch tools take an absolute fresh local study directory and
require the pinned sibling native/original studies. The generator copies the
unchanged original protocol and tools before applying the recorded observer
or half-route edits. Auditors take a retained study directory and optionally a
fresh report filename. The large raw records, model and native libraries stay
local. Worker patches describe research observers/route assertions, not new
serving APIs. Performance collection must use an uninstrumented worker after
these gates, without the target/survivor state hashes.
