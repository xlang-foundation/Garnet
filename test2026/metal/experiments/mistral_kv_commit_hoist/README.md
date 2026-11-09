# Compiler-proved early KV commits, original attention arithmetic

Experimental stage1 of a possible zero-copy two-product attention path.
`GARNET_METAL_CHUNK_KV_COMMIT_HOIST` defaults OFF. This stage moves the original
two private KV commits ahead of the unchanged attention operation and skips their
old positions. It does not implement two-product attention or change shader math,
precision, xModel operations, GPU layouts, allocations or scheduling APIs.
The native session adds only read-only `chunk_kv_hoisted_commits` instrumentation.

Prepare a separately verified source with:
`python3 prepare_native_source.py REPOSITORY ADMITTED_SOURCE ADMITTED_MANIFEST NEW_SOURCE`.
The admitted source/manifest are from the frozen prefix-policy native study;
preparation uses pinned commit7e5a16320, its28 verified overlays and this11-file
patch. Do not apply to/build the dirty worktree. Exact configure/build commands,
libraries and complete native logs are retained in
`out/research/mistral7b-local-coder/mistral-kv-commit-hoist1`.
Enable the option only with prepared private in-place KV and MPS prerequisites.

The compiler constructs its descriptor from immutable logical aliases before
scratch rebinding. It proves one bounded32/8-head D128/C4096 causal attention,
one K/V writer each, matching controls and returned-state groups, unique prior
Q/K/V producers, no intervening/late old-cache consumers, premature new-state
reads, extra cache writers or crossed token/control writes. Unsupported profiles
retain the original order. Runtime additionally requires both commits to target
the exact existing private cache views, no overlap with attention data/output,
and completed host controls proving contiguous active prefixes. Pending, gapped,
overwrite, unknown, standalone, verification and per-op diagnostic execution
retain the old path. B1/B2/B4 chunk profiles remain upper bounds.

All50 native tests pass in each compile mode. The pure logical proof rejects40
invalid alias/consumer/producer/order/shape layouts; native controls cover view
identity, readiness, pending data, gaps, overwrites, nonfinite flags, empty owners
and incompatible shapes. Existing native regression tests remain unchanged.

Two fresh serial OFF/ON engines execute four saved long prompts and8 output tokens
each. Actual ON counters observe1,088 reordered cache commits, including the sparse
tail; decode observes zero. All32 IDs/texts per engine agree with the saved control.
Complete physical32-layer KV and four-row hidden-bank digest witnesses at calls
1/16/18/25 agree across modes and with the earlier qualified original state.
There are260 matching digest pairs/520 observed hashes. All native owners close;
post-close Metal allocation stays under4MiB. Framework CoreAnalytics leak warnings
remain recorded and are not explained by these native ownership observations.

`python3 audit_model_state.py STUDY` validates the saved bounded model collection.
The research worker patch applies to the previous softmax state worker, changes
only read-only route instrumentation, and retains full state witnesses.

Two more fresh serial engines (hoist ON) exercise the frozen nine-request
uncached/cached HTTP return protocol. A/B/C/D occupy four rows; E queues before A
finishes. E, exact A, appended A, late-edited A and early-edited A reuse A's row at
new epochs while B/C/D continue. Cached reuse is211/128/128/0. All752 generated IDs
per engine and complete text match the prior saved original-policy results.
All3,900 survivor hash observations remain unchanged. Actual reordered commits
are768 uncached/512 cached. Both native workers close with zero owners and under
4MiB Metal allocation. HTTP wrappers end by the existing intentional SIGINT after
native graceful close; their raw exit status is retained.
`audit_http_hoist.py STUDY` validates added route/prior-control evidence alongside
the unchanged seven-corruption HTTP auditor in the retained `http1` subdirectory.
The HTTP worker patch applies to the frozen prefix-serving observer, adding only
read-only native route events. No comparator inference is rerun.

This is bounded semantic/lifecycle regression evidence. Full cache pressure,
actual reordered-model failure/partial-submit recovery, broader numerical/logit
contracts and useful coding/serving gates remain open. General nonfinite-model
behavior is not proved. No performance result or production/default promotion
follows. First extend those state/failure gates, then replace four products with
two under an explicit numerical/profile identity and independent validation.
Serving tables and strict coding0/8 remain unchanged; the full goal stays active.
