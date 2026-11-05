# Retained Mistral prefill experiments

These patches preserve measured experiments outside the selected runtime. They
apply in order to clean commit `85667ff7f317f18b9664822260038d31c04ce18b`:

Use `git apply --unidiff-zero PATCH` for these zero-context patches. Their
application is checked against the exact pinned base and compiled source hashes.

1. `mistral-prefill-tile-width.patch` adds the experimental
   `GARNET_METAL_MPP_REUSE64_TILE_K=32|64|128` option (default 64), matching native
   backend/benchmark metadata. The K=16 multiplication sequence and strict FP16
   operand/FP32 accumulation contract stay unchanged.
2. `mistral-prefill-packed-activations.patch` adds the operator-only
   `GARNET_METAL_MPP_REUSE64_PACK_HALF=ON` variant. It converts selected activation
   rows to compact FP16 once, then shares those values across output tiles.

The packed patch is **not a model/session integration**. Its scratch belongs to
the preparing `MetalContext`; resident sessions execute graphs through a
different context. Shared graph/session scratch ownership and binding must be
implemented and qualified before model use. The patch is retained because its
low-level operators were measured, not because serving supports it.

The pinned source snapshots, full configure argv, compiler logs and serial
collectors are under `out/research/mistral7b-local-coder/` in the parent workspace:
`mistral-prefill-tile-draft1`, `mistral-prefill-tile-source1` and
`mistral-prefill-pack-source2`. Each source has a captured SHA256 manifest.
`configure32-command.json`, `configure64-command.json`,
`configure128-command.json` and `configure-pack2-command.json` fix the full engine
flags, SDK/runtime and PCRE2 dependency paths. Reproduction must use new source,
build and output directories, preserving these original artifacts.

The compiled operator executable accepts:

```text
garnet_affine_fp16_projection_benchmark 9 3 128 dense NEW_EMPTY_WORDS_DIRECTORY
garnet_affine_fp16_projection_benchmark 9 3 128 indexed NEW_EMPTY_WORDS_DIRECTORY
```

Run all variants sequentially, then reverse variant order in a fresh cohort.
The separate relaxed-descriptor experiment uses the existing
`GARNET_METAL_MPP_REUSE64_RELAXED=ON` option with K=64 and no packed activations;
its fallback remains strict. It showed no material speed benefit here.

Audit tools are `audit_mistral_prefill_tiles.py` and
`audit_mistral_prefill_operand_study.py`. Raw collections are
`mistral-prefill-tile-operator1`, `mistral-prefill-pack-operator2` and
`mistral-prefill-relaxed-operator3`. The latter auditor's `--kind` selects `pack`
or `relaxed`. They check terminal nonoverlapping processes, compiled policy,
complete output bytes, original FP64 sample bounds and cleanup. These results
do not establish trained-model quality, comparative serving speed or a 2x gain.
