# Prepared half weights: exact operator outputs, no faster candidate

This experiment tests preparing Q4/group64 weights once as FP16, then feeding
the existing strict native MPP projection consumer. The purpose is to remove
repeated coefficient decoding from prefill without changing its numerical
contract. Earlier cached/streamed MPS experiments used a different consumer
and synthetic inputs; residency-set experiments did not prepare weight values.

The original native inline-Q4 kernel remains the timing control. Candidate
layouts are row-major FP16 and N64/K128 tiles padded to136 half values per row.
The tiled layout is also tested with the previously qualified vector/K32
schedule. A copied inline-Q4 kernel checks disabled writer-observer overhead.
All use the original M128/M256 tiles,256 threads,17,408 bytes of shared memory,
half operands, strict M16/N32/K16 matrix operations and FP32 accumulation/output.
This is an isolated experiment, with no runtime selection or xModel changes.

## Correctness and memory

Actual trained layer0/15/31 gate/down weights and native captured inputs/output
are replayed. The4096×4096 shape is a prefix of the gate weights/output and is
explicitly a shape proxy, not an actual Q/O projection capture. Both row tiles
cover127/128/129/255/256/257/383/384/511/512 selected rows with unique reordered
physical rows. The127-row operator probe does not enable native admission
below128 rows.

All1,620 cases pass exact full-parent comparisons:6,228,593,280 output words
and the same number of writer/guard words.243 independent FP64 samples pass
the unchanged2e-6 +4e-6×sum-absolute-products bound; maximum fraction0.018899.
Fourteen deliberate correctness-audit corruptions reject.

An independent Q4 decoder reconstructs every half value, tile-padding zero and
guard across18 complete prepared arrays, covering830,473,344 half words and
830,472,768 writer words. Counted preparation writes each element exactly once.
A separate uninstrumented preparation run proves the same complete half bits,
with writer arrays unchanged. Six complete raw half arrays are retained for
each preparation run, alongside16 complete projection output/writer arrays.

The largest tested row-major payload is117,440,512 bytes; tiled payload is
124,780,544 bytes. Both layouts are held only for this comparison, with a
recommended-working-set admission check and256MiB margin. Metadata named
`allocation_bytes` in the original host means requested buffer length, including
128 guard bytes. It does not prove actual physical allocation or peak memory.
The uninstrumented preparation host additionally records `allocatedSize`.
No full-model weight-cache budget, profile admission or production lifetime
has been qualified.

## Timing and decision

Two fresh processes run serially for each of two conditions: ordinary warmed
operator replay, and replay preceded by an explicit memory-pressure sweep.
Each mode/cell/process has five warmups and twenty measured rounds. Rotating
and reversing order puts every mode in each of five positions four times.
There are7,200 measured samples and1,800 warmup executions across128/256/512
rows and contiguous/permuted physical maps. Compilation and source/fixture
hashing precede each complete collection; postflight follows both processes.
Counted kernels and half-array hashing are excluded from timed collection.

The table gives control time / candidate time for512 contiguous rows. Above1
favors the candidate; ranges span the two process medians in each condition.

| Candidate | Warm gate | Warm down | Pressure gate | Pressure down |
|---|---:|---:|---:|---:|
| Prepared row-major FP16 |0.971–0.974× |0.940–0.943× |0.975–0.976× |0.956–0.958× |
| Prepared padded FP16 tiles |0.964–0.966× |0.861–0.864× |0.965–0.969× |0.957–0.959× |
| Prepared tiles plus vector/K32 |0.981–0.982× |0.929–0.956× |0.981–0.986× |0.981–0.984× |

**No replacement is selected.** All prepared candidates remain slower in the
largest real MLP cells under both tested conditions. Smaller rows and the gate
prefix proxy also remain recorded in the full timing report. These data do not
establish that all possible prepared-weight implementations are slower, or
identify a hardware-level cause.

Before every pressure-condition operator, a separate completed GPU command
reads256MiB and writes256MiB of alternating nonaliasing buffers. The sweep is
outside the measured operator interval. This tests sensitivity to that explicit
pressure condition; it is not proof that every hardware cache is cold. Both
the sweep and operator absolute GPU start/end timestamps are retained and
audited for serial ordering. Warmups are executed but their intervals are not
retained. The measured operator includes input gathering, projection and
inactive-output clearing when required. These are operator GPU durations,
not TTFT, token speed, model throughput or a matched vLLM kernel comparison.
Eighteen deliberate timing-evidence corruptions reject.

The latest serving comparison remains the
[fused MLP campaign](../mistral_fused_mlp_serving_benchmark/README.md).
No vLLM/SGLang inference was repeated. Coding remains0/8 on the strict suite;
the2× speed/throughput, half-TTFT, production integration, ANE/Gemma and original
Qwen objectives remain open. The original11 dirty source files are preserved.

## Reproduction

Dependencies are the pinned sibling native, trained-capture and strict-fragment
studies under `out/research/mistral7b-local-coder`. The generator requires a
fresh sibling directory. Do not run the following merely to remeasure an
unchanged candidate; saved-data auditors need no inference or GPU execution.

```sh
python3 -B prepare1.py /absolute/path/to/mistral7b-local-coder/FRESH-STUDY
python3 -B /absolute/path/to/FRESH-STUDY/run_correctness1.py
python3 -B audit_correctness1.py /absolute/path/to/FRESH-STUDY
python3 -B /absolute/path/to/FRESH-STUDY/run_normal_prepare1.py
python3 -B audit_normal_prepare1.py /absolute/path/to/FRESH-STUDY
python3 -B /absolute/path/to/FRESH-STUDY/run_timing1.py
python3 -B /absolute/path/to/FRESH-STUDY/prepare_pressure1.py
python3 -B /absolute/path/to/FRESH-STUDY/run_pressure1.py
python3 -B audit_timing1.py /absolute/path/to/FRESH-STUDY
```

For a saved-data check, each auditor accepts `STUDY FRESH-REPORT.json`.
Packaged auditors reproduce all three complete qualification reports exactly.
The generator reproduces both qualified shaders and all three host variants
exactly, without compilation or GPU work. Local studies are
`mistral-tiled-half-weight-screen1`,
`mistral-tiled-half-weight-replay-preparation1` and
`mistral-tiled-half-weight-replay-source1`.
