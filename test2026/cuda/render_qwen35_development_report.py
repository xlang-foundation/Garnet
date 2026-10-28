"""Render a qualified working report from completed saved evidence only."""
import argparse,datetime,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('output');a=p.parse_args()
root=Path(a.root);out=Path(a.output)
def read(name):
    path=root/name
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None
def link(name):return f'[{name}]({(root/name).as_posix()})'
def median(values):return statistics.median(values) if values else None
def fmt(value):return f'{value:,.1f}' if value is not None else 'unavailable'
lines=['# Qwen3.5 CUDA development report — in progress','',
    f'Generated from saved artifacts at {datetime.datetime.now(datetime.timezone.utc).isoformat()}.', '',
    '**Incomplete: Garnet has not achieved the 2× latency target. Broad correctness, '
    'isolated peak/residency coverage, and the attention prototype remain open. Current execution is Windows/CUDA; existing Metal paths are preserved and separately audited.**','',
    'The original screenshots came from another machine. They are context, not a quantitative baseline for this RTX4080 GPU0 run.', '',
    '## Protocol and precision','',
    '- Qwen3.5-2B and 4B use the same pinned checkpoints with BF16 large matrices; some checkpoint norm weights are FP32. vLLM load logs report casting some of these norms into BF16 parameters. Exact norm-weight/intermediate precision is therefore not matched across engines. Garnet counts the shared BF16 embedding/head once. Original Garnet cohorts exclude unused MTP weights and have speculation OFF; experimental MTP cohorts include their engines/buffers.',
    '- Correction (October23): saved LM Studio2B batches1/2/4/8 actually use MTP, with28/56/112/224 draft-acceptance events. The requested false flag did not establish execution. Saved LM4B logs have no drafting events, so their speculation status remains unresolved. Original vLLM cohorts configure speculation OFF; additional tested MTP profiles are reported separately. Exact proposal settings are not matched. See '+link('saved-comparator-speculation-audit.json')+'. Earlier contrary descriptions are superseded; raw timing logs are unchanged.',
    '- Garnet uses BF16 projection input/output with FP32 accumulation and reduced-precision reductions disabled; residual, norm, RoPE, attention and recurrent state are FP32. Other engines have different intermediate arithmetic. Equal BF16 source weights do not imply bit-exact execution.',
    '- Current fixed workload: 320×192 real-image variants, 60 merged visual tokens, 85 prompt tokens, 32 output-token limit, reasoning off, greedy sampling. A unique8×8 corner tile is added to each invocation; source/JPEG/decoded-RGB hashes are saved. Four related image families are not a broad quality dataset.',
    '- Seven rounds of four groups: two warmup rounds, five measured rounds. One model/engine/profile at a time. Batch groups contain B simultaneous requests inside that engine.',
    '- Current partitioned LM cohorts verify B slots,512-token slot capacity and `kv_unified=false`. Earlier unified-pool LM cohorts are retained as provisional/historical. Their slot ceiling alone did not prove independent per-row capacity.',
    '- Garnet and vLLM image hashes and prompt IDs must match. LM raw prompt IDs are unavailable; input counts and full prompt-evaluation logs are checked. Unique images and full prompt evaluations evidence actual work, not a verified setting disabling every internal cache.',
    '- Equal-output groups below contain32 sampled tokens per row in every engine. Group completion latency is not aggregate throughput or individual streaming latency. API timing boundaries differ.', '',
    'Protocol: '+link('inputs/matched512-batches/manifest.json')+'. Configuration sources: '
    '[matched profile](../../test2026/cuda/qwen35_matched_benchmark_profiles.json), '
    '[development methods](qwen35-cuda-development-log.md).','',
    '## Completed capacity-matched comparisons','',
    '| Model | Batch | Equal-output groups | Garnet ms | LM Studio ms | vLLM ms | 2× fastest-baseline target ms |',
    '|---|---:|---:|---:|---:|---:|---:|']
completed=[]
latest_internal=[(size,batch,read(f'packed-study-{size}-b{batch}-internal-stores-summary.json'),
    read(f'packed-study-{size}-b{batch}-internal-stores-all-baselines.json')) for size,batch in (('2b',1),('4b',8))]
if all(study and baseline for _,_,study,baseline in latest_internal):
    snapshot=['## Completed internal packed-store checkpoint','',
        'Completed DCD0 benchmark build: fresh sequential ABBA, 28 requests per role, all 20 warm groups retained. '
        'All-request inputs, outputs, stops, profiles, discrete MTP work and native owner counters pass. '
        'Only internalPackedStores changes. This remains opt-in.','',
        '| Model / batch | Control ms, all 20 | Internal stores ms, all 20 | Candidate latency reduction |',
        '|---|---:|---:|---:|']
    for size,batch,study,baseline in latest_internal:
        snapshot.append(f'| {size.upper()} B{batch} | {study["aggregate_group_median_ms"]["a"]:.2f} | '
            f'{study["aggregate_group_median_ms"]["b"]:.2f} | {study["comparisons"]["a_to_b"]["reduction_fraction"]*100:+.2f}% |')
    snapshot += ['','The following comparator table uses one common equal-output subset per profile, separately '
        'from the all-20 comparison above. Competitors are saved single cohorts.','',
        '| Model / batch | Common groups | Garnet internal ms | LM Studio ms | Fastest saved vLLM ms | 2× target ms |',
        '|---|---:|---:|---:|---:|---:|']
    for size,batch,study,baseline in latest_internal:
        v=baseline['median_group_ms'];snapshot.append(f'| {size.upper()} B{batch} | {baseline["common_equal32_groups"]} | '
            f'{v["garnet_b"]:.2f} | {v["lmstudio"]:.2f} | {v[baseline["fastest_tested_baseline"]]:.2f} | '
            f'{baseline["half_fastest_baseline_ms"]:.2f} |')
    snapshot += ['','Both requested 2B vLLM convolution-cache settings are included; actual allocated cache views '
        'are verified in the separate GPU diagnostic below. Norm-weight and intermediate precision still differ across engines. Neither model meets the '
        '2× target. All eight roles fall inside the sampled observer window, with zero recorded model/compiler/'
        'diagnostic overlap and zero observer metadata errors; sampling cannot prove complete isolation. '
        'Post-load owned savings are a separate measure from the warmed process counters below.','',
        'Internal-store latency chart: '+link('progress-internal-store-baselines.png')+'. Raw summaries: '
        +', '.join(link(f'packed-study-{size}-b{batch}-internal-stores-all-baselines.json') for size,batch,_,_ in latest_internal)+'.','']
    lines[8:8]=snapshot
    memory_pairs=[(size,batch,read(f'internal-stores-memory-{size}-b{batch}-pair-summary.json')) for size,batch,_,_ in latest_internal]
    if all(pair for _,_,pair in memory_pairs):
        current_memory=['### Internal-store profile warmed Windows process counters','',
            '| Model / batch | Control GiB | Internal stores GiB | Saved LM Studio GiB | Process-counter reduction MiB |',
            '|---|---:|---:|---:|---:|']
        for size,batch,pair in memory_pairs:
            lm=read(f'memory-lmstudio-{size}-b{batch}-512-partitioned-summary.json')
            assert pair['same_gpu_uuid_as_saved_lmstudio']==lm['adapter']['uuid']
            values=[pair['profiles'][mode]['warm_process_counter_bytes']['TotalCommitted']/(1<<30) for mode in ('control','candidate')]
            lm_value=lm['phases']['steady_after']['bytes']['TotalCommitted']['median']/(1<<30)
            current_memory.append(f'| {size.upper()} B{batch} | {values[0]:.3f} | {values[1]:.3f} | {lm_value:.3f} | '
                f'{pair["control_minus_candidate_counter_bytes"]["TotalCommitted"]/(1<<20):.1f} |')
        audit=read('internal-store-memory-observed-v1-process-audit.json')
        assert audit['all_four_cases_covered']
        current_memory += ['','Fresh sequential control/candidate processes, eight steady-after samples per case, '
            'six exact input/output/stop/profile controls, same physical GPU UUID as saved LM Studio. '
            'All four model/monitor intervals are covered by the Windows process observer; '
            f'{audit["known_overlap_count"]} known overlaps and {len(audit["observer_metadata_errors"])} metadata errors, '
            f'maximum snapshot gap {audit["maximum_snapshot_gap_seconds"]:.3f} s. This is sampled named-process presence. '
            'Counter medians and retained allocations are different measures; all phase statistics retain observed '
            'maxima, not true lifetime peaks or physical residency. 4B remains above saved LM Studio.','',
            'Memory chart: '+link('progress-internal-store-memory.png')+'. Pair evidence: '
            +', '.join(link(f'internal-stores-memory-{size}-b{batch}-pair-summary.json') for size,batch,_ in memory_pairs)+'.','']
        lines[8+len(snapshot):8+len(snapshot)]=current_memory
vision_studies=[(size,batch,read(f'packed-study-{size}-b{batch}-vision-graph-summary.json'),
    read(f'packed-study-{size}-b{batch}-vision-graph-all-baselines.json')) for size,batch in (('2b',1),('4b',8))]
if all(study and baseline for _,_,study,baseline in vision_studies):
    latest=['## Latest completed CUDA study — vision graph replay','',
        'Fresh sequential ABBA for 2B B1 then4B B8; all28 requests per role and all20 warm groups retained. '
        'Only visionCudaGraph changes. Internal packed stores remain OFF for2B and ON for4B in both roles. '
        'Every request matches outputs/stops/cache/discrete-MTP work and decoder/speculative ownership; '
        'fresh default controls also preserve the prior cohorts. Actual vision counters show one context, '
        'one capture, replay and no capture failures, with1,474,560 staging bytes counted explicitly.','',
        '| Model / batch | Control ms, all20 | Vision replay ms, all20 | Candidate reduction |',
        '|---|---:|---:|---:|']
    for size,batch,study,baseline in vision_studies:
        latest.append(f'| {size.upper()} B{batch} | {study["aggregate_group_median_ms"]["a"]:.2f} | '
            f'{study["aggregate_group_median_ms"]["b"]:.2f} | {study["comparisons"]["a_to_b"]["reduction_fraction"]*100:.2f}% |')
    latest += ['','Two repeats do not establish significance or causal benefit; this remains opt-in. '
        'The comparator table uses separate common32-token groups and saved competitor cohorts, including '
        'both tested2B vLLM cache settings. API boundaries, speculation, norm-weight casting and intermediate '
        'precision differ. Neither model meets the2× target.','',
        '| Model / batch | Common groups | Vision replay ms | LM Studio ms | Fastest vLLM ms | 2× target ms |',
        '|---|---:|---:|---:|---:|---:|']
    observer_notes=[]
    for size,batch,study,baseline in vision_studies:
        v=baseline['median_group_ms']
        latest.append(f'| {size.upper()} B{batch} | {baseline["common_equal32_groups"]} | {v["garnet_b"]:.2f} | '
            f'{v["lmstudio"]:.2f} | {v[baseline["fastest_tested_baseline"]]:.2f} | {baseline["half_fastest_baseline_ms"]:.2f} |')
        audit=read(f'vision-graph-observed-v1-{size}-b{batch}-process-audit.json')
        assert audit['all_four_roles_covered']
        observer_notes.append(f'{size.upper()} B{batch} observer: all roles covered, {audit["known_overlap_count"]} known '
            f'overlaps, {len(audit["observer_metadata_errors"])} metadata errors, maximum snapshot gap '
            f'{audit["maximum_snapshot_gap_seconds"]:.3f}s. Finite sampling does not prove complete isolation.')
    for note in observer_notes:latest+=['',note]
    latest += ['','Candidate derived post-first-sample group rates (B1/B2 run medians, tokens/s): '
        +'; '.join(f'{size.upper()} batch{batch}: {study["per_run"]["b1"]["median_derived_post_first_sample_tokens_per_second"]:.1f} / '
            f'{study["per_run"]["b2"]["median_derived_post_first_sample_tokens_per_second"]:.1f}' for size,batch,study,_ in vision_studies)
        +'. This subtracts one sampled token per row, includes host/stop overhead and aggregates batch rows; '
        'it is not per-request streaming speed. Raw phase medians remain separate and non-additive.','',
        'Latest chart: '+link('progress-vision-graph-baselines.png')+'.','']
    vision_memory=[(size,batch,read(f'vision-graph-memory-{size}-b{batch}-pair-summary.json'))
                   for size,batch,_,_ in vision_studies]
    if all(pair for _,_,pair in vision_memory):
        memory_audit=read('vision-graph-memory-observed-v1-process-audit.json')
        assert memory_audit['all_four_cases_covered']
        latest += ['### Exact vision-profile warmed process memory','',
            '| Model / batch | Control GiB | Vision replay GiB | Saved LM Studio GiB | Replay increase MiB |',
            '|---|---:|---:|---:|---:|']
        for size,batch,pair in vision_memory:
            lm=read(f'memory-lmstudio-{size}-b{batch}-512-partitioned-summary.json')
            assert pair['same_gpu_uuid_as_saved_lmstudio']==lm['adapter']['uuid']
            assert all(p['sample_count']>=3 for p in pair['profiles'].values())
            values=[pair['profiles'][mode]['warm_process_counter_bytes']['TotalCommitted']/(1<<30)
                    for mode in ('control','candidate')]
            lm_value=lm['phases']['steady_after']['bytes']['TotalCommitted']['median']/(1<<30)
            latest.append(f'| {size.upper()} B{batch} | {values[0]:.3f} | {values[1]:.3f} | {lm_value:.3f} | '
                f'{-pair["control_minus_candidate_counter_bytes"]["TotalCommitted"]/(1<<20):.1f} |')
        latest += ['','Fresh sequential processes use the exact completed timing profiles (internal stores OFF '
            'for 2B and ON for 4B), with six matching input/output/stop/profile checks per case. '
            'Replay adds 1,474,560 owned staging bytes and 4 MiB to warmed committed process-counter medians '
            'in each tested pair. Shared counters remain 0.076 GiB. 4B remains above saved LM Studio. '
            'These are sampled committed counters, not physical residency or true lifetime peaks; vLLM '
            'process totals remain unavailable. Allocator ownership and process counters are separate measures.','',
            f'All four model/monitor intervals are observer-covered: {memory_audit["known_overlap_count"]} '
            f'known overlaps, {len(memory_audit["observer_metadata_errors"])} metadata errors, maximum snapshot '
            f'gap {memory_audit["maximum_snapshot_gap_seconds"]:.3f} s. Sampling does not prove complete isolation.','',
            'Memory chart: '+link('progress-vision-graph-memory.png')+'. Pair evidence: '
            +', '.join(link(f'vision-graph-memory-{size}-b{batch}-pair-summary.json') for size,batch,_ in vision_memory)+'.','']
    else:
        latest += ['Exact vision-profile process memory is pending; the earlier internal-store memory table '
            'uses different 2B flags and must not be combined with this latency as one profile.','']
    lines[8:8]=latest
for size in ('2b','4b'):
    for batch in (1,2,4,8):
        name=f'matched512-{size}-b{batch}-partitioned-summary.json';data=read(name)
        if not data or not data.get('all_three_engines_present') or not data.get('lmstudio_independent_capacity_verified'):
            lines.append(f'| {size.upper()} | {batch} | pending | — | — | — | — |');continue
        assert data['all_input_hashes_equal'] and data['garnet_vllm_prompt_ids_equal']
        walls=data['equal_work_median_group_completion_ms'];target=min(walls['vllm'],walls['lmstudio'])/2
        lines.append(f'| {size.upper()} | {batch} | {data["equal_32_sample_groups"]} | {walls["garnet"]:.1f} | {walls["lmstudio"]:.1f} | {walls["vllm"]:.1f} | {target:.1f} |')
        completed.append((size,batch,name,data))
lines+=['','Evidence: '+', '.join(link(name) for _,_,name,_ in completed)+'.', '',
    '## Aggregate measured sampled throughput','',
    'Total sampled tokens divided by total group wall time across20 measured groups, including unequal stop lengths. '
    'This metric does not establish a per-request latency speedup.','',
    '| Model | Batch | Garnet tok/s | LM Studio tok/s | vLLM tok/s |','|---|---:|---:|---:|---:|']
for size,batch,_,data in completed:
    t=data['all_measured_aggregate_sample_throughput']
    lines.append(f'| {size.upper()} | {batch} | {t["garnet"]:.1f} | {t["lmstudio"]:.1f} | {t["vllm"]:.1f} |')
lines+=['','## Experimental direct-KV attention candidate','',
    'Opt-in `fusedAttention` is not accepted or enabled by default. Baseline files above remain unchanged. '
    'These comparisons reuse saved competitor cohorts and select four-way equal32-token groups.','',
    '| Model | Batch | Candidate | Groups | Prior Garnet ms | Candidate ms | Own latency reduction | Same sequences /28 |',
    '|---|---:|---|---:|---:|---:|---:|---:|']
for size in ('2b','4b'):
    for batch in (1,2,4,8):
        for suffix in ('-fused-attention','-fused-attention-workspace16k','-fused-attention-chunk16'):
            candidate=read(f'matched512-{size}-b{batch}{suffix}-candidate.json')
            if not candidate:continue
            w=candidate['median_group_ms'];fraction=candidate['candidate_control_latency_fraction']
            lines.append(f'| {size.upper()} | {batch} | {suffix[1:]} | {candidate["equal_32_four_way_groups"]} | {w["control"]:.1f} | {w["candidate"]:.1f} | {(1-fraction)*100:.1f}% | {candidate["same_sequence_cases"]}/28 |')
lines+=['','The strict batch4 decoder fixture passed; batch8 still fails a padded-step hidden tolerance. '
    'Six changed2B cases reproduced, with192 independent teacher-forced decisions:185 deterministic top1 matches, '
    '189 reference-maximum-set choices and192 reconstructed-head matches. Three non-tied differences remain. '
    'The oracle consumes native image embeddings; this is not independent vision or broad quality acceptance.','',
    'One instrumented2B request has GPU activity span466.237ms, union423.850ms and kernel-duration sum417.377ms. '
    'BF16 projection families and the head account for about83.2% of summed kernel time. Actual GPU copy time '
    'is6.239ms; blocking API durations must not be interpreted as transfer time. These are profiling diagnostics, '
    'not matched latency or a projected speedup. See '+link('profile-fused-attention-2b-b1-summary.json')+'.']
lines+=['','## Batch1 phase diagnostics','',
    'These are engine-specific diagnostics, not the primary latency ranking. Garnet TTFT includes reset/JPEG/vision/prefill; '
    'vLLM request metrics start after caller image decode; LM reports a server TTFT. '
    'Garnet decode uses post-prefill steps/time, vLLM uses first-to-last token timestamps, and LM supplies its own rate.','',
    '| Model | Engine / scope | Median TTFT ms | Median decode tok/s |', '|---|---|---:|---:|']
for size,batch,_,_ in completed:
    if batch!=1:continue
    g=[r for r in read(f'matched512-garnet-{size}-b1.json') if not r['warmup']]
    v=[r for r in read(f'matched512-vllm-{size}-b1.json') if not r['warmup']]
    lm=[r for r in read(f'matched512-lmstudio-{size}-b1-partitioned.json') if not r['warmup']]
    vttft=[];vdecode=[]
    for r in v:
        m=r['records'][0]['metrics'] or {}
        if m.get('first_token_latency') is not None:vttft.append(m['first_token_latency']*1000)
        delta=m.get('last_token_ts',0)-m.get('first_token_ts',0)
        if delta>0:vdecode.append((m['num_generation_tokens']-1)/delta)
    stats=[r['records'][0]['result']['stats'] for r in lm]
    phase=[('Garnet / Model API',median([r['ttft_seconds']*1000 for r in g]),median([r['decode_steps']/r['decode_seconds'] for r in g])),
        ('vLLM / scheduler request',median(vttft),median(vdecode)),
        ('LM Studio / server-reported',median([s['time_to_first_token_seconds']*1000 for s in stats]),median([s['tokens_per_second'] for s in stats]))]
    for engine,ttft,decode in phase:lines.append(f'| {size.upper()} | {engine} | {fmt(ttft)} | {fmt(decode)} |')
lines+=['','## Memory evidence and scope','',
    'The excessive-memory diagnosis led to one BF16 text-weight set per layer serving prefill/decode, '
    'one shared BF16 embedding/head, GPU-resident state, and explicit allocator tracking. '
    'This applies to the new Qwen3.5 CUDA path; the legacy Qwen3-VL path is not thereby proven memory-optimized.','',
    '| Model | Batch | Stable retained ownership GB (decimal) |', '|---|---:|---:|']
for size in ('2b','4b'):
    data=read(f'matched512-{size}-allocation-growth.json')
    if data:
        for profile in data['profiles']:lines.append(f'| {size.upper()} | {profile["batch"]} | {profile["bytes"]["owned_model_bytes"]/1e9:.3f} |')
lines+=['','Retained ownership excludes some driver/external-library/graph-executable/transient allocations. '
    'Engine live counters include weights and runtime buffers, not just workspace. Per-engine peaks are never added as a process peak. '
    'These values are not isolated process VRAM or proof of physical residency.', '',
    'Historical batch1 WDDM process-counter diagnostics were close to LM Studio, but used Garnet384 versus LM512 context; '
    'they do not substitute for current-profile isolated peak/residency measurements. WSL host VM counters had ambiguous duplicate instances; '
    'vLLM worker allocator counters were reported separately. See '+link('ISOLATED_MEMORY.md')+'.','',
    '## Correctness and regression evidence','',
    '- Component/state/masking/CUDA-graph tests cover selected independent fixtures. The real-model decoder oracle uses native image embeddings, so it bypasses independent vision validation.',
    '- Changed output trajectories after batched full-attention projections reproduced;2B and4B each matched64/64 teacher-forced decoder decisions. This is narrow trajectory evidence.',
    '- The4B batch8 candidate decoder reference had750/768 exact IDs and754/768 maximum-set choices; fourteen non-tied rank2 differences remain unresolved. A synthetic batch8 hidden check fails on both archived and current builds; layer tracing localizes its first error after an exactly matching recurrent layer.',
    '- Legacy Qwen3-VL-2B current-DLL smoke passes two16-token requests with exact text/count replay. Its preprocessing produces308patches/77visual tokens for this input, not Qwen3.5’s60. Raw IDs are not exposed by that legacy API. See '+link('qwen3-vl-matched-grid-regression.json')+'.',
    '- Metal existing-binary pretrained layer0/3 reference tests passed batches1/2 and16 stateful steps. Original/current4B Python factories produced identical8-token outputs twice on one real image. Remote native Metal sources differ from the Windows base; fresh local-source native build preservation and broad quality remain open. See '+link('metal-factory-preservation-summary.json')+'.','',
    '## Failed experiments retained','',
    '- TensorRT conditional branch around a plugin was rejected by the builder.',
    '- 4Bbatch8 attention builds exceeded64MiB and128MiB workspace caps; a256MiB layer probe and subsequent384-context full profile passed. This is not proof every512-context profile fits.',
    '- Original unified LMbatch8 load passed slot-ceiling checks but failed KV allocation on the first group. Partitioned non-unified512 slots completed. Prior shared-pool measurements remain historical/provisional.',
    '- The first legacy regression used an incompatible240-patch compiled profile; the retry derives its308-patch profile from native preprocessing.',
    '- Harness startup/schema failures and native precision discrepancies remain in raw logs; successful retries do not erase them.','',
    '## Remaining acceptance work','',
    '- All eight baseline profiles completed; preserve these artifacts and compare accepted optimizations on the same workload. Capacity and full prompt work are audited; a setting disabling every internal cache remains unverified.',
    '- Resolve or rigorously bound numerical discrepancies and add independent vision/held-out quality coverage, admission/cache-boundary tests at official-model scope.',
    '- Resolve numerical and broader pretrained-model gates for the opt-in attention candidate. TensorRT/component passing tests and selected speed measurements do not establish full model acceptance.',
    '- Repeat relevant isolated steady/peak/residency memory diagnostics on the accepted implementation and larger profiles; do not use allocator ownership as peak VRAM.',
    '- Retain the existing Metal compatibility audit when shared factories change; current benchmark execution is Windows/CUDA.',
    '- Achieve the requested2× full-request latency advantage over the fastest matched baseline with acceptable memory/correctness, then audit every requirement before marking the goal complete.','']
prototype_logs={kind:root/f'test-attention-core-racefix-{kind}.log' for kind in ('parity','racecheck','memcheck','synccheck')}
if all(path.exists() for path in prototype_logs.values()):
    contents={kind:path.read_text(errors='replace') for kind,path in prototype_logs.items()}
    passed=('QWEN35_ATTENTION_CORE_COMPLETE cases 7' in contents['parity'] and
        '0 hazards displayed (0 errors, 0 warnings)' in contents['racecheck'] and
        all('ERROR SUMMARY: 0 errors' in contents[k] for k in ('memcheck','synccheck')))
    lines+=['## Attention-core component status','',
        ('Standalone7-case parity, exact KV/input/guard/graph replay checks and memcheck/racecheck/synccheck pass. '
         'The original reduction raced despite matching outputs; its failure logs/binary are retained. '
         'Opt-in serving integration now has selected measurements above; component checks alone do not establish quality or promotion.' if passed else
         'Saved component logs do not establish a complete pass; inspect failures before integration.'),'']
memory=read('memory-attention-4b-b8-512-summary.json')
if memory:
    lines+=['## Attention candidate: isolated4B batch8 memory','',
        'Context512/chunk4, same attention binary as the candidate cohort. Timings excluded. '
        'PCI/CUDA-LUID/node resolved WDDM process counters; no ambiguous nonzero duplicate instances.','',
        '| Phase | Enclosed samples | Dedicated GB median | Shared GB median |','|---|---:|---:|---:|']
    for name in ('steady_before','steady_after','after_model_close'):
        phase=memory['phases'][name];values=phase['bytes']
        lines.append(f'| {name} | {phase["samples"]} | {values["DedicatedUsage"]["median"]/1e9:.3f} | {values["SharedUsage"]["median"]/1e9:.3f} |')
    lines+=['',f'Largest observed dedicated counter {memory["largest_observed_bytes"]["DedicatedUsage"]/1e9:.3f}GB; '
        'sampled maximum is not a true lifetime peak or physical-residency proof. '
        'There is no matched large-batch comparator memory cohort yet. See '+link('memory-attention-4b-b8-512-summary.json')+'.','']
lines+=['## MTP feasibility — separate from baseline acceptance','',
    'Experimental native MTP is now connected behind a default-OFF profile (depth1/3/7). '
    'On one traced2B trajectory, independent and native implementations propose the same31 tokens, with24/31 '
    'matching teacher next tokens. Target-head reconstruction is32/32. This consumes saved native target rows; '
    'it is not end-to-end inference or a latency claim. Three synthetic MTP profiles still fail the unchanged '
    'strict numerical bound; input/unwritten-cache preservation, graph replay and borrowed-head lifetime checks '
    'completed separately. A reciprocal-normalization hypothesis did not fix the failure and was reverted.','',
    'Conditional native drafting from teacher-correct prefixes suggests18/12/11 hypothetical target passes at '
    'depth1/3/7 on this single32-token trajectory. This omits actual verification/commit/corrected-draft work '
    'and does not establish rollout acceptance or speedup. The subsequent verifier/session implementation '
    'passed rollback/prefix/capacity/stop/reset tests: six synthetic profiles with24cases each, plus real2B/4B '
    'fresh-image probes with exact32tokens and all final target state bytes equal to native serial controls. '
    'This is native target equivalence, not broad independent-model quality acceptance.','',
    'The full2B B1 C16 depth3 MTP cohort completed28requests, all28sequences matching the C16 control. '
    'Across16four-way equal32groups, latency increased353.2359→384.22405ms (+8.77%); retained ownership '
    'increased4,578,699,912→4,745,831,440B (+167,131,528). Median20target forwards include12verification '
    'and8prefix-recomputation passes. This speed candidate is rejected and stays default off. '
    'Saved vLLM disables speculation; LM Studio runtime drafting is audited above. Algorithm settings differ. See '+link('matched512-2b-b1-verified-mtp-d3-candidate.json')+'.','',
    'Indexed draft heads and recurrent prefix replay are measured separately below. '
    'Every emitted proposal remains target-verified; no overall goal or quality acceptance is implied.','']
indexed=read('matched512-2b-b1-verified-mtp-d3-indexed8192-candidate.json')
if indexed:
    w=indexed['median_group_ms'];owned=indexed['retained_owned_bytes']['candidate']
    lines+=['## Indexed MTP head — measured experiment','',
        'The proposal head reads selected rows directly from the existing tied BF16 table. A GPU stable sort selects8192 '
        'candidates from the previous target distribution. Every emitted proposal is checked by the full target head; '
        'no second embedding table or persistent gathered weight matrix is stored. Fixed independent FP64 dot bounds, '
        'CPU stable sorting, guards/invalid/alias/nonfinite/graph tests and CUDA memory/race/synchronization checks passed. '
        'Native API6CPU-oracle cases and seven synthetic session profiles passed, plus fresh-image2B/4B probes with exact '
        'serial target outputs and all final target-state bytes. This preserves native target behavior on tested cases; '
        'the pre-existing independent-model quality gaps remain.','',
        f'The28-request2B B1 C16 D3 cohort matched all{indexed["same_sequence_cases"]} control output sequences. '
        f'Across{indexed["equal_32_four_way_groups"]}common equal32groups, median latency was{w["candidate"]:.2f}ms '
        f'versus{w["control"]:.2f}ms for non-speculative C16 Garnet, {w["lmstudio"]:.2f}ms for LM Studio and{w["vllm"]:.2f}ms for vLLM. '
        f'Retained ownership was{owned[0]/1e9:.3f}GB. Speculation differs from the saved OFF comparator cohorts. '
        'The2× target is unmet; this option remains default off. See '+link('matched512-2b-b1-verified-mtp-d3-indexed8192-candidate.json')+'.','',
        '![Completed2B experiments](../../../work/qwen35-cuda/progress-speculative-2b-b1.png)','']
replays=[read(f'matched512-2b-b1-verified-mtp-d{depth}-indexed8192-delta-replay-candidate.json') for depth in (3,7)]
if any(replays):
    lines+=['## Recurrent prefix replay — measured experiment','',
        'Partial commits retain projected inputs and rerun only the small convolution/delta recurrence from saved state. '
        'They restore unaccepted attention KV suffixes and select the accepted target hidden rows. No per-token copies '
        'of all recurrent states or second set of large projection weights is stored. Helper small weights and buffers, '
        'parent retained payloads and all session allocations are counted. The original restore/recompute path remains available.','',
        'Four transactional verifier profiles and seven24-case session profiles passed exact native state/output/slot and '
        'stop/limit/rollback/capacity checks. Fresh-image2B/4B probes each passed two MTP requests plus serial controls, '
        'matching all32output IDs and every final target-state byte. These tests do not close independent quality gaps.','',
        '| Depth | Equal32groups | Garnet ms | LM Studio ms | vLLM ms | Same sequences /28 | Owned GB |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for candidate in replays:
        if not candidate:continue
        w=candidate['median_group_ms'];owned=candidate['retained_owned_bytes']['candidate'];assert len(owned)==1
        lines.append(f'| {candidate["candidate_speculative_mtp_depth"]} | {candidate["equal_32_four_way_groups"]} | {w["candidate"]:.2f} | {w["lmstudio"]:.2f} | {w["vllm"]:.2f} | {candidate["same_sequence_cases"]}/28 | {owned[0]/1e9:.3f} |')
    lines+=['','All replay requests report zero full prefix recomputations; recurrent replays are counted separately. '
        'Saved vLLM disables speculation; LM Studio runtime drafting is audited separately; algorithms differ. Default OFF,2× target unmet. '
        'No isolated process peak/residency comparison is implied by retained ownership.','']
hybrids=[read(f'matched512-2b-b1-hybrid-p{width}-d4-mtp3-k8192-replay-candidate.json') for width in (32,64,128)]
if any(hybrids):
    lines+=['## Shared projection profiles — measured experiment','',
        'Separate bounded prefill/decode graphs borrow one readonly BF16 projection pool and share the same target state/last-hidden tensors. '
        'Header-audited target matrices total2,745,434,112B for2B and7,138,181,120B for4B; real-model retained pool counters match these payloads. '
        'Both graph sets, small constants, runtime buffers and the MTP session remain included in ownership totals. '
        'Synthetic constant-versus-pooled and independent explicit-copy controls pass; this is not process-residency proof.','',
        '| Prefill / decode | Equal32groups | Garnet ms | LM Studio ms | vLLM ms | Same sequences as older C16 /28 | Owned GB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for candidate in hybrids:
        if not candidate:continue
        w=candidate['median_group_ms'];owned=candidate['retained_owned_bytes']['candidate'];assert len(owned)==1
        width=candidate['candidate_suffix'].split('-p')[1].split('-')[0]
        lines.append(f'| {width} /4 | {candidate["equal_32_four_way_groups"]} | {w["candidate"]:.2f} | {w["lmstudio"]:.2f} | {w["vllm"]:.2f} | {candidate["same_sequence_cases"]}/28 | {owned[0]/1e9:.3f} |')
    lines+=['','Cross-profile precision remains open: a synthetic narrow-versus-wide comparison differs only in key-cache values (max7.15e-7), '
        'while an identical-shape control with independent explicit state copies matches byte-for-byte. '
        'Real2B and4B probes match their own serial target outputs and final state bytes;4B differs from the older saved caption after token26. '
        'That original assertion failure remains retained. Changed full-cohort2B trajectories also need independent analysis. '
        'All options stay default OFF; no quality/default/2× acceptance. Saved vLLM disables speculation; LM Studio runtime drafting is audited separately.','']
packed=read('matched512-2b-b1-packed-p64-d4-mtp3-k8192-replay-candidate.json')
lines+=['## Packed whole-decoder experiment','',
    'An explicit default-off profile compiles each target chunk into one graph with a canonical packed state tensor. '
    'Wide prefill and narrow decode share one readonly BF16 projection pool and state. FP32 layer boundaries remain materialized. '
    'Partial acceptance uses one graph of small delta-core replays and exact accepted-span KV selection; target matrix projections are not rerun. '
    'Eight two-layer packed-program profiles passed byte-exact same-shape native controls. Four transactional verifier profiles '
    'and seven24-case speculative profiles passed rollback, partial acceptance, EOS, length and capacity checks. '
    'Independent pretrained quality, process memory and2× acceptance remain open.','']
if packed:
    w=packed['median_group_ms'];owned=packed['retained_owned_bytes']['candidate'];assert len(owned)==1
    lines+=[f'Completed2B B1 cohort: {w["candidate"]:.2f}ms versus same-profile per-layer control {w["control"]:.2f}ms, '
        f'saved LM Studio {w["lmstudio"]:.2f}ms and vLLM {w["vllm"]:.2f}ms across{packed["equal_32_four_way_groups"]}equal32groups. '
        f'{packed["same_sequence_cases"]}/28 output sequences match the same-profile control; retained ownership {owned[0]/1e9:.3f}GB. '
        'Saved vLLM disables speculation; LM Studio runtime drafting is audited separately. No quality/default/2× acceptance is inferred.','']
else:lines+=['No completed packed-program matched timing cohort is available yet. Component/diagnostic times are not speed evidence.','']
resident=read('matched512-2b-b1-hybrid-p64-d4-mtp3-k8192-replay-resident-candidate.json')
if resident:
    w=resident['median_group_ms'];owned=resident['retained_owned_bytes']['candidate'];assert len(owned)==1
    lines+=['## Inline request controls and direct chunk embeddings','',
        'The default-off residentControls profile transports small scalars through CUDA launch parameters and gathers the shared BF16 table into caller-owned padded chunks. '
        'It changes data transport and temporary allocation behavior while preserving model arithmetic. Sixteen independent IO profiles, '
        'four parent controls, four transactional verifier profiles and eight24-case speculative profiles passed exact native comparisons; '
        'all three native IO sanitizer checks found zero errors.','',
        f'Completed2B B1 cohort: {w["candidate"]:.2f}ms versus {w["control"]:.2f}ms for the same-profile old IO control, '
        f'across{resident["equal_32_four_way_groups"]}equal32groups; all{resident["same_sequence_cases"]}/28sequences match. '
        f'Retained ownership {owned[0]/1e9:.3f}GB. Asynchronous prefill grew the MTP context pool from1to8contexts, '
        'adding31.1MB. A separate default-off serial graph reuse experiment addresses this growth; its result appears below when available. '
        'The2× target remains unmet, independent quality gaps remain, and saved vLLM disables speculation; LM Studio runtime drafting is audited separately.','']
serial=read('matched512-2b-b1-hybrid-p64-d4-mtp3-k8192-replay-resident-serial-candidate.json')
if serial:
    w=serial['median_group_ms'];owned=serial['retained_owned_bytes']['candidate'];assert len(owned)==1
    lines+=['## Serial MTP graph reuse — measured memory fix','',
        f'The same16equal32groups measure {w["candidate"]:.2f}ms versus {w["control"]:.2f}ms for resident IO with the old pool policy. '
        f'All{serial["same_sequence_cases"]}/28output sequences match. Retained ownership returns to {owned[0]/1e9:.3f}GB, '
        'saving31,098,368B with oneMTPcontext. No latency gain is claimed. '
        'Pending replay requires a captured graph, identical bindings, static shapes, same stream/thread semantics and no provider owner. '
        'Changed cases retain conservative retirement. Gated real-TRT pool tests and all three sanitizers passed; '
        'ten24-case MTP profiles preserve exact states/outputs/stops/limits. Default OFF; independent quality, physical memory and2× acceptance remain open.','']
register=read('matched512-2b-b1-hybrid-p64-d4-mtp3-k8192-replay-resident-serial-register128-candidate.json')
if register:
    w=register['median_group_ms'];owned=register['retained_owned_bytes']['candidate'];assert len(owned)==1
    lines+=['## Fixed128 recurrent state kernel — measured experiment','',
        'The optional kernel keeps the original sequential FP32 recurrence/reductions and convolution workspace, '
        'using fixed128-state loops and volatile shared Q/K. Native parent, prefix-commit and MTP controls pass exact bytes; '
        'all49pretrained final target-state hashes match the same-profile old kernel. Standalone independent FP64 and all three sanitizers pass. '
        'Native compiler resources differ from the standalone probe, so component speeds do not predict full-model gains.','',
        f'Completed2B B1 cohort: {w["candidate"]:.2f}ms versus {w["control"]:.2f}ms, '
        f'across{register["equal_32_four_way_groups"]}equal32groups, all{register["same_sequence_cases"]}/28output sequences matching. '
        f'Retained ownership remains {owned[0]/1e9:.3f}GB. Saved LM Studio {w["lmstudio"]:.2f}ms and vLLM {w["vllm"]:.2f}ms have different or unresolved proposal settings. '
        'The2× target, independent quality, physical-memory and all-profile acceptance remain open; option stays default OFF.','']
flat=read('matched512-2b-b1-packed-p64-d4-mtp3-k8192-replay-flat-candidate.json')
if flat:
    w=flat['median_group_ms'];owned=flat['retained_owned_bytes'];assert len(owned['candidate'])==len(owned['control'])==1
    lines+=['## Flat packed-state concatenation','',
        'The optional variadic concatenation joins layer payloads once instead of retaining a chain of intermediate concatenations. '
        'CPU row-order controls include batch8 and129 parts; native parent, verifier and speculative controls preserve exact bytes. '
        'The real2B probe preserves canonical-state and last-hidden hashes.','',
        f'The flat-only cohort measures {w["candidate"]:.2f} ms versus {w["control"]:.2f} ms for chained packing, '
        f'with {flat["same_sequence_cases"]}/28 sequences matching. Retained ownership is {owned["candidate"][0]/1e9:.3f} GB, '
        f'a reduction of {owned["control"][0]-owned["candidate"][0]:,} bytes. This is retained allocation accounting, not physical peak VRAM.','']
lines+=['## Combined packed, resident, serial-reuse and register profiles','',
    'These experiments combine several options. Their differences from the original target can include chunk-size arithmetic changes; '
    'same-sequence counts compare the saved control indicated by each artifact. '
    'Independent quality remains open. The2B batch1 row uses the completed vLLM MTP3 comparator; other vLLM rows use saved serial cohorts. '
    'LM Studio runtime drafting is audited above. Each row uses its own common equal-output subset.','',
    '| Model | Batch | Equal32 groups | Garnet ms | LM Studio ms | vLLM ms | 2× target ms | Owned GB | Same sequences /28 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
combined_suffix='-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat'
for size in ('2b','4b'):
    for batch in (1,2,4,8):
        candidate=read(f'matched512-{size}-b{batch}{combined_suffix}-mtp3-cache128-candidate.json') or read(f'matched512-{size}-b{batch}{combined_suffix}-candidate.json')
        candidate=read(f'matched512-{size}-b{batch}{combined_suffix}-targetserial-aux0-mtp3-cache128-candidate.json') or read(f'matched512-{size}-b{batch}{combined_suffix}-targetserial-aux0-candidate.json') or candidate
        candidate=read(f'matched512-{size}-b{batch}{combined_suffix}-targetserial-aux0-deferred-mtp3-cache128-candidate.json') or read(f'matched512-{size}-b{batch}{combined_suffix}-targetserial-aux0-deferred-candidate.json') or candidate
        candidate=read(f'matched512-{size}-b{batch}{combined_suffix}-targetserial-aux0-deferred-nativecommit-mtp3-cache128-candidate.json') or read(f'matched512-{size}-b{batch}{combined_suffix}-targetserial-aux0-deferred-nativecommit-candidate.json') or candidate
        candidate=read(f'matched512-{size}-b{batch}{combined_suffix}-targetserial-aux0-deferred-nativecommit-arena-mtp3-cache128-candidate.json') or read(f'matched512-{size}-b{batch}{combined_suffix}-targetserial-aux0-deferred-nativecommit-arena-candidate.json') or candidate
        if not candidate:continue
        w=candidate['median_group_ms'];owned=candidate['retained_owned_bytes']['candidate']
        target=min(w['lmstudio'],w['vllm'])/2
        owned_text=f'{min(owned)/1e9:.3f}' if len(owned)==1 else f'{min(owned)/1e9:.3f}–{max(owned)/1e9:.3f}'
        lines.append(f'| {size.upper()} | {batch} | {candidate["equal_32_four_way_groups"]} | {w["candidate"]:.2f} | '
            f'{w["lmstudio"]:.2f} | {w["vllm"]:.2f} | {target:.2f} | {owned_text} | {candidate["same_sequence_cases"]}/28 |')
lines+=['','Missing rows have no completed cohort. No projected speed or memory values are included. '
    'Ownership ranges retain allocation growth instead of selecting the lowest observation. '
    'The2× goal remains unmet; these options remain default OFF.','']
mtp_compare=read(f'matched512-2b-b1{combined_suffix}-mtp3-cache128-candidate.json')
if mtp_compare:
    w=mtp_compare['median_group_ms'];target=min(w['vllm'],w['lmstudio'])/2
    lines+=['## New vLLM MTP comparator — completed2B batch1','',
        'The same pinned BF16 checkpoint, JPEGs, prompt IDs,512-token capacity and32-token limit now run with vLLM native MTP depth3. '
        'Cache budgets96MiB and112MiB per sequence both failed admission; their errors remain preserved. '
        '128MiB is the smallest successful tested budget, not a proven mathematical minimum. '
        'The four-request admission smoke is separate from the full28-request cohort.','',
        f'Across {mtp_compare["equal_32_four_way_groups"]} common equal32 groups, Garnet measures {w["candidate"]:.2f} ms, '
        f'vLLM MTP {w["vllm"]:.2f} ms and LM Studio {w["lmstudio"]:.2f} ms. '
        f'The current2× threshold is {target:.2f} ms. Garnet is slower than this MTP comparator. '
        'vLLM MTP matches17/28 saved vLLM serial sequences; independent quality remains open. '
        'FP32 convolution/recurrent caches remain requested. Runtime selects the Triton GDN speculative path and logs that '
        'the fused CUDA path requires BF16 convolution cache. A separately identified native-cache profile may be faster and is still pending. '
        'Exact proposal settings and intermediate arithmetic are not equivalent across engines.','',
        'Evidence: '+link('matched512-vllm-2b-b1-mtp3-cache128.json')+', '+link(f'matched512-2b-b1{combined_suffix}-mtp3-cache128-candidate.json')+'.','']
capture_controls=[read(f'packed-capture-lifetime-g{g}-s{s}-memcheck-exit.json') for g,s in ((0,0),(1,0),(1,1))]
if all(capture_controls):
    lines+=['## CUDA capture lifetime gate — unresolved','',
        'All18 target-serial native integration profiles preserve states, tokens, stops, limits, rollback and one-context ownership. '
        'However, the full batch8 memcheck reports use-before-allocation errors in a TensorRT Myelin fused kernel. '
        'A same-profile target-reuse-OFF control also fails. Target-only isolation with no MTP/replay passes when capture is OFF '
        'and fails when capture is ON, with and without target serial reuse. '
        'Target serial reuse remains default OFF; no latency or memory improvement has been measured for it.','',
        'Eight independent plain-CUDA allocation/capture controls pass memcheck with zero errors. '
        'Skipping unchanged context binding setters did not fix the TensorRT failure; that diagnostic is rejected as the fix. '
        'The failing10,633,216-byte buffer is absent from the synchronous runtime allocator trace. '
        'Overriding the distinct TensorRT temporary-storage allocator also did not fix it. '
        'No sanitizer error is suppressed or treated as a pass. The current capture gate prevents acceptance.','']
zero_aux=read('zero-aux-capture-control-comparison.json')
if zero_aux:
    assert len(zero_aux)==3 and all(v['exit']==0 and v['all_states_ids_equal_old_noncapture'] for v in zero_aux)
    lines+=['## Single-stream capture route — validation in progress','',
        'The default wide target graph uses two auxiliary streams; decode and final norm use zero. '
        'Its captured prefill graph has52 nodes and no allocation/free/child nodes, so the failing buffer was allocated outside that graph. '
        'The optional zero-auxiliary-stream build passes all three target-only memcheck controls: capture OFF, capture ON, '
        'and capture ON with serial target reuse. All final state hashes, tokens and slots match the older uncaptured control. '
        'Only serial target reuse bounds all three graph context counts to one. '
        'This is a measured workaround for the isolated route, not a universal TensorRT bug diagnosis or performance claim.','']
    gates=[read(f'target-serial-session-b8-p64-{kind}-aux0-exit.json') for kind in ('memcheck','racecheck','synccheck')]
    lines+=['Full batch-profile and MTP sanitizer gates: '+('all three completed with exit0; inspect retained logs for scope.' if all(gates) and all(v['exit_code']==0 for v in gates) else 'full MTP memcheck passed; full integration racecheck was interrupted and remains incomplete. Full integration synccheck was not run.'),'']
    component=[read(f'native-register128-program-b8-c4-{kind}-aux0-exit.json') for kind in ('racecheck','synccheck')]
    if all(component) and all(v['exit_code']==0 for v in component):
        lines+=['Exact native-DLL two-layer C4/B8 packed-versus-per-layer component checks pass racecheck and synccheck with zero findings and12 exact records each. '
            'The Garnet kernel-name filter excludes vendor kernels. This bounded component coverage does not replace the incomplete full24-case MTP race/sync checks.','']
gm=read('memory-combined-4b-b8-512-summary.json')
lm=read('memory-lmstudio-4b-b8-512-partitioned-summary.json')
if gm and lm:
    ge=read('memory-combined-4b-b8-512-session.json');le=read('memory-lmstudio-4b-b8-512-partitioned-session.json')
    gh=[e['input_sha256'] for e in ge if 'input_sha256' in e]
    lh=[e['result']['input_sha256'] for e in le if 'result' in e]
    assert gh==lh and len(gh)==6
    lines+=['## Matched4B batch8 physical memory diagnostic','',
        'Each engine ran alone with the same six JPEG groups,8 slots and512-token independent capacity. '
        'Garnet uses the combined MTP profile; the saved4B GGUF lacks MTP tensors. Proposal configuration differs. '
        'Loading is observed only for Garnet. Sampling and idle holds invalidate latency measurement; the primary cohorts above are separate.','',
        '| Engine | Warm dedicated GB | Warm shared GB | Largest observed dedicated GB | Retained owned GB |',
        '|---|---:|---:|---:|---:|']
    for label,data in (('Garnet combined',gm),('LM Studio',lm)):
        steady=data['phases']['steady_after']['bytes'];owned=data['owned_bytes_after_warmup']
        lines.append(f'| {label} | {steady["DedicatedUsage"]["median"]/1e9:.3f} | {steady["SharedUsage"]["median"]/1e9:.3f} | '
            f'{data["largest_observed_bytes"]["DedicatedUsage"]/1e9:.3f} | '+(f'{max(owned)/1e9:.3f}' if owned else 'unavailable')+' |')
    lines+=['','The short Garnet diagnostic kept one target-prefill context. The longer28-request timing cohort allocated a second context '
        'and reached16.480GB retained ownership, with several requests above6seconds. Both observations are retained; '
        'the short diagnostic does not describe the longer cohort. Shared usage also remained after model close, so its full amount '
        'cannot be attributed to live model spill. A causal paging diagnosis remains unproven. '
        'LM Studio unloaded its worker before an enclosed close-phase sample was available. '
        'Largest sampled usage is not a true lifetime peak. Memory acceptance remains open.','',
        'Evidence: '+link('memory-combined-4b-b8-512-summary.json')+', '+link('memory-lmstudio-4b-b8-512-partitioned-summary.json')+'.','']
new_route=read(f'matched512-4b-b8{combined_suffix}-targetserial-aux0-candidate.json')
deferred=read(f'matched512-4b-b8{combined_suffix}-targetserial-aux0-deferred-candidate.json')
if new_route:
    w=new_route['median_group_ms']
    lines+=['## Bounded target contexts with zero auxiliary streams — measured4B batch8','',
        f'Across{new_route["equal_32_four_way_groups"]}common equal32groups, the new route measures {w["candidate"]:.2f}ms versus '
        f'{w["control"]:.2f}ms for the older combined control; all{new_route["same_sequence_cases"]}/28 output sequences match. '
        'The longer timing cohort now retains one wide target context, removing945,818,112B of duplicate context allocation. '
        'This combines serial target reuse with zero auxiliary streams; the latency change does not isolate either option. '
        'The short old memory diagnostic already had one context, so it cannot establish the same physical-memory improvement.','']
if deferred:
    w=deferred['median_group_ms'];owned=deferred['retained_owned_bytes'];target=min(w['vllm'],w['lmstudio'])/2
    lines+=['## Deferred canonical state commit — measured4B batch8','',
        'The optional transaction borrows canonical state unchanged while target verification computes proposed outputs. '
        'Rollback restores small request buffers; accepted prefixes write canonical state only at the final commit. '
        'Pending transactions reject other parent mutation. A final-write failure poisons the decoder until reset. '
        'This removes the full canonical backup without changing the CUDA recurrence or precision. Default OFF.','',
        'All12 integration profiles exited0, including parent ownership, invalid-use, partial/full acceptance and injected failure recovery. '
        'The batch8 verifier and full24-case MTP session both passed unfiltered memcheck with zero errors. '
        'These controls do not resolve existing independent pretrained quality gaps.','',
        f'The28-request real-image cohort measures {w["candidate"]:.2f}ms versus {w["control"]:.2f}ms on '
        f'{deferred["equal_32_four_way_groups"]}common equal32groups, with all{deferred["same_sequence_cases"]}/28 sequences unchanged. '
        f'Retained ownership falls from{max(owned["control"]):,}B to{max(owned["candidate"]):,}B '
        f'(-{max(owned["control"])-max(owned["candidate"]):,}B). '
        f'vLLM measures{w["vllm"]:.2f}ms, LM Studio{w["lmstudio"]:.2f}ms; the2× threshold{target:.2f}ms remains unmet.','',
        'Evidence: '+link(f'matched512-4b-b8{combined_suffix}-targetserial-aux0-deferred-candidate.json')+', '
        +link('deferred-state-verifier-b8-memcheck-exit.json')+', '+link('deferred-state-session-b8-memcheck-exit.json')+'.','']
route_memory=[('Garnet zero-aux / serial target','memory-combined-4b-b8-512-targetserial-aux0-summary.json'),
              ('Garnet plus deferred commit','memory-combined-4b-b8-512-targetserial-aux0-deferred-summary.json'),
              ('Garnet plus native prefix commit','memory-combined-4b-b8-512-targetserial-aux0-deferred-nativecommit-summary.json'),
              ('Garnet plus shared output storage','memory-combined-4b-b8-512-targetserial-aux0-deferred-nativecommit-arena-summary.json'),
              ('LM Studio','memory-lmstudio-4b-b8-512-partitioned-summary.json')]
if read(route_memory[1][1]):
    lines+=['## Latest isolated4B batch8 memory counters','',
        'Sequential runs use the same six JPEG groups. Dedicated/shared counters are sampled WDDM process usage; '
        'they do not prove lifetime peaks or physical residency. Loading was not observed for LM Studio. MTP configurations differ.','',
        '| Profile | Warm dedicated GB | Warm shared GB | Largest observed dedicated GB | Owned GB |',
        '|---|---:|---:|---:|---:|']
    for label,name in route_memory:
        data=read(name)
        if not data:continue
        steady=data['phases']['steady_after']['bytes'];owned=data['owned_bytes_after_warmup']
        lines.append(f'| {label} | {steady["DedicatedUsage"]["median"]/1e9:.3f} | {steady["SharedUsage"]["median"]/1e9:.3f} | '
            f'{data["largest_observed_bytes"]["DedicatedUsage"]/1e9:.3f} | '+(f'{max(owned)/1e9:.3f}' if owned else 'unavailable')+' |')
    lines+=['','Evidence: '+', '.join(link(name) for _,name in route_memory if read(name))+'. Memory acceptance remains open.','']
native_gates=[read(f'native-prefix-session-b8-{kind}-exit.json') for kind in ('memcheck','racecheck','synccheck')]
if all(native_gates) and all(g['exit_code']==0 for g in native_gates):
    lines+=['## Native state-only prefix commit — experimental','',
        'The default-off nativePackedPrefixCommit route replaces the packed replay TensorRT graph with separate CUDA state-only kernels. '
        'It recomputes accepted convolution/recurrent state in place and copies only accepted KV slots from authoritative target outputs. '
        'The canonical state stays unchanged until this final commit; failed final writes poison the decoder. '
        'It requires the deferred-state and register128 profiles. Small convolution/dt/A_log parameters and convolution workspace are explicitly counted. '
        'TensorRT input/output aliasing is not used.','',
        'All12 numerical profiles pass exact ordinary-native states/outputs, including B1/2/4/8, C4/C16 and P32/P64. '
        'The B8 verifier and full24-case MTP session pass unfiltered memcheck. '
        'Racecheck/synccheck pass with zero findings for the three new packedCommit kernels inside the full MTP session; other kernels are excluded from these two checks. '
        'Invalid spans/layouts are rejected before mutation, and post-write injected failures require reset. '
        'These scopes do not replace missing full integration race/sync coverage or existing independent pretrained quality gaps.','']
    native_probe=read('native-prefix-pretrained-4b-b8.json')
    if native_probe:
        lines+=['The real4B B8 probe '+('passes' if all(native_probe[k] for k in ('same_recorded_sequences','same_ordinary_outputs','all_state_hashes_equal')) else 'FAILS')+
            ': two native-prefix requests followed by a separately loaded non-MTP target. '
            'Full finite target-state hashes stream bounded column chunks; timings are invalidated by diagnostic hashing. '
            'Evidence: '+link('native-prefix-pretrained-4b-b8.json')+'.','']
    native_cohort=read(f'matched512-4b-b8{combined_suffix}-targetserial-aux0-deferred-nativecommit-candidate.json')
    if native_cohort:
        w=native_cohort['median_group_ms'];owned=native_cohort['retained_owned_bytes']
        lines+=[f'The full28-request4B B8 cohort measures {w["candidate"]:.2f}ms versus {w["control"]:.2f}ms on '
            f'{native_cohort["equal_32_four_way_groups"]}common equal32groups, with{native_cohort["same_sequence_cases"]}/28 unchanged sequences. '
            f'Retained ownership is{max(owned["candidate"]):,}B versus{max(owned["control"]):,}B. '
            f'vLLM serial{w["vllm"]:.2f}ms and LM Studio{w["lmstudio"]:.2f}ms; target{min(w["vllm"],w["lmstudio"])/2:.2f}ms remains unmet.','']
arena_storage=read('output-arena-storage-fusion.json')
if arena_storage:
    lines+=['## Shared packed output storage — experimental','',
        'A prospective design records one maximum-sized allocation with stable exact-shape SDK storage views for sequential packed prefill/decode outputs. '
        'The generic binding API checks static output shape, dtype, device, contiguity, alignment and storage span before accepting it; input/output overlap and late rebinding are rejected. '
        'The decoder counts the shared pool once and excludes borrowed output views from each graph\'s owned-output bytes. '
        'The initial test harness failed before engine execution because it omitted the required fusion graph boundary; its source/archive/log/exit1 are retained. '
        'The corrected harness preserves the same exact-output and lifecycle assertions.','',
        'The generic corrected test passes two interleaved graph shapes, capture/replay, caller-reference drop, closure of other owners and invalid storage boundaries. '
        'Model validation and sanitizer results are listed only when their explicit exit records exist. Default OFF. '
        'Evidence: '+link('shared-output-arena-design.json')+', '+link('output-arena-storage-fusion.json')+'.','']
    gates=[read(f'output-arena-{mode}-b8-memcheck-exit.json') for mode in ('storage','verifier','session')]
    if all(gates) and all(g['exit_code']==0 for g in gates):
        lines+=['All12 packed verifier/MTP profiles pass exact states and outputs. Generic storage, B8 verifier and full24-case B8 MTP session pass unfiltered memcheck with zero errors. '
            'This covers memory access/lifetime, not full integration race/synccheck or independent quality acceptance.','']
    arena_probe=read('output-arena-pretrained-4b-b8.json')
    if arena_probe:
        lines+=['The real4B B8 shared-output probe '+('passes' if all(arena_probe[k] for k in ('same_recorded_sequences','same_ordinary_outputs','all_state_hashes_equal')) else 'FAILS')+
            ': two production MTP requests followed by a separately loaded ordinary target with independent outputs. '
            'All full finite canonical-state and last-hidden hashes match. '
            f'Probe retained ownership is{arena_probe["native_memory"]["owned_model_bytes"]:,}B; this is not process VRAM or cohort timing. '
            'Evidence: '+link('output-arena-pretrained-4b-b8.json')+'.','']
    cohort=read(f'matched512-4b-b8{combined_suffix}-targetserial-aux0-deferred-nativecommit-arena-candidate.json')
    if cohort:
        w=cohort['median_group_ms'];owned=cohort['retained_owned_bytes']
        lines+=[f'Completed4B B8: {w["candidate"]:.2f}ms versus {w["control"]:.2f}ms on{cohort["equal_32_four_way_groups"]}common equal32groups; '
            f'{cohort["same_sequence_cases"]}/28 unchanged sequences. Owned{max(owned["candidate"]):,}B '
            f'versus{max(owned["control"]):,}B, a measured reduction of{max(owned["control"])-max(owned["candidate"]):,}B. '
            'Retained allocation accounting is separate from the isolated physical-counter table. The2× goal remains unmet.','']
whole_validation=read('whole-program-session-b8-p64-d4-exit.json')
if whole_validation:
    lines+=['## Scoped whole-program CUDA state-kernel compilation','',
        'The resource audit isolates relocatable device code as a register-count difference: the same native and volatile-template source bodies use168 registers without RDC and255 with it. '
        'Those object counts precede final device linking and do not establish model speed. '
        'The initial audit mirrored compiler FLAGS but omitted the hard-coded CMake rule addition; the second run adds only-rdc=true and records the correction.','',
        'The default-off GARNET_QWEN35_WHOLE_PROGRAM_CUDA option moves only the self-contained delta and packed-prefix CUDA units to a non-separable object target. '
        'Other CUDA sources retain their existing linking mode. No algorithm, precision, fast-math, projection or layout change is requested. '
        'The actual DLL reports168 registers/zero stack for target and prefix recurrence; a marker compiled in the CUDA unit identifies the real policy. '
        'Benchmark/probe labels must match that marker. All12 original verifier/MTP profiles pass exact state/output/lifecycle controls.','',
        'Evidence: '+link('qwen35-whole-program-design.json')+', '+link('register-template-audit-comparison.json')+', '+link('whole-program-dll-resources.txt')+'.','']
    lines+=['| Model / batch | Groups | Previous ms | Whole-program ms | Same sequences | Owned GB |',
        '|---|---:|---:|---:|---:|---:|']
    for size,batch in (('2b',1),('4b',8)):
        suffix='-mtp3-cache128' if size=='2b' else ''
        d=read(f'matched512-{size}-b{batch}{combined_suffix}-targetserial-aux0-deferred-nativecommit-arena-wholeprogram{suffix}-candidate.json')
        if not d:continue
        w=d['median_group_ms'];owned=d['retained_owned_bytes']['candidate'];assert d['candidate_whole_program_cuda'] is True
        lines.append(f'| {size.upper()} / {batch} | {d["equal_32_four_way_groups"]} | {w["control"]:.2f} | {w["candidate"]:.2f} | {d["same_sequence_cases"]}/28 | {max(owned)/1e9:.3f} |')
    lines+=['','Missing rows are pending. Independent quality, other batch profiles and 2× acceptance remain open. '
        'The 2B batch-1 result is a latency regression despite the lower register count. '
        'The option remains experimental and default off; this table reports measured candidates, not accepted improvements.','']
    for size,batch in (('2b',1),('4b',8)):
        name=f'whole-program-pretrained-{size}-b{batch}-cross-rdc-state-control.json';d=read(name)
        if d:
            lines+=[f'{size.upper()} B{batch} cross-build pretrained finite-state/output preservation: '+('PASS' if d['all_passed'] else 'FAIL')+
                '. Identical real inputs/profile; full ordinary and repeated speculative state hashes/signatures compared. Evidence: '+link(name)+'.','']
    exits=[read(f'whole-program-{mode}-b8-memcheck-exit.json') for mode in ('storage','verifier','session')]
    if all(exits) and all(d['exit_code']==0 for d in exits):
        lines+=['Generic storage, B8 verifier and full24-case MTP unfiltered memchecks pass with zero errors. '
            'Exact-DLL C4/B8 two-layer program/control race/sync checks pass with zero findings. '
            'New prefix-kernel race/sync checks also pass in the full MTP session. Their kernel-name filters limit scope; full integration race/sync remains incomplete.','']
finite_check=read('finite-f32-bytes-validation.json')
finite_memcheck=read('finite-f32-bytes-memcheck-exit.json')
finite_device=read('finite-f32-bytes-device-preservation.json')
if finite_check and finite_memcheck and finite_device:
    lines+=['## Bounded finite-state audit transfer','',
        'A host diagnostic exports at most 1,048,576 contiguous FP32 words (4 MiB) per call, '
        'rejects every NaN/Inf and returns exact little-endian bytes for SHA-256. '
        'The original column chunk order and full target-state coverage remain unchanged. '
        'The inference path does not call this helper, so this is not an inference speed improvement.','',
        f'Exact finite-byte, signed-zero, subnormal, maximum-finite, legacy SHA, rejection, '
        f'logical-view and lifetime checks: {len(finite_check["checks"])} passed. '
        f'Unfiltered memcheck exit: {finite_memcheck["exit_code"]}. '
        f'CUDA device-fatbin equality with the prior build: {finite_device["device_fatbin_equal"]}. '
        'Device equality does not by itself establish host correctness.','',
        'Evidence: '+link('finite-f32-bytes-validation.json')+', '+link('finite-f32-bytes-memcheck-exit.json')+', '+link('finite-f32-bytes-device-preservation.json')+'.','']
    interrupted=read('whole-program-pretrained-4b-b8-exit.json')
    if interrupted and interrupted.get('status')=='INTERRUPTED_INCOMPLETE':
        lines+=['The earlier 4B batch-8 scalar-conversion probe was interrupted while CPU-active after over 13 minutes. '
            'Its logs and incomplete exit are retained. No success, kernel failure or timing result follows from that interruption. '
            'Redirected stdout can be buffered, so the absence of a printed phase marker does not localize the delay.','']
    for size,batch in (('2b',1),('4b',8)):
        name=f'whole-program-fastbytes-pretrained-{size}-b{batch}-cross-rdc-state-control.json';d=read(name)
        if d:
            lines+=[f'{size.upper()} B{batch} bounded-byte real probe versus saved RDC full-state/output hashes: '
                +('PASS' if d['all_passed'] else 'FAIL')+'. Evidence: '+link(name)+'.','']
    d=read('finite-f32-bytes-2b1-cross-method-control.json')
    if d:
        lines+=['2B B1 bounded-byte versus saved same-policy legacy full-state/output hashes: '
            +('PASS' if d['all_passed'] else 'FAIL')+'. Evidence: '+link('finite-f32-bytes-2b1-cross-method-control.json')+'.','']
all_name='matched512-4b-b8-whole-program-all-baselines.json'
all_baselines=read(all_name)
if not all_baselines:
    all_name='matched512-4b-b8-native-arena-all-baselines.json';all_baselines=read(all_name)
if all_baselines:
    lines+=['## All tested 4B batch-8 comparators on one subset','',
        f'Every following value uses the same {all_baselines["common_equal32_groups"]} equal-32 groups, including serial and MTP vLLM. '
        'This is distinct from the 15-group four-way cohort table.','',
        '| Profile | Median group ms |','|---|---:|']
    for name,value in all_baselines['median_group_ms'].items():lines.append(f'| {name} | {value:.2f} |')
    lines+=['',f'The fastest tested baseline is {all_baselines["fastest_baseline"]}; the half-latency target of {all_baselines["half_fastest_baseline_ms"]:.2f} ms remains unmet. '
        'Evidence: '+link(all_name)+'. The source manifest identifies the exact candidate and control; '
        'the whole-program candidate remains experimental, with a 2B batch-1 regression and an unrepeated small 4B batch-8 difference.','']
v4mtp=read(f'matched512-4b-b8{combined_suffix}-targetserial-aux0-deferred-mtp3-cache128-convbfloat16-candidate.json')
if v4mtp:
    w=v4mtp['median_group_ms']
    lines+=['## Additional vLLM4B batch8 MTP baseline','',
        'A separate28-request vLLM MTP3 profile uses BF16 convolution cache and FP32 recurrent state, with128MiB cache per sequence and512-token capacity. '
        'Its runtime log explicitly selects the CUDA GDN speculative kernel. '
        f'On{v4mtp["equal_32_four_way_groups"]}common equal32groups it measures{w["vllm"]:.2f}ms versus '
        f'{w["candidate"]:.2f}ms for the deferred Garnet profile. '
        'The saved vLLM serial cohort is faster, so this slower MTP result does not replace it as the goal baseline. '
        'Proposal algorithms and intermediate arithmetic differ, and independent quality remains open.','']
matrix=read('cublaslt-profile-grid-summary.json')
if matrix:
    lines+=['## Isolated projection-layout screen','',
        f'{len(matrix["profiles"])} matrix shapes cover decode rows 4/8/16/32/64 and prefill rows 64/128/256/512 '
        'for the existing 2B/4B gate/up/down dimensions. Historical quick/full grids remain unchanged. '
        'Candidates retain BF16 inputs/output, FP32 accumulation, eligible compute-type reductions and the original 128 FP64 sample bounds per result.','',
        '| Screened candidate class | Median component ratio | Maximum component ratio |',
        '|---|---:|---:|']
    for key,label in [('best_same_layout_zero_workspace','Current layout, zero workspace'),
                      ('best_same_layout','Current layout, bounded workspace'),('best_any_layout','Allow transposed layout')]:
        ratios=[r[key]['sweep_speed_ratio'] for r in matrix['profiles']]
        lines.append(f'| {label} | {statistics.median(ratios):.3f} | {max(ratios):.3f} |')
    lines+=['','Ratios summarize heterogeneous isolated matrices and are not weighted model-speed estimates. '
        'The 256 MiB sweep does not establish L2 residency. Candidates were selected from multiple algorithms in one screen, '
        'so repeats and full-model numerical/lifecycle validation would be required before integration. '
        'Algorithm IDs alone do not describe complete heuristic configurations. No projection change was promoted. '
        'Evidence: '+link('cublaslt-profile-grid-summary.json')+'.','']
wide_exits=[read(f'wide8-session-b{b}-p{p}-d8-mtp7-exit.json') for b in (1,2,4,8) for p in (32,64)]
wide_exits += [read(f'wide8-verifier-b{b}-c{c}-exit.json') for b,c in ((8,8),(4,8),(8,16),(4,16))]
if all(wide_exits) and all(e['exit_code']==0 for e in wide_exits):
    lines+=['## Wider D8/MTP7 experiment','',
        'Four verifier profiles (B4/B8, C8/C16, verification width 8) and eight speculative-session profiles '
        '(B1/B2/B4/B8, P32/P64, D8/MTP7) pass exact state/output/lifecycle checks. '
        'Saved graph markers confirm the restored RDC build. These synthetic checks do not establish pretrained quality or speed.','']
    for size,batch in (('2b',1),('4b',8)):
        name=f'wide8-pretrained-{size}-b{batch}.json';d=read(name)
        if d:
            lines+=[f'{size.upper()} B{batch} real probe: ordinary-output match {d["same_ordinary_outputs"]}, '
                f'full finite-state hashes match {d["all_state_hashes_equal"]}, '
                f'saved D4 sequences match {d["same_saved_d4_sequences"]}. Evidence: '+link(name)+'.','']
        suffix='-mtp3-cache128' if size=='2b' else ''
        name=f'matched512-{size}-b{batch}-packed-p64-d8-mtp7-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena{suffix}-candidate.json'
        d=read(name)
        if d:
            w=d['median_group_ms']
            lines+=[f'{size.upper()} B{batch}, {d["equal_32_four_way_groups"]} equal-32 groups: '
                f'D8/MTP7 {w["candidate"]:.2f} ms versus D4/MTP3 {w["control"]:.2f} ms; '
                f'{d["same_sequence_cases"]}/28 same sequences. Independent quality and 2× acceptance remain open. '
                'Evidence: '+link(name)+'.','']
conditional=[read(f'conditional-control-api13-{kind}-exit.json') for kind in ('plain','memcheck','racecheck','synccheck')]
if all(conditional) and all(e['exit_code']==0 for e in conditional):
    lines+=['## Conditional GPU-control feasibility probe','',
        'An isolated WHILE graph containing a captured child graph passes 84 bounded-counter cases on SM89, '
        'including zero/short/32 limits, mixed rows, invalid limits and repeated reset. '
        'The source capture is destroyed before parent instantiation and launch. '
        'Normal execution and unfiltered memcheck/racecheck/synccheck pass with zero findings. '
        'The initial compile failure used an older API signature; its log and source remain retained.','',
        'This is a control/lifetime prerequisite, with no model weights, speculative acceptance algorithm or inference-speed claim. '
        'A future resident serving loop would need stable child graphs and resource ownership outside the conditional body. '
        'NVIDIA documents restricted body-node types in its '
        '[CUDA 13.2 graph guide](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/04-special-topics/cuda-graphs.html#conditional-node-body-graph-requirements). '
        'Evidence: '+link('conditional-control-api13-identity.json')+', '+link('conditional-control-design.json')+'.','']
static_name='matched512-2b-b1-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-staticbindings-mtp3-cache128-candidate.json'
static=read(static_name)
if static:
    w=static['median_group_ms']
    lines+=['## TensorRT binding-reuse diagnostic','',
        f'2B B1, {static["equal_32_four_way_groups"]} common equal-32 groups: '
        f'{w["candidate"]:.2f} ms versus historical control {w["control"]:.2f} ms and vLLM MTP {w["vllm"]:.2f} ms. '
        f'{static["same_sequence_cases"]}/28 same saved sequences; retained ownership unchanged. '
        'All 12 C4 numerical profiles, three unfiltered memchecks and a real full finite-state/output probe pass. '
        'The small timing difference is unrepeated; the option remains unselected/default off. '
        'This does not establish the 2× target. Evidence: '+link(static_name)+'.','']
packing=read('prompt-pack-native-plain.json');serving=read('prompt-pack-serving-fixed-plain.json')
if packing and serving:
    lines+=['## Copy-only prompt packing — validation in progress','',
        'A default-off packed-decoder option assembles one [time,batch,hidden] FP32 request buffer with pitched copies '
        'and exposes contiguous token views. Batched vision output is snapshotted before encoding the next row because the vision graph reuses its buffer. '
        'No model arithmetic, weights, precision, prompt cache or image cache changes.','',
        f'The native helper passes {len(packing["checks"])} exact-byte/legacy-copy/invalid-input/lifetime checks. '
        f'The production prompt-preparation path passes {len(serving["cases"])} cases using deliberately reused fake vision output with distinct images and ragged text. '
        'Both component suites pass normal execution plus unfiltered memcheck, racecheck and synccheck with zero findings. '
        'These are component results; broad pretrained quality and full-model race coverage are not established.','',
        'Source review caught a missing vision snapshot in the first serving integration before any pretrained timing cohort. '
        'The owned chain was interrupted, partial numerical logs retained, and the corrected chain uses separate evidence names. '
        'The CUDA device fatbin is byte-identical to the preceding RDC binary; host correctness is checked separately. '
        'Evidence: '+link('prompt-pack-device-preservation.json')+', '+link('prompt-pack-first-chain-interrupted-exit.json')+', '
        +link('prompt-pack-native-plain.json')+', '+link('prompt-pack-serving-fixed-plain.json')+'.','']
    for size,batch in (('2b',1),('4b',8)):
        suffix='-mtp3-cache128' if size=='2b' else ''
        name=f'matched512-{size}-b{batch}-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-promptpack{suffix}-candidate.json'
        d=read(name)
        if d:
            w=d['median_group_ms']
            lines+=[f'{size.upper()} B{batch}, {d["equal_32_four_way_groups"]} common equal-32 groups: '
                f'packed inputs {w["candidate"]:.2f} ms versus saved control {w["control"]:.2f} ms; '
                f'{d["same_sequence_cases"]}/28 same saved sequences. Requires paired repetition before promotion. '
                'Evidence: '+link(name)+'.','']
memory_rows=[]
for size,batch in (('2b',1),('4b',8)):
    name=f'memory-combined-{size}-b{batch}-512-targetserial-aux0-deferred-nativecommit-arena-promptpack-summary.json'
    d=read(name)
    if d:
        values=d['phases']['steady_after']['bytes']
        memory_rows.append((size,batch,values,name))
if memory_rows:
    lines+=['## Packed-input isolated Windows memory','',
        '| Model | Batch | Warm dedicated GB | Warm shared GB |','|---|---:|---:|---:|']
    for size,batch,values,name in memory_rows:
        lines.append(f'| {size.upper()} | {batch} | {values["DedicatedUsage"]["median"]/1e9:.6f} | {values["SharedUsage"]["median"]/1e9:.6f} |')
    lines+=['','Differences from the preceding memory profile are only 2 MiB for 2B B1 and 6 MiB for 4B B8; '
        'no meaningful memory reduction is claimed. Tracked Garnet allocations remain unchanged. '
        'All six image-hash groups match the saved reference and LM Studio 4B B8 memory run. '
        'These are isolated, sampled process GPU counters, separate from timing and not proof of true peak or physical residency. '
        'Evidence: '+', '.join(link(name) for _,_,_,name in memory_rows)+', '+link('prompt-pack-isolated-memory-input-control.json')+'.','']
trace=read('profile-packed-current-2b-b1-p64-d4-graph-summary.json')
if trace:
    g=trace['groups']
    lines+=['## Current packed decoder whole-graph trace','',
        f'One instrumented 2B B1 request records {g["GRAPH_TRACE"]["count"]} graph launches, '
        f'{g["MEMCPY"]["count"]} copies and {g["MEMSET"]["count"]} memsets. '
        f'Whole-graph plus uncaptured GPU activity union is {trace["gpu_activity_union_ms"]:.3f} ms '
        f'within a {trace["gpu_activity_span_ms"]:.3f} ms span; between-activity gaps total {trace["between_activity_idle_ms"]:.3f} ms. '
        'Whole-graph tracing hides internal gaps and node costs. API/idle overlap is not causation or recoverable time. '
        'The same saved input/output and request counters had 176.22 ms historical cohort latency; '
        'the instrumented Model request is 244.35 ms. This is not an interleaved profiler-overhead control or a speed forecast. '
        'Evidence: '+link('profile-packed-current-2b-b1-p64-d4-graph-summary.json')+'.','']
view=read('logical-views-fixed-plain.json');before=read('logical-views-before-fix-exit.json')
if view and before:
    lines+=['## Logical SDK-view I/O correction','',
        'The original native implementation fails the new byte-boundary oracle: zeroing a middle token view also zeroes the following token. '
        'The original failure is retained. Zero, clone and copy now use validated contiguous logical element bytes, '
        'while preserving backing-storage owners, grouped access leases and partial-overlap rejection. '
        f'All {len(view["cases"])} boundary/lifetime cases pass normal execution and unfiltered memcheck, racecheck and synccheck. '
        'CUDA device fatbin bytes are identical to the preceding DLL; host behavior changed and is checked separately. '
        'The packing inference path reads its token views and had already passed full-state preservation; '
        'this correction protects general writable-view operations. '
        'Evidence: '+link('logical-views-before-fix-exit.json')+', '+link('logical-views-fixed-plain.json')+', '+link('logical-views-device-preservation.json')+'.','']
    for size,batch in (('2b',1),('4b',8)):
        name=f'logical-views-pretrained-{size}-b{batch}-promptpack-cross-prior-state-control.json';d=read(name)
        if d and d['all_passed']:lines+=[f'{size.upper()} B{batch} complete finite target-state hashes and outputs match the preceding packed-input build: '+link(name)+'.','']
if read('reuse-verified-head-design.json'):
    lines+=['## Reusing verified target-head work','',
        'Current source inspection corrects an earlier working-note assumption: partial packed-prefix commit calls `_read()` '
        'and repeats full target projection/argmax. The traced case has one initial head, eleven verification heads and six repeated partial-commit heads; '
        'seventeen M4 CUTLASS launches plus one M1 GEMV agree with that count. '
        'The default-off change preserves masked normalization for MTP correction and reuses authoritative accepted-token predictions. '
        'Zero-prefix fallback, accepted-row candidate equality, rollback/fault behavior, state preservation and paired timing remain required. '
        'The option has no performance acceptance. Design: '+link('reuse-verified-head-design.json')+'.','']
    exits=list(root.glob('reuse-head-verifier-b*-c*-exit.json'))+list(root.glob('reuse-head-session-b*-p*-d4-exit.json'))
    passed=sum(json.loads(path.read_text())['exit_code']==0 for path in exits)
    if exits:
        lines+=[f'Default-off implementation: {passed}/12 numerical profile exits currently pass. '
            'Verifier tests compare accepted candidate IDs with a fresh full head using the same packed shape, '
            'check evaluation/reuse counters, preserve zero-prefix fallback and retain original rollback/fault assertions. '
            'Session tests retain all 24 MTP/ordinary state and stop cases per profile. '
            'These preservation tests do not establish independent pretrained quality.','']
    for size,batch in (('2b',1),('4b',8)):
        state_name=f'reuse-head-pretrained-{size}-b{batch}-promptpack-cross-prior-state-control.json'
        state=read(state_name)
        if state and state['all_passed']:
            lines+=[f'{size.upper()} B{batch} complete finite target-state hashes and output signatures match the preceding head-off build; '
                'the probe also compares a separately ordinary target. Evidence: '+link(state_name)+'.','']
        suffix='-mtp3-cache128' if size=='2b' else ''
        name=f'matched512-{size}-b{batch}-packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-promptpack-headreuse{suffix}-candidate.json'
        trial=read(name)
        if trial:
            w=trial['median_group_ms']
            lines+=[f'{size.upper()} B{batch}, {trial["equal_32_four_way_groups"]} common equal-32 groups: '
                f'head reuse {w["candidate"]:.2f} ms versus saved packing-only {w["control"]:.2f} ms; '
                f'{trial["same_sequence_cases"]}/28 same saved sequences. Kept GPU allocations unchanged. '
                'This single trial requires same-build repetition; it is not a 2× or independent-quality acceptance. Evidence: '+link(name)+'.','']
        study_name=f'packed-study-{size}-b{batch}-promptpack-reusehead-summary.json'
        study=read(study_name)
        if study:
            w=study['aggregate_group_median_ms']
            lines+=[f'Same-build {size.upper()} B{batch} ABCCBA repeat: A {w["a"]:.2f}, B {w["b"]:.2f}, C {w["c"]:.2f} ms. '
                'A is native-prefix/shared-arena D4, B adds prompt packing, C adds verified-head reuse. '
                'All 20 warm groups are retained, all 28 input/prompt/output/stop signatures match across six fresh processes. '
                'Each aggregate first takes the median of two repeats for each group, then the median across groups. '
                'Two repeats do not establish significance or causality; external competitors were not rerun in this study. '
                'Derived decode throughput and phase medians are recorded with their scopes in '+link(study_name)+'.','']
        comparison_name=f'packed-study-{size}-b{batch}-promptpack-reusehead-all-baselines.json'
        comparison=read(comparison_name)
        if comparison:
            w=comparison['median_group_ms'];fastest=comparison['fastest_tested_baseline']
            lines+=[f'Repeated {size.upper()} B{batch} versus all listed historical competitors on one '
                f'{comparison["common_equal32_groups"]}-group equal-32 subset: Garnet C {w["garnet_c"]:.2f} ms, '
                f'fastest tested baseline {fastest} {w[fastest]:.2f} ms, '
                f'2× threshold {comparison["half_fastest_baseline_ms"]:.2f} ms. '
                'Garnet uses per-group repeat medians; external cohorts were not rerun. '
                'API/arithmetic/speculative differences and independent quality remain unresolved. '
                'Evidence: '+link(comparison_name)+'.','']
host_rows=[]
for size,batch in (('2b',1),('4b',8)):
    name=f'host-time-{size}-b{batch}-p64-d4-promptpack-headreuse-summary.json';d=read(name)
    if d:
        assert d['same_all28_input_prompt_output_stop_profile_signatures']
        host_rows.append((size,batch,name,d))
if host_rows:
    lines+=['## Coarse host-clock diagnostic','',
        'Separate instrumented cohorts preserve all 28 input/prompt/output/stop/profile signatures for each tested model. '
        'Calling-thread CPU charges were observed in 15.625 ms steps; zero medians for short categories do not prove no CPU work. '
        'Thread CPU includes native driver polling and excludes worker threads. Categories overlap and must not be summed. '
        'The target-read wall interval waits for previously queued GPU work, so it is not standalone head cost. '
        'These clocks do not establish a CPU bottleneck, recoverable time, or speedup.','']
    for size,batch,name,d in host_rows:
        lines+=[f'{size.upper()} B{batch}: instrumented Model wall median {d["median_request_wall_ms"]:.2f} ms; '
            f'fresh target-read inclusive wall median {d["inclusive_categories_do_not_sum"]["target_read_fresh"]["median_inclusive_wall_ms"]:.2f} ms. '
            'Instrumented timings are separate from primary performance evidence. Evidence: '+link(name)+'.','']
    lines+=['Observed clock quantization: '+link('host-time-2b-b1-clock-quantization.json')+'.','']
families=read('packed-node-kernel-family-summary.json')
if families:
    lines+=['## Latest matched-request node traces','',
        'Both full 28-request diagnostic cohorts preserve input/prompt/output/stop/profile/speculation counters against saved study C1. '
        'Only warmed request index 8 is traced; its queued tail is synchronized before stopping capture. '
        'GPU kernel names identify BF16 GEMM families, including multiple body/vision/head operators. '
        'This is a name-based aggregate, not attribution to one operator, continuous busy-time proof, or recoverable latency.','']
    for size,batch in (('2b',1),('4b',8)):
        d=families[size]
        name=f'profile-packed-{size}-b{batch}-p64-d4-promptpack-headreuse-node-preservation.json'
        lines+=[f'{size.upper()} B{batch}: named BF16 GEMM families sum {d["named_bf16_gemm_sum_ms"]:.2f} ms '
            f'({d["named_bf16_gemm_fraction"]*100:.1f}% of summed kernel time); register-state kernels '
            f'{d["register_state128_sum_ms"]:.2f} ms. Instrumented timings are not primary performance evidence. '
            'Preservation: '+link(name)+'.','']
lm_memory=read('memory-lmstudio-2b-b1-512-partitioned-summary.json')
if lm_memory and read('matched-memory-2b1-packed-lm-input-control.json'):
    lm=lm_memory['phases']['steady_after']['bytes']
    gm=read('memory-combined-2b-b1-512-targetserial-aux0-deferred-nativecommit-arena-promptpack-summary.json')['phases']['steady_after']['bytes']
    lines+=['## Matched 2B B1 isolated memory coverage','',
        f'Garnet packed-input/head-off warm Windows dedicated+shared counters: {gm["TotalCommitted"]["median"]/1e9:.6f} GB; '
        f'LM Studio partitioned 512-slot run: {lm["TotalCommitted"]["median"]/1e9:.6f} GB. '
        'All six image-hash batches match. These separate sampled memory diagnostics are not true peak/residency measurements '
        'or a fresh physical-memory measurement of head reuse. Evidence: '+link('matched-memory-2b1-packed-lm-input-control.json')+'.','']
joined=read('joined-projection-screen-summary.json')
if joined:
    lines+=['## Same-input joined projection component screen','',
        'A standalone component screen compares four separate qkv/z/beta/a GEMMs with one joined BF16 matrix, '
        'using the unchanged production BF16 I/O / FP32-accumulation implementation. '
        'All 16 2B/4B decode/prefill matrix cases pass the existing fixed FP64 error bound, input preservation, output guards '
        'and exact output-split checks. Unfiltered memcheck reports zero errors and leaked bytes. '
        'All eight decode fixtures are BF16-identical; 4B M64 prefill changes 217 BF16 values, retained explicitly.','',
        'Including output splits, eviction-sweep component speed ratios for M4/8/16/32 are about 1.44–1.46× (2B) and '
        '1.16–1.26× (4B). Nine rotated graph repetitions and a 256 MiB eviction sweep are synthetic evidence, '
        'not verified L2 residency or a model speed prediction. The production candidate is default-off decode-only, '
        'with shared original/joined weight ownership; it has not been accepted as the default. '
        'Evidence: '+link('joined-projection-screen-summary.json')+', '+link('joined-decode-projection-production-design.json')+'.','']
joined_identity=read('joined-decode-before-sanitizers-identity.json')
if joined_identity:
    sha=joined_identity['garnet_binary_sha256']
    lines+=['## Joined decode production candidate — default off','',
        'The packed/shared decodeChunk4 path groups qkv/z/beta/a checkpoint matrices into one BF16 allocation, '
        'retains read-only original-member views for prefill, and uses one projection plus output slices during decode. '
        'Ownership counts the group once. The native CUDA device fatbin is unchanged from the pushed checkpoint; '
        'there is no arithmetic-precision change. DLL SHA256: `'+sha+'`.','',
        'The first integrated fixture run failed on an undefined Python layer-kind variable. Its failure is retained; '
        'the corrected source was archived before rerunning the gates. Grouped-weight byte preservation, cache/accounting, '
        'read-only alias/lifetime and external-weight graph checks passed plain and unfiltered memcheck. '
        'Full decoder state equivalence passed B1/2/4/8 plain and unfiltered memcheck.','',
        '| Numerical suite | Completed passing profiles | Required profiles |','|---|---:|---:|']
    for prefix in ('joined-regression','joined-decode'):
        paths=list(root.glob(prefix+'-verifier-b*-c*-exit.json'))+list(root.glob(prefix+'-session-b*-p*-d4-exit.json'))
        passes=sum((d:=json.loads(path.read_text()))['exit_code']==0 and d['garnet_dll_sha256']==sha for path in paths)
        lines.append(f'| {prefix} | {passes} | 12 |')
    lines+=['','| Joined sanitizer gate | Saved result | Scope |','|---|---|---|']
    for tag in ('verifier-b8-memcheck','session-b8-memcheck','session-b8-racecheck','session-b8-synccheck'):
        name='joined-decode-'+tag+'-exit.json';d=read(name)
        status='pending' if d is None else 'pass' if d['exit_code']==0 and d['garnet_dll_sha256']==sha else 'failed or binary mismatch'
        scope='unfiltered' if tag.endswith('memcheck') else 'packedCommit kernels only; not full integration'
        lines.append(f'| {tag} | {status} | {scope} |')
    lines+=['','Evidence: '+link('joined-decode-before-sanitizers-identity.json')+', '
        +link('joined-decode-device-preservation.json')+'. Real-weight state preservation, full-request gain, '
        'fresh isolated memory and repeated timings require separate evidence; fixture checks do not establish them.','']
    real_rows=[];real_notes=[]
    for size,batch,extra in (('2b',1,'-mtp3-cache128'),('4b',8,'')):
        tag='packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-promptpack-headreuse-joineddecode'
        name=f'matched512-{size}-b{batch}-{tag}{extra}-candidate.json';d=read(name)
        state=read(f'joined-decode-pretrained-{size}-b{batch}-promptpack-headreuse-cross-prior-state-control.json')
        if d and state and state['all_passed']:
            w=d['median_group_ms'];owned=d['retained_owned_bytes']['candidate']
            real_rows.append(f'| {size} B{batch} | {d["equal_32_four_way_groups"]} | {fmt(w["control"])} | {fmt(w["candidate"])} | {fmt(w["vllm"])} | {fmt(w["lmstudio"])} | {d["same_sequence_cases"]}/28 | {owned} |')
            real_notes+=['',f'{size} B{batch} real probe: complete finite target states and outputs/stop signatures match the prior control. '
                'Single-cohort timing uses historical controls/competitors; repeated same-build evidence remains separate. '
                'Retained allocation is not sampled process memory or a true peak. Evidence: '+link(name)+'.','']
        study_name=f'packed-study-{size}-b{batch}-joined-decode-summary.json';study=read(study_name)
        if study:
            w=study['aggregate_group_median_ms']
            real_notes+=['',f'Joined ABBA {size} B{batch}: control **{w["a"]:.3f} ms**, candidate **{w["b"]:.3f} ms**, '
                f'{100*study["comparisons"]["a_to_b"]["reduction_fraction"]:.2f}% latency reduction over all20 warm groups. '
                'All28 input/prompt/output/stop signatures match across four fresh sequential processes on the same DLL. '
                'Two repeats do not establish significance, causality or independent quality. Evidence: '+link(study_name)+'.','']
        workload_name=f'joined-decode-{size}-b{batch}-all28-workload-control.json';workload=read(workload_name)
        if workload and workload['all_passed']:
            real_notes+=['',f'{size} B{batch}: all28 first-trial inputs, prompt IDs, outputs/stops, discrete MTP calls, '
                'accepted-draft counts and iteration progress are unchanged; retained allocation is unchanged. '
                'This rules out a changed speculative workload in that cohort, without proving independent quality. '
                'Evidence: '+link(workload_name)+'.','']
        baseline_name=f'packed-study-{size}-b{batch}-joined-decode-all-baselines.json';baseline=read(baseline_name)
        if baseline:
            w=baseline['median_group_ms'];fastest=baseline['fastest_tested_baseline']
            real_notes+=['',f'{size} B{batch}, common {baseline["common_equal32_groups"]} equal32 groups across all listed engines: '
                f'joined Garnet **{w["garnet_b"]:.3f} ms**, fastest saved competitor ({fastest}) **{w[fastest]:.3f} ms**, '
                f'2x latency threshold **{baseline["half_fastest_baseline_ms"]:.3f} ms**. '
                'Competitors are historical single cohorts; API/arithmetic/proposal differences remain. Evidence: '+link(baseline_name)+'.','']
        memory_tag=f'memory-combined-{size}-b{batch}-512-targetserial-aux0-deferred-nativecommit-arena-promptpack-headreuse-joineddecode'
        memory=read(memory_tag+'-summary.json');memory_control=read(memory_tag+'-workload-control.json')
        if memory and memory_control and memory_control['all_six_input_hash_groups_exact'] and memory_control['all_six_garnet_output_stop_signatures_match_timing']:
            values=memory['phases']['steady_after']['bytes']
            real_notes+=['',f'{size} B{batch} joined isolated memory, warm Windows process-counter medians: dedicated '
                f'**{values["DedicatedUsage"]["median"]/1e9:.6f} GB**, shared **{values["SharedUsage"]["median"]/1e9:.6f} GB**, '
                f'total **{values["TotalCommitted"]["median"]/1e9:.6f} GB**. '
                'All six memory inputs match the LM memory run and Garnet timing cohort; Garnet output/stop signatures and profiles match. '
                'These sampled committed counters do not establish true peak memory or physical residency. Evidence: '+link(memory_tag+'-summary.json')+', '+link(memory_tag+'-workload-control.json')+'.','']
        context=read(memory_tag+'-context-requirements.json')
        if context:
            real_notes+=['',f'{size} B{batch}: summed TensorRT context upper bounds '
                f'**{context["sum_context_upper_bounds_bytes_not_measured_allocation"]/1e9:.6f} GB**, largest single-context bound '
                f'**{context["largest_single_context_upper_bound_bytes"]/1e9:.6f} GB**. '
                'These API bounds are not allocated-byte measurements or predicted savings; production workspace sharing is not accepted. '
                'Evidence: '+link(memory_tag+'-context-requirements.json')+'.','']
    if real_rows:lines+=['| Real cohort | Equal-output groups | Prior control ms | Joined ms | vLLM ms | LM Studio ms | Same token sequences | Retained bytes |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']+real_rows+['']
    lines+=real_notes
workspace_plain=read('shared-workspace-plain.json')
if workspace_plain:
    lines+=['## Standalone shared-context workspace prototype','',
        'Two fixed FP32 MatMul-ReLU-MatMul networks compare independently allocated contexts against one fixed shared '
        'scratch owner. Tests cover ordinary and captured execution, alternating streams, two host submission threads, '
        'complete output bits and a prospectively fixed FP64 forward-error bound, undersized-buffer rejection, '
        'padding guards and owner lifetime after caller release. The source and executable were archived before execution. '
        'This is not Garnet integration, BF16 model qualification, measured model savings or a latency result.','',
        '| Prototype gate | Saved result | Output checks |','|---|---|---:|']
    for kind in ('plain','memcheck','racecheck','synccheck'):
        name='shared-workspace-'+kind;exit_data=read(name+'-exit.json');data=read(name+'.json')
        passed=exit_data and exit_data['exit_code']==0 and data and data['all_passed']
        status='pending' if exit_data is None else 'pass' if passed else 'failed'
        checks=data.get('shared_output_checks','—') if data else '—'
        lines.append(f'| {kind} | {status} | {checks} |')
    lines+=['','Sanitizer runs are unfiltered. Each invocation builds its own engines, so selected tactics and reported '
        'scratch requirements may differ; comparisons use the same engine within each invocation. '
        'The production candidate shares only the two decoder-body programs and remains default-off pending separate qualification. '
        'Evidence: '+link('serial-context-workspace-design.json')+', '+link('shared-workspace-before-execution-identity.json')+', '
        +link('shared-decoder-activation-production-design.json')+'.','']
activation_identity=read('decoder-activation-legacy-output-after-build-identity.json')
if activation_identity:
    sha=activation_identity['garnet_binary_sha256']
    lines+=['## Managed decoder activation storage — implementation under validation','',
        'The working implementation adds an explicit user-managed context policy, one-context pools, a fixed per-decoder '
        'SDK buffer sized from the actual built-engine maximum, storage validation, overlap rejection and a shared WRITE '
        'lease in each input/output transaction. Python counts the single arena owner once. The profile defaults off. '
        'Current working DLL SHA256: `'+sha+'`; the device fatbin matches the preserved build. '
        'The joined model timing and process-memory measurements above are historical DLL3c measurements, '
        'not performance evidence for this new working build.','',
        'The first implementation (DLL2aa) passed all50 native/SDK output checks but failed unfiltered memcheck with '
        'one512-byte D2D use-after-free at runtime teardown. That failure is retained. A simpler default-workspace control '
        'did not reproduce it. Public compiled-graph outputs now use the tensor owner’s matching legacy allocation path; '
        'this corrects a documented allocation/cleanup mismatch but does not alone prove the full original failure’s cause. '
        'The original failing test source is unchanged for the rerun. Evidence: '
        +link('decoder-activation-sdk-memcheck-exit.json')+', '+link('compiled-output-allocation-design.json')+'.','',
        '| Current native gate | Saved result |','|---|---|']
    for name in ('sdk-plain','sdk-memcheck','sdk-racecheck','sdk-synccheck','default-grouped-io-plain','default-grouped-io-memcheck','default-external-plain','default-external-memcheck'):
        d=read('decoder-activation-legacy-output-'+name+'-exit.json')
        status='pending' if d is None else 'pass' if d['exit_code']==0 and d['garnet_dll_sha256'].lower()==sha.lower() else 'failed or binary mismatch'
        lines.append(f'| {name} | {status} |')
    lines+=['','The SDK test includes a shared pool, full output bits, missing/undersized/logical-view/host/symbolic/readonly '
        'binding rejection, input/output alias rejection, two-thread queued clones, partial close, caller-owner release and '
        'retained last outputs. Unfiltered sanitizer status is separate from its50 output assertions. '
        'Real-model preservation and measurement evidence is reported separately below; independent quality acceptance remains open.','']
    lines+=['| Fixture state gate | Saved result |','|---|---|']
    for kind in ('plain','memcheck'):
        d=read('activation-legacy-output-equivalence-'+kind+'-exit.json')
        status='pending' if d is None else 'pass' if d['exit_code']==0 and d['garnet_dll_sha256'].lower()==sha.lower() else 'failed or binary mismatch'
        lines.append(f'| B1/2/4/8 complete finite state/hidden/ID/slot equivalence, {kind} | {status} |')
    lines+=['','| Numerical mode | Current passing profiles | Required |','|---|---:|---:|']
    for mode in ('default','joined','shared'):
        paths=list(root.glob(f'activation-legacy-output-{mode}-verifier-b*-c*-exit.json'))+list(root.glob(f'activation-legacy-output-{mode}-session-b*-p*-d4-exit.json'))
        count=sum((d:=json.loads(path.read_text()))['exit_code']==0 and d['garnet_dll_sha256'].lower()==sha.lower() for path in paths)
        lines.append(f'| {mode} | {count} | 12 |')
    lines+=['','Fixture checks retain the original rollback, stop, fault, mask and oracle-branch assertions and numerical '
        'tolerances. They do not establish pretrained model quality or speed. Real-model validation remains a separate gate.','']
    lines+=['| Real activation cohort | Prior joined ms | Activation ms | Same token sequences | Prior / activation retained bytes |',
        '|---|---:|---:|---:|---:|']
    real_evidence=[]
    for size,batch,extra in (('2b',1,'-mtp3-cache128'),('4b',8,'')):
        tag='packed-p64-d4-mtp3-k8192-replay-resident-serial-register128-flat-targetserial-aux0-deferred-nativecommit-arena-promptpack-headreuse-joineddecode-activationarena'
        name=f'matched512-{size}-b{batch}-{tag}{extra}-candidate.json';data=read(name)
        state_name=f'activation-pretrained-{size}-b{batch}-promptpack-headreuse-joineddecode-cross-prior-state-control.json';state=read(state_name)
        workload_name=f'activation-{size}-b{batch}-all28-workload-control.json';workload=read(workload_name)
        if data and state and state['all_passed'] and workload and workload['all_passed']:
            w=data['median_group_ms'];owned=data['retained_owned_bytes']
            lines.append(f'| {size} B{batch} | {fmt(w["control"])} | {fmt(w["candidate"])} | {data["same_sequence_cases"]}/28 | {owned["control"]} / {owned["candidate"]} |')
            real_evidence.append(link(name)+', '+link(state_name)+', '+link(workload_name))
    lines+=['','Saved real probes require every finite target-state hash and ordinary/repeated outputs/stops to match '
        'the prior joined control. Cohort audits additionally require identical all28 discrete MTP calls, accepted drafts '
        'and iteration progress. Single activation timings compare against historical DLL3c controls; they include the '
        'working DLL’s allocation-contract correction and do not isolate sharing’s effect. Retained accounting is not '
        'sampled process memory or a true peak; same-build timing and memory pairs remain separate.','']
    if real_evidence:lines+=['Evidence: '+'; '.join(real_evidence)+'.','']
    lines+=['| Current integration gate | Saved result |','|---|---|']
    for name in ('verifier-b8-memcheck','session-b8-memcheck','session-b8-racecheck','session-b8-synccheck'):
        evidence='activation-'+name+'-exit.json';d=read(evidence)
        passed=d is not None and d['exit_code']==0 and d['garnet_dll_sha256'].lower()==sha.lower()
        lines.append(f'| {name} | {"pass" if passed else "pending or failed"} |')
    lines+=['','Integration memchecks are unfiltered; integration race/sync checks cover only `packedCommit` kernels. '
        'These scopes do not establish full integration race/sync coverage.','']
    for size,batch in (('2b',1),('4b',8)):
        evidence=f'packed-study-{size}-b{batch}-activation-arena-summary.json';study=read(evidence)
        if study:
            assert study['same_all_28_input_prompt_output_stop_signatures'] and study['garnet_dll_sha256'].lower()==sha.lower()
            values=study['aggregate_group_median_ms']
            lines += [f'Fresh same-build {size} B{batch}, all20 warm groups: independent context storage '
                f'{fmt(values["a"])} ms; shared activation storage {fmt(values["b"])} ms. '
                'All28 input/prompt/output/stop signatures match across the four processes. Evidence: '+link(evidence)+'.','']
            lines += ['| Shared-arena repeat | Full-request emitted output tok/s, batch aggregate | Derived post-first-sample decode tok/s, batch aggregate |',
                '|---|---:|---:|']
            for role in ('b1','b2'):
                metrics=study['per_run'][role]
                lines.append(f'| {role} | {fmt(metrics["median_full_request_output_tokens_per_second"])} | {fmt(metrics["median_derived_post_first_sample_tokens_per_second"])} |')
            lines += ['','Rates are medians over all20 warm groups in each run. Derived decode rate subtracts one sampled '
                'token per row and includes host/stop overhead; it is not standalone GPU kernel throughput. '
                'These timing-derived values carry the same external-build/interruption limitations as latency.','']
            baseline_name=f'packed-study-{size}-b{batch}-activation-arena-all-baselines.json';baseline=read(baseline_name)
            if baseline:
                w=baseline['median_group_ms']
                lines += [f'On the common {baseline["common_equal32_groups"]} equal32 groups across all listed engines, '
                    f'Garnet shared storage takes {fmt(w["garnet_b"])} ms; fastest saved competitor '
                    f'{baseline["fastest_tested_baseline"]} takes {fmt(w[baseline["fastest_tested_baseline"]])} ms; '
                    f'the2× threshold is {fmt(baseline["half_fastest_baseline_ms"])} ms. '
                    'Competitors are historical cohorts; APIs, intermediate arithmetic and speculative settings differ. '
                    'Evidence: '+link(baseline_name)+'.','']
        memory_name=f'activation-memory-{size}-b{batch}-pair-summary.json';memory=read(memory_name)
        if memory:
            assert memory['garnet_dll_sha256'].lower()==sha.lower()
            values=memory['profiles'];a=values['control'];b=values['candidate']
            lines += [f'Isolated same-build {size} B{batch} warmed Windows process counters: '
                f'control total {a["warm_process_counter_bytes"]["TotalCommitted"]} B; '
                f'shared storage total {b["warm_process_counter_bytes"]["TotalCommitted"]} B. '
                f'Retained owned allocations: {a["retained_owned_model_bytes"]} → {b["retained_owned_model_bytes"]} B. '
                'These are separate control/candidate processes with matched six-group workloads and profiles; '
                'sampled committed counters do not prove true peak or physical residency. Evidence: '+link(memory_name)+'.','']
            observed=[]
            for mode,suffix in (('control','activationcontrol'),('candidate','activationarena')):
                summary_name=f'memory-combined-{size}-b{batch}-512-targetserial-aux0-deferred-nativecommit-arena-promptpack-headreuse-joineddecode-{suffix}-summary.json'
                summary=read(summary_name)
                if summary:observed.append((mode,summary['largest_observed_bytes']['TotalCommitted'],summary_name))
            if len(observed)==2:
                lines += [f'Largest sampled process total across the monitored phases: {observed[0][1]} → {observed[1][1]} B. '
                    'Formatted-provider cadence and startup/shutdown gaps can miss transient peaks. These maxima are '
                    'distinct from warmed medians and do not establish a true lifetime peak. Evidence: '
                    +link(observed[0][2])+', '+link(observed[1][2])+'.','']
    resume=read('activation-paired-resume-record.json')
    if resume:lines += ['The2B ABBA study was interrupted after its successful first control process. '
        'The complete A1 evidence was preserved; remaining roles resumed after source/runtime identity checks. '
        'This timing gap is a limitation in addition to having only two repeats. No significance or causal claim is made. '
        'Evidence: '+link('activation-paired-resume-record.json')+'.','']
    interference=read('activation-paired-build-interference.json')
    if interference:lines += ['An unrelated Galaxy native build was observed concurrently with the2B A2 request sequence; '
        'compiler creation times fall inside that request envelope. This violates exclusive timing conditions. '
        'The2B activation-study latency result is provisional and cannot support promotion or attribution to sharing. '
        'All raw outputs and failures remain retained. The4B guard waited for active model/compiler processes to clear. '
        'Evidence: '+link('activation-paired-build-interference.json')+'.','']
    interference4=read('activation-4b-build-interference.json')
    if interference4:lines += ['A later external Galaxy/Cantor build overlapped the4B B1 request sequence. '
        'The4B activation timing also remains provisional and cannot support promotion or causal attribution. '
        'The remaining roles resumed with a validated process-snapshot observer. Event subscription was denied by Windows; '
        'snapshots can miss short processes between samples. No external processes were stopped. '
        'Evidence: '+link('activation-4b-build-interference.json')+', '+link('activation-observed-continuation-processes.jsonl')+'.','']
    package_design=read('qwen35-packaging-repair-design.json')
    if package_design:lines += ['## CUDA program packaging — bounded validation','',
        'The working CMake rules now preserve the flat model catalog while also supplying the `xModel` namespace and '
        '`runtime/cuda_serving.py` at the package root. New2B/4B manifests reference the actual CUDA factories and external '
        'checkpoints. Existing Metal profile files are preserved. CPU asset checks pass; native catalog, installation and '
        'real packaged serving remain separate gates. The package smoke checks actual installed module paths, import '
        'roots, fixed real-image token prefixes, replay, profile conflicts and unload. An initial CPython audit-hook '
        'guard did not observe XLang native open; it failed before loading any model and is retained. The source checkout '
        'remains accessible, so module-origin checks do not prove filesystem isolation or clean-machine deployment. '
        'The newly catalogued shared-hybrid profile templates have file/schema validation; the native packaged smokes '
        'below use the separately captured packed MTP profiles. Template-specific packaged inference remains open. '
        'Evidence: '+link('qwen35-packaging-repair-design.json')+'.','']
    package_build=read('qwen35-package-build-exit.json');catalog=read('qwen35-package-native-catalog-corrected-exit.json')
    if package_build and package_build['exit_code']==0 and package_build['garnet_unchanged'] and catalog and len(catalog)==2 and all(row['exit_code']==0 for row in catalog):
        lines += ['Package configure/install and both source/installed native catalog tests pass with eight model IDs. '
            'Only the standalone catalog test was compiled; the inference DLL is unchanged. The initial catalog harness '
            'used the wrong executable subdirectory and failed before launch; the corrected invocation and original failure are retained. '
            'Evidence: '+link('qwen35-package-build-exit.json')+', '+link('qwen35-package-native-catalog-corrected-exit.json')
            +', '+link('qwen35-package-catalog-path-correction.json')+'.','']
    package_serving=read('qwen35-packaged-serving-origin-summary.json')
    if package_serving and package_serving['all_passed']:
        lines += ['| Installed native serving smoke | Module origins inside package | Fixed16-token prefix/replay/conflict/unload |',
            '|---|---:|---|']
        for case in package_serving['cases']:
            lines.append(f'| {case["model"]} B{case["batch"]} | {case["module_count"]} | pass |')
        lines += ['','Both models ran sequentially on GPU0 from a working directory outside the checkout, using explicit '
            'installed model directories. Adapter and model module origins are inside the package and observed import roots '
            'exclude the checkout. The source directory remains accessible, and installed CUDA/TensorRT/XLang dependencies '
            'remain available. This does not prove filesystem isolation, clean-machine deployment, broad quality or performance. '
            'Evidence: '+link('qwen35-packaged-serving-origin-summary.json')+', '+link('qwen35-package-origin-method-correction.json')+'.','']
bf16conv=read('vllm-2b-b1-bf16-conv-workload-audit.json')
expandedconv=read('packed-study-2b-b1-activation-arena-all-baselines-expanded-conv.json')
if bf16conv and expandedconv:
    values=expandedconv['median_group_ms']
    lines += ['## Additional vLLM2B MTP convolution-cache profile','',
        f'A fresh28-request 2B B1 MTP3 run requests BF16 convolution cache and FP32 recurrent state, with128MiB cache per '
        f'sequence and512-token capacity. All28 inputs, prompt IDs, cache declarations and output limits match the saved '
        f'FP32-convolution profile; {bf16conv["same_output_stop_signatures_vs_saved_fp32_conv"]}/28 output/stop signatures match. '
        'Changed cases are retained and are not treated as independent quality acceptance. The raw loader also reports '
        'FP32 checkpoint norm values being loaded into BF16 parameters, illustrating the intermediate-arithmetic differences.','',
        f'On one common {expandedconv["common_equal32_groups"]}-group equal32 subset across all included profiles, BF16-convolution '
        f'vLLM takes {fmt(values["vllm-mtp3-cache128-convbfloat16"])} ms versus saved FP32-convolution '
        f'{fmt(values["vllm-mtp3-cache128"])} ms. The fastest tested2B baseline and2× threshold remain unchanged at '
        f'{fmt(values[expandedconv["fastest_tested_baseline"]])} and {fmt(expandedconv["half_fastest_baseline_ms"])} ms. '
        'One later cohort does not establish a causal dtype performance effect; Garnet activation-study timings remain provisional.','',
        'A60-second named-process idle gate passed before launch. The sampled Windows observer found no known named-process '
        'overlap in the request envelope, but can miss short processes and does not establish complete host/GPU isolation. '
        'Actual allocated cache tensor dtypes and engine/process memory remain separate diagnostic gates; frontend Torch '
        'allocator fields are not total engine memory. Evidence: '+link('vllm-2b-b1-bf16-conv-workload-audit.json')+', '
        +link('packed-study-2b-b1-activation-arena-all-baselines-expanded-conv.json')+'.','']
mlp_screen=read('mlp-joined-projection-rejection.json')
if mlp_screen:
    lines += ['## MLP gate/up joining — component screen rejected','',
        'An unwired extension of the existing projection probe compares two gate/up GEMMs with one joined GEMM, '
        'including exact output split cost. The production projection implementation, BF16 operands/output, FP32 '
        'accumulation policy and original FP64 bounds are unchanged. All16 shapes pass the original numerical, guard, '
        'input-byte and split checks; all eight decode-shaped fixtures have identical BF16 results.','',
        '| Component deployment anchor | Separate µs | Joined plus split µs | Separate / joined |',
        '|---|---:|---:|---:|']
    for case in mlp_screen['deployment_anchors']:
        lines.append(f'| {case["model"]}, M{case["m"]} | {fmt(case["separate_swept_graph_us"])} | {fmt(case["joined_split_swept_graph_us"])} | {case["joined_split_swept_speed_ratio"]:.3f}× |')
    lines += ['','The predeclared minimum5% swept-latency reduction at every deployment anchor fails. The broader '
        'candidate is rejected, production is unchanged, and memcheck was explicitly not run after the preliminary gate '
        'failed. One shape improves while another regresses; these synthetic nine-sample graph results do not establish '
        'full-model benefit or universal claims about all grouping implementations. Prefill fixtures and every changed '
        'result remain retained. Evidence: '+link('mlp-joined-projection-rejection.json')+', '
        +link('mlp-joined-projection-screen-preliminary.json')+'.','']
external_store=read('packed-store-external-alias-rejection.json')
if external_store:
    assert external_store['status']=='REJECTED_PREDECLARED_NETWORK_IO_ALIAS_GATE'
    lines += ['## Packed state stores — API probes','',
        'The external destination/input-output alias approach is rejected: the installed TensorRT engine did not '
        'report the required final network output alias after the two-plugin chain. The B1/D4 copy control passed22 '
        'complete buffer/guard checks and16 negative contracts, but the alias case failed before enqueue. Unfiltered '
        'diagnostics were not run for this rejected method, and production ownership rules were not relaxed. '
        'Evidence: '+link('packed-store-external-alias-rejection.json')+'.','']
internal_store=read('internal-packed-store-component-summary.json')
if internal_store:
    assert internal_store['status']=='COMPONENT_PASSED_PRODUCTION_UNCHANGED'
    lines += ['A distinct internal-root probe creates a small serialized TensorRT Fill operation and chains documented '
        'V3 aliases inside the engine. There is no destination network input and the public output remains separate '
        'from every network input. Both stores must receive exactly aliased root pointers, while hidden forwarding '
        'is copied. Batches1/2/4/8 and widths4/64 each test control/candidate, two deserializations, graph capture/replay, '
        'rebinding, complete uint32 bit patterns including signed zero and NaN payloads, readonly input preservation, '
        'guards and caller output survival after runtime teardown.','',
        '| Diagnostic | Cases | Full buffer/guard checks | Negative contracts |',
        '|---|---:|---:|---:|']
    for case in internal_store['diagnostics']:
        lines.append(f'| {case["mode"]} | {case["completed_cases"]} | {case["complete_buffer_and_guard_checks"]} | {case["negative_contract_checks"]} |')
    lines += ['','Unfiltered memcheck and synccheck report zero errors; unfiltered racecheck reports zero errors, '
        'warnings and hazards. These copy/Fill checks do not prove arbitrary concurrency safety.','',
        '| Synthetic batch/width | Copy context upper bound B | Internal alias context upper bound B |',
        '|---|---:|---:|']
    for pair in internal_store['synthetic_context_pairs']:
        lines.append(f'| B{pair["batch"]}/W{pair["chunk"]} | {pair["copy_context_upper_bound_bytes"]:,} | {pair["internal_alias_context_upper_bound_bytes"]:,} |')
    plan_sizes=sorted({(p['control_serialized_bytes'],p['candidate_serialized_bytes']) for p in internal_store['synthetic_context_pairs']})
    lines += ['','The serialized control/candidate plan sizes are '
        +', '.join(f'{a:,}/{b:,} B' for a,b in plan_sizes)+', rather than embedded full zero roots. '
        'These are context requirements for two synthetic copy stores; they exclude public outputs and other runtime '
        'allocations and are not model memory savings or latency results. The production DLL was unchanged during that API probe. '
        'Actual lowerer integration, original state/real-image/regression controls and full-request timing plus separate '
        'memory measurements remain open. The external-root rejection remains unchanged.','',
        'The initial guarded-workspace probe failed its512-byte alignment requirement; a separate internal-root attempt '
        'used unsupported scalar Fill setters for a rank2 root and failed before execution. The ND Fill correction '
        'first used an unavailable Dims1 convenience type and failed compilation; explicit Dims then compiled. Every '
        'original log, source/runtime archive and subsequent successful run is retained. Evidence: '
        +link('internal-packed-store-component-summary.json')+', '+link('packed-store-alias-workspace-alignment-correction.json')
        +', '+link('internal-packed-store-nd-fill-correction.json')+', '+link('internal-packed-store-production-design.json')+'.','']
internal_integration=read('internal-packed-store-integration-summary.json')
internal_partial=read('internal-store-integration-v1-decoder-equivalence-resumed-plain.json')
if not internal_integration and internal_partial and internal_partial.get('all_passed'):
    lines += ['## Internal packed stores — integration validation running','',
        'The default-off `internalPackedStores` path is implemented in native lowering, packed decoder/program and '
        'the serving adapter. The actual production plugin passed48 strongly typed ordinary cases, and the two-layer '
        'decoder fixture matches full finite state/hidden/ID/slot and retained payload hashes at B1/2/4/8. B4 exact '
        'comparison snapshots were recovered after the later B8 payload diagnostic reached its copy limit; B4 memory '
        'metadata was not saved. Streaming payload hashes retain every finite word and preserve the runtime limit.','',
        'Production component diagnostic status from saved exit records:','']
    for mode in ('plain','memcheck','racecheck','synccheck'):
        status=read('internal-store-integration-v1-production-probe-'+mode+'-exit.json')
        lines.append('- '+mode+(': exit '+str(status['exit_code']) if status else ': no terminal result yet'))
    lines += ['','No pretrained-model allocation, correctness or speed gain is established by this integration yet. '
        'Evidence: '+link('internal-store-integration-v1-decoder-equivalence-resumed-plain.json')+', '
        +link('internal-store-payload-diagnostic-correction.json')+'.','']
if internal_integration:
    assert internal_integration['status']=='PRODUCTION_PLUGIN_DIAGNOSTICS_AND_FIXTURE_PASSED'
    lines += ['## Internal packed stores — production integration, opt-in','',
        'The CUDA serving profile now accepts `internalPackedStores` (default off). It requires the packed/shared '
        'serial-target shared-output width4 profile. The lowerer passes the same existing hidden/state/payload values '
        'directly to a V3 store before per-layer concatenation, with one internal Fill root, a checked unbranched root '
        'chain and exactly one final hidden write. All existing layer FP32 barriers and public packed layout remain. '
        'Public output must not alias any input, and existing SDK ownership checks remain. Native statistics expose '
        'actual store/Fill counts and engine output alias metadata. Metal code is unchanged.','',
        'The native CUDA fatbin equals the prior build byte for byte; that establishes device-code preservation, '
        'not mathematical or performance equivalence. The actual production plugin passes the following strongly typed '
        'network tests with1/2/6 parts, B1/2/4/8, W4/W64, deserialization, exact special bit patterns, guards, '
        'capture/replay, rebinding and caller output lifetime:','',
        '| Production plugin diagnostic | Cases | Full buffer/guard checks | Negative contracts |',
        '|---|---:|---:|---:|']
    for case in internal_integration['production_plugin_diagnostics']:
        lines.append(f'| {case["mode"]} | {case["cases"]} | {case["full_buffer_guard_checks"]} | {case["negative_contract_checks"]} |')
    lines += ['','The separate decoder fixture comparison retains joined projections, shared output and shared '
        'activation storage in both modes, and enables retained decode payloads in both. All four batches match nine '
        'complete finite canonical-state/hidden/ID/slot snapshots, including eight full retained-payload hashes after '
        'masked decode steps. This is a small two-layer fixture, not pretrained-model validation.','',
        '| Fixture batch | Prior body contexts B | Internal-store body contexts B | Prior owned decoder B | Candidate owned decoder B |',
        '|---|---:|---:|---:|---:|']
    for case in internal_integration['fixture_records']:
        if case.get('native_memory_metadata_not_persisted'):
            lines.append(f'| {case["batch"]} | not saved | not saved | not saved | not saved |')
        else:
            old='/'.join(f'{n:,}' for n in case['old_body_context_bytes']);new='/'.join(f'{n:,}' for n in case['candidate_body_context_bytes'])
            lines.append(f'| {case["batch"]} | {old} | {new} | {case["old_owned_decoder_bytes"]:,} | {case["candidate_owned_decoder_bytes"]:,} |')
    lines += ['','The first batch8 baseline payload hash exceeded the bounded1048576-word diagnostic limit. The '
        'runtime limit was retained and the harness now hashes all words in bounded column chunks. Original successful '
        'B1/B2 records and the exact saved B4 nine-snapshot comparison were retained; only B8 resumed. B4 memory/native '
        'counter metadata was not persisted and is explicitly unavailable. Production runtime and compiled binaries '
        'were unchanged by this harness correction. These allocation figures exclude real-model/process peaks. '
        'Broader original numerical gates, pretrained controls, full28 cohorts, Qwen3VL, fresh timing and separate '
        'memory measurements remain required. Evidence: '+link('internal-packed-store-integration-summary.json')+', '
        +link('internal-store-payload-diagnostic-correction.json')+'.','']
internal_numbers=read('internal-packed-store-numerical-summary.json')
if internal_numbers:
    assert internal_numbers['status']=='ALL_SDK4_ORIGINAL36_INTERNAL12_FULLSTATE_MEMCHECK_PASSED'
    lines += ['## Internal packed stores — broader numerical gates','',
        'All four current-build SDK modes pass50 output checks each, including unfiltered memcheck/racecheck/synccheck. '
        'The original36 numerical cases and12 additional internal-store cases pass across the declared batch/prefill '
        'profiles, retaining original state/ID/mask/stop/prefix/rollback/fault/capacity assertions and bounds. '
        'The separate all-batch complete finite-state/hidden/payload comparison also passes unfiltered memcheck. '
        'Explicit oracle proposals are branch tests and never benchmark evidence. This is synthetic native coverage, '
        'not independent pretrained quality or measured speed. Evidence: '+link('internal-packed-store-numerical-summary.json')+'.','']
else:
    modes={}
    for mode in ('default','joined','shared','internal'):
        profiles=((8,4),(4,4),(8,16),(4,16)) if mode=='default' else ((1,4),(2,4),(4,4),(8,4))
        names=[f'internal-store-numerical-v1-{mode}-verifier-b{b}-c{c}' for b,c in profiles]
        names += [f'internal-store-numerical-v1-{mode}-session-b{b}-p{p}-d4' for b in (1,2,4,8) for p in (32,64)]
        modes[mode]=sum(bool((value:=read(name+'-exit.json')) and value['exit_code']==0) for name in names)
    if sum(modes.values()):
        lines += ['Broader internal-store numerical validation is incomplete. Saved successful terminal cases: '
            +', '.join(f'{m} {n}/12' for m,n in modes.items())+'. Final all-batch full-state memcheck remains a separate gate. '
            'Absence of a terminal file does not establish process status.','']
real_allocations=read('internal-packed-store-real-allocation-summary.json')
if real_allocations:
    lines += ['## Internal packed stores — actual model post-load allocations','',
        'Four fresh native processes load2B B1 control/candidate then4B B8 control/candidate sequentially. Both modes '
        'use the captured P64/D4 MTP3 joined/shared profiles; only internalPackedStores differs. Native counts, '
        'public output nonaliasing and workspace binding are asserted. State/request/output/weight/head owner '
        'categories are checked equal. This phase performs no inference and measures no request latency.','',
        '| Actual model | Prior body contexts B | Candidate body contexts B | Prior owned model B | Candidate owned model B | Owned reduction B |',
        '|---|---:|---:|---:|---:|---:|']
    for case in real_allocations['records']:
        old='/'.join(f'{n:,}' for n in case['control_body_context_bytes']);new='/'.join(f'{n:,}' for n in case['candidate_body_context_bytes'])
        lines.append(f'| {case["model"]} B{case["batch"]} | {old} | {new} | {case["control_owned_model_bytes"]:,} | {case["candidate_owned_model_bytes"]:,} | {case["owned_model_reduction_bytes"]:,} |')
    lines += ['','Actual initialized modules: '+', '.join(f'{c["model"]} vision={c["vision_initialized"]}, speculative={c["speculative_initialized"]}' for c in real_allocations['records'])+'. '
        'These post-load owned counters exclude possible driver/library/transient/build/graph-executable allocations; '
        'they are not warmed Windows process totals, physical residency or true lifetime peaks. The existing process '
        'memory chart remains a separate historical measurement. The later numerical, pretrained, full28 timing '
        'and warmed process sections provide their separate admission evidence. Evidence: '
        +link('internal-packed-store-real-allocation-summary.json')+'. Allocation chart: '
        +link('progress-internal-store-owned-allocations.png')+'.','']
metal_static=read('metal-factory-current-static-audit.json')
if metal_static:
    assert all(r['metal_constructor_calls_and_signatures_equal'] for r in metal_static['references'])
    lines += ['## Existing4B Metal path — current static audit','',
        'Constructor signatures and returned constructor arguments match both the pushed WorldSense checkpoint and '
        'its parent. The shared Metal decoder/vision files and4B serving profile are byte-identical to those references. '
        'The4B model wrapper moves imports into the existing factories and adds CUDA factories; the serving wrapper '
        'adds2B routing without changing the4B default vision sharing. All audited files are byte-identical to the '
        'pushed checkpoint. The CMake Metal branch returns before the CUDA project/source collection. '
        'No remote Mac action, fresh Metal build or inference is claimed. Equal source/call arguments are bounded '
        'compatibility evidence and do not prove complete current-source Metal numerical behavior. Evidence: '
        +link('metal-factory-current-static-audit.json')+', '+link('metal-factory-static-wrapper-differences.diff')+'.','']
work_audit=read('internal-store-workload-auditor-cpu-summary.json')
if work_audit:
    assert work_audit['all_passed']
    lines += ['The updated all-request workload auditor passes a CPU regression on saved matching activation cohorts '
        'and rejects a labelled synthetic fixture with one extra target verification call. The synthetic fixture is '
        'not a model run or timing result. Actual internal-store full28 cohorts are audited separately in the latest studies. '
        'Evidence: '+link('internal-store-workload-auditor-cpu-summary.json')+'.','']
cache_source=read('vllm-cache-view-source-inspection.json')
cache_cpu=read('vllm-cache-view-cpu-accounting.json')
cache_wsl_cpu=read('vllm-cache-view-cpu-accounting-wsl.json')
if cache_source and cache_cpu and cache_wsl_cpu:
    assert cache_cpu['all_passed'] and cache_wsl_cpu['all_passed']
    assert not cache_cpu['cuda_context_initialized'] and not cache_wsl_cpu['cuda_context_initialized']
    lines += ['## vLLM allocated cache views — diagnostic preparation','',
        'Read-only inspection of the installed worker/runner and Mamba cache binding code shows that raw runner '
        'cache allocations may be byte tensors. MambaBase binds separate typed state views, and the Qwen GDN layer '
        'uses the first for convolution state and the second for recurrent state. The raw allocation dtype alone '
        'therefore does not establish the allocated state precision. Installed source hashes are retained.','',
        'The diagnostic worker extension now has `qwen35_cache_snapshot`, which records actually bound layer tensor '
        'dtype, shape, stride, logical bytes and underlying storage metadata, plus existing worker allocator counters. '
        'Shared storage is deduplicated by device/base/size. `--cache-allocation-output` requires diagnostic mode and '
        'records loaded/after-warmup/complete observations; diagnostic rows are explicitly ineligible for primary '
        'performance acceptance. The normal benchmark path does not invoke this RPC.','',
        f'CPU checks pass with Windows Torch {cache_cpu["cpu_torch"]} and the installed WSL environment Torch '
        f'{cache_wsl_cpu["cpu_torch"]}, without initializing CUDA. BF16/FP32 views of one64-byte allocation count as64 B; '
        'repeated references remain deduplicated and a second allocation raises the count to128 B. The source bytes '
        'remain unchanged. These CPU checks establish the accounting behavior; the separate GPU diagnostics below '
        'establish the worker RPC and actual bound cache types. Worker allocated/reserved bytes are not Windows process totals, and mem_get_info is '
        'whole-device. Evidence: '+link('vllm-cache-view-source-inspection.json')+', '
        +link('vllm-cache-view-cpu-accounting.json')+', '+link('vllm-cache-view-cpu-accounting-wsl.json')+'.','']
cache_gpu=read('vllm-cache-precision-diagnostic-summary.json')
if cache_gpu:
    assert cache_gpu['all_three_diagnostics_passed'] and not cache_gpu['primary_performance_eligible']
    lines += ['## vLLM actual CUDA cache types and allocator observations','',
        'Three fresh sequential diagnostic processes reproduce all 12 prior input/prompt/token/stop/cache groups '
        'each, with options differing only by the worker extension. Loaded, after-warmup and complete snapshots '
        'verify the same physical RTX 4080 UUID as Garnet and saved LM Studio. These RPC/hold runs are excluded '
        'from primary timing.','',
        '| Model / batch | Conv cache | Recurrent cache | Referenced backing storage MiB | Worker allocated GiB | Worker reserved GiB |',
        '|---|---|---|---:|---:|---:|']
    for case in cache_gpu['cases']:
        dtype=lambda value:{'torch.float32':'FP32','torch.bfloat16':'BF16'}[value]
        lines.append(f'| {case["model"]} B{case["batch"]} | {dtype(case["conv_dtype"])} | '
            f'{dtype(case["recurrent_dtype"])} | {case["referenced_cache_storage_bytes"]/(1<<20):.2f} | '
            f'{case["worker_allocated_bytes"]/(1<<30):.3f} | {case["worker_reserved_bytes"]/(1<<30):.3f} |')
    lines += ['','Each inventory references one shared backing allocation. The 4B typed/raw logical views sum '
        'to 8,462,008,320 bytes but reference 1,069,547,520 allocated storage bytes; adding those view sizes would '
        'overcount. Raw runner views include int8 storage views and BF16 views, while the typed GDN views above '
        'establish actual state precision. The other bound layer-cache views are BF16. A fixed configured budget '
        'does not imply identical allocated bytes when convolution precision changes.','',
        'Load logs also retain FP32 checkpoint norm → BF16 parameter warnings. These are precision provenance, '
        'not a complete parameter census or proof of identical arithmetic. Worker allocated/reserved counters, '
        'deduplicated referenced cache storage, logical views and Windows committed process counters are separate '
        'measurements; no complete vLLM process VRAM, physical residency, true lifetime peak, independent quality '
        'or new latency claim follows. Evidence: '+link('vllm-cache-precision-diagnostic-summary.json')+'.','']
lines += ['Historical shared-activation charts: '+link('progress-activation-memory.png')+' (warmed process memory), '
    +link('progress-activation-baselines-expanded-conv.png')+' (provisional latency with additional cache profile).','']
pretrained_internal=read('internal-packed-store-pretrained-summary.json')
if pretrained_internal:
    assert pretrained_internal['status']=='BOTH_PRETRAINED_ORDINARY_PRIOR_REPEATED_EXACT_CONTROLS_PASSED'
    lines += ['## Internal packed stores — real pretrained state controls','',
        'The current CUDA build passes the 2B B1 and 4B B8 controls. Each model runs two repeated real-image '
        'native-prefix requests, closes that model, then loads an ordinary non-MTP target separately. Complete '
        'finite canonical-state and hidden-state hashes, output tokens and stop behavior match the ordinary '
        'target and the saved prior implementation. The profile changes only internalPackedStores.','',
        'These diagnostic requests include state hashing and cold calls. They establish the declared state/output '
        'controls for one image group per model; they do not establish full28 workload preservation, independent '
        'quality, primary latency or warmed process memory. Evidence: '
        +link('internal-packed-store-pretrained-summary.json')+'.','']
batch_vision=read('batched-vision-workspace128-v1-summary.json')
if batch_vision:
    assert batch_vision['all_six_profiles_screened']
    lines += ['## Batched vision component — original-bound screening','',
        'The existing batch-aware vision lowerer was screened separately for 2B and 4B at batches 2, 4 and 8. '
        'Each run uses eight distinct real images and sixteen repeated serial reference calls, then closes '
        'the serial engine before loading the batched engine. Two batched passes reverse row order. '
        'Original numerical bounds remain 2e-5 absolute plus 2e-4 relative; failures are retained.','',
        '| Model / batch | Rows outside bound / compared | Maximum absolute difference | Status |',
        '|---|---:|---:|---|']
    for row in batch_vision['cases']:
        lines.append(f'| {row["model"].upper()} B{row["batch"]} | {row["rows_outside_original_bound"]} / '
            f'{row["compared_rows"]} | {row["maximum_absolute_error"]:.6g} | {row["status"]} |')
    lines += ['','All six profiles failed the unchanged serial parity bound, so graph-replay and sanitizer '
        'admission were skipped. No batched vision routing is integrated into serving. Stable same-batch '
        'outputs under row permutation do not establish serial equivalence or independent vision quality. '
        'The initial 4B B8 build also failed its 64 MiB builder workspace cap: TensorRT requested 67,237,888 '
        'bytes. That failed run is retained. The separately identified retry uses 128 MiB explicitly; '
        'the ordinary encoder default remains unchanged. Serial64/batched128 also changes the build cap, '
        'so these runs do not isolate batching alone as the cause. These are component diagnostics with no '
        'performance, process-memory, full-request, or 2× acceptance. Evidence: '
        +link('batched-vision-workspace128-v1-summary.json')+'.','']
compact_kv=read('compact-kv-component-v1-summary.json')
if compact_kv:
    assert compact_kv['all_four_modes_passed'] and not compact_kv['serving_integrated']
    lines += ['## Compact KV output prototype — standalone admission only','',
        'A copy-only CUDA prototype packs updated full-cache slots into per-token FP32 K/V tensors and '
        'scatters accepted prefixes into canonical cache positions. Plain, unfiltered memcheck, racecheck '
        'and synccheck each pass 192 complete-buffer cases and 160 negative checks, with zero reported '
        'errors, hazards or warnings. The corpus covers KV heads2/4, batches1/2/4/8, capacities384/512, '
        'chunks4/64, boundary/middle/interior positions and zero/full/mixed/partial prefixes. Compact '
        'outputs, complete canonical caches, source caches, slot/active inputs and poisoned guards '
        'remain exact.','',
        'At that standalone checkpoint, the prototype was not connected to TensorRT or serving. Tensor-owner/alias rules, invalid-control '
        'flag consumption, output-layout schema, state transactions, rollback, poisoning, real-model '
        'correctness and isolated memory/latency remain unqualified. A CPU layout plan matches the '
        'current measured output-pool sizes and computes prospective logical pool differences of '
        '11.906 MiB for2B B1 and254 MiB for4B B8. Those are layout arithmetic, not measured allocation, '
        'process-counter or peak savings. No model or2× promotion follows from this component. Evidence: '
        +link('compact-kv-component-v1-summary.json')+'; '+link('compact-kv-output-layout-plan.json')+'.','']
workspace_isolation=read('vision-workspace-isolation-v1-summary.json')
if workspace_isolation:
    assert workspace_isolation['all_passed']
    lines += ['### Same-batch vision workspace isolation','',
        'Both models pass separate batch1 serial64-MiB versus serial128-MiB comparisons and unfiltered '
        'memcheck with zero reported errors. Per run, sixteen serial references and forty ordinary/replay '
        'comparisons over eight distinct images are bitwise identical. Source/staging checks and actual '
        'one-context/one-capture/replay counters pass. Workspace alone does not reproduce the failed '
        'batch2/4/8 parity on these controls. Batch-specific tactics, reduction order or other causes '
        'remain unisolated; this is not independent model-quality admission. Evidence: '
        +link('vision-workspace-isolation-v1-summary.json')+'.','']
manual_projection=read('cublaslt-manual-v1-summary.json')
if manual_projection:
    assert not manual_projection['all_three_component_gates_passed']
    row=manual_projection['records'][0];candidate=row['candidate']
    assert row['case']=='joined' and not row['gate_passed'] and candidate['selection_rank']==0
    lines += ['## Manual cuBLASLt algorithm screen — rejected first anchor','',
        'The actual 4B B8 joined decode shape M32/K2560/N12352 was screened with explicit algorithm, '
        'tile, stage and split-K settings under 8 MiB workspace, custom0/swizzle0 and BF16 I/O with FP32 '
        'accumulation. Twenty IDs yielded 2,073 checked configurations, 280 executed configurations and '
        'eight finalists. Rank0 was selected on three warm samples before nine new alternating '
        'baseline/candidate graph measurements with a 256-MiB eviction sweep outside the timed region.','',
        f'That prospective candidate measured {candidate["candidate_median_us"]:.3f} µs versus '
        f'{candidate["baseline_median_us"]:.3f} µs for production GEMM, '
        f'{candidate["component_speed_ratio"]:.3f}×; it failed the original 1.10× gate. '
        'Whole BF16 outputs were bitwise identical for rank0; the unchanged sampled FP64 bound, all '
        'buffer guards and full source/weight preservation passed. Other finalists cannot rescue this '
        'rejection. The queued MLP/proposal anchors and sanitizers were not run after the failed gate. '
        'No native provider, model-speed or quality promotion occurred. This limited search does not '
        'establish a global algorithm optimum or validated L2 residency. Evidence: '
        +link('cublaslt-manual-v1-summary.json')+'.','']
vision_components=read('vision-graph-component-summary.json')
if vision_components:
    assert vision_components['all_passed']
    lines += ['## Opt-in vision CUDA graph replay — component admission','',
        'The CUDA vision encoder previously ran its fixed graph without enabling CUDA graph replay. The new '
        'default-OFF visionCudaGraph profile adds one 1,474,560-byte FP32 staging input, refreshes it for every '
        'image, and uses serial context reuse. It loads no additional checkpoint/table and changes no model '
        'factory signature, Metal source, preprocessing, math or precision. Owned accounting includes staging.','',
        'Both models pass plain and unfiltered memcheck with zero reported errors. Per run,16 ordinary and16 '
        'replay calls on the same compiled engine preserve every finite visual FP32 word over8 unique images. '
        'Source patches remain unchanged; staging is poisoned before each refresh. Native counters prove one '
        'context, one capture, replays and no capture failures. An extra enqueue followed immediately by close '
        'preserves its output; double enable and closed calls are rejected.','',
        'These checks establish same-engine visual and ownership equivalence. The latest full28/ABBA section '
        'records separate request/timing admission; the exact-profile warmed memory section records its allocation cost. True batched '
        'vision is a separate future candidate. The full2× goal and broad independent quality remain open. '
        'Evidence: '+link('vision-graph-component-summary.json')+'.','']
legacy_tag='qwen3-vl-promptpack-dll-regression-internal-stores'
legacy_exit=read(legacy_tag+'-exit.json')
if not legacy_exit or legacy_exit['exit_code']!=0:
    legacy_tag='qwen3-vl-promptpack-dll-regression-packaged-programs'
    legacy_exit=read(legacy_tag+'-exit.json')
    if not legacy_exit or legacy_exit['exit_code']!=0:
        legacy_tag='qwen3-vl-promptpack-dll-regression-activation-arena'
        legacy_exit=read(legacy_tag+'-exit.json')
        if not legacy_exit or legacy_exit['exit_code']!=0:
            legacy_tag='qwen3-vl-promptpack-dll-regression-joined-decode'
            legacy_exit=read(legacy_tag+'-exit.json')
            if not legacy_exit or legacy_exit['exit_code']!=0:legacy_tag='qwen3-vl-promptpack-dll-regression-fixed-root'
legacy_name=legacy_tag+'.json';legacy=read(legacy_name);legacy_exit=read(legacy_tag+'-exit.json')
if legacy and legacy_exit and legacy_exit['exit_code']==0:
    rows=legacy['rows']
    assert legacy['status']['ready'] and len(rows)==2 and rows[0]['text']==rows[1]['text']
    assert [row['output_tokens'] for row in rows]==[16,16]
    legacy_scope='current internal-packed-store candidate DLL using installed model programs' if legacy_tag.endswith('internal-stores') else 'unchanged activation-storage candidate DLL using installed model programs' if legacy_tag.endswith('packaged-programs') else 'activation-storage candidate' if legacy_tag.endswith('activation-arena') else 'joined-candidate' if legacy_tag.endswith('joined-decode') else 'pushed checkpoint'
    lines+=['## Qwen3-VL regression smoke','',
        f'Qwen3-VL-2B-Instruct starts on the {legacy_scope} native DLL and completes two matching 16-token requests with 77 visual tokens each. '
        'The existing regression cache folder is reused. This is a legacy serving/reset/vision smoke test, '
        'not broad independent quality, historical input-equivalence proof or matched speed acceptance. '
        'The first harness attempt passed a program file where the API requires an xModel directory; its failure is retained. '
        'The corrected argument and full startup diagnostics preserve the readiness/output assertions. '
        'Evidence: '+link(legacy_name)+', '+link('qwen3-vl-startup-harness-correction.json')+'.','']
compact_fixture=read('compact-kv-fixture-matrix-v1-summary.json')
if compact_fixture:
    compact_terminal=read('compact-kv-fixture-matrix-v1-controller-exit.json')
    assert compact_terminal['exit_code']==0 and compact_fixture['all_16_fixture_profiles_passed']
    assert compact_fixture['six_min_max_format_sanitizer_cases_passed']
    compact_transactions=read('compact-kv-transaction-v1-summary.json')
    transaction_note='Speculative-prefix/session and exact error-path admission remain pending.'
    if compact_transactions:
        assert compact_transactions['garnet_dll_sha256']==compact_fixture['garnet_dll_sha256']
        assert compact_transactions['all_compact_verifier_session_profiles_passed']
        assert compact_transactions['ten_selected_unfiltered_sanitizer_cases_passed']
        transaction_note='The26 selected plain transaction/error profiles and10 selected sanitizer cases also passed.'
    lines[2:2]=['<!-- compact-kv-integration-status -->',
        '> Development candidate `'+compact_fixture['garnet_dll_sha256'][:8]+'`: compactKVOutputs is '
        'connected behind an OFF-by-default CUDA profile. All16 ordinary-decoder fixture profiles '
        '(B1/2/4/8, C384/512, both output layouts) passed complete finite state/hidden/prediction '
        'comparisons and guards. Six selected min/max-format sanitizer cases passed with zero '
        'reported errors, warnings or hazards. '+transaction_note+' These fixture checks do not '
        'by themselves establish pretrained quality, real-model latency or process memory. '
        'The earlier DCD0 tables are historical; current compact measurements are reported '
        'separately when complete. Evidence: '+link('compact-kv-fixture-matrix-v1-summary.json')+'.','']
compact_pretrained=read('compact-kv-pretrained-v1-summary.json')
if compact_pretrained:
    compact_real_terminal=read('compact-kv-real-validation-v1-controller-exit.json')
    assert compact_real_terminal['exit_code']==0
    assert compact_pretrained['all_four_pretrained_controls_passed']
    assert compact_pretrained['garnet_dll_sha256']==compact_fixture['garnet_dll_sha256']
    lines[2:2]=['<!-- compact-kv-pretrained-status -->',
        '> Compact CUDA real-model controls passed for2B batch1 and4B batch8, plain and '
        'unfiltered memcheck with zero reported errors. Each case resets and repeats two native '
        'MTP requests before loading a separate noncompact ordinary target. Complete finite '
        'canonical-state/hidden hashes match both current ordinary and saved prior controls; '
        'inputs, tokens, stops and discrete speculative work are preserved. This covers one '
        'real-image group per model. Full28 paired latency and separate Windows process '
        'memory measurements have their own evidence sections below. No independent quality or2x claim follows. Evidence: '
        +link('compact-kv-pretrained-v1-summary.json')+'.','']
timing=[]
for size,batch in (('2b',1),('4b',8)):
    diagnostic=read(f'request-retirement-{size}-b{batch}-v1-summary.json')
    if diagnostic:
        assert diagnostic['all28_input_output_stop_work_profile_controls_passed']
        assert diagnostic['primary_performance_eligible'] is False
        timing.append((size,batch,diagnostic))
if timing:
    note=['<!-- api-retirement-timing-status -->',
        '> Separate GPU-retirement diagnostic, not primary timing: saved tables measure '
        '`Model.infer_batch` API-return wall. CUDA barriers also include host/barrier/context '
        'overhead and do not isolate final MTP correction. No direct comparison with '
        'uninstrumented HTTP baselines follows.','',
        '| Diagnostic model / batch | API-return median ms | Post-return retirement median ms | Fully retired median ms |',
        '|---|---:|---:|---:|']
    for size,batch,diagnostic in timing:
        values=diagnostic['warm_milliseconds']
        note.append(f'| {size.upper()} B{batch} | {values["api_return_wall_seconds"]["median"]:.3f} | '
            f'{values["post_return_gpu_retirement_seconds"]["median"]:.3f} | '
            f'{values["fully_retired_wall_seconds"]["median"]:.3f} |')
    note.append('')
else:
    note=['<!-- api-retirement-timing-status -->',
        '> Timing boundary under investigation: saved Garnet request latency measures '
        '`Model.infer_batch` API-return wall. Source inspection shows final MTP cache correction '
        'can enqueue GPU work without a subsequent CPU observation before return. A separate '
        'full-cohort diagnostic will measure any post-return GPU retirement; its effect has not '
        'yet been measured. A diagnostic tail includes barrier/host/context overhead rather than '
        'isolating MTP cost. Competitor HTTP timing boundaries also differ.','']
lines[2:2]=note
recovery=read('source-recovery-v1-verification.json')
if recovery:
    assert recovery['all_registered_sources_and_runtime_exact']
    original=read('compact-kv-original-regressions-v3-summary.json')
    progress=('All53 original regressions and both legacy smokes passed: six verified v2 cases were preserved and47 ran under v3, with prior registered source, inputs and runtime unchanged.'
              if original and original.get('all_53_original_default_regressions_passed') else
              'The interrupted original regression runs are preserved; the v3 resume remains pending.')
    lines[2:2]=['<!-- isolated-source-recovery-status -->',
        '> Source recovery: the primary checkout was replaced during the original regression run. '
        'This task now uses the stable worktree `D:/CantorAI/work/qwen35-cuda/Garnet`, on '
        '`feature/gpt-oss-120b`. All274 registered source files and the native/SDK binaries matched '
        'the pre-replacement snapshot before path-only recovery edits. The primary replacement '
        'checkout is preserved. '+progress+' Evidence: '+link('source-recovery-v1-verification.json')+'.','']
compact_cases=[]
for size,batch,study_name,observer in (
    ('2b',1,'compact-kv-retry-v2','compact-kv-2b-retry-v2'),
    ('4b',8,'compact-kv','compact-kv-observed-study-v1')):
    comparison=read(f'packed-study-{size}-b{batch}-compact-kv-all-baselines.json')
    memory=read(f'compact-kv-memory-{size}-b{batch}-pair-summary.json')
    measured=read(f'packed-study-{size}-b{batch}-{study_name}-summary.json')
    observation=read(f'{observer}-{size}-b{batch}-process-audit.json')
    if comparison and memory and measured and observation:
        assert observation['all_four_roles_covered'] and observation['known_overlap_count']==0
        assert not observation['observer_metadata_errors']
        assert comparison['garnet_dll_sha256']==memory['garnet_dll_sha256']==measured['garnet_dll_sha256']
        compact_cases.append((size,batch,comparison,memory,measured))
if len(compact_cases)==2:
    memory_parent=read('compact-kv-observed-memory-v2-exit.json')
    memory_observation=read('compact-kv-observed-memory-v2-process-audit.json')
    assert memory_parent['exit_code']==0 and memory_observation['all_four_cases_covered']
    assert memory_observation['known_overlap_count']==0 and not memory_observation['observer_metadata_errors']
    section=['## Current compact CUDA measurements — DLL5EAD','',
        'The2B result uses the complete clean retry; the original2B attempt remains provisional '
        'because an external XLang test overlapped its first candidate. The original4B roles have '
        'their own zero-overlap audit and are retained. Every selected role covers28 real-image '
        'requests; all20 warm groups are retained in the within-Garnet study. Model/kernel/SDK '
        'and benchmark code stayed fixed across the controller/reporting changes.','',
        '| Model / batch | All20 ordinary ms | All20 compact ms | Ordinary / compact |',
        '|---|---:|---:|---:|']
    for size,batch,comparison,memory,measured in compact_cases:
        values=measured['aggregate_group_median_api_return_ms']
        section.append(f'| {size.upper()} B{batch} | {values["a"]:.3f} | {values["b"]:.3f} | {values["a"]/values["b"]:.3f}x |')
    section+=['',
        'These are API-return medians of two repeats per group followed by the group median. '
        'Two repeats do not establish statistical significance. Separate retirement diagnostics '
        'measure median tails of0.024ms for2B and0.031ms for4B, including barrier/host overhead.','',
        'For external comparisons, the common subset requires every engine to sample32tokens '
        'per row. The historical LM Studio and vLLM cohorts were not rerun. APIs, intermediate '
        'precision, cache formats and speculative settings differ; exact cross-engine numerical '
        'parity is not established.','',
        '| Model / batch | Common groups | Garnet ms | LM Studio ms | Fastest saved vLLM ms | vLLM / Garnet | Derived decode tok/s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for size,batch,comparison,memory,measured in compact_cases:
        values=comparison['median_group_ms'];fastest=comparison['fastest_tested_baseline']
        rate=comparison['garnet_common_group_derived_post_first_sample_tokens_per_second']['garnet_b']
        section.append(f'| {size.upper()} B{batch} | {comparison["common_equal32_groups"]} | '
            f'{values["garnet_b"]:.3f} | {values["lmstudio"]:.3f} | {values[fastest]:.3f} | '
            f'{values[fastest]/values["garnet_b"]:.3f}x | {rate:.1f} |')
    section+=['',
        'Derived decode throughput subtracts the first sample per row and includes host/stop '
        'overhead;4B throughput is aggregated across8rows. It is separate from full-request '
        'throughput. The2x fastest-baseline thresholds remain101.637ms and333.789ms, unmet.','',
        '| Model / batch | Ordinary committed GiB | Compact committed GiB | Reduction MiB | Compact shared GiB | After close GiB |',
        '|---|---:|---:|---:|---:|---:|']
    for size,batch,comparison,memory,measured in compact_cases:
        old=memory['profiles']['control'];new=memory['profiles']['candidate']
        after=new['all_phase_sampled_statistics']['after_model_close']['bytes']['TotalCommitted']['median']
        section.append(f'| {size.upper()} B{batch} | {old["warm_process_counter_bytes"]["TotalCommitted"]/2**30:.6f} | '
            f'{new["warm_process_counter_bytes"]["TotalCommitted"]/2**30:.6f} | '
            f'{memory["control_minus_candidate_counter_bytes"]["TotalCommitted"]/2**20:.1f} | '
            f'{new["warm_process_counter_bytes"]["SharedUsage"]/2**30:.6f} | {after/2**30:.6f} |')
    section+=['',
        'Memory sessions use fresh sequential processes and six exact input/output/stop/profile '
        'groups each. Dedicated/shared/total Windows counters are distinct from retained owned '
        'allocations. All four sessions have sampled observation coverage with no known named '
        'model/compiler/diagnostic overlap. Largest observations are not true peaks or physical '
        'residency; vLLM total process VRAM remains unavailable. The owned reductions are20.65MiB '
        'and254MiB, including new helper/metadata costs. The first memory harness attempt failed '
        'before model loading because of a reused PowerShell variable; its raw failure is retained.','',
        'Latency chart: '+link('progress-compact-kv-baselines.png')+'. Memory chart: '
        +link('progress-compact-kv-memory.png')+'.','']
    for size,batch,comparison,memory,measured in compact_cases:
        section.append('Evidence '+size.upper()+': '+link(comparison['study_summary'])+', '
            +link(f'packed-study-{size}-b{batch}-compact-kv-all-baselines.json')+', '
            +link(f'compact-kv-memory-{size}-b{batch}-pair-summary.json')+'.')
    section+=['']
    lines[2:2]=section
surveys=[(size,read(f'lossless-bf16-block-survey-{size}-v1.json')) for size in ('2b','4b')]
if all(data for _,data in surveys):
    lines+=['## Next experiment — lossless BF16 block storage survey','',
        'A read-only CPU survey ran sequentially after all model processes closed. It examines '
        'only stored BF16 text-projection matrices, counted once. Each256-word block uses a '
        '12-bit sign/fraction/exponent-delta record, an8-byte header and indexed full-word '
        'exceptions; a raw fallback is available. This is a storage-format estimate, not '
        'lower-precision quantization or an implemented GPU optimization.','',
        '| Model | Matrices | Source projection GiB | Estimated format GiB | Estimated saving MiB | Raw-fallback blocks |',
        '|---|---:|---:|---:|---:|---:|']
    for size,data in surveys:
        assert data['all_nonexception_bf16_words_algebraically_reconstructed']
        lines.append(f'| {size.upper()} | {len(data["records"])} | '
            f'{data["projection_source_bytes"]/2**30:.3f} | {data["estimated_format_bytes"]/2**30:.3f} | '
            f'{data["estimated_stored_bytes_saved"]/2**20:.1f} | {sum(r["raw_blocks"] for r in data["records"])} |')
    lines+=['',
        'All nonexception source words were algebraically reconstructed exactly; exceptions '
        'retain original words in the proposed format. Matrix-directory/alignment costs, '
        'embedding/head, vision, MTP and nonmatrix parameters are excluded. The survey alone '
        'does not establish byte-codec/GPU-decoder correctness or speed. Separate component '
        'results are reported below when present. Actual model ownership, workspace/cache '
        'effects, outputs and request latency remain unmeasured for compressed weights. '
        'Sources: '+link('lossless-bf16-block-survey-2b-v1.json')+', '
        +link('lossless-bf16-block-survey-4b-v1.json')+'.','']
storage_paths=read('lossless-bf16-storage-paths-v1-rejection-summary.json')
if storage_paths:
    lines+=['## Lossless-storage GPU paths — both current prototypes rejected','',
        f'The CPU physical-byte codec passes{storage_paths["cpu_byte_codec_checks_passed"]} checks: '
        'independent golden bytes, every BF16 bit pattern, tails, deterministic immutable-input '
        'round trips and malformed streams. Learned2B/4B joined-input and MLP-down matrices '
        'also round-trip every byte exactly.','',
        'The first standalone GPU attempt stopped on an upload-ordering gap between pageable '
        'copies and a nonblocking test stream. Its raw failure is retained. Explicit uploads '
        'on the test stream fixed the corrected run; no decoder arithmetic was changed. '
        'That run completes10 exact GPU decoder cases and32 plain matrix/row profiles, with '
        'all BF16 projection outputs equal to the unchanged production GEMM. Unfiltered '
        'memcheck covers12 profiles; race/sync cover all10 decoder-only variants. All report '
        'zero errors/hazards and intact guards.','',
        '| Staging variant | 2B selected cold ordinary / candidate | 4B selected cold ordinary / candidate | Required |',
        '|---|---:|---:|---:|']
    for variant in storage_paths['staging_decode_subset_scores']:
        ratios={m['model']:m['ratio'] for m in variant['models']}
        lines.append(f'| {variant["variant"]} | {ratios["2b"]:.3f}x | {ratios["4b"]:.3f}x | >=1.100x with no selected-op regression |')
    failure=storage_paths['fused_first_failure']
    lines+=['',
        'Both staging variants are rejected for speed. The format saves stored bytes, but '
        'materializing a full global BF16 matrix before GEMM adds work and temporary storage. '
        'These are component GPU-event measurements, not model latency or process-memory savings. '
        'The record includes cold/warm samples plus excluded eviction costs and totals.','',
        'A separate fused shared-tile decoder/MMA avoids the global staging matrix. It matches '
        f'production BF16 outputs on{storage_paths["fused_complete_profiles_before_failure"]} completed '
        f'profiles, then fails at{failure["fixture"]}, rows{failure["rows"]}: '
        f'{failure["different_bf16_words"]} BF16 output words differ while guards remain intact. '
        'Its completed small-row decode timings are also slower. That plain gate stops the '
        'remaining profiles and sanitizer dispatch; no4B result or complete fused acceptance '
        'is claimed. The prototype is rejected without weakening its equality gate.','',
        'Garnet production model/kernel/SDK behavior, DLL5EAD and the measured compact-KV '
        'latency/memory charts are unchanged. Neither compressed-weight path is integrated. '
        'Next revisions require a materially different measured approach and full precision, '
        'lifecycle and request-level evidence before any promotion. Evidence: '
        +link('lossless-bf16-storage-paths-v1-rejection-summary.json')+', '
        +link('lossless-bf16-projection-v2-summary.json')+', '
        +link('lossless-bf16-fused-v1-plain.jsonl')+'.','']
out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(lines),encoding='utf-8');print(out)
