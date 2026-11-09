"""Audit prepared-weight operator timings, including explicit cache pressure."""
from pathlib import Path
import collections,copy,hashlib,itertools,json,math,statistics,sys
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'timing-audit1.json'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def need(v,m):
    if not v:raise ValueError(m)
need(not DEST.exists(),'Never overwrite an audit')
correct=read(ROOT/'correctness-audit1.json');normal=read(ROOT/'normal-preparation-audit1.json')
need(correct['passed'] and normal['passed'] and normal['normal_and_counted_half_bits_exact'],'missing numerical admission')
baseline=read(ROOT/'correctness1/result1.json')['pipeline_resources']
need(len(baseline)==54,'wrong qualified pipeline count')
preflights={}
for prefix in ['timing','pressure']:
    pre=read(ROOT/(prefix+'1-preflight.json'));post=read(ROOT/(prefix+'1-postflight.json'))
    need(pre['passed'] and post['passed'] and post['all_gpu_processes_terminal'],'incomplete flight')
    need(pre['source_model_hashing_during_collection'] is False and pre['no_comparator_execution'],'wrong collection policy')
    need(pre['measured_rounds_per_mode_cell']==20 and pre['warmups_per_mode_cell']==5,'wrong collection plan')
    need(post['checked_files']==len(pre['files_sha256']),'wrong postflight count')
    for name,h in pre['files_sha256'].items():need(sha(Path(name))==h,'changed collection input '+name)
    preflights[prefix]=sha(ROOT/(prefix+'1-preflight.json'))
derivation=read(ROOT/'pressure-derivation1.json')
need(sha(ROOT/'combined1.metal')==correct['combined_sha256']==normal['qualified_shader_sha256']==derivation['qualified_shader_prefix_sha256'],'unqualified shader prefix')
need(sha(ROOT/'pressure1.metal')==derivation['new_shader_sha256'] and (ROOT/'pressure1.metal').read_bytes().startswith((ROOT/'combined1.metal').read_bytes()),'changed pressure shader')
host=(ROOT/'screen1.mm').read_text()
need(sha(ROOT/'screen1.mm')==derivation['original_host_sha256'],'changed original host')
for change in derivation['host_changes']:
    need(host.count(change['old'])==1,'ambiguous host derivation');host=host.replace(change['old'],change['new'])
need(host==(ROOT/'pressure1.mm').read_text() and sha(ROOT/'pressure1.mm')==derivation['pressure_host_sha256'],'unrecorded pressure host change')
SHAPES=['gate','down','gate-prefix-proxy'];DIMS={'gate':(4096,14336),'down':(14336,4096),'gate-prefix-proxy':(4096,4096)}
expected=[]
for shape,rows,sparse,round_,at in itertools.product(SHAPES,[128,256,512],[False,True],range(20),range(5)):
    index=(at+(round_+5)%5)%5
    if (round_+5)&1:index=4-index
    expected.append((shape,rows,sparse,round_,at,index-1))
def clock(x,start,end):
    a,b=x[start],x[end];need(all(type(v) in (float,int) and math.isfinite(v) for v in [a,b]) and 0<a<b,'missing/reversed GPU clock');return a,b
def verify(d,wall_ns,pressure):
    need(d['completed'] and d['timing'] and d['cases_count']==0 and d['samples_count']==1800,'wrong timing mode/count')
    need(not d['cases'] and not d['fp64'] and d['complete_parent_word_comparisons']==d['writer_word_comparisons']==0,'timing performed correctness observations')
    need(d['no_runtime_integration'] and not d['serving_speed_claim'] and not d['goal_complete'] and not d['exact_native_captured_controls'],'wrong claim scope')
    need(d['pipeline_resources']==baseline,'different qualified pipelines/resources')
    samples=d['samples'];actual=[(e['shape'],e['rows'],e['sparse'],e['round'],e['position'],e['mode']) for e in samples]
    need(actual==expected,'missing/duplicate/reordered cell/round/mode')
    prepared=d['prepared_weights'];need(len(prepared)==6,'missing prepared layout')
    positions=collections.defaultdict(collections.Counter);previous=0;total_gpu=0;last_shape_end=0;preparation=[]
    for si,shape in enumerate(SHAPES):
        k,n=DIMS[shape]
        for layout in range(2):
            p=prepared[2*si+layout];elements=n*k if layout==0 else n*k//128*136
            need((p['layer'],p['shape'],p['inner'],p['columns'],p['layout'],p['elements'])==(15,shape,k,n,layout,elements),'wrong prepared weight dimensions')
            need(p['payload_bytes']==elements*2 and p['allocation_bytes']==elements*2+128 and p['counted'] is False,'wrong cache bytes or instrumented preparation')
            need(not any(key in p for key in ['half_sha256','writers_sha256','raw_half']),'hashing during timing collection')
            a,b=clock(p,'gpu_start_seconds','gpu_end_seconds')
            need(a>=previous,'overlapping preparation or operator commands');previous=b;total_gpu+=(b-a)*1000
            need(math.isclose(p['gpu_ms'],(b-a)*1000,rel_tol=1e-12,abs_tol=1e-9),'preparation clock/duration disagreement')
            need(math.isfinite(p['allocation_encode_wait_wall_ms']) and p['allocation_encode_wait_wall_ms']>=p['gpu_ms'],'preparation exceeds measured wall interval')
            preparation.append(dict(shape=shape,layout=layout,payload_bytes=elements*2,requested_buffer_bytes=elements*2+128,gpu_ms=p['gpu_ms'],allocation_encode_wait_wall_ms=p['allocation_encode_wait_wall_ms']))
        for e in samples[si*600:(si+1)*600]:
            need(e['layer']==15 and type(e['sparse']) is bool and math.isfinite(e['gpu_ms']) and e['gpu_ms']>0,'wrong/nonfinite duration or layer')
            rows=e['rows'];tile=128 if shape=='gate-prefix-proxy' else (256 if rows>=(384 if shape=='down' else 256) else 128)
            need(e['tile']==tile,'different selected native tile')
            positions[(shape,rows,e['sparse'],e['mode'])][e['position']]+=1
            a,b=clock(e,'gpu_start_seconds','gpu_end_seconds')
            need(math.isclose(e['gpu_ms'],(b-a)*1000,rel_tol=1e-12,abs_tol=1e-9),'operator clock/duration disagreement')
            if pressure:
                sa,sb=clock(e,'sweep_gpu_start_seconds','sweep_gpu_end_seconds')
                need(e['sweep_bytes_read']==e['sweep_bytes_written']==256*1024*1024,'wrong pressure volume')
                need(sa>=previous and a>=sb,'pressure overlaps prior/current operator interval')
                total_gpu+=(sb-sa)*1000
            else:
                need(a>=previous and not any(key.startswith('sweep_') for key in e),'overlapping/unplanned command')
            previous=b;total_gpu+=e['gpu_ms'];last_shape_end=b
    need(len(positions)==90 and all(c==collections.Counter({i:4 for i in range(5)}) for c in positions.values()),'sample positions not balanced')
    need(total_gpu<=wall_ns/1e6,'recorded GPU work exceeds serial process wall time')
    result={}
    for shape,rows,sparse in itertools.product(SHAPES,[128,256,512],[False,True]):
        cell=[e for e in samples if (e['shape'],e['rows'],e['sparse'])==(shape,rows,sparse)]
        values={mode:[e['gpu_ms'] for e in cell if e['mode']==mode] for mode in range(-1,4)}
        med={m:statistics.median(v) for m,v in values.items()};rounds={(e['round'],e['mode']):e['gpu_ms'] for e in cell}
        result[shape+'-r'+str(rows)+'-'+('permuted' if sparse else 'contiguous')]=dict(shape=shape,rows=rows,sparse=sparse,median_gpu_ms={str(m):med[m] for m in med},range_gpu_ms={str(m):[min(v),max(v)] for m,v in values.items()},control_over_candidate={str(m):med[-1]/med[m] for m in range(4)},paired_round_ratio_median={str(m):statistics.median(rounds[(r,-1)]/rounds[(r,m)] for r in range(20)) for m in range(4)})
    return dict(cells=result,one_time_preparations=preparation,retained_gpu_interval_ms=total_gpu,process_wall_ms=wall_ns/1e6)
data={};statuses={};results={};all_status=[]
for prefix in ['timing','pressure']:
    results[prefix]={}
    for co in ['a','b']:
        name=prefix+'-'+co+'1';s=read(ROOT/(name+'-status.json'));statuses[name]=s;all_status.append(s)
        need(s['status']=='terminal' and s['returncode']==0 and 0<s['started_ns']<s['finished_ns'],'unfinished GPU process')
        cmd=read(ROOT/(name+'-command.json'));binary='pressure1' if prefix=='pressure' else 'screen1';shader='pressure1.metal' if prefix=='pressure' else 'combined1.metal'
        need(cmd==[str(ROOT/binary),str(ROOT/shader),str(ROOT.parent/'mistral-fused-mlp-trained1/assets1'),str(ROOT.parent/'mistral-fused-mlp-trained1/captures1'),str(ROOT/name),'timing'],'wrong executed command')
        data[name]=read(ROOT/name/'result1.json');results[prefix][co]=verify(data[name],s['finished_ns']-s['started_ns'],prefix=='pressure')
need(len({s['pid'] for s in all_status})==4 and all(b['started_ns']>=a['finished_ns'] for a,b in zip(all_status,all_status[1:])),'processes not fresh and serial')
rejected=[]
mutations={
 'missing_sample':lambda x:x['samples'].pop(),
 'duplicate_sample':lambda x:x['samples'].__setitem__(1,copy.deepcopy(x['samples'][0])),
 'wrong_tile':lambda x:x['samples'][0].update(tile=64),
 'wrong_order':lambda x:x['samples'][0].update(position=4),
 'nonfinite_duration':lambda x:x['samples'][0].update(gpu_ms=float('nan')),
 'negative_duration':lambda x:x['samples'][0].update(gpu_ms=-1),
 'duration_not_from_clocks':lambda x:x['samples'][0].update(gpu_ms=x['samples'][0]['gpu_ms']*1.01),
 'reversed_operator_clock':lambda x:x['samples'][0].update(gpu_start_seconds=x['samples'][0]['gpu_end_seconds']),
 'overlapping_operator_clock':lambda x:x['samples'][1].update(gpu_start_seconds=x['samples'][0]['gpu_start_seconds']),
 'wrong_qualified_resources':lambda x:x['pipeline_resources'][0].update(shared_bytes=0),
 'counted_preparation':lambda x:x['prepared_weights'][0].update(counted=True),
 'wrong_preparation_bytes':lambda x:x['prepared_weights'][0].update(payload_bytes=0),
 'missing_preparation':lambda x:x['prepared_weights'].pop(),
 'preparation_clock_mismatch':lambda x:x['prepared_weights'][0].update(gpu_ms=1),
 'pressure_inside_operator':lambda x:x['samples'][0].update(sweep_gpu_end_seconds=x['samples'][0]['gpu_end_seconds']),
 'wrong_pressure_bytes':lambda x:x['samples'][0].update(sweep_bytes_read=0),
 'reversed_pressure_clock':lambda x:x['samples'][0].update(sweep_gpu_start_seconds=x['samples'][0]['sweep_gpu_end_seconds']),
 'timing_half_hash_observation':lambda x:x['prepared_weights'][0].update(half_sha256='0'*64),
}
for name,mutate in mutations.items():
    d=copy.deepcopy(data['pressure-a1']);mutate(d);s=statuses['pressure-a1']
    try:verify(d,s['finished_ns']-s['started_ns'],True)
    except (ValueError,KeyError,TypeError,IndexError):rejected.append(name)
    else:raise ValueError('Corruption admitted: '+name)
largest={prefix:{shape:{str(mode):[results[prefix][co]['cells'][shape+'-r512-contiguous']['control_over_candidate'][str(mode)] for co in ['a','b']] for mode in range(4)} for shape in SHAPES} for prefix in results}
result=dict(passed=True,scope=__doc__,cohorts=results,largest_contiguous_control_over_candidate=largest,measured_samples=7200,warmup_samples=1800,measured_rounds_per_mode_cell_process=20,each_mode_sample_positions={str(i):4 for i in range(5)},negative_controls_rejected=rejected,
 mode_names={'-1':'unaltered native inline-Q4 kernel','0':'copied inline-Q4 kernel with disabled writer observer','1':'prepared row-major half weights, original K16 schedule','2':'prepared padded tiled half weights, original K16 schedule','3':'prepared padded tiled half weights, vector/K32 schedule'},
 numerical_gate_sha256=sha(ROOT/'correctness-audit1.json'),normal_preparation_gate_sha256=sha(ROOT/'normal-preparation-audit1.json'),sample_result_sha256={name:sha(ROOT/name/'result1.json') for name in data},preflight_sha256=preflights,
 selection_changed=False,timings_are_operator_gpu_durations_only=True,absolute_gpu_start_end_timestamps_retained=True,pressure_sweep_outside_operator_interval=True,pressure_is_not_proof_of_all_hardware_caches_cold=True,reported_allocation_bytes_are_requested_buffer_lengths=True,full_model_cache_budget_qualified=False,proxy_is_actual_QO=False,baseline_per_operator_kernel_selection_proven=False,native_runtime_changed=False,serving_speed_claim=False,goal_complete=False)
DEST.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='cohorts'},indent=2))
