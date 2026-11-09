"""Saved timing audit across transient, packed-half and GPU-KV-pack variants."""
from pathlib import Path
import copy,hashlib,json,math,statistics,sys
p=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
load=lambda q:json.loads(q.read_text())
def sha(q):
    h=hashlib.sha256()
    with q.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def audit(r,reverse,packed,kv_pack):
    assert r['collected'] and r['actual_trained_operands'] and not r['half_rounded_input_fixture']
    assert r['reverse'] is reverse and r['warmups_per_cell_variant']==3 and r['measured_rounds']==9
    assert r['all_prepared_before_measurement'] and r['math_mode']=='Safe' and r['relaxed_precision'] is packed
    assert r.get('packed_half_storage',False) is packed and r.get('Q_pack_inside_gpu_span',False) is packed
    assert r.get('KV_pack_inside_gpu_span',False) is kv_pack
    if packed:assert r['half_KV_fixture_prepared_before_timing'] is not kv_pack
    if kv_pack:assert r['half_scratch_poisoned_before_each_candidate']
    assert not r['native_integration'] and not r['serving_speed_claim'] and not r['quality_qualified'] and not r['goal_complete']
    assert r['half_policy_layout_capacities']==[8,16,16,8,16,16]*32 and r['register_layout_status']==[0]*32
    assert r['half_policy_coordinate_parent_words']==[0x7fc12345]*16+[1]*2560+[0]+[0x7fc12345]*16
    assert r['qkv_scale']==1 and r['probability_scale']==32768
    assert r['variant_names']==['original4mps','committed2mps','fp16_qkv_p16_v1','fp16_qkv_p22_v1']
    assert len(r['cases'])==3
    result=[]
    for ci,c in enumerate(r['cases']):
        assert [c[k] for k in ['batch','chunk','capacity','extent','gated']]==[4,128,4096,2048,False]
        assert c['first']==[1920]*4 and c['counts']==[128]*4
        assert c['params']==[4,128,32,8,128,4096,4100,1028,1028,4194304,4194304,131,133,4100,4100,0]
        assert c['checked_parent_words']==13*(c['readonly_parent_words']+c['output_parent_words'])
        assert len(c['samples'])==36
        times={v:[] for v in range(4)};pairs={v:[] for v in [2,3]}
        for rnd in range(9):
            values={}
            for order in range(4):
                s=c['samples'][rnd*4+order];slot=(rnd+order)%4;variant=3-slot if reverse else slot
                assert [s[k] for k in ['round','order','variant']]==[rnd,order,variant]
                assert all(math.isfinite(s[k]) and s[k]>0 for k in ['gpu_ms','wall_ms'])
                times[variant].append(s['gpu_ms']);values[variant]=s['gpu_ms']
            for v in [2,3]:pairs[v].append(values[1]/values[v])
        result.append(dict(layer=[0,15,31][ci],gpu_median_ms={str(v):statistics.median(x) for v,x in times.items()},gpu_range_ms={str(v):[min(x),max(x)] for v,x in times.items()},paired_two_MPS_over_candidate={str(v):statistics.median(x) for v,x in pairs.items()}))
    return result

assert load(p/'numeric-audit1.json')['passed'] and load(p/'numeric-audit2.json')['passed'] and load(p/'numeric-audit3.json')['passed']
cohorts=[]
for mode,indices,preflight,packed,kv_pack in [('transient',[1,2],2,False,False),('packed',[4,5],4,True,False),('gpu_kv_pack',[6,7],5,True,True)]:
    before=load(p/f'screen-preflight{preflight}.json')
    assert before['passed'] and before['no_compilation_or_hashing_during_timing']
    assert load(p/f'screen-postflight{preflight}.json')['passed']
    for q,h in before['files_sha256'].items():assert sha(Path(q))==h,q
    for order,n in enumerate(indices):
        assert load(p/f'trained-screen{n}-status.json')['returncode']==0
        cohorts.append(dict(mode=mode,cohort=n,cases=audit(load(p/f'trained-screen{n}/screen1.json'),order==1,packed,kv_pack)))
assert load(p/'trained-screen3-status.json')['returncode']==1
assert not (p/'trained-screen3/screen1.json').exists()
assert (p/'trained-screen3.stderr').read_text().strip()=='Missing Metal function'
negative_results={};base=load(p/'trained-screen6/screen1.json')
def negative(name,change):
    r=copy.deepcopy(base);change(r)
    try:audit(r,False,True,True)
    except (AssertionError,KeyError,IndexError,TypeError,ValueError):negative_results[name]=True
    else:raise AssertionError('Accepted timing corruption: '+name)
negative('missing_sample',lambda r:r['cases'][0]['samples'].pop())
negative('repeated_round',lambda r:r['cases'][0]['samples'][4].update(round=0))
negative('wrong_variant',lambda r:r['cases'][0]['samples'][0].update(variant=3))
negative('negative_time',lambda r:r['cases'][0]['samples'][0].update(gpu_ms=-1))
negative('nonfinite_time',lambda r:r['cases'][0]['samples'][0].update(gpu_ms=float('nan')))
negative('changed_descriptor',lambda r:r.update(relaxed_precision=False))
negative('false_Q_pack',lambda r:r.update(Q_pack_inside_gpu_span=False))
negative('false_KV_pack',lambda r:r.update(KV_pack_inside_gpu_span=False))
negative('missing_poison',lambda r:r.update(half_scratch_poisoned_before_each_candidate=False))
negative('changed_layout',lambda r:r['register_layout_status'].__setitem__(0,1))
negative('wrong_capacity',lambda r:r['half_policy_layout_capacities'].__setitem__(0,9))
negative('wrong_coordinate',lambda r:r['half_policy_coordinate_parent_words'].__setitem__(16,2))
negative('wrong_extent',lambda r:r['cases'][0].update(extent=4096))
negative('false_serving_claim',lambda r:r.update(serving_speed_claim=True))
negative('skipped_warmup',lambda r:r.update(warmups_per_cell_variant=0))
equal_pairs=0
for kind in ['synthetic','trained']:
    a=p/f'raw-{kind}1';b=p/f'raw-{kind}2'
    for q in a.glob('*-output[23].f32'):
        assert sha(q)==sha(b/q.name);equal_pairs+=q.stat().st_size//4
result=dict(passed=True,scope=__doc__,cohorts=cohorts,measured_command_buffers=648,warmup_attention_command_buffers=216,corruptions_rejected=len(negative_results),corruption_controls=negative_results,failed_stale_shader_attempt_retained=True,strict_and_packed_candidate_exact_output_parent_word_pairs=equal_pairs,interpretation='Ratios are medians of per-round two-MPS/candidate ratios; >1 favors candidate. Packed timing excludes prior KV conversion. GPU-KV-pack timing includes Q and KV conversion. These are component observations under distinct FP16 numerical policies, not full-model or serving qualification.',native_integration=False,quality_qualified=False,serving_speed_claim=False,goal_complete=False)
(p/'screens-audit1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':True,'measured_command_buffers':648,'corruptions_rejected':len(negative_results),'strict_packed_exact_output_words':equal_pairs},indent=2))
