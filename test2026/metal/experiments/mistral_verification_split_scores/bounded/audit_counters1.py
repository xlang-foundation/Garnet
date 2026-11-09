"""Audit current-target calibrated encoder regions with preserved batch boundaries."""
from pathlib import Path
import collections,copy,hashlib,json,math,sys
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'counter-audit1.json'
M=ROOT.parent;S=M/'mistral-fused-q4-decode-coder1'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def need(v,m):
    if not v:raise ValueError(m)
def union(intervals,scale):
    total=0;end=None
    for a,b in sorted(intervals):
        if end is None or a>end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total*scale
def command(e):
    need(e['diagnostic_only'] is True and e['mps_encoders_observed'] is False,'wrong counter scope')
    need(0<e['cpu0']<e['cpu1'] and 0<e['gpu0']<e['gpu1'],'invalid clock calibration')
    need(math.isfinite(e['gpu_ms']) and e['gpu_ms']>0 and e['gpu_ms']<=(e['cpu1']-e['cpu0'])/1e6+1,'invalid command GPU time')
    scale=(e['cpu1']-e['cpu0'])/(e['gpu1']-e['gpu0'])/1e6
    need(0<len(e['rows'])<=2048,'empty/overflowed sampled encoder list')
    spans=[];groups=collections.defaultdict(list);counts=collections.Counter();submissions=[]
    for r in e['rows']:
        need(e['gpu0']<=r['start']<=r['end']<=e['gpu1'] and r['start']>0,'encoder outside calibrated interval')
        need(math.isfinite(r['ms']) and r['ms']>=0 and abs((r['end']-r['start'])*scale-r['ms'])<=max(1e-6,r['ms']*1e-7),'wrong encoder duration units')
        need(type(r['command']) is int and 0<=r['command']<=e['partial_submissions'],'invalid command index')
        need(r['site'].startswith(('Encode','TryEncode')) and isinstance(r['kind'],str),'invalid encoder region')
        key=(r['site'],r['kind'],r['op'],r['rows'],r['columns']);span=(r['start'],r['end'])
        spans.append(span);groups[key].append(span);counts[key]+=1;submissions.append(r['command'])
    need(submissions==sorted(submissions),'command indices reordered')
    covered=union(spans,scale);span=(max(b for a,b in spans)-min(a for a,b in spans))*scale
    need(0<covered<=span+1e-6 and span<=e['gpu_ms']+1,'direct encoder span exceeds command')
    return dict(gpu_ms=e['gpu_ms'],direct_union_ms=covered,encoder_sum_ms=sum(r['ms'] for r in e['rows']),groups={k:union(v,scale) for k,v in groups.items()},counts=counts)
def verify_frame(events,stats,q):
    need(len(events)==2,'verification changed target/commit command boundaries')
    for event,key in zip(events,['target_gpu_ms','commit_gpu_ms']):
        need(abs(event['gpu_ms']-stats[key])<1e-4 and event['partial_submissions']==0,'target/commit GPU scope differs')
    need(abs(stats['sequential_gpu_duration_sum_ms']-stats['target_gpu_ms']-stats['commit_gpu_ms'])<1e-7,'wrong sequential GPU sum')
    for k in ['target_affine_dispatches','target_affine_misses','target_attention_dispatches','target_attention_layout_dispatches','target_attention_vector4_dispatches','target_affine_decode_block_dispatches','target_kv_backups','commit_kv_restores','consumed_inputs','emitted_tokens','eos']:
        need(stats[k]==q['metrics'][k],'instrumentation changed logical route/commit '+k)
    a,b=map(command,events);counts=collections.Counter()
    for row in events[0]['rows']:
        if row['kind']=='MatMul':
            need(row['rows']==4 and row['columns']==(14336 if row['op']==23 else 4096),'wrong physical verification projection shape')
            counts[row['op']]+=1
    need(counts==collections.Counter({6:32,7:32,8:32,14:32,20:32,21:32,23:32,1:1}),'missing/extra verification affine encoder')
    indices=[i for i,r in enumerate(events[0]['rows']) if r['kind']=='VerificationAttention']
    need(len(indices)==64 and all(indices[i+1]==indices[i]+1 for i in range(0,64,2)),'score/value pair order differs')
    # Native EncodeVerificationAttention records scores then context in order.
    scale=(events[0]['cpu1']-events[0]['cpu0'])/(events[0]['gpu1']-events[0]['gpu0'])/1e6
    for name,offset in [('score_union_ms',0),('value_union_ms',1)]:
        selected=[events[0]['rows'][indices[i]] for i in range(offset,64,2)]
        a[name]=union([(r['start'],r['end']) for r in selected],scale)
    return a,b
def main():
    need(not DEST.exists(),'never overwrite audit')
    need(read(ROOT/'counters1-status.json')['returncode']==0 and read(ROOT/'counters1-postflight.json')['passed'],'collection unfinished')
    for n,h in read(ROOT/'counters1-preflight.json')['files_sha256'].items():need(sha(Path(n))==h,'changed admitted input '+n)
    data=read(ROOT/'counters1-result.json');saved=read(ROOT/'counter-input1.json')
    need(data['completed'] and data['inputs']==saved['inputs'] and data['library']==saved['library'],'wrong target/profile')
    need(data['prepared']['target_graphs']==224 and len(data['records'])==1,'incomplete prepared profile/cases')
    memory=data['prepared']['memory'];need(memory['metal_allocated_bytes']<=memory['metal_recommended_working_set_bytes']-268435456,'prepared working set exceeded')
    expected=[];frame_rows={}
    for a,b in zip(data['records'],saved['records']):
        for k in ['case_id','prompt_ids','generated_ids','slots','state_sha256','hidden_sha256']:need(a[k]==b[k],'saved IDs/full state differ '+k)
        label=a['case_id'];expected.extend([(label,'reset',-1),(label,'prefill',-1)])
        need(len(a['rounds'])==len(b['rounds']),'changed target call coverage')
        for i,(x,y) in enumerate(zip(a['rounds'],b['rounds'])):
            phase='verify' if y['proposal'] else 'fallback';need(x['tick']==i and x['phase']==phase and x['active_tokens']==len(y['proposal'])+1 and x['consumed']==y['consumed'],'changed replay frame')
            key=(label,phase,i);expected.append(key);frame_rows[key]=(x,y)
        expected.append((label,'witness',-1))
    expected.append(('all','close',-1))
    cleanup=data['cleanup'];need(cleanup['metal_allocated_bytes']<=4194304 and all(cleanup[k]==0 for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']),'native owners retained')
    current=None;events=[];seen=[];groups=collections.defaultdict(float);counts=collections.Counter();target_gpu=target_union=target_sum=commit_gpu=0.;verify=fallback=other_commands=0;sample=None;score_union=value_union=0.
    with (ROOT/'counters1.log').open() as f:
        for line in f:
            if line.startswith('metal_op_profile '):raise ValueError('serialized operation profiling changed execution')
            if line.startswith('verification_gpu_frame '):
                marker=json.loads(line.split(' ',1)[1]);key=(marker['case'],marker['phase'],marker['tick'])
                if marker['event']=='begin':
                    need(current is None,'nested frame');current=key;events=[]
                else:
                    need(marker['event']=='end' and current==key,'unpaired/reordered frame boundary');seen.append(key)
                    if key[1]=='verify':
                        x,q=frame_rows[key];need(marker['stats']==x['stats'],'changed verification metadata');a,b=verify_frame(events,x['stats'],q);verify+=1
                        target_gpu+=a['gpu_ms'];target_union+=a['direct_union_ms'];target_sum+=a['encoder_sum_ms'];commit_gpu+=b['gpu_ms'];score_union+=a['score_union_ms'];value_union+=a['value_union_ms']
                        for k,v in a['groups'].items():groups[k]+=v
                        counts.update(a['counts'])
                        if sample is None:sample=(copy.deepcopy(events),copy.deepcopy(x['stats']),copy.deepcopy(q))
                    elif key[1]=='fallback':
                        x,q=frame_rows[key];need(len(events)==1 and marker['stats']==x['stats'] and abs(events[0]['gpu_ms']-x['stats']['gpu_ms'])<1e-4,'changed fallback GPU scope');command(events[0]);fallback+=1
                    else:
                        for e in events:command(e)
                        other_commands+=len(events)
                    current=None;events=[]
            elif line.startswith('metal_gpu_counters '):
                need(current is not None,'unbound counter command');events.append(json.loads(line.split(' ',1)[1]))
    need(current is None and seen==expected and verify==sum(bool(q['proposal']) for row in saved['records'] for q in row['rounds']) and fallback==sum(not q['proposal'] for row in saved['records'] for q in row['rounds']),'missing/extra diagnostic frames')
    negative=[]
    for name,mut in [('changed_duration',lambda e,s,q:e[0]['rows'][0].update(ms=1000.)),('clock_outside_range',lambda e,s,q:e[0]['rows'][0].update(end=e[0]['gpu1']+1)),('missing_encoder',lambda e,s,q:e[0].update(rows=[r for r in e[0]['rows'] if not(r['kind']=='MatMul' and r['op']==20)])),('wrong_projection_extent',lambda e,s,q:next(r for r in e[0]['rows'] if r['kind']=='MatMul').update(columns=4097)),('false_mps_coverage',lambda e,s,q:e[0].update(mps_encoders_observed=True)),('missing_commit_command',lambda e,s,q:e.pop()),('wrong_commit_scope',lambda e,s,q:s.update(commit_gpu_ms=1000.)),('wrong_accepted_prefix',lambda e,s,q:s['consumed_inputs'].__setitem__(0,0))]:
        e,s,q=copy.deepcopy(sample);mut(e,s,q)
        try:verify_frame(e,s,q)
        except (ValueError,KeyError,TypeError,IndexError):negative.append(name)
        else:raise ValueError('corruption admitted '+name)
    table=[dict(site=k[0],kind=k[1],op=k[2],physical_rows=k[3],input_columns=k[4],encoder_count=counts[k],region_union_ms=v,percent_target_command_gpu=100*v/target_gpu) for k,v in sorted(groups.items(),key=lambda pair:-pair[1])]
    result=dict(passed=True,scope=__doc__,cases=1,output_tokens=sum(len(r['generated_ids']) for r in data['records']),all_output_ids_full_kv_hidden_exact_saved_current_target=True,verification_frames=verify,ordinary_fallback_frames=fallback,target_gpu_ms=target_gpu,verification_score_union_ms=score_union,verification_value_union_ms=value_union,verification_score_percent=100*score_union/target_gpu,verification_value_percent=100*value_union/target_gpu,target_direct_union_ms=target_union,target_encoder_sum_ms=target_sum,direct_coverage=target_union/target_gpu,encoder_overlap_ms=target_sum-target_union,commit_gpu_ms=commit_gpu,other_commands=other_commands,groups=table,negative_controls_rejected=negative,files_sha256={str(p):sha(p) for p in [ROOT/'counters1.log',ROOT/'counters1-result.json',ROOT/'counters1-preflight.json',ROOT/'counter-input1.json']},graph_fusions_and_batch_boundaries_preserved=True,mps_owned_encoders_observed=False,counter_overhead_measured=False,region_unions_may_overlap=True,proposals_replayed_from_saved_data=True,learned_draft_inference_run=False,serving_speed_claim=False,coding_quality_qualified=False,goal_complete=False)
    DEST.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['files_sha256','groups']},indent=2));print(json.dumps(table[:12],indent=2))
if __name__=='__main__':main()
