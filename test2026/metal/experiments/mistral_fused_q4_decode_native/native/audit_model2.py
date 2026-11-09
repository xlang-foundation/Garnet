"""Audit native four-row fusion against frozen full Mistral FP32 state."""
from pathlib import Path
import copy,hashlib,json,math,struct,sys
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'model-audit2.json'
J=ROOT.parent/'mistral-prefill-completion-wide1';files={}
def need(v,m):
    if not v:raise ValueError(m)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def raw(p):
    b=p.read_bytes();files[str(p)]=hashlib.sha256(b).hexdigest();return b
def read(p):return json.loads(raw(p))
def events(p):
    result=[]
    for line in raw(p).splitlines():
        try:e=json.loads(line)
        except ValueError:continue
        if isinstance(e,dict) and 'event' in e:result.append(e)
    return result
def select(es,k):return [e for e in es if e['event']==k]
def check(es,control,inventory,profile):
    need(not select(es,'error') and len(select(es,'ready'))==len(select(es,'closed'))==1,'failed/unfinished model lifecycle')
    ready=select(es,'ready')[0];closed=select(es,'closed')[0]
    need(es[-1]==closed and ready['profile']==profile and profile['fused_decode_rows4'] is True and 'native_fused_q4_decode_rows4_default_v1' in ready['backend'],'wrong native fused profile')
    prep=select(es,'native_fused_q4_decode_prepared')
    need(len(prep)==1 and prep[0]['sites']==[1]*32 and prep[0]['profiles'] in [1,2] and prep[0]['scratch_bytes']==65536 and prep[0]['extra_weight_bytes']==0 and es.index(prep[0])<es.index(ready),'wrong/late native graph/profile preparation')
    need(ready['memory']['garnet_live_affine_weight_owners']==225,'extra/missing checkpoint weights')
    need(ready['memory']['metal_allocated_bytes']<=ready['memory']['metal_recommended_working_set_bytes']-268435456,'working set margin violated')
    new=select(es,'completed');old=select(control,'completed');need(len(new)==len(old)==4,'incomplete output coverage')
    for a,b in zip(new,old):
        need(all(a[k]==b[k] for k in ['id','prompt_ids','generated_ids','text','cached_tokens','finish_reason','native_request','native_row','native_epoch']),'saved native output differs')
        need(len(a['generated_ids'])==8 and a['cached_tokens']==0,'wrong diagnostic output/cache limits')
    for kind,count in [('diagnostic_full_state',4),('diagnostic_request_state',8)]:
        rows=select(es,kind);saved=select(control,kind);need(len(rows)==len(saved)==count and rows==saved,'complete KV/hidden state differs '+kind)
        for row in rows:
            values=row['hidden_values'];need(all(type(v) in [int,float] and math.isfinite(v) for v in values),'nonfinite hidden word')
            need(hashlib.sha256(struct.pack('<'+str(len(values))+'f',*values)).hexdigest()==row['hidden'],'hidden word/digest mismatch')
    batches=select(es,'native_batch');route=select(es,'diagnostic_fused_q4_decode')
    need(len(batches)==len(route)>0,'missing native frame route coverage')
    frames=blocks=copies=0
    for batch,r in zip(batches,route):
        active=sum(q['prefill_tokens'] for q in batch['requests']) if batch['stage']=='prefill' else len(batch['requests'])
        wanted=32 if batch['stage']=='decode' and active==4 else 0
        physical=4*batch['prefill_chunk'] if batch['stage']=='prefill' else 4
        need(r['dispatches']==wanted and r['active_tokens']==active and r['physical_tokens']==physical and type(r['input_copies']) is int and 0<=r['input_copies']<=wanted,'wrong native selection or alias copy count')
        frames+=int(wanted>0);blocks+=r['dispatches'];copies+=r['input_copies']
    need(frames==6 and blocks==192,'four-row decode fusion was not exercised')
    references=[ref for ref in inventory['references'] if ref['role']=='fused_decode_fp32_input']
    need(len(references)==1,'private FP32 input not inventoried exactly once')
    buffers={b['id']:b for b in inventory['buffers']};buffer=buffers[references[0]['buffer_id']]
    need(buffer['length_bytes']==65536 and buffer['allocated_bytes']>=65536,'wrong private input allocation')
    after=closed['memory'];need(after['metal_allocated_bytes']<=4194304,'device buffers retained after close')
    need(all(after[k]==0 for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']),'tracked native owners retained')
    return dict(output_tokens=32,whole_state_digest_pairs=260,request_state_digest_pairs=520,whole_hidden_values=65536,request_hidden_values=32768,four_row_decode_frames=frames,native_fused_blocks=blocks,native_input_copy_calls=copies,prepared_profiles=prep[0]['profiles'],extra_private_input_bytes=65536,extra_weight_cache_bytes=0,ready_metal_allocated_bytes=ready['memory']['metal_allocated_bytes'],close_metal_allocated_bytes=after['metal_allocated_bytes'])
def main():
    need(not DEST.exists(),'never overwrite audit')
    need(read(ROOT/'model2-status.json')['returncode']==0 and read(ROOT/'model2-postflight.json')['passed'],'model collection unfinished')
    for name,h in read(ROOT/'model2-preflight.json')['files_sha256'].items():need(sha(Path(name))==h,'changed model input '+name)
    es=events(ROOT/'model2.stdout');control=events(J/'model-control2.stdout');inventory=read(ROOT/'engine-inventory-model2.json');profile=read(ROOT/'profile-model2.json');report=check(es,control,inventory,profile)
    def event(e,k):return select(e,k)[0]
    mutations={
        'wrong_output':lambda e,i:event(e,'completed')['generated_ids'].__setitem__(0,-1),
        'wrong_whole_kv':lambda e,i:event(e,'diagnostic_full_state')['layers'][0].__setitem__(0,'0'*64),
        'wrong_row_kv':lambda e,i:event(e,'diagnostic_request_state')['kv'][0].__setitem__(0,'0'*64),
        'wrong_hidden_words':lambda e,i:event(e,'diagnostic_request_state')['hidden_values'].__setitem__(0,1e9),
        'wrong_epoch':lambda e,i:event(e,'completed').update(native_epoch=0),
        'unproved_graph_site':lambda e,i:event(e,'native_fused_q4_decode_prepared')['sites'].__setitem__(0,0),
        'fused_prefill':lambda e,i:event(e,'diagnostic_fused_q4_decode').update(dispatches=32),
        'missing_four_row_dispatch':lambda e,i:next(r for r in e if r['event']=='diagnostic_fused_q4_decode' and r['dispatches']).update(dispatches=0),
        'impossible_input_copy_count':lambda e,i:event(e,'diagnostic_fused_q4_decode').update(input_copies=33),
        'missing_private_input':lambda e,i:i.update(references=[r for r in i['references'] if r['role']!='fused_decode_fp32_input']),
        'retained_native_owner':lambda e,i:event(e,'closed')['memory'].update(garnet_live_matrix_owners=1),
    }
    rejected=[]
    for name,mutate in mutations.items():
        bad=copy.deepcopy(es);inv=copy.deepcopy(inventory);mutate(bad,inv)
        try:check(bad,control,inv,profile)
        except (ValueError,KeyError,IndexError,TypeError):rejected.append(name)
        else:raise ValueError('corruption admitted '+name)
    raw(ROOT/'model2.stderr')
    output=dict(passed=True,scope=__doc__,qualification=report,all_output_ids_text_and_complete_state_exact_frozen=True,negative_controls_rejected=rejected,files_sha256=files,performance_measurement=False,verification_rollback_qualified=False,prefix_cache_lineage_qualified=False,coding_quality_qualified=False,serving_speed_claim=False,goal_complete=False)
    DEST.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:v for k,v in output.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
