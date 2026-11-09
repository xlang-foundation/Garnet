"""Audit native N4 full-model state, dispatch policy, memory inventory and cleanup."""
from pathlib import Path
import copy,hashlib,json,math,struct,sys
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'model-audit3.json'
J=ROOT.parent/'mistral-prefill-completion-wide1';files={}
def require(v,m):
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
    require(not select(es,'error') and len(select(es,'ready'))==len(select(es,'closed'))==1,'missing or failed lifecycle')
    ready=select(es,'ready')[0];closed=select(es,'closed')[0];require(es[-1]==closed,'events after close')
    require(ready['profile']==profile and profile['n4_decode_mlp_rows4'] is True and 'mistral_n4_decode_mlp_rows4_v1' in ready['backend'] and 'n4_cache_budget_bytes=3221225472' in ready['backend'],'wrong N4 profile/backend')
    before=ready['memory'];after=closed['memory'];require(before['garnet_n4_reserved_or_live_bytes']==3170893824 and before['garnet_n4_weight_owners']==96 and before['garnet_n4_decode_encoded']==0,'wrong prepared weight count/bytes')
    require(before['metal_allocated_bytes']<=before['metal_recommended_working_set_bytes']-256*1024*1024,'working set margin exceeded')
    require(after['garnet_n4_weight_owners']==after['garnet_n4_reserved_or_live_bytes']==0 and after['metal_allocated_bytes']<=4*1024*1024,'N4 cache/buffers retained')
    require(all(after[k]==0 for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']),'tracked owner retained')
    done=select(es,'completed');old=select(control,'completed');require(len(done)==len(old)==4,'incomplete response coverage')
    for a,b in zip(done,old):
        require(all(a[k]==b[k] for k in ['id','prompt_ids','generated_ids','text','cached_tokens','finish_reason','native_request','native_row','native_epoch']),'native response/control differs')
        require(len(a['generated_ids'])==8 and a['cached_tokens']==0,'wrong diagnostic budget/cache')
    for kind,count in [('diagnostic_full_state',4),('diagnostic_request_state',8)]:
        new=select(es,kind);saved=select(control,kind);require(len(new)==len(saved)==count and new==saved,'complete canonical state/hidden differs '+kind)
        for row in new:
            values=row['hidden_values'];require(all(type(x) in [int,float] and math.isfinite(x) for x in values),'invalid hidden values')
            digest=hashlib.sha256(struct.pack('<'+str(len(values))+'f',*values)).hexdigest();require(digest==row['hidden'],'hidden words not bound to digest')
    batches=select(es,'native_batch');route=select(es,'diagnostic_n4_decode');require(len(batches)==len(route)>0,'N4 body route coverage missing')
    total=0;four=0
    for batch,e in zip(batches,route):
        expected=96 if batch['stage']=='decode' and len(batch['requests'])==4 else 0
        require(e==dict(event='diagnostic_n4_decode',stage=batch['stage'],dispatches=expected,weights=96,payload_bytes=3170893824),'N4 selected outside four-active-row MLP or owner budget changed');total+=expected;four+=int(expected>0)
    require(four==6 and total==576==after['garnet_n4_decode_encoded'],'N4 path never ran or count differs')
    buffers={b['id']:b for b in inventory['buffers']};roles={role:{ref['buffer_id'] for ref in inventory['references'] if ref['representation']==role} for role in ['affine_int4_n4_payload','affine_int4_n4_fp16_scales','affine_int4_n4_fp16_biases']}
    require(all(len(v)==96 for v in roles.values()),'missing N4 inventory owner family')
    ids=set().union(*roles.values());require(len(ids)==288,'N4 payload/scales/biases alias')
    length=sum(buffers[i]['length_bytes'] for i in ids);require(length==3170893824 and sum(buffers[i]['allocated_bytes'] for i in ids)>=length,'wrong inventoried N4 memory')
    return dict(output_tokens=32,whole_state_digest_pairs=260,request_state_digest_pairs=520,whole_hidden_values=65536,request_hidden_values=32768,four_row_decode_frames=four,n4_encoded=total,n4_weight_owners=96,n4_unique_buffers=288,n4_payload_bytes=length,ready_metal_allocated_bytes=before['metal_allocated_bytes'],close_metal_allocated_bytes=after['metal_allocated_bytes'])
def main():
    require(not DEST.exists(),'never overwrite audit');require(read(ROOT/'model2-status.json')['returncode']==0 and read(ROOT/'model2-postflight.json')['passed'],'unfinished new model')
    for name,h in read(ROOT/'model2-preflight.json')['files_sha256'].items():require(sha(Path(name))==h,'changed model input '+name)
    es=events(ROOT/'model2.stdout');control=events(J/'model-control2.stdout');inventory=read(ROOT/'engine-inventory-model2.json');profile=read(ROOT/'profile-model2.json');report=check(es,control,inventory,profile)
    rejected=[]
    def event(data,kind):return select(data,kind)[0]
    mutations={
        'changed_output':lambda e,i:event(e,'completed')['generated_ids'].__setitem__(0,-1),
        'changed_canonical_kv':lambda e,i:event(e,'diagnostic_full_state')['layers'][0].__setitem__(0,'0'*64),
        'changed_row_state':lambda e,i:event(e,'diagnostic_request_state')['kv'][0].__setitem__(0,'0'*64),
        'changed_hidden':lambda e,i:event(e,'diagnostic_request_state')['hidden_values'].__setitem__(0,1e8),
        'stale_row_epoch':lambda e,i:event(e,'completed').update(native_epoch=0),
        'wrong_n4_occupancy':lambda e,i:event(e,'diagnostic_n4_decode').update(dispatches=96),
        'missing_n4_dispatch':lambda e,i:next(x for x in e if x['event']=='diagnostic_n4_decode' and x['dispatches']).update(dispatches=0),
        'wrong_n4_budget':lambda e,i:event(e,'ready')['memory'].update(garnet_n4_reserved_or_live_bytes=1),
        'retained_n4_owner':lambda e,i:event(e,'closed')['memory'].update(garnet_n4_weight_owners=1),
        'missing_n4_inventory':lambda e,i:i.update(references=[ref for ref in i['references'] if ref['representation']!='affine_int4_n4_payload']),
    }
    for name,mutate in mutations.items():
        bad=copy.deepcopy(es);inv=copy.deepcopy(inventory);mutate(bad,inv)
        try:check(bad,control,inv,profile)
        except (ValueError,KeyError,IndexError,TypeError):rejected.append(name)
        else:raise ValueError('corrupt model evidence admitted '+name)
    raw(ROOT/'model2.stderr');raw(Path(__file__))
    result=dict(passed=True,scope=__doc__,qualification=report,all_outputs_and_complete_state_exact_frozen=True,negative_controls_rejected=rejected,files_sha256=files,performance_measurement=False,native_option_default_on=False,coding_quality_qualified=False,prefix_reuse_qualified_for_new_engine=False,serving_speed_claim=False,goal_complete=False)
    DEST.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
