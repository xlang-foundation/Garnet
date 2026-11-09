"""Audit real native MLP observations and unchanged complete model states."""
from pathlib import Path
import copy,hashlib,json,struct,sys
ROOT=Path(sys.argv[1]).resolve()
BASE=ROOT.parent/'mistral-half-attention-native1'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def require(v,m):
    if not v:raise ValueError(m)
def events(p):return [json.loads(line) for line in p.read_text().splitlines() if line.startswith('{')]
def validate(current,original,metadata):
    require(current[-1]['event']=='closed' and not any(e['event']=='error' for e in current),'incomplete/failed model')
    completed=lambda es:{e['id']:(e['prompt_ids'],e['generated_ids'],e['text']) for e in es if e['event']=='completed'}
    a,b=completed(current),completed(original);require(a==b and len(a)==4 and sum(len(v[1]) for v in a.values())==32,'changed actual model output')
    state=lambda es:{e['invocation']:{k:v for k,v in e.items() if k not in ['scope']} for e in es if e['event']=='diagnostic_full_state'}
    a,b=state(current),state(original);require(a==b and sorted(a)==[1,16,18,25],'changed complete physical model state')
    boundary=[e for e in current if e['event']=='diagnostic_native_kv_hoist' and e['invocation']==16]
    require(len(boundary)==1,'missing/duplicate observed boundary');boundary=boundary[0]
    fixture=read(Path(read(ROOT/'profile1.json')['diagnostic_fixture']))
    prompts=[r['token_ids'] for r in fixture['requests']]
    require(boundary['ids']==[token for p in prompts for token in p[1920:2048]] and boundary['slots']==[float(1920+t) for row in range(4) for t in range(128)] and boundary['active']==[1.0]*512,'wrong actual trained prefix/token controls')
    saved=[e for e in current if e['event']=='diagnostic_mlp_saved'];require([e['layer'] for e in saved]==[0,15,31],'missing/duplicate trained layer')
    for e in saved:
        m=e['metadata'];require(m==metadata[e['layer']] and e['invocation']==m['invocation']==16 and m['layer']==e['layer'] and m['rows']==512 and m['chunk']==128 and m['offset']==1920 and m['mask']==31 and m['arithmetic_changed'] is False and m['diagnostic_only'] is True,'wrong native MLP snapshot metadata')
        require(m['snapshot_bytes']==512*(2*4096+3*14336)*4 and len(m['source_identities'])==5 and all(isinstance(v,int) and v>0 for v in m['source_identities']),'wrong capture extent or provenance')
    rejections=[e['label'] for e in current if e['event']=='diagnostic_mlp_rejection']
    require(rejections==['unarmed-save','wrong-invocation','wrong-layer','duplicate-prepare','not-yet-complete-save','relative-save','existing-folder-save','duplicate-save'],'missing native API failure gates')
    close=current[-1]['memory'];require(all(close[k]==0 for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']) and close['metal_allocated_bytes']<=4*1024*1024,'native observer leak')
    half=[e for e in current if e['event']=='diagnostic_half_attention'];require(len(half)==25 and sum(e['dispatches'] for e in half)==544,'wrong actual half-prefill route')
    require(all(e['dispatches']==0 for e in current if e['event']=='native_chunk_attention_mps'),'executed old MPS attention')
    return dict(generated_tokens=32,full_state_digest_pairs=260,native_failure_gates=rejections,half_prefill_dispatches=544,postclose_metal_bytes=close['metal_allocated_bytes'])
def main():
    require(read(ROOT/'capture1-status.json')['returncode']==0 and read(ROOT/'capture1-postflight.json')['passed'],'failed capture process/postflight')
    for path,digest in read(ROOT/'capture1-preflight.json')['files_sha256'].items():require(sha(Path(path))==digest,'changed input '+path)
    current,original=events(ROOT/'capture1.stdout'),events(BASE/'model-on7.stdout')
    metadata={i:read(ROOT/'captures1'/('layer-'+str(i))/'metadata.json') for i in [0,15,31]}
    result=validate(current,original,metadata);files={};statistics={}
    import numpy as np
    for layer in [0,15,31]:
        folder=ROOT/'captures1'/('layer-'+str(layer));statistics[str(layer)]={}
        require(set(p.name for p in folder.iterdir())=={'norm-input.f32','gate.f32','up.f32','silu.f32','down.f32','metadata.json'},'incomplete/unexpected capture files')
        for name,width in [('norm-input',4096),('gate',14336),('up',14336),('silu',14336),('down',4096)]:
            path=folder/(name+'.f32');require(path.stat().st_size==512*width*4,'truncated capture')
            a=np.fromfile(path,dtype='<f4');require(np.all(np.isfinite(a)),'nonfinite trained observation')
            files[str(path.relative_to(ROOT))]=sha(path);statistics[str(layer)][name]=dict(words=int(a.size),maximum_absolute=float(np.max(np.abs(a))),rms=float(np.sqrt(np.mean(a.astype(np.float64)**2))))
        files[str((folder/'metadata.json').relative_to(ROOT))]=sha(folder/'metadata.json')
    require(not any(p.name.startswith('must-not-exist') for p in (ROOT/'captures1').iterdir()),'failed API gate created output')
    rejected=[]
    mutations={
      'wrong_generated_token':lambda e,m:next(x for x in e if x['event']=='completed')['generated_ids'].__setitem__(0,-1),
      'wrong_KV':lambda e,m:next(x for x in e if x['event']=='diagnostic_full_state')['layers'][0].__setitem__(0,'0'*64),
      'wrong_hidden':lambda e,m:next(x for x in e if x['event']=='diagnostic_full_state').update(hidden='0'*64),
      'wrong_capture_layer':lambda e,m:next(x for x in e if x['event']=='diagnostic_mlp_saved').update(layer=1),
      'wrong_mask':lambda e,m:next(x for x in e if x['event']=='diagnostic_mlp_saved')['metadata'].update(mask=15),
      'wrong_positions':lambda e,m:next(x for x in e if x['event']=='diagnostic_native_kv_hoist' and x['invocation']==16)['slots'].__setitem__(0,0.),
      'missing_failure_gate':lambda e,m:e.remove(next(x for x in e if x['event']=='diagnostic_mlp_rejection')),
      'owner_leak':lambda e,m:e[-1]['memory'].update(garnet_live_matrix_owners=1),
    }
    for label,mutate in mutations.items():
        e,m=copy.deepcopy((current,metadata));mutate(e,m)
        try:validate(e,original,m)
        except (ValueError,KeyError,IndexError,TypeError):rejected.append(label)
        else:raise ValueError('Corruption admitted '+label)
    result.update(passed=True,scope=__doc__,layers=[0,15,31],rows_per_layer=512,trained_weight_and_activation_inputs=True,capture_files_sha256=files,statistics=statistics,negative_controls_rejected=rejected,copy_unit=read(ROOT/'copy-test1.stdout'),stdout_sha256=sha(ROOT/'capture1.stdout'),original_stdout_sha256=sha(BASE/'model-on7.stdout'),source_binding_sha256=sha(ROOT/'source-bindings1.json'),serving_speed_claim=False,quality_qualified=False,goal_complete=False,warning='CoreAnalytics context-leak warnings also occur in the frozen control; zero native owner counters do not explain them.')
    out=Path(sys.argv[2]) if len(sys.argv)>2 else ROOT/'capture-audit1.json';require(not out.exists(),'Never overwrite audit');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['statistics','capture_files_sha256']},indent=2))
if __name__=='__main__':main()
