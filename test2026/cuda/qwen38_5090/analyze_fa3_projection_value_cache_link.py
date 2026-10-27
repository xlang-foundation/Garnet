"""Link saved actual pre-cache V projections to the retained full-model cache bytes."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_fp8_z128_full4k_model import checked
from test2026.cuda.qwen38_5090.gdn_chunk_stage_replay import array
from test2026.cuda.qwen38_5090.fa3_projection_witness import replay
from test2026.cuda.qwen38_5090.fa3_prefix_cache_witness import read_cache
CURRENT='fa3-projection-witness-attempt2';PRIOR='fa3-prefix-cache-witness-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';sha=lambda b:hashlib.sha256(b).hexdigest()
def projected_cache(raw):
    assert len(raw)==131072*4
    bits=np.frombuffer(raw,dtype='<u4');assert np.all((bits&0xffff)==0),'Actual FP8 plugin output must be exactly BF16-representable'
    values=(bits>>16).astype('<u2');assert np.all((values&0x7f80)!=0x7f80)
    result=np.zeros((4,4096,256),dtype='<u2');result[:,:128,:]=values.reshape(128,4,256).transpose(1,0,2)
    return result.tobytes()
def main():
    a=argparse.ArgumentParser();a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();proofs={};archives={}
    try:
        for job in (CURRENT,PRIOR):
            raw=(args.proofs/(job+'-verified.json')).read_bytes();assert raw==(args.proofs/(job+'-remote-verified.json')).read_bytes();proofs[job]=(json.loads(raw),sha(raw));file=args.proofs/(job+'-evidence.zip');assert sha(file.read_bytes())==proofs[job][0]['archive_sha256'];archives[job]=checked(file)[0]
        z=archives[CURRENT];base='results/'+CURRENT+'/';r=json.loads(z.read(base+'native/report.json'));fixture=z.read(P+'evidence/reconstructed-fa3-common-input.f32');checked_arrays=replay(r,lambda n:z.read(base+'native/'+n),fixture);assert checked_arrays==proofs[CURRENT][0]['arrays'] and checked_arrays['normalized_exact']
        old=archives[PRIOR];profiles={};observed={};predicted={}
        for t,mode in ((32,'baseline'),(128,'native')):
            row=next(c for c in r['cases'] if c['tokens']==t);raw=b''.join(array(chunk['arrays']['value'],lambda n:z.read(base+'native/'+n)) for chunk in row['repeats'][0]);predicted[t]=projected_cache(raw)
            oldbase='results/'+PRIOR+'/'+mode+'/';prior=json.loads(old.read(oldbase+'report.json'));assert prior['tokens']==t and prior['status']=='fa3_prefix_caches_collected_retained_ordinary_control_exact' and prior['prompt_ids'][:128]==r['source_input_metadata']['input_ids'] and prior['performance_admitted'] is False
            snapshot=next(s for s in prior['snapshots'] if s['prefix']==128);observed[t]=read_cache(snapshot['caches']['value'],lambda n:old.read(oldbase+n));cache=np.frombuffer(observed[t],dtype='<u2').reshape(4,4096,256);assert np.all((cache&0x7f80)!=0x7f80) and not np.any(cache[:,128:,:])
            actual=np.frombuffer(observed[t],dtype='<u2');expect=np.frombuffer(predicted[t],dtype='<u2');changed=np.flatnonzero(actual!=expect)
            profiles[str(t)]=dict(projection_f32_words=131072,complete_cache_BF16_values_compared=4194304,inactive_tail_zero=True,pre_cache_projection_reproduces_entire_retained_VALUE_cache=not bool(changed.size),different_BF16_values=int(changed.size),predicted_cache_sha256=sha(predicted[t]),retained_cache_sha256=sha(observed[t]),first_difference=dict(head=int(changed[0])//(4096*256),token=(int(changed[0])//256)%4096,dimension=int(changed[0])%256) if changed.size else None)
        a=np.frombuffer(observed[32],dtype='<u2');b=np.frombuffer(observed[128],dtype='<u2');x=np.flatnonzero(a!=b);p=np.frombuffer(predicted[32],dtype='<u2');q=np.frombuffer(predicted[128],dtype='<u2');y=np.flatnonzero(p!=q)
        linked=all(row['pre_cache_projection_reproduces_entire_retained_VALUE_cache'] for row in profiles.values()) and np.array_equal(x,y)
        result=dict(status='actual_FA3_pre_cache_VALUE_to_retained_cache_link_replayed',source_archives={job:doc[0]['archive_sha256'] for job,doc in proofs.items()},paired_source_proofs={job:doc[1] for job,doc in proofs.items()},same_first128_prompt_ids=True,normalized_inputs_byte_exact=True,profiles=profiles,retained_VALUE_cache_changed_values=int(x.size),projection_predicted_changed_values=int(y.size),all_retained_first128_VALUE_changes_reproduced_pre_cache=bool(linked),first_retained_change=dict(head=int(x[0])//(4096*256),token=(int(x[0])//256)%4096,dimension=int(x[0])%256,T32_BF16_bits=int(a[x[0]]),T128_BF16_bits=int(b[x[0]])) if x.size else None,KEY_norm_RoPE_or_full3968_chain_cause_established=False,model_or_quality_or_performance_qualified=False,two_times_target_achieved=False,analyzer_sha256=sha(Path(__file__).read_bytes()))
    finally:
        for z in archives.values():z.close()
    target=args.proofs/'fa3-projection-VALUE-cache-link-verified.json';raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw;target.write_bytes(raw);print(json.dumps(result));return 0
if __name__=='__main__':raise SystemExit(main())
