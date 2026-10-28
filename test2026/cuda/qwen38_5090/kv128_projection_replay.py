"""Full-array intervention comparison. No numerical threshold is relaxed."""
import hashlib,json,zipfile
from .fa3_projection_witness import replay as parent_replay
from .gdn_chunk_stage_replay import array
from .kv128_projection_layout import check_graph
PREFIX=b'GARNET_KV128_FACTORY '
CONFIG=[(n,v,4,v) for n,v in zip(('ID','TILE_ID','STAGES_ID','SPLITK_NUM','REDUCTION_SCHEME','CTA_SWIZZLING'),(35,15,18,7,2,1))]
APIS=['cudaGetDevice','cudaGetDeviceProperties','cublasLtGetVersion','cublasLtCreate','cublasLtMatmulDescCreate']+['cublasLtMatmulDescSetAttribute']*2+['cublasLtMatrixLayoutCreate']*5+['cublasLtMatmulPreferenceCreate','cublasLtMatmulPreferenceSetAttribute','cublasLtMatmulAlgoGetHeuristic']+['cublasLtMatmulAlgoConfigGetAttribute']*6+['cublasLtMatmulAlgoCheck']*2
def factory_events(raw):
    groups={}
    for line in raw.splitlines():
        if not line.startswith(PREFIX):continue
        row=json.loads(line[len(PREFIX):]);assert set(row)=={'id','seq','event','name','a','b','c','opaque'}
        assert type(row['id']) is int and row['id']>0;groups.setdefault(row['id'],[]).append(row)
    assert groups and sorted(groups)==list(range(1,len(groups)+1))
    for group in groups.values():
        assert [r['seq'] for r in group]==list(range(1,len(group)+1)) and group[0]['event']=='begin' and group[-1]['event']=='end'
    return groups
def factory_replay(raw,require_success):
    groups=factory_events(raw);created=cached=0;failed=[]
    for ident,group in groups.items():
        end=group[-1]['name'];apis=[r for r in group if r['event']=='api']
        if end not in ('created','cached'):
            failed.append(dict(id=ident,outcome=end,rejections=[r for r in group if r['event']=='reject']));continue
        assert not any(r['event']=='reject' for r in group)
        assert all(r['a']==(130400 if r['name']=='cublasLtGetVersion' else 0) for r in apis)
        if end=='cached':cached+=1;assert [r['name'] for r in apis]==['cudaGetDevice','cudaGetDeviceProperties','cublasLtGetVersion'];continue
        created+=1;assert [r['name'] for r in apis]==APIS
        configs=[(r['name'],r['a'],r['b'],r['c']) for r in group if r['event']=='configuration'];assert configs==CONFIG
        checks=[r for r in group if r['event']=='check'];assert [r['name'] for r in checks]==['M32','M128'] and all(r['a']==0 and 0<=r['b']<=4<<20 for r in checks)
        selected=[r for r in group if r['event']=='selection'];assert len(selected)==1 and selected[0]['name']=='documented_M32_control' and len(bytes.fromhex(selected[0]['opaque']))==64
    if require_success:assert created>=1 and not failed
    return dict(status='actual_KV128_factory_events_replayed',factories=len(groups),created=created,cached=cached,failed=failed,all_successful=created>=1 and not failed,SDK_control_and_actual_descriptors_checked=created>=1,math_or_performance_qualified=False)
def replay(report,read,fixture,parent_report,parent_read):
    result=parent_replay(report,read,fixture);old=parent_replay(parent_report,parent_read,fixture)
    assert old['normalized_exact'] and old['stages']['key']['changed_F32_words']==old['stages']['value']['changed_F32_words']==8
    assert report['weight_metadata']==parent_report['weight_metadata']
    compared=0
    for current,control in zip(report['cases'],parent_report['cases']):
        t=current['tokens'];assert t==control['tokens'];check_graph(current['graph'],t)
        for new_sequence,old_sequence in zip(current['repeats'],control['repeats']):
            for new,prior in zip(new_sequence,old_sequence):
                for k in ('input_hidden','input_hidden_after'):assert array(new[k],read)==array(prior[k],parent_read)
                names=('normalized','key','value') if t==32 else ('normalized',)
                for name in names:
                    a=array(new['arrays'][name],read);b=array(prior['arrays'][name],parent_read);assert a==b,('Control output drift',t,name);compared+=len(a)//4
    result.update(status='KV128_projection_intervention_complete_arrays_replayed',unchanged_control_output_F32_words=compared,T32_control_and_all_normalized_outputs_unchanged=True,original_eight_K_and_eight_V_differences_removed=result['normalized_exact'] and result['key_value_both_exact'],all_original_full_model_gates_still_required=True)
    return result
def replay_control(report,read,fixture,proofs):
    archive=proofs/'fa3-projection-witness-attempt2-evidence.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()=='75ce1c0bac482b86f2fadd7445f7c3bb3ab32499946d7a87e72ef00f46a1732f'
    base='results/fa3-projection-witness-attempt2/native/'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        return replay(report,read,fixture,json.loads(z.read(base+'report.json')),lambda n:z.read(base+n))
