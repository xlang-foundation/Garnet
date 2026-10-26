"""Replay the complete T32 partial result and actual postfailure helper observation; T128 remains unrun."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_layer2_ordinary_witness import JOB,M,STAGE,checked
from test2026.cuda.qwen38_5090.gdn_chunk_stage_replay import array
from test2026.cuda.qwen38_5090.layer2_ordinary_layout import INPUT_SOURCE,validate_profile,check_graph
from test2026.cuda.qwen38_5090.fp8_z128_context_audit import replay as replay_audit
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);a=parser.parse_args();archive=a.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive)
    try:
        base='results/'+JOB+'/';p=json.loads(z.read(M+'evidence/layer2-ordinary-witness-preparation.json'));w=json.loads(z.read(base+'workflow.json'));r=json.loads(z.read(base+'native/report.json'));proof=(a.proofs/(JOB+'-verified.json')).read_bytes();q=json.loads(proof)
        assert q['status']=='layer2_ordinary_failure_preserved' and q['archive_sha256']==sha(archive.read_bytes()) and q['source_runtime_SDK_checkpoint_restored'] and q['complete_array_replay'] is None
        assert w['exit_code']==1 and w['cleanup_finished'] and w['baseline_source_restored'] and w['baseline_library_restored'] and w['source_before_sha256']==w['source_after_sha256']==p['files']
        assert r['status']=='failed' and r['error']=='RuntimeError(\'TensorRT buildSerializedNetwork failed\')' and r['input_source']==INPUT_SOURCE and r['model_or_performance_qualified'] is False
        assert len(r['cases'])==1;row=r['cases'][0];spec=validate_profile(row['layer'],row['tokens'],row['diagnostic']);assert row['tokens']==32 and row['layout']==spec and row['input_lineage']==INPUT_SOURCE;check_graph(row['graph'],32,False)
        read=lambda name:z.read(base+'native/'+name);common=z.read(M+'evidence/layer2-ordinary-common-input.f32');assert len(common)==2621440 and sha(common)==p['common_input_sha256']==r['input_fixture_sha256'] and array(r['common_input'],read)==common
        assert np.array_equal(np.frombuffer(array(row['active'],read),dtype='<f4'),np.ones(32,dtype='<f4'))
        entries=2;words=0;repeats=[]
        assert len(row['repeats'])==2
        for repeat,sequence in enumerate(row['repeats']):
            assert len(sequence)==4;history=bytes(40960*4);state=bytes(786432*4);values=[]
            for i,chunk in enumerate(sequence):
                assert chunk['start']==i*32 and chunk['repeat']==repeat and set(chunk['arrays'])=={'primary'}
                hidden=array(chunk['input_hidden'],read);prior_h=array(chunk['input_history'],read);prior_s=array(chunk['input_state'],read)
                assert hidden==common[i*32*5120*4:(i+1)*32*5120*4] and prior_h==history and prior_s==state
                for name,value in (('input_hidden_after',hidden),('input_history_after',prior_h),('input_state_after',prior_s)):assert array(chunk[name],read)==value
                primary=array(chunk['arrays']['primary'],read);assert len(primary)==spec['primary_words']*4;history=primary[32*5120*4:32*5120*4+40960*4];state=primary[32*5120*4+40960*4:];values.append(primary);entries+=7;words+=len(primary)//4
            if repeats:assert values==repeats[0]
            repeats.append(values)
        assert entries==58 and words==7929856
        receipt=json.loads(z.read('results/'+STAGE+'/audit-library.json'));raw=z.read(base+'native/context-audit.json');audit=json.loads(raw);observed=replay_audit(audit,receipt['expected_m32_opaque']);bridge=r['initialization_context_audit']
        assert observed==bridge['validation']==q['initialization_context_observation'] and sha(raw)==bridge['report_sha256'] and bridge['native_package_ABI']==24 and bridge['library_sha256']==receipt['library_sha256'] and bridge['native_package_library_sha256']==receipt['native_package_library_sha256']
        assert observed['created'] is True and observed['first_successful_index']==0 and observed['matmul_calls']==0 and audit['cuda_error_before']==audit['cuda_error_after']==0
        hidden=b''.join(primary[:32*5120*4] for primary in repeats[0])
        result={'status':'layer2_T32_partial_arrays_and_postfailure_helper_replayed','archive_sha256':q['archive_sha256'],'original_failure_proof_sha256':sha(proof),'complete_finite_array_entries':entries,'complete_output_f32_words_replayed':words,'completed_profiles':1,'T32_warmups':3,'T32_repeats':2,'readonly_inputs_and_carried_states_exact':True,'T32_repeats_exact':True,'complete_layer2_T32_output_hidden_bytes':len(hidden),'complete_layer2_T32_output_hidden_sha256':sha(hidden),'T128_initialization_failed':True,'T128_outputs_collected':False,'T32_T128_arithmetic_comparison_available':False,'postfailure_helper_observation':observed,'postfailure_helper_raw_sha256':sha(raw),'postfailure_candidates':audit['heuristics'][0]['candidates'],'postfailure_helper_info':audit['helper_info'],'source_runtime_SDK_checkpoint_restored':True,'actual_failed_Garnet_plugin_or_helper_boundary_identified':False,'initialization_cause_or_reliability_resolved':False,'full_model_cause_established':False,'model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False,'analyzer_sha256':sha(Path(__file__).read_bytes()),'scope':'Actual complete ordinary layer2 T32 result survives failed subsequent T128 construction. Same-process/thread postfailure observer uses unchanged Z128 helper in a separate TU and succeeds; it does not prove which actual Garnet plugin failed or its earlier call status/opaque. Do not infer algorithm-selection cause, arithmetic parity, quality or performance.'}
    finally:z.close()
    raw=json.dumps(result,indent=2).encode();target=a.proofs/(JOB+'-analysis.json');assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'complete_finite_array_entries':entries,'postfailure_helper_created':observed['created']}));return 0
if __name__=='__main__':raise SystemExit(main())
