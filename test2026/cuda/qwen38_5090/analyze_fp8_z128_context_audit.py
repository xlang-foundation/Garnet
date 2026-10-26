"""Independent partial T32 primary/state/repeat/baseline replay after failed T128 build."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.gdn_chunk_stage_replay import array
from test2026.cuda.qwen38_5090.gdn_chunk_stage_layout import STAGES,layout
from test2026.cuda.qwen38_5090.fp8_z128_stage_layout import check_graph
JOB='fp8-z128-context-audit-attempt1';PREVIOUS='gdn-chunk-stage-witness-attempt2';sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);args=parser.parse_args();proof_raw=(args.proofs/(JOB+'-verified.json')).read_bytes();proof=json.loads(proof_raw)
    assert proof['status']=='fp8_z128_context_audit_and_partial_main_failure_preserved' and proof['workflow_exit_code']==1 and proof['profiles_observed']==2 and proof['complete_finite_array_entries']==203 and set(proof['context_observations'])=={'cold','native'}
    archive=args.proofs/(JOB+'-evidence.zip');assert sha(archive.read_bytes())==proof['archive_sha256'];base='results/'+JOB+'/native/'
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None
        for name,h in manifest['files'].items():assert sha(z.read(name))==h
        report=json.loads(z.read(base+'report.json'));w=json.loads(z.read('results/'+JOB+'/workflow.json'))
        assert w['exit_code']==1 and w['cleanup_finished'] and w['baseline_source_restored'] and w['baseline_library_restored'] and report['status']=='failed'
        assert report['error']=="RuntimeError('TensorRT buildSerializedNetwork failed')" and [(row['layer'],row['tokens'],row['diagnostic']) for row in report['cases']]==[(0,32,False),(0,32,True)]
        assert len(report['frontend_rejections'])==10 and all(row['rejected'] and row['error'] for row in report['frontend_rejections'])
        oldproof=json.loads((args.proofs/(PREVIOUS+'-verified.json')).read_bytes());assert oldproof['status']=='gdn_chunk_stage_witness_arrays_replayed' and sha((args.proofs/(PREVIOUS+'-evidence.zip')).read_bytes())==oldproof['archive_sha256']
        prompt=array(report['prompt_input'],lambda name:z.read(base+name));assert len(prompt)==128*5120*4 and prompt!=prompt[:16*5120*4]*8
        frames={};words=0;compared=0
        with zipfile.ZipFile(args.proofs/(PREVIOUS+'-evidence.zip')) as old:
            assert old.testzip() is None;oldbase='results/'+PREVIOUS+'/native/';prior=json.loads(old.read(oldbase+'report.json'));assert report['input_ids']==prior['input_ids']
            for row in report['cases']:
                diagnostic=row['diagnostic'];check_graph(row['graph'],32,diagnostic);spec=layout(32,diagnostic);assert row['layout']==spec and len(row['repeats'])==2
                active=array(row['active'],lambda name:z.read(base+name));assert active==np.ones(32,dtype='<f4').tobytes()
                earlier=next(item for item in prior['cases'] if item['layer']==0 and item['tokens']==32 and item['diagnostic']==diagnostic);repeats=[]
                for repeat,sequence in enumerate(row['repeats']):
                    assert len(sequence)==4;history=bytes(40960*4);state=bytes(786432*4);values=[]
                    for index,chunk in enumerate(sequence):
                        assert chunk['start']==index*32 and chunk['repeat']==repeat
                        oldchunk=earlier['repeats'][repeat][index]
                        inputs={name:array(chunk[name],lambda name:z.read(base+name)) for name in ('input_hidden','input_history','input_state','input_hidden_after','input_history_after','input_state_after')}
                        assert inputs['input_hidden']==prompt[index*32*5120*4:(index+1)*32*5120*4]==inputs['input_hidden_after'] and inputs['input_history']==history==inputs['input_history_after'] and inputs['input_state']==state==inputs['input_state_after']
                        assert set(chunk['arrays'])=={part['name'] for part in spec['rows']};arrays={name:array(value,lambda name:z.read(base+name)) for name,value in chunk['arrays'].items()};words+=sum(len(value)//4 for value in arrays.values())
                        primary=arrays['primary'];h=32*5120*4;assert len(primary)==spec['primary_words']*4;history=primary[h:h+40960*4];state=primary[h+40960*4:]
                        if diagnostic:
                            assert arrays['masked_input']==inputs['input_hidden'] and arrays['output_hidden']==primary[:h]
                            for name,width in STAGES:assert len(arrays[name])==32*width*4
                        for name,value in inputs.items():assert value==array(oldchunk[name],lambda name:old.read(oldbase+name));compared+=1
                        assert set(arrays)==set(oldchunk['arrays'])
                        for name,value in arrays.items():assert value==array(oldchunk['arrays'][name],lambda name:old.read(oldbase+name));compared+=1
                        values.append(arrays)
                    if repeat:assert values==repeats[0]
                    repeats.append(values)
                frames[diagnostic]=repeats[0]
            assert all(a['primary']==b['primary'] for a,b in zip(frames[False],frames[True])) and compared==200
        result={'status':'fp8_z128_context_audit_partial_T32_primary_and_baseline_replayed','archive_sha256':proof['archive_sha256'],'failure_proof_sha256':sha(proof_raw),'completed_policy_controls':0,'linked_parent_fresh_policy_controls':10,'completed_frontend_rejections':10,'complete_finite_array_entries':203,'partial_profiles_replayed':2,'partial_output_f32_words_replayed':words,'input_readonly_carried_state_and_repeats_exact':True,'diagnostic_primary_exact_to_ordinary':True,'unchanged_old_T32_baseline_arrays_compared':compared,'old_T32_baseline_archive_sha256':oldproof['archive_sha256'],'observed_failure':report['error'],'failed_next_profile':{'layer':0,'tokens':128,'diagnostic':False},'T128_profile_built_or_inferred':False,'plugin_initialization_root_cause_established':False,'model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False,'analyzer_sha256':sha(Path(__file__).read_bytes()),'scope':'Only two completed layer0T32ordinary/diagnostic profiles, actual realfirst128input/readonly/carriedstate/repeats/all11stages/primary and200priorbaseline arrays. Tenfreshpolicygraphs areclosed only inlinked originalfailedstageparent, notreexecuted here. SameDLL mainfailure preserved withpostfailure separateobservedactualhelper, completeopaquechoices/coldcontrol. DirectGarnetplugin/fullmodelcausality remainsunqualified. First mainT128ordinary buildfails TensorRTPluginV2Runner.allocateResources; no T128stagevalues orfullmodel/quality/servingmetric admission. Requires initialization diagnostics, not numerical-gate relaxation.'}
    raw=json.dumps(result,indent=2).encode();target=args.proofs/(JOB+'-analysis.json');assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'analysis_sha256':sha(raw),'output_words':words,'old_baseline_arrays':compared}));return 0

if __name__=='__main__':raise SystemExit(main())
