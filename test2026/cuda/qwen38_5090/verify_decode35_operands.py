"""Independent raw same-DLL off/on replay; preserve failures without interpreting invalid operands."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.b4_full4k_inputs import load_corpus
from test2026.cuda.qwen38_5090.decode35_operand_analysis import PHASES,primary_control,witness_trace,analyze
from test2026.cuda.qwen38_5090.b4_qk_m16_short_route import assert_disabled
from test2026.cuda.qwen38_5090.b4_k_m4x4_control52_route import route_trace
JOB='decode35-operands-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';B='results/'+JOB+'/';sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);args=parser.parse_args();archive=args.proofs/(JOB+'-evidence.zip')
    with zipfile.ZipFile(archive) as z:
        mf=json.loads(z.read('evidence-manifest.json'))['files'];assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(mf)+1
        for n,h in mf.items():assert sha(z.read(n))==h
        raw=z.read(P+'evidence/decode35-operands-preparation.json');p=json.loads(raw);w=json.loads(z.read(B+'workflow.json'))
        assert sha(raw)==w['preparation_sha256'] and w['cleanup_finished'] and w['all_restored'] and type(w['exit_code']) is int
        for first,last,key in [('source_before','source_after','files'),('runtime_before','runtime_after','runtime'),('environment_before','environment_after','environment'),('checkpoint_before','checkpoint_after','checkpoints'),('cuda_objects_before','cuda_objects_after','cuda_objects')]:assert w[first]==w[last]==p[key]
        assert w['cache_before_sha256']==w['cache_after_sha256']==p['cache_before_sha256'] and len(p['cuda_objects'])==34 and not w['model_or_performance_qualified']
        for n in p['archive_files']:assert sha(z.read(n))==p['files'][n]
        for n,h in p['checkpoints'].items():assert sha(z.read(n))==h
        for n in ('decode35_operand_analysis.py','decode35_operand_program.py','verify_decode35_operands.py'):assert sha(Path(__file__).with_name(n).read_bytes())==p['files'][P+n]
        bundle=z.read(p['bundle']);assert sha(bundle)==p['bundle_sha256']
        with zipfile.ZipFile(io.BytesIO(bundle)) as source:assert source.testzip() is None and len(source.namelist())==len(set(source.namelist()))==86 and {n:sha(source.read(n)) for n in source.namelist()}==p['source_files']
        with zipfile.ZipFile(io.BytesIO(z.read(B+'original-source.zip'))) as old:assert {n:sha(old.read(n)) for n in old.namelist()}=={n:h for n,h in p['candidate_base'].items() if h is not None}
        for job,row in p['parents'].items():
            raw=z.read('results/'+job+'-verified.json');q=json.loads(raw);assert sha(raw)==row['proof_sha256'] and q['status']==row['status'] and raw==(args.proofs/(job+'-verified.json')).read_bytes() and sha((args.proofs/(job+'-evidence.zip')).read_bytes())==q['archive_sha256']==row['archive_sha256']
        build=json.loads(z.read('results/decode35-stage-build-attempt1-verified.json'));assert build['exit_code']==0 and build['candidate_files']==86 and build['candidate_library_sha256']==p['library_sha256']
        stages={}
        for stage in ('cpu',)+PHASES:
            name=B+stage+'-ledger/commands.jsonl'
            if name not in mf:continue
            events=[json.loads(s) for s in z.read(name).splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished'
            expected=p['files'] if stage=='cpu' else p['installed_files'];assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==expected;stages[stage]=events[1]['exit_code']
        if stages.get('cpu')==0:assert b'Ran 81 tests' in z.read(B+'cpu-ledger/commands.log') and b'\nOK' in z.read(B+'cpu-ledger/commands.log')
        assert [row['phase'] for row in w['phases']]==list(PHASES[:len(w['phases'])])
        for row in w['phases']:
            assert row['source_installed']==p['installed_files'] and row['library_sha256']==p['library_sha256'] and row['exit_code']==stages.get(row['phase'])
            assert row['mode']==row['phase'].split('-')[1] and row['enabled'] is row['phase'].startswith('on-')
        corpus=z.read(P+'evidence/fixed-length-4096-corpus.zip');assert sha(corpus)==p['input_corpus_sha256'];selection=load_corpus(corpus)
        result_analysis=None;replayed_controls={};reports={};reads={}
        with zipfile.ZipFile(args.proofs/'step45-layer-boundary-attempt2-evidence.zip') as parent:
            pb='results/step45-layer-boundary-attempt2/'
            historical={role:json.loads(parent.read(pb+role+'/report.json')) for role in ('reference','candidate')}
            old_reads={role:(lambda n,role=role:parent.read(pb+role+'/'+n)) for role in historical}
            for phase in PHASES:
                if stages.get(phase)!=0:continue
                role=phase.split('-')[1];enabled=phase.startswith('on-');batch=1 if role=='reference' else 4
                report=json.loads(z.read(B+phase+'/report.json'));assert report['library_sha256']==p['library_sha256'] and report['phase']==phase and report['operand_witness'] is enabled
                if not enabled:assert report['internal_observer'] is None and report['rows'][0]['operand_observations']==[]
                reports[phase]=report;reads[phase]=lambda n,phase=phase:z.read(B+phase+'/'+n)
                log=z.read(B+phase+'-ledger/commands.log')
                try:
                    original=assert_disabled(log) if role=='reference' else route_trace(log,'off')
                    route={'original':original,'decode35':witness_trace(log,batch,enabled)}
                except Exception as error:
                    assert w['exit_code']!=0 and phase not in w['routes'] and w['error']==repr(error)
                    continue
                else:assert w['routes'][phase]==route
                try:
                    if enabled:control=primary_control(report,reports['off-'+role],reads[phase],reads['off-'+role],p['selected_key'],p['library_sha256'],p['library_sha256'],True,False)
                    else:control=primary_control(report,historical[role],reads[phase],old_reads[role],p['selected_key'],p['library_sha256'],p['historical_libraries'][role],False,True)
                except Exception as error:
                    assert w['exit_code']!=0 and phase not in w['controls'] and w['error']==repr(error)
                else:assert control==w['controls'][phase];replayed_controls[phase]=control
            if 'operand_analysis' in w:
                assert set(replayed_controls)==set(PHASES) and set(w['routes'])==set(PHASES)
                result_analysis=analyze(selection,p['selected_key'],reports,historical,reads,old_reads,p['library_sha256'],p['historical_libraries'])
                assert result_analysis==w['operand_analysis']==json.loads(z.read(B+'analysis.json'))
        passed=w['exit_code']==0
        if passed:assert w['status']=='same_DLL_decode35_operands_collected_unqualified' and stages==dict.fromkeys(('cpu',)+PHASES,0) and result_analysis
        else:assert w['status']=='failed' and isinstance(w.get('error'),str)
        result={'status':'same_DLL_decode35_operands_replayed' if passed else 'same_DLL_decode35_operands_failure_replayed','exit_code':w['exit_code'],'error':w.get('error'),'archive_sha256':sha(archive.read_bytes()),'files_verified':len(mf),'stage_exit_codes':stages,'controls':replayed_controls,'analysis':result_analysis,'CPU_controls':81 if stages.get('cpu')==0 else None,'source_runtime_SDK_config_index_CMake_all34_baseline_restored':True,'model_quality_HTTP_or_performance_qualified':False,'original_full4K_still_failed':True,'two_times_target_achieved':False,'reader_sha256':sha(Path(__file__).read_bytes()),'scope':p['scope']}
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'exit_code':result['exit_code'],'proof_sha256':sha(raw),'first_differing_key_path_operands':[{k:r[k] for k in ('step','first_differing_active_key_path_operand')} for r in result_analysis['rows']] if result_analysis else None}));return 0
if __name__=='__main__':raise SystemExit(main())
