"""Validate source/runtime identity before reusing immutable ordinary off captures."""
import hashlib,json
J='decode35-operands-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';B='results/'+J+'/'

def checked_reuse(p,z,q):
    assert q['status']=='same_DLL_decode35_operands_failure_replayed' and q['exit_code']==1 and q['analysis'] is None
    assert q['error']=="AssertionError('Observer or DLL changed complete primary state')" and q['CPU_controls']==81
    assert q['source_runtime_SDK_config_index_CMake_all34_baseline_restored']
    assert set(q['controls'])=={'off-reference','off-candidate','on-reference'}
    assert q['archive_sha256']==p['parents'][J]['archive_sha256']
    raw=z.read(P+'evidence/decode35-operands-preparation.json');old=json.loads(raw);w=json.loads(z.read(B+'workflow.json'))
    assert hashlib.sha256(raw).hexdigest()==w['preparation_sha256'] and w['cleanup_finished'] and w['all_restored'] and w['exit_code']==1
    for key in ('runtime','environment','checkpoints','cuda_objects','cache_before_sha256','input_corpus_sha256','selected_key','bundle','bundle_sha256','source_files','candidate_base','library','library_sha256','flags','historical_libraries'):
        assert old[key]==p[key],('Reused off identity drift',key)
    assert len(old['source_files'])==len(old['candidate_base'])==86 and len(old['cuda_objects'])==34
    assert all(p['files'].get(n)==h for n,h in old['files'].items()),'Old off source inventory changed'
    assert w['source_before']==w['source_after']==old['files']
    assert w['cuda_objects_before']==w['cuda_objects_after']==old['cuda_objects']
    assert [r['phase'] for r in w['phases']]==['off-reference','off-candidate','on-reference','on-candidate'] and all(r['exit_code']==0 for r in w['phases'])
    reports={};reads={};provenance={}
    for phase,batch in (('off-reference',1),('off-candidate',4)):
        row=next(r for r in w['phases'] if r['phase']==phase)
        assert row['source_installed']==old['installed_files'] and row['library_sha256']==p['library_sha256']
        report=json.loads(z.read(B+phase+'/report.json'))
        assert report['phase']==phase and report['mode']==phase.split('-')[1] and report['physical_batch']==batch
        assert report['operand_witness'] is False and report['internal_observer'] is None and report['rows'][0]['operand_observations']==[]
        assert report['library_sha256']==p['library_sha256']
        reports[phase]=report;reads[phase]=lambda n,phase=phase:z.read(B+phase+'/'+n)
        provenance[phase]={'job':J,'phase':phase,'library_sha256':p['library_sha256'],'source_bundle_sha256':p['bundle_sha256'],'closed_control':q['controls'][phase],'fresh_process_in_this_experiment':False}
    return reports,reads,{'status':'immutable_ordinary_off_reuse_identity_verified','parent_job':J,'parent_archive_sha256':q['archive_sha256'],'profiles':provenance,'old_source_inventory_preserved':True,'all34_objects_identical':True,'all_new_on_primary_controls_required':True}
