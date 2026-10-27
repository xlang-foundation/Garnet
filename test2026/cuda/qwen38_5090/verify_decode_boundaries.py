"""Independent archive replay; exact primary parity precedes interpretation."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.decode_boundary_analysis import STATUS,primary_parity,analyze
from test2026.cuda.qwen38_5090.b4_b1_short_m16_gate import route_trace
JOB='decode-boundaries-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';BASE='results/'+JOB+'/';sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);a=parser.parse_args();archive=a.proofs/(JOB+'-evidence.zip')
    with zipfile.ZipFile(archive) as z:
        mf=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(mf['files'])+1
        for n,h in mf['files'].items():assert sha(z.read(n))==h
        raw=z.read(P+'evidence/decode-boundaries-preparation.json');p=json.loads(raw);w=json.loads(z.read(BASE+'workflow.json'));assert sha(raw)==w['preparation_sha256'] and type(w['exit_code']) is int and w['cleanup_finished'] is w['baseline_source_restored'] is w['baseline_library_restored'] is True
        assert w['source_before']==w['source_after']==p['files'] and w['runtime_before']==w['runtime_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints']
        assert w['cuda_objects_before']==w['cuda_objects_after']==p['cuda_objects'] and w['cache_before_sha256']==w['cache_after_sha256']==p['cache_before_sha256'] and w['build_artifacts_unchanged']
        assert w['performance_admitted'] is w['full4k_MTP_805_qualified'] is False
        for n,h in p['archive_files'].items():assert sha(z.read(n))==h
        names=('decode_boundary_analysis.py','decode_boundary_layout.py','decode_boundary_program.py','b4_b1_exact_gate.py','b4_b1_short_m16_gate.py','verify_decode_boundaries.py')
        for n in names:assert Path(__file__).with_name(n).read_bytes()==z.read(P+n)
        for mode in ('reference','candidate'):
            assert sha(z.read(p[mode+'_bundle']))==p[mode+'_bundle_sha256']
            with zipfile.ZipFile(io.BytesIO(z.read(p[mode+'_bundle']))) as source:assert source.testzip() is None and len(source.namelist())==len(set(source.namelist()))==len(p[mode+'_source_files']) and {n:sha(source.read(n)) for n in source.namelist()}==p[mode+'_source_files']
        for job,row in p['parents'].items():
            raw=z.read('results/'+job+'-verified.json');q=json.loads(raw);assert sha(raw)==row['proof_sha256'] and q['status']==row['status'] and raw==(a.proofs/(job+'-verified.json')).read_bytes() and sha((a.proofs/(job+'-evidence.zip')).read_bytes())==q['archive_sha256']==row['archive_sha256']
        parent_path=a.proofs/Path(p['primary_parent_archive']).name;assert sha(parent_path.read_bytes())==p['primary_parent_archive_sha256'];parent_proof=json.loads((a.proofs/'b4-b1-short-m16-admission-attempt1-verified.json').read_bytes());assert parent_proof['exit_code']==1 and parent_proof['strict_short_admission_passed'] is False
        original=a.proofs/'paged-batch8-model-preservation-attempt1-evidence.zip';assert sha(original.read_bytes())==p['input_parent_archive_sha256']
        with zipfile.ZipFile(original) as old:case=json.loads(old.read('results/paged-batch8-model-preservation-attempt1/model/report.json'))['cases'][0];assert case['group']=='short-mixed' and case['actual_limit']==8;inputs=case['inputs'][:4]
        assert [len(r['prompt_ids']) for r in inputs]==[41,104,93,93]
        for row in inputs:
            name='results/native-mtp3-http/'+row['key']+'-request.json';assert sha(z.read(name))==p['audit_dependencies'][name]==row['request_sha256']
        for record in ('decode-boundary-program-derivation.json','decode-boundaries-method-derivation.json','decode-boundary-owner-derivation.json'):
            record=json.loads(z.read(P+'evidence/'+record));rows=record.get('derivations',[record])
            for row in rows:
                raw=z.read(P+row['source']);assert sha(raw)==row['source_sha256']
                for change in row['changes']:assert raw.count(change['old'].encode())==change['count'];raw=raw.replace(change['old'].encode(),change['new'].encode())
                assert raw==z.read(P+row['target']) and sha(raw)==row['target_sha256']
        def commands(name,expected):
            events=[json.loads(line) for line in z.read(BASE+name+'-ledger/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and type(events[1]['exit_code']) is int;assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==expected;return events[1]['exit_code']
        assert commands('cpu',p['files'])==0 and b'Ran 17 tests' in z.read(BASE+'cpu-ledger/commands.log') and b'\nOK' in z.read(BASE+'cpu-ledger/commands.log')
        reports={}
        for phase in w['phases']:
            mode=phase['mode'];assert mode not in reports and mode in ('reference','candidate') and phase['source_installed']==p[mode+'_installed_files'] and phase['library_sha256']==p[mode+'_library_sha256'] and commands(mode,p[mode+'_installed_files'])==phase['exit_code']
            name=BASE+mode+'/report.json'
            if name in mf['files']:
                report=json.loads(z.read(name));assert report['inputs']==inputs and report['library_sha256']==p[mode+'_library_sha256'];reports[mode]=report
        parities={};parity_error=None;observation=route=None;diagnostic_passed=False
        if set(reports)=={'reference','candidate'} and all(report['status']==STATUS for report in reports.values()):
            with zipfile.ZipFile(parent_path) as parent:
                for mode,report in reports.items():
                    base='results/b4-b1-short-m16-admission-attempt1/'+mode+'/'
                    try:parities[mode]=primary_parity(report,json.loads(parent.read(base+'report.json')),lambda n:z.read(BASE+mode+'/'+n),lambda n:parent.read(base+n))
                    except AssertionError as error:parity_error=repr(error);break
            assert parities==w.get('primary_parity',{})
            if parity_error is None:
                assert parities==json.loads(z.read(BASE+'primary-parity.json'));observation=analyze(reports['reference'],reports['candidate'],lambda n:z.read(BASE+'reference/'+n),lambda n:z.read(BASE+'candidate/'+n));assert observation==w['observation']==json.loads(z.read(BASE+'analysis.json'))
                assert b'GARNET_GDNM16_FACTORY ' not in z.read(BASE+'reference-ledger/commands.log');route=route_trace(z.read(BASE+'candidate-ledger/commands.log'));assert route==w['actual_GDN_M16_route']==json.loads(z.read(BASE+'route-analysis.json'));diagnostic_passed=True
        if w['exit_code']==0:assert w['status']=='readonly_decode_boundary_diagnostic_primary_parity_verified_model_gate_unchanged' and [v['mode'] for v in w['phases']]==['reference','candidate'] and all(v['exit_code']==0 for v in w['phases']) and diagnostic_passed
        else:assert w['status']=='failed' and isinstance(w['error'],str)
        result=dict(status='readonly_decode_boundary_primary_parity_and_arrays_replayed' if diagnostic_passed else 'readonly_decode_boundary_diagnostic_failure_replayed',exit_code=w['exit_code'],error=w.get('error'),archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size,archive_members=len(z.namelist()),reference_library_sha256=p['reference_library_sha256'],candidate_library_sha256=p['candidate_library_sha256'],phase_statuses={k:v['status'] for k,v in reports.items()},phase_errors={k:v.get('error') for k,v in reports.items()},primary_parity=parities,primary_parity_error=parity_error,observation=observation,actual_GDN_M16_route=route,diagnostic_passed=diagnostic_passed,source_runtime_SDK_checkpoint_CMake_all34objects_restored=True,original_failed_model_job_promoted=False,full4k_MTP_805_HTTP_or_performance_qualified=False,two_times_target_achieved=False,replay_verifier_sha256=sha(Path(__file__).read_bytes()),scope=p['scope'])
    target=a.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps(dict(status=result['status'],exit_code=result['exit_code'],proof_sha256=sha(raw),diagnostic_passed=diagnostic_passed)));return 0
if __name__=='__main__':raise SystemExit(main())
