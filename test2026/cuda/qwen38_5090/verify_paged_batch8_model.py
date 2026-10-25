"""Independently compare the new input path to retained complete B8 model bytes."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.paged_batch8_model import replay_preserved
JOB='paged-batch8-model-preservation-attempt1';BASE='bf16-batch8-model-admission-attempt1';COMP='paged-prompt-native-component-attempt1'
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();sha=lambda raw:hashlib.sha256(raw).hexdigest()
    archive=args.proofs/(JOB+'-evidence.zip');prefix='results/'+JOB+'/'
    with zipfile.ZipFile(archive) as z:
        doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json');assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for n,h in manifest['files'].items():assert sha(z.read(n))==h
        raw=z.read('Garnet/test2026/cuda/qwen38_5090/evidence/paged-batch8-model-preparation.json');p=json.loads(raw);w=doc(prefix+'workflow.json')
        assert sha(raw)==w['preparation_sha256'] and w['source_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime']
        assert w['environment_before']==w['environment_after']==p['environment'] and w['cleanup_finished'] is w['library_unchanged'] is True and w['parents']==p['parents']
        for n,h in p['new_files'].items():assert sha(z.read(n))==h
        assert sha(Path(__file__).read_bytes())==p['new_files']['Garnet/test2026/cuda/qwen38_5090/'+Path(__file__).name]
        for name,row in p['parents'].items():
            parent=args.proofs/(name+'-evidence.zip');qraw=z.read('results/'+name+'-verified.json');q=json.loads(qraw)
            assert sha(parent.read_bytes())==q['archive_sha256']==row['archive_sha256']==manifest['linked_parent_archives'][name] and sha(qraw)==row['proof_sha256'] and q['status']==row['status']
            with zipfile.ZipFile(parent) as prior:
                old=json.loads(prior.read('results/'+name+'/workflow.json'));assert old['exit_code']==0 and old['cleanup_finished'] and old['source_sha256']==old['source_after_sha256'] and old['runtime_dependencies']==old['runtime_dependencies_after']==p['runtime']
                if name==COMP:assert p['files']=={**old['source_after_sha256'],**p['retained_b8_source_files'],**p['new_files']}
                if name==BASE:
                    for n,h in p['retained_b8_source_files'].items():assert sha(prior.read(n))==h==old['source_after_sha256'][n]
        passed=w['exit_code']==0;replayed=None;memory=None
        if passed:
            assert w['status']=='paged_batch8_model_collected_needs_replay'
            for name in ('cpu-ledger','gpu-ledger'):
                events=[json.loads(line) for line in z.read(prefix+name+'/commands.jsonl').splitlines()]
                assert len(events)==2 and events[-1]['event']=='finished' and events[-1]['exit_code']==0
            log=z.read(prefix+'cpu-ledger/commands.log');assert b'Ran 26 tests' in log and b'\nOK' in log
            with zipfile.ZipFile(args.proofs/(BASE+'-evidence.zip')) as original:
                report=doc(prefix+'model/report.json');reference=json.loads(original.read('results/'+BASE+'/model/report.json'))
                assert report['library_sha256']==reference['library_sha256']==w['library_sha256']==p['runtime']['garnet.dll']
                replayed=replay_preserved(report,lambda n:z.read(prefix+'model/'+n),reference,lambda n:original.read('results/'+BASE+'/model/'+n))
                assert replayed==report['retained_baseline_replay']
            resources=doc(prefix+'resources/resource-monitor.json');samples=[json.loads(line) for line in z.read(prefix+'resources/resource-vram.jsonl').splitlines()]
            assert resources['child_exit_code']==0 and resources['gpu']['uuid']==p['gpu_uuid'] and max(s['used_bytes'] for s in samples)==resources['sampled_memory']['sampled_peak_adapter_used_bytes'];memory=resources['sampled_memory']
        result={'status':'paged_batch8_model_preservation_replayed' if passed else 'paged_batch8_model_preservation_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),
                'library_sha256':w['library_sha256'],'preserved_model':replayed,'correctness_adapter_sampled_memory':memory,'external_b1_b4_math_qualified':False,'mtp_qualified':False,
                'broad_quality_qualified':False,'http_qualified':False,'serving_memory_saving_qualified':False,'performance_admitted':False,'two_times_target_achieved':False,'scope':p['scope']}
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'archive_sha256':result['archive_sha256'],'files_verified':result['files_verified'],'proof_sha256':sha(raw),'preserved_states':29 if passed else None}));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
