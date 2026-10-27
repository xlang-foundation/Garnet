"""Retain complete first128 boundary arrays; explicitly scope independent replay.

Full3968 raw archive and full-array verifier remain on the RTX5090 machine.
This capsule never claims to carry or locally replay the other boundary frames.
"""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');JOB='layer-boundary-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';BASE='results/'+JOB+'/';sha=lambda b:hashlib.sha256(b).hexdigest()
def collect():
    archive=ROOT/'results'/(JOB+'-evidence.zip');proof=ROOT/'results'/(JOB+'-verified.json');raw=proof.read_bytes();q=json.loads(raw)
    assert q['exit_code']==0 and q['status']=='actual_complete_layer0_to4_boundaries_and_primary_equivalence_replayed' and q['source_library_runtime_SDK_checkpoint_restored']
    assert sha(archive.read_bytes())==q['archive_sha256'] and q['archive_bytes']==archive.stat().st_size
    with zipfile.ZipFile(archive) as z:
        original=json.loads(z.read('evidence-manifest.json'));selected={n:z.read(n) for n in original['files'] if not n.endswith('.f32')}
        for phase in ('baseline','native'):
            prefix=BASE+phase+'/';r=json.loads(z.read(prefix+'report.json'))
            for row in r['boundaries'][:4 if phase=='baseline' else 1]:
                for v in row['layers'].values():selected[prefix+v['file']]=z.read(prefix+v['file'])
            for name in ('empty_state','ordinary_prefill','ordinary_final','repeat_prefill'):
                if name in r:
                    for v in r[name]['vectors'].values():selected[prefix+v['file']]=z.read(prefix+v['file'])
        for n,b in selected.items():assert sha(b)==original['files'][n]
        selected['original-evidence-manifest.json']=z.read('evidence-manifest.json')
    selected['results/'+JOB+'-verified.json']=raw
    selected['results/layer-boundary-install.json']=(ROOT/'results/layer-boundary-install.json').read_bytes()
    selected[P+'layer_boundary_capsule.py']=Path(__file__).read_bytes()
    manifest=dict(scope='Complete first128 raw boundaries layers0..4, all original metadata/source bundles/commands/resources and primary raw vectors. Full3968 raw boundary replay is remote only; omitted cache parts are reused byte-exactly from retained prefix parent during independent primary replay.',files={n:sha(b) for n,b in selected.items()},original_archive_sha256=q['archive_sha256'],original_archive_bytes=q['archive_bytes'],remote_full_replay_proof_sha256=sha(raw),full_boundary_tokens=3968,capsule_boundary_tokens=128,locally_claim_full_boundary_array_replay=False)
    target=ROOT/'results'/(JOB+'-first128-capsule.zip');assert not target.exists()
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for n,b in selected.items():z.writestr(n,b)
        z.writestr('capsule-manifest.json',json.dumps(manifest,indent=2))
    print(json.dumps(dict(status='bounded_first128_capsule_retained_full_raw_archive_unchanged',sha256=sha(target.read_bytes()),bytes=target.stat().st_size,files=len(selected))))

def verify(capsule,proofs):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    from test2026.cuda.qwen38_5090.layer_boundary_replay import compare,primary_equivalence,REFERENCE,SHA_REFERENCE
    parent=proofs/(REFERENCE+'-evidence.zip');assert sha(parent.read_bytes())==SHA_REFERENCE
    with zipfile.ZipFile(capsule) as z,zipfile.ZipFile(parent) as old:
        m=json.loads(z.read('capsule-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(m['files'])|{'capsule-manifest.json'}
        for n,h in m['files'].items():assert sha(z.read(n))==h
        assert m['capsule_boundary_tokens']==128 and m['full_boundary_tokens']==3968 and m['locally_claim_full_boundary_array_replay'] is False
        assert z.read(P+'layer_boundary_capsule.py')==Path(__file__).read_bytes()
        proof=z.read('results/'+JOB+'-verified.json');q=json.loads(proof);assert sha(proof)==m['remote_full_replay_proof_sha256'] and q['archive_sha256']==m['original_archive_sha256'] and q['archive_bytes']==m['original_archive_bytes']
        assert q['exit_code']==0 and q['status']=='actual_complete_layer0_to4_boundaries_and_primary_equivalence_replayed' and q['model_quality_or_performance_qualified'] is False
        original=json.loads(z.read('original-evidence-manifest.json'));p=json.loads(z.read(P+'evidence/layer-boundary-preparation.json'));w=json.loads(z.read(BASE+'workflow.json'))
        for n,h in original['files'].items():
            if n in m['files']:assert m['files'][n]==h
        for n,h in p['new_files'].items():assert sha(z.read(n))==h
        for n,h in {**p['audit_dependencies'],**p['checkpoints']}.items():assert sha(z.read(n))==h
        assert w['exit_code']==0 and w['cleanup_finished'] is w['baseline_source_restored'] is w['baseline_library_restored'] is True
        assert w['source_before_sha256']==w['source_after_sha256']==p['files'] and w['installed_source_sha256']==p['installed_files']
        assert w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints']
        reports={phase:json.loads(z.read(BASE+phase+'/report.json')) for phase in ('baseline','native')}
        def read(phase,name):
            n=BASE+phase+'/'+name
            blob=z.read(n) if n in m['files'] else old.read('results/'+REFERENCE+'/'+phase+'/'+name)
            assert sha(blob)==original['files'][n];return blob
        reads={phase:lambda n,phase=phase:read(phase,n) for phase in reports};primary=primary_equivalence(reports,reads,proofs)
        assert primary==q['arrays']['primary_equivalence']
        scoped={}
        for phase,r in reports.items():
            rows=r['boundaries'][:4 if phase=='baseline' else 1];scoped[phase]={**r,'boundaries':rows,'prompt_ids':r['prompt_ids'][:128],'boundary_runs':2*len(rows)}
        observed=compare(scoped['baseline'],scoped['native'],reads['baseline'],reads['native'],128)
        assert observed['first_different_layer']==q['arrays']['first_different_layer']
        for layer,row in observed['layers'].items():
            full=q['arrays']['layers'][layer];assert row['first_difference']==full['first_difference'] and row['byte_exact']==full['byte_exact']
            assert row['changed_F32_words']<=full['changed_F32_words'] and row['maximum_absolute_difference']<=full['maximum_absolute_difference']
        result=dict(status='first128_all5_boundaries_and_complete_primary_equivalence_independently_replayed',capsule_sha256=sha(capsule.read_bytes()),original_full_archive_sha256=m['original_archive_sha256'],remote_full_replay_proof_sha256=m['remote_full_replay_proof_sha256'],remote_full_tokens=3968,remote_full_F32_words_compared=q['arrays']['complete_F32_words_compared'],independent_local_boundary_tokens=128,independent_local_boundary_F32_words_compared=observed['complete_F32_words_compared'],independent_local_full3968_boundary_replay=False,primary_equivalence=primary,first128_boundaries=observed,source_library_runtime_SDK_checkpoint_restoration_replayed=True,model_or_performance_qualified=False)
    target=proofs/(JOB+'-first128-capsule-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw;target.write_bytes(raw);print(json.dumps(dict(status=result['status'],proof_sha256=sha(raw),first_difference=observed['first_different_layer'])))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=('collect','verify'));a.add_argument('--capsule',type=Path);a.add_argument('--proofs',type=Path);args=a.parse_args()
    if args.mode=='collect':collect()
    else:assert args.capsule is not None and args.proofs is not None;verify(args.capsule,args.proofs)
