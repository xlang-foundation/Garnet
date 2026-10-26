"""Bind the latest closed component before starting the next exclusive reference."""
from pathlib import Path
import hashlib,io,json,re,zipfile
JOB='prefill128-commit-component-attempt1'
sha=lambda b:hashlib.sha256(b).hexdigest()

def admit_component(archive_raw,proof_raw):
    q=json.loads(proof_raw)
    assert q['status']=='prefill128_commit_component_full_state_replayed' and q['archive_sha256']==sha(archive_raw)
    assert q['performance_admitted'] is q['two_times_target_achieved'] is False
    r=q['full_state_replay'];assert (r['valid_cases'],r['invalid_cases'],r['graph_replays'],r['full_canonical_words_replayed'])==(16,35,3,5448010944)
    with zipfile.ZipFile(io.BytesIO(archive_raw)) as z:
        manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for n,h in manifest['files'].items():assert sha(z.read(n))==h
        w=json.loads(z.read('results/'+JOB+'/workflow.json'))
    assert type(w['exit_code']) is int and w['exit_code']==0 and w['cleanup_finished'] is True
    assert w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after']
    assert w['environment']==w['environment_after'] and w['sdk_before']==w['sdk_after']
    return {'job':JOB,'archive_sha256':q['archive_sha256'],'proof_sha256':sha(proof_raw),'proof_status':q['status'],'exit_code':0,'cleanup_finished':True}

def closed_component(root,digest):
    if re.fullmatch('[0-9a-f]{64}',digest) is None:raise ValueError('Actual closed component archive required')
    raw=(root/'results'/(JOB+'-evidence.zip')).read_bytes();assert sha(raw)==digest
    result=admit_component(raw,(root/'results'/(JOB+'-verified.json')).read_bytes())
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        assert z.read('results/'+JOB+'/workflow.json')==(root/'results'/JOB/'workflow.json').read_bytes()
    return result
