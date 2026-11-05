"""CPU-only synthetic three-cohort namespace fixture from an actual admission.

The temporary copies test restart scoping and corruption rejection only.
They are deleted and never represent fresh GPU runs or performance evidence.
"""
import copy,json,shutil,subprocess,sys,tempfile
from pathlib import Path
from audit_local_coder_batch_campaign import audit
source=Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory)/'synthetic-cpu-fixture';shutil.copytree(source,root)
    spec=json.loads((root/'engine.json').read_text());old=str(source)
    spec['argv']=[a.replace(old,str(root)) for a in spec['argv']]
    spec['argv']=[str(root/'sources/profile-{cohort}.json') if a==str(root/'sources/profile.json') else
                  str(root/'server/cohort-{cohort}') if a==str(root/'server') else a for a in spec['argv']]
    spec['native_cohort_profiles']={str(n):f'profile-{n}.json' for n in range(1,4)}
    spec['native_cohort_log_roots']={str(n):f'server/cohort-{n}' for n in range(1,4)}
    template=root/'server-original';(root/'server').rename(template);(root/'server').mkdir()
    for number in range(1,4):
        key=str(number);profile=json.loads((root/'sources/profile.json').read_text());profile['memory_inventory_path']=str(root/f'inventory-{number}.json')
        (root/'sources'/f'profile-{number}.json').write_text(json.dumps(profile))
        if number>1:shutil.copytree(root/'campaign/cohort-1',root/f'campaign/cohort-{number}')
        folder=root/f'campaign/cohort-{number}';state=json.loads((folder/'cohort.json').read_text());state['pid']+=number;state['spawn_monotonic']+=10000*number
        state['server_argv']=[a.replace('{cohort}',key) for a in spec['argv']];(folder/'cohort.json').write_text(json.dumps(state))
        server=root/spec['native_cohort_log_roots'][key];shutil.copytree(template,server)
        path=server/'worker-stdout.jsonl';lines=[]
        for line in path.read_text().splitlines():
            outer=json.loads(line)
            try:event=json.loads(outer['line'])
            except json.JSONDecodeError:lines.append(line);continue
            if isinstance(event,dict) and event.get('event')=='ready':event['profile']=profile;outer['line']=json.dumps(event)
            lines.append(json.dumps(outer))
        path.write_text('\n'.join(lines)+'\n')
    shutil.rmtree(template)
    import hashlib
    spec['sources']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'sources').iterdir() if p.is_file()}
    (root/'engine.json').write_text(json.dumps(spec))
    metadata=json.loads((root/'campaign/frozen/metadata.json').read_text());metadata['engine']=spec
    metadata['manifest']['required_fresh_process_cohorts']=3
    if 'cohorts' in metadata['manifest']:metadata['manifest']['cohorts']=3
    (root/'campaign/frozen/metadata.json').write_text(json.dumps(metadata))
    status=json.loads((root/'campaign/status.json').read_text());status['completed_cohorts']=3;(root/'campaign/status.json').write_text(json.dumps(status))
    report=audit(root);assert report['status']=='passed',report['issues'];assert len(report['cohorts'])==3
    assert all(len(p['report']['requests'])==2 for p in report['cohorts']) and len(report['requests'])==6
    # Identical request IDs and epochs in different synthetic process scopes
    # must not collide; every component is still independently audited.
    subprocess.run([sys.executable,str(Path(__file__).with_name('test_local_coder_batch_campaign_audit.py')),str(root)],check=True)
print('CPU synthetic multi-cohort namespace/argv/restart and corruption tests PASS; no inference engine ran')
