"""Freeze a qualified native engine and run the exact three-cohort reference plan.

No model build, install or download occurs here. Every fresh native process
gets independent exclusive logs/inventory files; all runtime policies match.
"""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
from run_local_coder_baseline import engine_processes

def read(path):return json.loads(path.read_text())
def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()
def write(path,value):
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('experiment-root','qualified','admission-template','reference-manifest','asset-manifest','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[2];source=Path(__file__).parent
    if engine_processes():raise RuntimeError('A competing engine is live')
    q=args.qualified.resolve();template=args.admission_template.resolve();root=args.output.resolve();experiment=args.experiment_root.resolve()
    artifacts=read(q/'artifacts.json');quality=read(q/'evaluation.json');accounting=read(q/'input-output-audit.json')
    if (quality.get('passed')!=8 or quality.get('total')!=8 or accounting.get('status')!='passed' or accounting.get('issues') or
        len(accounting.get('tasks',[]))!=8 or not all(t.get('matches_single_reference') is True for t in accounting['tasks'])):
        raise ValueError('Full coding and full token-sequence qualification required')
    for name,wanted in artifacts['files'].items():
        if digest(q/'frozen'/name)!=wanted:raise ValueError('Qualified source changed: '+name)
    if digest(q/'libgarnet.dylib')!=artifacts['library_sha256']:raise ValueError('Qualified native library changed')
    for name,current in (('metal_batch_factory.py',repo/'xModel/qwen38/vl_27b/metal_batch_study.py'),('metal_decoder.py',repo/'xModel/qwen35/vl_0_8b/metal_decoder.py')):
        if digest(current)!=digest(q/'frozen'/name):raise ValueError('Runtime xModel source differs from qualified artifact')
    if read(template/'accounting-audit1.json').get('status')!='passed':raise ValueError('Serving admission must pass before the matched campaign')
    if digest(template/'sources/libgarnet.dylib')!=digest(q/'libgarnet.dylib') or digest(template/'sources/local_coder_batch_worker.py')!=digest(q/'frozen/local_coder_batch_worker.py'):
        raise ValueError('Serving admission must use this exact qualified library and worker')
    spec=read(template/'engine.json');manifest=read(args.reference_manifest)
    if (manifest.get('required_fresh_process_cohorts')!=3 or manifest.get('warmup_waves_per_cell')!=2 or manifest.get('measured_waves_per_cell')!=5 or
        {p['id'] for p in manifest['plans']}!={f'p{p}-c{c}-o128' for p in (256,2048) for c in (1,2,4)}):
        raise ValueError('The complete frozen phase1 reference workload is required')
    if digest(args.asset_manifest)!=manifest['model_asset_manifest_sha256']:raise ValueError('Pinned model asset manifest differs from the reference')
    assets=read(args.asset_manifest);checkpoint=Path(assets['root']).resolve()
    if Path(spec['argv'][spec['argv'].index('--checkpoint')+1]).resolve()!=checkpoint:raise ValueError('Serving checkpoint differs from pinned assets')
    for entry in assets['files']:
        path=checkpoint/entry['name']
        if path.stat().st_size!=entry['bytes'] or digest(path)!=entry['sha256']:raise ValueError('Pinned model asset changed: '+entry['name'])
        print('verified pinned model asset',entry['name'],flush=True)
    root.mkdir(parents=True,exist_ok=False);sources=root/'sources';sources.mkdir()
    for name in spec['sources']:
        if name=='profile.json':continue
        origin=q/'libgarnet.dylib' if name=='libgarnet.dylib' else q/'frozen'/name
        shutil.copy2(origin,sources/name)
    profiles=[read(p) for p in (q/'frozen').glob('*.json') if isinstance(read(p),dict) and 'profile_id' in read(p)]
    if len(profiles)!=1:raise ValueError('Exactly one frozen native profile required')
    original=profiles[0];admitted=read(template/'sources/profile.json')
    if {k:v for k,v in original.items() if k!='memory_inventory_path'}!={k:v for k,v in admitted.items() if k!='memory_inventory_path'}:
        raise ValueError('Qualified and serving profiles differ')
    canonical=dict(original,memory_inventory_path=str(root/'memory-inventory-unused.json'))
    write(sources/'profile.json',canonical);profile_names={};log_roots={};changes={}
    for number in range(1,4):
        key=str(number);name=f'profile-{number}.json';profile=dict(original,memory_inventory_path=str(root/f'memory-inventory-{number}.json'))
        write(sources/name,profile);profile_names[key]=name;log_roots[key]=f'server/cohort-{number}'
        changes[key]=dict(before=original['memory_inventory_path'],after=profile['memory_inventory_path'])
    shutil.copy2(args.reference_manifest,sources/'benchmark-manifest.json')
    spec['argv']=[value.replace(str(template),str(root)) for value in spec['argv']]
    spec['argv']=[str(sources/'profile-{cohort}.json') if value==str(sources/'profile.json') else
                  str(root/'server/cohort-{cohort}') if value==str(root/'server') else value for value in spec['argv']]
    spec.update(scope='Full matched phase1 native campaign: three fresh sequential processes, six cells, two warmups/five measured waves per cell; retained raw data, no automatic speed claim.',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        native_cohort_profiles=profile_names,native_cohort_log_roots=log_roots,qualified_coding_archive=str(q),
        reference_manifest_sha256=digest(args.reference_manifest),sources={p.name:digest(p) for p in sources.iterdir()},
        qualification_provenance=dict(artifacts_sha256=digest(q/'artifacts.json'),audit_sha256=digest(q/'input-output-audit.json'),
            evaluation_sha256=digest(q/'evaluation.json'),profile_inventory_path_changes=changes))
    write(root/'engine.json',spec)
    write(root/'asset-verification.json',dict(status='passed',manifest_sha256=digest(args.asset_manifest),repo=assets['repo'],revision=assets['revision'],
        files=len(assets['files']),bytes=sum(f['bytes'] for f in assets['files']),scope='All pinned checkpoint bytes rehashed before starting any measured engine'))
    commands=[['run_local_coder_baseline.py','--experiment-root',str(experiment),'--manifest',str(sources/'benchmark-manifest.json'),
        '--engine-spec',str(root/'engine.json'),'--output',str(root/'campaign'),'--cohorts','3'],
        ['audit_local_coder_batch_campaign.py','--root',str(root),'--output',str(root/'accounting-audit1.json')],
        ['test_local_coder_batch_campaign_audit.py',str(root)]]
    write(root/'commands.json',commands)
    write(root/'controller-provenance.json',dict(controller_sha256=digest(Path(__file__)),reference_manifest_sha256=digest(args.reference_manifest),
        collector_sha256=digest(source/'run_local_coder_baseline.py'),native_auditor_sha256=digest(source/'audit_local_coder_batch_campaign.py')))
    with (root/'driver.log').open('x') as log:
        for command in commands:
            print('starting',command[0],flush=True)
            result=subprocess.run([sys.executable,str(source/command[0]),*command[1:]],stdout=log,stderr=subprocess.STDOUT)
            print('finished',command[0],result.returncode,flush=True)
            if result.returncode:raise SystemExit(result.returncode)
    print(json.dumps(dict(status=read(root/'accounting-audit1.json')['status'],root=str(root))),flush=True)
if __name__=='__main__':main()
