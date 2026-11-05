# SPDX-License-Identifier: Apache-2.0
"""Serial fresh controls then cached Mistral HTTP arrivals; no speed claim."""
import argparse
import concurrent.futures
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from urllib.request import urlopen
from urllib.error import URLError
from run_local_coder_baseline import engine_processes,stop_owned
from local_coder_stream_benchmark import run_request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('engine-spec','fixture','serving-manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if engine_processes():raise RuntimeError('Competing inference engine: '+str(engine_processes()))
    args.output.mkdir(parents=True,exist_ok=False)
    spec=json.loads(args.engine_spec.read_text());profile=json.loads(Path(spec['argv'][-1]).read_text())
    assert profile['native_cached_chunks'] and profile['prefix_cache'] and profile['prefill_chunk']==64
    fixture=json.loads(args.fixture.read_text());coding=[c for c in fixture['cases'] if c['kind']=='coding']
    plans={p['id']:p for p in json.loads(args.serving_manifest.read_text())['plans']}
    bodies={}
    for case in coding[:5]:
        plan=plans[case['id']];path=Path(plan['cases'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==plan['cases_sha256']
        bodies[case['id']]=json.loads(path.read_text())[0]
    entries=[]
    for label,index,budget in [('A',0,8),('B',1,256),('C',2,320),('D',3,384),('E',4,16),('A-return',0,32)]:
        entries.append(dict(label=label,case_id=coding[index]['id'],body=dict(bodies[coding[index]['id']],max_tokens=budget)))
    for label in ('append','changed'):
        body=copy.deepcopy(bodies[coding[0]['id']]);body['max_tokens']=16
        content=body['messages'][0]['content']
        body['messages'][0]['content']=(content+'\nInclude a short docstring.' if label=='append' else content.replace('imports needed.','imports required.'))
        assert body['messages'][0]['content']!=content
        entries.append(dict(label=label,case_id='new-'+label,body=body))
    manifest=dict(scope=__doc__,engine=spec,requests=entries,fixture=str(args.fixture),source_manifest=str(args.serving_manifest))
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    frozen=args.output/'frozen';frozen.mkdir()
    for name in ('run_mistral_cached_http.py','local_coder_stream_benchmark.py','run_local_coder_baseline.py'):
        shutil.copyfile(Path(__file__).with_name(name),frozen/name)
    report=dict(status='starting',phases={},transitions=[])
    def save():(args.output/'status.json').write_text(json.dumps(report,indent=2)+'\n')
    @contextmanager
    def phase(label,cache):
        if engine_processes():raise RuntimeError('Prior engine still live')
        folder=args.output/label;folder.mkdir()
        active=copy.deepcopy(profile);active.update(prefix_cache=cache,native_cached_chunks=cache,
                                                   profile_id=profile['profile_id']+('-cached' if cache else '-fresh'))
        profile_path=folder/'profile.json';profile_path.write_text(json.dumps(active,indent=2)+'\n')
        command=list(spec['argv']);command[-1]=str(profile_path)
        command[command.index('--log-root')+1]=str(folder/'server')
        state=dict(status='starting',command=command);report['phases'][label]=state;save()
        log=(folder/'server.log').open('x');process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,**spec.get('env',{})),start_new_session=True)
        state['pid']=process.pid;save()
        try:
            while True:
                if process.poll() is not None or (folder/'server/worker-exit.json').exists():raise RuntimeError('Native startup failed')
                try:
                    with urlopen(spec['url']+'/health',timeout=2) as response:
                        if response.status==200:state['health']=json.load(response);break
                except (URLError,TimeoutError):pass
                time.sleep(.5)
            state['status']='running';save()
            yield folder
            state['status']='collected'
        finally:
            state['shutdown']=stop_owned(process,graceful_native=True);log.close();save()
    def call(folder,entry,barrier):
        dest=folder/entry['label'];dest.mkdir()
        result=run_request(spec['url'],entry['body'],dest/'request-0',barrier,600,None,True)
        if result['status']!='completed':raise RuntimeError('Request failed: '+entry['label']+' '+repr(result))
        return result
    try:
        with phase('fresh',False) as folder:
            for entry in entries[6:]:call(folder,entry,threading.Barrier(1))
        with phase('cached',True) as folder:
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                barrier=threading.Barrier(4)
                jobs={e['label']:pool.submit(call,folder,e,barrier) for e in entries[:4]}
                previous='A'
                for entry in entries[4:]:
                    result=jobs[previous].result()
                    report['transitions'].append(dict(completed=previous,submitted=entry['label'],monotonic=time.monotonic()))
                    report['status']='running_'+entry['label'];save()
                    jobs[entry['label']]=pool.submit(call,folder,entry,threading.Barrier(1));previous=entry['label']
                for future in jobs.values():future.result()
        report['status']='collected_pending_independent_audit'
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:save()
    print(json.dumps(dict(status=report['status'],phases=report['phases'])))


if __name__=='__main__':main()
