# SPDX-License-Identifier: Apache-2.0
"""Three serial uncached Mistral2K/B4 rounds in one engine: cold, warm1, warm2. Frozen128 output tokens each. Native tracked bulk scheduling OFF/ON, no snapshot retention; no comparator rerun or coding admission."""
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
    parser.add_argument('--case-id',required=True,choices=['p2048-c1-o128','p2048-c4-o128'])
    args=parser.parse_args()
    concurrency=int(args.case_id.split('-c')[1].split('-')[0])
    if engine_processes():raise RuntimeError('Competing inference engine: '+str(engine_processes()))
    args.output.mkdir(parents=True,exist_ok=False)
    spec=json.loads(args.engine_spec.read_text());profile=json.loads(Path(spec['argv'][-1]).read_text())
    assert profile['prefix_shared_blocks'] and profile['prefix_entries']==0 and profile['prefix_host_bytes']==0
    assert type(profile['prefix_cache']) is bool and profile['native_cached_chunks']==(profile['prefix_cache'] or profile.get('native_leased_bulk_chunks',0)>0) and profile['prefill_chunk']==128
    fixture=json.loads(args.fixture.read_text())
    plans={p['id']:p for p in json.loads(args.serving_manifest.read_text())['plans']}
    plan=plans[args.case_id];path=Path(plan['cases'])
    assert str(path)==spec['frozen_case_file'] # SHA checked before the campaign and after terminal
    originals=json.loads(path.read_text());assert len(originals)==concurrency
    coding=[dict(id=args.case_id+'-'+str(i)) for i in range(concurrency)]
    bodies={coding[i]['id']:originals[i] for i in range(concurrency)}
    entries=[dict(label=label,case_id=coding[i]['id'],body=dict(bodies[coding[i]['id']],max_tokens=128)) for i,label in enumerate(['A','B','C','D'][:concurrency])]
    original_entries=entries
    entries=[dict(e,label=round_name+'/'+e['label'],round=round_name) for round_name in ['cold','warm1','warm2'] for e in original_entries]
    manifest=dict(scope=__doc__,engine=spec,requests=entries,fixture=str(args.fixture),source_manifest=str(args.serving_manifest))
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=dict(status='starting',phases={},transitions=[])
    def save():(args.output/'status.json').write_text(json.dumps(report,indent=2)+'\n')
    @contextmanager
    def phase(label,cache):
        if engine_processes():raise RuntimeError('Prior engine still live')
        folder=args.output/label;folder.mkdir()
        active=copy.deepcopy(profile);active.update(prefix_cache=cache,native_cached_chunks=(cache or profile.get('native_leased_bulk_chunks',0)>0),
                                                   profile_id=profile['profile_id']+('-cached' if cache else '-fresh'))
        active['memory_inventory_path']=str(folder/'prepared-inventory.json')
        profile_path=folder/'profile.json';profile_path.write_text(json.dumps(active,indent=2)+'\n')
        command=list(spec['argv']);command[-1]=str(profile_path)
        command[command.index('--log-root')+1]=str(folder/'server')
        state=dict(status='starting',command=command);report['phases'][label]=state;save()
        log=(folder/'server.log').open('x');process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,**spec.get('env',{})),start_new_session=True)
        state['pid']=process.pid;save()
        try:
            deadline=time.monotonic()+600
            while True:
                if time.monotonic()>deadline:raise RuntimeError('Native startup deadline exceeded')
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
        dest=folder/entry['label'];dest.mkdir(parents=True)
        result=run_request(spec['url'],entry['body'],dest/'request-0',barrier,600,None,True)
        if result['status']!='completed':raise RuntimeError('Request failed: '+entry['label']+' '+repr(result))
        return result
    try:
        report['fresh_control_reference']=spec['saved_native_control']
        with phase('observed',profile['prefix_cache']) as folder:
            report['rounds']=[]
            for round_name in ['cold','warm1','warm2']:
                record=dict(round=round_name,started_ns=time.time_ns());report['rounds'].append(record);save()
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    barrier=threading.Barrier(concurrency)
                    jobs={e['label']:pool.submit(call,folder,e,barrier) for e in entries if e['round']==round_name}
                    for future in jobs.values():future.result()
                record.update(finished_ns=time.time_ns(),status='collected');save()
        report['status']='collected_pending_independent_audit'
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:save()
    print(json.dumps(dict(status=report['status'],phases=report['phases'])))


if __name__=='__main__':main()
