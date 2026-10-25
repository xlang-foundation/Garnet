"""Sequential Nsight phase diagnosis; exact retained4K IDs, no HTTP benchmark."""
from pathlib import Path
import argparse,hashlib,json,sqlite3,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from native_fixed_length_http_bridge import Worker
from replay_native_host_stage_trace import replay
from test2026.cuda.qwen38_5090.replay_native_cuda_phases import replay as replay_cuda
from run_logged import run
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--cache',type=Path,required=True);args=a.parse_args();root=args.root;m=root/'Garnet/test2026/cuda/qwen38_5090';out=args.output
    assert not out.exists() and out.resolve().is_relative_to((root/'results').resolve())
    cases=m/'evidence/native-host-stage-cases.json';p=json.loads(cases.read_bytes());parent='fixed-length-http-study-attempt1';proof=root/'results'/(parent+'-verified.json');q=json.loads(proof.read_bytes())
    assert sha(proof)==p['parent_proof_sha256'] and q['status']=='fixed_length_http_study_replayed' and q['native_fixed_4k_metrics_replayed'] and sha(root/'results'/(parent+'-evidence.zip'))==p['parent_archive_sha256']==q['archive_sha256']
    assert sha(root/'build/bin/garnet.dll')==p['library_sha256']==q['library_sha256'] and p['status']=='native_host_stage_retained_cases_prepared_gpu_unrun'
    assert len(p['cases'])==4 and p['orders']==[[0,1,2,3],[2,0,3,1],[0,1,2,3]]
    out.mkdir();report={'status':'running','cases_sha256':sha(cases),'parent_archive_sha256':p['parent_archive_sha256'],'modes':{},'performance_admitted':False,'scope':'Actual unchanged1a1d B1 fixed4K model under Nsight node tracing, with explicit unique HtoD request/phase markers. Four retained requests in three orders for each ordinary/MTP mode. Exact target IDs and image bytes required. Preserve first-use and late repeats. Instrumented GPU/API durations only, no HTTP, uninstrumented speedup or causal bottleneck claim.'}
    def save():(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    save()
    try:
        for mode in ('ordinary','mtp'):
            target=out/mode;target.mkdir();trace=target/'host-stages.jsonl';worker=None;record={'status':'starting','requests':[]};report['modes'][mode]=record;save();expected={}
            try:
                command=['cmd.exe','/d','/c',str(m/'serve_native_cuda_phase_worker.bat'),mode,str(args.cache),str(trace)];record['command']=command;save();worker=Worker(command,target/'worker');deadline=time.monotonic()+5400
                while not worker.healthy() or getattr(worker,'ready_info',None) is None:
                    if worker.process.poll() is not None:raise RuntimeError('Diagnostic worker exited before ready')
                    if time.monotonic()>deadline:raise TimeoutError('Diagnostic worker ready deadline; inspect owned process')
                    time.sleep(.25)
                ready=worker.ready_info;assert ready['library_sha256']==p['library_sha256'] and ready['generation_mode']==mode and ready['batch']==1 and ready['capacity']==4096 and ready['qualified_model_source'] is True
                record['ready']=ready;record['status']='requests_running';save()
                for repeat,order in enumerate(p['orders']):
                    for index in order:
                        case=p['cases'][index];key=mode+'-r'+str(repeat)+'-'+case['key'];expected[key]={'case':case['case'],'prompt_ids':case['prompt_ids'],'generated_ids':case['generated_ids_by_mode'][mode],'image_sha256':case['image_sha256']}
                        values=list(worker.generate(case['body'],key));end=values[-1];assert end['event']=='completed' and end['prompt_ids']==case['prompt_ids'] and end['generated_ids']==case['generated_ids_by_mode'][mode] and end['image_sha256']==case['image_sha256']
                        record['requests'].append({'id':key,'case_index':index,'repeat':repeat,'parent_request_key':case['key'],'prompt_tokens':len(end['prompt_ids']),'output_tokens':len(end['generated_ids']),'retained_ids_exact':True});save()
                record['status']='requests_completed'
            finally:
                if worker is not None:
                    worker.close();record['worker_exit_code']=worker.process.returncode;assert record['worker_exit_code']==0;save()
            rows=[json.loads(line) for line in trace.read_bytes().splitlines()];events=[];ignored=[]
            for line in (target/'worker/worker-stdout.jsonl').read_bytes().splitlines():
                outer=json.loads(line)
                try:event=json.loads(outer['line'])
                except ValueError:ignored.append(outer);continue
                assert isinstance(event,dict) and event.get('event') in ('ready','token','completed');events.append(event)
            result=replay(rows,events,expected);assert len(result['requests'])==12;record['replay']=result;record['non_json_stdout_lines']=ignored;record['status']='instrumented_host_stages_replayed';save()
            cleanup=json.loads(trace.with_suffix('.model-closed.json').read_bytes())
            assert cleanup['status']=='native_model_close_calls_returned' and cleanup['mode']==mode and cleanup['marker_count']==144 and cleanup['library_sha256']==p['library_sha256']
            record['native_model_cleanup']=cleanup;save()
            nsys=root/'tools/n26/target-windows-x64/nsys.exe'
            code=run([str(nsys),'stats','--report','cuda_gpu_kern_sum,cuda_api_sum','--format','csv',str(target/'cuda-trace.nsys-rep')],target/'stats-ledger','correctness','Export actual full4K trace; all phase windows require real unique HtoD markers',[nsys,m/'replay_native_cuda_phases.py'])
            record['stats_exit_code']=code;save();assert type(code) is int and code==0
            markers=[json.loads(line) for line in trace.with_suffix('.cuda-markers.jsonl').read_bytes().splitlines()]
            with sqlite3.connect('file:'+(target/'cuda-trace.sqlite').as_posix()+'?mode=ro',uri=True) as connection:gpu=replay_cuda(connection,markers,rows)
            assert gpu['requests']==12 and gpu['unique_markers']==144
            (target/'cuda-phases-replayed.json').write_text(json.dumps(gpu,indent=2),encoding='utf8')
            record['cuda_phase_replay']={key:gpu[key] for key in ('status','requests','unique_markers','owner')};save()
        report['status']='native_host_stage_diagnostic_collected_needs_independent_replay';report['cuda_phase_status']='both_modes_actual_cuda_markers_collected_needs_independent_replay';save();print('NATIVE_CUDA_PHASE_DIAGNOSTIC_SAVED',flush=True)
    except BaseException as error:report.update(status='failed',error=repr(error));save();raise
    return 0


if __name__=='__main__':raise SystemExit(main())
