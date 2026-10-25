"""One-mode driver profiled from outside; normal worker IPC, no HTTP timing."""
from pathlib import Path
import argparse,hashlib,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from native_fixed_length_http_bridge import Worker
from replay_native_host_stage_trace import replay
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--cache',type=Path,required=True);a.add_argument('--mode',choices=('ordinary','mtp'),required=True);args=a.parse_args();root=args.root;m=root/'Garnet/test2026/cuda/qwen38_5090';out=args.output
    assert out.is_dir() and not (out/args.mode).exists() and not (out/('report-'+args.mode+'.json')).exists() and out.resolve().is_relative_to((root/'results').resolve())
    cases=m/'evidence/native-host-stage-cases.json';p=json.loads(cases.read_bytes());parent='fixed-length-http-study-attempt1';proof=root/'results'/(parent+'-verified.json');q=json.loads(proof.read_bytes())
    assert sha(proof)==p['parent_proof_sha256'] and q['status']=='fixed_length_http_study_replayed' and q['native_fixed_4k_metrics_replayed'] and sha(root/'results'/(parent+'-evidence.zip'))==p['parent_archive_sha256']==q['archive_sha256']
    assert sha(root/'build/bin/garnet.dll')==p['library_sha256']==q['library_sha256'] and p['status']=='native_host_stage_retained_cases_prepared_gpu_unrun'
    assert len(p['cases'])==4 and p['orders']==[[0,1,2,3],[2,0,3,1],[0,1,2,3]]
    report={'status':'running','cases_sha256':sha(cases),'parent_archive_sha256':p['parent_archive_sha256'],'modes':{},'performance_admitted':False,'scope':'One-mode ordinary/T4MTP driver under outer Nsight process-tree collection; normal owned worker stdin/stdout unchanged. Four retained3968+128 requests in three orders, exact target IDs/image bytes and native phase markers. First-use/late repeats retained. Instrumented diagnosis, no HTTP, comparative speed or causal bottleneck claim.'}
    def save():(out/('report-'+args.mode+'.json')).write_text(json.dumps(report,indent=2),encoding='utf8')
    save()
    try:
        for mode in (args.mode,):
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
        report['status']='native_host_stage_diagnostic_collected_needs_independent_replay';save();print('NATIVE_CUDA_PHASE_DRIVER_SAVED',flush=True)
    except BaseException as error:report.update(status='failed',error=repr(error));save();raise
    return 0


if __name__=='__main__':raise SystemExit(main())
