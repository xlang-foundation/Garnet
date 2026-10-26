"""One explicit outer Nsight process tree, normal inner pipes and retained failures."""
from pathlib import Path
import argparse,json,sqlite3,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.native_cuda_phase_study import profiler
from test2026.cuda.qwen38_5090.replay_native_cuda_phases import replay
from test2026.cuda.qwen38_5090.run_logged import run

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--cache',type=Path,required=True);args=a.parse_args()
    root=args.root.resolve();out=args.output.resolve();m=Path(__file__).resolve().parent
    assert root==Path('D:/CantorAI/5090-study-20261023').resolve() and out.is_relative_to(root/'results') and not out.exists()
    p=json.loads((m/'evidence/native-cuda-stdio-profiled-preparation.json').read_bytes())
    tool=profiler(root);assert tool==p['profiler'];nsys=root/tool['path'];out.mkdir()
    report={'status':'starting','profiler_before':tool,'profiled':True,'performance_admitted':False,
            'scope':'One actual retained3968+128 ordinary text control under explicit pinned Nsight. Bootstrap/root-cause and actual CUDA activity only; no HTTP/speed/quality admission.'}
    def save():(out/'report.json').write_bytes(json.dumps(report,indent=2).encode())
    save();code=1
    try:
        application=subprocess.list2cmdline([sys.executable,str(m/'probe_native_cuda_stdio_bootstrap.py'),
            '--root',str(root),'--output',str(out/'control'),'--cache',str(args.cache)])
        stem=out/'cuda-trace'
        command=[str(nsys),'profile','--trace=cuda','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node',
                 '--force-overwrite=false','--output='+str(stem),'cmd.exe','/d','/c',application]
        report['profile_command']=command;save()
        code=run(command,out/'profile-ledger','correctness','Pinned outer Nsight and original exact3968+128 native control; diagnostic only',
                 [nsys,m/'probe_native_cuda_stdio_bootstrap.py',m/'serve_native_cuda_phase_stdio_worker.bat',m/'native_cuda_phase_stdio_entry.py'])
        if code:raise RuntimeError('Profiled entry control exited '+str(code))
        control=json.loads((out/'control/report.json').read_bytes())
        assert control['status']=='native_stdio_bootstrap_retained_control_completed' and control['profiled'] is True
        assert type(control['worker_exit_code']) is int and control['worker_exit_code']==0
        report['control_status']=control['status'];save()
        code=run([str(nsys),'stats','--report','cuda_gpu_kern_sum,cuda_api_sum','--format','csv',str(stem)+'.nsys-rep'],
            out/'stats-ledger','correctness','Export real CUDA records; preserve instrumented scope',[nsys,m/'replay_native_cuda_phases.py'])
        if code:raise RuntimeError('Actual CUDA stats export failed')
        markers=[json.loads(v) for v in (out/'control/host-stages.cuda-markers.jsonl').read_bytes().splitlines()]
        host=[json.loads(v) for v in (out/'control/host-stages.jsonl').read_bytes().splitlines()]
        with sqlite3.connect('file:'+Path(str(stem)+'.sqlite').as_posix()+'?mode=ro',uri=True) as connection:
            gpu=replay(connection,markers,host)
        assert gpu['requests']==1 and gpu['unique_markers']==12
        report.update(status='native_stdio_profiled_control_collected_needs_replay',cuda_phase_replay=gpu);code=0
    except BaseException as error:report.update(status='failed',error=repr(error));code=code or 1
    finally:
        report.update(exit_code=code,profiler_after=profiler(root));save()
        assert report['profiler_after']==report['profiler_before']
    print(json.dumps({'status':report['status'],'exit_code':code}),flush=True);return code

if __name__=='__main__':raise SystemExit(main())
