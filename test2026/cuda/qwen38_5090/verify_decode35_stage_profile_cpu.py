"""Independent raw CPU guard replay; downloaded executable is never loaded."""
from pathlib import Path
import argparse,hashlib,json,zipfile

EXPECTED_INPUTS = {'qwen38_decode35_stage_profile.h': 'c45e15bdbb1b48641fa6a9ff0d6fb0442548a86e6cafe4893883282e7df3b8f0', 'qwen38_full_chunk_stage_profile.h': '2c3658933497d0b863551a682f3cf7c5d083f7393ab0f63897239d58ad6a8037', 'probe_decode35_stage_profile.cpp': '97e82cdfcdfde2c2637799e89b344147a19da4159d8efea8d08189c8840a56f4', 'build_decode35_stage_profile_cpu.bat': 'e93281e738ed5e1f91d39d40823a58a332c8b3adec75036a288534b88e4dba4f', 'runner.py': '109ba8139e9bf131725d4f2dc70dab1bcbd60a3a00f41386ff4a75ee13ee21af', 'preparation.json': 'a93793a210d644bd261103cbf747f6767ef8315f0ded799754cef629b8f0c02d'}
J='decode35-stage-profile-cpu-attempt1'
ROOT=Path('D:/CantorAI/5090-study-20261023')
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    with zipfile.ZipFile(args.archive) as z:
        mf=json.loads(z.read('evidence-manifest.json'))['files']
        assert len(z.namelist())==len(set(z.namelist()))==len(mf)+1 and z.testzip() is None
        for n,h in mf.items():assert sha(z.read(n))==h
        assert len(EXPECTED_INPUTS)==6
        for n,h in EXPECTED_INPUTS.items():assert sha(z.read('input/'+n))==h
        p=json.loads(z.read('input/preparation.json'));w=json.loads(z.read('workflow.json'))
        assert w['preparation_sha256']==EXPECTED_INPUTS['preparation.json'] and w['cleanup_finished'] and not w['model_or_GPU_executed']
        expected={k:p[k] for k in ('files','runtime','environment','checkpoints','cuda_objects','cache')}
        assert w['baseline_before']==expected and len(p['files'])==1892 and len(p['cuda_objects'])==34
        assert w['all_baseline_unchanged']==(w['baseline_before']==w['baseline_after'])
        source={str(ROOT/'results'/J/'input'/n):h for n,h in EXPECTED_INPUTS.items()}
        stages={}
        for stage in ('build','contract'):
            name=stage+'-ledger/commands.jsonl'
            if name not in mf:continue
            events=[json.loads(s) for s in z.read(name).splitlines()]
            assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished'
            assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==source
            stages[stage]=events[1]['exit_code']
        passed=w['exit_code']==0
        if passed:
            assert w['status']=='decode35_stage_native_CPU_profile_passed' and w['all_baseline_unchanged']
            assert stages=={'build':0,'contract':0} and w['build_exit_code']==w['contract_exit_code']==0
            wanted={'status':'passed','new_valid':2,'new_rejected':103,'old_valid':2,'old_rejected':2,'failures':0,'cases':109}
            records=[json.loads(s) for s in z.read('contract-ledger/commands.log').decode().splitlines() if s.startswith('{"status"')]
            assert records==[wanted] and w['contract']==wanted and sha(z.read('bin/contract.exe'))==w['executable_sha256']
        else:assert w['status']=='failed' and isinstance(w.get('error'),str)
        result={'status':'decode35_stage_native_CPU_profile_replayed' if passed else 'decode35_stage_native_CPU_profile_failure_replayed','exit_code':w['exit_code'],'error':w.get('error'),'archive_sha256':sha(args.archive.read_bytes()),'files_verified':len(mf),'source_sha256':EXPECTED_INPUTS,'source_bundle_sha256':p['source_bundle_sha256'],'source_files':p['source_files'],'local_source_controls':p['local_source_controls'],'stage_exit_codes':stages,'native_contract':w.get('contract'),'baseline_unchanged':w['all_baseline_unchanged'],'baseline_files':1892,'cuda_objects':34,'reader_sha256':sha(Path(__file__).read_bytes()),'model_or_GPU_or_performance_qualified':False,'original_full4K_still_failed':True,'two_times_target_achieved':False,'scope':p['scope']}
    raw=json.dumps(result,indent=2).encode();assert not args.output.exists() or args.output.read_bytes()==raw
    if not args.output.exists():args.output.write_bytes(raw)
    print(json.dumps({'status':result['status'],'exit_code':result['exit_code'],'proof_sha256':sha(raw),'native_contract':result['native_contract']}));return 0

if __name__=='__main__':raise SystemExit(main())
