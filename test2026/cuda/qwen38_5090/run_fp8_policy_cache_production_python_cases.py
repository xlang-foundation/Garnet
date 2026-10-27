"""Sequential exploratory cache behavior calls on an explicitly unqualified DLL."""
from pathlib import Path
import hashlib,json,os,sys
def main():
    root,out=map(Path,sys.argv[1:3]);m=root/'Garnet/test2026/cuda/qwen38_5090';sys.path[:0]=[str(root/'Garnet'),str(m)];import run_logged as ledger
    from test2026.cuda.qwen38_5090.fp8_policy_cache_production_contract import CASES,replay
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();p=json.loads((m/'evidence/fp8-policy-cache-production-python-preparation.json').read_bytes());assert not out.exists();out.mkdir();(out/'scratch').mkdir();cache=out/'cache';os.chdir(root/'Garnet')
    ledger.source_identity=lambda:{'scope':'Explicit exploratory production cache diagnostic sources; protected unrelated bodies excluded; no promotion or benchmark','owned_source_sha256':{n:sha(root/n) for n in p['installed_files']}}
    report=dict(status='cases_running',fresh_process_per_case=True,cases=[],model_quality_or_performance_admitted=False,library_sha256=sha(root/'build/bin/garnet.dll'),native_build_artifact_identity_gate_passed=False,exploratory_only=True)
    def save():(out/'report.json').write_bytes(json.dumps(report,indent=2).encode())
    save()
    try:
        for index,spec in enumerate(CASES):
            target=out/spec['key'];code=ledger.run([sys.executable,str(m/'launch_fp8_policy_cache_production_python_case.py'),str(root),str(target),str(cache),str(index)],out/(spec['key']+'-ledger'),'correctness','Actual production load_model cache '+spec['key']+' on preserved unqualified binary; no VLM or performance admission',[m/'fp8_policy_cache_production_contract.py',m/'probe_fp8_policy_cache_production_python.py',m/'launch_fp8_policy_cache_production_python_case.py',m/'fp8_policy_cache_fixture.py',root/'build/bin/garnet.dll'])
            if (target/'report.json').is_file():report['cases'].append(json.loads((target/'report.json').read_bytes()));save()
            assert code==0,('Exploratory production case failed; preserve exact native error',spec['key'],code)
        report['status']='actual_production_cache_contracts_collected';report['local_replay']=replay(report,report['library_sha256']);save()
    except BaseException as e:report.update(status='failed',error=repr(e));save();raise
    return 0
if __name__=='__main__':raise SystemExit(main())
