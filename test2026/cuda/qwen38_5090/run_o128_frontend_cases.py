"""Sequential actual frontend graph cases, no model decode or speed measurement."""
from pathlib import Path
import hashlib,json,os,sys
def main():
    root,out,cache=map(Path,sys.argv[1:4]);m=root/'Garnet/test2026/cuda/qwen38_5090';sys.path[:0]=[str(root/'Garnet'),str(m)]
    import run_logged as ledger
    from test2026.cuda.qwen38_5090.o128_frontend_contract import CASES,replay
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();p=json.loads((m/'evidence/o128-frontend-preparation.json').read_bytes());assert not out.exists();out.mkdir();os.chdir(root/'Garnet')
    ledger.source_identity=lambda:{'scope':'Explicit actual O128 frontend source inventory; protected unrelated bodies excluded','owned_source_sha256':{n:sha(root/n) for n in p['installed_files']}}
    report=dict(status='cases_running',fresh_process_per_case=True,cases=[],model_or_performance_qualified=False,library_sha256=sha(root/'build/bin/garnet.dll'))
    def save():(out/'report.json').write_bytes(json.dumps(report,indent=2).encode())
    save()
    try:
        for index,spec in enumerate(CASES):
            target=out/spec['key'];code=ledger.run([sys.executable,str(m/'launch_o128_frontend_case.py'),str(root),str(target),str(cache),str(index)],out/(spec['key']+'-ledger'),'correctness','Fresh actual Garnet native frontend '+spec['key']+'; no numerical/performance admission',[m/'o128_frontend_contract.py',m/'probe_o128_frontend.py',m/'launch_o128_frontend_case.py',root/'build/bin/garnet.dll'])
            if (target/'report.json').is_file():report['cases'].append(json.loads((target/'report.json').read_bytes()));save()
            assert code==0,('Actual frontend case failed; preserve raw native error',spec['key'],code)
        report['status']='actual_O128_frontend_contracts_collected';report['local_replay']=replay(report);save()
    except BaseException as e:report.update(status='failed',error=repr(e));save();raise
    return 0
if __name__=='__main__':raise SystemExit(main())
